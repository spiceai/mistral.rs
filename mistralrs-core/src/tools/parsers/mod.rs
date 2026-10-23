//! Model-specific tool call format parsers.
//!
//! Each model family emits tool calls in a different format.  Parsers are
//! registered in [`PARSERS`] and tried in order — the first match wins.

pub(crate) mod atem;
mod deepseek;
mod dsml;
mod gemma4;
mod gemma4_strict;
pub(crate) mod harmony;
mod hunyuan;
mod liquid;
mod llama;
mod mistral_nemo;
mod qwen;

use candle_core::Result;
use llguidance::api::TopLevelGrammar;

use crate::Tool;

const REQUIRED_TOOL_CALL_CONTROL_TOKENS: &[&str] = &[
    "<tool_call>",
    "</tool_call>",
    "<|python_tag|>",
    "[TOOL_CALLS]",
    "<tool_calls>",
    "</tool_calls>",
    "<|message|>",
    "<|eom|>",
    "<|start|>",
];

/// Identifies the detected tool call format so that the correct grammar
/// can be constructed for mid-stream constrained decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// Muse Glimmer ATEM recipient message with XML-like function markup.
    Atem,
    /// `<tool_call>{"name":"...","arguments":{...}}</tool_call>`
    Qwen,
    /// `<|python_tag|>{"name":"...","parameters":{...}}`
    Llama,
    /// `<|tool_call_start|>[name(arg="value")]<|tool_call_end|>`
    Liquid,
    /// `[TOOL_CALLS][{"name":"...","arguments":{...}}]`
    MistralNemo,
    /// `<tool_calls>[{"name":"...","arguments":{...}}]</tool_calls>`
    Hunyuan,
    /// Multi-line with ` ```json` fence and `<｜tool▁call▁end｜>` delimiter
    DeepSeek,
    /// DeepSeek-V4's DSML markup: `<｜DSML｜tool_calls>` around `invoke` and `parameter` tags
    DeepSeekDsml,
    /// `<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>` (non-JSON)
    Gemma4,
    /// Harmony commentary channel tool call with `to=...` recipient.
    Harmony,
}

/// A parser that can detect and extract tool calls from model output.
pub trait ToolFormatParser: Send + Sync {
    /// Returns `true` if `text` contains a token sequence that could be (or
    /// is) a tool call in this format.  Used during streaming to decide
    /// whether to buffer output.
    fn could_be_tool_call(&self, text: &str) -> bool;

    /// Whether `text` ends inside a call in this format: one has opened and not closed. A
    /// format whose calls run to the end of the message is inside one from its marker on.
    fn open_at_end(&self, text: &str) -> bool {
        self.could_be_tool_call(text)
    }

    /// Where `message` opens a call in this format, finished or not, for a format that claims
    /// the messages it opens a call in: one is read only in that format, so text inside its
    /// calls never becomes a call in another. A call opened inside another format's call is
    /// that call's text, and gives no claim.
    fn claim_start(&self, _message: &str) -> Option<usize> {
        None
    }

    /// Attempt to parse complete tool calls from `message`.
    ///
    /// Returns `Ok(Some(json))` if tool calls were found and parsed (the
    /// JSON is a serialised `Vec<CalledFunctionParameters>` or a single
    /// `CalledFunctionParameters`).
    ///
    /// Returns `Ok(None)` if this parser doesn't match the message format,
    /// OR if the tool call is incomplete (still being generated).
    ///
    /// The caller will fall through to the next parser on `Ok(None)`.
    fn parse(&self, message: &str) -> Result<Option<String>>;

    /// The tool call format this parser handles.
    fn format(&self) -> ToolCallFormat;

    /// Build an llguidance grammar that constrains model output to a valid
    /// tool call in this format.  The grammar covers the **post-prefix**
    /// content (the prefix itself is already generated when grammar
    /// activation occurs).
    ///
    /// `text` is the full generated text up to the point of grammar
    /// activation — parsers like DeepSeek can use it to extract the tool
    /// name from the prefix.
    fn tool_call_grammar(&self, tools: &[Tool], text: &str) -> TopLevelGrammar;

    /// Build an llguidance grammar for a complete tool call in this format.
    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar;
}

/// Static registry of all supported tool-call format parsers, tried in order.
///
/// **Order matters**: Gemma 4 must come before Qwen because `<|tool_call>`
/// contains `<tool_call>` as a substring.
static PARSERS: std::sync::LazyLock<Vec<Box<dyn ToolFormatParser>>> =
    std::sync::LazyLock::new(|| {
        vec![
            Box::new(dsml::DsmlParser),
            Box::new(atem::AtemParser),
            Box::new(gemma4::Gemma4Parser),
            Box::new(liquid::LiquidParser),
            Box::new(llama::LlamaParser),
            Box::new(qwen::QwenParser),
            Box::new(hunyuan::HunyuanParser),
            Box::new(mistral_nemo::MistralNemoParser),
            Box::new(deepseek::DeepSeekParser),
        ]
    });

/// The parsers to read `message` with, in order. A format that opens a call in it outside any
/// other format's call reads it alone; one whose call opens inside another format's call does
/// not read it at all.
fn parsers_for(message: &str) -> Vec<&'static dyn ToolFormatParser> {
    let all = || PARSERS.iter().map(|parser| parser.as_ref());
    for parser in all() {
        let Some(start) = parser.claim_start(message) else {
            continue;
        };
        let before = &message[..start];
        let enclosed =
            all().any(|other| other.format() != parser.format() && other.open_at_end(before));
        return if enclosed {
            all()
                .filter(|other| other.format() != parser.format())
                .collect()
        } else {
            vec![parser]
        };
    }
    all().collect()
}

/// Check whether `text` could contain a tool call prefix from any supported
/// format.
pub fn contains_tool_call_prefix(text: &str) -> bool {
    parsers_for(text)
        .into_iter()
        .any(|p| p.could_be_tool_call(text))
}

/// Build a tool call grammar for the first matching format detected in
/// `text`.  Returns `None` if no format matches or if the format is not
/// yet ready for grammar activation (e.g. DeepSeek before the JSON fence).
pub fn build_tool_call_grammar(text: &str, tools: &[Tool]) -> Option<TopLevelGrammar> {
    for parser in parsers_for(text) {
        if parser.could_be_tool_call(text) {
            // DeepSeek: wait until the JSON fence is present so we don't
            // activate grammar before the function name is complete.
            if parser.format() == ToolCallFormat::DeepSeek && !text.contains("```json\n") {
                return None;
            }
            // DSML tags open with `｜DSML｜`, a token of its own. A grammar written in text would
            // make the model spell it byte by byte, so DSML calls are read, not constrained.
            if parser.format() == ToolCallFormat::DeepSeekDsml {
                return None;
            }
            return Some(parser.tool_call_grammar(tools, text));
        }
    }
    None
}

pub fn build_required_tool_call_grammar(
    format: Option<ToolCallFormat>,
    tools: &[Tool],
) -> TopLevelGrammar {
    if format == Some(ToolCallFormat::Harmony) {
        return harmony::required_tool_call_grammar(tools, false);
    }

    if let Some(format) = format {
        for parser in PARSERS.iter() {
            if parser.format() == format {
                return parser.required_tool_call_grammar(tools);
            }
        }
    }

    qwen::QwenParser.required_tool_call_grammar(tools)
}

pub fn specialize_required_tool_call_grammar(
    grammar: &mut TopLevelGrammar,
    tok_trie: &toktrie::TokTrie,
) {
    let Some(lark) = grammar
        .grammars
        .first_mut()
        .and_then(|grammar| grammar.lark_grammar.as_mut())
    else {
        return;
    };

    for token in REQUIRED_TOOL_CALL_CONTROL_TOKENS {
        if tok_trie.get_special_token(token).is_some() {
            *lark = lark.replace(&format!(r#""{token}""#), token);
        }
    }
    // DSML's marker is spelled through one rule, as the special token's id where the
    // tokenizer has one, since its text has no brackets to name it by.
    if let Some(mark) = tok_trie.get_special_token(dsml::DSML) {
        *lark = lark
            .replace(dsml::MARK_RULE, &format!("dsml_mark: <[{mark}]>"))
            .replace(dsml::VALUE_RULES, dsml::VALUE_RULES_WITH_MARK_TOKEN);
    }
}

/// Whether `text` ends after an `open` delimiter with no `close` after it.
fn open_between(text: &str, open: &str, close: &str) -> bool {
    text.rfind(open)
        .is_some_and(|at| !text[at + open.len()..].contains(close))
}

/// Why a required call in `format` cannot hold one of `tools` to its strict schema, if it
/// cannot.
pub(crate) fn unenforceable_strict_tool(
    format: Option<ToolCallFormat>,
    tools: &[Tool],
) -> Option<String> {
    (format == Some(ToolCallFormat::DeepSeekDsml))
        .then(|| dsml::unenforceable(tools))
        .flatten()
}

/// Whether `message` opens a call in a format that no grammar constrains, and has not finished
/// it: a DSML block.
pub(crate) fn call_unfinished(message: &str) -> bool {
    dsml::unfinished_block(message)
}

/// Try each parser in order to extract tool calls from `message`.
/// Returns the original message unchanged if no parser matches.
pub fn process_model_specific_message(message: &str) -> Result<String> {
    for parser in parsers_for(message) {
        if let Some(json) = parser.parse(message)? {
            return Ok(json);
        }
    }
    Ok(message.to_string())
}

/// The calls a model-specific format reads out of `message`, as JSON, with the text around
/// them and the format they were read in.
pub(crate) fn extract_model_specific_call(
    message: &str,
) -> Result<Option<(String, String, ToolCallFormat)>> {
    for parser in parsers_for(message) {
        if let Some(json) = parser.parse(message)? {
            let text = strip_tool_call_segments(message, parser.format());
            return Ok(Some((json, text, parser.format())));
        }
    }
    Ok(None)
}

fn strip_tool_call_segments(message: &str, format: ToolCallFormat) -> String {
    match format {
        ToolCallFormat::Qwen => strip_delimited_segments(message, "<tool_call>", "</tool_call>"),
        ToolCallFormat::Gemma4 => strip_delimited_segments(message, "<|tool_call>", "<tool_call|>"),
        ToolCallFormat::DeepSeek => {
            strip_delimited_segments(message, "<｜tool▁call▁begin｜>", "<｜tool▁call▁end｜>")
        }
        ToolCallFormat::DeepSeekDsml => dsml::strip_calls(message),
        ToolCallFormat::MistralNemo => strip_from_first(message, "[TOOL_CALLS]"),
        ToolCallFormat::Hunyuan => {
            strip_delimited_segments(message, "<tool_calls>", "</tool_calls>")
        }
        ToolCallFormat::Llama => strip_from_first(message, "<|python_tag|>"),
        ToolCallFormat::Liquid => {
            strip_delimited_segments(message, "<|tool_call_start|>", "<|tool_call_end|>")
        }
        ToolCallFormat::Atem => {
            strip_delimited_segments(message, "<atem:function_calls>", "</atem:function_calls>")
        }
        ToolCallFormat::Harmony => message.to_string(),
    }
}

fn strip_from_first(message: &str, start: &str) -> String {
    message
        .find(start)
        .map_or_else(|| message.to_string(), |pos| message[..pos].to_string())
}

fn strip_delimited_segments(message: &str, start: &str, end: &str) -> String {
    let mut rest = message;
    let mut out = String::new();
    while let Some(start_pos) = rest.find(start) {
        out.push_str(&rest[..start_pos]);
        let after_start = &rest[start_pos + start.len()..];
        let Some(end_pos) = after_start.find(end) else {
            return out;
        };
        rest = &after_start[end_pos + end.len()..];
    }
    out.push_str(rest);
    out
}
