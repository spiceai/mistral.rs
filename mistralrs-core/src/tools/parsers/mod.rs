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
        let enclosed = open_in_other_format(&message[..start], parser.format());
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
            // DSML: the grammar takes over right after a block's opening tag, and spells the
            // `｜DSML｜` marker through one rule that `specialize_tool_call_grammar` makes the
            // tokenizer's special token.
            if parser.format() == ToolCallFormat::DeepSeekDsml {
                return dsml::continuation_grammar(tools, text);
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

/// Spells a tool-call grammar's control tokens, and DSML's marker, as the tokenizer's special
/// tokens where it has them, for a required call's grammar and for the grammar that takes over a
/// call the model began.
pub fn specialize_tool_call_grammar(grammar: &mut TopLevelGrammar, tok_trie: &toktrie::TokTrie) {
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

/// The delimiters of the formats whose calls close, in the order the parsers are tried: Gemma 4
/// before Qwen, as `<|tool_call>` holds `<tool_call>`.
const CLOSED_CALL_DELIMITERS: [(ToolCallFormat, &str, &str); 6] = [
    (
        ToolCallFormat::Atem,
        "<atem:function_calls>",
        "</atem:function_calls>",
    ),
    (ToolCallFormat::Gemma4, "<|tool_call>", "<tool_call|>"),
    (ToolCallFormat::Qwen, "<tool_call>", "</tool_call>"),
    (ToolCallFormat::Hunyuan, "<tool_calls>", "</tool_calls>"),
    (
        ToolCallFormat::DeepSeek,
        "<｜tool▁call▁begin｜>",
        "<｜tool▁call▁end｜>",
    ),
    (
        ToolCallFormat::Liquid,
        "<|tool_call_start|>",
        "<|tool_call_end|>",
    ),
];

/// `text` with every finished call of a format whose calls close blanked out, so a marker the
/// model wrote inside one, as an argument's text, does not open a call in another format. The
/// calls are read left to right: the earliest opener claims the text through its closer, and
/// a marker inside that span, whichever format's, is the call's text. An opener that never
/// closes is a call that holds everything after it, which is left as it is.
fn without_closed_calls(text: &str) -> String {
    let mut out = text.to_string();
    let mut from = 0;
    loop {
        let Some((format, open, start, end)) = CLOSED_CALL_DELIMITERS
            .iter()
            .filter_map(|(format, start, end)| {
                out[from..]
                    .find(start)
                    .map(|offset| (*format, from + offset, *start, *end))
            })
            .min_by_key(|(_, open, start, _)| (*open, std::cmp::Reverse(start.len())))
        else {
            return out;
        };
        let Some(close) = call_end(format, &out, open, start, end) else {
            return out;
        };
        let blank = " ".repeat(out[open..close].chars().count());
        out.replace_range(open..close, &blank);
        from = open + blank.len();
    }
}

/// Where the call in `format` that `start` opens at `open` in `text` ends, past its `end`, or
/// `None` while it is still open. A closer inside one of the call's values is the value's text:
/// a Gemma 4 value quoted in `<|"|>`, a Qwen `<parameter=…>` or Atem `<atem:parameter>` value,
/// a quoted Liquid argument, and a JSON string of a Qwen JSON, Hunyuan or DeepSeek call. A
/// JSON-format call ends only at a closer up to which it reads as a call, so one that does not
/// read stays open and holds what follows. Any other call ends at its first closer.
fn call_end(
    format: ToolCallFormat,
    text: &str,
    open: usize,
    start: &str,
    end: &str,
) -> Option<usize> {
    let body = open + start.len();
    match format {
        ToolCallFormat::Gemma4 => gemma4::call_end(text, open),
        ToolCallFormat::Qwen if text[body..].trim_start().starts_with("<function=") => {
            end_outside_tagged_values(text, body, end, "<parameter=", "</parameter>")
        }
        ToolCallFormat::Atem => {
            end_outside_tagged_values(text, body, end, "<atem:parameter", "</atem:parameter>")
        }
        ToolCallFormat::Liquid => end_outside_quotes(text, body, end),
        ToolCallFormat::Qwen if text[body..].trim_start().starts_with(['{', '[']) => {
            end_outside_json_strings(text, body, end, |at| is_json(&text[body..at]))
        }
        ToolCallFormat::Hunyuan if text[body..].trim_start().starts_with('[') => {
            end_outside_json_strings(text, body, end, |at| is_json(&text[body..at]))
        }
        ToolCallFormat::DeepSeek => end_outside_json_strings(text, body, end, |at| {
            deepseek::DeepSeekParser
                .parse(&text[open..at + end.len()])
                .is_ok_and(|calls| calls.is_some())
        }),
        _ => text[body..]
            .find(end)
            .map(|length| body + length + end.len()),
    }
}

/// Whether `text` is one JSON value.
fn is_json(text: &str) -> bool {
    serde_json::from_str::<serde::de::IgnoredAny>(text).is_ok()
}

/// Where the first `end` from `body` in `text` that is outside JSON strings, where `\` escapes
/// the next character, and at which `reads` holds ends; `None` before one.
fn end_outside_json_strings(
    text: &str,
    body: usize,
    end: &str,
    reads: impl Fn(usize) -> bool,
) -> Option<usize> {
    let mut inside = false;
    let mut escaped = false;
    for (offset, c) in text[body..].char_indices() {
        let at = body + offset;
        if inside {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                inside = false;
            }
        } else if c == '"' {
            inside = true;
        } else if text[at..].starts_with(end) && reads(at) {
            return Some(at + end.len());
        }
    }
    None
}

/// Where the first `end` from `body` in `text` that is not inside a value, from a `value_open`
/// tag through its `value_close`, ends; `None` before one, or inside a value still open.
fn end_outside_tagged_values(
    text: &str,
    body: usize,
    end: &str,
    value_open: &str,
    value_close: &str,
) -> Option<usize> {
    let mut from = body;
    loop {
        let close = from + text[from..].find(end)?;
        match text[from..close].find(value_open) {
            Some(offset) => {
                let value = from + offset + value_open.len();
                from = value + text[value..].find(value_close)? + value_close.len();
            }
            None => return Some(close + end.len()),
        }
    }
}

/// Where the first `end` from `body` in `text` that is not inside a string quoted in `"` or
/// `'`, where `\` escapes the next character, ends; `None` before one, or inside a string.
fn end_outside_quotes(text: &str, body: usize, end: &str) -> Option<usize> {
    let mut quote = None;
    let mut escaped = false;
    for (offset, c) in text[body..].char_indices() {
        match quote {
            Some(_) if escaped => escaped = false,
            Some(_) if c == '\\' => escaped = true,
            Some(open) if c == open => quote = None,
            Some(_) => {}
            None if c == '"' || c == '\'' => quote = Some(c),
            None if text[body + offset..].starts_with(end) => {
                return Some(body + offset + end.len());
            }
            None => {}
        }
    }
    None
}

/// Whether a call in `format`, one of the formats whose calls close, is open at the end of
/// `text`, and where it opens: a call is open from its opener until `call_end` finds its end.
pub(super) fn open_call_start(text: &str, format: ToolCallFormat) -> Option<usize> {
    let (_, start, end) = CLOSED_CALL_DELIMITERS
        .iter()
        .find(|(delimited, _, _)| *delimited == format)?;
    let mut from = 0;
    while let Some(offset) = text[from..].find(start) {
        let open = from + offset;
        let Some(close) = call_end(format, text, open, start, end) else {
            return Some(open);
        };
        from = close;
    }
    None
}

/// Whether a call in a format other than `format` is open at the end of `text`, a bare-JSON call
/// included. A marker inside a finished call of some format is that call's text, and opens
/// nothing.
pub(super) fn open_in_other_format(text: &str, format: ToolCallFormat) -> bool {
    if in_bare_json_call_string(text) {
        return true;
    }
    let outside = without_closed_calls(&without_bare_json_call(text));
    PARSERS
        .iter()
        .any(|parser| parser.format() != format && parser.open_at_end(&outside))
}

/// `text` with the bare-JSON call it begins with, a call or an array of calls written as JSON
/// with nothing around it, blanked out once the call is finished, so a marker quoted in one of
/// its strings does not open a call in another format.
fn without_bare_json_call(text: &str) -> std::borrow::Cow<'_, str> {
    let json = text.trim_start();
    let start = text.len() - json.len();
    let finished = |length: Option<usize>| length.map(|length| start..start + length);
    let Some(call) = json
        .starts_with(['{', '['])
        .then(|| {
            finished(finished_json_length::<super::CalledFunctionParameters>(
                json,
            ))
            .or_else(|| {
                finished(finished_json_length::<Vec<super::CalledFunctionParameters>>(json))
            })
        })
        .flatten()
    else {
        return std::borrow::Cow::Borrowed(text);
    };
    let mut out = text.to_string();
    let blank = " ".repeat(out[call.clone()].chars().count());
    out.replace_range(call, &blank);
    std::borrow::Cow::Owned(out)
}

/// The length of the `T` that `json` begins with, written as JSON, once it is finished.
fn finished_json_length<T: serde::de::DeserializeOwned>(json: &str) -> Option<usize> {
    let mut values = serde_json::Deserializer::from_str(json).into_iter::<T>();
    matches!(values.next(), Some(Ok(_))).then(|| values.byte_offset())
}

/// Whether `text` begins a bare-JSON call, a call or an array of calls written as JSON with
/// nothing around it, and ends inside one of its strings, where a marker is the string's text.
fn in_bare_json_call_string(text: &str) -> bool {
    let json = text.trim_start();
    json.starts_with(['{', '[']) && unfinished_json_call(json) && ends_inside_json_string(json)
}

/// Whether `json` is the start of a call or of an array of calls written as JSON, cut off.
fn unfinished_json_call(json: &str) -> bool {
    super::could_be_json::<super::CalledFunctionParameters>(json).0
        || super::could_be_json::<Vec<super::CalledFunctionParameters>>(json).0
}

/// Whether `text` ends inside a call of a format whose calls are `marker` and then JSON running
/// to the end of the message: after some `marker`, the text `begins` a call, and the call's JSON
/// has yet to begin or has begun and not finished. A marker followed by anything else, such as
/// prose or the rest of a JSON string it is quoted in, begins no call.
fn json_call_open_after(text: &str, marker: &str, begins: impl Fn(&str) -> bool) -> bool {
    text.match_indices(marker).any(|(at, _)| {
        let rest = &text[at + marker.len()..];
        begins(rest) && (rest.trim().is_empty() || unfinished_json_call(rest))
    })
}

/// Whether the JSON `text` ends inside a string.
fn ends_inside_json_string(text: &str) -> bool {
    let mut inside = false;
    let mut escaped = false;
    for c in text.chars() {
        if escaped {
            escaped = false;
        } else if inside {
            match c {
                '\\' => escaped = true,
                '"' => inside = false,
                _ => {}
            }
        } else {
            inside = c == '"';
        }
    }
    inside
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

/// `message` without the DSML blocks that made a call, where `made` says whether the call at
/// an index among the message's calls, in order, was made. A block none of whose calls were
/// made is content, as a malformed one is.
pub(crate) fn strip_dsml_blocks_making_calls(
    message: &str,
    made: impl Fn(usize) -> bool,
) -> String {
    dsml::strip_blocks_making_calls(message, made)
}

fn strip_tool_call_segments(message: &str, format: ToolCallFormat) -> String {
    if let Some((_, start, end)) = CLOSED_CALL_DELIMITERS
        .iter()
        .find(|(delimited, _, _)| *delimited == format)
    {
        return strip_delimited_segments(message, start, end);
    }
    match format {
        ToolCallFormat::DeepSeekDsml => dsml::strip_calls(message),
        ToolCallFormat::MistralNemo => strip_from_first(message, "[TOOL_CALLS]"),
        ToolCallFormat::Llama => strip_from_first(message, "<|python_tag|>"),
        _ => message.to_string(),
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
