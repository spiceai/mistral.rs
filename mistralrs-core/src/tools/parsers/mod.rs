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
        if parser.claim_start(message).is_some() {
            return vec![parser];
        }
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

/// The markers that begin a call, as the parsers are tried: Gemma 4 before Qwen, as
/// `<|tool_call>` holds `<tool_call>`.
const CALL_OPENERS: [(ToolCallFormat, &str); 9] = [
    (ToolCallFormat::DeepSeekDsml, dsml::OPEN),
    (ToolCallFormat::Atem, "<atem:function_calls>"),
    (ToolCallFormat::Gemma4, "<|tool_call>"),
    (ToolCallFormat::Liquid, "<|tool_call_start|>"),
    (ToolCallFormat::Llama, "<|python_tag|>"),
    (ToolCallFormat::Qwen, "<tool_call>"),
    (ToolCallFormat::Hunyuan, "<tool_calls>"),
    (ToolCallFormat::MistralNemo, "[TOOL_CALLS]"),
    (ToolCallFormat::DeepSeek, "<｜tool▁call▁begin｜>"),
];

/// A call found in a message: its format, `None` for a bare-JSON call, where it opens, and how
/// it ends.
pub(super) struct Span {
    pub(super) format: Option<ToolCallFormat>,
    pub(super) open: usize,
    pub(super) end: SpanEnd,
}

/// How a call found in a message ends.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SpanEnd {
    /// It ends at this offset.
    At(usize),
    /// It has not ended, and holds everything after it.
    Open,
    /// A DSML block whose tags cannot be read: it holds everything after it, and the message
    /// makes no call.
    Unreadable,
}

/// The calls of `text`, left to right, in one pass: a bare-JSON call at its start, then from
/// the earliest marker that begins one, each call to where its format ends it, with the next
/// looked for after it, so a marker inside a call, whichever format's, is that call's text. A
/// call that has not ended, such as one that does not read once closed, holds everything after
/// it, and is the last. Each marker is looked for from where the last call ended, and each
/// call's text is read once, so the pass takes time linear in `text`.
pub(super) fn call_spans(text: &str) -> Vec<Span> {
    let mut spans = Vec::new();
    let mut from = 0;
    if let Some((open, end)) = bare_json_call(text) {
        match end {
            Some(end) => {
                spans.push(Span {
                    format: None,
                    open,
                    end: SpanEnd::At(end),
                });
                from = end;
            }
            None => {
                spans.push(Span {
                    format: None,
                    open,
                    end: SpanEnd::Open,
                });
                return spans;
            }
        }
    }
    // Where each closer and tag a call is framed by next occurs, carried from call to call.
    let mut ahead = Ahead::new(text);
    // Where each marker next occurs at or after `from`, found again only once `from` passes it.
    let mut next: Vec<Option<usize>> = CALL_OPENERS
        .iter()
        .map(|(_, marker)| text[from..].find(marker).map(|at| from + at))
        .collect();
    loop {
        for ((_, marker), at) in CALL_OPENERS.iter().zip(next.iter_mut()) {
            if at.is_some_and(|at| at < from) {
                *at = text[from..].find(marker).map(|offset| from + offset);
            }
        }
        let Some((format, marker, open)) = CALL_OPENERS
            .iter()
            .zip(&next)
            .filter_map(|(&(format, marker), at)| at.map(|at| (format, marker, at)))
            .min_by_key(|&(_, marker, at)| (at, std::cmp::Reverse(marker.len())))
        else {
            return spans;
        };
        let body = open + marker.len();
        let end = if format == ToolCallFormat::DeepSeekDsml {
            match dsml::frame(text, open) {
                dsml::Frame::NotABlock => {
                    from = body;
                    continue;
                }
                dsml::Frame::Block(end) => SpanEnd::At(end),
                dsml::Frame::Open => SpanEnd::Open,
                dsml::Frame::Unreadable => SpanEnd::Unreadable,
            }
        } else {
            call_end(format, &mut ahead, open, body).map_or(SpanEnd::Open, SpanEnd::At)
        };
        spans.push(Span {
            format: Some(format),
            open,
            end,
        });
        match end {
            SpanEnd::At(end) => from = end,
            SpanEnd::Open | SpanEnd::Unreadable => return spans,
        }
    }
}

/// Whether more text could still make `text` a call: it ends inside a call that has not ended,
/// or partway through a marker that begins one outside any call, or holds a Mistral Nemo or
/// Llama marker, whose parser reads the rest of the message from it as one call, so none of
/// what follows may be read on its own.
pub(crate) fn may_still_call(text: &str) -> bool {
    if text.contains("[TOOL_CALLS]") || text.contains("<|python_tag|>") {
        return true;
    }
    let after = match call_spans(text).last() {
        Some(Span {
            end: SpanEnd::At(end),
            ..
        }) => *end,
        Some(_) => return true,
        None => 0,
    };
    CALL_OPENERS.iter().any(|(_, marker)| {
        marker
            .char_indices()
            .skip(1)
            .any(|(at, _)| text.ends_with(&marker[..at]) && text.len() - at >= after)
    })
}

/// The bare-JSON call that `text` begins with, a call or an array of calls written as JSON with
/// nothing around it: where it opens, and where it ends, `None` while it has not.
fn bare_json_call(text: &str) -> Option<(usize, Option<usize>)> {
    fn after(text: &str, c: char) -> Option<&str> {
        text.strip_prefix(c).map(str::trim_start)
    }
    let object =
        |text: &str| after(text, '{').is_some_and(|rest| rest.is_empty() || rest.starts_with('"'));
    let start = text.len() - text.trim_start().len();
    let json = &text[start..];
    (object(json) || after(json, '[').is_some_and(|rest| rest.is_empty() || object(rest)))
        .then(|| (start, json_end(text, start)))
}

/// Where the JSON value that begins at `start` in `text` ends: at the bracket closing its
/// first, counting only brackets outside its strings, where `\` escapes the next character.
/// A value that does not read once closed is never finished. `None` while it is not.
fn json_end(text: &str, start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut inside = false;
    let mut escaped = false;
    for (offset, c) in text[start..].char_indices() {
        if inside {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                inside = false;
            }
            continue;
        }
        match c {
            '"' => inside = true,
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    let end = start + offset + c.len_utf8();
                    return is_json(&text[start..end]).then_some(end);
                }
            }
            _ => {}
        }
    }
    None
}

/// The next occurrence of each closer and tag a call is framed by, at or after a position in a
/// text: each is looked for again only once a lookup passes the occurrence found, so the
/// lookups of one pass over the text, which move forward, take time linear in it.
struct Ahead<'a> {
    text: &'a str,
    found: Vec<(&'static str, usize, Option<usize>)>,
    /// Where the next call begins at or after a position, and the position it was found from.
    begins: Option<(usize, Option<usize>)>,
}

impl<'a> Ahead<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            found: Vec::new(),
            begins: None,
        }
    }

    /// Where the next marker at or after `from` that begins a call, as the text right after it
    /// shows (`starts_call`), opens.
    fn next_call(&mut self, from: usize) -> Option<usize> {
        if let Some((searched, next)) = self.begins {
            if searched <= from && next.is_none_or(|at| at >= from) {
                return next;
            }
        }
        let mut at = from;
        let next = loop {
            let Some((format, marker, open)) = CALL_OPENERS
                .iter()
                .filter_map(|&(format, marker)| {
                    self.find(marker, at).map(|open| (format, marker, open))
                })
                .min_by_key(|&(_, marker, open)| (open, std::cmp::Reverse(marker.len())))
            else {
                break None;
            };
            if starts_call(format, self.text, open, open + marker.len()) {
                break Some(open);
            }
            at = open + marker.len();
        };
        self.begins = Some((from, next));
        next
    }

    /// Where `needle` next occurs in the text at or after `from`.
    fn find(&mut self, needle: &'static str, from: usize) -> Option<usize> {
        let text = self.text;
        let search = || text[from..].find(needle).map(|at| from + at);
        match self.found.iter_mut().find(|(found, _, _)| *found == needle) {
            Some((_, searched, next)) => {
                if *searched > from || next.is_some_and(|at| at < from) {
                    *searched = from;
                    *next = search();
                }
                *next
            }
            None => {
                let next = search();
                self.found.push((needle, from, next));
                next
            }
        }
    }
}

/// Where the call in `format` that opens at `open` in the text `ahead` reads, its body from
/// `body`, ends, or `None` while it is still open. A closer inside one of the call's values is
/// the value's text: a Gemma 4 value quoted in `<|"|>`, a Qwen `<parameter=…>` or Atem
/// `<atem:parameter>` value, a quoted Liquid argument, and a JSON string of a Qwen JSON,
/// Hunyuan or DeepSeek call. A JSON-format call ends at the closer after its JSON, once that
/// reads as a call, so one that does not read stays open and holds what follows. A Mistral Nemo
/// or Llama call is its marker and then the JSON of a call running to the end of the message:
/// it ends with that JSON, and a marker followed by anything else, such as prose or the rest of
/// a string it is quoted in, begins no call. Neither does an Atem call to `self` or `user`, nor
/// a marker its parser would not read a call after (`begins_call`), such as a stray opener
/// before a real call. Each search stops at the call's own closer, carried in `ahead`.
fn call_end(format: ToolCallFormat, ahead: &mut Ahead, open: usize, body: usize) -> Option<usize> {
    let text = ahead.text;
    let rest = &text[body..];
    let json = body + (rest.len() - rest.trim_start().len());
    let starts_json = |brackets: &[char]| text[json..].starts_with(brackets);
    let closer = CLOSED_CALL_DELIMITERS
        .iter()
        .find_map(|&(delimited, _, end)| (delimited == format).then_some(end));
    if !begins_call(format, ahead, body, closer)? {
        return Some(body);
    }
    match (format, closer) {
        (ToolCallFormat::Gemma4, Some(end)) => {
            // `call:NAME{...}` runs to the brace matching its first, which must come before the
            // first closer, and a closer inside a `<|"|>`-quoted value is that value's text.
            let first = ahead.find(end, body)?;
            match text[body..first].find('{') {
                Some(offset) => {
                    let (_, after) = gemma4::extract_matched_braces(text, body + offset)?;
                    ahead.find(end, after).map(|at| at + end.len())
                }
                None => Some(first + end.len()),
            }
        }
        (ToolCallFormat::Atem, Some(end)) => {
            end_outside_tagged_values(ahead, body, end, "<atem:parameter", "</atem:parameter>")
                .or_else(|| {
                    matches!(atem::channel_recipient(text, open), Some("self" | "user"))
                        .then_some(body)
                })
        }
        (ToolCallFormat::Qwen, Some(end)) if starts_json(&['{', '[']) => {
            end_after_json(text, json, end)
        }
        (ToolCallFormat::Qwen, Some(end)) => {
            end_outside_tagged_values(ahead, body, end, "<parameter=", "</parameter>")
        }
        (ToolCallFormat::Hunyuan, Some(end)) if starts_json(&['[']) => {
            end_after_json(text, json, end)
        }
        (ToolCallFormat::DeepSeek, Some(end)) => {
            // The arguments' JSON comes before the first closer; the call ends at the closer
            // after it, once the call reads.
            let first = ahead.find(end, body)?;
            let arguments = body + text[body..first].find(['{', '['])?;
            let value_end = json_end(text, arguments)?;
            let close = ahead.find(end, value_end)? + end.len();
            deepseek::DeepSeekParser
                .parse(&text[open..close])
                .is_ok_and(|calls| calls.is_some())
                .then_some(close)
        }
        (ToolCallFormat::Liquid, Some(end)) => end_outside_quotes(text, body, end),
        (_, Some(end)) => ahead.find(end, body).map(|at| at + end.len()),
        (ToolCallFormat::MistralNemo, None) if rest.is_empty() => None,
        (ToolCallFormat::MistralNemo, None) if rest.starts_with('[') => json_end(text, body),
        (ToolCallFormat::Llama, None) if rest.trim_start().is_empty() => None,
        (ToolCallFormat::Llama, None) if starts_json(&['{', '[']) => json_end(text, json),
        (_, None) => Some(body),
    }
}

/// Whether the text after a marker, from `body` in the text `ahead` reads, begins a call in
/// `format` as the format's parser reads one, `None` while too little of it is written to tell:
/// a Gemma 4 call opens with `call:`, a DeepSeek one with `function<｜tool▁sep｜>`, a Hunyuan or
/// Liquid one with `[`, a Qwen one with JSON or a `<function=…>` before its closer, and an Atem
/// one with an `<atem:invoke` before its closer. A marker followed by anything else is text, so
/// a stray opener does not swallow, or end inside, a call written after it.
fn begins_call(
    format: ToolCallFormat,
    ahead: &mut Ahead,
    body: usize,
    closer: Option<&'static str>,
) -> Option<bool> {
    let rest = &ahead.text[body..];
    let opens_with = |prefix: &str| {
        if rest.starts_with(prefix) {
            Some(true)
        } else {
            (!prefix.starts_with(rest)).then_some(false)
        }
    };
    let tag = match format {
        ToolCallFormat::Gemma4 => return opens_with("call:"),
        ToolCallFormat::DeepSeek => return opens_with("function<｜tool▁sep｜>"),
        ToolCallFormat::Hunyuan | ToolCallFormat::Liquid => {
            let trimmed = rest.trim_start();
            return (!trimmed.is_empty()).then(|| trimmed.starts_with('['));
        }
        ToolCallFormat::Qwen if rest.trim_start().starts_with(['{', '[']) => return Some(true),
        ToolCallFormat::Qwen => "<function=",
        ToolCallFormat::Atem => "<atem:invoke",
        _ => return Some(true),
    };
    // The tag counts only inside the call's own extent: before its closer, and before any
    // other call begins, so one quoted in a later call's value does not.
    let tag_at = ahead.find(tag, body);
    let close = closer.and_then(|end| ahead.find(end, body));
    let later = ahead.next_call(body);
    match tag_at {
        Some(at) => {
            Some(close.is_none_or(|close| at < close) && later.is_none_or(|later| at < later))
        }
        None if close.is_none() && later.is_none() => None,
        None => Some(false),
    }
}

/// Whether the marker of `format` that opens at `open` in `text` begins a call, judged by the
/// text right after it, from `body`: what `begins_call` reads, with a Qwen or Atem call taken to
/// begin only where its JSON, `<function=` or `<atem:invoke` follows the marker directly.
fn starts_call(format: ToolCallFormat, text: &str, open: usize, body: usize) -> bool {
    let rest = &text[body..];
    let trimmed = rest.trim_start();
    match format {
        ToolCallFormat::DeepSeekDsml => dsml::opens_block(&text[open..]),
        ToolCallFormat::Gemma4 => rest.starts_with("call:"),
        ToolCallFormat::DeepSeek => rest.starts_with("function<｜tool▁sep｜>"),
        ToolCallFormat::Hunyuan | ToolCallFormat::Liquid | ToolCallFormat::MistralNemo => {
            trimmed.starts_with('[')
        }
        ToolCallFormat::Llama => trimmed.starts_with(['{', '[']),
        ToolCallFormat::Qwen => {
            trimmed.starts_with(['{', '[']) || trimmed.starts_with("<function=")
        }
        ToolCallFormat::Atem => trimmed.starts_with("<atem:invoke"),
        ToolCallFormat::Harmony => false,
    }
}

/// Whether `text` is one JSON value.
fn is_json(text: &str) -> bool {
    serde_json::from_str::<serde::de::IgnoredAny>(text).is_ok()
}

/// Where a call whose JSON begins at `json` in `text` ends: past `end`, which only whitespace
/// may separate from the JSON once it reads. `None` while it has not, or ever will.
fn end_after_json(text: &str, json: usize, end: &str) -> Option<usize> {
    let value_end = json_end(text, json)?;
    let after = &text[value_end..];
    let close = value_end + (after.len() - after.trim_start().len());
    text[close..].starts_with(end).then(|| close + end.len())
}

/// Where the first `end` from `body` in the text `ahead` reads that is not inside a value, from
/// a `value_open` tag through its `value_close`, ends; `None` before one, or inside a value still
/// open. Every search is carried in `ahead`.
fn end_outside_tagged_values(
    ahead: &mut Ahead,
    body: usize,
    end: &'static str,
    value_open: &'static str,
    value_close: &'static str,
) -> Option<usize> {
    let mut from = body;
    loop {
        let close = ahead.find(end, from)?;
        match ahead.find(value_open, from).filter(|&at| at < close) {
            Some(at) => {
                let value = at + value_open.len();
                from = ahead.find(value_close, value)? + value_close.len();
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

/// The format that `message` is read in alone, whatever format the template prefers: DSML, once
/// it opens a block outside any other format's call.
pub(crate) fn claimed_format(message: &str) -> Option<ToolCallFormat> {
    dsml::claims(message).then_some(ToolCallFormat::DeepSeekDsml)
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
