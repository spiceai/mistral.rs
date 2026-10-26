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

/// The calls of `text`, left to right: a bare-JSON call at its start, then from the earliest
/// marker that begins one, each call to where its format ends it, with the next looked for
/// after it, so a marker inside a call, whichever format's, is that call's text.
///
/// A marker begins a call only if the call can be read to its end, in its format, before
/// another marker in its framing, its text outside its values, begins a call that can be: a
/// marker whose call another cuts short is text, so its closer, written in the other call's
/// value, does not end it there. A call that has not ended, or may yet begin, holds everything
/// after it, and is the last. Each marker is read once, and the markers read as text are passed
/// over together, so the pass takes time linear in `text`.
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
    let mut ahead = Ahead::new(text);
    let mut index = ahead.first_marker_at(from);
    loop {
        let (next, reading) = match ahead.next_not_text(index) {
            Ok(Some(next)) => next,
            Ok(None) => return spans,
            Err(unread) => {
                ahead.resolve(unread);
                index = unread;
                continue;
            }
        };
        let (open, format, _) = ahead.markers[next];
        let end = match reading {
            Reading::Call(end) => SpanEnd::At(end),
            Reading::Open => SpanEnd::Open,
            Reading::Unreadable => SpanEnd::Unreadable,
            Reading::Text => {
                index = next + 1;
                continue;
            }
        };
        spans.push(Span {
            format: Some(format),
            open,
            end,
        });
        let SpanEnd::At(end) = end else {
            return spans;
        };
        index = ahead.first_marker_at(end);
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

/// How the text from a marker that may begin a call reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reading {
    /// The marker begins a call that ends at this offset.
    Call(usize),
    /// The marker begins no call: it is text.
    Text,
    /// The marker begins a call that has not ended, or may yet begin one: it holds everything
    /// after it.
    Open,
    /// The marker opens a DSML block whose tags cannot be read: it holds everything after it,
    /// and the message makes no call.
    Unreadable,
}

/// A value, or, as `Err`, the index of the later marker not yet read that it waits on.
type Waits<T> = std::result::Result<T, usize>;

/// How a call's JSON value reads.
#[derive(Clone, Copy)]
enum Json {
    /// It ends at this offset.
    Ends(usize),
    /// It has not ended, or does not read: the call holds everything after it.
    Open,
    /// A marker outside its strings begins a call, which cuts this one short.
    Interrupted,
}

/// The markers of a text that may begin a call, how each reads once read, and where each closer
/// and tag a call is framed by occurs.
struct Ahead<'a> {
    text: &'a str,
    /// Each marker, as its offset, format and text, in order. A marker inside an earlier one,
    /// as `<tool_call>` is inside `<|tool_call>`, is part of it, and is left out.
    markers: Vec<(usize, ToolCallFormat, &'static str)>,
    /// How the marker at each index reads, once read.
    readings: Vec<Option<Reading>>,
    /// For a marker read as text, a later index such that every marker between the two is
    /// read as text too, so a search for the next marker that is not passes them together.
    skip: Vec<usize>,
    /// Every occurrence of each closer and tag looked for, in order.
    occurrences: Vec<(&'static str, Vec<usize>)>,
    /// How a call whose values are tagged, by the tag that opens one, reads from an offset
    /// where its framing goes on, once read: two calls read to the same offset read the same
    /// from there.
    tails: std::collections::HashMap<(usize, &'static str), Reading>,
    /// How the JSON value that begins at an offset reads, once read.
    json_values: std::collections::HashMap<usize, Json>,
    /// Where the Gemma 4 brace at an offset, outside a quoted value, is matched, once read, and
    /// `None` where it never is.
    braces: std::collections::HashMap<usize, Option<usize>>,
    /// Where the Liquid bracket at an offset, outside a string, is matched before any closer,
    /// once read, and `None` where it never is.
    brackets: std::collections::HashMap<usize, Option<usize>>,
}

impl<'a> Ahead<'a> {
    fn new(text: &'a str) -> Self {
        let mut found: Vec<(usize, ToolCallFormat, &'static str)> = CALL_OPENERS
            .iter()
            .flat_map(|&(format, marker)| {
                text.match_indices(marker)
                    .map(move |(at, _)| (at, format, marker))
            })
            .collect();
        found.sort_unstable_by_key(|&(at, _, marker)| (at, std::cmp::Reverse(marker.len())));
        let mut markers = Vec::with_capacity(found.len());
        let mut end = 0;
        for (at, format, marker) in found {
            if at >= end {
                markers.push((at, format, marker));
                end = at + marker.len();
            }
        }
        let count = markers.len();
        Self {
            text,
            markers,
            readings: vec![None; count],
            skip: (1..=count).collect(),
            occurrences: Vec::new(),
            tails: std::collections::HashMap::new(),
            json_values: std::collections::HashMap::new(),
            braces: std::collections::HashMap::new(),
            brackets: std::collections::HashMap::new(),
        }
    }

    /// The index of the first marker at or after the offset `from`.
    fn first_marker_at(&self, from: usize) -> usize {
        self.markers.partition_point(|&(at, _, _)| at < from)
    }

    /// Where `needle` next occurs in the text at or after `from`.
    fn find(&mut self, needle: &'static str, from: usize) -> Option<usize> {
        let index = match self
            .occurrences
            .iter()
            .position(|&(found, _)| found == needle)
        {
            Some(index) => index,
            None => {
                let at = self.text.match_indices(needle).map(|(at, _)| at).collect();
                self.occurrences.push((needle, at));
                self.occurrences.len() - 1
            }
        };
        let at = &self.occurrences[index].1;
        at.get(at.partition_point(|&at| at < from)).copied()
    }

    /// The first marker at or after the index `from` that is not read as text, as its index
    /// and reading; or, as `Err`, the first one there not yet read.
    fn next_not_text(&mut self, from: usize) -> Waits<Option<(usize, Reading)>> {
        let mut at = from;
        let found = loop {
            match self.readings.get(at).copied() {
                None => break Ok(None),
                Some(None) => break Err(at),
                Some(Some(Reading::Text)) => at = self.skip[at],
                Some(Some(reading)) => break Ok(Some((at, reading))),
            }
        };
        // Every marker passed on the way is text, up to where the search stopped.
        let mut passed = from;
        while passed < at {
            let next = self.skip[passed];
            self.skip[passed] = at;
            passed = next;
        }
        found
    }

    /// Reads the marker at `index`, and each later marker its reading waits on, the last first.
    fn resolve(&mut self, index: usize) {
        let mut waiting = vec![index];
        while let Some(&top) = waiting.last() {
            if self.readings[top].is_some() {
                waiting.pop();
                continue;
            }
            match self.read(top, true) {
                Ok(reading) => {
                    self.readings[top] = Some(reading);
                    waiting.pop();
                }
                Err(unread) => waiting.push(unread),
            }
        }
    }

    /// Whether the stretch of a call's framing from `from` to `to`, or to the text's end, cuts
    /// the call's reading short: as `Text` when the first marker there that is not text begins
    /// a call that ends or a DSML block that cannot be read, as `Open` when it begins one that
    /// has not ended, so the call around it has not either, and `None` when there is none.
    /// With `inline`, a marker there not yet read is read here, unless its own reading waits on
    /// a later one, which is returned as `Err`, as any unread marker is without it.
    fn cut(&mut self, from: usize, to: Option<usize>, inline: bool) -> Waits<Option<Reading>> {
        let within = |at: usize| to.is_none_or(|to| at < to);
        let mut index = self.first_marker_at(from);
        loop {
            match self.next_not_text(index) {
                Ok(Some((next, reading))) if within(self.markers[next].0) => {
                    return Ok(match reading {
                        Reading::Call(_) | Reading::Unreadable => Some(Reading::Text),
                        Reading::Open => Some(Reading::Open),
                        Reading::Text => None,
                    });
                }
                Ok(_) => return Ok(None),
                Err(unread) if !within(self.markers[unread].0) => return Ok(None),
                Err(unread) if inline => {
                    let reading = self.read(unread, false)?;
                    self.readings[unread] = Some(reading);
                    index = unread;
                }
                Err(unread) => return Err(unread),
            }
        }
    }

    /// Whether a marker at `at`, outside a call's values, cuts the call short, as `cut` reads
    /// it.
    fn cut_at(&mut self, at: usize, inline: bool) -> Waits<Option<Reading>> {
        self.cut(at, Some(at + 1), inline)
    }

    /// How the marker at `index` reads, from the readings of the later markers in its call's
    /// framing; `Err` with the first of those not yet read, when one is needed.
    ///
    /// A Gemma 4 call opens with `call:`, a DeepSeek one with `function<｜tool▁sep｜>`, a
    /// Hunyuan or Liquid one with `[`, a Qwen one with JSON or a `<function=…>` before its
    /// closer, an Atem one with an `<atem:invoke` before its closer, and a Mistral Nemo or
    /// Llama one with JSON: a marker followed by anything else is text. A closer inside one of
    /// the call's values is the value's text: a Gemma 4 value in its braces, a Qwen
    /// `<parameter=…>` or Atem `<atem:parameter>` value, a quoted Liquid argument, and a JSON
    /// string. A JSON-format call ends at the closer after its JSON, once that reads as a call,
    /// so one that does not read stays open and holds what follows. A Mistral Nemo or Llama
    /// call is its marker and then the JSON of a call running to the end of the message. An
    /// Atem call to `self` or `user` begins no call. A DSML block is read as DSML reads it.
    fn read(&mut self, index: usize, inline: bool) -> Waits<Reading> {
        let (open, format, marker) = self.markers[index];
        let text = self.text;
        let body = open + marker.len();
        let rest = &text[body..];
        let trimmed = rest.trim_start();
        let json = text.len() - trimmed.len();
        let closer = CLOSED_CALL_DELIMITERS
            .iter()
            .find_map(|&(delimited, _, end)| (delimited == format).then_some(end));
        let opens_with = |prefix: &str| {
            if rest.starts_with(prefix) {
                Some(true)
            } else {
                (!prefix.starts_with(rest)).then_some(false)
            }
        };
        Ok(match (format, closer) {
            (ToolCallFormat::DeepSeekDsml, _) => match dsml::frame(text, open) {
                dsml::Frame::NotABlock => Reading::Text,
                dsml::Frame::Block(end) => Reading::Call(end),
                dsml::Frame::Open => Reading::Open,
                dsml::Frame::Unreadable => Reading::Unreadable,
            },
            (ToolCallFormat::Gemma4, Some(end)) => match opens_with("call:") {
                Some(true) => self.gemma4(body, end, inline)?,
                Some(false) => Reading::Text,
                None => Reading::Open,
            },
            (ToolCallFormat::DeepSeek, Some(end)) => match opens_with("function<｜tool▁sep｜>")
            {
                Some(true) => self.deepseek(open, body, end, inline)?,
                Some(false) => Reading::Text,
                None => Reading::Open,
            },
            (ToolCallFormat::MistralNemo, _) if rest.is_empty() => Reading::Open,
            (ToolCallFormat::MistralNemo, _) if rest.starts_with('[') => {
                self.json_reading(body, inline)?
            }
            (ToolCallFormat::Hunyuan | ToolCallFormat::Liquid | ToolCallFormat::Llama, _)
                if trimmed.is_empty() =>
            {
                Reading::Open
            }
            (ToolCallFormat::Llama, _) if trimmed.starts_with(['{', '[']) => {
                self.json_reading(json, inline)?
            }
            (ToolCallFormat::Hunyuan, Some(end)) if trimmed.starts_with('[') => {
                self.json_call(json, end, inline)?
            }
            (ToolCallFormat::Liquid, Some(end)) if trimmed.starts_with('[') => {
                self.liquid(body, end)
            }
            (ToolCallFormat::Qwen, Some(end)) if trimmed.starts_with(['{', '[']) => {
                self.json_call(json, end, inline)?
            }
            (ToolCallFormat::Qwen, Some(end)) => self.tagged(
                body,
                end,
                "<function=",
                "<parameter=",
                "</parameter>",
                inline,
            )?,
            (ToolCallFormat::Atem, Some(end)) => {
                let tag = "<atem:invoke";
                let reading = self.tagged(
                    body,
                    end,
                    tag,
                    "<atem:parameter",
                    "</atem:parameter>",
                    inline,
                )?;
                let to_itself = reading == Reading::Open
                    && self.find(tag, body).is_some()
                    && matches!(atem::channel_recipient(text, open), Some("self" | "user"));
                if to_itself {
                    Reading::Text
                } else {
                    reading
                }
            }
            _ => Reading::Text,
        })
    }

    /// A Gemma 4 call, from after its `call:`: its name, then its arguments' braces, then its
    /// closer, or its closer right after the name for a call with no arguments.
    fn gemma4(&mut self, body: usize, end: &'static str, inline: bool) -> Waits<Reading> {
        let first = self.find(end, body);
        let brace = self
            .find("{", body)
            .filter(|&at| first.is_none_or(|first| at < first));
        if let Some(reading) = self.cut(body, brace.or(first), inline)? {
            return Ok(reading);
        }
        let Some(brace) = brace else {
            return Ok(first.map_or(Reading::Open, |first| Reading::Call(first + end.len())));
        };
        let Some(after) = self.gemma4_braces(brace) else {
            return Ok(Reading::Open);
        };
        let close = self.find(end, after);
        if let Some(reading) = self.cut(after, close, inline)? {
            return Ok(reading);
        }
        Ok(close.map_or(Reading::Open, |close| Reading::Call(close + end.len())))
    }

    /// Where the braces of a Gemma 4 call's arguments, which open at `brace`, close, counting
    /// only braces outside its values quoted in `<|"|>`; `None` while they have not. Gemma 4
    /// reads an unquoted value as its text, so a marker in the braces is an argument's text, as
    /// a marker in a quoted value is. Each brace met on the way is matched once: a brace another
    /// call's braces reach, outside a quoted value, is matched as it is on its own, so its
    /// region is passed over in one step, and the braces of one text are read in time linear in
    /// it.
    fn gemma4_braces(&mut self, brace: usize) -> Option<usize> {
        const QUOTE: &str = "<|\"|>";
        if let Some(&matched) = self.braces.get(&brace) {
            return matched;
        }
        let text = self.text;
        // The braces opened and not yet matched, the innermost last.
        let mut open = vec![brace];
        let mut quoted = false;
        let mut at = brace + 1;
        while let Some(c) = text[at..].chars().next() {
            if text[at..].starts_with(QUOTE) {
                quoted = !quoted;
                at += QUOTE.len();
                continue;
            }
            if !quoted {
                match c {
                    '{' => match self.braces.get(&at) {
                        Some(&Some(matched)) => {
                            at = matched;
                            continue;
                        }
                        // A brace inside these that never closes leaves them open too.
                        Some(&None) => break,
                        None => open.push(at),
                    },
                    '}' => {
                        if let Some(inner) = open.pop() {
                            self.braces.insert(inner, Some(at + 1));
                        }
                        if open.is_empty() {
                            return Some(at + 1);
                        }
                    }
                    _ => {}
                }
            }
            at += c.len_utf8();
        }
        for unmatched in open {
            self.braces.insert(unmatched, None);
        }
        None
    }

    /// A DeepSeek call, from after its `function<｜tool▁sep｜>`: its name and fence, its
    /// arguments' JSON before its first closer, and the closer after that JSON, once the
    /// DeepSeek parser reads the whole as a call.
    fn deepseek(
        &mut self,
        open: usize,
        body: usize,
        end: &'static str,
        inline: bool,
    ) -> Waits<Reading> {
        let first = self.find(end, body);
        let arguments = [self.find("{", body), self.find("[", body)]
            .into_iter()
            .flatten()
            .min()
            .filter(|&at| first.is_none_or(|first| at < first));
        if let Some(reading) = self.cut(body, arguments.or(first), inline)? {
            return Ok(reading);
        }
        let Some(arguments) = arguments else {
            return Ok(Reading::Open);
        };
        let value_end = match self.json_value(arguments, inline)? {
            Json::Ends(value_end) => value_end,
            Json::Open => return Ok(Reading::Open),
            Json::Interrupted => return Ok(Reading::Text),
        };
        let close = self.find(end, value_end);
        if let Some(reading) = self.cut(value_end, close, inline)? {
            return Ok(reading);
        }
        let Some(close) = close.map(|close| close + end.len()) else {
            return Ok(Reading::Open);
        };
        let reads = deepseek::DeepSeekParser
            .parse(&self.text[open..close])
            .is_ok_and(|calls| calls.is_some());
        Ok(if reads {
            Reading::Call(close)
        } else {
            Reading::Open
        })
    }

    /// A call whose JSON begins at `json` and is followed by its closer `end`, which only
    /// whitespace may separate from the JSON once it reads. Other text after the JSON leaves the
    /// call open, as one that does not read, unless it is a marker that begins a call.
    fn json_call(&mut self, json: usize, end: &'static str, inline: bool) -> Waits<Reading> {
        let value_end = match self.json_value(json, inline)? {
            Json::Ends(value_end) => value_end,
            Json::Open => return Ok(Reading::Open),
            Json::Interrupted => return Ok(Reading::Text),
        };
        let after = &self.text[value_end..];
        let close = value_end + (after.len() - after.trim_start().len());
        if self.text[close..].starts_with(end) {
            return Ok(Reading::Call(close + end.len()));
        }
        Ok(match self.cut_at(close, inline)? {
            Some(Reading::Text) => Reading::Text,
            _ => Reading::Open,
        })
    }

    /// A call that is its JSON, which begins at `start`, as a Mistral Nemo or Llama call is.
    fn json_reading(&mut self, start: usize, inline: bool) -> Waits<Reading> {
        Ok(match self.json_value(start, inline)? {
            Json::Ends(end) => Reading::Call(end),
            Json::Open => Reading::Open,
            Json::Interrupted => Reading::Text,
        })
    }

    /// How the JSON value that begins at `start` reads: where it ends, at the bracket closing
    /// its first, counting only brackets outside its strings, where `\` escapes the next
    /// character, once it reads; or where a marker outside its strings, where JSON cannot go
    /// on, begins a call.
    fn json_value(&mut self, start: usize, inline: bool) -> Waits<Json> {
        if let Some(&json) = self.json_values.get(&start) {
            return Ok(json);
        }
        let json = self.read_json_value(start, inline)?;
        self.json_values.insert(start, json);
        Ok(json)
    }

    /// How the JSON value that begins at `start` reads, as `json_value` gives it, read anew.
    fn read_json_value(&mut self, start: usize, inline: bool) -> Waits<Json> {
        let text = self.text;
        let mut depth = 0usize;
        let mut inside = false;
        let mut escaped = false;
        let mut marker = self.first_marker_at(start);
        for (offset, c) in text[start..].char_indices() {
            let at = start + offset;
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
            if matches!(c, '<' | '[') {
                while self
                    .markers
                    .get(marker)
                    .is_some_and(|&(open, _, _)| open < at)
                {
                    marker += 1;
                }
                if self
                    .markers
                    .get(marker)
                    .is_some_and(|&(open, _, _)| open == at)
                {
                    return Ok(match self.cut_at(at, inline)? {
                        Some(Reading::Text) => Json::Interrupted,
                        _ => Json::Open,
                    });
                }
            }
            match c {
                '"' => inside = true,
                '{' | '[' => depth += 1,
                '}' | ']' => {
                    let Some(outer) = depth.checked_sub(1) else {
                        return Ok(Json::Open);
                    };
                    depth = outer;
                    if depth == 0 {
                        let end = at + c.len_utf8();
                        return Ok(if is_json(&text[start..end]) {
                            Json::Ends(end)
                        } else {
                            Json::Open
                        });
                    }
                }
                _ => {}
            }
        }
        Ok(Json::Open)
    }

    /// A Liquid call, from its `[`: it ends at the first closer `end` outside a string quoted
    /// in `"` or `'`, where `\` escapes the next character, once its list has closed right
    /// before it, every bracket outside the strings matched; one that has not stays open and
    /// holds what follows. Liquid reads an unquoted value as its text, so a marker in the list
    /// is an argument's text, as a marker in a string is. Each bracket met on the way is matched
    /// once, as `gemma4_braces` matches a brace.
    fn liquid(&mut self, body: usize, end: &'static str) -> Reading {
        let text = self.text;
        let list = body + (text[body..].len() - text[body..].trim_start().len());
        // The brackets opened and not yet matched, the innermost last.
        let mut open = Vec::new();
        let mut quote = None;
        let mut escaped = false;
        let mut last = ' ';
        let mut at = list;
        match self.brackets.get(&list) {
            Some(&Some(matched)) => {
                at = matched;
                last = text[..matched].chars().next_back().unwrap_or(' ');
            }
            Some(&None) => return Reading::Open,
            None => {}
        }
        let reading = loop {
            let Some(c) = text[at..].chars().next() else {
                break Reading::Open;
            };
            if let Some(opened) = quote {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == opened {
                    quote = None;
                    last = c;
                }
                at += c.len_utf8();
                continue;
            }
            let closed = open.is_empty() && last == ']';
            if text[at..].starts_with(end) {
                break if closed {
                    Reading::Call(at + end.len())
                } else {
                    Reading::Open
                };
            }
            // Only whitespace may come between the closed list and the closer.
            if closed && !c.is_whitespace() {
                break Reading::Open;
            }
            match c {
                '"' | '\'' => quote = Some(c),
                '[' | '(' | '{' if !open.is_empty() => match self.brackets.get(&at) {
                    Some(&Some(matched)) => {
                        at = matched;
                        last = text[..matched].chars().next_back().unwrap_or(c);
                        continue;
                    }
                    // A bracket inside these that never closes leaves them open too.
                    Some(&None) => break Reading::Open,
                    None => open.push(at),
                },
                '[' | '(' | '{' => open.push(at),
                ']' | ')' | '}' => match open.pop() {
                    Some(opened) => {
                        self.brackets.insert(opened, Some(at + c.len_utf8()));
                    }
                    None => break Reading::Open,
                },
                _ => {}
            }
            if !c.is_whitespace() {
                last = c;
            }
            at += c.len_utf8();
        };
        for unmatched in open {
            self.brackets.insert(unmatched, None);
        }
        reading
    }

    /// A call whose start is `tag`, before its closer `end`, and whose values run from each
    /// `value_open` to the `value_close` after it. A closer before the tag leaves no call.
    fn tagged(
        &mut self,
        body: usize,
        end: &'static str,
        tag: &'static str,
        value_open: &'static str,
        value_close: &'static str,
        inline: bool,
    ) -> Waits<Reading> {
        let tag_at = self.find(tag, body);
        let close = self.find(end, body);
        if close.is_some_and(|close| tag_at.is_none_or(|tag_at| close < tag_at)) {
            return Ok(Reading::Text);
        }
        if let Some(reading) = self.cut(body, tag_at, inline)? {
            return Ok(reading);
        }
        let Some(mut from) = tag_at else {
            return Ok(Reading::Open);
        };
        let mut passed = Vec::new();
        let reading = loop {
            if let Some(&reading) = self.tails.get(&(from, value_open)) {
                break reading;
            }
            passed.push(from);
            let close = self.find(end, from);
            let value = self
                .find(value_open, from)
                .filter(|&at| close.is_none_or(|close| at < close));
            if let Some(reading) = self.cut(from, value.or(close), inline)? {
                break reading;
            }
            match (value, close) {
                (Some(at), _) => match self.find(value_close, at + value_open.len()) {
                    Some(value_end) => from = value_end + value_close.len(),
                    None => break Reading::Open,
                },
                (None, Some(close)) => break Reading::Call(close + end.len()),
                (None, None) => break Reading::Open,
            }
        };
        for from in passed {
            self.tails.insert((from, value_open), reading);
        }
        Ok(reading)
    }
}

/// Whether `text` is one JSON value.
fn is_json(text: &str) -> bool {
    serde_json::from_str::<serde::de::IgnoredAny>(text).is_ok()
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
