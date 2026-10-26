//! DeepSeek-V4's tool calls, written in its DSML markup:
//!
//! ```text
//! <｜DSML｜tool_calls>
//! <｜DSML｜invoke name="get_weather">
//! <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
//! <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
//! </｜DSML｜invoke>
//! </｜DSML｜tool_calls>
//! ```
//!
//! A parameter marked `string="true"` is its text exactly as written; any other value is
//! JSON, read to the first closing tag that leaves it readable as JSON, so a string may hold
//! a tag's text. The markup is read leniently, as quantized checkpoints write it: spaces, tabs
//! or line breaks may follow the `｜DSML｜` token, and the block may be named `tool_calls`,
//! `function_calls` or `calls`. It is read strictly otherwise: a block holds only invocations,
//! an invocation only parameters, every tag closes before the next opens, a tag holds only the
//! attributes the template writes, each once, and a parameter appears once. A block that
//! breaks any of these stays content, so no call is made from it. A message that opens a block
//! outside any other format's call is read only as DSML, and whole: text inside a DSML call
//! never becomes a call in another format, and nothing after the block streams out before the
//! turn ends; a block opened inside another format's call is that call's text.

use std::collections::HashMap;

use candle_core::Result;
use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use serde::Serialize;
use serde_json::Value;

use super::ToolFormatParser;
use crate::Tool;

/// The token every DSML tag starts with after its `<` or `</`.
pub(super) const DSML: &str = "｜DSML｜";
pub(super) const OPEN: &str = "<｜DSML｜";
const CLOSE: &str = "</｜DSML｜";
/// The whitespace a tag may hold, as the grammars' value lexemes know it: `[ \t\r\n]`. The
/// scanner reads no other whitespace in a tag, so a tag a grammar lets through as value text,
/// such as one with a no-break space before its name, is value text to the scanner too.
const WHITESPACE: [char; 4] = [' ', '\t', '\r', '\n'];
/// The names a block of calls is written under, longest first.
const BLOCK_NAMES: [&str; 3] = ["function_calls", "tool_calls", "calls"];
/// The block name the chat template uses, and the one a grammar writes.
const BLOCK: &str = "tool_calls";
/// The grammar rule that spells the `｜DSML｜` marker in every tag. Where the tokenizer has
/// the marker as a special token, `specialize_tool_call_grammar` makes the rule that
/// token, which text cannot match.
pub(super) const MARK_RULE: &str = r#"dsml_mark: "｜DSML｜""#;

pub struct DsmlParser;

impl ToolFormatParser for DsmlParser {
    /// A block that is not finished could become a call. Once a message has opened a block,
    /// finished or not, readable or not, the text from the block on is held back: a message
    /// that opens a block is read whole and only as DSML, so streamed out, the text after the
    /// block would be read on its own, and a block or another format's call in it could make
    /// a call the message does not. A complete call ends the turn, so a call costs nothing.
    fn could_be_tool_call(&self, text: &str) -> bool {
        block_start(text).is_some() || matches!(scan(text), Scan::More)
    }

    fn claim_start(&self, message: &str) -> Option<usize> {
        block_start(message)
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::DeepSeekDsml
    }

    fn tool_call_grammar(&self, tools: &[Tool], text: &str) -> TopLevelGrammar {
        continuation_grammar(tools, text)
            .unwrap_or_else(|| build_dsml_grammar(tools, Block::Rest { name: BLOCK }))
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        build_dsml_grammar(tools, Block::Whole)
    }

    fn parse(&self, message: &str) -> Result<Option<String>> {
        let Scan::Done(found) = scan(message) else {
            return Ok(None);
        };
        if found.calls.is_empty() {
            return Ok(None);
        }
        serde_json::to_string(&found.calls)
            .map(Some)
            .map_err(candle_core::Error::msg)
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct DsmlToolCall {
    name: String,
    arguments: Value,
}

/// How far a piece of DSML text reads.
enum Scan<T> {
    /// It reads as `T`.
    Done(T),
    /// It ends before what it started is complete, so more text could finish it.
    More,
    /// It cannot be read, whatever follows.
    Bad,
}

/// The calls in a message, in order, and the blocks they were read from.
struct Found {
    calls: Vec<DsmlToolCall>,
    blocks: Vec<FoundBlock>,
}

/// Where a block of calls sits in a message, and how many of the message's calls it holds.
struct FoundBlock {
    span: std::ops::Range<usize>,
    calls: usize,
}

/// A tag at the start of `text` that opens (or, with `closing`, closes) one of `names`, as
/// `(attributes, length)`.
fn tag_at<'a>(text: &'a str, closing: bool, names: &[&str]) -> Scan<(&'a str, usize)> {
    let marker = if closing { CLOSE } else { OPEN };
    let Some(after) = text.strip_prefix(marker) else {
        return if marker.starts_with(text) {
            Scan::More
        } else {
            Scan::Bad
        };
    };
    let rest = after.trim_start_matches(WHITESPACE);
    if rest.is_empty() {
        return Scan::More;
    }
    let mut could_become = false;
    for name in names {
        if let Some(tail) = rest.strip_prefix(name) {
            if tail.is_empty() {
                return Scan::More;
            }
            // The name ends at whitespace or the tag's end, so `calls` does not read `callsX`.
            if !tail.starts_with(|c: char| c == '>' || WHITESPACE.contains(&c)) {
                continue;
            }
            // The tag ends at its `>`; a `<` first means another tag began inside it.
            return match tail.find(['<', '>']) {
                Some(end) if tail[end..].starts_with('>') => {
                    Scan::Done((&tail[..end], text.len() - tail.len() + end + 1))
                }
                Some(_) => Scan::Bad,
                None => Scan::More,
            };
        }
        could_become |= name.starts_with(rest);
    }
    if could_become {
        Scan::More
    } else {
        Scan::Bad
    }
}

/// A tag's attributes, as the `key="value"` pairs that can be read from it, and whether the tag
/// holds exactly what the template writes: pairs set apart by whitespace, each key among
/// `keys` and given once. The pairs are read even from a tag that does not, so a parameter's
/// `string` flag still frames its value while the stray attribute makes the block content.
fn attribute_pairs<'a>(text: &'a str, keys: &[&str]) -> (Vec<(&'a str, &'a str)>, bool) {
    let mut pairs: Vec<(&str, &str)> = Vec::new();
    let mut well_formed = true;
    let mut rest = text;
    loop {
        let pair = rest.trim_start_matches(WHITESPACE);
        if pair.is_empty() {
            return (pairs, well_formed);
        }
        if pair.len() == rest.len() {
            well_formed = false;
        }
        let Some((key, after)) = pair.split_once("=\"") else {
            return (pairs, false);
        };
        let Some((value, after)) = after.split_once('"') else {
            return (pairs, false);
        };
        if !keys.contains(&key) || pairs.iter().any(|(seen, _)| *seen == key) {
            well_formed = false;
        } else {
            pairs.push((key, value));
        }
        rest = after;
    }
}

/// The value of attribute `key` among `pairs`.
fn attribute<'a>(pairs: &[(&str, &'a str)], key: &str) -> Option<&'a str> {
    pairs
        .iter()
        .find_map(|&(name, value)| (name == key).then_some(value))
}

fn leading_whitespace(text: &str) -> usize {
    text.len() - text.trim_start_matches(WHITESPACE).len()
}

/// A parameter's value, which runs from the end of its opening tag, the start of `text`, to
/// its own closing tag: anything else in it, DSML tags included, is part of the value. The
/// flag says whether the closing tag carried attributes, which the template never writes.
///
/// A `json` value that begins a JSON string, object or array may spell a closing tag inside a
/// string, so it ends at the closing tag after its JSON, found in one pass over it, once the
/// JSON reads; until then, more text could finish it. Any other value ends at the first
/// closing tag, as no JSON it could be holds one; a value that does not parse is read as text.
fn value_at(text: &str, json: bool) -> Scan<(&str, usize, bool)> {
    let lead = leading_whitespace(text);
    if json && text[lead..].starts_with(['"', '{', '[']) {
        let Some(end) = json_text_end(text, lead) else {
            return Scan::More;
        };
        let close = end + leading_whitespace(&text[end..]);
        return match tag_at(&text[close..], true, &["parameter"]) {
            Scan::Done((attributes, length)) if is_json(&text[..close]) => {
                Scan::Done((&text[..close], close + length, !bare(attributes)))
            }
            _ => Scan::More,
        };
    }
    let mut from = 0;
    while let Some(offset) = text[from..].find(CLOSE) {
        let at = from + offset;
        match tag_at(&text[at..], true, &["parameter"]) {
            Scan::Done((attributes, length)) => {
                return Scan::Done((&text[..at], at + length, !bare(attributes)));
            }
            Scan::More => return Scan::More,
            Scan::Bad => from = at + CLOSE.len(),
        }
    }
    Scan::More
}

/// Where the JSON string, object or array that begins at `start` in `text` ends, read in one
/// pass: at the quote closing its string, or the bracket closing its first, counting only
/// brackets outside its strings, where `\` escapes the next character. `None` while it has not.
fn json_text_end(text: &str, start: usize) -> Option<usize> {
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
                if depth == 0 {
                    return Some(start + offset + 1);
                }
            }
            continue;
        }
        match c {
            '"' => inside = true,
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(start + offset + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether `text` is one JSON value in form, which reading it as `IgnoredAny` checks without
/// converting its numbers. A number too large to hold is JSON in form, so a value holding one
/// ends where its JSON does, and is refused for what it holds (`invocation_at`) rather than
/// read on past its closing tag.
fn is_json(text: &str) -> bool {
    serde_json::from_str::<serde::de::IgnoredAny>(text.trim()).is_ok()
}

/// Whether a tag's attributes are none at all.
fn bare(attributes: &str) -> bool {
    attributes.trim_matches(WHITESPACE).is_empty()
}

/// One invocation, from the end of its opening tag, the start of `text`, through its closing
/// tag, and whether it breaks a rule of the markup while its tags stay in place: a parameter
/// without a name or with attributes the template does not write, a parameter given twice,
/// or a closing tag with attributes. Such an invocation still has a known end.
fn invocation_at(text: &str) -> Scan<(serde_json::Map<String, Value>, usize, bool)> {
    let mut arguments = serde_json::Map::new();
    let mut malformed = false;
    let mut at = 0;
    loop {
        at += leading_whitespace(&text[at..]);
        let rest = &text[at..];
        if rest.is_empty() {
            return Scan::More;
        }
        let close = tag_at(rest, true, &["invoke"]);
        if let Scan::Done((attributes, length)) = close {
            return Scan::Done((arguments, at + length, malformed || !bare(attributes)));
        }
        let (attributes, length) = match tag_at(rest, false, &["parameter"]) {
            Scan::Done(tag) => tag,
            Scan::More => return Scan::More,
            Scan::Bad if matches!(close, Scan::More) => return Scan::More,
            Scan::Bad => return Scan::Bad,
        };
        let (attributes, well_formed) = attribute_pairs(attributes, &["name", "string"]);
        let name = attribute(&attributes, "name");
        // A parameter name the markup cannot write (`""`, `1st`, `a b`) is one no grammar would
        // have written, and no tool declares: the invocation is content, its framing kept.
        malformed |= !well_formed || !name.is_some_and(writable_name);
        at += length;
        let json = attribute(&attributes, "string") != Some("true");
        let (raw, length, closed_with_attributes) = match value_at(&text[at..], json) {
            Scan::Done(value) => value,
            Scan::More => return Scan::More,
            Scan::Bad => return Scan::Bad,
        };
        malformed |= closed_with_attributes;
        at += length;
        let value = if !json {
            Value::String(raw.to_string())
        } else if let Ok(value) = serde_json::from_str(raw.trim()) {
            value
        } else {
            // Text that is not JSON is read as text, as a model writing without a grammar may
            // put it in a `string="false"` value. JSON the parser cannot hold, a number out of
            // range, is no value to call with: read as text, it would call with a string where
            // the model wrote a number.
            malformed |= is_json(raw);
            Value::String(raw.to_string())
        };
        // A parameter given twice has no one value to call with.
        if let Some(name) = name {
            malformed |= arguments.insert(name.to_string(), value).is_some();
        }
    }
}

/// The name of the block a complete opening tag `tag` opens.
fn block_name(tag: &str) -> &'static str {
    let rest = tag
        .strip_prefix(OPEN)
        .unwrap_or(tag)
        .trim_start_matches(WHITESPACE);
    BLOCK_NAMES
        .into_iter()
        .find(|name| rest.starts_with(name))
        .unwrap_or(BLOCK)
}

/// One block's calls, from the end of its opening tag, the start of `text`, through its
/// closing tag under the `name` it opened with, and whether it breaks a rule of the markup
/// while its tags stay in place: no invocation, an invocation without a name or with
/// attributes the template does not write, a malformed invocation, or a closing tag with
/// attributes.
fn block_at(text: &str, name: &str) -> Scan<(Vec<DsmlToolCall>, usize, bool)> {
    let mut calls = Vec::new();
    let mut malformed = false;
    let mut at = 0;
    loop {
        at += leading_whitespace(&text[at..]);
        let rest = &text[at..];
        if rest.is_empty() {
            return Scan::More;
        }
        let close = tag_at(rest, true, &[name]);
        if let Scan::Done((attributes, length)) = close {
            let malformed = malformed || calls.is_empty() || !bare(attributes);
            return Scan::Done((calls, at + length, malformed));
        }
        let (attributes, length) = match tag_at(rest, false, &["invoke"]) {
            Scan::Done(tag) => tag,
            Scan::More => return Scan::More,
            Scan::Bad if matches!(close, Scan::More) => return Scan::More,
            Scan::Bad => return Scan::Bad,
        };
        let (attributes, well_formed) = attribute_pairs(attributes, &["name"]);
        let name = attribute(&attributes, "name");
        malformed |= !well_formed || name.is_none();
        at += length;
        let (arguments, length, invocation_malformed) = match invocation_at(&text[at..]) {
            Scan::Done(invocation) => invocation,
            Scan::More => return Scan::More,
            Scan::Bad => return Scan::Bad,
        };
        malformed |= invocation_malformed;
        at += length;
        if let Some(name) = name {
            calls.push(DsmlToolCall {
                name: name.to_string(),
                arguments: Value::Object(arguments),
            });
        }
    }
}

/// How a DSML tag at an offset of a message frames what follows it.
pub(super) enum Frame {
    /// The tag opens no block: it is text.
    NotABlock,
    /// It opens a block that ends at this offset, read or not.
    Block(usize),
    /// It opens a block, or may, that has not ended.
    Open,
    /// It opens a block whose tags cannot be read, which leaves no place to resume.
    Unreadable,
}

/// How the DSML tag at `start` in `text` frames what follows it, for `call_spans`.
pub(super) fn frame(text: &str, start: usize) -> Frame {
    match tag_at(&text[start..], false, &BLOCK_NAMES) {
        Scan::Done((_, open)) => {
            let name = block_name(&text[start..start + open]);
            match block_at(&text[start + open..], name) {
                Scan::Done((_, length, _)) => Frame::Block(start + open + length),
                Scan::More => Frame::Open,
                Scan::Bad => Frame::Unreadable,
            }
        }
        Scan::More => Frame::Open,
        Scan::Bad if names_block(&text[start..]) => Frame::Unreadable,
        Scan::Bad => Frame::NotABlock,
    }
}

/// Reads every block of calls in `message` outside any other format's call, as `call_spans`
/// finds them. A block that breaks a rule of the markup while its tags stay in place is
/// content, and reading resumes after it; its values are read as values, so a block written
/// inside one never becomes a call. A block whose tags cannot be read leaves no place to resume,
/// so the message makes no call.
fn scan(message: &str) -> Scan<Found> {
    let mut found = Found {
        calls: Vec::new(),
        blocks: Vec::new(),
    };
    let spans = super::call_spans(message);
    let blocks = spans
        .iter()
        .filter(|span| span.format == Some(super::ToolCallFormat::DeepSeekDsml));
    for span in blocks {
        let end = match span.end {
            super::SpanEnd::At(end) => end,
            super::SpanEnd::Open => return Scan::More,
            super::SpanEnd::Unreadable => return Scan::Bad,
        };
        let start = span.open;
        let Scan::Done((attributes, open)) = tag_at(&message[start..], false, &BLOCK_NAMES) else {
            return Scan::Bad;
        };
        let name = block_name(&message[start..start + open]);
        let Scan::Done((calls, _, malformed)) = block_at(&message[start + open..end], name) else {
            return Scan::Bad;
        };
        if !malformed && bare(attributes) {
            found.blocks.push(FoundBlock {
                span: start..end,
                calls: calls.len(),
            });
            found.calls.extend(calls);
        }
    }
    if partial_opener_outside(message, &spans).is_some() {
        Scan::More
    } else {
        Scan::Done(found)
    }
}

/// Where `message` ends partway through the marker that opens a block, outside any other
/// format's call: prefixes of the marker end in different characters, so at most one of them
/// ends the message.
fn partial_opener_outside(message: &str, spans: &[super::Span]) -> Option<usize> {
    let after = match spans.last() {
        Some(super::Span {
            end: super::SpanEnd::At(end),
            ..
        }) => *end,
        Some(_) => return None,
        None => 0,
    };
    OPEN.char_indices()
        .skip(1)
        .map(|(at, _)| at)
        .find(|&at| message.ends_with(&OPEN[..at]))
        .map(|at| message.len() - at)
        .filter(|&start| start >= after)
}

/// Where `message` opens its first block of calls outside any other format's call, finished
/// or not, or names one in a tag that cannot be read: such a message calls nothing, in DSML
/// or in another format. A message that ends inside a tag that could still open a block, its
/// marker included, is claimed as well, so an example in another format earlier in it is not
/// dispatched as a call while the block is still being written.
fn block_start(message: &str) -> Option<usize> {
    let spans = super::call_spans(message);
    spans
        .iter()
        .find(|span| span.format == Some(super::ToolCallFormat::DeepSeekDsml))
        .map(|span| span.open)
        .or_else(|| partial_opener_outside(message, &spans))
}

/// Whether the DSML tag at the start of `text` names a block, whether or not the tag can be
/// read.
fn names_block(text: &str) -> bool {
    let rest = text
        .strip_prefix(OPEN)
        .unwrap_or(text)
        .trim_start_matches(WHITESPACE);
    BLOCK_NAMES.into_iter().any(|name| {
        rest.strip_prefix(name).is_some_and(|tail| {
            tail.starts_with(|c: char| c == '>' || c == '<' || WHITESPACE.contains(&c))
        })
    })
}

/// Whether `message` opens a block of calls outside any other format's call, which claims the
/// message for DSML.
pub(super) fn claims(message: &str) -> bool {
    block_start(message).is_some()
}

/// Whether `message` opens a block of calls it does not finish, the block's opening tag
/// included: text cut short partway through one could still be a call.
pub(super) fn unfinished_block(message: &str) -> bool {
    matches!(scan(message), Scan::More)
}

/// `message` without its blocks of calls: the text the model wrote around them.
pub(super) fn strip_calls(message: &str) -> String {
    strip_blocks_making_calls(message, |_| true)
}

/// `message` without the blocks that made a call, where `made` says whether the call at an
/// index among the message's calls, in order, was made. A block none of whose calls were made
/// is content, as a malformed one is.
pub(super) fn strip_blocks_making_calls(message: &str, made: impl Fn(usize) -> bool) -> String {
    let Scan::Done(found) = scan(message) else {
        return message.to_string();
    };
    let mut out = String::new();
    let mut cursor = 0;
    let mut first = 0;
    for block in found.blocks {
        let calls = first..first + block.calls;
        first = calls.end;
        if !calls.into_iter().any(&made) {
            continue;
        }
        out.push_str(&message[cursor..block.span.start]);
        cursor = block.span.end;
    }
    out.push_str(&message[cursor..]);
    out
}

/// `value` as a grammar's string literal, escaped as a JSON string is, control characters
/// included, which a literal cannot hold as they are.
fn lark_literal(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| format!("{value:?}"))
}

/// A string parameter's value and its closing tag, as a grammar rule.
const VALUE_THEN_CLOSE: &str = "dsml_vc";
/// The sub-grammar for any JSON value, the `string="false"` value of a parameter no schema holds.
const ANY_JSON: &str = "dsml_json";

/// The value rules where the marker is text: the value runs to its first closing tag, which
/// ends it, and holds any text but a closing tag of a parameter, where the scanner would end it
/// (`value_at`), so it may hold other DSML tags. The value is lazy: the parser's lexer extends a
/// lexeme as far as it can, so a value written as the longest text before a `</` would run on
/// past its closing tag, and never close.
pub(super) const VALUE_RULES: &str = r#"dsml_vc[suffix="</｜DSML｜parameter>\n"]: /(?s:.*)/ & ~/(?s:.*)<\/｜DSML｜[ \t\r\n]*parameter[> \t\r\n](?s:.*)/"#;

/// The value rules where the marker is a special token: text between markers, where a marker
/// after `</` begins the closing tag only when `parameter` follows it, so other DSML tags are
/// part of the value, and markers may follow one another. No text lexeme holds the marker's
/// text: the parser reads a marker spelled in ordinary tokens as the token, so a value spelling
/// its closing tag that way would read as the end of the value and parameters after it.
pub(super) const VALUE_RULES_WITH_MARK_TOKEN: &str = r#"dsml_vc: dsml_vc_open | dsml_mark dsml_vc_open
dsml_vc_open: DSML_LT_SLASH dsml_mark dsml_vc_closing | DSML_OTHER dsml_mark dsml_vc_open | dsml_mark dsml_vc_open
dsml_vc_closing: "parameter>\n" | DSML_NP_LT_SLASH dsml_mark dsml_vc_closing | DSML_NP_OTHER dsml_mark dsml_vc_open | dsml_mark dsml_vc_open
DSML_LT_SLASH: /(?s:.*)<\// & ~/(?s:.*)｜DSML｜(?s:.*)/
DSML_OTHER: /(?s:.+)/ & ~/(?s:.*)<\// & ~/(?s:.*)｜DSML｜(?s:.*)/
DSML_NP_LT_SLASH: /(?s:.*)<\// & ~/[ \t\r\n]*parameter[> \t\r\n](?s:.*)/ & ~/(?s:.*)｜DSML｜(?s:.*)/
DSML_NP_OTHER: /(?s:.+)/ & ~/(?s:.*)<\// & ~/[ \t\r\n]*parameter[> \t\r\n](?s:.*)/ & ~/(?s:.*)｜DSML｜(?s:.*)/"#;

/// A tag in a grammar: `<`, or `</` with `closing`, the marker, then the rest as text.
fn tag(closing: bool, rest: &str) -> String {
    format!(
        "{} dsml_mark {}",
        lark_literal(if closing { "</" } else { "<" }),
        lark_literal(rest)
    )
}

/// How much of a block a grammar writes.
#[derive(Clone, Copy)]
enum Block<'a> {
    /// The whole block, from its opening tag.
    Whole,
    /// The rest of a block the model has opened under `name`: what follows its opening tag and
    /// any whitespace the model wrote after it.
    Rest { name: &'a str },
}

/// The grammar for the rest of the block that `text`, the message so far, ends by opening:
/// its opening tag is complete, and only whitespace follows it. That block is the last the
/// message opens, after any it has closed, an empty or malformed one included, as a block
/// after one that makes no call still makes its own. The grammar writes the
/// invocations and the closing tag under the name the block opened with, so a call the model
/// writes on its own is held to the offered tools and to a strict tool's schema, as in the other
/// formats; a strict tool whose schema the grammar cannot hold (`unenforceable`) is written as
/// one that is not strict, and its calls are held to the schema when read. `None` once the
/// block's body has begun, where a grammar could not know what the model has written, and when
/// the opening tag cannot be read.
pub(super) fn continuation_grammar(tools: &[Tool], text: &str) -> Option<TopLevelGrammar> {
    let start = match super::call_spans(text).last() {
        Some(super::Span {
            format: Some(super::ToolCallFormat::DeepSeekDsml),
            open,
            end: super::SpanEnd::Open,
        }) => *open,
        _ => return None,
    };
    let Scan::Done((attributes, length)) = tag_at(&text[start..], false, &BLOCK_NAMES) else {
        return None;
    };
    if !bare(attributes) {
        return None;
    }
    // Whitespace after the opening tag, however much the token that closed the tag carried, is
    // the model's; the grammar matches none of it. Anything else begins the block's body.
    if !text[start + length..]
        .chars()
        .all(|c| WHITESPACE.contains(&c))
    {
        return None;
    }
    let name = block_name(&text[start..start + length]);
    Some(build_dsml_grammar(tools, Block::Rest { name }))
}

/// A grammar for DSML calls to `tools`, writing `block`.
fn build_dsml_grammar(tools: &[Tool], block: Block) -> TopLevelGrammar {
    let mut rules = Vec::new();
    let mut schemas = Vec::new();
    let mut invokes = Vec::new();
    for (index, tool) in tools.iter().enumerate() {
        let invoke = format!("invoke_{index}");
        let arguments = emit_arguments(tool, index, &mut rules, &mut schemas);
        rules.push(format!(
            "{invoke}: {open} {arguments} {close}",
            open = tag(false, &format!("invoke name=\"{}\">\n", tool.function.name)),
            close = tag(true, "invoke>\n"),
        ));
        invokes.push(invoke);
    }
    let calls = format!("({})+", invokes.join(" | "));
    let start = match block {
        Block::Whole => format!(
            "start: {} {calls} {}",
            tag(false, &format!("{BLOCK}>\n")),
            tag(true, &format!("{BLOCK}>"))
        ),
        // Whitespace the model may still write before the first call, after whatever it wrote
        // already, which the grammar does not see.
        Block::Rest { name } => {
            format!("start: DSML_WS? {calls} {}", tag(true, &format!("{name}>")))
        }
    };
    let mut lines = vec![start];
    lines.extend(rules);
    lines.extend([
        r"DSML_WS: /[ \t\r\n]+/".to_string(),
        // Any parameter: text as `string="true"`, or one complete JSON value as `string="false"`,
        // which the parser reads as JSON; a value that is not JSON would be read as text, and
        // one cut short would leave the block unfinished.
        format!(
            "dsml_parameter: {} PARAM_NAME {} (\"true\" {after} {VALUE_THEN_CLOSE} | \"false\" {after} @{ANY_JSON} {close})",
            tag(false, "parameter name=\""),
            lark_literal("\" string=\""),
            after = lark_literal("\">"),
            close = tag(true, "parameter>\n"),
        ),
        // A value is any text but its closing tag, which `</` and then the marker start, so
        // ordinary markup such as `</div>` is part of it. A lexeme reads the value with the
        // `</` that ends it: written apart, a value ending in `<` or `</` would read the
        // closing tag's `<` as more value.
        VALUE_RULES.to_string(),
        "PARAM_NAME: /[a-zA-Z_][a-zA-Z0-9_.-]*/".to_string(),
        MARK_RULE.to_string(),
    ]);
    let mut grammars = vec![GrammarWithLexer::from_lark(lines.join("\n"))];
    grammars.extend(schemas);
    // The value of a `string="false"` parameter of a tool that is not strict: any JSON.
    grammars.push(GrammarWithLexer {
        name: Some(ANY_JSON.to_string()),
        json_schema: Some(serde_json::json!({})),
        ..Default::default()
    });
    TopLevelGrammar {
        grammars,
        max_tokens: None,
    }
}

/// How a strict tool's property is written: as text, as one of a closed set of strings, or as
/// JSON held to its schema.
enum PropertyValue {
    Text,
    OneOf(Vec<String>),
    Json,
}

/// The keys a string property may have and still take any text: `format` is an annotation,
/// not a constraint.
const ANNOTATIONS: [&str; 7] = [
    "type",
    "description",
    "title",
    "default",
    "examples",
    "format",
    "$comment",
];

fn property_value(property: &Value) -> PropertyValue {
    let Some(object) = property.as_object() else {
        return PropertyValue::Json;
    };
    // A shortcut stands in for the schema only when it is the property's one constraint: on
    // a string, or with no type, and beside annotations alone.
    let alone = |keyword: &str| {
        object.get("type").is_none_or(|kind| kind == "string")
            && object
                .keys()
                .all(|key| key == keyword || ANNOTATIONS.contains(&key.as_str()))
    };
    let strings = |values: &[Value]| {
        values
            .iter()
            .map(|value| value.as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()
    };
    if alone("enum") {
        if let Some(values) = object
            .get("enum")
            .and_then(Value::as_array)
            .and_then(|v| strings(v))
        {
            return PropertyValue::OneOf(values);
        }
    }
    if alone("const") {
        if let Some(value) = object.get("const").and_then(Value::as_str) {
            return PropertyValue::OneOf(vec![value.to_string()]);
        }
    }
    if object.get("type").and_then(Value::as_str) == Some("string") && alone("type") {
        PropertyValue::Text
    } else {
        PropertyValue::Json
    }
}

/// The keys a strict tool's parameters schema may have for a required call's grammar to hold
/// the object it describes: the grammar writes each declared property at most once, its
/// required ones always, and no other. An annotation constrains nothing, so it is allowed.
const ENFORCED_ROOT_KEYS: [&str; 17] = [
    "type",
    "properties",
    "required",
    "additionalProperties",
    "description",
    "title",
    "$defs",
    "definitions",
    "$schema",
    "$id",
    "$anchor",
    "$comment",
    "examples",
    "default",
    "deprecated",
    "readOnly",
    "writeOnly",
];

/// Why a required call cannot hold one of `tools` to its strict schema: a constraint on the
/// object as a whole, which the parameter-by-parameter grammar cannot express.
pub(super) fn unenforceable(tools: &[Tool]) -> Option<String> {
    tools.iter().find_map(unenforceable_tool)
}

/// Why a required call cannot hold `tool` to its strict schema, if it cannot.
fn unenforceable_tool(tool: &Tool) -> Option<String> {
    let schema = tool.function.strict_parameters_schema()?;
    let object = schema.as_object()?;
    let name = &tool.function.name;
    if let Some(keyword) = object
        .keys()
        .find(|key| !ENFORCED_ROOT_KEYS.contains(&key.as_str()))
    {
        return Some(format!(
            "Tool '{name}' is strict, and its parameters schema constrains the object as a whole with `{keyword}`, which a required DeepSeek-V4 tool call cannot enforce. Remove `strict` from the tool, or the `{keyword}` from its schema."
        ));
    }
    // The grammar writes declared properties only, so a schema that admits others, as a map of
    // parameters does through `additionalProperties`, is one it cannot hold.
    if object
        .get("additionalProperties")
        .is_some_and(|additional| *additional != Value::Bool(false))
    {
        return Some(format!(
            "Tool '{name}' is strict, and its parameters schema admits parameters it does not declare (`additionalProperties`), which a required DeepSeek-V4 tool call cannot write. Declare the parameters in `properties` and set `additionalProperties` to `false`, or remove `strict` from the tool."
        ));
    }
    // The grammar writes declared properties only, so a required one must be declared.
    let declared = object.get("properties").and_then(Value::as_object);
    if let Some(unwritable) =
        declared.and_then(|declared| declared.keys().find(|property| !writable_name(property)))
    {
        return Some(format!(
            "Tool '{name}' is strict, and its parameter `{unwritable}` has a name a DeepSeek-V4 tool call cannot write: a name is letters, digits, `_`, `.` and `-`, and does not start with a digit. Rename the parameter, or remove `strict` from the tool."
        ));
    }
    // A `$id` inside the schema starts a schema resource of its own, against which the
    // references in it resolve; the grammar follows references against the root only.
    if let Some(pointer) = nested_identifier(&schema) {
        return Some(format!(
            "Tool '{name}' is strict, and its parameters schema declares `$id` at `{pointer}`, which starts a schema resource of its own, whose references a DeepSeek-V4 tool call's grammar cannot follow. Remove that `$id`, or `strict` from the tool."
        ));
    }
    // The grammar follows a reference into the schema only.
    if let Some(reference) = unfollowable_reference(&schema) {
        return Some(format!(
            "Tool '{name}' is strict, and its parameters schema refers to `{reference}`, which a DeepSeek-V4 tool call cannot follow: a reference names the schema's root (`#`), a pointer into it (`#/...`), or an `$anchor` it declares. Point the reference into the schema, or remove `strict` from the tool."
        ));
    }
    let required = object.get("required").and_then(Value::as_array)?;
    // A property whose schema is `false`, or refers to one, admits no value, so one that is
    // required cannot be written at all.
    let anchors = anchors(&schema);
    if let Some(unsatisfiable) = required.iter().filter_map(Value::as_str).find(|required| {
        declared
            .and_then(|declared| declared.get(*required))
            .is_some_and(|property| admits_nothing(&schema, property, &anchors))
    }) {
        return Some(format!(
            "Tool '{name}' is strict, and its parameters schema requires `{unsatisfiable}` with the schema `false`, which no value satisfies, so no call to it can be written. Allow a value for `{unsatisfiable}`, or stop requiring it."
        ));
    }
    let missing = required
        .iter()
        .filter_map(Value::as_str)
        .find(|required| declared.is_none_or(|declared| !declared.contains_key(*required)))?;
    Some(format!(
        "Tool '{name}' is strict, and its parameters schema requires `{missing}` without declaring it in `properties`, so a required DeepSeek-V4 tool call cannot write it. Declare `{missing}` in `properties`, or remove `strict` from the tool."
    ))
}

/// The names a tool's parameters `schema` gives its parameters, in order: those its
/// `properties` declare and its `required` lists, and those of every `allOf`, `anyOf` and
/// `oneOf` branch, followed through the local references the schema makes, as a call may
/// write any of them and may need some of them.
fn parameter_names(schema: &Value) -> Vec<String> {
    let anchors = anchors(schema);
    let mut names = std::collections::BTreeSet::new();
    let mut pending = vec![schema];
    let mut followed = std::collections::HashSet::new();
    while let Some(part) = pending.pop() {
        let Some(object) = part.as_object() else {
            continue;
        };
        if let Some(properties) = object.get("properties").and_then(Value::as_object) {
            names.extend(properties.keys().cloned());
        }
        if let Some(required) = object.get("required").and_then(Value::as_array) {
            names.extend(
                required
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string),
            );
        }
        for keyword in ["allOf", "anyOf", "oneOf"] {
            if let Some(branches) = object.get(keyword).and_then(Value::as_array) {
                pending.extend(branches);
            }
        }
        // A reference is followed once, so a schema that refers to itself is read through.
        if let Some(target) = object
            .get("$ref")
            .and_then(Value::as_str)
            .and_then(|reference| reference_target(reference, &anchors))
        {
            if followed.insert(target.clone()) {
                pending.extend(schema.pointer(&target));
            }
        }
    }
    names.into_iter().collect()
}

/// Whether a parameter name can be written in a tag's `name="..."` attribute and read back:
/// the names `PARAM_NAME` spells, letters, digits, `_`, `.` and `-`, not starting with a digit.
fn writable_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

/// The definition the whole tool schema is kept under in a property's schema.
const TOOL_SCHEMA: &str = "tool";

/// The schema of property `name` of the tool schema `root`, as a schema of its own: a reference
/// into a copy of the tool schema, kept under a definition, with every reference in the copy
/// rebased onto it. A reference in the tool schema is relative to the tool schema's root, so
/// one to the root's `$defs`, to another property, or into the property's own definitions
/// resolves as it does there; a property copied out alone would lose the first two, and could
/// bind the first to definitions of its own. The copy holds only the parts of the tool schema
/// the property reaches (`reached_copy`), each in its place, so a tool's grammar grows with its
/// schema rather than with its schema times its parameters; `anchors` are the tool schema's.
fn property_schema(root: &Value, name: &str, anchors: &HashMap<String, String>) -> Value {
    let property = format!("/properties/{}", pointer_segment(name));
    let mut tool = reached_copy(root, &property, anchors).unwrap_or_else(|| root.clone());
    rebase_references(&mut tool, &format!("#/$defs/{TOOL_SCHEMA}"));
    serde_json::json!({
        "$ref": format!("#/$defs/{TOOL_SCHEMA}{property}"),
        "$defs": { TOOL_SCHEMA: tool },
    })
}

/// A copy of the parts of `root` that the schema at `pointer` reaches: that schema, and each
/// schema a reference in a part copied names, transitively, each at its own place, so the
/// references resolve as they do in `root`. `None` where only a copy of the whole keeps them in
/// place: a part that is the root itself, sits inside an array, or is not found.
fn reached_copy(root: &Value, pointer: &str, anchors: &HashMap<String, String>) -> Option<Value> {
    let within = |pointer: &str, ancestor: &str| {
        pointer == ancestor
            || pointer
                .strip_prefix(ancestor)
                .is_some_and(|rest| rest.starts_with('/'))
    };
    let mut reached: Vec<String> = Vec::new();
    let mut pending = vec![pointer.to_string()];
    while let Some(pointer) = pending.pop() {
        if pointer.is_empty() {
            return None;
        }
        if reached.iter().any(|seen| within(&pointer, seen)) {
            continue;
        }
        let part = root.pointer(&pointer)?;
        for inner in schema_pointers(part) {
            if let Some(reference) = part
                .pointer(&inner)
                .and_then(|schema| schema.get("$ref"))
                .and_then(Value::as_str)
            {
                pending.push(reference_target(reference, anchors)?);
            }
        }
        reached.retain(|seen| !within(seen, &pointer));
        reached.push(pointer);
    }
    /// `source`'s part at `segments`, placed at the same segments in `target`.
    fn place(target: &mut Value, source: &Value, segments: &[String]) -> Option<()> {
        let (first, rest) = segments.split_first()?;
        let source = source.as_object()?.get(first)?;
        let map = target.as_object_mut()?;
        if rest.is_empty() {
            map.insert(first.clone(), source.clone());
            return Some(());
        }
        let next = map
            .entry(first.clone())
            .or_insert_with(|| Value::Object(serde_json::Map::new()));
        place(next, source, rest)
    }
    let mut copy = Value::Object(serde_json::Map::new());
    for pointer in &reached {
        let segments: Vec<String> = pointer
            .split('/')
            .skip(1)
            .map(|segment| segment.replace("~1", "/").replace("~0", "~"))
            .collect();
        place(&mut copy, root, &segments)?;
    }
    Some(copy)
}

/// The pointer, in its document, of the schema `reference` names: `#` is the root, `#/...` a
/// pointer from it, and `#name` the schema declaring `$anchor: "name"`, among `anchors`.
fn reference_target(reference: &str, anchors: &HashMap<String, String>) -> Option<String> {
    let fragment = reference.strip_prefix('#')?;
    if fragment.is_empty() || fragment.starts_with('/') {
        Some(fragment.to_string())
    } else {
        anchors.get(fragment).cloned()
    }
}

/// Whether `schema`, in the tool schema `root` whose anchors are `anchors`, admits no value: it
/// is `false`, or a `$ref` that leads, through more of them, to `false`.
fn admits_nothing(root: &Value, schema: &Value, anchors: &HashMap<String, String>) -> bool {
    let mut schema = schema;
    // A cycle of references leads nowhere; the bound ends one.
    for _ in 0..64 {
        if *schema == Value::Bool(false) {
            return true;
        }
        let Some(target) = schema
            .get("$ref")
            .and_then(Value::as_str)
            .and_then(|reference| reference_target(reference, anchors))
            .and_then(|pointer| root.pointer(&pointer))
        else {
            return false;
        };
        schema = target;
    }
    false
}

/// `key` as a segment of a JSON pointer, which escapes `~` and `/`.
fn pointer_segment(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

/// Every `$ref` in `schema` into its own document made relative to `base`, as a pointer, which
/// is all the grammar follows: `#` names the root, `#/...` is a pointer from it, and `#name` is
/// the schema declaring `$anchor: "name"`, written as the pointer to it. A reference the grammar
/// cannot follow (`unfollowable_reference`) is left as written.
fn rebase_references(schema: &mut Value, base: &str) {
    let anchors = anchors(schema);
    for pointer in schema_pointers(schema) {
        let Some(Value::String(reference)) = schema
            .pointer_mut(&pointer)
            .and_then(|schema| schema.get_mut("$ref"))
        else {
            continue;
        };
        if let Some(rebased) = rebased(reference, base, &anchors) {
            *reference = rebased;
        }
    }
}

/// `reference` as a pointer from `base`: `#` is `base`, `#/...` is `base/...`, and `#name` is
/// `base` and then the pointer to the schema declaring the anchor `name`, when one does. Any
/// other reference has no pointer the grammar can follow.
fn rebased(reference: &str, base: &str, anchors: &HashMap<String, String>) -> Option<String> {
    let fragment = reference.strip_prefix('#')?;
    if fragment.is_empty() {
        return Some(base.to_string());
    }
    if fragment.starts_with('/') {
        return Some(format!("{base}{fragment}"));
    }
    anchors
        .get(fragment)
        .map(|pointer| format!("{base}{pointer}"))
}

/// The first `$ref` in `schema` the grammar cannot follow, if any. The grammar follows a
/// reference into the schema's own document only: `#`, a pointer `#/...` that names something
/// in it, or the name of an `$anchor` it declares.
fn unfollowable_reference(schema: &Value) -> Option<String> {
    let anchors = anchors(schema);
    schema_pointers(schema).into_iter().find_map(|pointer| {
        let reference = schema.pointer(&pointer)?.get("$ref")?.as_str()?;
        let followable = match reference.strip_prefix('#') {
            Some("") => true,
            Some(fragment) if fragment.starts_with('/') => schema.pointer(fragment).is_some(),
            Some(fragment) => anchors.contains_key(fragment),
            None => false,
        };
        (!followable).then(|| reference.to_string())
    })
}

/// The pointer of the first schema inside `schema`, the root excepted, that declares an `$id`,
/// if any: such a schema is a resource of its own, against which the references in it resolve.
fn nested_identifier(schema: &Value) -> Option<String> {
    schema_pointers(schema).into_iter().find(|pointer| {
        !pointer.is_empty()
            && schema
                .pointer(pointer)
                .is_some_and(|schema| schema.get("$id").is_some())
    })
}

/// The pointer of each schema in `schema` that declares an `$anchor`, by the anchor's name. The
/// first declaration of a name is kept; a schema declaring one twice is not a valid schema.
fn anchors(schema: &Value) -> HashMap<String, String> {
    let mut anchors = HashMap::new();
    for pointer in schema_pointers(schema) {
        let Some(name) = schema
            .pointer(&pointer)
            .and_then(|schema| schema.get("$anchor"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        anchors.entry(name.to_string()).or_insert(pointer);
    }
    anchors
}

/// The JSON pointer of each schema in `schema`, itself included. Only the keywords that hold
/// schemas are entered: `const`, `enum`, `default`, `examples` and the annotations hold data,
/// where a `$ref` or `$anchor` key is a value's own.
fn schema_pointers(schema: &Value) -> Vec<String> {
    fn collect(schema: &Value, pointer: String, out: &mut Vec<String>) {
        let Some(object) = schema.as_object() else {
            return;
        };
        for (key, value) in object {
            let under = format!("{pointer}/{}", pointer_segment(key));
            match key.as_str() {
                // A schema, a map of schemas, or an array of schemas.
                "additionalProperties"
                | "unevaluatedProperties"
                | "propertyNames"
                | "items"
                | "additionalItems"
                | "unevaluatedItems"
                | "contains"
                | "not"
                | "if"
                | "then"
                | "else"
                | "contentSchema"
                | "properties"
                | "patternProperties"
                | "$defs"
                | "definitions"
                | "dependentSchemas"
                | "allOf"
                | "anyOf"
                | "oneOf"
                | "prefixItems" => {
                    if value.is_object() && !is_schema_map(key) {
                        collect(value, under, out);
                    } else {
                        each(value, under, out);
                    }
                }
                // A map of schemas or of property-name arrays.
                "dependencies" => each(value, under, out),
                _ => {}
            }
        }
        out.push(pointer);
    }
    fn each(values: &Value, pointer: String, out: &mut Vec<String>) {
        if let Some(map) = values.as_object() {
            for (name, schema) in map {
                collect(schema, format!("{pointer}/{}", pointer_segment(name)), out);
            }
        } else if let Some(items) = values.as_array() {
            for (index, item) in items.iter().enumerate() {
                collect(item, format!("{pointer}/{index}"), out);
            }
        }
    }
    let mut out = Vec::new();
    collect(schema, String::new(), &mut out);
    out
}

/// Whether `keyword` holds a map from names to schemas, rather than one schema.
fn is_schema_map(keyword: &str) -> bool {
    matches!(
        keyword,
        "properties" | "patternProperties" | "$defs" | "definitions" | "dependentSchemas"
    )
}

/// The rule for a call's parameters. A strict tool's parameters are its declared properties,
/// each once, with its required ones present and every value held to its schema: a string
/// property that takes any text as text, one with an `enum` or `const` of strings as one of
/// them, and any other as JSON. A tool that is not strict takes any parameters.
fn emit_arguments(
    tool: &Tool,
    tool_index: usize,
    rules: &mut Vec<String>,
    schemas: &mut Vec<GrammarWithLexer>,
) -> String {
    let arguments = format!("arguments_{tool_index}");
    // A strict tool whose schema the grammar cannot hold is written as one that is not strict;
    // its calls are held to the schema when read.
    let strict = tool.function.strict == Some(true);
    let schema = tool
        .function
        .strict_parameters_schema()
        .filter(|_| unenforceable_tool(tool).is_none());
    let Some(schema) = schema else {
        // A tool that is not strict takes any value. Its declared parameters are written at
        // most once each, as the parser reads a parameter given twice as malformed; one that
        // declares none takes any parameters.
        let Some(parameters) = tool.function.parameters.as_ref().filter(|parameters| {
            parameters
                .get("properties")
                .and_then(Value::as_object)
                .is_some_and(|declared| !declared.is_empty())
        }) else {
            rules.push(format!("{arguments}: dsml_parameter*"));
            return arguments;
        };
        // A name the markup cannot write is left out: the model could not write it either.
        let schema = Value::Object(
            parameters
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        let names = parameter_names(&schema)
            .into_iter()
            .filter(|name| writable_name(name))
            .collect::<Vec<_>>();
        let mut sequence = Vec::new();
        // Each parameter's tag is written as the same opening, then its name, so the name of a
        // parameter written as another (`other_parameter_…`) is read as a name, whatever
        // declared name it begins as: a tag spelled whole around a declared name would leave
        // the parser's lexer reading `command` as the start of `cwd`, and failing.
        for (property_index, name) in names.iter().enumerate() {
            let parameter = format!("parameter_{tool_index}_{property_index}");
            rules.push(format!(
                "{parameter}: {open} {name} {string} (\"true\" {after} {VALUE_THEN_CLOSE} | \"false\" {after} @{ANY_JSON} {close})",
                open = tag(false, "parameter name=\""),
                name = lark_literal(name),
                string = lark_literal("\" string=\""),
                after = lark_literal("\">"),
                close = tag(true, "parameter>\n"),
            ));
            sequence.push(format!("{parameter}?"));
        }
        // A strict tool's schema may require a parameter it gives nowhere `parameter_names`
        // reads, through `dependentRequired` or `if` and `then`, so the grammar standing in
        // for one it cannot hold also takes any other parameter the markup can write; the call
        // is held to the whole schema when read.
        if strict {
            let other = format!("other_parameter_{tool_index}");
            let other_name = format!("OTHER_NAME_{tool_index}");
            rules.push(format!(
                "{other}: {open} {other_name} {string} (\"true\" {after} {VALUE_THEN_CLOSE} | \"false\" {after} @{ANY_JSON} {close})",
                open = tag(false, "parameter name=\""),
                string = lark_literal("\" string=\""),
                after = lark_literal("\">"),
                close = tag(true, "parameter>\n"),
            ));
            let declared = names
                .iter()
                .map(|name| regex::escape(name))
                .collect::<Vec<_>>()
                .join("|");
            rules.push(format!(
                "{other_name}: /[a-zA-Z_][a-zA-Z0-9_.-]*/ & ~/(?:{declared})/"
            ));
            sequence.push(format!("{other}*"));
        }
        rules.push(format!("{arguments}: {}", sequence.join(" ")));
        return arguments;
    };
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        rules.push(format!("{arguments}:"));
        return arguments;
    };
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .collect::<std::collections::HashSet<_>>()
        })
        .unwrap_or_default();
    let mut names = properties.keys().collect::<Vec<_>>();
    names.sort();
    let mut sequence = Vec::new();
    let anchors = anchors(&schema);
    for (property_index, name) in names.into_iter().enumerate() {
        let parameter = format!("parameter_{tool_index}_{property_index}");
        let property = &properties[name];
        // A property whose schema is `false`, or refers to one, admits no value: the call
        // leaves it out.
        if admits_nothing(&schema, property, &anchors) {
            continue;
        }
        let open = tag(false, &format!("parameter name=\"{name}\" string=\""));
        let close = tag(true, "parameter>\n");
        let as_text = lark_literal("true\">");
        let as_json = lark_literal("false\">");
        // A value written as JSON, held to the property's schema as a sub-grammar.
        let mut held = || {
            let grammar = format!("dsml_schema_{tool_index}_{property_index}");
            schemas.push(GrammarWithLexer {
                name: Some(grammar.clone()),
                json_schema: Some(property_schema(&schema, name, &anchors)),
                ..Default::default()
            });
            format!("{as_json} @{grammar} {close}")
        };
        let written = match property_value(property) {
            // Text, or a JSON string: text cannot hold the parameter's closing tag.
            PropertyValue::Text => format!("{as_text} {VALUE_THEN_CLOSE} | {}", held()),
            PropertyValue::OneOf(values) => {
                // A value holding the parameter's closing tag is written as JSON with its `<`
                // escaped, so the parser cannot end the value inside it.
                let (raw, closing): (Vec<&String>, Vec<&String>) =
                    values.iter().partition(|value| !value.contains(CLOSE));
                let alternatives = |values: Vec<String>| {
                    values
                        .iter()
                        .map(|value| lark_literal(value))
                        .collect::<Vec<_>>()
                        .join(" | ")
                };
                let mut written = Vec::new();
                if !raw.is_empty() {
                    let raw = alternatives(raw.into_iter().cloned().collect());
                    written.push(format!("{as_text} ({raw}) {close}"));
                }
                if !closing.is_empty() {
                    let json = alternatives(
                        closing
                            .iter()
                            .map(|value| {
                                Value::String((*value).clone())
                                    .to_string()
                                    .replace('<', "\\u003c")
                            })
                            .collect(),
                    );
                    written.push(format!("{as_json} ({json}) {close}"));
                }
                written.join(" | ")
            }
            PropertyValue::Json => held(),
        };
        rules.push(format!("{parameter}: {open} ({written})"));
        sequence.push(if required.contains(name.as_str()) {
            parameter
        } else {
            format!("{parameter}?")
        });
    }
    rules.push(format!("{arguments}: {}", sequence.join(" ")));
    arguments
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::tools::parsers::specialize_tool_call_grammar;

    fn calls(message: &str) -> Option<Value> {
        DsmlParser
            .parse(message)
            .expect("parse")
            .map(|json| serde_json::from_str(&json).expect("JSON"))
    }

    fn bash(strict: bool) -> Tool {
        serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {
                "name": "Bash",
                "strict": strict,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                    "required": ["command"],
                },
            },
        }))
        .expect("tool")
    }

    #[test]
    fn the_templates_markup_reads_as_calls_with_typed_arguments() {
        let message = "Checking.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n<｜DSML｜parameter name=\"units\" string=\"false\">[\"c\", \"f\"]</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"now\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        assert_eq!(
            calls(message),
            Some(serde_json::json!([
                {"name": "get_weather", "arguments": {"city": "Paris", "days": 3, "units": ["c", "f"]}},
                {"name": "now", "arguments": {}},
            ]))
        );
        assert_eq!(strip_calls(message), "Checking.\n\n");
    }

    /// Verbatim from DeepSeek-V4.1-Flash (Q2_K GGUF) served by the ring backend: whitespace
    /// after the DSML token, and the block named `calls`.
    #[test]
    fn a_quantized_checkpoints_spacing_reads_the_same() {
        let message = "I'll run that command for you.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"Bash\">\n<｜DSML｜ parameter name=\"command\" string=\"true\">python3 --version</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        assert_eq!(
            calls(message),
            Some(serde_json::json!([
                {"name": "Bash", "arguments": {"command": "python3 --version"}},
            ]))
        );
        assert_eq!(strip_calls(message), "I'll run that command for you.\n\n");
    }

    #[test]
    fn a_string_parameter_keeps_its_text_exactly_tags_included() {
        let message = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"write\"><｜DSML｜parameter name=\"body\" string=\"true\">  <div>{\"not\": \"parsed\"}</div></｜DSML｜invoke>\n</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";
        assert_eq!(
            calls(message),
            Some(serde_json::json!([
                {"name": "write", "arguments": {"body": "  <div>{\"not\": \"parsed\"}</div></｜DSML｜invoke>\n"}},
            ]))
        );
    }

    #[test]
    fn an_unfinished_block_is_not_a_call_yet_but_could_become_one() {
        let whole = "Text <｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"Bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">ls</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        // Every prefix from the opening `<` on.
        let mut cut = whole.find('<').expect("a tag") + 1;
        while cut < whole.len() {
            let prefix = &whole[..cut];
            assert_eq!(calls(prefix), None, "{prefix:?}");
            assert!(DsmlParser.could_be_tool_call(prefix), "{prefix:?}");
            cut += whole[cut..].chars().next().expect("a char").len_utf8();
        }
        assert!(calls(whole).is_some());
        // The message is DSML's from the block on; the complete call ends the turn.
        assert!(DsmlParser.could_be_tool_call(whole));
    }

    #[test]
    fn text_without_markup_is_not_a_call() {
        for text in [
            "a < b",
            "x <｜DSML｜ is mentioned>",
            "<tool_calls>[]</tool_calls>",
            "",
        ] {
            assert_eq!(calls(text), None, "{text:?}");
            assert!(!DsmlParser.could_be_tool_call(text), "{text:?}");
        }
    }

    /// regression test for spiceai/mistral.rs#54: malformed blocks made calls. A block that
    /// breaks a rule while its tags stay in place is content, and reading resumes after it; one
    /// whose tags cannot be read is content with everything after it. Either holds back the
    /// message from the block on, so nothing after it is read on its own.
    #[test]
    fn a_malformed_block_stays_content() {
        let open = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"Bash\">";
        let parameter = |name: &str, value: &str| {
            format!(
                "<｜DSML｜parameter name=\"{name}\" string=\"true\">{value}</｜DSML｜parameter>"
            )
        };
        let readable = [
            // A parameter given twice.
            format!(
                "{open}{}{}</｜DSML｜invoke></｜DSML｜tool_calls>",
                parameter("command", "ls"),
                parameter("command", "rm -rf /")
            ),
            // An invocation without a name.
            "<｜DSML｜tool_calls><｜DSML｜invoke></｜DSML｜invoke></｜DSML｜tool_calls>"
                .to_string(),
            // A block with nothing in it.
            "<｜DSML｜tool_calls></｜DSML｜tool_calls>".to_string(),
        ];
        let valid_after = |case: &str| {
            format!(
                "{case}\n<｜DSML｜tool_calls><｜DSML｜invoke name=\"Bash\">{}</｜DSML｜invoke></｜DSML｜tool_calls>",
                parameter("command", "ls")
            )
        };
        for case in readable {
            assert_eq!(calls(&case), None, "{case}");
            assert!(DsmlParser.could_be_tool_call(&case), "{case}");
            assert_eq!(strip_calls(&case), case);
            // Reading resumes after it: a valid block after it calls.
            let followed = valid_after(&case);
            assert_eq!(
                calls(&followed),
                Some(serde_json::json!([{"name": "Bash", "arguments": {"command": "ls"}}])),
                "{followed}"
            );
        }
        let unreadable = [
            // A named invocation that never closes, then an unnamed one.
            format!(
                "{open}{}<｜DSML｜invoke>{}</｜DSML｜invoke></｜DSML｜tool_calls>",
                parameter("command", "ls"),
                parameter("command", "rm -rf /")
            ),
            // A valid call, then an invocation that is opened and never finished.
            format!(
                "{open}{}</｜DSML｜invoke><｜DSML｜invoke name=\"Bash\"</｜DSML｜tool_calls>",
                parameter("command", "ls")
            ),
            // A parameter outside any invocation.
            format!(
                "<｜DSML｜tool_calls>{}</｜DSML｜tool_calls>",
                parameter("command", "ls")
            ),
        ];
        for case in unreadable {
            assert_eq!(calls(&case), None, "{case}");
            assert!(DsmlParser.could_be_tool_call(&case), "{case}");
            assert_eq!(strip_calls(&case), case);
            // Whatever follows is held with it, and makes no call.
            let followed = valid_after(&case);
            assert_eq!(calls(&followed), None, "{followed}");
            assert!(DsmlParser.could_be_tool_call(&followed), "{followed}");
        }
    }

    #[test]
    fn the_required_grammar_writes_the_templates_markup() {
        let grammar = build_dsml_grammar(&[bash(false)], Block::Whole);
        let lark = grammar.grammars[0].lark_grammar.as_deref().expect("lark");
        assert!(
            lark.starts_with(
                "start: \"<\" dsml_mark \"tool_calls>\\n\" (invoke_0)+ \"</\" dsml_mark \"tool_calls>\""
            ),
            "{lark}"
        );
        assert!(
            lark.contains(
                "invoke_0: \"<\" dsml_mark \"invoke name=\\\"Bash\\\">\\n\" arguments_0 \"</\" dsml_mark \"invoke>\\n\""
            ),
            "{lark}"
        );
        assert!(
            lark.contains("arguments_0: parameter_0_0? parameter_0_1?\n"),
            "{lark}"
        );
    }

    /// regression test for spiceai/mistral.rs#54: strict tools took any parameters
    #[test]
    fn a_strict_tools_grammar_holds_its_declared_parameters() {
        let grammar = build_dsml_grammar(&[bash(true)], Block::Whole);
        let lark = grammar.grammars[0].lark_grammar.as_deref().expect("lark");
        assert!(
            lark.contains("arguments_0: parameter_0_0 parameter_0_1?"),
            "{lark}"
        );
        // A plain string is text, or a JSON string held to its schema.
        assert!(
            lark.contains(
                "parameter_0_0: \"<\" dsml_mark \"parameter name=\\\"command\\\" string=\\\"\" (\"true\\\">\" dsml_vc | \"false\\\">\" @dsml_schema_0_0 \"</\" dsml_mark \"parameter>\\n\")\n"
            ),
            "{lark}"
        );
        assert!(
            lark.contains(
                "parameter_0_1: \"<\" dsml_mark \"parameter name=\\\"timeout\\\" string=\\\"\" (\"false\\\">\" @dsml_schema_0_1 \"</\" dsml_mark \"parameter>\\n\")\n"
            ),
            "{lark}"
        );
        // Each property's schema, then the any-JSON grammar for parameters no schema holds.
        assert_eq!(grammar.grammars.len(), 4);
        // A property's schema refers into a copy of the parts of the tool schema it reaches, in
        // their places, so its references resolve as they do there.
        let tool_schema = bash(true).function.parameters.expect("parameters");
        for (index, name) in [(1, "command"), (2, "timeout")] {
            assert_eq!(
                grammar.grammars[index].json_schema,
                Some(serde_json::json!({
                    "$ref": format!("#/$defs/tool/properties/{name}"),
                    "$defs": {
                        "tool": { "properties": { name: tool_schema["properties"][name] } },
                    },
                })),
                "{name}"
            );
        }
        assert_eq!(grammar.grammars[3].name.as_deref(), Some("dsml_json"));
    }

    /// regression test for spiceai/mistral.rs#54: a tool that is not strict could write a
    /// `string="false"` value that is not JSON, or JSON cut short, which the parser reads as
    /// unfinished, so a forced call could end with no call read from it
    #[test]
    fn a_tools_json_parameter_is_one_complete_json_value() {
        for (value, accepted) in [
            ("5", true),
            ("{\"a\": [1, 2]}", true),
            ("\"text\"", true),
            ("{", false),
            ("[1,", false),
            ("maybe", false),
        ] {
            // A declared parameter of a tool that is not strict.
            assert_eq!(
                forced_call_accepts(
                    &[bash(false)],
                    &one_call("Bash", &[("timeout", "false", value)])
                ),
                accepted,
                "declared {value:?}"
            );
            // Any parameter of a tool that declares none.
            let any = tool(serde_json::json!({ "type": "object" }));
            let mut any = any;
            any.function.strict = Some(false);
            assert_eq!(
                forced_call_accepts(
                    std::slice::from_ref(&any),
                    &one_call("set", &[("x", "false", value)])
                ),
                accepted,
                "undeclared {value:?}"
            );
        }
        // Text takes anything.
        assert!(forced_call_accepts(
            &[bash(false)],
            &one_call("Bash", &[("command", "true", "{")])
        ));
    }

    /// regression test for spiceai/mistral.rs#54: rebasing a property schema's references also
    /// rewrote a `$ref` key inside a `const` value, which is data, so the forced grammar
    /// required a value the schema does not allow
    #[test]
    fn rebasing_references_leaves_a_schemas_data_alone() {
        let root = serde_json::json!({
            "type": "object",
            "$defs": { "Any": {} },
            "properties": {
                "pinned": { "type": "object", "const": { "$ref": "#/original" } },
                "listed": { "enum": [{ "$ref": "#/original" }], "default": { "$ref": "#/d" } },
                "nested": { "type": "array", "items": { "$ref": "#/$defs/Any" } },
            },
            "required": ["pinned", "listed", "nested"],
            "additionalProperties": false,
        });
        let copy =
            |name: &str| property_schema(&root, name, &anchors(&root))["$defs"]["tool"].clone();
        assert_eq!(
            copy("pinned")["properties"]["pinned"]["const"],
            serde_json::json!({ "$ref": "#/original" })
        );
        assert_eq!(
            copy("listed")["properties"]["listed"]["enum"],
            serde_json::json!([{ "$ref": "#/original" }])
        );
        assert_eq!(
            copy("listed")["properties"]["listed"]["default"],
            serde_json::json!({ "$ref": "#/d" })
        );
        assert_eq!(
            copy("nested")["properties"]["nested"]["items"]["$ref"],
            serde_json::json!("#/$defs/tool/$defs/Any")
        );
        // Through the forced grammar: the constant as the schema gives it, and nothing else.
        let pinned = tool(serde_json::json!({
            "type": "object",
            "properties": { "pinned": { "type": "object", "const": { "$ref": "#/original" } } },
            "required": ["pinned"],
            "additionalProperties": false,
        }));
        assert!(forced_call_accepts(
            std::slice::from_ref(&pinned),
            &one_call("set", &[("pinned", "false", r##"{"$ref": "#/original"}"##)])
        ));
        assert!(!forced_call_accepts(
            &[pinned],
            &one_call(
                "set",
                &[("pinned", "false", r##"{"$ref": "#/$defs/tool/original"}"##)]
            )
        ));
    }

    struct GreedyTokenizerEnv {
        trie: toktrie::TokTrie,
    }

    impl toktrie::TokenizerEnv for GreedyTokenizerEnv {
        fn tok_trie(&self) -> &toktrie::TokTrie {
            &self.trie
        }

        fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<toktrie::TokenId> {
            self.trie.greedy_tokenize(bytes)
        }

        fn tokenize_is_canonical(&self) -> bool {
            false
        }
    }

    /// Every byte but 0xFF, which marks a special token, as a token, then `<eos>` and `｜DSML｜`
    /// as special tokens, as DeepSeek-V4's
    /// tokenizer registers the marker.
    fn dsml_env() -> (toktrie::TokEnv, toktrie::TokenId) {
        let env = byte_env(&["<eos>", DSML]);
        let mark = env
            .tok_trie()
            .get_special_token(DSML)
            .expect("the marker is a special token");
        (env, mark)
    }

    /// Every byte but 0xFF, which marks a special token, as a token, then `specials`.
    fn byte_env(specials: &[&str]) -> toktrie::TokEnv {
        let mut tokens = (0_u8..=254).map(|byte| vec![byte]).collect::<Vec<_>>();
        let eos = u32::try_from(tokens.len()).expect("fits");
        for token in specials {
            let mut bytes = vec![toktrie::TokTrie::SPECIAL_TOKEN_MARKER];
            bytes.extend_from_slice(token.as_bytes());
            tokens.push(bytes);
        }
        let vocab_size = u32::try_from(tokens.len()).expect("fits");
        let trie = toktrie::TokTrie::from(&toktrie::TokRxInfo::new(vocab_size, eos), &tokens);
        Arc::new(GreedyTokenizerEnv { trie })
    }

    /// The tokens of a DSML text, with every marker as the special token.
    fn dsml_tokens(text: &str, mark: toktrie::TokenId) -> Vec<toktrie::TokenId> {
        let mut tokens = Vec::new();
        for (index, piece) in text.split(DSML).enumerate() {
            if index > 0 {
                tokens.push(mark);
            }
            tokens.extend(piece.bytes().map(u32::from));
        }
        tokens
    }

    /// regression test for spiceai/mistral.rs#54: a forced call had to spell the marker as
    /// text, and could not write a closing tag inside a value
    #[test]
    fn a_forced_call_writes_the_marker_token_and_markup_in_values() {
        let (env, mark) = dsml_env();
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        let mut grammar = build_dsml_grammar(&[bash(true)], Block::Whole);
        specialize_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());
        let lark = grammar.grammars[0].lark_grammar.as_deref().expect("lark");
        assert!(lark.contains(&format!("dsml_mark: <[{mark}]>")), "{lark}");
        let mut matcher = llguidance::Matcher::new(factory.create_parser(grammar));
        matcher
            .consume_tokens(&[u32::from(b'<')])
            .expect("the call opens with `<`");
        let mask = matcher.compute_mask().expect("a mask");
        assert!(mask.is_allowed(mark), "the marker token follows `<`");
        let call = format!(
            "{DSML}tool_calls>\n<{DSML}invoke name=\"Bash\">\n<{DSML}parameter name=\"command\" string=\"true\">echo '<div>Hi</div>'</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
        );
        matcher
            .consume_tokens(&dsml_tokens(&call, mark))
            .expect("the whole call");
        assert!(matcher.is_accepting().expect("state"));
    }

    /// regression test for spiceai/mistral.rs#54: an attribute was found by substring, so
    /// `xname` named the tool and `xstring` changed a value's type
    #[test]
    fn an_attribute_is_read_by_its_whole_key() {
        assert_eq!(
            calls(&format!(
                "{OPEN}tool_calls>\n{OPEN}invoke xname=\"Bash\">\n{OPEN}parameter name=\"command\" string=\"true\">ls</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
            )),
            None
        );
        assert_eq!(
            calls(&format!(
                "{OPEN}tool_calls>\n{OPEN}invoke name=\"note\">\n{OPEN}parameter name=\"count\" xstring=\"true\">3</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
            )),
            None
        );
        assert_eq!(
            calls(&format!(
                "{OPEN}tool_calls>\n{OPEN}invoke title=\"see name=\" name=\"note\">\n</{DSML}invoke>\n</{DSML}tool_calls>"
            )),
            None
        );
    }

    /// regression test for spiceai/mistral.rs#54: with no special token for the marker, a
    /// forced call's string value was the longest text before a `</`, which the parser's lexer
    /// ran on past the value's closing tag, so no string value could close, and a value could
    /// not hold another DSML closing tag, which the scanner reads as the value's text. A value
    /// now runs to its first closing tag, and holds whatever the scanner reads as its text.
    #[test]
    fn a_forced_call_without_the_marker_token_writes_any_value_the_scanner_reads() {
        let env = byte_env(&["<eos>"]);
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        let accepts = |call: &str| {
            let mut grammar = build_dsml_grammar(&[bash(true)], Block::Whole);
            specialize_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());
            let mut matcher = llguidance::Matcher::new(factory.create_parser(grammar));
            let tokens: Vec<u32> = call.bytes().map(u32::from).collect();
            matcher.consume_tokens(&tokens).is_ok() && matcher.is_accepting().unwrap_or(false)
        };
        for value in [
            "ls",
            "echo '<div>Hi</div>'",
            "echo '</｜DSML｜invoke>'",
            "a ｜DSML｜ b",
        ] {
            let call = one_call("Bash", &[("command", "true", value)]);
            assert!(accepts(&call), "{value}");
            assert_eq!(
                calls(&call),
                Some(serde_json::json!([{ "name": "Bash", "arguments": { "command": value } }])),
                "{value}"
            );
        }
        // A closing tag of a parameter ends the value where the scanner ends it, so a value
        // cannot hold one.
        let call = one_call("Bash", &[("command", "true", "x</｜DSML｜parameter >y")]);
        assert!(!accepts(&call));
        assert_eq!(calls(&call), None);
    }

    /// Runs `call` through the forced-call grammar for `tools`, from its opening `<`: whether
    /// the grammar accepts it whole.
    fn forced_call_accepts(tools: &[Tool], call: &str) -> bool {
        let (env, mark) = dsml_env();
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        let mut grammar = build_dsml_grammar(tools, Block::Whole);
        specialize_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());
        let mut matcher = llguidance::Matcher::new(factory.create_parser(grammar));
        matcher.consume_tokens(&dsml_tokens(call, mark)).is_ok()
            && matcher.is_accepting().unwrap_or(false)
    }

    fn one_call(name: &str, parameters: &[(&str, &str, &str)]) -> String {
        let mut call = format!("<{DSML}tool_calls>\n<{DSML}invoke name=\"{name}\">\n");
        for (parameter, string, value) in parameters {
            call.push_str(&format!(
                "<{DSML}parameter name=\"{parameter}\" string=\"{string}\">{value}</{DSML}parameter>\n"
            ));
        }
        call.push_str(&format!("</{DSML}invoke>\n</{DSML}tool_calls>"));
        call
    }

    /// regression test for spiceai/mistral.rs#54: a forced string value could not end in `<`
    /// or `</`, whose `<` the grammar read as the start of more value
    #[test]
    fn a_forced_value_may_end_in_the_start_of_a_tag() {
        for value in ["<", "a <", "x<<", "a</", "</", "", "<b>bold</b>"] {
            assert!(
                forced_call_accepts(
                    &[bash(true)],
                    &one_call("Bash", &[("command", "true", value)])
                ),
                "{value:?}"
            );
        }
    }

    fn tool(parameters: Value) -> Tool {
        serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": { "name": "set", "strict": true, "parameters": parameters },
        }))
        .expect("a tool")
    }

    /// regression test for spiceai/mistral.rs#54: a strict string property's `enum` and
    /// `const` were dropped, so a forced call could hold any text
    #[test]
    fn a_forced_string_keeps_to_its_enum_and_const() {
        let units = tool(serde_json::json!({
            "type": "object",
            "properties": {
                "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] },
                "scale": { "type": "string", "const": "linear" },
            },
            "required": ["unit", "scale"],
            "additionalProperties": false,
        }));
        let call = |unit: &str, scale: &str| {
            one_call("set", &[("scale", "true", scale), ("unit", "true", unit)])
        };
        assert!(forced_call_accepts(
            std::slice::from_ref(&units),
            &call("celsius", "linear")
        ));
        assert!(!forced_call_accepts(
            std::slice::from_ref(&units),
            &call("kelvin", "linear")
        ));
        assert!(!forced_call_accepts(
            std::slice::from_ref(&units),
            &call("", "linear")
        ));
        assert!(!forced_call_accepts(&[units], &call("celsius", "log")));
    }

    /// regression test for spiceai/mistral.rs#54: a property's `$ref` lost the tool schema's
    /// `$defs`, so the forced grammar did not compile
    #[test]
    fn a_forced_argument_resolves_references_against_the_tool_schema() {
        let options = tool(serde_json::json!({
            "type": "object",
            "$defs": {
                "Options": {
                    "type": "object",
                    "properties": { "depth": { "type": "integer" } },
                    "required": ["depth"],
                    "additionalProperties": false,
                },
            },
            "properties": { "options": { "$ref": "#/$defs/Options" } },
            "required": ["options"],
            "additionalProperties": false,
        }));
        assert!(forced_call_accepts(
            std::slice::from_ref(&options),
            &one_call("set", &[("options", "false", r#"{"depth": 2}"#)])
        ));
        assert!(!forced_call_accepts(
            &[options],
            &one_call("set", &[("options", "false", r#"{"depth": "deep"}"#)])
        ));
    }

    /// regression test for spiceai/mistral.rs#54: a property copied out of the tool schema as a
    /// schema of its own lost a reference to another property, and a reference to the root's
    /// `$defs` could bind to definitions of the property's own, so a forced argument was held
    /// to the wrong schema, or the grammar did not compile and the call went unconstrained
    #[test]
    fn a_forced_arguments_references_resolve_as_in_the_tool_schema() {
        let refs = tool(serde_json::json!({
            "type": "object",
            "$defs": { "Count": { "type": "integer" } },
            "properties": {
                "left": { "$ref": "#/$defs/Count" },
                // To another property, through the root.
                "right": { "$ref": "#/properties/left" },
                // The root's `Count`, not the property's own, which only a longer pointer names.
                "inner": {
                    "type": "object",
                    "$defs": { "Count": { "type": "string" } },
                    "properties": { "n": { "$ref": "#/$defs/Count" } },
                    "required": ["n"],
                    "additionalProperties": false,
                },
            },
            "required": ["left", "right", "inner"],
            "additionalProperties": false,
        }));
        let call = |left: &str, right: &str, inner: &str| {
            one_call(
                "set",
                &[
                    ("inner", "false", inner),
                    ("left", "false", left),
                    ("right", "false", right),
                ],
            )
        };
        let tools = std::slice::from_ref(&refs);
        assert!(forced_call_accepts(tools, &call("1", "2", r#"{"n": 3}"#)));
        assert!(!forced_call_accepts(
            tools,
            &call("1", "\"two\"", r#"{"n": 3}"#)
        ));
        assert!(!forced_call_accepts(
            tools,
            &call("1", "2", r#"{"n": "three"}"#)
        ));
    }

    /// regression test for spiceai/mistral.rs#54: a `$ref` to a named anchor, `#Count` for the
    /// schema declaring `$anchor: "Count"`, was rebased like a pointer, to `#/$defs/toolCount`,
    /// which names nothing, so the forced grammar did not compile. The grammar follows pointers
    /// only, so the reference is written as the pointer to the schema declaring the anchor.
    #[test]
    fn a_forced_arguments_reference_to_an_anchor_resolves_to_the_schema_declaring_it() {
        use crate::tools::{ToolCallingMatcher, ToolChoice};
        let schema = serde_json::json!({
            "type": "object",
            "$defs": { "Count": { "$anchor": "Count", "type": "integer" } },
            "properties": {
                "n": { "$ref": "#Count" },
                // A pointer to the same schema.
                "m": { "$ref": "#/$defs/Count" },
                // An anchor declared on a property.
                "min": { "$anchor": "Min", "type": "integer", "minimum": 0 },
                "max": { "$ref": "#Min" },
            },
            "required": ["n", "m", "max"],
            "additionalProperties": false,
        });
        let copy =
            |name: &str| property_schema(&schema, name, &anchors(&schema))["$defs"]["tool"].clone();
        assert_eq!(
            copy("n")["properties"]["n"]["$ref"],
            serde_json::json!("#/$defs/tool/$defs/Count")
        );
        assert_eq!(
            copy("m")["properties"]["m"]["$ref"],
            serde_json::json!("#/$defs/tool/$defs/Count")
        );
        assert_eq!(
            copy("max")["properties"]["max"]["$ref"],
            serde_json::json!("#/$defs/tool/properties/min")
        );
        let anchored = tool(schema);
        assert_eq!(unenforceable_tool(&anchored), None);
        // The parameters in the grammar's order: the names sorted.
        let call = |n: &str, max: &str| {
            one_call(
                "set",
                &[
                    ("m", "false", "4"),
                    ("max", "false", max),
                    ("n", "false", n),
                ],
            )
        };
        let tools = std::slice::from_ref(&anchored);
        assert!(forced_call_accepts(tools, &call("3", "7")));
        assert!(!forced_call_accepts(tools, &call("\"three\"", "7")));
        assert!(!forced_call_accepts(tools, &call("3", "-1")));
        // A call read is held to the anchor's schema too.
        let matcher = ToolCallingMatcher::new(ToolChoice::Auto, Some(tools)).expect("a matcher");
        assert_eq!(matcher.get_call(&call("3", "7")).expect("read").len(), 1);
        assert!(matcher
            .get_call(&call("\"three\"", "7"))
            .expect("read")
            .is_empty());
    }

    /// A strict tool's schema with a reference the grammar cannot follow, to nothing in the
    /// schema or outside it, is one the grammar cannot hold: a required call refuses it, naming
    /// the reference, and a call the model writes on its own is read unconstrained and held to
    /// the schema when read, so the other tools stay held to theirs.
    #[test]
    fn a_reference_the_grammar_cannot_follow_makes_a_strict_tool_unenforceable() {
        let opener = format!("<{DSML}tool_calls>");
        let body = |name: &str, parameters: &str| {
            format!("\n<{DSML}invoke name=\"{name}\">\n{parameters}</{DSML}invoke>\n</{DSML}tool_calls>")
        };
        for reference in [
            "#Missing",
            "#/$defs/Missing",
            "https://example.com/count.json",
            "count.json#/Count",
        ] {
            let unfollowable = tool(serde_json::json!({
                "type": "object",
                "$defs": { "Count": { "$anchor": "Count", "type": "integer" } },
                "properties": { "n": { "$ref": reference } },
                "required": ["n"],
                "additionalProperties": false,
            }));
            let why = unenforceable_tool(&unfollowable)
                .unwrap_or_else(|| panic!("{reference} makes the tool unenforceable"));
            assert!(why.contains(&format!("refers to `{reference}`")), "{why}");
            // Written as a tool that is not strict, beside one that stays held to its schema.
            let tools = [unfollowable, bash(true)];
            assert!(
                continuation_accepts(
                    &tools,
                    &opener,
                    &body(
                        "set",
                        &format!(
                            "<{DSML}parameter name=\"n\" string=\"true\">anything</{DSML}parameter>\n"
                        )
                    )
                ),
                "{reference}"
            );
            assert!(
                !continuation_accepts(&tools, &opener, &body("Bash", "")),
                "{reference}"
            );
        }
    }

    /// regression test for spiceai/mistral.rs#54: a strict tool's schema with an annotation at
    /// its root, `$anchor`, `deprecated`, `readOnly` or `writeOnly`, was refused as constraining
    /// the object as a whole. An annotation constrains nothing: a required call holds the schema
    /// as it does without it, and a `$ref` to the root's anchor names the root.
    #[test]
    fn a_strict_schemas_root_annotations_do_not_make_it_unenforceable() {
        let schema = serde_json::json!({
            "$anchor": "Args",
            "deprecated": false,
            "readOnly": false,
            "writeOnly": false,
            "type": "object",
            "properties": {
                "n": { "type": "integer" },
                "again": { "$ref": "#Args" },
            },
            "required": ["n"],
            "additionalProperties": false,
        });
        let annotated = tool(schema.clone());
        assert_eq!(unenforceable_tool(&annotated), None);
        assert_eq!(
            property_schema(&schema, "again", &anchors(&schema))["$defs"]["tool"]["properties"]
                ["again"]["$ref"],
            serde_json::json!("#/$defs/tool")
        );
        let tools = std::slice::from_ref(&annotated);
        assert!(forced_call_accepts(
            tools,
            &one_call("set", &[("n", "false", "3")])
        ));
        assert!(forced_call_accepts(
            tools,
            &one_call(
                "set",
                &[("again", "false", r#"{"n": 1}"#), ("n", "false", "3")]
            )
        ));
        assert!(!forced_call_accepts(
            tools,
            &one_call(
                "set",
                &[("again", "false", r#"{"n": "one"}"#), ("n", "false", "3")]
            )
        ));
        assert!(!forced_call_accepts(
            tools,
            &one_call("set", &[("n", "false", "\"three\"")])
        ));
    }

    /// regression test for spiceai/mistral.rs#54: a `$id` inside a strict tool's schema starts a
    /// schema resource of its own, against which the references in it resolve, but the grammar
    /// followed them against the schema's root: with `$defs.Count` an integer in the resource and
    /// a string at the root, the grammar held a call to the string and refused a valid integer.
    /// The grammar follows references against the root only, so such a schema is one it cannot
    /// hold: a required call refuses it, naming the `$id`, and a call the model writes on its own
    /// is read unconstrained and held to the schema, resource and all, when read.
    #[test]
    fn a_strict_schema_with_a_nested_id_is_unenforceable() {
        use crate::tools::{ToolCallingMatcher, ToolChoice};
        let nested = tool(serde_json::json!({
            "type": "object",
            "$defs": {
                "Count": { "type": "string" },
                "Inner": {
                    "$id": "https://example.com/inner.json",
                    "type": "object",
                    "$defs": { "Count": { "type": "integer" } },
                    "properties": { "n": { "$ref": "#/$defs/Count" } },
                    "required": ["n"],
                    "additionalProperties": false,
                },
            },
            "properties": { "inner": { "$ref": "#/$defs/Inner" } },
            "required": ["inner"],
            "additionalProperties": false,
        }));
        let why = unenforceable_tool(&nested).expect("a nested `$id` is unenforceable");
        assert!(why.contains("declares `$id` at `/$defs/Inner`"), "{why}");
        // Written as a tool that is not strict, so the grammar no longer refuses the valid call.
        let tools = std::slice::from_ref(&nested);
        let call = |n: &str| one_call("set", &[("inner", "false", &format!(r#"{{"n": {n}}}"#))]);
        assert!(forced_call_accepts(tools, &call("3")));
        // Held to the schema, resource and all, when read.
        let matcher = ToolCallingMatcher::new(ToolChoice::Auto, Some(tools)).expect("a matcher");
        assert_eq!(matcher.get_call(&call("3")).expect("read").len(), 1);
        assert!(matcher
            .get_call(&call("\"three\""))
            .expect("read")
            .is_empty());
    }

    /// regression test for spiceai/mistral.rs#54: the `enum` and `const` shortcuts dropped the
    /// property's other constraints
    #[test]
    fn a_forced_enum_keeps_the_propertys_other_constraints() {
        let short = tool(serde_json::json!({
            "type": "object",
            "properties": {
                "code": { "type": "string", "enum": ["abc", "abcdef"], "maxLength": 3 },
            },
            "required": ["code"],
            "additionalProperties": false,
        }));
        let code = |string: &str, value: &str| one_call("set", &[("code", string, value)]);
        assert!(forced_call_accepts(
            std::slice::from_ref(&short),
            &code("false", "\"abc\"")
        ));
        assert!(!forced_call_accepts(
            std::slice::from_ref(&short),
            &code("false", "\"abcdef\"")
        ));
        assert!(!forced_call_accepts(&[short], &code("true", "abcdef")));

        let pinned = tool(serde_json::json!({
            "type": "object",
            "properties": { "mode": { "enum": ["a", "b"], "const": "a" } },
            "required": ["mode"],
            "additionalProperties": false,
        }));
        let mode = |string: &str, value: &str| one_call("set", &[("mode", string, value)]);
        assert!(forced_call_accepts(
            std::slice::from_ref(&pinned),
            &mode("false", "\"a\"")
        ));
        assert!(!forced_call_accepts(
            std::slice::from_ref(&pinned),
            &mode("false", "\"b\"")
        ));
        assert!(!forced_call_accepts(&[pinned], &mode("true", "b")));
    }

    /// regression test for spiceai/mistral.rs#54: a block's opening tag and every closing tag
    /// took any attributes
    #[test]
    fn a_tag_the_template_writes_bare_takes_no_attributes() {
        for (open, invoke_close, parameter_close, block_close) in [
            (" x=\"1\"", "", "", ""),
            ("", " x=\"1\"", "", ""),
            ("", "", " x=\"1\"", ""),
            ("", "", "", " x=\"1\""),
        ] {
            let message = format!(
                "{OPEN}tool_calls{open}>\n{OPEN}invoke name=\"Bash\">\n{OPEN}parameter name=\"command\" string=\"true\">ls</{DSML}parameter{parameter_close}>\n</{DSML}invoke{invoke_close}>\n</{DSML}tool_calls{block_close}>"
            );
            assert_eq!(calls(&message), None, "{message}");
            assert_eq!(strip_calls(&message), message, "it stays content");
        }
    }

    /// regression test for spiceai/mistral.rs#54: a malformed block discarded every call in the
    /// message, not only its own
    #[test]
    fn a_malformed_block_stays_content_and_later_blocks_still_call() {
        let valid = format!(
            "{OPEN}tool_calls>\n{OPEN}invoke name=\"Bash\">\n{OPEN}parameter name=\"command\" string=\"true\">ls</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
        );
        let empty = format!("{OPEN}tool_calls></{DSML}tool_calls>");
        assert_eq!(
            calls(&format!("{empty}\n{valid}")),
            Some(serde_json::json!([{"name": "Bash", "arguments": {"command": "ls"}}]))
        );
        assert_eq!(
            strip_calls(&format!("{empty}\n{valid}")),
            format!("{empty}\n")
        );

        // A block whose invocation has no name stays content, and the call written inside its
        // value is never made, though it reads as a block on its own.
        let nameless = format!(
            "{OPEN}tool_calls>\n{OPEN}invoke>\n{OPEN}parameter name=\"note\" string=\"true\">{OPEN}tool_calls>\n{OPEN}invoke name=\"Bash\">\n</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
        );
        assert_eq!(calls(&nameless), None);
        assert_eq!(
            calls(&format!("{nameless}\n{valid}")).map(|c| c.as_array().map(Vec::len)),
            Some(Some(1))
        );
    }

    /// regression test for spiceai/mistral.rs#54: a forced value could not hold `</｜`, though
    /// only `</｜DSML｜` reads as its closing tag
    #[test]
    fn a_forced_value_may_hold_the_markers_bar() {
        for value in ["ls", "</b>", "</｜ordinary>", "a</｜", "｜ alone"] {
            let call = one_call("Bash", &[("command", "true", value)]);
            assert!(forced_call_accepts(&[bash(true)], &call), "{value:?}");
        }
    }

    /// regression test for spiceai/mistral.rs#54: a block closed under another of the names a
    /// block opens under was read as closed
    #[test]
    fn a_block_closes_under_the_name_it_opened_with() {
        let body = format!("\n{OPEN}invoke name=\"Bash\">\n</{DSML}invoke>\n");
        let mixed = format!("{OPEN}calls>{body}</{DSML}tool_calls>");
        assert_eq!(calls(&mixed), None);
        assert_eq!(strip_calls(&mixed), mixed);
        assert_eq!(
            calls(&format!("{OPEN}calls>{body}</{DSML}calls>")),
            Some(serde_json::json!([{"name": "Bash", "arguments": {}}]))
        );
        assert_eq!(
            calls(&format!("{OPEN} tool_calls>{body}</{DSML} tool_calls >")),
            Some(serde_json::json!([{"name": "Bash", "arguments": {}}]))
        );
    }

    /// regression test for spiceai/mistral.rs#54: a block's opening tag cut short was not an
    /// unfinished block, so a required call's deadline forced a second one after it
    #[test]
    fn a_cut_short_opening_tag_is_an_unfinished_block() {
        for partial in [
            "<",
            "<｜DS",
            &format!("{OPEN}tool_cal"),
            &format!("{OPEN}tool_calls"),
        ] {
            assert!(
                unfinished_block(&format!("Let me check. {partial}")),
                "{partial:?}"
            );
        }
        assert!(!unfinished_block("Let me check."));
    }

    /// regression test for spiceai/mistral.rs#54: a forced value could not hold a DSML tag
    /// other than its own closing tag
    #[test]
    fn a_forced_value_may_hold_dsml_tags_but_its_closing_tag() {
        for value in [
            format!("echo '</{DSML}invoke>'"),
            format!("<{DSML}invoke name=\"x\">"),
            format!("a</{DSML}tool_calls>b</{DSML} parameters"),
            format!("{DSML} at the start"),
            "  <div>{\"not\": \"parsed\"}</div>".to_string(),
        ] {
            let call = one_call("Bash", &[("command", "true", &value)]);
            assert!(forced_call_accepts(&[bash(true)], &call), "{value:?}");
        }
        // The parser's own example, a value holding the invocation's closing tag.
        let example = format!(
            "<{DSML}tool_calls>\n<{DSML}invoke name=\"Bash\">\n<{DSML}parameter name=\"command\" string=\"true\">  <div>{{\"not\": \"parsed\"}}</div></{DSML}invoke>\n</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
        );
        assert!(forced_call_accepts(&[bash(true)], &example));
        // The closing tag ends the value wherever it appears.
        let cut = one_call(
            "Bash",
            &[("command", "true", &format!("a</{DSML}parameter>b"))],
        );
        assert!(!forced_call_accepts(&[bash(true)], &cut));
    }

    /// regression test for spiceai/mistral.rs#54: a block's opening tag that could not be
    /// read was skipped, and a block written inside it made a call
    #[test]
    fn a_broken_opening_tag_of_a_block_calls_nothing() {
        let inner = format!(
            "{OPEN}tool_calls>{OPEN}invoke name=\"Bash\"></{DSML}invoke></{DSML}tool_calls>"
        );
        assert_eq!(calls(&format!("{OPEN}tool_calls x=\"{inner}\">")), None);
        // An unrelated DSML tag is still passed over.
        assert!(calls(&format!("{OPEN}note x=\"1\"> {inner}")).is_some());
    }

    /// regression test for spiceai/mistral.rs#54: a forced call to a tool that is not strict
    /// could write a parameter twice, which the parser refuses
    #[test]
    fn a_forced_call_writes_each_parameter_once() {
        let twice = one_call(
            "Bash",
            &[("command", "true", "ls"), ("command", "true", "pwd")],
        );
        assert!(!forced_call_accepts(&[bash(false)], &twice));
        let once = one_call(
            "Bash",
            &[("command", "true", "ls"), ("timeout", "false", "5")],
        );
        assert!(forced_call_accepts(&[bash(false)], &once));
    }

    /// The tokens of a call to `name` with one parameter, whose `value` is text: any marker in
    /// it is written out, as a grammar's literal lets the model write it.
    fn call_with_text_value(name: &str, string: &str, value: &str, mark: u32) -> Vec<u32> {
        let mut tokens = dsml_tokens(
            &format!("<{DSML}tool_calls>\n<{DSML}invoke name=\"{name}\">\n<{DSML}parameter name=\"body\" string=\"{string}\">"),
            mark,
        );
        tokens.extend(value.bytes().map(u32::from));
        tokens.extend(dsml_tokens(
            &format!("</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"),
            mark,
        ));
        tokens
    }

    /// regression test for spiceai/mistral.rs#54: a `const` holding a parameter's closing tag
    /// was written as raw text, which the parser read as two parameters
    #[test]
    fn a_forced_const_holding_a_closing_tag_reads_back_whole() {
        let value =
            format!("x</{DSML}parameter>\n<{DSML}parameter name=\"injected\" string=\"true\">y");
        let pinned = tool(serde_json::json!({
            "type": "object",
            "properties": { "body": { "type": "string", "const": value } },
            "required": ["body"],
            "additionalProperties": false,
        }));
        let (env, mark) = dsml_env();
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        let accepts = |tokens: &[u32]| {
            let mut grammar = build_dsml_grammar(std::slice::from_ref(&pinned), Block::Whole);
            specialize_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());
            let mut matcher = llguidance::Matcher::new(factory.create_parser(grammar));
            matcher.consume_tokens(tokens).is_ok() && matcher.is_accepting().unwrap_or(false)
        };
        // Written raw, the value would read back as two parameters.
        assert!(!accepts(&call_with_text_value("set", "true", &value, mark)));
        // Written as JSON with its `<` escaped, it reads back whole.
        let escaped = serde_json::to_string(&value)
            .expect("JSON")
            .replace('<', "\\u003c");
        assert!(accepts(&call_with_text_value(
            "set", "false", &escaped, mark
        )));
        let text = format!(
            "<{DSML}tool_calls>\n<{DSML}invoke name=\"set\">\n<{DSML}parameter name=\"body\" string=\"false\">{escaped}</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
        );
        assert_eq!(
            calls(&text),
            Some(serde_json::json!([{"name": "set", "arguments": {"body": value}}]))
        );
    }

    /// regression test for spiceai/mistral.rs#54: where the marker is a token, a forced value
    /// could still spell `｜DSML｜` in ordinary tokens, which the parser read as the marker: a
    /// strict string could end its parameter early and write parameters the schema does not have
    #[test]
    fn a_forced_value_spells_the_marker_only_as_its_token() {
        let body_only = tool(serde_json::json!({
            "type": "object",
            "properties": { "body": { "type": "string" } },
            "required": ["body"],
            "additionalProperties": false,
        }));
        let (env, mark) = dsml_env();
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        let accepts = |tokens: &[u32]| {
            let mut grammar = build_dsml_grammar(std::slice::from_ref(&body_only), Block::Whole);
            specialize_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());
            let mut matcher = llguidance::Matcher::new(factory.create_parser(grammar));
            matcher.consume_tokens(tokens).is_ok() && matcher.is_accepting().unwrap_or(false)
        };
        let injected =
            format!("x</{DSML}parameter>\n<{DSML}parameter name=\"injected\" string=\"true\">y");
        // Spelled in ordinary tokens, the marker is refused wherever it is in the value.
        assert!(!accepts(&call_with_text_value(
            "set", "true", &injected, mark
        )));
        assert!(!accepts(&call_with_text_value(
            "set",
            "true",
            &format!("see {DSML} here"),
            mark
        )));
        assert!(!accepts(&call_with_text_value(
            "set",
            "true",
            &format!("echo '</{DSML}invoke>'"),
            mark
        )));
        // As the token, another DSML tag is part of the value.
        let mut tokens = dsml_tokens(
            &format!(
                "<{DSML}tool_calls>\n<{DSML}invoke name=\"set\">\n<{DSML}parameter name=\"body\" string=\"true\">echo '</{DSML}invoke>'</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
            ),
            mark,
        );
        assert!(accepts(&tokens));
        // As the token, the closing tag ends the value, and a parameter after it is refused.
        tokens = dsml_tokens(
            &format!(
                "<{DSML}tool_calls>\n<{DSML}invoke name=\"set\">\n<{DSML}parameter name=\"body\" string=\"true\">{injected}</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
            ),
            mark,
        );
        assert!(!accepts(&tokens));
    }

    /// regression test for spiceai/mistral.rs#54: a JSON value that spelled its parameter's
    /// closing tag inside a string ended the parameter there, so a forced call could end with
    /// its required parameters left out, or could not be read at all
    #[test]
    fn a_json_value_may_hold_its_closing_tag_in_a_string() {
        let json_call = |body: &str, after: &str| {
            format!(
                "<{DSML}tool_calls>\n<{DSML}invoke name=\"set\">\n<{DSML}parameter name=\"body\" string=\"false\">{body}</{DSML}parameter>\n{after}</{DSML}invoke>\n</{DSML}tool_calls>"
            )
        };
        let count =
            format!("<{DSML}parameter name=\"count\" string=\"false\">2</{DSML}parameter>\n");
        // The review's case: the whole block's closing tags inside the string.
        let tags = format!("ok</{DSML}parameter></{DSML}invoke></{DSML}tool_calls>");
        let message = json_call(&format!("\"{tags}\""), &count);
        assert_eq!(
            calls(&message),
            Some(serde_json::json!([{"name": "set", "arguments": {"body": tags, "count": 2}}]))
        );
        // Cut before the closing quote, the call is not complete: the string may still close.
        let cut = &message[..message.find(&format!("{tags}\"")).expect("the value") + tags.len()];
        assert_eq!(calls(cut), None);
        assert!(DsmlParser.could_be_tool_call(cut));
        // A closing tag alone in the string.
        assert_eq!(
            calls(&json_call(&format!("\"</{DSML}parameter>\""), "")),
            Some(
                serde_json::json!([{"name": "set", "arguments": {"body": format!("</{DSML}parameter>")}}])
            )
        );
        // A value that is not JSON is still its text, ending at the first closing tag.
        assert_eq!(
            calls(&json_call("maybe", &count)),
            Some(serde_json::json!([{"name": "set", "arguments": {"body": "maybe", "count": 2}}]))
        );
        // One that begins like JSON and never reads as JSON could still be cut short.
        assert_eq!(calls(&json_call("{\"a\": ", &count)), None);
        // Through the forced grammar, as JSON held to its schema: the string is written in
        // ordinary tokens, tags included, and reads back whole.
        let strict_json = tool(serde_json::json!({
            "type": "object",
            "properties": { "body": { "type": "string", "minLength": 1 } },
            "required": ["body"],
            "additionalProperties": false,
        }));
        let (env, mark) = dsml_env();
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        let mut grammar = build_dsml_grammar(std::slice::from_ref(&strict_json), Block::Whole);
        specialize_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());
        let mut matcher = llguidance::Matcher::new(factory.create_parser(grammar));
        let value = format!("\"{tags}\"");
        let tokens = call_with_text_value("set", "false", &value, mark);
        assert!(matcher.consume_tokens(&tokens).is_ok() && matcher.is_accepting().expect("state"));
        let written = format!(
            "<{DSML}tool_calls>\n<{DSML}invoke name=\"set\">\n<{DSML}parameter name=\"body\" string=\"false\">{value}</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
        );
        assert_eq!(
            calls(&written),
            Some(serde_json::json!([{"name": "set", "arguments": {"body": tags}}]))
        );
    }

    /// regression test for spiceai/mistral.rs#54: the scanner read any Unicode whitespace in a
    /// tag, but the grammars' value lexemes know only `[ \t\r\n]`, so a closing tag with a
    /// no-break space before its name passed a forced value as text and ended the parameter in
    /// the scanner, with parameters the schema does not have after it
    #[test]
    fn a_tag_with_other_whitespace_is_value_text_to_the_scanner_too() {
        let body_only = tool(serde_json::json!({
            "type": "object",
            "properties": { "body": { "type": "string" } },
            "required": ["body"],
            "additionalProperties": false,
        }));
        let value = format!(
            "x</{DSML}\u{a0}parameter>\n<{DSML}parameter name=\"injected\" string=\"true\">y"
        );
        let written = format!(
            "<{DSML}tool_calls>\n<{DSML}invoke name=\"set\">\n<{DSML}parameter name=\"body\" string=\"true\">{value}</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
        );
        // The grammar lets the value through, with the markers as the token.
        let (env, mark) = dsml_env();
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        let mut grammar = build_dsml_grammar(std::slice::from_ref(&body_only), Block::Whole);
        specialize_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());
        let mut matcher = llguidance::Matcher::new(factory.create_parser(grammar));
        assert!(
            matcher.consume_tokens(&dsml_tokens(&written, mark)).is_ok()
                && matcher.is_accepting().expect("state")
        );
        // The scanner reads the same value, whole.
        assert_eq!(
            calls(&written),
            Some(serde_json::json!([{"name": "set", "arguments": {"body": value}}]))
        );
        // The template's own spacing still reads.
        assert!(calls(&format!(
            "<{DSML} tool_calls>\n<{DSML} invoke name=\"set\">\n<{DSML} parameter name=\"body\" string=\"true\">v</{DSML} parameter>\n</{DSML} invoke>\n</{DSML} tool_calls>"
        ))
        .is_some());
    }

    /// regression test for spiceai/mistral.rs#54: a parameter tag with an attribute the template
    /// does not write lost its `string` flag along with it, so a text value beginning like JSON
    /// read as JSON cut short, and the block never finished: a valid block after it was lost
    #[test]
    fn a_parameter_with_a_stray_attribute_keeps_its_string_framing() {
        let stray = format!(
            "<{DSML}tool_calls>\n<{DSML}invoke name=\"note\">\n<{DSML}parameter name=\"body\" string=\"true\" extra=\"1\">{{ literal text</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
        );
        // The stray attribute makes the block content, read whole.
        assert_eq!(calls(&stray), None);
        assert_eq!(strip_calls(&stray), stray);
        // Reading resumes after it.
        let valid = format!(
            "<{DSML}tool_calls>\n<{DSML}invoke name=\"Bash\">\n<{DSML}parameter name=\"command\" string=\"true\">ls</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
        );
        assert_eq!(
            calls(&format!("{stray}\n{valid}")),
            Some(serde_json::json!([{"name": "Bash", "arguments": {"command": "ls"}}]))
        );
        // Without the flag the same value is JSON cut short, and the block waits for more.
        let unflagged = stray.replace(" string=\"true\" extra=\"1\"", " extra=\"1\"");
        assert_eq!(calls(&format!("{unflagged}\n{valid}")), None);
    }

    /// regression test for spiceai/mistral.rs#54: with no grammar active, a parameter named
    /// `""`, `1st` or `a b` was read as an argument of an executable call, though no grammar
    /// writes such a name and no tool declares one
    #[test]
    fn a_parameter_name_the_markup_cannot_write_makes_the_block_content() {
        let valid = format!(
            "<{DSML}tool_calls>\n<{DSML}invoke name=\"Bash\">\n<{DSML}parameter name=\"command\" string=\"true\">ls</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
        );
        for bad in ["", "1st", "a b", "x>y"] {
            let block = format!(
                "<{DSML}tool_calls>\n<{DSML}invoke name=\"Bash\">\n<{DSML}parameter name=\"{bad}\" string=\"true\">{{ text</{DSML}parameter>\n<{DSML}parameter name=\"command\" string=\"true\">ls</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
            );
            assert_eq!(calls(&block), None, "{bad:?}");
            assert_eq!(strip_calls(&block), block, "{bad:?}");
            // The block ends where the model closed it, so a valid block after it still calls.
            assert_eq!(
                calls(&format!("{block}\n{valid}")),
                Some(serde_json::json!([{"name": "Bash", "arguments": {"command": "ls"}}])),
                "{bad:?}"
            );
        }
        assert!(calls(&valid.replace("command", "a.b-c_9")).is_some());
    }

    /// regression test for spiceai/mistral.rs#54: an optional property whose schema is a `$ref`
    /// to a `false` schema, one the call must leave out, was compiled into a sub-grammar no value
    /// satisfies, which does not compile, so no call to any tool offered beside it was
    /// constrained. A property's schema is resolved through its references before the check.
    #[test]
    fn a_strict_property_referring_to_a_false_schema_is_left_out() {
        let schema = |required: &[&str]| {
            tool(serde_json::json!({
                "type": "object",
                "$defs": { "Never": false, "Alias": { "$ref": "#/$defs/Never" } },
                "properties": {
                    "command": { "type": "string" },
                    "blocked": { "$ref": "#/$defs/Never" },
                    "aliased": { "$ref": "#/$defs/Alias" },
                },
                "required": required,
                "additionalProperties": false,
            }))
        };
        let optional = schema(&["command"]);
        let tools = std::slice::from_ref(&optional);
        assert_eq!(unenforceable(tools), None);
        let (env, _) = dsml_env();
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        assert_eq!(
            factory
                .create_parser(build_dsml_grammar(tools, Block::Whole))
                .err()
                .map(|error| error.to_string()),
            None
        );
        assert!(forced_call_accepts(
            tools,
            &one_call("set", &[("command", "true", "ls")])
        ));
        for written in ["blocked", "aliased"] {
            assert!(
                !forced_call_accepts(
                    tools,
                    &one_call("set", &[(written, "true", "x"), ("command", "true", "ls")])
                ),
                "{written}"
            );
        }
        // Required, it cannot be written at all.
        let why = unenforceable(&[schema(&["command", "aliased"])]).expect("refused");
        assert!(
            why.contains("`aliased`") && why.contains("`false`"),
            "{why}"
        );
    }

    /// regression test for spiceai/mistral.rs#54: each parameter's sub-grammar held a copy of
    /// the whole tool schema, so a strict tool's grammar grew with the square of its parameters:
    /// 1000 parameters with 1 KB descriptions copied 1 GB. Each now holds only what the
    /// parameter's schema reaches. Four times the parameters must take less than six times the
    /// space (linear is four, square sixteen), and 1000 of them under 64 MB.
    #[test]
    fn a_strict_tools_grammar_grows_linearly_with_its_parameters() {
        let copied = |parameters: usize| {
            let properties: serde_json::Map<String, Value> = (0..parameters)
                .map(|index| {
                    let schema =
                        serde_json::json!({ "type": "string", "description": "d".repeat(1024) });
                    (format!("p{index}"), schema)
                })
                .collect();
            let strict = tool(serde_json::json!({
                "type": "object",
                "properties": properties,
                "additionalProperties": false,
            }));
            build_dsml_grammar(std::slice::from_ref(&strict), Block::Whole)
                .grammars
                .iter()
                .filter_map(|grammar| grammar.json_schema.as_ref())
                .map(|schema| serde_json::to_string(schema).expect("JSON").len())
                .sum::<usize>()
        };
        let (small, large) = (copied(250), copied(1000));
        let hundredths = large * 100 / small;
        let ratio = format!("{}.{:02} times", hundredths / 100, hundredths % 100);
        eprintln!("250 parameters: {small} bytes of sub-grammar schemas; 1000: {large}; {ratio}");
        assert!(large < small * 6, "{ratio}");
        assert!(large < 64 << 20, "{large} bytes");
    }

    /// regression test for spiceai/mistral.rs#54: a strict schema that admits parameters it does
    /// not declare, a map through `additionalProperties`, was held to its declared ones, none,
    /// so every call the grammar let the model write had no arguments
    #[test]
    fn a_strict_schema_admitting_undeclared_parameters_is_written_with_them() {
        use crate::tools::{ToolCallingMatcher, ToolChoice};
        let map = tool(serde_json::json!({
            "type": "object",
            "additionalProperties": { "type": "string" },
        }));
        let tools = std::slice::from_ref(&map);
        let why = unenforceable(tools).expect("refused for a required call");
        assert!(why.contains("`additionalProperties`"), "{why}");
        let opener = format!("<{DSML}tool_calls>");
        let body = |parameters: &str| {
            format!(
                "\n<{DSML}invoke name=\"set\">\n{parameters}</{DSML}invoke>\n</{DSML}tool_calls>"
            )
        };
        let parameter = |name: &str, string: &str, value: &str| {
            format!(
                "<{DSML}parameter name=\"{name}\" string=\"{string}\">{value}</{DSML}parameter>\n"
            )
        };
        let (env, _) = dsml_env();
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        let grammar = continuation_grammar(tools, &opener).expect("a grammar");
        assert_eq!(
            factory
                .create_parser(grammar)
                .err()
                .map(|error| error.to_string()),
            None
        );
        let k = parameter("k", "true", "v");
        assert!(continuation_accepts(tools, &opener, &body(&k)));
        assert!(continuation_accepts(tools, &opener, &body("")));
        // The calls are held to the schema when read: a value that is not a string is refused.
        let matcher = ToolCallingMatcher::new(ToolChoice::Auto, Some(tools)).expect("a matcher");
        let read = |parameters: &str| {
            matcher
                .get_call(&format!("{opener}{}", body(parameters)))
                .expect("read")
                .into_iter()
                .map(|call| call.function.arguments)
                .collect::<Vec<_>>()
        };
        assert_eq!(read(&k), [r#"{"k":"v"}"#]);
        assert_eq!(read(&parameter("k", "false", "3")), Vec::<String>::new());
    }

    /// regression test for spiceai/mistral.rs#54: an optional property whose schema is `false`,
    /// one the call must leave out, was compiled into a sub-grammar no value satisfies, which
    /// does not compile, so no call to any tool offered beside it was constrained
    #[test]
    fn a_strict_property_whose_schema_is_false_is_left_out() {
        let blocked = tool(serde_json::json!({
            "type": "object",
            "properties": { "command": { "type": "string" }, "blocked": false },
            "required": ["command"],
            "additionalProperties": false,
        }));
        let tools = std::slice::from_ref(&blocked);
        assert_eq!(unenforceable(tools), None);
        let (env, _) = dsml_env();
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        assert_eq!(
            factory
                .create_parser(build_dsml_grammar(tools, Block::Whole))
                .err()
                .map(|error| error.to_string()),
            None
        );
        assert!(forced_call_accepts(
            tools,
            &one_call("set", &[("command", "true", "ls")])
        ));
        assert!(!forced_call_accepts(
            tools,
            &one_call(
                "set",
                &[("blocked", "true", "x"), ("command", "true", "ls")]
            )
        ));
        // A required property whose schema is `false` cannot be written at all.
        let unsatisfiable = tool(serde_json::json!({
            "type": "object",
            "properties": { "blocked": false },
            "required": ["blocked"],
            "additionalProperties": false,
        }));
        let why = unenforceable(&[unsatisfiable]).expect("refused");
        assert!(
            why.contains("`blocked`") && why.contains("`false`"),
            "{why}"
        );
    }

    /// regression test for spiceai/mistral.rs#54: a strict `enum` value holding a tab, a
    /// carriage return or another control character was written into the grammar unescaped,
    /// which does not compile, so no call to any tool offered beside it was constrained
    #[test]
    fn a_strict_enum_value_with_a_control_character_compiles_and_is_written() {
        let values = [",", "\t", "\r", "\u{1}"];
        let strict = tool(serde_json::json!({
            "type": "object",
            "properties": { "delimiter": { "type": "string", "enum": values } },
            "required": ["delimiter"],
            "additionalProperties": false,
        }));
        let tools = std::slice::from_ref(&strict);
        let (env, _) = dsml_env();
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        assert_eq!(
            factory
                .create_parser(build_dsml_grammar(tools, Block::Whole))
                .err()
                .map(|error| error.to_string()),
            None
        );
        for value in values {
            assert!(
                forced_call_accepts(tools, &one_call("set", &[("delimiter", "true", value)])),
                "{value:?}"
            );
        }
        assert!(!forced_call_accepts(
            tools,
            &one_call("set", &[("delimiter", "true", ";")])
        ));
    }

    /// regression test for spiceai/mistral.rs#54: a strict tool whose schema the grammar cannot
    /// hold was written with its declared parameters only, so one its schema requires elsewhere,
    /// here through `allOf`, could not be written, and every call the grammar let the model
    /// write was refused when read
    #[test]
    fn an_unenforceable_strict_tool_takes_a_parameter_its_schema_requires_elsewhere() {
        use crate::tools::{ToolCallingMatcher, ToolChoice};
        let strict = tool(serde_json::json!({
            "type": "object",
            "properties": { "mode": { "type": "string" } },
            "allOf": [{ "required": ["path"] }],
        }));
        let tools = std::slice::from_ref(&strict);
        assert!(unenforceable(tools).is_some_and(|why| why.contains("`allOf`")));
        let opener = format!("<{DSML}tool_calls>");
        let body = |parameters: &[&str]| {
            let parameters: String = parameters
                .iter()
                .map(|name| {
                    format!(
                        "<{DSML}parameter name=\"{name}\" string=\"true\">x</{DSML}parameter>\n"
                    )
                })
                .collect();
            format!(
                "\n<{DSML}invoke name=\"set\">\n{parameters}</{DSML}invoke>\n</{DSML}tool_calls>"
            )
        };
        assert!(continuation_accepts(
            tools,
            &opener,
            &body(&["mode", "path"])
        ));
        assert!(continuation_accepts(tools, &opener, &body(&["path"])));
        assert!(continuation_accepts(tools, &opener, &body(&["mode"])));
        // A declared parameter is still written at most once, as the parser reads a parameter
        // given twice as malformed.
        assert!(!continuation_accepts(
            tools,
            &opener,
            &body(&["mode", "mode"])
        ));
        // The calls are held to the whole schema when read.
        let matcher = ToolCallingMatcher::new(ToolChoice::Auto, Some(tools)).expect("a matcher");
        let read = |parameters: &[&str]| {
            matcher
                .get_call(&format!("{opener}{}", body(parameters)))
                .expect("read")
                .len()
        };
        assert_eq!(read(&["mode", "path"]), 1);
        assert_eq!(read(&["mode"]), 0);
    }

    /// regression test for spiceai/mistral.rs#54: a strict property whose name the markup cannot
    /// write (`a"b`, `a>b`) was written into `name="..."` as it was, so a required call could
    /// complete its grammar and not read back as a call; a tool that is not strict wrote it too
    #[test]
    fn a_parameter_name_the_markup_cannot_write_is_refused_or_left_out() {
        for bad in ["a\"b", "a>b", "a b", "1st", "a<b"] {
            let strict = tool(serde_json::json!({
                "type": "object",
                "properties": { bad: { "type": "string" }, "ok": { "type": "string" } },
                "required": ["ok"],
                "additionalProperties": false,
            }));
            let why = unenforceable(std::slice::from_ref(&strict)).expect("refused");
            assert!(why.contains(&format!("`{bad}`")), "{why}");
            // The grammar that takes over a call the model began writes the tool as one that
            // is not strict: its declared parameters the markup can write, and other names it
            // can write, but never one it cannot.
            let opener = format!("<{DSML}tool_calls>");
            let body = |parameters: &str| {
                format!(
                    "\n<{DSML}invoke name=\"set\">\n{parameters}</{DSML}invoke>\n</{DSML}tool_calls>"
                )
            };
            let ok = format!("<{DSML}parameter name=\"ok\" string=\"true\">x</{DSML}parameter>\n");
            assert!(
                continuation_accepts(std::slice::from_ref(&strict), &opener, &body(&ok)),
                "{bad:?}"
            );
            assert!(
                !continuation_accepts(
                    std::slice::from_ref(&strict),
                    &opener,
                    &body(&format!(
                        "{ok}<{DSML}parameter name=\"{bad}\" string=\"true\">x</{DSML}parameter>\n"
                    ))
                ),
                "{bad:?}"
            );
        }
        for good in ["ok", "_x", "a.b-c", "camelCase9"] {
            let strict = tool(serde_json::json!({
                "type": "object",
                "properties": { good: { "type": "string" } },
                "required": [good],
                "additionalProperties": false,
            }));
            assert_eq!(unenforceable(&[strict]), None, "{good}");
        }
        // A tool that is not strict leaves such a parameter out of its grammar.
        let mut lax = tool(serde_json::json!({
            "type": "object",
            "properties": { "a>b": { "type": "string" }, "ok": { "type": "string" } },
        }));
        lax.function.strict = Some(false);
        let tools = std::slice::from_ref(&lax);
        assert!(forced_call_accepts(
            tools,
            &one_call("set", &[("ok", "true", "x")])
        ));
        assert!(!forced_call_accepts(
            tools,
            &one_call("set", &[("a>b", "true", "x")])
        ));
    }

    /// regression test for spiceai/mistral.rs#54: a forced value could not hold two markers in
    /// a row, which the scanner reads as value text
    #[test]
    fn a_forced_value_may_hold_adjacent_markers() {
        let value = format!("a{DSML}{DSML}b");
        let call = one_call("Bash", &[("command", "true", &value)]);
        assert!(forced_call_accepts(&[bash(true)], &call));
        assert_eq!(
            calls(&call),
            Some(serde_json::json!([{"name": "Bash", "arguments": {"command": value}}]))
        );
    }

    /// Runs `rest`, written after the block that `opener` ends by opening, through the
    /// continuation grammar: whether the grammar accepts it whole.
    fn continuation_accepts(tools: &[Tool], opener: &str, rest: &str) -> bool {
        let (env, mark) = dsml_env();
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        let Some(mut grammar) = continuation_grammar(tools, opener) else {
            return false;
        };
        specialize_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());
        let mut matcher = llguidance::Matcher::new(factory.create_parser(grammar));
        matcher.consume_tokens(&dsml_tokens(rest, mark)).is_ok()
            && matcher.is_accepting().unwrap_or(false)
    }

    /// regression test for spiceai/mistral.rs#54: no grammar held a call the model wrote on its
    /// own to a strict tool's schema, so `Bash`, which requires `command`, could be called
    /// without it
    #[test]
    fn a_call_the_model_writes_is_held_to_a_strict_tools_schema() {
        let opener = format!("<{DSML}tool_calls>");
        let body = |parameters: &str| {
            format!(
                "\n<{DSML}invoke name=\"Bash\">\n{parameters}</{DSML}invoke>\n</{DSML}tool_calls>"
            )
        };
        let command =
            format!("<{DSML}parameter name=\"command\" string=\"true\">ls</{DSML}parameter>\n");
        assert!(continuation_accepts(
            &[bash(true)],
            &opener,
            &body(&command)
        ));
        assert!(!continuation_accepts(&[bash(true)], &opener, &body("")));
        assert!(!continuation_accepts(
            &[bash(true)],
            &opener,
            &body(&format!(
                "{command}<{DSML}parameter name=\"cwd\" string=\"true\">/</{DSML}parameter>\n"
            ))
        ));
        // A tool that is not offered is refused.
        assert!(!continuation_accepts(
            &[bash(true)],
            &opener,
            &body(&command).replace("Bash", "Shell")
        ));
        // Whitespace the model wrote after the opening tag, a newline or more, is not written
        // again.
        for written in ["\n", "\n\n", "\n  \n"] {
            assert!(
                continuation_accepts(
                    &[bash(true)],
                    &format!("{opener}{written}"),
                    &body(&command)[1..]
                ),
                "{written:?}"
            );
        }
        // A block opened under another name closes under it.
        let calls_opener = format!("I'll run it.\n\n<{DSML} calls>");
        assert!(continuation_accepts(
            &[bash(true)],
            &calls_opener,
            &body(&command).replace("tool_calls>", "calls>")
        ));
        assert!(!continuation_accepts(
            &[bash(true)],
            &calls_opener,
            &body(&command)
        ));
    }

    /// The continuation grammar takes over right after a block's opening tag, before the model
    /// has begun the block's body, and only for a block the grammar can write.
    #[test]
    fn the_continuation_grammar_starts_right_after_a_blocks_opening_tag() {
        for text in [
            format!("<{DSML}tool_calls>"),
            format!("Sure.\n<{DSML}tool_calls>"),
            format!("<{DSML}tool_calls>\n"),
            format!("<{DSML}tool_calls>\n\n"),
            format!("<{DSML}tool_calls>\n  \n"),
            format!("<{DSML} calls>"),
        ] {
            assert!(
                continuation_grammar(&[bash(true)], &text).is_some(),
                "{text:?}"
            );
        }
        for text in [
            format!("<{DSML}tool_"),
            format!("<{DSML}tool_calls>\n<"),
            format!("<{DSML}tool_calls>\n<{DSML}invoke name=\"Bash\">\n"),
            format!("<{DSML}tool_calls x=\"1\">"),
            format!("<{DSML}tool_calls>!"),
            "Just text".to_string(),
        ] {
            assert!(
                continuation_grammar(&[bash(true)], &text).is_none(),
                "{text:?}"
            );
        }
        // A strict tool whose schema the grammar cannot hold is written as any parameters, and
        // held to its schema when read, so the other tools stay held to theirs.
        let either = tool(serde_json::json!({
            "type": "object",
            "oneOf": [
                { "properties": { "a": { "type": "string" } }, "required": ["a"] },
                { "properties": { "b": { "type": "string" } }, "required": ["b"] },
            ],
        }));
        let opener = format!("<{DSML}tool_calls>");
        let tools = [either, bash(true)];
        assert!(continuation_grammar(&tools, &opener).is_some());
        let body = |name: &str, parameters: &str| {
            format!("\n<{DSML}invoke name=\"{name}\">\n{parameters}</{DSML}invoke>\n</{DSML}tool_calls>")
        };
        let parameter = |name: &str, value: &str| {
            format!("<{DSML}parameter name=\"{name}\" string=\"true\">{value}</{DSML}parameter>\n")
        };
        assert!(continuation_accepts(
            &tools,
            &opener,
            &body(
                "set",
                &format!("{}{}", parameter("a", "1"), parameter("c", "3"))
            )
        ));
        assert!(continuation_accepts(
            &tools,
            &opener,
            &body("Bash", &parameter("command", "ls"))
        ));
        assert!(!continuation_accepts(&tools, &opener, &body("Bash", "")));
    }

    /// regression test for spiceai/mistral.rs#54: a `string="false"` value that is JSON but a
    /// number the parser cannot hold (`1e309`) read as the string "1e309", and one inside an
    /// array left the block unfinished, so the call was lost
    #[test]
    fn a_json_number_the_parser_cannot_hold_makes_the_block_content() {
        let ls = one_call("Bash", &[("command", "true", "ls")]);
        for value in ["1e309", "[1e309]", "{\"n\": -1e400}"] {
            let block = one_call("set", &[("n", "false", value)]);
            assert_eq!(calls(&block), None, "{value}");
            assert!(!unfinished_block(&block), "{value}");
            // The block is content, and reading resumes after it.
            assert_eq!(
                calls(&format!("{block}\n{ls}")),
                Some(serde_json::json!([{"name": "Bash", "arguments": {"command": "ls"}}])),
                "{value}"
            );
        }
        // Text that is not JSON in a `string="false"` value is still read as text.
        assert_eq!(
            calls(&one_call("set", &[("n", "false", "soon")])),
            Some(serde_json::json!([{"name": "set", "arguments": {"n": "soon"}}]))
        );
    }

    /// regression test for spiceai/mistral.rs#54: a strict tool's plain string parameter was
    /// written as text only, which cannot hold the parameter's closing tag, so a value with
    /// `</｜DSML｜parameter>` in it could not be written at all
    #[test]
    fn a_forced_plain_string_may_hold_its_closing_tag_as_a_json_string() {
        let value = format!("the tag is </{DSML}parameter>, and that is fine");
        let json = Value::String(value.clone()).to_string();
        // The parser reads the JSON string back as its text.
        assert_eq!(
            calls(&one_call("Bash", &[("command", "false", json.as_str())])),
            Some(serde_json::json!([{"name": "Bash", "arguments": {"command": value}}]))
        );
        // The forced grammar writes it. The JSON string's text is in ordinary tokens, the
        // marker in it included: the special token cannot be part of a JSON value.
        let (env, mark) = dsml_env();
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        let mut grammar = build_dsml_grammar(&[bash(true)], Block::Whole);
        specialize_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());
        let accepts = |value: &str| {
            let mut tokens = dsml_tokens(
                &format!(
                    "<{DSML}tool_calls>\n<{DSML}invoke name=\"Bash\">\n<{DSML}parameter name=\"command\" string=\"false\">"
                ),
                mark,
            );
            tokens.extend(value.bytes().map(u32::from));
            tokens.extend(dsml_tokens(
                &format!("</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"),
                mark,
            ));
            let mut matcher = llguidance::Matcher::new(factory.create_parser(grammar.clone()));
            matcher.consume_tokens(&tokens).is_ok() && matcher.is_accepting().unwrap_or(false)
        };
        assert!(accepts(&json));
        // As JSON, the value is held to the property's schema: a string.
        assert!(!accepts("42"));
        assert!(!accepts("[\"ls\"]"));
    }

    /// A tool that declares no parameters takes any in a forced call, under any names: a grammar
    /// cannot tell a name from one written before it. The parser reads a parameter given twice
    /// as a block that is content, and a required call then fails (see the matcher's tests).
    #[test]
    fn a_tool_declaring_no_parameters_takes_any_names_in_a_forced_call() {
        let any: Tool = serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": { "name": "set", "parameters": { "type": "object" } },
        }))
        .expect("a tool");
        let distinct = one_call("set", &[("a", "true", "1"), ("b", "true", "2")]);
        let twice = one_call("set", &[("a", "true", "1"), ("a", "true", "2")]);
        assert!(forced_call_accepts(std::slice::from_ref(&any), &distinct));
        assert!(forced_call_accepts(&[any], &twice));
        assert_eq!(
            calls(&distinct),
            Some(serde_json::json!([{"name": "set", "arguments": {"a": "1", "b": "2"}}]))
        );
        assert_eq!(calls(&twice), None);
    }

    /// regression test for spiceai/mistral.rs#54: the grammar wrote a tool's parameters from
    /// its schema's root `properties` only, so a parameter an `allOf` branch declares and
    /// requires could not be written, by a tool that is not strict or by a strict one, and a
    /// forced or continued call could only leave it out.
    #[test]
    fn a_parameter_an_allof_branch_requires_can_be_written() {
        let tool = |strict: Option<bool>| -> Tool {
            let mut function = serde_json::json!({
                "name": "Bash",
                "parameters": {
                    "type": "object",
                    "properties": { "cwd": { "type": "string" } },
                    "allOf": [{
                        "required": ["command"],
                        "properties": { "command": { "type": "string" } },
                    }],
                },
            });
            if let Some(strict) = strict {
                function["strict"] = serde_json::json!(strict);
            }
            serde_json::from_value(serde_json::json!({ "type": "function", "function": function }))
                .expect("tool")
        };
        let opener = format!("<{DSML}tool_calls>");
        for strict in [None, Some(false), Some(true)] {
            let tools = [tool(strict)];
            for call in [
                one_call("Bash", &[("command", "true", "ls")]),
                one_call(
                    "Bash",
                    &[("command", "true", "ls"), ("cwd", "true", "/tmp")],
                ),
            ] {
                assert!(forced_call_accepts(&tools, &call), "{strict:?} {call}");
                assert!(
                    continuation_accepts(&tools, &opener, &call[opener.len()..]),
                    "{strict:?} {call}"
                );
            }
            // A name the schema gives nowhere is still not written.
            let other = one_call("Bash", &[("shell", "true", "sh")]);
            if strict != Some(true) {
                assert!(!forced_call_accepts(&tools, &other), "{strict:?}");
            }
        }
        // A strict tool's schema may require a name it gives nowhere the grammar reads, and the
        // grammar standing in for the schema takes it, whatever declared name it begins as.
        let dependent: Tool = serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {
                "name": "Bash",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "properties": { "cwd": { "type": "string" } },
                    "dependentRequired": { "cwd": ["command"] },
                },
            },
        }))
        .expect("tool");
        for name in ["command", "cwdx", "shell"] {
            let call = one_call("Bash", &[(name, "true", "ls")]);
            assert!(
                forced_call_accepts(std::slice::from_ref(&dependent), &call),
                "{name}"
            );
        }
    }

    /// regression test for spiceai/mistral.rs#54: the continuation grammar took over only at
    /// the message's first block, so after a block that makes no call, empty or malformed with
    /// its tags in place, a later block, whose calls are made, was written with no grammar,
    /// and a strict tool's call in it was not held to its schema.
    #[test]
    fn the_continuation_grammar_takes_over_a_block_after_one_that_makes_no_call() {
        let tools = [bash(true)];
        let body = format!(
            "\n<{DSML}invoke name=\"Bash\">\n<{DSML}parameter name=\"command\" string=\"true\">ls</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
        );
        let without_command =
            format!("\n<{DSML}invoke name=\"Bash\">\n</{DSML}invoke>\n</{DSML}tool_calls>");
        for before in [
            format!("<{DSML}tool_calls>\n</{DSML}tool_calls>\n"),
            format!("<{DSML}tool_calls>\n<{DSML}invoke>\n</{DSML}invoke>\n</{DSML}tool_calls>\n"),
        ] {
            let opener = format!("{before}<{DSML}tool_calls>");
            assert!(
                continuation_grammar(&tools, &opener).is_some(),
                "{opener:?}"
            );
            assert!(continuation_accepts(&tools, &opener, &body), "{opener:?}");
            assert!(
                !continuation_accepts(&tools, &opener, &without_command),
                "{opener:?}"
            );
            assert_eq!(
                calls(&format!("{opener}{body}")),
                Some(serde_json::json!([{ "name": "Bash", "arguments": { "command": "ls" } }])),
                "{opener:?}"
            );
        }
    }
}
