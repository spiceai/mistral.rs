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
//! JSON. The markup is read leniently, as quantized checkpoints write it: whitespace may
//! follow the `｜DSML｜` token, and the block may be named `tool_calls`, `function_calls` or
//! `calls`. It is read strictly otherwise: a block holds only invocations, an invocation only
//! parameters, every tag closes before the next opens, a tag holds only the attributes the
//! template writes, each once, and a parameter appears once. A block that breaks any of these
//! stays content, so no call is made from it. A message that opens a block outside any other
//! format's call is read only as DSML, so text inside a DSML call never becomes a call in
//! another format; a block opened inside another format's call is that call's text.

use candle_core::Result;
use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use serde::Serialize;
use serde_json::Value;

use super::ToolFormatParser;
use crate::Tool;

/// The token every DSML tag starts with after its `<` or `</`.
pub(super) const DSML: &str = "｜DSML｜";
const OPEN: &str = "<｜DSML｜";
const CLOSE: &str = "</｜DSML｜";
/// The names a block of calls is written under, longest first.
const BLOCK_NAMES: [&str; 3] = ["function_calls", "tool_calls", "calls"];
/// The block name the chat template uses, and the one a grammar writes.
const BLOCK: &str = "tool_calls";
/// The grammar rule that spells the `｜DSML｜` marker in every tag. Where the tokenizer has
/// the marker as a special token, `specialize_required_tool_call_grammar` makes the rule that
/// token, which text cannot match.
pub(super) const MARK_RULE: &str = r#"dsml_mark: "｜DSML｜""#;

pub struct DsmlParser;

impl ToolFormatParser for DsmlParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        matches!(scan(text), Scan::More)
    }

    fn claim_start(&self, message: &str) -> Option<usize> {
        block_start(message)
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::DeepSeekDsml
    }

    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        build_dsml_grammar(tools, false)
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        build_dsml_grammar(tools, true)
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

/// The calls in a message, and where each block of them sits in it.
struct Found {
    calls: Vec<DsmlToolCall>,
    blocks: Vec<std::ops::Range<usize>>,
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
    let rest = after.trim_start();
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
            if !tail.starts_with(|c: char| c == '>' || c.is_whitespace()) {
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

/// A tag's attributes, as `key="value"` pairs set apart by whitespace, each key among `keys`
/// and given once; `None` for anything else.
fn attribute_pairs<'a>(text: &'a str, keys: &[&str]) -> Option<Vec<(&'a str, &'a str)>> {
    let mut pairs: Vec<(&str, &str)> = Vec::new();
    let mut rest = text;
    loop {
        let pair = rest.trim_start();
        if pair.is_empty() {
            return Some(pairs);
        }
        if pair.len() == rest.len() {
            return None;
        }
        let (key, after) = pair.split_once("=\"")?;
        if !keys.contains(&key) || pairs.iter().any(|(seen, _)| *seen == key) {
            return None;
        }
        let (value, after) = after.split_once('"')?;
        pairs.push((key, value));
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
    text.len() - text.trim_start().len()
}

/// A parameter's value, which runs from the end of its opening tag, the start of `text`, to
/// its own closing tag: anything else in it, DSML tags included, is part of the value. The
/// flag says whether the closing tag carried attributes, which the template never writes.
fn value_at(text: &str) -> Scan<(&str, usize, bool)> {
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

/// Whether a tag's attributes are none at all.
fn bare(attributes: &str) -> bool {
    attributes.trim().is_empty()
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
        let attributes = attribute_pairs(attributes, &["name", "string"]).unwrap_or_default();
        let name = attribute(&attributes, "name");
        malformed |= name.is_none();
        at += length;
        let (raw, length, closed_with_attributes) = match value_at(&text[at..]) {
            Scan::Done(value) => value,
            Scan::More => return Scan::More,
            Scan::Bad => return Scan::Bad,
        };
        malformed |= closed_with_attributes;
        at += length;
        let value = if attribute(&attributes, "string") == Some("true") {
            Value::String(raw.to_string())
        } else {
            serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_string()))
        };
        // A parameter given twice has no one value to call with.
        if let Some(name) = name {
            malformed |= arguments.insert(name.to_string(), value).is_some();
        }
    }
}

/// One block's calls, from the end of its opening tag, the start of `text`, through its
/// closing tag, and whether it breaks a rule of the markup while its tags stay in place: no
/// invocation, an invocation without a name or with attributes the template does not write,
/// a malformed invocation, or a closing tag with attributes.
fn block_at(text: &str) -> Scan<(Vec<DsmlToolCall>, usize, bool)> {
    let mut calls = Vec::new();
    let mut malformed = false;
    let mut at = 0;
    loop {
        at += leading_whitespace(&text[at..]);
        let rest = &text[at..];
        if rest.is_empty() {
            return Scan::More;
        }
        let close = tag_at(rest, true, &BLOCK_NAMES);
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
        let attributes = attribute_pairs(attributes, &["name"]).unwrap_or_default();
        let name = attribute(&attributes, "name");
        malformed |= name.is_none();
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

/// Reads every block of calls in `message`. A block that breaks a rule of the markup while
/// its tags stay in place is content, and reading resumes after it; its values are read as
/// values, so a block written inside one never becomes a call. A block whose tags cannot be
/// read leaves no place to resume, so the message makes no call.
fn scan(message: &str) -> Scan<Found> {
    let mut found = Found {
        calls: Vec::new(),
        blocks: Vec::new(),
    };
    let mut from = 0;
    while let Some(offset) = message[from..].find(OPEN) {
        let start = from + offset;
        let (attributes, open) = match tag_at(&message[start..], false, &BLOCK_NAMES) {
            Scan::Done(tag) => tag,
            Scan::More => return Scan::More,
            // A DSML tag that does not open a block is not a call.
            Scan::Bad => {
                from = start + OPEN.len();
                continue;
            }
        };
        let (calls, length, malformed) = match block_at(&message[start + open..]) {
            Scan::Done(block) => block,
            Scan::More => return Scan::More,
            Scan::Bad => return Scan::Bad,
        };
        let end = start + open + length;
        if !malformed && bare(attributes) {
            found.calls.extend(calls);
            found.blocks.push(start..end);
        }
        from = end;
    }
    // The message may end partway through the marker that opens the next block.
    let tail = &message[from..];
    let partial = OPEN
        .char_indices()
        .skip(1)
        .any(|(at, _)| tail.ends_with(&OPEN[..at]));
    if partial {
        Scan::More
    } else {
        Scan::Done(found)
    }
}

/// Where `message` opens its first block of calls, finished or not.
fn block_start(message: &str) -> Option<usize> {
    message
        .match_indices(OPEN)
        .map(|(start, _)| start)
        .find(|&start| {
            matches!(
                tag_at(&message[start..], false, &BLOCK_NAMES),
                Scan::Done(_)
            )
        })
}

/// Whether `message` opens a block of calls it does not finish.
pub(super) fn unfinished_block(message: &str) -> bool {
    block_start(message).is_some() && matches!(scan(message), Scan::More)
}

/// `message` without its blocks of calls: the text the model wrote around them.
pub(super) fn strip_calls(message: &str) -> String {
    let Scan::Done(found) = scan(message) else {
        return message.to_string();
    };
    let mut out = String::new();
    let mut cursor = 0;
    for block in found.blocks {
        out.push_str(&message[cursor..block.start]);
        cursor = block.end;
    }
    out.push_str(&message[cursor..]);
    out
}

fn lark_literal(value: &str) -> String {
    format!(
        "\"{}\"",
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    )
}

/// A string parameter's value and its closing tag, as a grammar.
const VALUE_THEN_CLOSE: &str = r#"DSML_VALUE_LT_SLASH dsml_mark "parameter>\n""#;

/// A tag in a grammar: `<`, or `</` with `closing`, the marker, then the rest as text.
fn tag(closing: bool, rest: &str) -> String {
    format!(
        "{} dsml_mark {}",
        lark_literal(if closing { "</" } else { "<" }),
        lark_literal(rest)
    )
}

/// A grammar for DSML calls to `tools`: the whole block when `required`, otherwise what
/// follows a block's opening tag.
fn build_dsml_grammar(tools: &[Tool], required: bool) -> TopLevelGrammar {
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
    let block_close = tag(true, &format!("{BLOCK}>"));
    let start = if required {
        format!(
            "start: {} {calls} {block_close}",
            tag(false, &format!("{BLOCK}>\n"))
        )
    } else {
        format!("start: \"\\n\"? {calls} {block_close}")
    };
    let mut lines = vec![start];
    lines.extend(rules);
    lines.extend([
        format!(
            "dsml_parameter: {} PARAM_NAME {} (\"true\" | \"false\") {} {VALUE_THEN_CLOSE}",
            tag(false, "parameter name=\""),
            lark_literal("\" string=\""),
            lark_literal("\">"),
        ),
        // A value is any text but its closing tag, which `</` and then the marker start, so
        // ordinary markup such as `</div>` is part of it. One lexeme reads the value with the
        // `</` that ends it: written apart, a value ending in `<` or `</` would read the
        // closing tag's `<` as more value.
        // The lexeme is text through a final `</` that holds no `</｜DSML｜`, the text the
        // closing tag reads as whichever way the marker is written.
        r#"DSML_VALUE_LT_SLASH: /(?s:.*)<\// & ~/(?s:.*)<\/｜DSML｜(?s:.*)/"#.to_string(),
        "PARAM_NAME: /[a-zA-Z_][a-zA-Z0-9_.-]*/".to_string(),
        MARK_RULE.to_string(),
    ]);
    let mut grammars = vec![GrammarWithLexer::from_lark(lines.join("\n"))];
    grammars.extend(schemas);
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
/// required ones always, and no other.
const ENFORCED_ROOT_KEYS: [&str; 13] = [
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
    "$comment",
    "examples",
    "default",
];

/// Why a required call cannot hold one of `tools` to its strict schema: a constraint on the
/// object as a whole, which the parameter-by-parameter grammar cannot express.
pub(super) fn unenforceable(tools: &[Tool]) -> Option<String> {
    tools.iter().find_map(|tool| {
        let schema = tool.function.strict_parameters_schema()?;
        let keyword = schema
            .as_object()?
            .keys()
            .find(|key| !ENFORCED_ROOT_KEYS.contains(&key.as_str()))?
            .clone();
        Some(format!(
            "Tool '{}' is strict, and its parameters schema constrains the object as a whole with `{keyword}`, which a required DeepSeek-V4 tool call cannot enforce. Remove `strict` from the tool, or the `{keyword}` from its schema.",
            tool.function.name
        ))
    })
}

/// `property` with the definitions of the tool schema `root`, which its references are
/// relative to.
fn with_definitions(property: &Value, root: &Value) -> Value {
    let mut property = property.clone();
    if let Some(object) = property.as_object_mut() {
        for key in ["$defs", "definitions"] {
            if let (Some(definitions), false) = (root.get(key), object.contains_key(key)) {
                object.insert(key.to_string(), definitions.clone());
            }
        }
    }
    property
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
    let Some(schema) = tool.function.strict_parameters_schema() else {
        rules.push(format!("{arguments}: dsml_parameter*"));
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
    for (property_index, name) in names.into_iter().enumerate() {
        let parameter = format!("parameter_{tool_index}_{property_index}");
        let property = &properties[name];
        let (string, value) = match property_value(property) {
            PropertyValue::Text => ("true", VALUE_THEN_CLOSE.to_string()),
            PropertyValue::OneOf(values) => {
                let values = values
                    .iter()
                    .map(|value| lark_literal(value))
                    .collect::<Vec<_>>()
                    .join(" | ");
                ("true", format!("({values}) {}", tag(true, "parameter>\n")))
            }
            PropertyValue::Json => {
                let grammar = format!("dsml_schema_{tool_index}_{property_index}");
                schemas.push(GrammarWithLexer {
                    name: Some(grammar.clone()),
                    json_schema: Some(with_definitions(property, &schema)),
                    ..Default::default()
                });
                ("false", format!("@{grammar} {}", tag(true, "parameter>\n")))
            }
        };
        rules.push(format!(
            "{parameter}: {open} {value}",
            open = tag(
                false,
                &format!("parameter name=\"{name}\" string=\"{string}\">")
            ),
        ));
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
    use crate::tools::parsers::specialize_required_tool_call_grammar;

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
        assert!(!DsmlParser.could_be_tool_call(whole));
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

    /// regression test for spiceai/mistral.rs#54: malformed blocks made calls
    #[test]
    fn a_malformed_block_stays_content() {
        let open = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"Bash\">";
        let parameter = |name: &str, value: &str| {
            format!(
                "<｜DSML｜parameter name=\"{name}\" string=\"true\">{value}</｜DSML｜parameter>"
            )
        };
        let cases = [
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
            // A parameter given twice.
            format!(
                "{open}{}{}</｜DSML｜invoke></｜DSML｜tool_calls>",
                parameter("command", "ls"),
                parameter("command", "rm -rf /")
            ),
            // A parameter outside any invocation.
            format!(
                "<｜DSML｜tool_calls>{}</｜DSML｜tool_calls>",
                parameter("command", "ls")
            ),
            // An invocation without a name.
            "<｜DSML｜tool_calls><｜DSML｜invoke></｜DSML｜invoke></｜DSML｜tool_calls>"
                .to_string(),
            // A block with nothing in it.
            "<｜DSML｜tool_calls></｜DSML｜tool_calls>".to_string(),
        ];
        for case in cases {
            assert_eq!(calls(&case), None, "{case}");
            assert!(!DsmlParser.could_be_tool_call(&case), "{case}");
            assert_eq!(strip_calls(&case), case);
        }
    }

    #[test]
    fn the_required_grammar_writes_the_templates_markup() {
        let grammar = build_dsml_grammar(&[bash(false)], true);
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
        assert!(lark.contains("arguments_0: dsml_parameter*"), "{lark}");
    }

    /// regression test for spiceai/mistral.rs#54: strict tools took any parameters
    #[test]
    fn a_strict_tools_grammar_holds_its_declared_parameters() {
        let grammar = build_dsml_grammar(&[bash(true)], true);
        let lark = grammar.grammars[0].lark_grammar.as_deref().expect("lark");
        assert!(
            lark.contains("arguments_0: parameter_0_0 parameter_0_1?"),
            "{lark}"
        );
        assert!(
            lark.contains(
                "parameter_0_0: \"<\" dsml_mark \"parameter name=\\\"command\\\" string=\\\"true\\\">\" DSML_VALUE_LT_SLASH dsml_mark \"parameter>\\n\""
            ),
            "{lark}"
        );
        assert!(
            lark.contains(
                "parameter_0_1: \"<\" dsml_mark \"parameter name=\\\"timeout\\\" string=\\\"false\\\">\" @dsml_schema_0_1"
            ),
            "{lark}"
        );
        assert_eq!(grammar.grammars.len(), 2);
        assert_eq!(
            grammar.grammars[1].json_schema,
            Some(serde_json::json!({"type": "integer"}))
        );
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
        let mut grammar = build_dsml_grammar(&[bash(true)], true);
        specialize_required_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());
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

    /// Runs `call` through the forced-call grammar for `tools`, from its opening `<`: whether
    /// the grammar accepts it whole.
    fn forced_call_accepts(tools: &[Tool], call: &str) -> bool {
        let (env, mark) = dsml_env();
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        let mut grammar = build_dsml_grammar(tools, true);
        specialize_required_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());
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
}
