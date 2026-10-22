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
//! A parameter marked `string="true"` is the string as written; any other value is JSON. The
//! markup is read leniently, as quantized checkpoints write it: whitespace may follow the
//! `｜DSML｜` token, and the block may be named `tool_calls`, `function_calls` or `calls`.

use candle_core::Result;
use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use serde::Serialize;
use serde_json::Value;

use super::ToolFormatParser;
use crate::Tool;

/// The token every DSML tag starts with after its `<` or `</`.
const DSML: &str = "｜DSML｜";
const OPEN: &str = "<｜DSML｜";
const CLOSE: &str = "</｜DSML｜";
/// The names a block of calls is written under, longest first so a prefix never shadows one.
const BLOCK_NAMES: [&str; 3] = ["function_calls", "tool_calls", "calls"];
/// The block name the chat template uses, and the one a grammar writes.
const BLOCK: &str = "tool_calls";

pub struct DsmlParser;

impl ToolFormatParser for DsmlParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        // An open block that has not closed, a DSML tag still being written, or a tail that
        // could become one.
        open_block_unclosed(text) || unfinished_tag(text) || ends_with_tag_prefix(text)
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
        let Some(calls) = parse_calls(message) else {
            return Ok(None);
        };
        serde_json::to_string(&calls)
            .map(Some)
            .map_err(candle_core::Error::msg)
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct DsmlToolCall {
    name: String,
    arguments: Value,
}

/// A DSML tag found in the text.
struct Tag<'a> {
    /// Where the tag's `<` is.
    start: usize,
    /// Just past the tag's `>`.
    end: usize,
    /// What follows the name inside the tag, such as its attributes.
    attributes: &'a str,
}

/// The first complete tag at or after `from` that opens (or, with `closing`, closes) one of
/// `names`.
fn find_tag<'a>(text: &'a str, from: usize, closing: bool, names: &[&'a str]) -> Option<Tag<'a>> {
    let marker = if closing { CLOSE } else { OPEN };
    let mut cursor = from;
    while let Some(offset) = text[cursor..].find(marker) {
        let start = cursor + offset;
        let after = start + marker.len();
        let rest = text[after..].trim_start();
        let name_at = text.len() - rest.len();
        cursor = after;
        let Some(&name) = names.iter().find(|name| rest.starts_with(**name)) else {
            continue;
        };
        let after_name = name_at + name.len();
        // The name ends at whitespace or the tag's end, so `calls` does not match `callsX`.
        if !text[after_name..].starts_with(|c: char| c == '>' || c.is_whitespace()) {
            continue;
        }
        let close = text[after_name..].find('>')?;
        return Some(Tag {
            start,
            end: after_name + close + 1,
            attributes: &text[after_name..after_name + close],
        });
    }
    None
}

/// The value of `attribute="..."` in a tag's attributes.
fn attribute<'a>(attributes: &'a str, attribute: &str) -> Option<&'a str> {
    let prefix = format!(r#"{attribute}=""#);
    let start = attributes.find(&prefix)? + prefix.len();
    let end = attributes[start..].find('"')? + start;
    Some(&attributes[start..end])
}

/// Every call in every complete block of `message`, or `None` when there is no complete block,
/// or a block holds a call that cannot be read, so the text is left as the model wrote it.
fn parse_calls(message: &str) -> Option<Vec<DsmlToolCall>> {
    let mut calls = Vec::new();
    let mut cursor = 0;
    while let Some(open) = find_tag(message, cursor, false, &BLOCK_NAMES) {
        let close = find_tag(message, open.end, true, &BLOCK_NAMES)?;
        calls.extend(parse_invocations(&message[open.end..close.start])?);
        cursor = close.end;
    }
    (!calls.is_empty()).then_some(calls)
}

fn parse_invocations(block: &str) -> Option<Vec<DsmlToolCall>> {
    let mut calls = Vec::new();
    let mut cursor = 0;
    while let Some(open) = find_tag(block, cursor, false, &["invoke"]) {
        let name = attribute(open.attributes, "name")?;
        let close = find_tag(block, open.end, true, &["invoke"])?;
        calls.push(DsmlToolCall {
            name: name.to_string(),
            arguments: Value::Object(parse_parameters(&block[open.end..close.start])?),
        });
        cursor = close.end;
    }
    Some(calls)
}

fn parse_parameters(invocation: &str) -> Option<serde_json::Map<String, Value>> {
    let mut parameters = serde_json::Map::new();
    let mut cursor = 0;
    while let Some(open) = find_tag(invocation, cursor, false, &["parameter"]) {
        let name = attribute(open.attributes, "name")?;
        let close = find_tag(invocation, open.end, true, &["parameter"])?;
        let raw = &invocation[open.end..close.start];
        let value = if attribute(open.attributes, "string") == Some("true") {
            Value::String(raw.to_string())
        } else {
            serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_string()))
        };
        parameters.insert(name.to_string(), value);
        cursor = close.end;
    }
    Some(parameters)
}

/// `message` without its blocks of calls: the text the model wrote around them.
pub(super) fn strip_calls(message: &str) -> String {
    let mut out = String::new();
    let mut cursor = 0;
    while let Some(open) = find_tag(message, cursor, false, &BLOCK_NAMES) {
        out.push_str(&message[cursor..open.start]);
        let Some(close) = find_tag(message, open.end, true, &BLOCK_NAMES) else {
            return out;
        };
        cursor = close.end;
    }
    out.push_str(&message[cursor..]);
    out
}

fn open_block_unclosed(text: &str) -> bool {
    let mut cursor = 0;
    while let Some(open) = find_tag(text, cursor, false, &BLOCK_NAMES) {
        let Some(close) = find_tag(text, open.end, true, &BLOCK_NAMES) else {
            return true;
        };
        cursor = close.end;
    }
    false
}

/// Whether the text's last DSML marker starts a tag that has not reached its `>`.
fn unfinished_tag(text: &str) -> bool {
    text.rfind(DSML)
        .is_some_and(|at| !text[at + DSML.len()..].contains('>'))
}

/// Whether the text ends partway through `<｜DSML｜` or `</｜DSML｜`.
fn ends_with_tag_prefix(text: &str) -> bool {
    [OPEN, CLOSE].iter().any(|marker| {
        marker
            .char_indices()
            .skip(1)
            .map(|(at, _)| &marker[..at])
            .any(|prefix| text.ends_with(prefix))
    })
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

/// A grammar for DSML calls to `tools`: the whole block when `required`, otherwise what
/// follows a block's opening tag.
fn build_dsml_grammar(tools: &[Tool], required: bool) -> TopLevelGrammar {
    let mut rules = Vec::new();
    let mut invokes = Vec::new();
    for (index, tool) in tools.iter().enumerate() {
        let invoke = format!("invoke_{index}");
        rules.push(format!(
            "{invoke}: {open} dsml_parameter* {close}",
            open = lark_literal(&format!("{OPEN}invoke name=\"{}\">\n", tool.function.name)),
            close = lark_literal(&format!("{CLOSE}invoke>\n")),
        ));
        invokes.push(invoke);
    }
    let calls = format!("({})+", invokes.join(" | "));
    let block_close = lark_literal(&format!("{CLOSE}{BLOCK}>"));
    let start = if required {
        format!(
            "start: {} {calls} {block_close}",
            lark_literal(&format!("{OPEN}{BLOCK}>\n"))
        )
    } else {
        format!("start: \"\\n\"? {calls} {block_close}")
    };
    let mut lines = vec![start];
    lines.extend(rules);
    lines.extend([
        format!(
            "dsml_parameter: {} PARAM_NAME {} (\"true\" | \"false\") {} dsml_value {}",
            lark_literal(&format!("{OPEN}parameter name=\"")),
            lark_literal("\" string=\""),
            lark_literal("\">"),
            lark_literal(&format!("{CLOSE}parameter>\n")),
        ),
        "dsml_value: (dsml_text | dsml_lt)*".to_string(),
        "dsml_text: /[^<]+/".to_string(),
        r#"dsml_lt: "<" /[^\/]/"#.to_string(),
        "PARAM_NAME: /[a-zA-Z_][a-zA-Z0-9_.-]*/".to_string(),
    ]);
    TopLevelGrammar {
        grammars: vec![GrammarWithLexer::from_lark(lines.join("\n"))],
        max_tokens: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calls(message: &str) -> Option<Value> {
        DsmlParser
            .parse(message)
            .expect("parse")
            .map(|json| serde_json::from_str(&json).expect("JSON"))
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
    fn a_string_parameter_keeps_its_text_exactly() {
        let message = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"write\"><｜DSML｜parameter name=\"body\" string=\"true\">  {\"not\": \"parsed\"}\n</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";
        assert_eq!(
            calls(message),
            Some(serde_json::json!([
                {"name": "write", "arguments": {"body": "  {\"not\": \"parsed\"}\n"}},
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
        }
        assert!(!DsmlParser.could_be_tool_call("a < b and c"));
        assert!(!DsmlParser.could_be_tool_call("done."));
    }

    #[test]
    fn an_invocation_without_a_name_leaves_the_text_as_written() {
        let message = "<｜DSML｜tool_calls><｜DSML｜invoke><｜DSML｜parameter name=\"a\" string=\"true\">1</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";
        assert_eq!(calls(message), None);
    }

    #[test]
    fn the_required_grammar_writes_the_templates_markup() {
        let tool: Tool = serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {"name": "Bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}
        }))
        .expect("tool");
        let grammar = build_dsml_grammar(&[tool], true);
        let lark = grammar.grammars[0].lark_grammar.as_deref().expect("lark");
        assert!(
            lark.starts_with(
                "start: \"<｜DSML｜tool_calls>\\n\" (invoke_0)+ \"</｜DSML｜tool_calls>\""
            ),
            "{lark}"
        );
        assert!(
            lark.contains("invoke_0: \"<｜DSML｜invoke name=\\\"Bash\\\">\\n\""),
            "{lark}"
        );
    }
}
