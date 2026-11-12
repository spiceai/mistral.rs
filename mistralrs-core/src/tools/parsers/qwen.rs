//! Qwen tool call parser.
//!
//! Formats:
//! `<tool_call>{"name":"...", "arguments":{...}}</tool_call>`
//! `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`
//! `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value></tool_call>` (GLM-4.5 and later)

use candle_core::Result;
use llguidance::api::TopLevelGrammar;
use regex::Regex;
use serde_json::{Map, Value};
use std::sync::OnceLock;

use super::ToolFormatParser;
use crate::Tool;

static QWEN_REGEX: OnceLock<Regex> = OnceLock::new();
static QWEN_FUNCTION_REGEX: OnceLock<Regex> = OnceLock::new();
static QWEN_PARAMETER_REGEX: OnceLock<Regex> = OnceLock::new();
static GLM_CALL_REGEX: OnceLock<Regex> = OnceLock::new();

pub struct QwenParser;

impl ToolFormatParser for QwenParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("<tool_call>")
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Qwen
    }

    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            qwen_tool_call_lark(tools, false),
            tools,
            "arguments",
            false,
        )
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            qwen_tool_call_lark(tools, true),
            tools,
            "arguments",
            false,
        )
    }

    fn parse(&self, message: &str) -> Result<Option<String>> {
        let re = QWEN_REGEX
            .get_or_init(|| Regex::new(r"(?s)<tool_call>(?P<inner>.*?)</tool_call>").unwrap());

        if !re.is_match(message) {
            Ok(None)
        } else {
            parse_qwen_tool_calls(message)
        }
    }
}

fn qwen_tool_call_lark(tools: &[Tool], include_wrapper: bool) -> String {
    // Parallel calls arrive as consecutive <tool_call> blocks, so the grammar stays open for more
    let start = if include_wrapper {
        r#"start: "<tool_call>" (json_call | xml_call | glm_call) ("\n"? "<tool_call>" (json_call | xml_call | glm_call))*"#
    } else {
        r#"start: (json_call | xml_call | glm_call) ("\n"? <tool_call> (json_call | xml_call | glm_call))*"#
    };
    let json_call = if include_wrapper {
        r#"json_call: @json_body "</tool_call>""#
    } else {
        "json_call: @json_body </tool_call>"
    };
    let xml_end = if include_wrapper {
        r#""</tool_call>""#
    } else {
        "</tool_call>"
    };

    format!(
        r#"{start}
{json_call}
xml_call: "\n"? xml_function ("\n"? xml_function)* "\n"? {xml_end}
{}
{}
{}
glm_call: glm_function {xml_end}
{}
{}
{}"#,
        qwen_xml_function_rules(tools),
        value_rules("xml_param_value", "</parameter>"),
        qwen_xml_generic_rules(tools),
        glm_function_rules(tools),
        value_rules("glm_value", "</arg_value>"),
        glm_generic_rules(tools),
    )
}

// The rules for `name`, a value that runs to its closing tag `close` (`</parameter>`,
// `</arg_value>`) and may hold any other text, a `<` of its own included (HTML, XML, code). It
// may hold a start of `close` that does not go on to the whole tag, and may end in one right
// before the tag. A start of `close` is never taken as more of the value when `<`, which may
// begin `close` itself, follows it, so the tag that closes the value is never read as its text.
// `close` holds no `<` but its first character.
fn value_rules(name: &str, close: &str) -> String {
    let class_char = |c: char| match c {
        '/' | '\\' | ']' | '^' | '-' => format!("\\{c}"),
        c => c.to_string(),
    };
    let prefixes = (1..close.len())
        .map(|k| lark_string(&close[..k]))
        .collect::<Vec<_>>();
    let starts = (1..close.len())
        .map(|k| {
            let next = class_char(close[k..].chars().next().unwrap_or('<'));
            format!("{} /[^{next}<]/", lark_string(&close[..k]))
        })
        .collect::<Vec<_>>();
    format!(
        "{name}: {name}_part* {name}_prefix*\n\
         {name}_part: /[^<]+/ | {name}_prefix* {name}_start\n\
         {name}_start: {}\n\
         {name}_prefix: {}",
        starts.join(" | "),
        prefixes.join(" | "),
    )
}

#[derive(serde::Serialize)]
struct QwenToolCall {
    name: String,
    arguments: Value,
}

const WRAPPER_OPEN: &str = "<tool_call>";
const WRAPPER_CLOSE: &str = "</tool_call>";
const GLM_KEY: &str = "<arg_key>";
const GLM_KEY_END: &str = "</arg_key>";
const GLM_VALUE: &str = "<arg_value>";
const GLM_VALUE_END: &str = "</arg_value>";

/// How the text after a `<tool_call>` marker reads as a GLM call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum GlmCall {
    /// It is not a GLM call.
    Not,
    /// It is one, which ends this far into the text, past its closer.
    Ends(usize),
    /// It is one, or may be once more streams in, and has not ended.
    Open,
}

/// How `body`, the text after a `<tool_call>` marker, reads as a GLM call closed by `closer`:
/// the function's name after any whitespace, then each `<arg_key>` and `<arg_value>` pair, then
/// the closer. A key ends at its first `</arg_key>`. A value is raw text, so a closer, a marker
/// or a `</arg_value>` in it is its text: it ends at the first `</arg_value>` that the next
/// `<arg_key>` or the call's closer follows. This one reading frames the call for the parser,
/// for the text stripped from the content, and for the scanner that tells calls from text.
pub(super) fn glm_call(body: &str, closer: &str) -> GlmCall {
    // the text from `at` is the start of `tag`, and more may complete it
    let partial = |at: usize, tag: &str| tag.starts_with(&body[at..]);
    let skip_space = |at: usize| body.len() - body[at..].trim_start().len();
    let mut at = skip_space(0);
    if !body[at..].starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        return GlmCall::Not;
    }
    at += body[at..]
        .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':' | '-')))
        .unwrap_or(body.len() - at);
    let mut first = true;
    loop {
        at = skip_space(at);
        if body[at..].starts_with(closer) {
            return GlmCall::Ends(at + closer.len());
        }
        if partial(at, closer) || partial(at, GLM_KEY) {
            return GlmCall::Open;
        }
        if !body[at..].starts_with(GLM_KEY) {
            // after the name, other text is a sentence's, not a call; after a pair it cannot
            // occur, as a value runs on to a `</arg_value>` the next pair or the closer follows
            return if first { GlmCall::Not } else { GlmCall::Open };
        }
        first = false;
        at += GLM_KEY.len();
        let Some(key_end) = body[at..].find(GLM_KEY_END) else {
            return GlmCall::Open;
        };
        at = skip_space(at + key_end + GLM_KEY_END.len());
        if partial(at, GLM_VALUE) {
            return GlmCall::Open;
        }
        if !body[at..].starts_with(GLM_VALUE) {
            return GlmCall::Not;
        }
        at += GLM_VALUE.len();
        at = loop {
            let Some(found) = body[at..].find(GLM_VALUE_END) else {
                return GlmCall::Open;
            };
            let after = at + found + GLM_VALUE_END.len();
            let next = skip_space(after);
            if body[next..].starts_with(GLM_KEY) || body[next..].starts_with(closer) {
                break after;
            }
            if partial(next, GLM_KEY) || partial(next, closer) {
                return GlmCall::Open;
            }
            at = after;
        };
    }
}

/// The body of each `<tool_call>` in `message`, and where its call ends: a GLM call to its own
/// closer, its values read as text, and any other to the first closer.
fn qwen_call_bodies(message: &str) -> Vec<(&str, usize)> {
    let mut bodies = Vec::new();
    let mut from = 0;
    while let Some(open) = message[from..].find(WRAPPER_OPEN) {
        let start = from + open + WRAPPER_OPEN.len();
        let body = &message[start..];
        let end = match glm_call(body, WRAPPER_CLOSE) {
            GlmCall::Ends(end) => end,
            GlmCall::Not | GlmCall::Open => match body.find(WRAPPER_CLOSE) {
                Some(close) => close + WRAPPER_CLOSE.len(),
                None => break,
            },
        };
        bodies.push((&body[..end - WRAPPER_CLOSE.len()], start + end));
        from = start + end;
    }
    bodies
}

/// `message` without its calls, each cut out to where `qwen_call_bodies` ends it; a call left
/// open takes the rest of the message with it.
pub(super) fn strip_qwen_calls(message: &str) -> String {
    let mut out = String::new();
    let mut from = 0;
    for (_, end) in qwen_call_bodies(message) {
        if let Some(open) = message[from..].find(WRAPPER_OPEN) {
            out.push_str(&message[from..from + open]);
        }
        from = end;
    }
    match message[from..].find(WRAPPER_OPEN) {
        Some(open) => out.push_str(&message[from..from + open]),
        None => out.push_str(&message[from..]),
    }
    out
}

fn parse_qwen_tool_calls(message: &str) -> Result<Option<String>> {
    let mut xml_calls = Vec::new();
    let mut json_calls = Vec::new();

    for (inner, _) in qwen_call_bodies(message) {
        let inner = inner.trim();
        if inner.is_empty() {
            continue;
        }

        // A GLM call is read first: it must account for the whole body, so a `<function=` it
        // holds is a value's text, while the XML format finds its calls anywhere in the body.
        if let Some(call) = parse_glm_tool_call(inner) {
            xml_calls.push(call);
            continue;
        }

        let parsed_xml = parse_qwen_xml_tool_call(inner)?;
        if !parsed_xml.is_empty() {
            xml_calls.extend(parsed_xml);
            continue;
        }

        match serde_json::from_str::<Value>(inner) {
            Ok(value) => json_calls.push(value),
            Err(_) => return Ok(None),
        }
    }

    if !xml_calls.is_empty() {
        return Ok(Some(
            serde_json::to_string(&xml_calls).map_err(candle_core::Error::msg)?,
        ));
    }

    match json_calls.len() {
        0 => Ok(None),
        1 => Ok(Some(
            serde_json::to_string(&json_calls[0]).map_err(candle_core::Error::msg)?,
        )),
        _ => Ok(Some(
            serde_json::to_string(&json_calls).map_err(candle_core::Error::msg)?,
        )),
    }
}

fn parse_qwen_xml_tool_call(inner: &str) -> Result<Vec<QwenToolCall>> {
    let function_re = QWEN_FUNCTION_REGEX.get_or_init(|| {
        Regex::new(r"(?s)<function=(?P<name>[^>\n]+)>\s*(?P<body>.*?)\s*</function>").unwrap()
    });
    let parameter_re = QWEN_PARAMETER_REGEX.get_or_init(|| {
        Regex::new(r"(?s)<parameter=(?P<key>[^>\n]+)>(?P<value>.*?)</parameter>").unwrap()
    });

    let mut calls = Vec::new();
    for caps in function_re.captures_iter(inner) {
        let name = caps.name("name").unwrap().as_str().trim().to_string();
        let body = caps.name("body").unwrap().as_str();
        let mut arguments = Map::new();
        for param_caps in parameter_re.captures_iter(body) {
            let key = param_caps.name("key").unwrap().as_str().trim().to_string();
            let value = param_caps.name("value").unwrap().as_str();
            arguments.insert(key, qwen_xml_param_value(value));
        }
        calls.push(QwenToolCall {
            name,
            arguments: Value::Object(arguments),
        });
    }
    Ok(calls)
}

/// One call in GLM's format (GLM-4.5, 4.6, 5.x): the function's name, then each argument as an
/// `<arg_key>` and `<arg_value>` pair, with newlines between the parts as the template writes
/// them. A call with no arguments is the name alone. Anything else between the tags, a JSON
/// object above all, is not this format.
///
/// Values are raw text, as in the XML format above: GLM's template writes a string verbatim and
/// any other value as JSON, and the tool schema's types are applied to them later.
fn parse_glm_tool_call(inner: &str) -> Option<QwenToolCall> {
    let call_re = GLM_CALL_REGEX.get_or_init(|| {
        Regex::new(r"(?s)^(?P<name>[A-Za-z_][A-Za-z0-9_.:\-]*)\s*(?P<arguments><arg_key>.*)?$")
            .unwrap()
    });
    let caps = call_re.captures(inner)?;
    let name = caps.name("name")?.as_str().to_string();
    let arguments = parse_glm_arguments(caps.name("arguments").map_or("", |m| m.as_str()))?;
    Some(QwenToolCall {
        name,
        arguments: Value::Object(arguments),
    })
}

/// Every `<arg_key>` and `<arg_value>` pair of a GLM call in turn, to the end; `None` when text
/// the pairs do not account for means the body is not a GLM call. A key ends at its first
/// `</arg_key>`, so a key without a value of its own is malformed rather than read up to a
/// later key's tag. A value is raw text, which may hold `</arg_value>` itself: it ends at the
/// first `</arg_value>` that the next pair, or the end of the call, follows.
fn parse_glm_arguments(mut rest: &str) -> Option<Map<String, Value>> {
    const KEY: &str = "<arg_key>";
    const KEY_END: &str = "</arg_key>";
    const VALUE: &str = "<arg_value>";
    const VALUE_END: &str = "</arg_value>";
    let mut arguments = Map::new();
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            return Some(arguments);
        }
        rest = rest.strip_prefix(KEY)?;
        let key_end = rest.find(KEY_END)?;
        let key = rest[..key_end].trim().to_string();
        rest = rest[key_end + KEY_END.len()..]
            .trim_start()
            .strip_prefix(VALUE)?;
        let mut from = 0;
        let value_end = loop {
            let at = from + rest[from..].find(VALUE_END)?;
            let next = rest[at + VALUE_END.len()..].trim_start();
            if next.is_empty() || next.starts_with(KEY) {
                break at;
            }
            from = at + VALUE_END.len();
        };
        arguments.insert(key, Value::String(rest[..value_end].to_string()));
        rest = &rest[value_end + VALUE_END.len()..];
    }
}

// Values are kept verbatim minus the template's single framing newlines; types come from the tool schema later
fn qwen_xml_param_value(raw: &str) -> Value {
    let value = raw.strip_prefix('\n').unwrap_or(raw);
    let value = value.strip_suffix('\n').unwrap_or(value);
    Value::String(value.to_string())
}

fn qwen_xml_function_rules(tools: &[Tool]) -> String {
    let mut rules = Vec::new();
    let mut branches = Vec::new();

    for (tool_idx, tool) in tools.iter().enumerate() {
        let branch = format!("qwen_xml_tool_{tool_idx}");
        let args = qwen_xml_args_rule(tool_idx, tool, &mut rules);
        let opener = lark_string(&format!("<function={}>", tool.function.name));
        rules.push(format!(
            "{branch}: {opener} \"\\n\"? {args} \"</function>\""
        ));
        branches.push(branch);
    }

    if branches.is_empty() {
        rules.push("xml_function: qwen_xml_generic_function".to_string());
    } else {
        rules.push(format!("xml_function: {}", branches.join(" | ")));
    }

    rules.join("\n")
}

fn qwen_xml_args_rule(tool_idx: usize, tool: &Tool, rules: &mut Vec<String>) -> String {
    let args_rule = format!("qwen_xml_args_{tool_idx}");
    let Some(parameters) = tool.function.parameters.as_ref() else {
        rules.push(format!("{args_rule}: qwen_xml_generic_params"));
        return args_rule;
    };
    let Some(Value::Object(properties)) = parameters.get("properties") else {
        rules.push(format!("{args_rule}: qwen_xml_generic_params"));
        return args_rule;
    };

    let required = parameters
        .get("required")
        .and_then(|v| v.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<std::collections::BTreeSet<_>>()
        })
        .unwrap_or_default();

    let mut required_pairs = Vec::new();
    let mut optional_pairs = Vec::new();
    let mut property_names = properties.keys().cloned().collect::<Vec<_>>();
    property_names.sort();

    for (prop_idx, name) in property_names.iter().enumerate() {
        let pair = format!("qwen_xml_arg_{tool_idx}_{prop_idx}");
        let opener = lark_string(&format!("<parameter={name}>"));
        rules.push(format!(
            "{pair}: {opener} \"\\n\"? xml_param_value \"</parameter>\" \"\\n\"?"
        ));
        if required.contains(name.as_str()) {
            required_pairs.push(pair);
        } else {
            optional_pairs.push(pair);
        }
    }

    // Qwen does not always emit parameters in schema order, so accept any order (llama.cpp permutes too)
    let mut parts = required_pairs;
    parts.extend(optional_pairs);

    if parts.is_empty() {
        rules.push(format!("{args_rule}:"));
    } else {
        rules.push(format!("{args_rule}: ({})*", parts.join(" | ")));
    }

    args_rule
}

fn qwen_xml_generic_rules(tools: &[Tool]) -> &'static str {
    if tools.iter().any(|tool| tool.function.parameters.is_none()) {
        r#"qwen_xml_generic_function: "<function=" /[a-zA-Z_][a-zA-Z0-9_]*/ ">" "\n"? qwen_xml_generic_params "</function>"
qwen_xml_generic_params: (qwen_xml_generic_param "\n"?)*
qwen_xml_generic_param: "<parameter=" /[a-zA-Z_][a-zA-Z0-9_]*/ ">" "\n"? xml_param_value "</parameter>""#
    } else {
        ""
    }
}

/// GLM's call to each offered tool: its name, then each of its arguments as a key and value
/// pair, in any order, as the GLM template writes them; a tool without a schema takes any keys.
fn glm_function_rules(tools: &[Tool]) -> String {
    let mut rules = Vec::new();
    let mut branches = Vec::new();
    for (tool_idx, tool) in tools.iter().enumerate() {
        let branch = format!("glm_tool_{tool_idx}");
        let name = lark_string(&tool.function.name);
        let args = glm_args_rule(tool_idx, tool, &mut rules);
        rules.push(format!("{branch}: {name} \"\\n\"? {args}"));
        branches.push(branch);
    }
    if branches.is_empty() {
        rules.push("glm_function: glm_generic_function".to_string());
    } else {
        rules.push(format!("glm_function: {}", branches.join(" | ")));
    }
    rules.join("\n")
}

fn glm_args_rule(tool_idx: usize, tool: &Tool, rules: &mut Vec<String>) -> String {
    let args_rule = format!("glm_args_{tool_idx}");
    let properties = tool.function.parameters.as_ref().and_then(|parameters| {
        match parameters.get("properties") {
            Some(Value::Object(properties)) => Some(properties),
            _ => None,
        }
    });
    let Some(properties) = properties else {
        rules.push(format!("{args_rule}: glm_generic_args"));
        return args_rule;
    };
    let mut property_names = properties.keys().cloned().collect::<Vec<_>>();
    property_names.sort();
    let mut pairs = Vec::new();
    for (prop_idx, name) in property_names.iter().enumerate() {
        let pair = format!("glm_arg_{tool_idx}_{prop_idx}");
        let key = lark_string(&format!("<arg_key>{name}</arg_key>"));
        rules.push(format!(
            "{pair}: {key} \"\\n\"? \"<arg_value>\" glm_value \"</arg_value>\" \"\\n\"?"
        ));
        pairs.push(pair);
    }
    if pairs.is_empty() {
        rules.push(format!("{args_rule}:"));
    } else {
        rules.push(format!("{args_rule}: ({})*", pairs.join(" | ")));
    }
    args_rule
}

fn glm_generic_rules(tools: &[Tool]) -> &'static str {
    if tools.is_empty() || tools.iter().any(|tool| tool.function.parameters.is_none()) {
        r#"glm_generic_function: /[a-zA-Z_][a-zA-Z0-9_]*/ "\n"? glm_generic_args
glm_generic_args: ("<arg_key>" /[a-zA-Z_][a-zA-Z0-9_]*/ "</arg_key>" "\n"? "<arg_value>" glm_value "</arg_value>" "\n"?)*"#
    } else {
        ""
    }
}

fn lark_string(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    format!("\"{escaped}\"")
}

#[cfg(test)]
mod tests {
    use super::{parse_qwen_tool_calls, QwenParser};
    use crate::tools::parsers::{specialize_tool_call_grammar, ToolFormatParser};
    use mistralrs_mcp::{Function, ToolType};
    use serde_json::json;
    use serde_json::Value;
    use std::collections::HashMap;
    use std::sync::Arc;

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

    fn wrapper_token_trie() -> toktrie::TokTrie {
        let mut tokens = (0_u8..=127).map(|byte| vec![byte]).collect::<Vec<_>>();
        let eos = u32::try_from(tokens.len()).expect("test vocabulary fits in u32");
        tokens.push(b"\xff<eos>".to_vec());
        tokens.push(b"\xff<tool_call>".to_vec());
        tokens.push(b"\xff</tool_call>".to_vec());
        let vocab_size = u32::try_from(tokens.len()).expect("test vocabulary fits in u32");
        toktrie::TokTrie::from(&toktrie::TokRxInfo::new(vocab_size, eos), &tokens)
    }

    #[test]
    fn parses_qwen_json_tool_call() {
        let parsed = QwenParser
            .parse(r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#)
            .unwrap()
            .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value["name"], "get_weather");
        assert_eq!(value["arguments"]["city"], "Paris");
    }

    /// GLM-5.3's call as Claude Code received it as text through Spice before this format was read.
    #[test]
    fn parses_a_glm_tool_call() {
        let parsed = parse_qwen_tool_calls(
            "<tool_call>WebSearch<arg_key>query</arg_key><arg_value>Anthropic Claude latest news</arg_value></tool_call>",
        )
        .unwrap()
        .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(
            value,
            json!([{"name": "WebSearch", "arguments": {"query": "Anthropic Claude latest news"}}])
        );
    }

    /// The layout GLM's chat template writes: newlines between the parts, a non-string value as
    /// JSON, and a value that holds tags of its own.
    #[test]
    fn parses_glm_tool_calls_as_the_template_writes_them() {
        let parsed = parse_qwen_tool_calls(
            "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>\n<tool_call>write_file\n<arg_key>content</arg_key>\n<arg_value>if a < b:\n    print(\"<b>\")</arg_value>\n</tool_call>",
        )
        .unwrap()
        .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(
            value,
            json!([
                {"name": "get_weather", "arguments": {"city": "Paris", "days": "3"}},
                {"name": "write_file", "arguments": {"content": "if a < b:\n    print(\"<b>\")"}},
            ])
        );
    }

    #[test]
    fn a_glm_call_without_arguments_is_its_name() {
        let parsed = parse_qwen_tool_calls("<tool_call>get_time\n</tool_call>")
            .unwrap()
            .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value, json!([{"name": "get_time", "arguments": {}}]));
    }

    /// Text between the tags that is neither JSON nor a whole GLM call is not a call.
    #[test]
    fn a_body_that_is_not_a_glm_call_is_not_read_as_one() {
        for message in [
            "<tool_call>not a call: hello</tool_call>",
            "<tool_call>get_weather<arg_key>city</arg_key></tool_call>",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value> trailing</tool_call>",
            // a key with no value of its own: the next key's closing tag does not end it
            "<tool_call>get_weather<arg_key>city</arg_key><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>",
        ] {
            assert_eq!(parse_qwen_tool_calls(message).unwrap(), None, "{message}");
        }
    }

    #[test]
    fn parses_qwen_xml_tool_call() {
        let parsed = parse_qwen_tool_calls(
            r#"<tool_call>
<function=get_weather>
<parameter=locations>
[{"country":"France","city":"Paris"}]
</parameter>
<parameter=temp_units>
celsius
</parameter>
</function>
</tool_call>"#,
        )
        .unwrap()
        .unwrap();

        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value[0]["name"], "get_weather");
        // Raw text: the matcher applies the tool schema types afterwards
        assert_eq!(
            value[0]["arguments"]["locations"],
            r#"[{"country":"France","city":"Paris"}]"#
        );
        assert_eq!(value[0]["arguments"]["temp_units"], "celsius");
    }

    #[test]
    fn qwen_xml_values_keep_inner_whitespace() {
        let parsed = parse_qwen_tool_calls(
            "<tool_call>\n<function=write_file>\n<parameter=content>\n    indented\n\n</parameter>\n</function>\n</tool_call>",
        )
        .unwrap()
        .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value[0]["arguments"]["content"], "    indented\n");
    }

    #[test]
    fn parses_multiple_qwen_xml_tool_calls() {
        let parsed = parse_qwen_tool_calls(
            r#"<tool_call>
<function=get_weather>
<parameter=city>Tokyo</parameter>
</function>
</tool_call><tool_call>
<function=get_time>
<parameter=timezone>Asia/Tokyo</parameter>
</function>
</tool_call>"#,
        )
        .unwrap()
        .unwrap();

        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value.as_array().unwrap().len(), 2);
        assert_eq!(value[0]["arguments"]["city"], "Tokyo");
        assert_eq!(value[1]["arguments"]["timezone"], "Asia/Tokyo");
    }

    #[test]
    fn parses_qwen_xml_code_with_less_than() {
        let parsed = parse_qwen_tool_calls(
            r#"<tool_call>
<function=mistralrs_execute_python>
<parameter=code>
print(1 < 2)
</parameter>
</function>
</tool_call>"#,
        )
        .unwrap()
        .unwrap();

        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value[0]["name"], "mistralrs_execute_python");
        assert_eq!(value[0]["arguments"]["code"], "print(1 < 2)");
    }

    #[test]
    fn qwen_xml_grammar_requires_code_without_forcing_newline_before_close() {
        let parameters: HashMap<String, Value> = serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "code": { "type": "string" },
                "outputs": { "type": "array" }
            },
            "required": ["code"]
        }))
        .unwrap();
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "mistralrs_execute_python".to_string(),
                description: None,
                parameters: Some(parameters),
                strict: Some(true),
            },
        };

        let grammar = QwenParser.tool_call_grammar(&[tool], "");
        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();
        assert!(lark.contains("\"<parameter=code>\" \"\\n\"? xml_param_value \"</parameter>\""));
        assert!(!lark.contains("\"\\n</parameter>\""));
    }

    #[test]
    fn required_grammar_uses_tokenizer_special_wrappers() {
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: None,
                strict: None,
            },
        };
        let mut grammar = QwenParser.required_tool_call_grammar(&[tool]);
        let trie = wrapper_token_trie();
        let start_token = trie.get_special_token("<tool_call>").unwrap();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let parser = factory.create_parser(grammar.clone()).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(parser));

        assert!(!matcher.compute_mask().unwrap().is_allowed(start_token));

        specialize_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());

        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();
        assert!(lark.contains("start: <tool_call> (json_call | xml_call | glm_call)"));
        assert!(lark.contains("json_call: @json_body </tool_call>"));

        let parser = factory.create_parser(grammar).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(parser));
        let mask = matcher.compute_mask().unwrap();

        assert!(mask.is_allowed(start_token));
        matcher.consume_token(start_token).unwrap();
    }

    fn weather_tool() -> crate::Tool {
        let parameters: HashMap<String, Value> = serde_json::from_value(json!({
            "type": "object",
            "properties": { "city": { "type": "string" }, "days": { "type": "integer" } },
            "required": ["city"]
        }))
        .unwrap();
        crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: Some(parameters),
                strict: None,
            },
        }
    }

    /// Feeds `body` to `matcher` byte by byte, each of which the grammar must allow.
    fn accepts(matcher: &mut llguidance::Matcher, trie: &toktrie::TokTrie, body: &str) {
        for token in trie.greedy_tokenize(body.as_bytes()) {
            assert!(
                matcher.compute_mask().unwrap().is_allowed(token),
                "rejected byte {token} of {body:?}"
            );
            matcher.consume_token(token).unwrap();
        }
    }

    /// The grammar that constrains a call once `<tool_call>` opens it lets a GLM model write its
    /// call: the name, then each argument as a key and value pair, in any order, a value holding
    /// `<` of its own, then the closer.
    #[test]
    fn continuation_grammar_allows_a_glm_call() {
        let grammar = QwenParser.tool_call_grammar(&[weather_tool()], "<tool_call>");
        let trie = wrapper_token_trie();
        let end_token = trie.get_special_token("</tool_call>").unwrap();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie: trie.clone() });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(factory.create_parser(grammar).unwrap()));
        let body = "get_weather\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n<arg_key>city</arg_key>\n<arg_value>Paris </b> <arg_v</arg_value>\n";
        accepts(&mut matcher, &trie, body);
        assert!(matcher.compute_mask().unwrap().is_allowed(end_token));
        matcher.consume_token(end_token).unwrap();
    }

    /// A required call may be written in GLM's format, and only to an offered tool and its keys.
    #[test]
    fn required_grammar_allows_a_glm_call_to_an_offered_tool() {
        let mut grammar = QwenParser.required_tool_call_grammar(&[weather_tool()]);
        let trie = wrapper_token_trie();
        let start_token = trie.get_special_token("<tool_call>").unwrap();
        let end_token = trie.get_special_token("</tool_call>").unwrap();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie: trie.clone() });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        specialize_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());
        let parser = factory.create_parser(grammar.clone()).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(parser));
        matcher.consume_token(start_token).unwrap();
        accepts(
            &mut matcher,
            &trie,
            "get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>",
        );
        assert!(matcher.compute_mask().unwrap().is_allowed(end_token));

        // an unoffered key is masked where it would begin
        let parser = factory.create_parser(grammar).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(parser));
        matcher.consume_token(start_token).unwrap();
        accepts(&mut matcher, &trie, "get_weather<arg_key>");
        let z = trie.greedy_tokenize(b"z")[0];
        assert!(!matcher.compute_mask().unwrap().is_allowed(z));
    }

    /// A value may end in `<`, or in the start of its own closing tag, right before that tag:
    /// the tag still closes it, and is never read as more of the value, so nothing but the
    /// call's framing may follow it.
    #[test]
    fn a_value_may_end_in_the_start_of_its_closing_tag() {
        let trie = wrapper_token_trie();
        let end_token = trie.get_special_token("</tool_call>").unwrap();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie: trie.clone() });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let y = trie.greedy_tokenize(b"y")[0];
        for (body, close) in [
            (
                "get_weather<arg_key>city</arg_key><arg_value>x<",
                "</arg_value>",
            ),
            (
                "get_weather<arg_key>city</arg_key><arg_value>x</arg_v",
                "</arg_value>",
            ),
            (
                "get_weather<arg_key>city</arg_key><arg_value>x</a",
                "</arg_value>",
            ),
            (
                "get_weather<arg_key>city</arg_key><arg_value><<",
                "</arg_value>",
            ),
            (
                "<function=get_weather><parameter=city>x<",
                "</parameter></function>",
            ),
            (
                "<function=get_weather><parameter=city>x</param",
                "</parameter></function>",
            ),
        ] {
            let grammar = QwenParser.tool_call_grammar(&[weather_tool()], "<tool_call>");
            let mut matcher = llguidance::Matcher::new(Ok(factory.create_parser(grammar).unwrap()));
            accepts(&mut matcher, &trie, body);
            accepts(&mut matcher, &trie, close);
            assert!(
                matcher.compute_mask().unwrap().is_allowed(end_token),
                "{body}{close} closes the call"
            );
            assert!(
                !matcher.compute_mask().unwrap().is_allowed(y),
                "{body}{close}: the closing tag is not read as more of the value"
            );
        }
    }

    #[test]
    fn continuation_grammar_allows_eos_or_another_call_after_a_block() {
        let parameters: HashMap<String, Value> = serde_json::from_value(json!({
            "type": "object",
            "properties": { "city": { "type": "string" }, "days": { "type": "integer" } },
            "required": ["city"]
        }))
        .unwrap();
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: Some(parameters),
                strict: None,
            },
        };
        let grammar = QwenParser.tool_call_grammar(&[tool], "<tool_call>");
        let trie = wrapper_token_trie();
        let start_token = trie.get_special_token("<tool_call>").unwrap();
        let end_token = trie.get_special_token("</tool_call>").unwrap();
        let eos = trie.eos_token();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie: trie.clone() });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let parser = factory.create_parser(grammar).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(parser));

        // Parameters in non-schema order, a `</` inside a value
        let body = "\n<function=get_weather>\n<parameter=days>\n3\n</parameter>\n<parameter=city>\nParis </b>\n</parameter>\n</function>\n";
        for token in trie.greedy_tokenize(body.as_bytes()) {
            assert!(
                matcher.compute_mask().unwrap().is_allowed(token),
                "rejected byte {token}"
            );
            matcher.consume_token(token).unwrap();
        }
        assert!(matcher.compute_mask().unwrap().is_allowed(end_token));
        matcher.consume_token(end_token).unwrap();

        let mask = matcher.compute_mask().unwrap();
        assert!(
            mask.is_allowed(eos),
            "EOS must close the turn after a complete call"
        );
        let newline = trie.greedy_tokenize(b"\n")[0];
        assert!(mask.is_allowed(newline));
        matcher.consume_token(newline).unwrap();
        assert!(matcher.compute_mask().unwrap().is_allowed(start_token));
        matcher.consume_token(start_token).unwrap();
    }
}
