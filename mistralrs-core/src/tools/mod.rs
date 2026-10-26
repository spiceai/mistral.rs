pub(crate) mod grammar;
pub(crate) mod parsers;
mod request;
mod response;
pub(crate) mod state;
pub(crate) mod strategy;

use candle_core::Result;
pub(crate) use parsers::specialize_tool_call_grammar;
pub use parsers::ToolCallFormat;
pub use request::*;
pub use response::*;
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde_json::{Map, Value};
pub(crate) use state::ToolCallState;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use uuid::Uuid;

use mistralrs_mcp::CalledFunction;

pub use mistralrs_mcp::{ToolCallback, ToolCallbackWithTool};

/// Collection of callbacks keyed by tool name.
pub type ToolCallbacks = HashMap<String, Arc<ToolCallback>>;

/// Collection of callbacks with their tool definitions keyed by tool name.
pub type ToolCallbacksWithTools = HashMap<String, ToolCallbackWithTool>;

fn contains_tool_call_prefix(prefix: &str) -> bool {
    parsers::contains_tool_call_prefix(prefix)
}

fn process_model_specific_message(message: &str) -> Result<String> {
    parsers::process_model_specific_message(message)
}

pub struct ToolCallingMatcher {
    tool_choice: ToolChoice,
    known_tool_names: Option<std::collections::HashSet<String>>,
    tools: Option<Arc<Vec<crate::Tool>>>,
}

// Same as CalledFunction, but has different cases for variations on the names
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunctionParameters {
    #[serde(alias = "function")]
    pub name: String,
    #[serde(alias = "arguments", deserialize_with = "flexible_args")]
    pub parameters: Value,
}

// Accept either `{...}` **or** a `"stringified { ... }"`
fn flexible_args<'de, D>(d: D) -> std::result::Result<Value, D::Error>
where
    D: Deserializer<'de>,
{
    struct ArgVisitor;

    impl<'de> Visitor<'de> for ArgVisitor {
        type Value = Value;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("an object or a JSON-encoded string containing an object")
        }

        // Case 1 – the good case: already a JSON object
        fn visit_map<M>(self, mut m: M) -> std::result::Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut map = Map::new();
            while let Some((k, v)) = m.next_entry()? {
                map.insert(k, v);
            }
            Ok(Value::Object(map))
        }

        // Case 2 – got a *string*; try parsing it as JSON
        fn visit_str<E>(self, s: &str) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            serde_json::from_str(s).map_err(|e| E::custom(format!("inner JSON error: {e}")))
        }
    }

    d.deserialize_any(ArgVisitor)
}

/// Fixup potentially broken JSON
/// 1) allow/handle arguments as maps in quotations
fn fix_broken_json(raw: &str) -> anyhow::Result<String> {
    // Only apply the fix if the first pattern matches - otherwise we might corrupt valid JSON
    // where arguments is a properly escaped string containing `}`
    if raw.contains(r#""arguments":"{"#) {
        // 1) Delete the opening quote that shouldn't be there
        let tmp = raw.replacen(r#""arguments":"{"#, r#""arguments":{"#, 1);
        // 2) Delete the closing quote that matches it
        let fixed = tmp.replacen(r#"}"}"#, r#"}}"#, 1);
        Ok(fixed)
    } else {
        Ok(raw.to_string())
    }
}

/// `message` with `fix_broken_json`'s repair, unless `format` writes exact JSON of its own: a
/// DSML call's arguments are what the parser read, so a string parameter named `arguments`
/// holding `{}` is a string, which the repair would turn into an object.
fn repaired_unless_exact(
    message: String,
    format: Option<ToolCallFormat>,
) -> anyhow::Result<String> {
    if format == Some(ToolCallFormat::DeepSeekDsml) {
        Ok(message)
    } else {
        fix_broken_json(&message)
    }
}

impl ToolCallingMatcher {
    #[cfg(test)]
    pub fn new(tool_choice: ToolChoice, tools: Option<&[crate::Tool]>) -> anyhow::Result<Self> {
        Self::new_with_format(tool_choice, tools, None)
    }

    pub fn new_with_format(
        tool_choice: ToolChoice,
        tools: Option<&[crate::Tool]>,
        _preferred_tool_call_format: Option<ToolCallFormat>,
    ) -> anyhow::Result<Self> {
        let selected_tools = match &tool_choice {
            ToolChoice::Builtin(choice) => {
                anyhow::bail!(
                    "tool_choice forcing hosted tool `{}` is not supported.",
                    choice.tp.kind()
                );
            }
            ToolChoice::AllowedTools(choice) => {
                let tools = tools.unwrap_or_default();
                let mut seen = std::collections::HashSet::new();
                let mut matching_tools = Vec::new();
                for allowed_tool in &choice.tools {
                    let AllowedToolChoice::Function { name } = allowed_tool else {
                        anyhow::bail!(
                            "tool_choice.allowed_tools contains hosted tool `{}`; hosted tool forcing is not supported.",
                            allowed_tool.kind()
                        );
                    };
                    if !seen.insert(name.as_str()) {
                        continue;
                    }
                    let Some(tool) = tools.iter().find(|tool| tool.function.name == *name) else {
                        anyhow::bail!("tool_choice references unknown tool `{name}`.");
                    };
                    matching_tools.push(tool.clone());
                }
                if matching_tools.is_empty() {
                    anyhow::bail!("tool_choice.allowed_tools requires at least one function tool.");
                }
                Some(matching_tools)
            }
            _ => {
                if let Some(name) = tool_choice.forced_function_name() {
                    let tools = tools.unwrap_or_default();
                    let matching_tools = tools
                        .iter()
                        .filter(|tool| tool.function.name == name)
                        .cloned()
                        .collect::<Vec<_>>();
                    if matching_tools.is_empty() {
                        anyhow::bail!("tool_choice references unknown tool `{name}`.");
                    }
                    Some(matching_tools)
                } else {
                    tools.map(|tools| tools.to_vec())
                }
            }
        };
        let known_tool_names = selected_tools.as_ref().map(|t| {
            t.iter()
                .map(|tool| tool.function.name.clone())
                .collect::<std::collections::HashSet<_>>()
        });
        let tools_arc = selected_tools.map(Arc::new);
        Ok(Self {
            tool_choice,
            known_tool_names,
            tools: tools_arc,
        })
    }

    pub fn requires_tool_call(&self) -> bool {
        self.tool_choice.requires_tool_call()
    }

    pub(crate) fn allows_tool_call(&self) -> bool {
        !matches!(self.tool_choice, ToolChoice::None)
    }

    pub(crate) fn tools(&self) -> Option<&[crate::Tool]> {
        self.tools.as_ref().map(|tools| tools.as_slice())
    }

    // Checks if the `message_prefix` could be a tool call. If false, either
    // [`ToolChoice::None`] was selected, or the prefix could not match.
    //
    // If the start of a message could be a tool call, then it looks like an incomplete JSON of a given structure, e.g. `{"name": "foo", "param`.
    //
    // Returns a tuple of `(could_be_tool, is_complete_tool)`.
    pub fn prefix_could_be_tool(&self, message_prefix: &str) -> Result<(bool, bool)> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok((false, false));
        }
        let raw_prefix = message_prefix;
        let (message_prefix, format) = match parsers::extract_model_specific_call(message_prefix)? {
            Some((json, _, format)) => (json, Some(format)),
            None => (process_model_specific_message(message_prefix)?, None),
        };
        let message_prefix =
            repaired_unless_exact(message_prefix, format).map_err(candle_core::Error::msg)?;

        // Check if the prefix could be a JSON serialization of any of the following types.
        let (could_be_tool, is_complete_tool) = [
            could_be_json::<CalledFunctionParameters>,
            could_be_json::<Vec<CalledFunctionParameters>>,
        ]
        .iter()
        .find_map(|check| {
            let (could_be_tool, is_complete_tool) = check(&message_prefix);
            if could_be_tool || is_complete_tool {
                Some((could_be_tool, is_complete_tool))
            } else {
                None
            }
        })
        .unwrap_or((contains_tool_call_prefix(&message_prefix), false));
        // Qwen emits parallel calls as consecutive blocks; keep buffering until the model closes
        // the turn. Only a Qwen call is read this way: another format's value may hold a Qwen
        // closer as text.
        if is_complete_tool
            && format == Some(ToolCallFormat::Qwen)
            && qwen_more_calls_possible(raw_prefix)
        {
            return Ok((true, false));
        }
        // A DSML block makes the calls that survive being read: a call to a strict tool that its
        // schema refuses, or to a tool not offered, is not made. A block that makes none is
        // content, as a malformed one is, and the message stays held back from it until the turn
        // ends. A required call that makes none fails when read, so it stays complete and fails
        // at once.
        if is_complete_tool
            && format == Some(ToolCallFormat::DeepSeekDsml)
            && !self.tool_choice.requires_tool_call()
        {
            let (_, calls) = self
                .get_call_with_content(raw_prefix)
                .map_err(candle_core::Error::msg)?;
            if calls.is_empty() {
                return Ok((true, false));
            }
        }
        Ok((could_be_tool, is_complete_tool))
    }

    pub fn get_call(&self, message: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
        self.get_call_with_content(message).map(|(_, calls)| calls)
    }

    pub fn get_call_with_content(
        &self,
        message: &str,
    ) -> anyhow::Result<(Option<String>, Vec<ToolCallResponse>)> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok((Some(message.to_string()), Vec::new()));
        }
        let written = message;
        let (message, content, format) = if let Some((message, content, format)) =
            parsers::extract_model_specific_call(message)?
        {
            let content = content.trim_start().to_string();
            let content = if content.is_empty() {
                None
            } else {
                Some(content)
            };
            (message, content, Some(format))
        } else {
            (process_model_specific_message(message)?, None, None)
        };
        let message = repaired_unless_exact(message, format)?;

        let mut calls = if let Ok(deser) =
            serde_json::from_str::<CalledFunctionParameters>(&message)
        {
            let id = format!("call-{}", Uuid::new_v4());
            vec![ToolCallResponse {
                index: 0,
                id,
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: deser.name,
                    arguments: serde_json::to_string(&deser.parameters)?,
                },
            }]
        } else if let Ok(deser) = serde_json::from_str::<Vec<CalledFunctionParameters>>(&message) {
            deser
                .into_iter()
                .enumerate()
                .map(|(idx, deser)| {
                    let id = format!("call-{}", Uuid::new_v4());
                    Ok(ToolCallResponse {
                        index: idx,
                        id,
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: deser.name,
                            arguments: serde_json::to_string(&deser.parameters)?,
                        },
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            if self.tool_choice.requires_tool_call() {
                anyhow::bail!("Tool choice was required but no tools were called.")
            }
            // No call: the content is the message as the model wrote it, not as it was repaired
            // to be read as one.
            return Ok((Some(written.to_string()), Vec::new()));
        };

        // DSML marks each value's type itself, so its arguments are already what the model
        // wrote; a string `null` stays a string. A call to a strict tool is held to the tool's
        // schema here, as the grammar that holds the model to it while it writes may not have
        // run: the token that closed the block's opening tag carried more than whitespace, or
        // the schema is one the grammar cannot hold.
        if let Some(tools) = self.tools.as_deref() {
            if format == Some(ToolCallFormat::DeepSeekDsml) {
                retain_calls_within_strict_schemas(
                    tools,
                    &mut calls,
                    self.tool_choice.requires_tool_call(),
                )?;
            } else {
                coerce_arguments_by_schema(tools, &mut calls)?;
            }
        }

        // Filter out hallucinated tool names.
        if let Some(ref known) = self.known_tool_names {
            let before = calls.len();
            calls.retain(|tc| {
                let valid = known.contains(&tc.function.name);
                if !valid {
                    tracing::warn!(
                        "Dropping hallucinated tool call `{}` (not in defined tools: {:?})",
                        tc.function.name,
                        known
                    );
                }
                valid
            });
            if calls.is_empty() && before > 0 && self.tool_choice.requires_tool_call() {
                anyhow::bail!("Tool choice was required but model called unknown tools.");
            }
        }

        // A DSML block that makes no call is content, as a malformed one is: the message is
        // returned as the model wrote it, the block included, rather than with the block gone
        // and no call in its place.
        if calls.is_empty() && format == Some(ToolCallFormat::DeepSeekDsml) {
            return Ok((Some(written.to_string()), calls));
        }

        Ok((content, calls))
    }
}

const QWEN_TOOL_CALL_OPEN: &str = "<tool_call>";
const QWEN_TOOL_CALL_CLOSE: &str = "</tool_call>";

// True while the text after the last complete <tool_call> block is empty or an unfinished opener of the next block
fn qwen_more_calls_possible(text: &str) -> bool {
    let Some(last_close) = text.rfind(QWEN_TOOL_CALL_CLOSE) else {
        return false;
    };
    let tail = text[last_close + QWEN_TOOL_CALL_CLOSE.len()..].trim_start();
    tail.is_empty()
        || QWEN_TOOL_CALL_OPEN.starts_with(tail)
        || tail.starts_with(QWEN_TOOL_CALL_OPEN)
}

/// Drops from `calls` each call to a strict tool whose arguments do not match the tool's
/// parameters schema, and fails when a call is `required` and none is left: a strict tool is
/// promised arguments that match its schema, so a call with others is not made.
fn retain_calls_within_strict_schemas(
    tools: &[crate::Tool],
    calls: &mut Vec<ToolCallResponse>,
    required: bool,
) -> anyhow::Result<()> {
    let mut dropped = None;
    calls.retain(|call| {
        let Some(schema) = tools
            .iter()
            .find(|tool| tool.function.name == call.function.name)
            .and_then(|tool| tool.function.strict_parameters_schema())
        else {
            return true;
        };
        let Some(why) = strict_schema_mismatch(&schema, &call.function.arguments) else {
            return true;
        };
        tracing::warn!("Dropping tool call `{}`: it {why}", call.function.name);
        dropped.get_or_insert_with(|| format!("`{}` {why}", call.function.name));
        false
    });
    if let Some(dropped) = dropped.filter(|_| required && calls.is_empty()) {
        anyhow::bail!("Tool choice was required but the model's call to strict tool {dropped}");
    }
    Ok(())
}

/// Why `arguments` do not match the strict tool's parameters `schema`, if they do not.
fn strict_schema_mismatch(schema: &Value, arguments: &str) -> Option<String> {
    let arguments: Value = match serde_json::from_str(arguments) {
        Ok(arguments) => arguments,
        Err(error) => return Some(format!("has arguments that are not JSON: {error}")),
    };
    let validator = match jsonschema::options()
        .should_validate_formats(false)
        .build(schema)
    {
        Ok(validator) => validator,
        Err(error) => {
            return Some(format!(
                "has a parameters schema its arguments cannot be checked against: {error}. Fix the schema, or remove `strict` from the tool"
            ))
        }
    };
    validator.validate(&arguments).err().map(|error| {
        format!(
            "has arguments that do not match its parameters schema at `{}`: {error}",
            error.instance_path()
        )
    })
}

/// XML-style tool formats (Qwen3.5) carry every argument as text; the tool schema decides its JSON type (vLLM parity).
fn coerce_arguments_by_schema(
    tools: &[crate::Tool],
    calls: &mut [ToolCallResponse],
) -> anyhow::Result<()> {
    for call in calls.iter_mut() {
        let Some(tool) = tools
            .iter()
            .find(|tool| tool.function.name == call.function.name)
        else {
            continue;
        };
        let Some(properties) = tool
            .function
            .parameters
            .as_ref()
            .and_then(|parameters| parameters.get("properties"))
            .and_then(Value::as_object)
        else {
            continue;
        };
        let Ok(Value::Object(mut arguments)) =
            serde_json::from_str::<Value>(&call.function.arguments)
        else {
            continue;
        };
        for (key, value) in arguments.iter_mut() {
            let Value::String(raw) = value else {
                continue;
            };
            let param_type = properties
                .get(key)
                .and_then(|property| property.get("type"))
                .and_then(Value::as_str)
                .unwrap_or("string");
            *value = coerce_param_value(raw, param_type);
        }
        call.function.arguments = serde_json::to_string(&arguments)?;
    }
    Ok(())
}

fn coerce_param_value(raw: &str, param_type: &str) -> Value {
    if raw.eq_ignore_ascii_case("null") {
        return Value::Null;
    }
    let param_type = param_type.trim().to_ascii_lowercase();
    let integer_like = ["int", "uint", "long", "short", "unsigned"]
        .iter()
        .any(|prefix| param_type.starts_with(prefix));
    let number_like = param_type.starts_with("num") || param_type.starts_with("float");
    let structured = matches!(param_type.as_str(), "object" | "array" | "arr" | "sequence")
        || param_type.starts_with("dict")
        || param_type.starts_with("list");
    if integer_like {
        raw.trim()
            .parse::<i64>()
            .map(Value::from)
            .unwrap_or_else(|_| Value::String(raw.to_string()))
    } else if number_like {
        let trimmed = raw.trim();
        trimmed
            .parse::<i64>()
            .map(Value::from)
            .or_else(|_| trimmed.parse::<f64>().map(Value::from))
            .unwrap_or_else(|_| Value::String(raw.to_string()))
    } else if matches!(param_type.as_str(), "boolean" | "bool" | "binary") {
        Value::Bool(raw.trim().eq_ignore_ascii_case("true"))
    } else if structured {
        serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_string()))
    } else {
        Value::String(raw.to_string())
    }
}

/// Checks if the given prefix could be the start of, or the entire JSON serialization of a given type, `T`.
///
/// Returns a tuple of `(could_be_tool, is_entire_tool)`.
fn could_be_json<T>(text_prefix: &str) -> (bool, bool)
where
    T: serde::de::DeserializeOwned,
{
    if text_prefix.trim().is_empty() {
        return (false, false);
    }
    match serde_json::from_str::<T>(text_prefix) {
        Ok(_) => (false, true),
        // EOF show that JSON parsing was successful up to the end of the entire string.
        Err(e) if e.is_eof() => (true, false),
        _ => (false, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Function, Tool, ToolType};
    use serde_json::json;

    fn test_tool(name: &str) -> Tool {
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: None,
                name: name.to_string(),
                parameters: None,
                strict: None,
            },
        }
    }

    #[test]
    fn deserializes_responses_named_function_tool_choice() {
        let choice: ToolChoice =
            serde_json::from_value(json!({ "type": "function", "name": "get_weather" })).unwrap();

        let ToolChoice::NamedFunction(choice) = choice else {
            panic!("expected named function tool choice");
        };
        assert_eq!(choice.name, "get_weather");
    }

    #[test]
    fn deserializes_chat_function_tool_choice() {
        let choice: ToolChoice = serde_json::from_value(json!({
            "type": "function",
            "function": { "name": "get_weather" }
        }))
        .unwrap();

        let ToolChoice::Tool(tool) = choice else {
            panic!("expected chat function tool choice");
        };
        assert_eq!(tool.function.name, "get_weather");
    }

    #[test]
    fn tool_call_allowed_tools_deserializes_required_function_subset() {
        let choice: ToolChoice = serde_json::from_value(json!({
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{ "type": "function", "name": "get_weather" }]
        }))
        .unwrap();

        let ToolChoice::AllowedTools(choice) = choice else {
            panic!("expected allowed_tools tool choice");
        };
        assert_eq!(choice.mode, AllowedToolsMode::Required);
        assert_eq!(choice.tools.len(), 1);
    }

    #[test]
    fn specific_tool_choice_rejects_unknown_tool() {
        let tools = vec![test_tool("get_weather")];
        let choice: ToolChoice =
            serde_json::from_value(json!({ "type": "function", "name": "get_customer" })).unwrap();

        assert!(ToolCallingMatcher::new(choice, Some(&tools)).is_err());
    }

    #[test]
    fn specific_tool_choice_constrains_called_tool() {
        let tools = vec![test_tool("get_weather"), test_tool("get_customer")];
        let choice: ToolChoice =
            serde_json::from_value(json!({ "type": "function", "name": "get_weather" })).unwrap();
        let matcher = ToolCallingMatcher::new(choice, Some(&tools)).unwrap();

        assert!(matcher
            .get_call(r#"{"name":"get_customer","parameters":{}}"#)
            .is_err());
        let calls = matcher
            .get_call(r#"{"name":"get_weather","parameters":{}}"#)
            .unwrap();
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn tool_call_allowed_tools_required_constrains_called_tool() {
        let tools = vec![test_tool("get_weather"), test_tool("get_customer")];
        let choice: ToolChoice = serde_json::from_value(json!({
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{ "type": "function", "name": "get_weather" }]
        }))
        .unwrap();
        let matcher = ToolCallingMatcher::new(choice, Some(&tools)).unwrap();

        assert!(matcher.requires_tool_call());
        assert!(matcher
            .get_call(r#"{"name":"get_customer","parameters":{}}"#)
            .is_err());
        let calls = matcher
            .get_call(r#"{"name":"get_weather","parameters":{}}"#)
            .unwrap();
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn tool_call_rejects_forced_hosted_tool_choice() {
        let tools = vec![test_tool("get_weather")];
        let choice: ToolChoice =
            serde_json::from_value(json!({ "type": "web_search_preview" })).unwrap();

        assert!(matches!(choice, ToolChoice::Builtin(_)));
        assert!(ToolCallingMatcher::new(choice, Some(&tools)).is_err());
    }

    #[test]
    fn qwen_xml_arguments_follow_the_tool_schema() {
        let mut tool = test_tool("get_weather");
        tool.function.parameters = Some(
            serde_json::from_value(json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "days": { "type": "integer" },
                    "units": { "type": "string" },
                    "detailed": { "type": "boolean" },
                    "coords": { "type": "array" }
                }
            }))
            .unwrap(),
        );
        let matcher = ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool])).unwrap();
        let calls = matcher
            .get_call(
                "<tool_call>\n<function=get_weather>\n<parameter=city>\n12345\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=units>\n\"C\"\n</parameter>\n<parameter=detailed>\ntrue\n</parameter>\n<parameter=coords>\n[1, 2]\n</parameter>\n</function>\n</tool_call>",
            )
            .unwrap();
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "12345");
        assert_eq!(args["days"], 3);
        assert_eq!(args["units"], "\"C\"");
        assert_eq!(args["detailed"], true);
        assert_eq!(args["coords"], json!([1, 2]));
    }

    #[test]
    fn qwen_parallel_calls_keep_buffering_until_the_turn_ends() {
        let tools = vec![test_tool("get_weather"), test_tool("get_time")];
        let matcher = ToolCallingMatcher::new(ToolChoice::Auto, Some(&tools)).unwrap();
        let first = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(matcher.prefix_could_be_tool(first).unwrap(), (true, false));
        assert_eq!(
            matcher
                .prefix_could_be_tool(&format!("{first}\n<tool_call>\n<function=get_time>"))
                .unwrap(),
            (true, false)
        );
        assert_eq!(
            matcher
                .prefix_could_be_tool(&format!("{first}\nDone."))
                .unwrap(),
            (false, true)
        );
        let both = format!(
            "{first}\n<tool_call>\n<function=get_time>\n<parameter=zone>\nUTC\n</parameter>\n</function>\n</tool_call>"
        );
        assert_eq!(matcher.get_call(&both).unwrap().len(), 2);
    }

    /// regression test for spiceai/mistral.rs#54: a DSML string `null` became JSON null
    #[test]
    fn a_dsml_string_value_keeps_its_text_through_the_matcher() {
        let tool: Tool = serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "note",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                },
            },
        }))
        .unwrap();
        let matcher = ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool])).unwrap();
        let calls = matcher
            .get_call(
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"note\">\n<｜DSML｜parameter name=\"text\" string=\"true\">null</｜DSML｜parameter>\n<｜DSML｜parameter name=\"count\" string=\"false\">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            )
            .unwrap();
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, json!({"text": "null", "count": 2}));
    }

    /// regression test for spiceai/mistral.rs#54: while a DSML call was unfinished, a
    /// `<tool_call>` example in one of its values read as a complete call in Qwen's format
    #[test]
    fn an_unfinished_dsml_call_keeps_other_formats_out_of_its_values() {
        let tool = |name: &str| -> Tool {
            serde_json::from_value(json!({
                "type": "function",
                "function": { "name": name, "parameters": { "type": "object" } },
            }))
            .unwrap()
        };
        let matcher =
            ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool("Bash"), tool("write_file")]))
                .unwrap();
        let prefix = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"content\" string=\"true\">Call it as <tool_call>{\"name\": \"Bash\", \"arguments\": {}}</tool_call> and then";
        assert_eq!(matcher.prefix_could_be_tool(prefix).unwrap(), (true, false));
        assert!(matcher.get_call(prefix).unwrap().is_empty());
        let finished =
            format!("{prefix}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>");
        let calls = matcher.get_call(&finished).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "write_file");
    }

    /// regression test for spiceai/mistral.rs#54: a DSML opener inside another format's call
    /// claimed the message for DSML, and the call it sat in was lost
    #[test]
    fn a_dsml_opener_inside_another_formats_call_is_that_calls_text() {
        let tool: Tool = serde_json::from_value(json!({
            "type": "function",
            "function": { "name": "write_file", "parameters": { "type": "object" } },
        }))
        .unwrap();
        let matcher = ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool])).unwrap();
        let calls = matcher
            .get_call(r#"<tool_call>{"name": "write_file", "arguments": {"content": "<｜DSML｜tool_calls>"}}</tool_call>"#)
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "write_file");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, json!({"content": "<｜DSML｜tool_calls>"}));
    }

    /// regression test for spiceai/mistral.rs#54: a closed `<tool_call>` example before a DSML
    /// block counted as enclosing it, and a Qwen call inside the block's value was made
    #[test]
    fn a_closed_example_before_a_dsml_block_does_not_enclose_it() {
        let tool = |name: &str| -> Tool {
            serde_json::from_value(json!({
                "type": "function",
                "function": { "name": name, "parameters": { "type": "object" } },
            }))
            .unwrap()
        };
        let matcher =
            ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool("Bash"), tool("write_file")]))
                .unwrap();
        let message = "An empty example: <tool_call></tool_call>\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"content\" string=\"true\">Run <tool_call>{\"name\": \"Bash\", \"arguments\": {}}</tool_call></｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let calls = matcher.get_call(message).unwrap();
        let names: Vec<&str> = calls
            .iter()
            .map(|call| call.function.name.as_str())
            .collect();
        assert_eq!(names, ["write_file"]);
    }

    /// regression test for spiceai/mistral.rs#54: a DSML opener inside an earlier call in
    /// another format took DSML out of the message, and a later block was read as Qwen
    #[test]
    fn a_dsml_opener_inside_an_earlier_call_leaves_later_blocks_to_dsml() {
        let tool = |name: &str| -> Tool {
            serde_json::from_value(json!({
                "type": "function",
                "function": { "name": name, "parameters": { "type": "object" } },
            }))
            .unwrap()
        };
        let matcher =
            ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool("Bash"), tool("write_file")]))
                .unwrap();
        let message = "<tool_call>{\"name\": \"note\", \"arguments\": {\"text\": \"<｜DSML｜tool_calls>\"}}</tool_call>\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"content\" string=\"true\">Run <tool_call>{\"name\": \"Bash\", \"arguments\": {}}</tool_call></｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let calls = matcher.get_call(message).unwrap();
        let names: Vec<&str> = calls
            .iter()
            .map(|call| call.function.name.as_str())
            .collect();
        assert_eq!(names, ["write_file"]);
    }

    /// regression test for spiceai/mistral.rs#54: a DSML block opener that could not be read
    /// claimed nothing, and a Qwen call written inside it was made
    #[test]
    fn a_broken_dsml_block_opener_keeps_other_formats_out() {
        let tool: Tool = serde_json::from_value(json!({
            "type": "function",
            "function": { "name": "Bash", "parameters": { "type": "object" } },
        }))
        .unwrap();
        let matcher = ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool])).unwrap();
        for message in [
            "<｜DSML｜tool_calls x=\"<tool_call>{\"name\": \"Bash\", \"arguments\": {}}</tool_call>\">",
            "<｜DSML｜tool_calls <tool_call>{\"name\": \"Bash\", \"arguments\": {}}</tool_call>",
        ] {
            assert!(matcher.get_call(message).unwrap().is_empty(), "{message}");
        }
    }

    /// regression test for spiceai/mistral.rs#54: a block whose tags could not be read was
    /// streamed out as text, and what the model wrote after it was read on its own, so a block
    /// or a `<tool_call>` example after it made a call the whole message does not
    #[test]
    fn a_block_that_cannot_be_read_holds_the_message_and_calls_nothing() {
        let tool = |name: &str| -> Tool {
            serde_json::from_value(json!({
                "type": "function",
                "function": { "name": name, "parameters": { "type": "object" } },
            }))
            .unwrap()
        };
        let matcher =
            ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool("Bash"), tool("write_file")]))
                .unwrap();
        let broken = "Sure.\n<｜DSML｜tool_calls>!";
        assert_eq!(matcher.prefix_could_be_tool(broken).unwrap(), (true, false));
        let later_block = format!(
            "{broken}\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"content\" string=\"true\">x</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        );
        assert_eq!(
            matcher.prefix_could_be_tool(&later_block).unwrap(),
            (true, false)
        );
        assert!(matcher.get_call(&later_block).unwrap().is_empty());
        let example =
            format!("{broken} <tool_call>{{\"name\": \"Bash\", \"arguments\": {{}}}}</tool_call>");
        assert_eq!(
            matcher.prefix_could_be_tool(&example).unwrap(),
            (true, false)
        );
        assert!(matcher.get_call(&example).unwrap().is_empty());
    }

    /// regression test for spiceai/mistral.rs#54: a marker written as argument text inside a
    /// finished call in one format counted as opening a call in another, which took a DSML
    /// block after it out of DSML's hands, and a `<tool_call>` example in that block was made
    #[test]
    fn a_marker_inside_a_finished_call_does_not_enclose_a_later_dsml_block() {
        let tool = |name: &str| -> Tool {
            serde_json::from_value(json!({
                "type": "function",
                "function": { "name": name, "parameters": { "type": "object" } },
            }))
            .unwrap()
        };
        let matcher =
            ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool("Bash"), tool("write_file")]))
                .unwrap();
        let message = "<tool_call>{\"name\": \"note\", \"arguments\": {\"text\": \"[TOOL_CALLS]\"}}</tool_call>\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"content\" string=\"true\">Run <tool_call>{\"name\": \"Bash\", \"arguments\": {}}</tool_call></｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let calls = matcher.get_call(message).unwrap();
        let names: Vec<&str> = calls
            .iter()
            .map(|call| call.function.name.as_str())
            .collect();
        assert_eq!(names, ["write_file"]);
    }

    /// regression test for spiceai/mistral.rs#54: a block that broke a rule while its tags
    /// stayed in place streamed out as content, and what the model wrote after it was read on
    /// its own, so a `<tool_call>` example after it made a call the whole message does not
    #[test]
    fn a_message_that_opened_a_block_streams_nothing_after_it() {
        let tool = |name: &str| -> Tool {
            serde_json::from_value(json!({
                "type": "function",
                "function": { "name": name, "parameters": { "type": "object" } },
            }))
            .unwrap()
        };
        let matcher =
            ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool("Bash"), tool("write_file")]))
                .unwrap();
        let empty = "Sure.\n<｜DSML｜tool_calls></｜DSML｜tool_calls>";
        assert_eq!(matcher.prefix_could_be_tool(empty).unwrap(), (true, false));
        let example =
            format!("{empty} <tool_call>{{\"name\": \"Bash\", \"arguments\": {{}}}}</tool_call>");
        assert_eq!(
            matcher.prefix_could_be_tool(&example).unwrap(),
            (true, false)
        );
        assert!(matcher.get_call(&example).unwrap().is_empty());
        // A valid block after it still calls, and completes the message.
        let later_block = format!(
            "{empty}\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"content\" string=\"true\">x</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        );
        assert_eq!(
            matcher.prefix_could_be_tool(&later_block).unwrap(),
            (false, true)
        );
        let calls = matcher.get_call(&later_block).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "write_file");
    }

    /// regression test for spiceai/mistral.rs#54: a finished call's span was blanked format by
    /// format, so an opener of one format written inside a call of another, with the closer
    /// after that call, blanked the enclosing call's closer and left it open
    #[test]
    fn a_marker_inside_a_finished_call_does_not_hide_that_calls_closer() {
        let tool = |name: &str| -> Tool {
            serde_json::from_value(json!({
                "type": "function",
                "function": { "name": name, "parameters": { "type": "object" } },
            }))
            .unwrap()
        };
        let matcher =
            ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool("Bash"), tool("write_file")]))
                .unwrap();
        let message = "<tool_call>{\"name\": \"note\", \"arguments\": {\"text\": \"<atem:function_calls>\"}}</tool_call></atem:function_calls>\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"content\" string=\"true\">Run <tool_call>{\"name\": \"Bash\", \"arguments\": {}}</tool_call></｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let calls = matcher.get_call(message).unwrap();
        let names: Vec<&str> = calls
            .iter()
            .map(|call| call.function.name.as_str())
            .collect();
        assert_eq!(names, ["write_file"]);
    }

    /// regression test for spiceai/mistral.rs#54: a Gemma 4 call was framed at the first
    /// `<tool_call|>`, even one inside a `<|"|>`-quoted value, so a DSML block later in that
    /// value read as outside any call and claimed the message for DSML
    #[test]
    fn a_closer_quoted_in_a_gemma_value_does_not_end_its_call() {
        let tool = |name: &str| -> Tool {
            serde_json::from_value(json!({
                "type": "function",
                "function": { "name": name, "parameters": { "type": "object" } },
            }))
            .unwrap()
        };
        let matcher =
            ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool("Bash"), tool("write_file")]))
                .unwrap();
        let value = "End a call with <tool_call|>, then run:\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"Bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">ls</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let message =
            format!("<|tool_call>call:write_file{{content:<|\"|>{value}<|\"|>}}<tool_call|>");
        let calls = matcher.get_call(&message).unwrap();
        let names: Vec<&str> = calls
            .iter()
            .map(|call| call.function.name.as_str())
            .collect();
        assert_eq!(names, ["write_file"]);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, json!({ "content": value }));
    }

    /// regression test for spiceai/mistral.rs#54: a DSML call to a strict tool was read as
    /// written, with nothing holding it to the tool's schema when no grammar had held the model
    /// to it, so `Bash`, which requires `command`, could be called without it
    #[test]
    fn a_dsml_call_to_a_strict_tool_is_held_to_its_schema_when_read() {
        let bash: Tool = serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "Bash",
                "strict": true,
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
        .unwrap();
        let call = |parameters: &str| {
            format!(
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"Bash\">\n{parameters}</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
            )
        };
        let command =
            "<｜DSML｜parameter name=\"command\" string=\"true\">ls</｜DSML｜parameter>\n";
        let without_command = call("");
        let timeout_as_text = call(&format!(
            "{command}<｜DSML｜parameter name=\"timeout\" string=\"true\">30</｜DSML｜parameter>\n"
        ));
        let valid = call(&format!(
            "{command}<｜DSML｜parameter name=\"timeout\" string=\"false\">30</｜DSML｜parameter>\n"
        ));
        let auto =
            ToolCallingMatcher::new(ToolChoice::Auto, Some(std::slice::from_ref(&bash))).unwrap();
        assert!(auto.get_call(&without_command).unwrap().is_empty());
        assert!(auto.get_call(&timeout_as_text).unwrap().is_empty());
        let calls = auto.get_call(&valid).unwrap();
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, json!({"command": "ls", "timeout": 30}));
        let required =
            ToolCallingMatcher::new(ToolChoice::Required, Some(std::slice::from_ref(&bash)))
                .unwrap();
        let error = required.get_call(&without_command).unwrap_err().to_string();
        assert!(
            error.contains("strict tool `Bash`")
                && error.contains("\"command\" is a required property"),
            "{error}"
        );
        assert_eq!(required.get_call(&valid).unwrap().len(), 1);
        // A tool that is not strict is read as written.
        let mut loose = bash;
        loose.function.strict = None;
        let auto = ToolCallingMatcher::new(ToolChoice::Auto, Some(&[loose])).unwrap();
        assert_eq!(auto.get_call(&without_command).unwrap().len(), 1);
    }

    /// regression test for spiceai/mistral.rs#54: a DSML block whose every call was refused when
    /// read, as a call to a strict tool with arguments its schema does not allow or to a tool
    /// not offered is, was still reported complete, with the block stripped from the content, so
    /// a stream ended with `finish_reason: "tool_calls"`, no calls, and the block gone. A block
    /// that makes no call is content, as a malformed one is: the message stays held back from it
    /// until the turn ends, then is returned as the model wrote it.
    #[test]
    fn a_dsml_block_whose_calls_are_all_refused_is_content_not_a_complete_call() {
        let bash: Tool = serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "Bash",
                "strict": true,
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
        .unwrap();
        let block = |invocations: &str| {
            format!("Running it.\n\n<｜DSML｜tool_calls>\n{invocations}</｜DSML｜tool_calls>")
        };
        let invoke = |name: &str, parameters: &str| {
            format!("<｜DSML｜invoke name=\"{name}\">\n{parameters}</｜DSML｜invoke>\n")
        };
        let command =
            "<｜DSML｜parameter name=\"command\" string=\"true\">ls</｜DSML｜parameter>\n";
        let timeout =
            "<｜DSML｜parameter name=\"timeout\" string=\"false\">30</｜DSML｜parameter>\n";
        let auto =
            ToolCallingMatcher::new(ToolChoice::Auto, Some(std::slice::from_ref(&bash))).unwrap();
        // Refused by the tool's schema, and a tool that is not offered.
        for refused in [
            block(&invoke("Bash", timeout)),
            block(&invoke("Shell", command)),
        ] {
            assert_eq!(
                auto.prefix_could_be_tool(&refused).unwrap(),
                (true, false),
                "{refused}"
            );
            let (content, calls) = auto.get_call_with_content(&refused).unwrap();
            assert_eq!(content.as_deref(), Some(refused.as_str()), "{refused}");
            assert!(calls.is_empty(), "{refused}");
        }
        // A block that makes a call beside one it does not still ends the turn with that call.
        let one_of_two = block(&format!(
            "{}{}",
            invoke("Bash", timeout),
            invoke("Bash", command)
        ));
        assert_eq!(
            auto.prefix_could_be_tool(&one_of_two).unwrap(),
            (false, true)
        );
        let (content, calls) = auto.get_call_with_content(&one_of_two).unwrap();
        assert_eq!(content.as_deref(), Some("Running it.\n\n"));
        let args: Vec<Value> = calls
            .iter()
            .map(|call| serde_json::from_str(&call.function.arguments).unwrap())
            .collect();
        assert_eq!(args, [json!({"command": "ls"})]);
        // A required call that makes none fails as soon as it is read.
        let required = ToolCallingMatcher::new(ToolChoice::Required, Some(&[bash])).unwrap();
        let refused = block(&invoke("Bash", timeout));
        assert_eq!(
            required.prefix_could_be_tool(&refused).unwrap(),
            (false, true)
        );
        let error = required.get_call(&refused).unwrap_err().to_string();
        assert!(
            error.contains("\"command\" is a required property"),
            "{error}"
        );
    }

    /// regression test for spiceai/mistral.rs#54: the check for a Qwen block of parallel calls
    /// still being written ran on a DSML call too, so a string value holding
    /// `</tool_call><tool_call>` kept a complete block from being reported complete: the stream
    /// stayed buffered, and a required call, never marked satisfied, could have a second block
    /// forced into the turn at its deadline
    #[test]
    fn a_qwen_closer_inside_a_dsml_value_does_not_hold_the_block_open() {
        let tool = test_tool("write");
        let value = "</tool_call><tool_call>";
        let message = format!(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write\">\n<｜DSML｜parameter name=\"body\" string=\"true\">{value}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        );
        for tool_choice in [ToolChoice::Auto, ToolChoice::Required] {
            let matcher =
                ToolCallingMatcher::new(tool_choice, Some(std::slice::from_ref(&tool))).unwrap();
            assert_eq!(
                matcher.prefix_could_be_tool(&message).unwrap(),
                (false, true)
            );
            let calls = matcher.get_call(&message).unwrap();
            assert_eq!(calls.len(), 1);
            let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
            assert_eq!(args, json!({"body": value}));
        }
    }

    /// regression test for spiceai/mistral.rs#54: a message that made no call was returned as
    /// content after the repair for a model's stringified `arguments`, so a DSML block the parser
    /// made content, followed by the text `{"arguments":"{}"}`, came back with `{"arguments":{}}`
    #[test]
    fn content_that_makes_no_call_is_returned_as_written() {
        let matcher = ToolCallingMatcher::new(ToolChoice::Auto, Some(&[test_tool("set")])).unwrap();
        let block_then_example =
            "<｜DSML｜tool_calls></｜DSML｜tool_calls>\n{\"arguments\":\"{}\"}";
        for message in [
            block_then_example,
            "The shape is {\"arguments\":\"{}\"} with the braces quoted.",
        ] {
            let (content, calls) = matcher.get_call_with_content(message).unwrap();
            assert_eq!(content.as_deref(), Some(message), "{message}");
            assert!(calls.is_empty(), "{message}");
        }
        // The block is DSML's, so the message is held back until the turn ends.
        assert_eq!(
            matcher.prefix_could_be_tool(block_then_example).unwrap(),
            (true, false)
        );
    }

    /// A tool that declares no parameters takes any in a forced call, so no grammar keeps the
    /// model from giving a parameter twice. Such a block is content, and a required call ends
    /// the request with an error rather than calling with one of the two values.
    #[test]
    fn a_required_call_with_a_parameter_given_twice_is_an_error_not_a_call() {
        let tool: Tool = serde_json::from_value(json!({
            "type": "function",
            "function": { "name": "set", "parameters": { "type": "object" } },
        }))
        .unwrap();
        let twice = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"set\">\n<｜DSML｜parameter name=\"a\" string=\"true\">1</｜DSML｜parameter>\n<｜DSML｜parameter name=\"a\" string=\"true\">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let required = ToolCallingMatcher::new(ToolChoice::Required, Some(&[tool])).unwrap();
        assert_eq!(
            required.get_call(twice).unwrap_err().to_string(),
            "Tool choice was required but no tools were called."
        );
    }

    /// regression test for spiceai/mistral.rs#54: a DSML block opener still being written, or
    /// only its marker, claimed nothing, so a finished `<tool_call>` example earlier in the
    /// message was dispatched as a complete call and the turn ended before the DSML call
    #[test]
    fn a_partial_dsml_opener_claims_the_message_before_an_example_is_dispatched() {
        let tool = |name: &str| -> Tool {
            serde_json::from_value(json!({
                "type": "function",
                "function": { "name": name, "parameters": { "type": "object" } },
            }))
            .unwrap()
        };
        let matcher =
            ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool("Bash"), tool("write_file")]))
                .unwrap();
        let example = "Like this: <tool_call>{\"name\": \"note\", \"arguments\": {}}</tool_call>\nNow for real:\n";
        for partial in [
            "<｜",
            "<｜DSML｜",
            "<｜DSML｜tool_cal",
            "<｜DSML｜tool_calls>\n<｜DSML｜inv",
        ] {
            let prefix = format!("{example}{partial}");
            assert_eq!(
                matcher.prefix_could_be_tool(&prefix).unwrap(),
                (true, false),
                "{partial:?}"
            );
            assert!(matcher.get_call(&prefix).unwrap().is_empty(), "{partial:?}");
        }
        let whole = format!(
            "{example}<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"content\" string=\"true\">x</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        );
        let calls = matcher.get_call(&whole).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "write_file");
        // A tag that is not a block's, or a marker inside another format's open call, claims nothing.
        assert!(
            matcher
                .prefix_could_be_tool(&format!("{example}<｜DSML｜note>"))
                .unwrap()
                .1
        );
    }

    /// regression test for spiceai/mistral.rs#54: the repair for a model's stringified
    /// `arguments` ran on the DSML parser's own JSON, so a string parameter named `arguments`
    /// holding `{}` reached the tool as an object
    #[test]
    fn a_dsml_string_parameter_named_arguments_stays_a_string() {
        let tool: Tool = serde_json::from_value(json!({
            "type": "function",
            "function": { "name": "note", "parameters": { "type": "object" } },
        }))
        .unwrap();
        let matcher = ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool])).unwrap();
        let message = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"note\">\n<｜DSML｜parameter name=\"arguments\" string=\"true\">{}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        assert_eq!(
            matcher.prefix_could_be_tool(message).unwrap(),
            (false, true)
        );
        let calls = matcher.get_call(message).unwrap();
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, json!({"arguments": "{}"}));
    }
}
