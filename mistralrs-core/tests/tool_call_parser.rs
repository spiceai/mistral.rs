//! The tool-call parser through the crate's public API, as another engine calls it on the
//! text a model generated.

use std::collections::HashMap;

use mistralrs_core::{Function, Tool, ToolCallingMatcher, ToolChoice, ToolType};
use serde_json::json;

fn weather_tool() -> Tool {
    let parameters: HashMap<String, serde_json::Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"],
    }))
    .expect("the schema is an object");
    Tool {
        tp: ToolType::Function,
        function: Function {
            name: "get_weather".to_string(),
            description: Some("The weather in a city".to_string()),
            parameters: Some(parameters),
            strict: None,
        },
    }
}

#[test]
fn a_qwen_tool_call_parses_into_its_name_and_arguments() {
    let tools = [weather_tool()];
    let matcher = ToolCallingMatcher::new_with_format(ToolChoice::Auto, Some(&tools), None)
        .expect("a matcher for one function tool");
    let text = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

    let (content, calls) = matcher
        .get_call_with_content(text)
        .expect("the text parses");

    assert_eq!(content.as_deref(), Some("Let me check.\n"));
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments)
            .expect("the arguments are JSON"),
        json!({ "city": "Paris" })
    );
}

#[test]
fn an_unfinished_tool_call_is_a_prefix_to_hold_back() {
    let tools = [weather_tool()];
    let matcher = ToolCallingMatcher::new_with_format(ToolChoice::Auto, Some(&tools), None)
        .expect("a matcher for one function tool");

    let (could_be_tool, is_complete) = matcher
        .prefix_could_be_tool("<tool_call>\n{\"name\": \"get_wea")
        .expect("a prefix is checked");

    assert_eq!((could_be_tool, is_complete), (true, false));
}

#[test]
fn tool_choice_none_returns_the_text_as_content() {
    let tools = [weather_tool()];
    let matcher = ToolCallingMatcher::new_with_format(ToolChoice::None, Some(&tools), None)
        .expect("a matcher for one function tool");
    let text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

    let (content, calls) = matcher
        .get_call_with_content(text)
        .expect("the text parses");

    assert!(calls.is_empty(), "no call is parsed under tool_choice none: {calls:?}");
    assert_eq!(content.as_deref(), Some(text));
}
