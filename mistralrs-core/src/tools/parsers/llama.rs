//! Llama tool call parser.
//!
//! Format: `<|python_tag|>{"name":"...", "parameters":{...}}`

use llguidance::api::TopLevelGrammar;

use super::ToolFormatParser;
use crate::Tool;

pub struct LlamaParser;

impl ToolFormatParser for LlamaParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("<|python_tag|>")
    }

    /// A call is the marker and then the JSON of a call, or of an array of calls, running to
    /// the end of the message.
    fn open_at_end(&self, text: &str) -> bool {
        super::json_call_open_after(text, "<|python_tag|>", |rest| {
            let rest = rest.trim_start();
            rest.is_empty() || rest.starts_with(['{', '['])
        })
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Llama
    }

    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            r#"start: @json_body"#.to_string(),
            tools,
            "parameters",
            false,
        )
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            r#"start: "<|python_tag|>" @json_body"#.to_string(),
            tools,
            "parameters",
            false,
        )
    }

    fn parse(&self, message: &str) -> candle_core::Result<Option<String>> {
        let prefix = "<|python_tag|>";
        if let Some(pos) = message.find(prefix) {
            Ok(Some(message[pos + prefix.len()..].to_string()))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaParser;
    use crate::tools::parsers::ToolFormatParser;

    #[test]
    fn parses_tool_call_after_text() {
        let parsed = LlamaParser
            .parse(r#"I'll check.<|python_tag|>{"name":"search","parameters":{"query":"rust"}}"#)
            .unwrap()
            .unwrap();
        assert_eq!(parsed, r#"{"name":"search","parameters":{"query":"rust"}}"#);
    }
}
