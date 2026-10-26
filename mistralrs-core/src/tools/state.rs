use candle_core::Result;
use llguidance::api::TopLevelGrammar;

use crate::tools::{
    strategy::{
        AtemToolCallStrategy, HarmonyToolCallStrategy, TextToolCallStrategy, ToolCallStrategy,
    },
    ToolCallFormat, ToolCallResponse, ToolCallingMatcher, ToolChoice,
};

const REQUIRED_TOOL_CALL_DEADLINE_DIVISOR: usize = 4;
const REQUIRED_TOOL_CALL_DEADLINE_MIN_TOKENS: usize = 1024;
const REQUIRED_TOOL_CALL_DEADLINE_MAX_TOKENS: usize = 4096;

#[derive(Clone, Copy, Debug, Default)]
struct ToolObligation {
    satisfied: bool,
    forced: bool,
}

impl ToolObligation {
    fn unsatisfied(self, requires_tool_call: bool) -> bool {
        requires_tool_call && !self.satisfied
    }

    fn mark_satisfied(&mut self, requires_tool_call: bool) {
        if requires_tool_call {
            self.satisfied = true;
        }
    }

    fn should_force(
        self,
        requires_tool_call: bool,
        max_generation_len: usize,
        remaining: usize,
    ) -> bool {
        if !self.unsatisfied(requires_tool_call) || self.forced {
            return false;
        }
        remaining <= required_tool_call_deadline_tokens(max_generation_len)
    }

    fn mark_forced(&mut self) {
        self.forced = true;
    }

    fn clear_forced(&mut self) {
        self.forced = false;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum ToolGrammarState {
    #[default]
    Inactive,
    Active {
        forced: bool,
    },
}

pub(crate) struct ToolCallParse {
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCallResponse>,
    pub tool_use_still_possible: bool,
    pub tool_use_is_done: bool,
}

impl ToolCallParse {
    fn empty(content: Option<String>) -> Self {
        Self {
            content,
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_use_still_possible: false,
            tool_use_is_done: false,
        }
    }
}

pub(crate) struct ToolCallState {
    matcher: ToolCallingMatcher,
    strategy: Box<dyn ToolCallStrategy>,
    grammar: ToolGrammarState,
    obligation: ToolObligation,
}

impl ToolCallState {
    pub(crate) fn new(
        tool_choice: ToolChoice,
        tools: Option<&[crate::Tool]>,
        preferred_format: Option<ToolCallFormat>,
    ) -> anyhow::Result<Self> {
        let matcher = ToolCallingMatcher::new_with_format(tool_choice, tools, preferred_format)?;
        if matcher.requires_tool_call() {
            // Only the tools a call may be made to are held to their schemas.
            if let Some(why) = crate::tools::parsers::unenforceable_strict_tool(
                preferred_format,
                matcher.tools().unwrap_or_default(),
            ) {
                anyhow::bail!(why);
            }
        }
        let strategy: Box<dyn ToolCallStrategy> = match preferred_format {
            Some(ToolCallFormat::Harmony) => Box::new(HarmonyToolCallStrategy::new()?),
            Some(ToolCallFormat::Atem) => {
                Box::new(AtemToolCallStrategy::new(if matcher.allows_tool_call() {
                    matcher.tools()
                } else {
                    None
                }))
            }
            _ => Box::new(TextToolCallStrategy::new(preferred_format)),
        };
        Ok(Self {
            matcher,
            strategy,
            grammar: ToolGrammarState::Inactive,
            obligation: ToolObligation::default(),
        })
    }

    pub(crate) fn observe_token(&mut self, token: u32, bytes: &[u8]) {
        self.strategy.observe_token(token, bytes);
    }

    pub(crate) fn requires_special_tokens(&self) -> bool {
        true
    }

    pub(crate) fn has_reasoning(&self) -> bool {
        self.strategy.has_reasoning()
    }

    pub(crate) fn required_tool_call_unsatisfied(&self) -> bool {
        self.obligation
            .unsatisfied(self.matcher.requires_tool_call())
    }

    pub(crate) fn content_delta(&mut self) -> Option<String> {
        self.strategy.content_delta()
    }

    pub(crate) fn reasoning_delta(&mut self) -> Option<String> {
        self.strategy.reasoning_delta()
    }

    pub(crate) fn content(&self) -> Option<String> {
        self.strategy.content()
    }

    pub(crate) fn reasoning_content(&self) -> Option<String> {
        self.strategy.reasoning_content()
    }

    pub(crate) fn finalize(&mut self) {
        self.strategy.finalize();
    }

    pub(crate) fn maybe_activate_continuation_grammar(
        &mut self,
        text: Option<&str>,
    ) -> Option<TopLevelGrammar> {
        if self.grammar != ToolGrammarState::Inactive || !self.matcher.allows_tool_call() {
            return None;
        }
        let tools = self.matcher.tools()?;
        self.strategy.continuation_grammar(text, tools)
    }

    pub(crate) fn maybe_force_required_grammar(
        &mut self,
        remaining: usize,
        max_generation_len: usize,
        force_now: bool,
        content: Option<&str>,
    ) -> Option<TopLevelGrammar> {
        if self.grammar != ToolGrammarState::Inactive {
            return None;
        }
        let requires_tool_call = self.matcher.requires_tool_call();
        if !self.obligation.unsatisfied(requires_tool_call)
            || (!force_now
                && !self
                    .obligation
                    .should_force(requires_tool_call, max_generation_len, remaining))
        {
            return None;
        }
        // A forced call would start a new block inside the one being written; the model
        // finishes its own call instead. `content` is the response without its reasoning, as
        // prefix detection reads it, so a call written as an example in the reasoning counts
        // for nothing.
        if content.is_some_and(|content| self.strategy.call_unfinished(content)) {
            return None;
        }
        let tools = self.matcher.tools()?;
        let boundary = self.strategy.required_boundary();
        Some(self.strategy.required_grammar(tools, boundary))
    }

    pub(crate) fn required_tool_call_deadline_status(
        generated: usize,
        max_generation_len: usize,
    ) -> (usize, usize, usize) {
        let deadline = required_tool_call_deadline_tokens(max_generation_len);
        let remaining = max_generation_len.saturating_sub(generated);
        (generated, remaining, deadline)
    }

    pub(crate) fn mark_grammar_active(&mut self, forced: bool) {
        self.grammar = ToolGrammarState::Active { forced };
        if forced {
            self.obligation.mark_forced();
        }
    }

    pub(crate) fn clear_active_grammar(&mut self) -> bool {
        let ToolGrammarState::Active { forced } = self.grammar else {
            return false;
        };
        self.grammar = ToolGrammarState::Inactive;
        if forced || self.strategy.has_tool_calls() {
            self.obligation
                .mark_satisfied(self.matcher.requires_tool_call());
        }
        self.obligation.clear_forced();
        true
    }

    pub(crate) fn is_stop_token_blocked(
        &self,
        tok: u32,
        eos_tok: Option<&[u32]>,
        stop_tokens: &[u32],
    ) -> bool {
        self.obligation
            .unsatisfied(self.matcher.requires_tool_call())
            && (eos_tok.is_some_and(|tokens| tokens.contains(&tok)) || stop_tokens.contains(&tok))
    }

    pub(crate) fn prefix_status(&self, message_prefix: &str) -> Result<(bool, bool)> {
        self.matcher.prefix_could_be_tool(message_prefix)
    }

    pub(crate) fn stops_after_complete_tool_call(&self) -> bool {
        self.strategy.stops_after_complete_tool_call()
    }

    pub(crate) fn complete_if_tool_call(
        &mut self,
        message: &str,
    ) -> anyhow::Result<Vec<ToolCallResponse>> {
        let calls = self.matcher.get_call(message)?;
        if !calls.is_empty() {
            self.obligation
                .mark_satisfied(self.matcher.requires_tool_call());
        }
        Ok(calls)
    }

    pub(crate) fn parse_streaming(
        &mut self,
        content_delta: Option<String>,
        raw_delta: &str,
        parser_text: Option<&str>,
        has_external_reasoning: bool,
        is_done: bool,
    ) -> Result<ToolCallParse> {
        if self.strategy.has_reasoning() {
            if is_done && self.strategy.has_tool_calls() {
                self.obligation
                    .mark_satisfied(self.matcher.requires_tool_call());
                return Ok(ToolCallParse {
                    content: content_delta,
                    reasoning_content: None,
                    tool_calls: self.strategy.finalize_tool_calls(),
                    tool_use_still_possible: false,
                    tool_use_is_done: true,
                });
            }
            return Ok(ToolCallParse::empty(content_delta));
        }

        let visible_text = match content_delta {
            Some(content_delta) => content_delta,
            None if has_external_reasoning => return Ok(ToolCallParse::empty(None)),
            None => raw_delta.to_string(),
        };
        let parse_text = parser_text.unwrap_or(&visible_text);
        let (tool_use_still_possible, tool_use_is_done) =
            self.matcher.prefix_could_be_tool(parse_text)?;
        let (mut content, tool_calls) = self
            .matcher
            .get_call_with_content(parse_text)
            .map_err(candle_core::Error::msg)?;
        if parser_text.is_some() && tool_calls.is_empty() {
            content = self
                .matcher
                .get_call_with_content(&visible_text)
                .map_err(candle_core::Error::msg)?
                .0;
        }
        if !tool_calls.is_empty() {
            self.obligation
                .mark_satisfied(self.matcher.requires_tool_call());
        }
        Ok(ToolCallParse {
            content,
            reasoning_content: None,
            tool_use_still_possible,
            tool_use_is_done: tool_use_is_done || !tool_calls.is_empty(),
            tool_calls,
        })
    }

    pub(crate) fn finalize_for_response(
        &mut self,
        raw_text: &str,
        parsed_content: Option<String>,
        reasoning_content: Option<String>,
        parser_text: Option<&str>,
    ) -> Result<ToolCallParse> {
        if self.strategy.has_reasoning() {
            let tool_calls = self.strategy.finalize_tool_calls();
            if !tool_calls.is_empty() {
                self.obligation
                    .mark_satisfied(self.matcher.requires_tool_call());
            }
            return Ok(ToolCallParse {
                content: self.strategy.content(),
                reasoning_content: self.strategy.reasoning_content(),
                tool_use_still_possible: false,
                tool_use_is_done: !tool_calls.is_empty(),
                tool_calls,
            });
        }

        let visible_text = parsed_content.unwrap_or_else(|| raw_text.to_string());
        let parse_text = parser_text.unwrap_or(&visible_text);
        let (mut content, tool_calls) = self
            .matcher
            .get_call_with_content(parse_text)
            .map_err(candle_core::Error::msg)?;
        if parser_text.is_some() && tool_calls.is_empty() {
            content = self
                .matcher
                .get_call_with_content(&visible_text)
                .map_err(candle_core::Error::msg)?
                .0;
        }
        if !tool_calls.is_empty() {
            self.obligation
                .mark_satisfied(self.matcher.requires_tool_call());
        }
        Ok(ToolCallParse {
            content,
            reasoning_content,
            tool_use_still_possible: false,
            tool_use_is_done: !tool_calls.is_empty(),
            tool_calls,
        })
    }
}

pub(crate) fn required_tool_call_deadline_tokens(max_generation_len: usize) -> usize {
    (max_generation_len / REQUIRED_TOOL_CALL_DEADLINE_DIVISOR).clamp(
        REQUIRED_TOOL_CALL_DEADLINE_MIN_TOKENS,
        REQUIRED_TOOL_CALL_DEADLINE_MAX_TOKENS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::NamedFunctionToolChoice;
    use crate::{Function, Tool, ToolType};
    use serde_json::json;

    fn tool(name: &str) -> Tool {
        Tool {
            tp: ToolType::Function,
            function: Function {
                name: name.to_string(),
                description: None,
                parameters: None,
                strict: None,
            },
        }
    }

    fn lark(grammar: &TopLevelGrammar) -> &str {
        grammar.grammars[0].lark_grammar.as_ref().unwrap()
    }

    #[test]
    fn auto_text_prefix_tool_call_activates_continuation_grammar() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(ToolChoice::Auto, Some(&tools), None).unwrap();

        let grammar = state
            .maybe_activate_continuation_grammar(Some("<tool_call>"))
            .unwrap();

        assert!(lark(&grammar).contains("json_call"));
        assert_eq!(grammar.grammars.len(), 2);
    }

    /// regression test for spiceai/mistral.rs#54: no grammar took over a DSML call the model
    /// began on its own, so a strict tool's schema held only for a forced call
    #[test]
    fn a_dsml_block_opener_activates_the_continuation_grammar() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(
            ToolChoice::Auto,
            Some(&tools),
            Some(ToolCallFormat::DeepSeekDsml),
        )
        .unwrap();
        assert!(state
            .maybe_activate_continuation_grammar(Some("Let me check.\n<｜DSML｜tool_"))
            .is_none());
        let grammar = state
            .maybe_activate_continuation_grammar(Some("Let me check.\n<｜DSML｜tool_calls>"))
            .unwrap();
        assert!(
            lark(&grammar)
                .starts_with("start: DSML_WS? (invoke_0)+ \"</\" dsml_mark \"tool_calls>\"\n"),
            "{}",
            lark(&grammar)
        );
        assert!(
            lark(&grammar).contains(
                "\ninvoke_0: \"<\" dsml_mark \"invoke name=\\\"get_weather\\\">\\n\" arguments_0 \"</\" dsml_mark \"invoke>\\n\"\n"
            ),
            "{}",
            lark(&grammar)
        );
    }

    #[test]
    fn tool_call_choice_none_does_not_activate_continuation_grammar() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(ToolChoice::None, Some(&tools), None).unwrap();

        assert!(state
            .maybe_activate_continuation_grammar(Some("<tool_call>"))
            .is_none());
    }

    #[test]
    fn atem_tool_call_choice_none_does_not_return_calls() {
        let tools = vec![tool("get_weather")];
        let mut state =
            ToolCallState::new(ToolChoice::None, Some(&tools), Some(ToolCallFormat::Atem)).unwrap();
        state.observe_token(
            0,
            b" to=get_weather<|message|><atem:function_calls><atem:invoke name=\"get_weather\"></atem:invoke></atem:function_calls><|eot|>",
        );

        let parsed = state.parse_streaming(None, "", None, false, true).unwrap();
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn required_tool_call_deadline_forces_text_grammar() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(
            ToolChoice::Required,
            Some(&tools),
            Some(ToolCallFormat::Gemma4),
        )
        .unwrap();

        let grammar = state
            .maybe_force_required_grammar(2048, 8192, false, None)
            .unwrap();

        assert!(lark(&grammar).contains("start: <|tool_call> tool_call_body"));
    }

    /// regression test for spiceai/mistral.rs#54: a required call's deadline, reached inside
    /// an unfinished DSML call, forced a new block of calls into its value; and it did so
    /// again when the template preferred no format, though a DSML block is read whatever the
    /// template prefers, so the deadline reads the unfinished call from the content alone
    #[test]
    fn required_tool_call_deadline_waits_for_an_unfinished_dsml_call() {
        let tools = vec![tool("get_weather")];
        for preferred in [Some(ToolCallFormat::DeepSeekDsml), None] {
            let mut state =
                ToolCallState::new(ToolChoice::Required, Some(&tools), preferred).unwrap();
            let calling = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Par";
            assert!(
                state
                    .maybe_force_required_grammar(0, 8192, false, Some(calling))
                    .is_none(),
                "{preferred:?}"
            );
            assert!(
                state
                    .maybe_force_required_grammar(0, 8192, true, Some(calling))
                    .is_none(),
                "{preferred:?}"
            );
            assert!(
                state
                    .maybe_force_required_grammar(
                        0,
                        8192,
                        false,
                        Some("The weather? Let me check. ")
                    )
                    .is_some(),
                "{preferred:?}"
            );
        }
    }

    /// regression test for spiceai/mistral.rs#54: a DSML example in the reasoning hid an
    /// unfinished call in the answer, and the deadline forced a second block into it. The
    /// deadline reads the response without its reasoning, as prefix detection does.
    #[test]
    fn a_dsml_example_in_reasoning_does_not_hide_an_unfinished_call() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(
            ToolChoice::Required,
            Some(&tools),
            Some(ToolCallFormat::DeepSeekDsml),
        )
        .unwrap();
        // What the model wrote, its reasoning included, is never what the deadline reads.
        state.observe_token(
            0,
            "<think>The wrapper is <｜DSML｜tool_calls> followed by invokes.</think>".as_bytes(),
        );
        let answer = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Par";
        assert!(state
            .maybe_force_required_grammar(0, 8192, false, Some(answer))
            .is_none());
    }

    /// regression test for spiceai/mistral.rs#54: a DSML call to a strict tool, refused by the
    /// tool's schema when read, was reported complete while the stream was still being read, so
    /// the stream ended with `finish_reason: "tool_calls"`, no calls, and the block gone from the
    /// content. The block is content: held back until the turn ends, then returned as written.
    #[test]
    fn a_stream_holds_back_a_dsml_block_whose_calls_are_all_refused_until_the_turn_ends() {
        let bash: Tool = serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "Bash",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" },
                        "timeout": { "type": "integer" },
                    },
                    "required": ["command"],
                },
            },
        }))
        .unwrap();
        let tools = vec![bash];
        let mut state = ToolCallState::new(
            ToolChoice::Auto,
            Some(&tools),
            Some(ToolCallFormat::DeepSeekDsml),
        )
        .unwrap();
        let refused = "Running it.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"Bash\">\n<｜DSML｜parameter name=\"timeout\" string=\"false\">30</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        // While the turn goes on: a call is still possible and none is complete, so nothing is
        // sent.
        assert_eq!(state.prefix_status(refused).unwrap(), (true, false));
        assert!(state.complete_if_tool_call(refused).unwrap().is_empty());
        let parsed = state
            .parse_streaming(Some(refused.to_string()), refused, None, false, false)
            .unwrap();
        assert!(parsed.tool_use_still_possible);
        assert!(!parsed.tool_use_is_done);
        assert!(parsed.tool_calls.is_empty());
        // When the turn ends: the message as the model wrote it, and no call ended the turn.
        let parsed = state
            .parse_streaming(Some(refused.to_string()), refused, None, false, true)
            .unwrap();
        assert_eq!(parsed.content.as_deref(), Some(refused));
        assert!(parsed.tool_calls.is_empty());
        assert!(!parsed.tool_use_is_done);
        // A call the schema allows ends the turn.
        let accepted = refused.replace(
            "name=\"timeout\" string=\"false\">30",
            "name=\"command\" string=\"true\">ls",
        );
        assert_eq!(state.prefix_status(&accepted).unwrap(), (false, true));
        let parsed = state
            .parse_streaming(Some(accepted.clone()), &accepted, None, false, false)
            .unwrap();
        assert_eq!(parsed.content.as_deref(), Some("Running it.\n\n"));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert!(parsed.tool_use_is_done);
    }

    /// regression test for spiceai/mistral.rs#54: a required call refused an unenforceable
    /// strict tool that `tool_choice` left out of the call
    #[test]
    fn a_required_dsml_call_checks_only_the_tools_it_may_call() {
        let strict_either: Tool = serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {
                "name": "pick",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "properties": { "a": { "type": "string" } },
                    "oneOf": [{ "required": ["a"] }],
                },
            },
        }))
        .unwrap();
        let tools = vec![tool("get_weather"), strict_either];
        let named = ToolChoice::NamedFunction(NamedFunctionToolChoice {
            tp: ToolType::Function,
            name: "get_weather".to_string(),
        });
        assert!(
            ToolCallState::new(named, Some(&tools), Some(ToolCallFormat::DeepSeekDsml)).is_ok()
        );
    }

    /// regression test for spiceai/mistral.rs#54: a strict schema requiring a property it
    /// does not declare was forced with no arguments
    #[test]
    fn a_required_dsml_call_refuses_required_properties_it_cannot_write() {
        for parameters in [
            serde_json::json!({ "type": "object", "required": ["command"] }),
            serde_json::json!({
                "type": "object",
                "properties": { "cwd": { "type": "string" } },
                "required": ["command"],
            }),
        ] {
            let strict: Tool = serde_json::from_value(serde_json::json!({
                "type": "function",
                "function": { "name": "Bash", "strict": true, "parameters": parameters },
            }))
            .unwrap();
            let error = ToolCallState::new(
                ToolChoice::Required,
                Some(&[strict]),
                Some(ToolCallFormat::DeepSeekDsml),
            )
            .err()
            .expect("a required call is refused");
            assert_eq!(
                error.to_string(),
                "Tool 'Bash' is strict, and its parameters schema requires `command` without declaring it in `properties`, so a required DeepSeek-V4 tool call cannot write it. Declare `command` in `properties`, or remove `strict` from the tool."
            );
        }
    }

    #[test]
    fn required_harmony_tool_call_deadline_forces_native_grammar() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(
            ToolChoice::Required,
            Some(&tools),
            Some(ToolCallFormat::Harmony),
        )
        .unwrap();

        let grammar = state
            .maybe_force_required_grammar(8192, 8192, true, None)
            .unwrap();

        assert!(lark(&grammar).contains("<|channel|>"));
        assert!(lark(&grammar).contains("commentary to=functions.get_weather "));
    }

    #[test]
    fn atem_strategy_separates_reasoning_content_and_tools() {
        let tools = vec![tool("get_weather")];
        let mut state =
            ToolCallState::new(ToolChoice::Auto, Some(&tools), Some(ToolCallFormat::Atem)).unwrap();
        let output = b" to=self<|message|>checking<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls><atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Paris</atem:parameter></atem:invoke></atem:function_calls><|eot|>";
        state.observe_token(0, output);

        assert_eq!(state.reasoning_delta().as_deref(), Some("checking"));
        let parsed = state.parse_streaming(None, "", None, false, true).unwrap();
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
        assert_eq!(
            parsed.tool_calls[0].function.arguments,
            r#"{"city":"Paris"}"#
        );
    }

    #[test]
    fn atem_streaming_buffers_a_split_multibyte_character() {
        let mut state =
            ToolCallState::new(ToolChoice::Auto, None, Some(ToolCallFormat::Atem)).unwrap();
        state.observe_token(0, b" to=user<|message|>abc");
        assert_eq!(state.content_delta().as_deref(), Some("abc"));

        state.observe_token(0, &[0xc3]);
        assert_eq!(state.content_delta(), None);

        state.observe_token(0, &[0xa9]);
        assert_eq!(state.content_delta().as_deref(), Some("\u{e9}"));
        assert_eq!(state.content().as_deref(), Some("abc\u{e9}"));
    }

    #[test]
    fn atem_streaming_replaces_invalid_utf8_without_replacing_an_incomplete_tail() {
        let mut state =
            ToolCallState::new(ToolChoice::Auto, None, Some(ToolCallFormat::Atem)).unwrap();
        state.observe_token(0, b" to=user<|message|>abc\xff\xc3");
        assert_eq!(state.content_delta().as_deref(), Some("abc\u{fffd}"));

        state.observe_token(0, &[0xa9]);
        assert_eq!(state.content_delta().as_deref(), Some("\u{e9}"));
        assert_eq!(state.content().as_deref(), Some("abc\u{fffd}\u{e9}"));
    }

    #[test]
    fn atem_waits_for_turn_end_and_returns_parallel_calls() {
        let tools = vec![tool("get_weather"), tool("search")];
        let mut state =
            ToolCallState::new(ToolChoice::Auto, Some(&tools), Some(ToolCallFormat::Atem)).unwrap();
        let first = " to=get_weather<|message|><atem:function_calls><atem:invoke name=\"get_weather\"></atem:invoke></atem:function_calls>";
        state.observe_token(0, first.as_bytes());

        assert_eq!(state.prefix_status(first).unwrap(), (false, true));
        assert!(!state.stops_after_complete_tool_call());

        state.observe_token(
            0,
            b"<|eom|><|start|>assistant to=search<|message|><atem:function_calls><atem:invoke name=\"search\"></atem:invoke></atem:function_calls><|eot|>",
        );
        let parsed = state.finalize_for_response("", None, None, None).unwrap();

        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
        assert_eq!(parsed.tool_calls[1].function.name, "search");
    }

    #[test]
    fn atem_continuation_grammar_activates_once_per_wrapper() {
        let tools = vec![tool("get_weather"), tool("search")];
        let mut state =
            ToolCallState::new(ToolChoice::Auto, Some(&tools), Some(ToolCallFormat::Atem)).unwrap();
        state.observe_token(0, b" to=get_weather<|message|><atem:function_calls>");
        assert!(state.maybe_activate_continuation_grammar(None).is_some());
        state.mark_grammar_active(false);
        state.observe_token(
            0,
            b"<atem:invoke name=\"get_weather\"></atem:invoke></atem:function_calls>",
        );
        assert!(state.clear_active_grammar());
        assert!(state.maybe_activate_continuation_grammar(None).is_none());

        state.observe_token(
            0,
            b"<|eom|><|start|>assistant to=search<|message|><atem:function_calls>",
        );
        assert!(state.maybe_activate_continuation_grammar(None).is_some());
    }

    #[test]
    fn atem_named_choice_uses_native_required_grammar() {
        let tools = vec![tool("get_weather"), tool("search")];
        let choice = ToolChoice::NamedFunction(NamedFunctionToolChoice {
            tp: ToolType::Function,
            name: "search".to_string(),
        });
        let mut state =
            ToolCallState::new(choice, Some(&tools), Some(ToolCallFormat::Atem)).unwrap();

        let grammar = state
            .maybe_force_required_grammar(8192, 8192, true, None)
            .unwrap();
        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();
        assert!(lark.contains("to=search"));
        assert!(!lark.contains("to=get_weather"));
    }

    #[test]
    fn completed_atem_grammar_satisfies_required_choice_before_eos() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(
            ToolChoice::Required,
            Some(&tools),
            Some(ToolCallFormat::Atem),
        )
        .unwrap();
        state.mark_grammar_active(true);
        state.observe_token(
            0,
            b" to=get_weather<|message|><atem:function_calls><atem:invoke name=\"get_weather\"></atem:invoke></atem:function_calls",
        );

        assert!(state.clear_active_grammar());
        assert!(!state.required_tool_call_unsatisfied());

        state.observe_token(0, b">");
        assert_eq!(
            state
                .finalize_for_response("", None, None, None)
                .unwrap()
                .tool_calls
                .len(),
            1
        );
    }

    #[test]
    fn forced_atem_call_closes_an_open_reasoning_message() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(
            ToolChoice::Required,
            Some(&tools),
            Some(ToolCallFormat::Atem),
        )
        .unwrap();
        state.observe_token(0, b" to=self<|message|>still thinking");

        let grammar = state
            .maybe_force_required_grammar(0, 8192, true, None)
            .unwrap();
        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();
        assert!(lark.contains(r#""<|eom|>" "<|start|>" "assistant to=get_weather""#));
    }

    #[test]
    fn forced_atem_call_reuses_an_emitted_assistant_boundary() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(
            ToolChoice::Required,
            Some(&tools),
            Some(ToolCallFormat::Atem),
        )
        .unwrap();
        state.observe_token(0, b" to=self<|message|>done<|eom|><|start|>assistant");

        let grammar = state
            .maybe_force_required_grammar(0, 8192, true, None)
            .unwrap();
        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();
        assert!(lark.contains(r#"" to=get_weather""#));
        assert!(!lark.contains(r#""<|start|>""#));
    }

    #[test]
    fn stop_token_under_required_tool_call_is_blocked() {
        let tools = vec![tool("get_weather")];
        let state = ToolCallState::new(ToolChoice::Required, Some(&tools), None).unwrap();

        assert!(state.is_stop_token_blocked(2, Some(&[2]), &[]));
        assert!(state.is_stop_token_blocked(9, None, &[9]));
    }

    #[test]
    fn grammar_completion_deactivates_tool_grammar_state() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(ToolChoice::Auto, Some(&tools), None).unwrap();

        state.mark_grammar_active(false);

        assert!(state.clear_active_grammar());
        assert!(!state.clear_active_grammar());
    }

    #[test]
    fn completed_tool_call_satisfies_required_obligation() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(ToolChoice::Required, Some(&tools), None).unwrap();

        let calls = state
            .complete_if_tool_call(r#"{"name":"get_weather","parameters":{"city":"Paris"}}"#)
            .unwrap();

        assert_eq!(calls.len(), 1);
        assert!(!state.required_tool_call_unsatisfied());
    }

    #[test]
    fn named_tool_choice_narrows_allowed_tool_name_and_schema() {
        let tools = vec![tool("get_weather"), tool("search")];
        let choice = ToolChoice::NamedFunction(NamedFunctionToolChoice {
            tp: ToolType::Function,
            name: "search".to_string(),
        });
        let mut state = ToolCallState::new(choice, Some(&tools), None).unwrap();

        let grammar = state
            .maybe_force_required_grammar(8192, 8192, true, None)
            .unwrap();
        let schema = grammar.grammars[1].json_schema.as_ref().unwrap();
        let names = schema["properties"]["name"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["search"]);
    }

    #[test]
    fn freeform_text_before_forced_text_tool_call_is_preserved() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(ToolChoice::Required, Some(&tools), None).unwrap();

        let parsed = state
            .finalize_for_response(
                r#"Before <tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#,
                None,
                None,
                None,
            )
            .unwrap();

        assert_eq!(parsed.content, Some("Before ".to_string()));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(
            parsed.tool_calls[0].function.arguments,
            json!({"city":"Paris"}).to_string()
        );
    }

    #[test]
    fn text_tool_strategy_parses_a_hidden_stop_delimiter_without_exposing_it() {
        let tools = vec![tool("get_weather")];
        let visible = r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}"#;
        let parser_text = format!("{visible}</tool_call>");

        let mut final_state =
            ToolCallState::new(ToolChoice::Auto, Some(&tools), Some(ToolCallFormat::Qwen)).unwrap();
        let parsed = final_state
            .finalize_for_response(visible, None, None, Some(&parser_text))
            .unwrap();
        assert_eq!(parsed.content, None);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "get_weather");

        let mut streaming_state =
            ToolCallState::new(ToolChoice::Auto, Some(&tools), Some(ToolCallFormat::Qwen)).unwrap();
        let parsed = streaming_state
            .parse_streaming(
                Some(visible.to_string()),
                visible,
                Some(&parser_text),
                false,
                true,
            )
            .unwrap();
        assert_eq!(parsed.content, None);
        assert_eq!(parsed.tool_calls.len(), 1);
    }

    #[test]
    fn hidden_parser_text_is_not_returned_when_no_tool_call_matches() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(ToolChoice::Auto, Some(&tools), None).unwrap();

        let parsed = state
            .finalize_for_response(
                "ordinary response",
                None,
                None,
                Some("ordinary response<STOP_BOUNDARY>"),
            )
            .unwrap();

        assert_eq!(parsed.content.as_deref(), Some("ordinary response"));
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn hunyuan_partial_prefix_activates_before_merged_array_token() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(
            ToolChoice::Auto,
            Some(&tools),
            Some(ToolCallFormat::Hunyuan),
        )
        .unwrap();

        let grammar = state
            .maybe_activate_continuation_grammar(Some("<tool_calls"))
            .expect("Hunyuan grammar must activate before the merged `>[` token");

        assert!(lark(&grammar).contains(r#"start: ">" @json_body "</tool_calls>""#));
    }
}
