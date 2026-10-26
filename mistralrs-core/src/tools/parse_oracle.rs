//! A differential corpus for the tool-call parsers: messages that splice one format's call
//! markers into another format's calls, and their prefixes, each read by the matcher here and
//! compared with what the base read from the same text; and each message streamed through
//! `parse_streaming` one character at a time and in runs drawn from `STREAM_SEED`, compared
//! with what the base's streaming of the same chunks sent and called. The base is `spiceai` at
//! fcd94520dd6ed6ee7c1800b7c60351c4772ffccf, before DeepSeek-V4's DSML calls were read; its
//! readings are recorded in `testdata/tool_call_parse_oracle.json`, which pairs every entry of
//! `corpus()` with what `outcome()` returned on the base for each of its cuts, and what
//! `streaming_outcome()` returned for each of its chunkings. To record it
//! again, check out the base, add this module there, and write `oracle_json(&corpus())` to that
//! file.
//!
//! A reading here may differ from the base's only where it reaches into a DSML block outside
//! any other format's call, whose calls `dsml_calls` states, or as `intended_difference` lists:
//! everything else, a call the base did not make above all, is a regression.

use serde_json::{json, Value};

use super::state::ToolCallState;
use super::{ToolCallingMatcher, ToolChoice};
use crate::Tool;

/// The seed the generated entries are drawn from.
pub(super) const SEED: u64 = 0x5eed_0054_d5e1_0001;
/// How many entries are generated.
const GENERATED: usize = 300;
/// The seed the streaming arm's random chunkings are drawn from.
pub(super) const STREAM_SEED: u64 = 0x5eed_0054_57e4_0001;
/// The seed the mutated entries are drawn from.
pub(super) const MUTATION_SEED: u64 = 0x5eed_0054_6d75_0001;

/// One message of the corpus, the prefixes of it read (`cuts`, byte lengths, the whole message
/// last), and the DSML calls it makes where it writes a DSML block outside any other format's
/// call.
pub(super) struct Entry {
    pub(super) id: String,
    pub(super) message: String,
    pub(super) cuts: Vec<usize>,
    pub(super) dsml: Option<TopLevelDsml>,
}

/// A DSML block outside any other format's call: from `at`, the prefixes read as DSML, and each
/// block that has closed by a prefix's end makes its calls.
pub(super) struct TopLevelDsml {
    pub(super) reason: &'static str,
    pub(super) at: usize,
    pub(super) blocks: Vec<(usize, Vec<(String, String)>)>,
}

/// The tools every entry is read with.
fn tools() -> Vec<Tool> {
    [
        "write_file",
        "delete_all",
        "Bash",
        "note",
        "get_weather",
        "get_time",
    ]
    .into_iter()
    .map(|name| {
        serde_json::from_value(json!({
            "type": "function",
            "function": { "name": name, "parameters": { "type": "object" } },
        }))
        .expect("a tool")
    })
    .collect()
}

/// A matcher for the tools every entry is read with.
pub(super) fn matcher() -> ToolCallingMatcher {
    ToolCallingMatcher::new(ToolChoice::Auto, Some(&tools())).expect("a matcher")
}

/// What the matcher reads from `text`, as `[streaming, read]`: the streaming state as `"tf"`
/// for `(true, false)` and so on, and what is read as `[content, calls]`, each call as its name
/// and its arguments' JSON, with content written exactly as `text` as `true`. An error is
/// `{"error": message}` in place of either.
pub(super) fn outcome(matcher: &ToolCallingMatcher, text: &str) -> Value {
    let flag = |value: bool| if value { 't' } else { 'f' };
    let streaming = match matcher.prefix_could_be_tool(text) {
        Ok((could_be, complete)) => json!(format!("{}{}", flag(could_be), flag(complete))),
        Err(error) => json!({ "error": error.to_string() }),
    };
    let read = match matcher.get_call_with_content(text) {
        Ok((content, calls)) => {
            let content = match content {
                Some(content) if content == text => Value::Bool(true),
                Some(content) => Value::String(content),
                None => Value::Null,
            };
            let calls: Vec<Value> = calls
                .into_iter()
                .map(|call| json!([call.function.name, call.function.arguments]))
                .collect();
            json!([content, calls])
        }
        Err(error) => json!({ "error": error.to_string() }),
    };
    json!([streaming, read])
}

/// The chunkings `message` is streamed in, as the byte offsets the chunks end at: one character
/// at a time, and runs of one to eight characters drawn from `rng`.
pub(super) fn chunkings(message: &str, rng: &mut Rng) -> [Vec<usize>; 2] {
    let ends: Vec<usize> = message
        .char_indices()
        .map(|(at, c)| at + c.len_utf8())
        .collect();
    let mut runs = Vec::new();
    let mut at = 0;
    while at < ends.len() {
        at = (at + 1 + rng.below(8)).min(ends.len());
        runs.push(ends[at - 1]);
    }
    [ends, runs]
}

/// What streaming `message` in chunks ending at `ends` reads, as a chat response with no
/// reasoning parser streams: after each chunk the text not yet sent is checked, and is sent
/// through `parse_streaming` once it cannot be or is a call, or the message has ended; a call
/// ends the response. As `[content, calls]`: the content sent, joined, `true` when it is the
/// whole message, and each call as its name and its arguments' JSON; an error is
/// `{"error": message}`.
pub(super) fn streaming_outcome(message: &str, ends: &[usize]) -> Value {
    let tools = tools();
    let mut state = match ToolCallState::new(ToolChoice::Auto, Some(&tools), None) {
        Ok(state) => state,
        Err(error) => return json!({ "error": error.to_string() }),
    };
    let mut sent = 0;
    let mut content = String::new();
    let mut calls = Vec::new();
    for (index, &end) in ends.iter().enumerate() {
        let last = index + 1 == ends.len();
        let pending = &message[sent..end];
        let (possible, done) = match state.prefix_status(pending) {
            Ok(status) => status,
            Err(error) => return json!({ "error": error.to_string() }),
        };
        if possible && !done && !last {
            continue;
        }
        let parsed =
            match state.parse_streaming(Some(pending.to_string()), pending, None, false, last) {
                Ok(parsed) => parsed,
                Err(error) => return json!({ "error": error.to_string() }),
            };
        sent = end;
        content.push_str(parsed.content.as_deref().unwrap_or_default());
        let made = !parsed.tool_calls.is_empty();
        calls.extend(
            parsed
                .tool_calls
                .into_iter()
                .map(|call| json!([call.function.name, call.function.arguments])),
        );
        if parsed.tool_use_is_done || made {
            break;
        }
    }
    let content = if content == message {
        Value::Bool(true)
    } else {
        Value::String(content)
    };
    json!([content, Value::Array(calls)])
}

/// The stream the streaming arm's chunkings are drawn from, in corpus order.
pub(super) fn stream_rng() -> Rng {
    Rng(STREAM_SEED)
}

/// The corpus with what the matcher reads from each cut, and what streaming each message in
/// each of its chunkings reads, as the oracle file records them.
pub(super) fn oracle_json(entries: &[Entry]) -> Value {
    let matcher = matcher();
    let mut rng = stream_rng();
    let entries: Vec<Value> = entries
        .iter()
        .map(|entry| {
            let outcomes: Vec<Value> = entry
                .cuts
                .iter()
                .map(|&cut| outcome(&matcher, &entry.message[..cut]))
                .collect();
            let chunkings = chunkings(&entry.message, &mut rng);
            let streaming: Vec<Value> = chunkings
                .iter()
                .map(|ends| streaming_outcome(&entry.message, ends))
                .collect();
            json!({
                "id": entry.id,
                "message": entry.message,
                "cuts": entry.cuts,
                "outcomes": outcomes,
                "chunks": chunkings.iter().map(Vec::len).collect::<Vec<_>>(),
                "streaming": streaming,
            })
        })
        .collect();
    json!({
        "seed": format!("{SEED:#x}"),
        "stream_seed": format!("{STREAM_SEED:#x}"),
        "mutation_seed": format!("{MUTATION_SEED:#x}"),
        "entries": entries,
    })
}

/// Why streaming `entry` may read `head` where the base's streaming read `base`, if it may.
/// `read` is what the base read from each prefix of the message, the whole message included,
/// that ends before the entry's DSML block, if it writes one outside any other format's call. A
/// message with such a block may make the block's calls; any message may make the calls the
/// base's streaming made, with the content before them cut elsewhere, or only calls one of
/// `read` makes, or fail as one of `read` does; and a message that makes no call is its content
/// as written.
pub(super) fn intended_streaming_difference(
    entry: &Entry,
    base: &Value,
    head: &Value,
    read: &[&Value],
) -> Option<&'static str> {
    let calls = |reading: &Value| -> Option<Vec<Value>> {
        reading.get(1).and_then(Value::as_array).cloned()
    };
    let within = |made: &[Value], read: Option<Vec<Value>>| {
        let mut read = read.unwrap_or_default();
        made.iter().all(|call| {
            read.iter()
                .position(|candidate| candidate == call)
                .map(|at| read.remove(at))
                .is_some()
        })
    };
    if let Some(dsml) = &entry.dsml {
        let blocks: Vec<Value> = dsml
            .blocks
            .iter()
            .flat_map(|(_, calls)| {
                calls
                    .iter()
                    .map(|(name, arguments)| json!([name, arguments]))
            })
            .collect();
        if calls(head).as_deref() == Some(blocks.as_slice()) {
            return Some(dsml.reason);
        }
    }
    let no_call = json!([]);
    if base.get(1) == Some(&no_call) && *head == json!([true, no_call]) {
        return Some(AS_WRITTEN);
    }
    if base.get(1).is_some() && base.get(1) == head.get(1) {
        return Some(SAME_STREAMED_CALLS);
    }
    let as_read = match head.get("error") {
        Some(error) => read
            .iter()
            .any(|reading| reading[1].get("error") == Some(error)),
        None => {
            let made = calls(head)?;
            read.iter().any(|reading| within(&made, calls(&reading[1])))
        }
    };
    as_read.then_some(STREAMED_AS_READ)
}

/// The differences from the base this crate intends, one per line with its reason. A DSML
/// block outside any other format's call (`TopLevelDsml`) is read from its opener on by
/// `DSML_CALLS` or `DSML_OWNS`; and:
const AS_WRITTEN: &str = "a message that makes no call is its content as the model wrote it, where the base returned the text a parser had read or repaired from it";
const PARTIAL_OPENER: &str = "a message ending partway through a marker that begins a call, outside any other call, is held back and makes no call yet, as it may be opening one, a DSML block that would claim the message included, where the base streamed it or read a call before it";
const HELD_IN_CALL: &str = "a message inside a call that has not ended, such as a bare-JSON call that does not read, is held back and makes no call yet, where the base streamed it, so the rest of the call is never read on its own";
const COMPLETE_OUTSIDE_QWEN: &str = "a finished call in a format other than Qwen is complete, and makes the same calls, where the base held it back as if more Qwen calls could follow, though it reads only the first format's calls";
const SAME_STREAMED_CALLS: &str = "streamed, a message makes the calls the base's streaming made, with the content sent before them cut elsewhere, as a marker that may begin a call is held back";
const STREAMED_AS_READ: &str = "streamed in chunks that cut a call's marker, a message makes only calls the base reads from it or from a prefix of it before any DSML block, or fails as one does, where the base streamed the cut marker out as text and missed them";

/// The calls a cut of `entry` makes as DSML, with the reason, when it reaches into a DSML
/// block outside any other format's call: from the block's opener on, the prefix makes
/// exactly the calls of the blocks closed by its end, whatever the base read.
pub(super) fn dsml_calls(entry: &Entry, cut: usize) -> Option<(&'static str, Value)> {
    let dsml = entry.dsml.as_ref().filter(|dsml| cut > dsml.at)?;
    let calls: Vec<Value> = dsml
        .blocks
        .iter()
        .filter(|(closes, _)| *closes <= cut)
        .flat_map(|(_, calls)| {
            calls
                .iter()
                .map(|(name, arguments)| json!([name, arguments]))
        })
        .collect();
    Some((dsml.reason, Value::Array(calls)))
}

/// Why the reading `head` of the cut `cut` of `entry` may differ from the base's, `base`, outside
/// a DSML block (`dsml_calls`), if it may: a prefix ending partway through a call marker is
/// held back with no call; a prefix inside a call that has not ended, which the base streamed,
/// is held back with no call; a finished call in another format than Qwen is complete where
/// the base waited for more Qwen calls; and a prefix that makes no call, read with the same
/// streaming state, is content as written.
pub(super) fn intended_difference(
    entry: &Entry,
    cut: usize,
    base: &Value,
    head: &Value,
) -> Option<&'static str> {
    let no_call = json!([]);
    let text = &entry.message[..cut];
    let held = head[0] == json!("tf") && head[1] == json!([true, no_call]);
    let partial = MARKERS.iter().any(|marker| {
        marker
            .char_indices()
            .skip(1)
            .any(|(at, _)| text.ends_with(&marker[..at]))
    });
    if held && partial {
        return Some(PARTIAL_OPENER);
    }
    if held && base[0] == json!("ff") && base[1][1] == no_call {
        return Some(HELD_IN_CALL);
    }
    if head[0] == json!("ft")
        && base[0] == json!("tf")
        && head[1] == base[1]
        && base[1][1] != no_call
    {
        return Some(COMPLETE_OUTSIDE_QWEN);
    }
    (head[0] == base[0] && base[1][1] == no_call && head[1] == json!([true, no_call]))
        .then_some(AS_WRITTEN)
}

/// The markers that begin a call, in every format.
const MARKERS: [&str; 9] = [
    "<｜DSML｜",
    "<atem:function_calls>",
    "<|tool_call>",
    "<|tool_call_start|>",
    "<|python_tag|>",
    "<tool_call>",
    "<tool_calls>",
    "[TOOL_CALLS]",
    "<｜tool▁call▁begin｜>",
];

const DELETE_ALL: &str = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"delete_all\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
const BASH_EXAMPLE: &str = "Run <tool_call>{\"name\": \"Bash\", \"arguments\": {}}</tool_call>";
const OPENER: &str = "<｜DSML｜tool_calls>";
const BLOCK_CLOSER: &str = "</｜DSML｜tool_calls>";

/// A DSML block calling `write_file` whose `content` holds a Qwen call to `Bash`.
fn write_file_block() -> String {
    format!(
        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"content\" string=\"true\">{BASH_EXAMPLE}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    )
}

fn call(name: &str, arguments: &Value) -> (String, String) {
    (
        name.to_string(),
        serde_json::to_string(arguments).expect("JSON"),
    )
}

fn write_file_call() -> (String, String) {
    call("write_file", &json!({ "content": BASH_EXAMPLE }))
}

fn delete_all_call() -> (String, String) {
    call("delete_all", &json!({}))
}

const DSML_CALLS: &str = "a DSML block outside any other format's call is a call: DSML is the format this PR reads, and the base reads it as text";
const DSML_OWNS: &str = "a message with a DSML block outside any other format's call is read only as DSML, so a call written in a DSML value, or after a block that cannot be read, is text";

/// The prefixes of `message` read for a hand-written entry: the end of every tag, quote, line,
/// marker and bracket, the start of every tag, and the whole message.
fn hand_cuts(message: &str) -> Vec<usize> {
    let mut cuts: Vec<usize> = message
        .char_indices()
        .flat_map(|(at, c)| {
            let after = at + c.len_utf8();
            let mut cuts = Vec::new();
            if matches!(c, '>' | '"' | '\n' | '｜' | '{' | '}' | '[' | ']' | '\'') {
                cuts.push(after);
            }
            if c == '<' && at > 0 {
                cuts.push(at);
            }
            cuts
        })
        .collect();
    cuts.push(message.len());
    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

fn hand(id: &str, message: impl Into<String>) -> Entry {
    let message = message.into();
    Entry {
        id: format!("hand:{id}"),
        cuts: hand_cuts(&message),
        message,
        dsml: None,
    }
}

/// A hand-written entry whose last DSML block opener begins a block outside any other
/// format's call, closing at the end of the message with `calls`.
fn hand_dsml(
    id: &str,
    message: impl Into<String>,
    reason: &'static str,
    calls: Vec<(String, String)>,
) -> Entry {
    let mut entry = hand(id, message);
    let at = entry.message.rfind(OPENER).expect("a block");
    let blocks = if calls.is_empty() {
        Vec::new()
    } else {
        let closes = entry.message.rfind(BLOCK_CLOSER).expect("a closer") + BLOCK_CLOSER.len();
        vec![(closes, calls)]
    };
    entry.dsml = Some(TopLevelDsml { reason, at, blocks });
    entry
}

fn hand_corpus() -> Vec<Entry> {
    let block = DELETE_ALL;
    let write = write_file_block();
    let mut entries = vec![
        // Each format alone.
        hand(
            "qwen-json",
            r#"<tool_call>{"name":"write_file","arguments":{"content":"hello"}}</tool_call>"#,
        ),
        hand(
            "qwen-json-array",
            r#"<tool_call>[{"name":"write_file","arguments":{"content":"hello"}}]</tool_call>"#,
        ),
        hand(
            "qwen-xml",
            "<tool_call>\n<function=write_file>\n<parameter=content>\nhello\n</parameter>\n</function>\n</tool_call>",
        ),
        hand(
            "qwen-xml-leading-text",
            "<tool_call>Example:\n<function=write_file>\n<parameter=content>\nhello\n</parameter>\n</function>\n</tool_call>",
        ),
        hand(
            "qwen-xml-parallel",
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_time>\n<parameter=zone>\nUTC\n</parameter>\n</function>\n</tool_call>",
        ),
        hand(
            "hunyuan",
            r#"<tool_calls>[{"name":"write_file","arguments":{"content":"hello"}}]</tool_calls>"#,
        ),
        hand(
            "deepseek",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>write_file\n```json\n{\"content\":\"hello\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        ),
        hand(
            "mistral-nemo",
            r#"[TOOL_CALLS][{"name":"write_file","arguments":{"content":"hello"}}]"#,
        ),
        hand(
            "llama",
            r#"<|python_tag|>{"name":"write_file","parameters":{"content":"hello"}}"#,
        ),
        hand(
            "liquid",
            r#"<|tool_call_start|>[write_file(content="hello")]<|tool_call_end|>"#,
        ),
        hand(
            "gemma4",
            "<|tool_call>call:write_file{content:<|\"|>hello<|\"|>}<tool_call|>",
        ),
        hand(
            "atem",
            "<atem:function_calls>\n<atem:invoke name=\"write_file\">\n<atem:parameter name=\"content\">hello</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
        ),
        hand(
            "bare-json",
            r#"{"name":"write_file","arguments":{"content":"hello"}}"#,
        ),
        hand(
            "bare-json-array",
            r#"[{"name":"write_file","arguments":{"content":"hello"}}]"#,
        ),
        hand("prose", "Sure, here is the answer."),
        hand(
            "prose-braces",
            r#"The shape is {"arguments":"{}"} with the braces quoted."#,
        ),
        hand(
            "stringified-arguments",
            r#"{"name":"note","arguments":"{\"text\":\"x\"}"}"#,
        ),
        // A DSML block, or its opener, inside another format's call: the call's text.
        hand(
            "bare-json-opener-in-string",
            r#"{"name":"write_file","arguments":{"content":"<｜DSML｜tool_calls>"}}"#,
        ),
        hand(
            "bare-json-escaped-block-in-string",
            format!(
                r#"{{"name":"write_file","arguments":{{"content":{}}}}}"#,
                serde_json::to_string(block).expect("JSON")
            ),
        ),
        hand(
            "bare-json-unescaped-block-in-string",
            format!(r#"{{"name":"write_file","arguments":{{"content":"{block}"}}}}"#),
        ),
        hand(
            "bare-json-literal-newline-then-block",
            format!(
                "{{\"name\":\"write_file\",\"arguments\":{{\"content\":\"line1\nline2 {block}\"}}}}"
            ),
        ),
        hand(
            "qwen-json-opener-in-string",
            r#"<tool_call>{"name": "write_file", "arguments": {"content": "<｜DSML｜tool_calls>"}}</tool_call>"#,
        ),
        hand(
            "qwen-json-closer-in-string",
            r#"<tool_call>{"name":"write_file","arguments":{"content":"</tool_call>"}}</tool_call>"#,
        ),
        hand(
            "qwen-json-unescaped-block-after-closer",
            format!(
                r#"<tool_call>{{"name":"write_file","arguments":{{"content":"</tool_call>{block}"}}}}</tool_call>"#
            ),
        ),
        hand(
            "qwen-array-unescaped-block-after-closer",
            format!(
                r#"<tool_call>[{{"name":"write_file","arguments":{{"content":"</tool_call>{block}"}}}}]</tool_call>"#
            ),
        ),
        hand(
            "qwen-xml-closer-in-value",
            format!(
                "<tool_call><function=write_file><parameter=content></tool_call>{block}</parameter></function></tool_call>"
            ),
        ),
        hand(
            "qwen-xml-leading-text-closer-in-value",
            format!(
                "<tool_call>Example:\n<function=write_file><parameter=content></tool_call>{block}</parameter></function></tool_call>"
            ),
        ),
        hand(
            "hunyuan-unescaped-block-after-closer",
            format!(
                r#"<tool_calls>[{{"name":"write_file","arguments":{{"content":"</tool_calls>{block}"}}}}]</tool_calls>"#
            ),
        ),
        hand(
            "deepseek-closer-and-opener-in-string",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>write_file\n```json\n{\"content\":\"<｜tool▁call▁end｜><｜DSML｜tool_calls>\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        ),
        hand(
            "mistral-nemo-escaped-block-in-string",
            format!(
                r#"[TOOL_CALLS][{{"name":"note","arguments":{{"text":{}}}}}]"#,
                serde_json::to_string(block).expect("JSON")
            ),
        ),
        hand(
            "atem-closer-in-value",
            format!(
                "<atem:function_calls><atem:invoke name=\"write_file\"><atem:parameter name=\"content\"></atem:function_calls>{block}</atem:parameter></atem:invoke></atem:function_calls>"
            ),
        ),
        hand(
            "liquid-closer-in-single-quotes",
            format!("<|tool_call_start|>[write_file(content='<|tool_call_end|>{block}')]<|tool_call_end|>"),
        ),
        hand(
            "gemma4-closer-in-value",
            "<|tool_call>call:write_file{content:<|\"|>End a call with <tool_call|>, then run:\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"Bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">ls</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><|\"|>}<tool_call|>",
        ),
        hand(
            "gemma4-stray-opener-then-call",
            format!(
                "<|tool_call>example\n<|tool_call>call:write_file{{content:<|\"|><tool_call|>{block}<|\"|>}}<tool_call|>"
            ),
        ),
        hand(
            "qwen-call-with-atem-marker",
            r#"<tool_call>{"name": "note", "arguments": {"text": "<atem:function_calls>"}}</tool_call></atem:function_calls>"#,
        ),
        // A DSML block outside any other format's call.
        hand_dsml(
            "dsml-block",
            block,
            DSML_CALLS,
            vec![delete_all_call()],
        ),
        hand_dsml(
            "dsml-block-with-qwen-example",
            write.clone(),
            DSML_OWNS,
            vec![write_file_call()],
        ),
        hand_dsml(
            "closed-example-then-block",
            format!("An empty example: <tool_call></tool_call>\n{write}"),
            DSML_OWNS,
            vec![write_file_call()],
        ),
        hand_dsml(
            "qwen-call-with-opener-then-block",
            format!(
                "<tool_call>{{\"name\": \"note\", \"arguments\": {{\"text\": \"<｜DSML｜tool_calls>\"}}}}</tool_call>\n{write}"
            ),
            DSML_OWNS,
            vec![write_file_call()],
        ),
        hand_dsml(
            "qwen-call-with-marker-then-block",
            format!(
                "<tool_call>{{\"name\": \"note\", \"arguments\": {{\"text\": \"[TOOL_CALLS]\"}}}}</tool_call>\n{write}"
            ),
            DSML_OWNS,
            vec![write_file_call()],
        ),
        hand_dsml(
            "qwen-call-with-atem-marker-then-block",
            format!(
                "<tool_call>{{\"name\": \"note\", \"arguments\": {{\"text\": \"<atem:function_calls>\"}}}}</tool_call></atem:function_calls>\n{write}"
            ),
            DSML_OWNS,
            vec![write_file_call()],
        ),
        hand_dsml(
            "qwen-json-closer-in-string-then-block",
            format!(
                r#"<tool_call>{{"name":"write_file","arguments":{{"content":"</tool_call>"}}}}</tool_call>
{block}"#
            ),
            DSML_CALLS,
            vec![delete_all_call()],
        ),
        hand_dsml(
            "qwen-xml-then-block",
            format!(
                "<tool_call>\n<function=write_file>\n<parameter=content>\nhello\n</parameter>\n</function>\n</tool_call>\n{block}"
            ),
            DSML_CALLS,
            vec![delete_all_call()],
        ),
        hand_dsml(
            "mistral-nemo-quoting-hunyuan-then-block",
            format!(r#"[TOOL_CALLS][{{"name":"note","arguments":{{"text":"<tool_calls>"}}}}]
{write}"#),
            DSML_OWNS,
            vec![write_file_call()],
        ),
        hand_dsml(
            "prose-mistral-marker-then-block",
            format!("Write [TOOL_CALLS] before a Mistral call.\n{write}"),
            DSML_OWNS,
            vec![write_file_call()],
        ),
        hand_dsml(
            "prose-llama-marker-then-block",
            format!("Write <|python_tag|> before a Llama call.\n{write}"),
            DSML_OWNS,
            vec![write_file_call()],
        ),
        hand_dsml(
            "example-then-block",
            format!(
                "Like this: <tool_call>{{\"name\": \"note\", \"arguments\": {{}}}}</tool_call>\nNow for real:\n{write}"
            ),
            DSML_OWNS,
            vec![write_file_call()],
        ),
        hand_dsml(
            "empty-block-then-qwen-call",
            "Sure.\n<｜DSML｜tool_calls></｜DSML｜tool_calls> <tool_call>{\"name\": \"Bash\", \"arguments\": {}}</tool_call>",
            DSML_OWNS,
            Vec::new(),
        ),
        hand_dsml(
            "unreadable-block-then-qwen-call",
            "Sure.\n<｜DSML｜tool_calls>! <tool_call>{\"name\": \"Bash\", \"arguments\": {}}</tool_call>",
            DSML_OWNS,
            Vec::new(),
        ),
    ];
    // Another format's opener quoted in a finished bare-JSON call, then a DSML block.
    for (index, marker) in [
        "<tool_calls>",
        "<|tool_call>",
        "<tool_call>",
        "<|tool_call_start|>",
        "<atem:function_calls>",
        "<｜tool▁call▁begin｜>",
        "[TOOL_CALLS]",
        "<|python_tag|>",
    ]
    .into_iter()
    .enumerate()
    {
        let quoted = serde_json::to_string(marker).expect("JSON");
        for (shape, call) in [
            (
                "object",
                format!(r#"{{"name":"note","arguments":{{"text":{quoted}}}}}"#),
            ),
            (
                "array",
                format!(r#"[{{"name":"note","arguments":{{"text":{quoted}}}}}]"#),
            ),
        ] {
            entries.push(hand_dsml(
                &format!("bare-json-{shape}-quoting-marker-{index}-then-block"),
                format!("{call}\n{write}"),
                DSML_OWNS,
                vec![write_file_call()],
            ));
        }
    }
    entries
}

/// A xorshift64* stream, so the generated entries are the same on every machine.
pub(super) struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(n).expect("n")).expect("an index")
    }
}

/// Text spliced into another format's values: every format's markers, a DSML block, quotes,
/// backslashes, and control characters.
const FRAGMENTS: &[&str] = &[
    OPENER,
    BLOCK_CLOSER,
    DELETE_ALL,
    "<tool_call>",
    "</tool_call>",
    "<tool_calls>",
    "</tool_calls>",
    "[TOOL_CALLS]",
    "<|python_tag|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|\"|>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "<｜tool▁call▁begin｜>",
    "<｜tool▁call▁end｜>",
    "<atem:function_calls>",
    "</atem:function_calls>",
    "<parameter=x>",
    "</parameter>",
    "<atem:parameter name=\"x\">",
    "</atem:parameter>",
    "<function=Bash>",
    "<tool_call>{\"name\":\"Bash\",\"arguments\":{}}</tool_call>",
    "\"",
    "'",
    "\\",
    "\n",
    "\t",
    "\r",
    "{",
    "}",
    "[",
    "]",
    "text ",
    "```",
];

/// The markers that end a call, in every format whose calls close.
const CLOSERS: [&str; 7] = [
    "</tool_call>",
    "</tool_calls>",
    "<tool_call|>",
    "<|tool_call_end|>",
    "<｜tool▁call▁end｜>",
    "</atem:function_calls>",
    BLOCK_CLOSER,
];

/// Where the JSON value of a generated call goes, and the call's text after it.
const JSON_CALLS: &[(&str, &str, &str)] = &[
    (
        "qwen-json",
        r#"<tool_call>{"name":"write_file","arguments":{"content":"#,
        "}}</tool_call>",
    ),
    (
        "qwen-array",
        r#"<tool_call>[{"name":"write_file","arguments":{"content":"#,
        "}}]</tool_call>",
    ),
    (
        "hunyuan",
        r#"<tool_calls>[{"name":"write_file","arguments":{"content":"#,
        "}}]</tool_calls>",
    ),
    (
        "deepseek",
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>write_file\n```json\n{\"content\":",
        "}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    ),
    (
        "mistral-nemo",
        r#"[TOOL_CALLS][{"name":"write_file","arguments":{"content":"#,
        "}}]",
    ),
    (
        "llama",
        r#"<|python_tag|>{"name":"write_file","parameters":{"content":"#,
        "}}",
    ),
    (
        "bare-json",
        r#"{"name":"write_file","arguments":{"content":"#,
        "}}",
    ),
    (
        "bare-json-array",
        r#"[{"name":"write_file","arguments":{"content":"#,
        "}}]",
    ),
];

/// One generated entry: a call in one format whose value splices in other formats' markers.
/// A value written as its format writes one, escaped, keeps its markers as its text; one
/// written unescaped into a JSON string is never JSON, so the call is never finished. A
/// finished call may be followed by a DSML block, outside any call.
fn generated(rng: &mut Rng, case: usize) -> Entry {
    let mut payload = String::new();
    for _ in 0..=rng.below(5) {
        payload.push_str(FRAGMENTS[rng.below(FRAGMENTS.len())]);
    }
    let kind = rng.below(JSON_CALLS.len() + 5);
    let escaped = kind >= JSON_CALLS.len() || rng.below(2) == 0;
    let (name, head, tail) = if let Some(&(name, open, close)) = JSON_CALLS.get(kind) {
        let value = if escaped {
            serde_json::to_string(&payload).expect("JSON")
        } else {
            // Unescaped text that keeps the JSON from ever reading: a control character, or a
            // DSML block whose attribute's quotes end the string early.
            let escaped_payload = serde_json::to_string(&payload).expect("JSON");
            let inner = &escaped_payload[1..escaped_payload.len() - 1];
            let breaker = ["\n", "\t", "\r", DELETE_ALL][rng.below(4)];
            let split = inner
                .char_indices()
                .map(|(at, _)| at)
                .filter(|&at| !inner[..at].ends_with('\\'))
                .nth(rng.below(inner.chars().count() + 1))
                .unwrap_or(inner.len());
            format!("\"{}{breaker}{}\"", &inner[..split], &inner[split..])
        };
        (name, format!("{open}{value}"), close.to_string())
    } else {
        match kind - JSON_CALLS.len() {
            0 => {
                let value = payload.replace("</parameter>", "");
                let lead = if rng.below(2) == 0 { "Example:\n" } else { "\n" };
                (
                    "qwen-xml",
                    format!("<tool_call>{lead}<function=write_file>\n<parameter=content>\n{value}"),
                    "\n</parameter>\n</function>\n</tool_call>".to_string(),
                )
            }
            1 => (
                "atem",
                format!(
                    "<atem:function_calls>\n<atem:invoke name=\"write_file\">\n<atem:parameter name=\"content\">{}",
                    payload.replace("</atem:parameter>", "")
                ),
                "</atem:parameter>\n</atem:invoke>\n</atem:function_calls>".to_string(),
            ),
            2 => (
                "liquid-double",
                format!(
                    "<|tool_call_start|>[write_file(content=\"{}",
                    payload.replace('\\', "\\\\").replace('"', "\\\"")
                ),
                "\")]<|tool_call_end|>".to_string(),
            ),
            3 => (
                "liquid-single",
                format!(
                    "<|tool_call_start|>[write_file(content='{}",
                    payload.replace('\\', "\\\\").replace('\'', "\\'")
                ),
                "')]<|tool_call_end|>".to_string(),
            ),
            _ => (
                "gemma4",
                format!(
                    "<|tool_call>call:write_file{{content:<|\"|>{}",
                    payload.replace("<|\"|>", "")
                ),
                "<|\"|>}<tool_call|>".to_string(),
            ),
        }
    };
    let finished = escaped;
    let mut message = format!("{head}{tail}");
    let mut cuts = vec![head.len()];
    if let Some(at) = message.find(OPENER) {
        cuts.push(at + OPENER.len());
    }
    let mut dsml = None;
    if finished && rng.below(5) < 2 {
        let (block, calls) = if rng.below(2) == 0 {
            (DELETE_ALL.to_string(), vec![delete_all_call()])
        } else {
            (write_file_block(), vec![write_file_call()])
        };
        let at = message.len() + 1;
        message.push('\n');
        message.push_str(&block);
        cuts.extend([at - 1, at, at + OPENER.len()]);
        dsml = Some(TopLevelDsml {
            reason: DSML_OWNS,
            at,
            blocks: vec![(message.len(), calls)],
        });
    }
    // Where a call written in the value could be complete, so a call streaming makes there is
    // checked against what the base reads there.
    for closer in CLOSERS {
        cuts.extend(
            message
                .match_indices(closer)
                .map(|(at, _)| at + closer.len()),
        );
    }
    let boundaries: Vec<usize> = message.char_indices().map(|(at, _)| at).collect();
    for _ in 0..2 {
        cuts.push(boundaries[rng.below(boundaries.len())]);
    }
    cuts.push(message.len());
    cuts.retain(|&cut| cut > 0);
    cuts.sort_unstable();
    cuts.dedup();
    Entry {
        id: format!(
            "generated:{name}:case={case}:{}",
            if escaped { "escaped" } else { "unescaped" }
        ),
        message,
        cuts,
        dsml,
    }
}

/// The formats a mutated entry writes its real call in, by name: a call to `write_file` whose
/// `content` is `value`, escaped as the format escapes a value, and the format's opener and
/// closer, if its calls close.
const SHAPES: [(&str, &str, Option<&str>); 9] = [
    ("qwen-json", "<tool_call>", Some("</tool_call>")),
    ("qwen-xml", "<tool_call>", Some("</tool_call>")),
    ("hunyuan", "<tool_calls>", Some("</tool_calls>")),
    (
        "deepseek",
        "<｜tool▁call▁begin｜>",
        Some("<｜tool▁call▁end｜>"),
    ),
    ("mistral-nemo", "[TOOL_CALLS]", None),
    ("llama", "<|python_tag|>", None),
    ("liquid", "<|tool_call_start|>", Some("<|tool_call_end|>")),
    ("gemma4", "<|tool_call>", Some("<tool_call|>")),
    (
        "atem",
        "<atem:function_calls>",
        Some("</atem:function_calls>"),
    ),
];

/// The openers and closers a mutation splices in: every format's, DSML's included.
const SPLICED: [(&str, Option<&str>); 9] = [
    ("<tool_call>", Some("</tool_call>")),
    ("<tool_calls>", Some("</tool_calls>")),
    ("<｜tool▁call▁begin｜>", Some("<｜tool▁call▁end｜>")),
    ("[TOOL_CALLS]", None),
    ("<|python_tag|>", None),
    ("<|tool_call_start|>", Some("<|tool_call_end|>")),
    ("<|tool_call>", Some("<tool_call|>")),
    ("<atem:function_calls>", Some("</atem:function_calls>")),
    (OPENER, Some(BLOCK_CLOSER)),
];

/// What each spliced opener's parser reads as the start of a call after it, in `SPLICED` order:
/// the text a recognition check looks for.
const RECOGNITION: [&str; 9] = [
    "<function=x>",
    "[",
    "function<｜tool▁sep｜>",
    "[",
    "{",
    "[",
    "call:x",
    "<atem:invoke name=\"x\">",
    "<｜DSML｜invoke name=\"x\">",
];

/// A call to `write_file` in the format `SHAPES[shape]`, its `content` the text `value`,
/// escaped as that format escapes a value, so the whole of `value` is the value's text.
fn shaped_call(shape: usize, value: &str) -> String {
    let json = serde_json::to_string(value).expect("JSON");
    match SHAPES[shape].0 {
        "qwen-json" => format!(
            r#"<tool_call>{{"name":"write_file","arguments":{{"content":{json}}}}}</tool_call>"#
        ),
        "qwen-xml" => format!(
            "<tool_call>\n<function=write_file>\n<parameter=content>\n{}\n</parameter>\n</function>\n</tool_call>",
            value.replace("</parameter>", "")
        ),
        "hunyuan" => format!(
            r#"<tool_calls>[{{"name":"write_file","arguments":{{"content":{json}}}}}]</tool_calls>"#
        ),
        "deepseek" => format!(
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>write_file\n```json\n{{\"content\":{json}}}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        ),
        "mistral-nemo" => format!(
            r#"[TOOL_CALLS][{{"name":"write_file","arguments":{{"content":{json}}}}}]"#
        ),
        "llama" => format!(r#"<|python_tag|>{{"name":"write_file","parameters":{{"content":{json}}}}}"#),
        "liquid" => format!(
            "<|tool_call_start|>[write_file(content='{}')]<|tool_call_end|>",
            value.replace('\\', "\\\\").replace('\'', "\\'")
        ),
        "gemma4" => format!(
            "<|tool_call>call:write_file{{content:<|\"|>{}<|\"|>}}<tool_call|>",
            value.replace("<|\"|>", "")
        ),
        _ => format!(
            "<atem:function_calls>\n<atem:invoke name=\"write_file\">\n<atem:parameter name=\"content\">{}</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
            value.replace("</atem:parameter>", "")
        ),
    }
}

/// The mutated entries, drawn from `MUTATION_SEED`: for every format, a real call preceded by a
/// stray opener of its own format that never closes; the same call with its opener written
/// twice; a call whose value holds another format's closer after an unmatched opener of that
/// format before the call; a call whose value holds another format's opener, followed by a real
/// call in that format; and an unmatched opener, then a call whose value holds what that
/// opener's parser reads as a call's start and its closer, then a DSML block. Each value may
/// hold a DSML block, which, inside a value, is text.
fn mutations() -> Vec<Entry> {
    let mut rng = Rng(MUTATION_SEED);
    let mut entries = Vec::new();
    let fillers = ["example\n", "Like this: ", "", "text "];
    for (shape, &(name, opener, closer)) in SHAPES.iter().enumerate() {
        let value = |rng: &mut Rng, around: &str| {
            let block = if rng.below(4) == 0 { "" } else { DELETE_ALL };
            format!("{around}{block}")
        };
        let filler = |rng: &mut Rng| fillers[rng.below(fillers.len())];
        let own_closer = closer.unwrap_or("");
        let stray = format!(
            "{opener}{}{}",
            filler(&mut rng),
            shaped_call(shape, &value(&mut rng, own_closer))
        );
        entries.push(mutated(format!("stray-opener:{name}"), stray, None));
        let single = shaped_call(shape, &value(&mut rng, own_closer));
        let doubled = single.replacen(opener, &format!("{opener}{opener}"), 1);
        entries.push(mutated(format!("two-openers:{name}"), doubled, None));
        for (other, &(other_opener, other_closer)) in SPLICED.iter().enumerate() {
            if other_opener == opener {
                continue;
            }
            if let Some(other_closer) = other_closer {
                let message = format!(
                    "{other_opener}{}{}",
                    filler(&mut rng),
                    shaped_call(shape, &value(&mut rng, other_closer))
                );
                let dsml = (other_opener == OPENER).then_some(0);
                entries.push(mutated(
                    format!("unmatched-opener-then-closer-in-value:{name}:{other}"),
                    message,
                    dsml.map(|at| (at, Vec::new())),
                ));
            }
            if let Some(other_closer) = other_closer {
                let message = format!(
                    "{other_opener}{}{}",
                    filler(&mut rng),
                    shaped_call(
                        shape,
                        &format!("{}{other_closer}{DELETE_ALL}", RECOGNITION[other])
                    )
                );
                let dsml = (other_opener == OPENER).then_some((0, Vec::new()));
                entries.push(mutated(
                    format!("unmatched-opener-then-its-start-and-closer-in-value:{name}:{other}"),
                    message,
                    dsml,
                ));
            }
            let inner = value(&mut rng, other_opener);
            let first = shaped_call(shape, &inner);
            let real = SHAPES
                .iter()
                .position(|&(_, opener, _)| opener == other_opener)
                .map_or_else(write_file_hello_block, |real| shaped_call(real, "hello"));
            let message = format!("{first}\n{real}");
            let dsml = (other_opener == OPENER).then(|| {
                (
                    first.len() + 1,
                    vec![call("write_file", &json!({ "content": "hello" }))],
                )
            });
            entries.push(mutated(
                format!("opener-in-value-then-call:{name}:{other}"),
                message,
                dsml,
            ));
        }
    }
    entries
}

/// Messages heavy with markers, `n` of them each, by name: Qwen openers rejected for want of a
/// call, repeated Gemma 4 calls with no arguments, DSML openers in a bare-JSON string, closing
/// tags in a JSON parameter value, and, for every format, `n` stray openers before a real call.
/// Reading any of them takes time linear in its length.
pub(super) fn marker_heavy_messages(n: usize) -> Vec<(String, String)> {
    let mut messages = vec![
        (
            "qwen-rejected-openers".to_string(),
            format!("{}</tool_call>", "<tool_call>text ".repeat(n)),
        ),
        (
            "gemma4-calls-without-arguments".to_string(),
            "<|tool_call>call:x<tool_call|>".repeat(n),
        ),
        (
            "dsml-openers-in-bare-json".to_string(),
            format!(
                r#"{{"name":"write_file","arguments":{{"content":"{}"#,
                OPENER.repeat(n)
            ),
        ),
        (
            "closing-tags-in-a-json-value".to_string(),
            format!(
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"content\" string=\"false\">{}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
                serde_json::to_string(&"</｜DSML｜parameter>".repeat(n)).expect("JSON")
            ),
        ),
    ];
    for (shape, &(name, opener, _)) in SHAPES.iter().enumerate() {
        messages.push((
            format!("stray-openers:{name}"),
            format!(
                "{}{}",
                format!("{opener}text ").repeat(n),
                shaped_call(shape, "hello")
            ),
        ));
    }
    messages
}

/// A DSML block calling `write_file` with `content` "hello".
fn write_file_hello_block() -> String {
    "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"content\" string=\"true\">hello</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>".to_string()
}

/// A mutated entry: read at every marker's end and the whole message, with the DSML block
/// outside any other call at `dsml.0`, if any, making `dsml.1`.
fn mutated(id: String, message: String, dsml: Option<(usize, Vec<(String, String)>)>) -> Entry {
    let mut cuts = Vec::new();
    for marker in MARKERS.iter().chain(CLOSERS.iter()) {
        cuts.extend(
            message
                .match_indices(marker)
                .map(|(at, _)| at + marker.len()),
        );
        cuts.extend(message.match_indices(marker).map(|(at, _)| at));
    }
    // Where a JSON call could be complete.
    cuts.extend(message.match_indices(['}', ']']).map(|(at, _)| at + 1));
    cuts.push(message.len());
    cuts.retain(|&cut| cut > 0);
    cuts.sort_unstable();
    cuts.dedup();
    let dsml = dsml.map(|(at, calls)| TopLevelDsml {
        reason: DSML_OWNS,
        at,
        blocks: if calls.is_empty() {
            Vec::new()
        } else {
            vec![(message.len(), calls)]
        },
    });
    Entry {
        id: format!("mutation:{id}"),
        message,
        cuts,
        dsml,
    }
}

/// The hand-written entries, then `GENERATED` drawn from `SEED`, then the mutations.
pub(super) fn corpus() -> Vec<Entry> {
    let mut entries = hand_corpus();
    let mut rng = Rng(SEED);
    entries.extend((0..GENERATED).map(|case| generated(&mut rng, case)));
    entries.extend(mutations());
    entries
}
