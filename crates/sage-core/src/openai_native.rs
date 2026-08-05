use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashSet};
use std::time::Duration;
use tokio::sync::mpsc;

const NATIVE_PROVIDER_REQUEST_TIMEOUT: Duration = Duration::from_secs(180);
const NATIVE_PROVIDER_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_NATIVE_SSE_LINE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub struct NativeToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeAssistantMessage {
    pub content: String,
    pub tool_calls: Vec<NativeToolCall>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NativeChatMessage {
    System(String),
    User(String),
    Assistant(NativeAssistantMessage),
    ToolResult { call_id: String, content: String },
}

impl NativeChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System(content.into())
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User(content.into())
    }

    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult {
            call_id: call_id.into(),
            content: content.into(),
        }
    }

    fn to_wire_value(&self) -> Value {
        match self {
            Self::System(content) => json!({"role": "system", "content": content}),
            Self::User(content) => json!({"role": "user", "content": content}),
            Self::Assistant(message) => {
                let mut value = Map::from_iter([
                    ("role".to_string(), json!("assistant")),
                    ("content".to_string(), json!(message.content)),
                ]);
                if !message.tool_calls.is_empty() {
                    value.insert(
                        "tool_calls".to_string(),
                        Value::Array(
                            message
                                .tool_calls
                                .iter()
                                .map(|call| {
                                    json!({
                                        "id": call.id,
                                        "type": "function",
                                        "function": {
                                            "name": call.name,
                                            "arguments": call.arguments.to_string(),
                                        }
                                    })
                                })
                                .collect(),
                        ),
                    );
                }
                Value::Object(value)
            }
            Self::ToolResult { call_id, content } => json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeFinishReason {
    Stop,
    ToolCalls,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeProviderSignal {
    Content(String),
    Event,
}

const MAX_NATIVE_TOOL_CALLS: usize = 8;

#[derive(Clone, Debug, PartialEq)]
pub struct NativeAssistantTurn {
    pub content: String,
    pub tool_calls: Vec<NativeToolCall>,
    pub finish_reason: NativeFinishReason,
}

#[derive(Clone, Debug)]
pub struct NativeTurnRequest {
    pub model: String,
    pub messages: Vec<NativeChatMessage>,
    pub tools: Vec<NativeToolDefinition>,
    pub max_tokens: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum NativeProviderError {
    #[error("native provider transport failed: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("native provider returned HTTP {status}: {body}")]
    Http {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("native provider protocol error: {0}")]
    Protocol(String),
    #[error("native provider timed out during {phase}")]
    Timeout { phase: &'static str },
}

pub struct OpenAiNativeClient {
    client: Client,
    api_url: String,
    api_key: String,
    temperature: f64,
    request_timeout: Duration,
    stream_idle_timeout: Duration,
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Default)]
struct NativeStreamState {
    content: String,
    tool_calls: BTreeMap<u64, PartialToolCall>,
    finish_reason: Option<NativeFinishReason>,
    saw_done: bool,
}

impl OpenAiNativeClient {
    pub fn new(client: Client, api_url: String, api_key: String, temperature: f64) -> Self {
        Self {
            client,
            api_url,
            api_key,
            temperature,
            request_timeout: NATIVE_PROVIDER_REQUEST_TIMEOUT,
            stream_idle_timeout: NATIVE_PROVIDER_STREAM_IDLE_TIMEOUT,
        }
    }

    #[cfg(test)]
    fn new_with_timeouts(
        client: Client,
        api_url: String,
        api_key: String,
        temperature: f64,
        request_timeout: Duration,
        stream_idle_timeout: Duration,
    ) -> Self {
        Self {
            client,
            api_url,
            api_key,
            temperature,
            request_timeout,
            stream_idle_timeout,
        }
    }

    pub async fn stream_turn(
        &self,
        request: NativeTurnRequest,
        signal_sender: Option<mpsc::UnboundedSender<NativeProviderSignal>>,
    ) -> Result<NativeAssistantTurn, NativeProviderError> {
        tokio::time::timeout(
            self.request_timeout,
            self.stream_turn_inner(request, signal_sender),
        )
        .await
        .map_err(|_| NativeProviderError::Timeout {
            phase: "request_total",
        })?
    }

    async fn stream_turn_inner(
        &self,
        request: NativeTurnRequest,
        signal_sender: Option<mpsc::UnboundedSender<NativeProviderSignal>>,
    ) -> Result<NativeAssistantTurn, NativeProviderError> {
        let mut body = Map::from_iter([
            ("model".to_string(), json!(request.model)),
            (
                "messages".to_string(),
                Value::Array(
                    request
                        .messages
                        .iter()
                        .map(NativeChatMessage::to_wire_value)
                        .collect(),
                ),
            ),
            ("temperature".to_string(), json!(self.temperature)),
            ("max_tokens".to_string(), json!(request.max_tokens)),
            ("stream".to_string(), json!(true)),
        ]);
        if !request.tools.is_empty() {
            body.insert(
                "tools".to_string(),
                Value::Array(
                    request
                        .tools
                        .iter()
                        .map(|tool| {
                            json!({
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": tool.parameters,
                                }
                            })
                        })
                        .collect(),
                ),
            );
            body.insert("tool_choice".to_string(), json!("auto"));
        }

        let response = self
            .client
            .post(format!(
                "{}/chat/completions",
                self.api_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
            .json(&Value::Object(body))
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(NativeProviderError::Http {
                status,
                body: truncate(&body, 500),
            });
        }

        let mut state = NativeStreamState::default();
        let mut buffer = Vec::new();
        let mut stream = response.bytes_stream();
        loop {
            let next = tokio::time::timeout(self.stream_idle_timeout, stream.next())
                .await
                .map_err(|_| NativeProviderError::Timeout {
                    phase: "stream_idle",
                })?;
            let Some(chunk) = next else {
                break;
            };
            buffer.extend_from_slice(&chunk?);
            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = String::from_utf8_lossy(&buffer[..newline])
                    .trim_end_matches('\r')
                    .to_string();
                buffer.drain(..=newline);
                consume_sse_line(&line, &mut state, &signal_sender)?;
            }
            validate_sse_buffer_len(buffer.len())?;
        }
        if !buffer.is_empty() {
            validate_sse_buffer_len(buffer.len())?;
            let line = String::from_utf8_lossy(&buffer)
                .trim_end_matches(['\r', '\n'])
                .to_string();
            consume_sse_line(&line, &mut state, &signal_sender)?;
        }
        finish_stream(state)
    }
}

fn validate_sse_buffer_len(buffer_len: usize) -> Result<(), NativeProviderError> {
    if buffer_len > MAX_NATIVE_SSE_LINE_BYTES {
        return Err(NativeProviderError::Protocol(format!(
            "provider sent an SSE line longer than {MAX_NATIVE_SSE_LINE_BYTES} bytes"
        )));
    }
    Ok(())
}

fn consume_sse_line(
    line: &str,
    state: &mut NativeStreamState,
    signal_sender: &Option<mpsc::UnboundedSender<NativeProviderSignal>>,
) -> Result<(), NativeProviderError> {
    let Some(data) = line.strip_prefix("data:").map(str::trim) else {
        return Ok(());
    };
    if data.is_empty() {
        return Ok(());
    }
    if data == "[DONE]" {
        state.saw_done = true;
        return Ok(());
    }
    let value: Value = serde_json::from_str(data)
        .map_err(|error| NativeProviderError::Protocol(format!("malformed SSE JSON: {error}")))?;
    if let Some(provider_error) = value.get("error") {
        return Err(NativeProviderError::Protocol(format!(
            "provider streamed an error: {provider_error}"
        )));
    }
    let choices = value
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(|| NativeProviderError::Protocol("SSE event omitted choices".to_string()))?;
    if choices.is_empty() {
        return Ok(());
    }
    if choices.len() != 1 {
        return Err(NativeProviderError::Protocol(
            "SSE event returned multiple choices".to_string(),
        ));
    }
    let choice = &choices[0];
    let empty_delta = Map::new();
    let delta = match choice.get("delta") {
        None | Some(Value::Null) => &empty_delta,
        Some(Value::Object(delta)) => delta,
        Some(_) => {
            return Err(NativeProviderError::Protocol(
                "choice delta was not an object".to_string(),
            ));
        }
    };
    let mut emitted_content = false;
    if let Some(content) = optional_string(delta.get("content"), "delta.content")? {
        state.content.push_str(content);
        if !content.is_empty() {
            if let Some(sender) = signal_sender {
                let _ = sender.send(NativeProviderSignal::Content(content.to_string()));
            }
            emitted_content = true;
        }
    }
    for field in ["reasoning", "reasoning_content"] {
        let _ = optional_string(delta.get(field), field)?;
    }
    if let Some(tool_calls) = delta.get("tool_calls") {
        let tool_calls = tool_calls.as_array().ok_or_else(|| {
            NativeProviderError::Protocol("delta.tool_calls was not an array".to_string())
        })?;
        for wire_call in tool_calls {
            let index = wire_call
                .get("index")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    NativeProviderError::Protocol(
                        "streamed Tool call omitted its numeric index".to_string(),
                    )
                })?;
            if !state.tool_calls.contains_key(&index)
                && state.tool_calls.len() == MAX_NATIVE_TOOL_CALLS
            {
                return Err(NativeProviderError::Protocol(format!(
                    "provider selected more than the allowed maximum of {MAX_NATIVE_TOOL_CALLS} Tool calls"
                )));
            }
            let partial = state.tool_calls.entry(index).or_default();
            append_optional_string(&mut partial.id, wire_call.get("id"), "tool_call.id")?;
            if let Some(function) = wire_call.get("function") {
                let function = function.as_object().ok_or_else(|| {
                    NativeProviderError::Protocol(
                        "streamed Tool call function was not an object".to_string(),
                    )
                })?;
                append_optional_string(
                    &mut partial.name,
                    function.get("name"),
                    "tool_call.function.name",
                )?;
                append_optional_string(
                    &mut partial.arguments,
                    function.get("arguments"),
                    "tool_call.function.arguments",
                )?;
            }
        }
    }
    if let Some(reason) = choice.get("finish_reason").filter(|value| !value.is_null()) {
        let reason = reason.as_str().ok_or_else(|| {
            NativeProviderError::Protocol("finish_reason was not a string".to_string())
        })?;
        let finish_reason = match reason {
            "stop" => NativeFinishReason::Stop,
            "tool_calls" => NativeFinishReason::ToolCalls,
            "length" => {
                return Err(NativeProviderError::Protocol(
                    "provider reached its token limit".to_string(),
                ));
            }
            other => {
                return Err(NativeProviderError::Protocol(format!(
                    "unsupported finish reason '{other}'"
                )));
            }
        };
        if state.finish_reason.replace(finish_reason).is_some() {
            return Err(NativeProviderError::Protocol(
                "provider sent multiple finish reasons".to_string(),
            ));
        }
    }
    if !emitted_content && !delta.is_empty() {
        if let Some(sender) = signal_sender {
            let _ = sender.send(NativeProviderSignal::Event);
        }
    }
    Ok(())
}

fn finish_stream(state: NativeStreamState) -> Result<NativeAssistantTurn, NativeProviderError> {
    if !state.saw_done {
        return Err(NativeProviderError::Protocol(
            "provider stream ended before [DONE]".to_string(),
        ));
    }
    let finish_reason = state.finish_reason.ok_or_else(|| {
        NativeProviderError::Protocol("provider stream omitted a finish reason".to_string())
    })?;
    let mut tool_calls = Vec::with_capacity(state.tool_calls.len());
    let mut call_ids = HashSet::with_capacity(state.tool_calls.len());
    for (_, call) in state.tool_calls {
        if call.id.trim().is_empty() || call.name.trim().is_empty() {
            return Err(NativeProviderError::Protocol(
                "provider returned an incomplete Tool call".to_string(),
            ));
        }
        let raw_arguments = call.arguments.trim();
        let arguments = if raw_arguments.is_empty() {
            json!({})
        } else {
            serde_json::from_str::<Value>(raw_arguments).map_err(|error| {
                NativeProviderError::Protocol(format!(
                    "Tool '{}' returned malformed arguments: {error}",
                    call.name
                ))
            })?
        };
        if !arguments.is_object() {
            return Err(NativeProviderError::Protocol(format!(
                "Tool '{}' arguments were not a JSON object",
                call.name
            )));
        }
        if !call_ids.insert(call.id.clone()) {
            return Err(NativeProviderError::Protocol(format!(
                "provider returned duplicate Tool call id '{}'",
                call.id
            )));
        }
        tool_calls.push(NativeToolCall {
            id: call.id,
            name: call.name,
            arguments,
        });
    }
    match finish_reason {
        NativeFinishReason::Stop if !tool_calls.is_empty() => {
            return Err(NativeProviderError::Protocol(
                "provider stopped while returning Tool calls".to_string(),
            ));
        }
        NativeFinishReason::Stop if state.content.trim().is_empty() => {
            return Err(NativeProviderError::Protocol(
                "provider stopped without an answer".to_string(),
            ));
        }
        NativeFinishReason::ToolCalls if tool_calls.is_empty() => {
            return Err(NativeProviderError::Protocol(
                "provider selected Tools without returning a Tool call".to_string(),
            ));
        }
        NativeFinishReason::ToolCalls | NativeFinishReason::Stop => {}
    }
    Ok(NativeAssistantTurn {
        content: state.content,
        tool_calls,
        finish_reason,
    })
}

fn optional_string<'a>(
    value: Option<&'a Value>,
    field: &str,
) -> Result<Option<&'a str>, NativeProviderError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(NativeProviderError::Protocol(format!(
            "{field} was not a string"
        ))),
    }
}

fn append_optional_string(
    destination: &mut String,
    value: Option<&Value>,
    field: &str,
) -> Result<(), NativeProviderError> {
    if let Some(value) = optional_string(value, field)? {
        destination.push_str(value);
    }
    Ok(())
}

fn truncate(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::{
        consume_sse_line, finish_stream, validate_sse_buffer_len, NativeAssistantMessage,
        NativeChatMessage, NativeFinishReason, NativeProviderSignal, NativeStreamState,
        NativeToolCall, NativeToolDefinition, NativeTurnRequest, OpenAiNativeClient,
        MAX_NATIVE_SSE_LINE_BYTES,
    };
    use axum::{
        body::{Body, Bytes},
        extract::State,
        http::{header::CONTENT_TYPE, StatusCode},
        response::{IntoResponse, Response},
        routing::post,
        Json, Router,
    };
    use serde_json::{json, Value};
    use std::convert::Infallible;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio::sync::mpsc;

    #[derive(Clone, Default)]
    struct CapturedRequests(Arc<Mutex<Vec<Value>>>);

    async fn completion(
        State(captured): State<CapturedRequests>,
        Json(body): Json<Value>,
    ) -> impl IntoResponse {
        let request_index = {
            let mut requests = captured
                .0
                .lock()
                .expect("capture lock should remain healthy");
            requests.push(body);
            requests.len()
        };
        let stream = if request_index == 1 {
            concat!(
                "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"knowledge_search\",\"arguments\":\"{\\\"query\\\":\"}}]},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"freedom guide\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
                "data: [DONE]\n\n"
            )
        } else {
            concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"Grounded \"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"answer.\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n"
            )
        };
        (
            StatusCode::OK,
            [("content-type", "text/event-stream")],
            stream,
        )
    }

    #[tokio::test]
    async fn native_provider_round_trips_tool_calls_and_results() {
        let captured = CapturedRequests::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test provider should bind");
        let address = listener.local_addr().expect("test provider address");
        let app = Router::new()
            .route("/v1/chat/completions", post(completion))
            .with_state(captured.clone());
        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("test provider should serve");
        });

        let client = OpenAiNativeClient::new(
            reqwest::Client::new(),
            format!("http://{address}/v1"),
            "test-key".to_string(),
            0.1,
        );
        let tool = NativeToolDefinition {
            name: "knowledge_search".to_string(),
            description: "Search uploaded Documents.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": false
            }),
        };
        let first = client
            .stream_turn(
                NativeTurnRequest {
                    model: "glm-5-2".to_string(),
                    messages: vec![
                        NativeChatMessage::system("You are Sage."),
                        NativeChatMessage::user("What does the guide say?"),
                    ],
                    tools: vec![tool],
                    max_tokens: 8192,
                },
                None,
            )
            .await
            .expect("native Tool selection should parse");

        assert_eq!(first.finish_reason, NativeFinishReason::ToolCalls);
        assert_eq!(
            first.tool_calls,
            vec![NativeToolCall {
                id: "call-1".to_string(),
                name: "knowledge_search".to_string(),
                arguments: json!({"query": "freedom guide"}),
            }]
        );

        let (signal_sender, mut signal_receiver) = mpsc::unbounded_channel();
        let second = client
            .stream_turn(
                NativeTurnRequest {
                    model: "glm-5-2".to_string(),
                    messages: vec![
                        NativeChatMessage::system("You are Sage."),
                        NativeChatMessage::user("What does the guide say?"),
                        NativeChatMessage::Assistant(NativeAssistantMessage {
                            content: first.content,
                            tool_calls: first.tool_calls,
                        }),
                        NativeChatMessage::tool_result("call-1", "The guide recommends safety."),
                    ],
                    tools: Vec::new(),
                    max_tokens: 8192,
                },
                Some(signal_sender),
            )
            .await
            .expect("final native answer should stream");

        assert_eq!(second.finish_reason, NativeFinishReason::Stop);
        assert_eq!(second.content, "Grounded answer.");
        assert_eq!(
            vec![signal_receiver.recv().await, signal_receiver.recv().await,],
            vec![
                Some(NativeProviderSignal::Content("Grounded ".to_string())),
                Some(NativeProviderSignal::Content("answer.".to_string())),
            ]
        );

        let requests = captured.0.lock().expect("captured requests");
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0].pointer("/tools/0/function/name"),
            Some(&json!("knowledge_search"))
        );
        assert_eq!(
            requests[0].pointer("/tools/0/function/parameters/properties/query/type"),
            Some(&json!("string"))
        );
        assert_eq!(
            requests[1].pointer("/messages/2/tool_calls/0/id"),
            Some(&json!("call-1"))
        );
        assert_eq!(
            requests[1].pointer("/messages/3/tool_call_id"),
            Some(&json!("call-1"))
        );
        assert!(requests[1].get("tools").is_none());
    }

    #[test]
    fn tool_call_only_delta_emits_a_content_free_provider_event() {
        let mut state = NativeStreamState::default();
        let (signal_sender, mut signal_receiver) = mpsc::unbounded_channel();

        consume_sse_line(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"knowledge_search","arguments":"{\"query\":\"guide\"}"}}]},"finish_reason":"tool_calls"}]}"#,
            &mut state,
            &Some(signal_sender),
        )
        .expect("Tool-call-only event should parse");

        assert_eq!(
            signal_receiver.try_recv().expect("provider event signal"),
            NativeProviderSignal::Event
        );
    }

    #[test]
    fn terminal_null_delta_preserves_finish_reason() {
        let mut state = NativeStreamState::default();
        state.content.push_str("Complete answer.");

        consume_sse_line(
            r#"data: {"choices":[{"delta":null,"finish_reason":"stop"}]}"#,
            &mut state,
            &None,
        )
        .expect("a terminal null delta should still carry its finish reason");
        consume_sse_line("data: [DONE]", &mut state, &None).expect("DONE marker should parse");

        let turn = finish_stream(state).expect("terminal stream should finish");
        assert_eq!(turn.finish_reason, NativeFinishReason::Stop);
        assert_eq!(turn.content, "Complete answer.");
    }

    #[test]
    fn empty_tool_arguments_become_an_empty_object() {
        let mut state = NativeStreamState::default();
        consume_sse_line(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"read_status","arguments":"   "}}]},"finish_reason":"tool_calls"}]}"#,
            &mut state,
            &None,
        )
        .expect("empty Tool arguments should parse as stream fragments");
        consume_sse_line("data: [DONE]", &mut state, &None).expect("DONE marker should parse");

        let turn = finish_stream(state).expect("empty arguments should normalize");
        assert_eq!(turn.tool_calls[0].arguments, json!({}));
    }

    #[test]
    fn unterminated_sse_line_is_bounded() {
        validate_sse_buffer_len(MAX_NATIVE_SSE_LINE_BYTES)
            .expect("the exact line budget should be accepted");
        let error = validate_sse_buffer_len(MAX_NATIVE_SSE_LINE_BYTES + 1)
            .expect_err("an oversized unterminated line must be rejected");
        assert!(error.to_string().contains("SSE line longer"));
    }

    #[tokio::test]
    async fn stalled_stream_body_hits_the_native_idle_timeout() {
        async fn completion() -> Response {
            let body = Body::from_stream(async_stream::stream! {
                yield Ok::<_, Infallible>(Bytes::from_static(
                    b"data: {\"choices\":[{\"delta\":{\"content\":\"Started\"},\"finish_reason\":null}]}\n\n",
                ));
                std::future::pending::<()>().await;
            });
            Response::builder()
                .header(CONTENT_TYPE, "text/event-stream")
                .body(body)
                .expect("stream response should build")
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test provider should bind");
        let address = listener.local_addr().expect("test provider address");
        let app = Router::new().route("/v1/chat/completions", post(completion));
        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("test provider should serve");
        });

        let client = OpenAiNativeClient::new_with_timeouts(
            reqwest::Client::new(),
            format!("http://{address}/v1"),
            "test-key".to_string(),
            0.1,
            Duration::from_millis(500),
            Duration::from_millis(25),
        );
        let error = client
            .stream_turn(
                NativeTurnRequest {
                    model: "glm-5-2".to_string(),
                    messages: vec![NativeChatMessage::user("hello")],
                    tools: Vec::new(),
                    max_tokens: 8192,
                },
                None,
            )
            .await
            .expect_err("a stalled body must not wait indefinitely");

        assert!(matches!(
            error,
            super::NativeProviderError::Timeout {
                phase: "stream_idle"
            }
        ));
    }

    #[test]
    fn duplicate_tool_call_ids_are_rejected_before_execution() {
        let mut state = NativeStreamState::default();
        consume_sse_line(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"duplicate","type":"function","function":{"name":"knowledge_search","arguments":"{}"}},{"index":1,"id":"duplicate","type":"function","function":{"name":"find_resources","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#,
            &mut state,
            &None,
        )
        .expect("stream fragments should parse before correlation validation");
        consume_sse_line("data: [DONE]", &mut state, &None).expect("DONE marker should parse");

        let error = finish_stream(state).expect_err("duplicate call IDs must be rejected");
        assert!(error.to_string().contains("duplicate Tool call id"));
    }

    #[test]
    fn oversized_native_tool_batch_is_rejected() {
        let tool_calls = (0..9)
            .map(|index| {
                json!({
                    "index": index,
                    "id": format!("call-{index}"),
                    "type": "function",
                    "function": {"name": "knowledge_search", "arguments": "{}"}
                })
            })
            .collect::<Vec<_>>();
        let line = format!(
            "data: {}",
            json!({"choices": [{"delta": {"tool_calls": tool_calls}, "finish_reason": null}]})
        );
        let mut state = NativeStreamState::default();

        let error = consume_sse_line(&line, &mut state, &None)
            .expect_err("oversized Tool batch must be rejected at the provider boundary");

        assert!(error.to_string().contains("maximum of 8 Tool calls"));
    }
}
