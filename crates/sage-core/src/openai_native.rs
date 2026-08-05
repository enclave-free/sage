use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::mpsc;

const MAX_NATIVE_SSE_LINE_BYTES: usize = 1024 * 1024;
pub const MAX_NATIVE_CONTINUITY_STATE_BYTES: usize = 1024 * 1024;

/// OpenAI-compatible reasoning effort sent unchanged to the Model Provider.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NativeReasoningEffort {
    #[default]
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl NativeReasoningEffort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

impl fmt::Display for NativeReasoningEffort {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for NativeReasoningEffort {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "minimal" => Ok(Self::Minimal),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::XHigh),
            "max" => Ok(Self::Max),
            _ => Err(format!(
                "unsupported reasoning effort {value:?}; expected none, minimal, low, medium, high, xhigh, or max"
            )),
        }
    }
}

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
    /// Provider-supplied opaque continuity state for the current native Tool loop.
    pub continuity_state: Option<String>,
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
                if let Some(continuity_state) = &message.continuity_state {
                    value.insert("reasoning_content".to_string(), json!(continuity_state));
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
    Usage(NativeModelUsage),
    Event,
}

const MAX_NATIVE_TOOL_CALLS: usize = 8;

#[derive(Clone, Debug, PartialEq)]
pub struct NativeAssistantTurn {
    pub content: String,
    pub tool_calls: Vec<NativeToolCall>,
    pub continuity_state: Option<String>,
    pub usage: Option<NativeModelUsage>,
    pub finish_reason: NativeFinishReason,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeModelUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
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
    #[error("native provider emitted no stream event before the first-event deadline")]
    PreResponseStall,
    #[error("native provider transport failed: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("native provider returned HTTP {status}: {body}")]
    Http {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("native provider protocol error: {0}")]
    Protocol(String),
}

pub struct OpenAiNativeClient {
    client: Client,
    api_url: String,
    api_key: String,
    temperature: f64,
    reasoning_effort: NativeReasoningEffort,
    stream_usage_supported: Arc<AtomicBool>,
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
    continuity_state: String,
    usage: Option<NativeModelUsage>,
    tool_calls: BTreeMap<u64, PartialToolCall>,
    finish_reason: Option<NativeFinishReason>,
    saw_done: bool,
}

impl OpenAiNativeClient {
    pub fn new(client: Client, api_url: String, api_key: String, temperature: f64) -> Self {
        let stream_usage_supported = stream_usage_capability(&api_url);
        Self {
            client,
            api_url,
            api_key,
            temperature,
            reasoning_effort: NativeReasoningEffort::default(),
            stream_usage_supported,
        }
    }

    pub fn with_reasoning_effort(mut self, reasoning_effort: NativeReasoningEffort) -> Self {
        self.reasoning_effort = reasoning_effort;
        self
    }

    pub async fn stream_turn(
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
            (
                "reasoning_effort".to_string(),
                json!(self.reasoning_effort.as_str()),
            ),
            ("max_tokens".to_string(), json!(request.max_tokens)),
            ("stream".to_string(), json!(true)),
        ]);
        if self.stream_usage_supported.load(Ordering::Relaxed) {
            body.insert("stream_options".to_string(), json!({"include_usage": true}));
        }
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
        } else {
            body.insert("tool_choice".to_string(), json!("none"));
        }

        let mut response = self.send_request(&body).await?;
        let mut status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::BAD_REQUEST
                && body.contains_key("stream_options")
                && rejects_stream_usage_options(&error_body)
            {
                self.stream_usage_supported.store(false, Ordering::Relaxed);
                body.remove("stream_options");
                response = self.send_request(&body).await?;
                status = response.status();
                if status.is_success() {
                    return consume_stream_response(response, signal_sender).await;
                }
                let fallback_body = response.text().await.unwrap_or_default();
                return Err(NativeProviderError::Http {
                    status,
                    body: truncate(&fallback_body, 500),
                });
            }
            return Err(NativeProviderError::Http {
                status,
                body: truncate(&error_body, 500),
            });
        }

        consume_stream_response(response, signal_sender).await
    }

    async fn send_request(
        &self,
        body: &Map<String, Value>,
    ) -> Result<reqwest::Response, reqwest::Error> {
        self.client
            .post(format!(
                "{}/chat/completions",
                self.api_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
            .json(&Value::Object(body.clone()))
            .send()
            .await
    }
}

fn stream_usage_capability(api_url: &str) -> Arc<AtomicBool> {
    static CAPABILITIES: OnceLock<Mutex<HashMap<String, Arc<AtomicBool>>>> = OnceLock::new();
    let key = api_url.trim_end_matches('/').to_string();
    let mut capabilities = CAPABILITIES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    capabilities
        .entry(key)
        .or_insert_with(|| Arc::new(AtomicBool::new(true)))
        .clone()
}

fn rejects_stream_usage_options(body: &str) -> bool {
    let normalized = body.to_ascii_lowercase();
    normalized.contains("stream_options") || normalized.contains("include_usage")
}

async fn consume_stream_response(
    response: reqwest::Response,
    signal_sender: Option<mpsc::UnboundedSender<NativeProviderSignal>>,
) -> Result<NativeAssistantTurn, NativeProviderError> {
    let mut state = NativeStreamState::default();
    let mut buffer = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
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
        if let Some(sender) = signal_sender {
            let _ = sender.send(NativeProviderSignal::Event);
        }
        return Err(NativeProviderError::Protocol(format!(
            "provider streamed an error: {provider_error}"
        )));
    }
    let usage_value = value.get("usage").filter(|usage| !usage.is_null());
    let mut emitted_usage = false;
    if state.usage.is_none() {
        if let Some(usage) = usage_value.and_then(parse_usage) {
            if let Some(sender) = signal_sender {
                let _ = sender.send(NativeProviderSignal::Usage(usage.clone()));
            }
            state.usage = Some(usage);
            emitted_usage = true;
        }
    }
    let choices = match value.get("choices").and_then(Value::as_array) {
        Some(choices) => choices,
        None if usage_value.is_some() => {
            if !emitted_usage {
                if let Some(sender) = signal_sender {
                    let _ = sender.send(NativeProviderSignal::Event);
                }
            }
            return Ok(());
        }
        None => {
            return Err(NativeProviderError::Protocol(
                "SSE event omitted choices".to_string(),
            ));
        }
    };
    if choices.is_empty() {
        if !emitted_usage {
            if let Some(sender) = signal_sender {
                let _ = sender.send(NativeProviderSignal::Event);
            }
        }
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
    let reasoning_content =
        optional_string(delta.get("reasoning_content"), "delta.reasoning_content")?;
    let reasoning = optional_string(delta.get("reasoning"), "delta.reasoning")?;
    if let Some(continuity_delta) = reasoning_content.or(reasoning) {
        if state
            .continuity_state
            .len()
            .checked_add(continuity_delta.len())
            .is_none_or(|size| size > MAX_NATIVE_CONTINUITY_STATE_BYTES)
        {
            return Err(NativeProviderError::Protocol(format!(
                "provider continuity state exceeded {MAX_NATIVE_CONTINUITY_STATE_BYTES} bytes"
            )));
        }
        state.continuity_state.push_str(continuity_delta);
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
    if !emitted_usage && !emitted_content {
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
        continuity_state: (!state.continuity_state.is_empty()).then_some(state.continuity_state),
        usage: state.usage,
        finish_reason,
    })
}

fn parse_usage(value: &Value) -> Option<NativeModelUsage> {
    let usage = value.as_object()?;
    let prompt_details = usage
        .get("prompt_tokens_details")
        .and_then(Value::as_object);
    let completion_details = usage
        .get("completion_tokens_details")
        .and_then(Value::as_object);
    Some(NativeModelUsage {
        prompt_tokens: usage.get("prompt_tokens").and_then(Value::as_u64),
        completion_tokens: usage.get("completion_tokens").and_then(Value::as_u64),
        total_tokens: usage.get("total_tokens").and_then(Value::as_u64),
        cached_tokens: prompt_details
            .and_then(|details| details.get("cached_tokens"))
            .and_then(Value::as_u64),
        reasoning_tokens: completion_details
            .and_then(|details| details.get("reasoning_tokens"))
            .and_then(Value::as_u64),
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
        NativeChatMessage, NativeFinishReason, NativeProviderSignal, NativeReasoningEffort,
        NativeStreamState, NativeToolCall, NativeToolDefinition, NativeTurnRequest,
        OpenAiNativeClient, MAX_NATIVE_CONTINUITY_STATE_BYTES, MAX_NATIVE_SSE_LINE_BYTES,
    };
    use axum::{
        extract::State, http::StatusCode, response::IntoResponse, routing::post, Json, Router,
    };
    use serde_json::{json, Value};
    use std::sync::{Arc, Mutex};
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
                "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"private continuity \",\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"knowledge_search\",\"arguments\":\"{\\\"query\\\":\"}}]},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"state\",\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"freedom guide\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
                "data: [DONE]\n\n"
            )
        } else {
            concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"Grounded \"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"answer.\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":41,\"completion_tokens\":17,\"total_tokens\":58,\"prompt_tokens_details\":{\"cached_tokens\":11},\"completion_tokens_details\":{\"reasoning_tokens\":7}}}\n\n",
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
        )
        .with_reasoning_effort(NativeReasoningEffort::High);
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
                    tools: vec![tool.clone()],
                    max_tokens: 8192,
                },
                None,
            )
            .await
            .expect("native Tool selection should parse");

        assert_eq!(first.finish_reason, NativeFinishReason::ToolCalls);
        assert_eq!(
            first.continuity_state.as_deref(),
            Some("private continuity state")
        );
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
                            continuity_state: first.continuity_state,
                        }),
                        NativeChatMessage::tool_result("call-1", "The guide recommends safety."),
                    ],
                    tools: vec![tool],
                    max_tokens: 8192,
                },
                Some(signal_sender),
            )
            .await
            .expect("final native answer should stream");

        assert_eq!(second.finish_reason, NativeFinishReason::Stop);
        assert_eq!(second.content, "Grounded answer.");
        assert_eq!(second.continuity_state, None);
        let usage = second.usage.expect("terminal usage should be observed");
        assert_eq!(usage.prompt_tokens, Some(41));
        assert_eq!(usage.completion_tokens, Some(17));
        assert_eq!(usage.total_tokens, Some(58));
        assert_eq!(usage.cached_tokens, Some(11));
        assert_eq!(usage.reasoning_tokens, Some(7));
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
        assert_eq!(requests[0].get("tool_choice"), Some(&json!("auto")));
        assert_eq!(requests[0].get("reasoning_effort"), Some(&json!("high")));
        assert_eq!(
            requests[0].pointer("/stream_options/include_usage"),
            Some(&json!(true))
        );
        assert_eq!(
            requests[1].pointer("/messages/2/tool_calls/0/id"),
            Some(&json!("call-1"))
        );
        assert_eq!(
            requests[1].pointer("/messages/2/reasoning_content"),
            Some(&json!("private continuity state"))
        );
        assert_eq!(
            requests[1].pointer("/messages/3/tool_call_id"),
            Some(&json!("call-1"))
        );
        assert_eq!(
            requests[1].pointer("/tools/0/function/name"),
            Some(&json!("knowledge_search"))
        );
        assert_eq!(requests[1].get("tool_choice"), Some(&json!("auto")));
        assert_eq!(requests[1].get("reasoning_effort"), Some(&json!("high")));
    }

    #[tokio::test]
    async fn unsupported_usage_extension_is_negotiated_once_before_inference() {
        async fn completion(
            State(captured): State<CapturedRequests>,
            Json(body): Json<Value>,
        ) -> axum::response::Response {
            let request_number = {
                let mut requests = captured.0.lock().unwrap();
                requests.push(body.clone());
                requests.len()
            };
            if request_number == 1 && body.get("stream_options").is_some() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": {"message": "Unsupported parameter: stream_options"}})),
                )
                    .into_response();
            }
            (
                StatusCode::OK,
                [("content-type", "text/event-stream")],
                concat!(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"Available.\"},\"finish_reason\":\"stop\"}]}\n\n",
                    "data: [DONE]\n\n"
                ),
            )
                .into_response()
        }

        let captured = CapturedRequests::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/v1/chat/completions", post(completion))
            .with_state(captured.clone());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = OpenAiNativeClient::new(
            reqwest::Client::new(),
            format!("http://{address}/v1"),
            "test-key".to_string(),
            0.1,
        );
        let request = || NativeTurnRequest {
            model: "compatible-model".to_string(),
            messages: vec![NativeChatMessage::user("hello")],
            tools: Vec::new(),
            max_tokens: 64,
        };

        let first = client
            .stream_turn(request(), None)
            .await
            .expect("unsupported optional usage should fall back before inference");
        let later_turn_client = OpenAiNativeClient::new(
            reqwest::Client::new(),
            format!("http://{address}/v1"),
            "test-key".to_string(),
            0.1,
        );
        let second = later_turn_client
            .stream_turn(request(), None)
            .await
            .expect("capability result should be retained for later requests");

        assert_eq!(first.content, "Available.");
        assert_eq!(second.content, "Available.");
        let requests = captured.0.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(
            requests[0].pointer("/stream_options/include_usage"),
            Some(&json!(true))
        );
        assert!(requests[1].get("stream_options").is_none());
        assert!(requests[2].get("stream_options").is_none());
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
    fn provider_error_event_still_closes_silent_retry_eligibility() {
        let mut state = NativeStreamState::default();
        let (signal_sender, mut signal_receiver) = mpsc::unbounded_channel();

        let error = consume_sse_line(
            r#"data: {"error":{"message":"upstream stopped"}}"#,
            &mut state,
            &Some(signal_sender),
        )
        .expect_err("provider error payload should remain a protocol failure");

        assert!(error.to_string().contains("provider streamed an error"));
        assert_eq!(
            signal_receiver.try_recv().expect("provider event signal"),
            NativeProviderSignal::Event
        );
    }

    #[test]
    fn malformed_optional_usage_is_best_effort_and_never_breaks_completion() {
        let mut state = NativeStreamState::default();
        let (signal_sender, mut signal_receiver) = mpsc::unbounded_channel();

        consume_sse_line(
            r#"data: {"usage":{"prompt_tokens":41,"completion_tokens":"unknown","prompt_tokens_details":"unsupported"}}"#,
            &mut state,
            &Some(signal_sender),
        )
        .expect("optional usage metadata must not fail an otherwise valid stream");
        consume_sse_line(
            r#"data: {"choices":[{"delta":{"content":"Complete."},"finish_reason":"stop"}]}"#,
            &mut state,
            &None,
        )
        .expect("answer should still parse");
        consume_sse_line("data: [DONE]", &mut state, &None).expect("DONE should parse");

        assert_eq!(
            signal_receiver
                .try_recv()
                .expect("usage observation signal"),
            NativeProviderSignal::Usage(super::NativeModelUsage {
                prompt_tokens: Some(41),
                completion_tokens: None,
                total_tokens: None,
                cached_tokens: None,
                reasoning_tokens: None,
            })
        );
        let turn = finish_stream(state).expect("usage metadata must not invalidate the answer");
        assert_eq!(turn.content, "Complete.");
        assert_eq!(turn.usage.unwrap().prompt_tokens, Some(41));
    }

    #[test]
    fn duplicate_usage_observations_keep_the_first_valid_observation() {
        let mut state = NativeStreamState::default();
        consume_sse_line(
            r#"data: {"choices":[],"usage":{"prompt_tokens":11}}"#,
            &mut state,
            &None,
        )
        .expect("first usage observation should parse");
        consume_sse_line(
            r#"data: {"choices":[],"usage":{"prompt_tokens":99}}"#,
            &mut state,
            &None,
        )
        .expect("duplicate optional usage must not fail the stream");
        consume_sse_line(
            r#"data: {"choices":[{"delta":{"content":"Complete."},"finish_reason":"stop"}]}"#,
            &mut state,
            &None,
        )
        .expect("answer should parse");
        consume_sse_line("data: [DONE]", &mut state, &None).expect("DONE should parse");

        assert_eq!(
            finish_stream(state).unwrap().usage.unwrap().prompt_tokens,
            Some(11)
        );
    }

    #[test]
    fn aggregate_provider_continuity_state_is_bounded() {
        let mut state = NativeStreamState {
            continuity_state: "x".repeat(MAX_NATIVE_CONTINUITY_STATE_BYTES),
            ..NativeStreamState::default()
        };

        let error = consume_sse_line(
            r#"data: {"choices":[{"delta":{"reasoning_content":"y"},"finish_reason":null}]}"#,
            &mut state,
            &None,
        )
        .expect_err("aggregate continuity state must remain bounded");

        assert!(error.to_string().contains("continuity state exceeded"));
        assert_eq!(
            state.continuity_state.len(),
            MAX_NATIVE_CONTINUITY_STATE_BYTES
        );
    }

    #[test]
    fn each_tool_batch_keeps_only_its_corresponding_continuity_state() {
        let messages = [
            NativeChatMessage::Assistant(NativeAssistantMessage {
                content: String::new(),
                tool_calls: vec![NativeToolCall {
                    id: "call-1".to_string(),
                    name: "knowledge_search".to_string(),
                    arguments: json!({"query": "first"}),
                }],
                continuity_state: Some("first private state".to_string()),
            }),
            NativeChatMessage::tool_result("call-1", "first result"),
            NativeChatMessage::Assistant(NativeAssistantMessage {
                content: String::new(),
                tool_calls: vec![NativeToolCall {
                    id: "call-2".to_string(),
                    name: "knowledge_search".to_string(),
                    arguments: json!({"query": "second"}),
                }],
                continuity_state: Some("second private state".to_string()),
            }),
            NativeChatMessage::tool_result("call-2", "second result"),
            NativeChatMessage::Assistant(NativeAssistantMessage {
                content: "final answer".to_string(),
                tool_calls: Vec::new(),
                continuity_state: None,
            }),
        ]
        .map(|message| message.to_wire_value());

        assert_eq!(messages[0]["reasoning_content"], "first private state");
        assert_eq!(messages[0]["tool_calls"][0]["id"], "call-1");
        assert_eq!(messages[2]["reasoning_content"], "second private state");
        assert_eq!(messages[2]["tool_calls"][0]["id"], "call-2");
        assert!(messages[4].get("reasoning_content").is_none());
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
        assert_eq!(turn.continuity_state, None);
        assert_eq!(turn.usage, None);
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
