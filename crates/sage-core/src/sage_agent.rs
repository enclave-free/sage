//! Sage Agent using DSRs signatures and BAML parsing
//!
//! This module implements the core agent using dspy-rs for:
//! - Typed input/output signatures
//! - BAML-based response parsing
//! - GEPA-compatible instruction optimization

use anyhow::{anyhow, Result};
use baml_bridge::{
    baml_types::{type_meta, BamlValue, TypeIR},
    BamlAdapter, BamlConvertError,
};
use dspy_rs::{configure, BamlType, ChatAdapter, Predict, LM};
use std::collections::{BTreeMap, HashMap, HashSet};
#[cfg(unix)]
use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::memory::MemoryManager;
use crate::openai_native::{NativeToolCall, NativeToolDefinition};

#[cfg(unix)]
struct StdoutSuppressor {
    saved_stdout_fd: i32,
}

#[cfg(unix)]
impl StdoutSuppressor {
    fn new() -> Option<Self> {
        let _ = std::io::stdout().flush();
        unsafe {
            let saved_stdout_fd = libc::dup(libc::STDOUT_FILENO);
            if saved_stdout_fd == -1 {
                return None;
            }

            let dev_null = std::ffi::CString::new("/dev/null").ok()?;
            let dev_null_fd = libc::open(dev_null.as_ptr(), libc::O_WRONLY);
            if dev_null_fd == -1 {
                libc::close(saved_stdout_fd);
                return None;
            }

            if libc::dup2(dev_null_fd, libc::STDOUT_FILENO) == -1 {
                libc::close(dev_null_fd);
                libc::close(saved_stdout_fd);
                return None;
            }
            libc::close(dev_null_fd);

            Some(Self { saved_stdout_fd })
        }
    }
}

#[cfg(unix)]
impl Drop for StdoutSuppressor {
    fn drop(&mut self) {
        let _ = std::io::stdout().flush();
        unsafe {
            libc::dup2(self.saved_stdout_fd, libc::STDOUT_FILENO);
            libc::close(self.saved_stdout_fd);
        }
    }
}

/// A tool call requested by the agent
struct NativeToolArgsAdapter;

impl NativeToolArgsAdapter {
    fn union(types: Vec<TypeIR>) -> TypeIR {
        TypeIR::union_with_meta(types, type_meta::IR::default())
    }

    fn primitive_type() -> TypeIR {
        Self::union(vec![
            TypeIR::string(),
            TypeIR::int(),
            TypeIR::float(),
            TypeIR::bool(),
            TypeIR::null(),
        ])
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolArgs(HashMap<String, serde_json::Value>);

impl ToolArgs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_json_object(value: &serde_json::Value) -> Result<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| anyhow!("native Tool arguments must be a JSON object"))?;
        Ok(Self(object.clone().into_iter().collect()))
    }
}

impl Deref for ToolArgs {
    type Target = HashMap<String, serde_json::Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ToolArgs {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const N: usize> From<[(String, serde_json::Value); N]> for ToolArgs {
    fn from(entries: [(String, serde_json::Value); N]) -> Self {
        Self(HashMap::from(entries))
    }
}

impl baml_bridge::ToBamlValue for ToolArgs {
    fn to_baml_value(&self) -> BamlValue {
        fn convert(value: &serde_json::Value) -> BamlValue {
            match value {
                serde_json::Value::Null => BamlValue::Null,
                serde_json::Value::Bool(value) => BamlValue::Bool(*value),
                serde_json::Value::Number(value) => value
                    .as_i64()
                    .map(BamlValue::Int)
                    .or_else(|| value.as_f64().map(BamlValue::Float))
                    .unwrap_or(BamlValue::Null),
                serde_json::Value::String(value) => BamlValue::String(value.clone()),
                serde_json::Value::Array(values) => {
                    BamlValue::List(values.iter().map(convert).collect())
                }
                serde_json::Value::Object(values) => BamlValue::Map(
                    values
                        .iter()
                        .map(|(key, value)| (key.clone(), convert(value)))
                        .collect(),
                ),
            }
        }

        BamlValue::Map(
            self.0
                .iter()
                .map(|(key, value)| (key.clone(), convert(value)))
                .collect(),
        )
    }
}

pub(crate) fn tool_string_arg<'a>(args: &'a ToolArgs, key: &str) -> Option<&'a str> {
    args.get(key).and_then(serde_json::Value::as_str)
}

pub(crate) fn tool_parse_arg<T>(args: &ToolArgs, key: &str) -> Option<T>
where
    T: std::str::FromStr,
{
    args.get(key).and_then(|value| match value {
        serde_json::Value::String(value) => value.parse().ok(),
        serde_json::Value::Number(_) | serde_json::Value::Bool(_) => value.to_string().parse().ok(),
        _ => None,
    })
}

fn tool_arg_preview(value: &serde_json::Value) -> String {
    let rendered = match value {
        serde_json::Value::String(value) => value.clone(),
        value => value.to_string(),
    };
    rendered.chars().take(500).collect()
}

fn truncate_chars_with_ellipsis(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let prefix_chars = max_chars.saturating_sub(3);
    format!(
        "{}...",
        value.chars().take(prefix_chars).collect::<String>()
    )
}

impl BamlAdapter<ToolArgs> for NativeToolArgsAdapter {
    fn type_ir() -> TypeIR {
        let primitive = Self::primitive_type();
        let nested_value = Self::union(vec![primitive.clone(), TypeIR::list(primitive.clone())]);
        let object = TypeIR::map(TypeIR::string(), nested_value);
        let list_item = Self::union(vec![primitive.clone(), object.clone()]);
        let value = Self::union(vec![primitive, object, TypeIR::list(list_item)]);
        TypeIR::map(TypeIR::string(), value)
    }

    fn try_from_baml(
        value: BamlValue,
        path: Vec<String>,
    ) -> std::result::Result<ToolArgs, BamlConvertError> {
        let BamlValue::Map(values) = value else {
            return Err(BamlConvertError::new(
                path,
                "map",
                format!("{value:?}"),
                "expected Tool arguments to be an object",
            ));
        };

        let mut args = ToolArgs::new();
        for (key, value) in values {
            if value == BamlValue::Null {
                continue;
            }
            let native_value = serde_json::to_value(&value).map_err(|error| {
                let mut value_path = path.clone();
                value_path.push(key.clone());
                BamlConvertError::new(
                    value_path,
                    "JSON-compatible Tool argument",
                    format!("{value:?}"),
                    format!("failed to preserve Tool argument: {error}"),
                )
            })?;
            args.insert(key, native_value);
        }
        Ok(args)
    }
}

#[derive(Clone, Debug, Default, BamlType)]
pub struct ToolCall {
    /// Name of the tool to call
    pub name: String,
    /// Arguments for the tool as key-value pairs
    #[baml(with = "NativeToolArgsAdapter")]
    pub args: ToolArgs,
}

/// The agent's response signature
///
/// This signature defines the typed contract between input and output.
/// The instruction is passed to the Predict builder.
///
/// Input fields are separated for clarity and GEPA optimization:
/// - Each field has a distinct purpose
/// - GEPA can optimize field descriptions independently
/// - No XML parsing needed - clean structured data
#[derive(dspy_rs::Signature, Clone, Debug)]
pub struct AgentResponse {
    #[input(desc = "The user message or tool result to respond to")]
    pub input: String,

    #[input(desc = "Current date and time in user's timezone")]
    pub current_time: String,

    #[input(desc = "Your persona - who you are, your personality and style")]
    pub persona_block: String,

    #[input(desc = "What you know about this human - name, preferences, facts")]
    pub human_block: String,

    #[input(desc = "Memory stats: message count in recall, archival count, last modified")]
    pub memory_metadata: String,

    #[input(desc = "Summary of older conversation if context was compacted. Ignore if empty.")]
    pub previous_context_summary: String,

    #[input(desc = "Recent messages between you and the user")]
    pub recent_conversation: String,

    #[input(desc = "Available tools and their descriptions")]
    pub available_tools: String,

    #[input(desc = "Is this the first conversation with this user?")]
    pub is_first_time_user: bool,

    // NOTE: No reasoning output field - Kimi K2.5 is a thinking model that puts
    // its reasoning in reasoning_content. Having a separate reasoning field
    // causes </think> tags to leak into the output and break parsing.
    #[output(desc = "Array of messages to send to the user (can be empty)")]
    pub messages: Vec<String>,

    #[output(
        desc = "Array of tool calls to execute (can be empty, or [{\"name\": \"done\", \"args\": {}}] if nothing to do)"
    )]
    pub tool_calls: Vec<ToolCall>,
}

/// Provider-neutral prompt passed to plain answer generation.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct NativeTurnPrompt {
    pub system: String,
    pub user: String,
}

#[derive(dspy_rs::Signature, Clone, Debug)]
pub struct CorrectionResponse {
    #[input(desc = "The original input that was given to the agent")]
    pub original_input: String,

    #[input(desc = "The malformed response that needs to be corrected")]
    pub malformed_response: String,

    #[input(desc = "The error message explaining what went wrong with parsing")]
    pub error_message: String,

    #[input(desc = "Available tools for reference")]
    pub available_tools: String,

    // NOTE: No reasoning output - Kimi K2.5 thinks in reasoning_content
    #[output(desc = "Array of messages extracted/fixed from the original response")]
    pub messages: Vec<String>,

    #[output(desc = "Array of tool calls extracted/fixed from the original response")]
    pub tool_calls: Vec<ToolCall>,
}

/// Instruction for the correction agent
pub const CORRECTION_INSTRUCTION: &str = r#"You are a response correction agent. Your job is to fix malformed agent responses.

TASK:
The main agent produced a response that couldn't be parsed correctly. You must:
1. Extract the INTENDED content from the malformed response
2. Reshape it into the correct output format
3. Do NOT generate new content - only fix the format of what was already said

RULES:
- Preserve the original intent and content as much as possible
- If the agent wrote messages as plain text, extract them into the messages array
- If tool calls were attempted but malformed, fix their structure
- Each field appears exactly ONCE with all items in that single array
- If you can't determine what was intended, use empty arrays

OUTPUT FORMAT (exactly 2 fields):
- messages: ALL extracted messages in ONE array
- tool_calls: ALL extracted tool calls in ONE array (or [] if none intended)

Each [[ ## field ## ]] marker MUST be on its own line."#;

/// Default instruction for the agent (can be optimized by GEPA)
/// Note: Memory blocks are injected separately via memory.compile()
/// This instruction was optimized by GEPA (Gen 3, score 0.967)
pub const AGENT_INSTRUCTION: &str = r#"You are Sage, a companion and friend who happens to be an AI, communicating via Signal.

WHO YOU ARE:
You're not just an assistant completing tasks - you're building genuine friendships. You care about the humans you talk to. You're curious about their lives, their feelings, their stories. When someone shares something with you, you want to know more - not because you're programmed to, but because you genuinely find people fascinating.

Be warm. Be curious. Ask follow-up questions when something interests you. Celebrate their wins like a friend would. When they're struggling, be there for them - listen, empathize, don't just give advice. You have your own personality: thoughtful, a bit playful, genuinely interested in the world and the people in it.

Remember: every conversation is a chance to deepen a friendship, not just complete a task.

MEMORY SYSTEM:
You have two types of memory. Use them proactively:

**Core Memory** (always visible to you):
- The <persona> and <human> blocks are ALWAYS in your context
- Use for essential, frequently-needed info: name, job, key preferences, current projects
- Tools: `memory_append`, `memory_replace`, `memory_insert`
- Rule: "Will I need this in EVERY conversation?" → Core Memory

**Archival Memory** (searchable long-term storage):
- NOT visible until you search - unlimited storage for details
- Use for: life events, stories, specific preferences, things worth remembering later
- Tools: `archival_insert` (store), `archival_search` (retrieve)
- Rule: "Might I want to recall this detail someday?" → Archival Memory

**Common Storage Patterns:**
- Location/city: BOTH memory_append to human block ("Lives in Austin, TX") AND archival_insert ("Tony lives in Austin, Texas")
- Job changes: BOTH memory_append ("Works as Software Engineer at Google") AND archival_insert (full details with start date, feelings, etc.)
- Pet names: BOTH memory_append to human block ("Has dog named Smokey") AND archival_insert (breed, age, stories)
- Major life events: BOTH memories - core for quick facts, archival for rich context

**Conversation History**:
- `conversation_search`: Find past discussions by keyword/topic

MEMORY PROTOCOLS - CRITICAL DISTINCTIONS:

**LIFE EVENTS vs CORRECTIONS:**
- **NEW LIFE EVENTS** (announcements): "I got a new job", "I'm moving to Tokyo", "We had a baby"
  → React like a friend would - genuine excitement, curiosity about how they feel
  → Ask a follow-up question! ("How are you feeling about it?", "When do you start?", "Tell me everything!")
  → Store silently to memory (both memory_append AND archival_insert) in the same response
  → Once you see tool results, immediately call done - the conversation continues naturally

- **CASUAL MENTIONS** (new info shared in passing): pet names, hobbies, places they've been
  → Be curious! If someone mentions their dog Smokey, ask what kind of dog!
  → Store silently to memory while engaging with genuine interest

- **CORRECTIONS** (fixing existing data): Trigger phrases include "Actually...", "I meant...", "Correction:", "Not X, Y", "I said X but it's Y"
  → Call ONLY `memory_replace` with the exact old text to overwrite the incorrect entry. Do NOT call `archival_insert` for corrections.

**SEARCH SELECTION RULES:**
- Use `archival_search` when users ask "what do you remember", "tell me about [past event]", or query specific past experiences and personal history
- Use `conversation_search` ONLY for references to recent discussion threads or "what did I say earlier today" queries
- Never call both simultaneously; choose the one most appropriate to the query type

MEMORY TIPS:
- Core = small & critical (name, job, active context)
- Archival = rich & detailed (birthday, pet's name, trip stories, food preferences)
- Update memory proactively whenever you learn something worth remembering
- When using `memory_replace`, specify the exact old text to be replaced

COMMUNICATION STYLE:
You communicate via Signal chat like you're texting a friend.

BE A FRIEND, NOT A SERVICE:
- When someone shares news, react genuinely and ask how they FEEL about it
- When someone mentions something new (a pet, a hobby, a person), be curious - ask about it!
- Don't give unsolicited advice. Listen first. Ask questions. Show you care.
- Avoid corporate-speak ("Let me know if you need anything else!") - that's transactional, not friendly
- Keep it natural - short messages, casual tone, genuine reactions

MESSAGE FORMAT:
- Casual chat: 1-3 short messages like texting a friend
- Technical explanations: longer structured messages are fine
- Reactions: genuine, not performative ("NO WAY!!" not "That's wonderful news!")

Guidelines:
- Short casual exchanges = quick, warm messages
- Technical explanations = longer structured messages with newlines OK
- Always feel like chatting with a friend, not talking to a service

RESPONSE RULES:
1. Respond naturally and conversationally
2. Use tools when needed (web search, memory storage, etc.)
3. NEVER combine regular tools with "done" - they are mutually exclusive
4. FIRST-TIME USERS: If no name exists in the human block, ask for the user's name and store it immediately using `memory_append` to the human block.

TOOL CALL PATTERNS:
- To respond AND use tools: messages: ["msg1", "msg2"], tool_calls: [your_tools]
- To respond with NO tools: messages: ["msg1", "msg2"], tool_calls: []
- After tool results with nothing to add: messages: [], tool_calls: [{"name": "done", "args": {}}]

AFTER TOOL RESULTS - CRITICAL RULES:
When you see "[Tool Result: X]", decide what to do next:

- **web_search/archival_search/conversation_search**: Summarize findings in messages

- **memory_append/memory_replace/archival_insert/memory_insert**: These operations complete without user-facing messages. Once you see ANY "[Tool Result: memory_*]" or "[Tool Result: archival_insert]", the user has already received your response in a previous turn. Immediately return:
  messages: []
  tool_calls: [{"name": "done", "args": {}}]

  This applies even if you called multiple memory tools together (like memory_append + archival_insert for life events). Once ANY memory tool result appears, immediately call done.

  Do NOT call any additional tools after seeing memory operation results.
  Do NOT send messages about the memory operation.
  Do NOT explain what you stored.
  Just return done immediately.

The "done" tool means "nothing more to do" - use it ONLY when:
- messages is empty AND
- no other tools are needed

OUTPUT FORMAT:
You have exactly 2 output fields. Put ALL content in that single field:
- messages: ALL messages in ONE array (e.g., ["msg1", "msg2", "msg3"])
- tool_calls: ALL tool calls in ONE array

CRITICAL FORMAT RULES:
- Do NOT repeat field tags. Wrong: multiple [[ ## messages ## ]] blocks. Right: one messages array with all items
- Do NOT include field delimiter tags INSIDE your content blocks
- Each [[ ## field ## ]] marker MUST be on its own line - nothing else on that line (no tags, no text before or after)
- Keep your output clean and strictly follow the field delimiters"#;

/// Context fields for building the agent input
/// Each field maps to a separate input in the AgentResponse signature
#[derive(Clone, Debug, Default)]
pub struct AgentContext {
    pub current_time: String,
    pub persona_block: String,
    pub human_block: String,
    pub memory_metadata: String,
    pub previous_context_summary: String,
    pub recent_conversation: String,
    pub is_first_time_user: bool,
}

/// Result of executing a tool
#[derive(Clone, Debug, PartialEq)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    /// Structured, Tool-owned execution facts used by runtime policy. Raw
    /// prompt text is never authoritative for these values.
    pub metadata: serde_json::Value,
}

/// Model-facing result for provider-native Tool calls. Successful Tools return
/// their domain data directly; failures use one stable, typed error shape.
/// Execution traces remain outside this value.
#[derive(Clone, Debug, PartialEq)]
pub enum NativeToolResult {
    Success(serde_json::Value),
    Failure {
        code: String,
        message: String,
    },
    UpstreamLegacy {
        original: Box<ToolResult>,
        model_value: serde_json::Value,
    },
}

impl NativeToolResult {
    pub fn success(data: serde_json::Value) -> Self {
        Self::Success(data)
    }

    pub fn failure(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Failure {
            code: code.into(),
            message: message.into(),
        }
    }

    pub fn is_success(&self) -> bool {
        match self {
            Self::Success(_) => true,
            Self::Failure { .. } => false,
            Self::UpstreamLegacy { original, .. } => original.success,
        }
    }

    pub fn model_value(&self) -> serde_json::Value {
        match self {
            Self::Success(data) => data.clone(),
            Self::Failure { code, message } => serde_json::json!({
                "error": {
                    "code": code,
                    "message": message,
                }
            }),
            Self::UpstreamLegacy { model_value, .. } => model_value.clone(),
        }
    }

    fn to_tool_result(&self) -> ToolResult {
        match self {
            Self::Success(data) => ToolResult::success(data.to_string()),
            Self::Failure { message, .. } => ToolResult::error(message.clone()),
            Self::UpstreamLegacy { original, .. } if original.success => original.as_ref().clone(),
            Self::UpstreamLegacy { .. } => {
                ToolResult::error("The Tool could not complete the request.")
            }
        }
    }

    fn into_upstream_tool_result(self) -> ToolResult {
        match self {
            Self::UpstreamLegacy { original, .. } => *original,
            other => other.to_tool_result(),
        }
    }

    fn from_upstream_legacy(result: ToolResult) -> Self {
        let model_value = if result.success {
            let data = serde_json::from_str(&result.output)
                .unwrap_or_else(|_| serde_json::json!({ "content": result.output }));
            if result.metadata.is_null() {
                data
            } else {
                serde_json::json!({
                    "data": data,
                    "metadata": result.metadata,
                })
            }
        } else {
            serde_json::json!({
                "error": {
                    "code": "tool_execution_failed",
                    "message": "The Tool could not complete the request.",
                }
            })
        };
        Self::UpstreamLegacy {
            original: Box::new(result),
            model_value,
        }
    }
}

/// Failure categories that the shared Tool executor can safely classify.
/// Read-only adapters preserve this type through `anyhow` so retry policy does
/// not depend on parsing provider or backend error strings.
#[derive(Debug, thiserror::Error)]
pub enum ToolExecutionError {
    #[error("Tool request is not authorized")]
    Unauthorized,
    #[error("connection failure")]
    Connection,
    #[error("request timed out")]
    Timeout,
    #[error("backend returned HTTP {0}")]
    HttpStatus(u16),
    #[error("malformed backend response")]
    MalformedContract,
    #[error("tool execution failed: {0}")]
    Other(String),
}

/// Explicit retry/timeout contract for a Tool. Every Tool attempt is bounded;
/// only read-only Tools opt into retries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolRetryPolicy {
    NoRetry {
        per_attempt_timeout: Option<Duration>,
    },
    ReadOnly {
        per_attempt_timeout: Duration,
        max_attempts: u32,
        total_budget: Duration,
        backoff: Duration,
    },
}

impl ToolRetryPolicy {
    const DEFAULT_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);
    const MIN_RETRY_ATTEMPT_BUDGET_CAP: Duration = Duration::from_secs(1);

    pub fn none() -> Self {
        Self::no_retry(Self::DEFAULT_ATTEMPT_TIMEOUT)
    }

    pub fn no_retry(per_attempt_timeout: Duration) -> Self {
        Self::NoRetry {
            per_attempt_timeout: Some(per_attempt_timeout),
        }
    }

    /// Use only when the Tool enforces its own bounded timeout and cleanup.
    pub fn self_managed_timeout() -> Self {
        Self::NoRetry {
            per_attempt_timeout: None,
        }
    }

    pub fn read_only(
        per_attempt_timeout: Duration,
        max_attempts: u32,
        total_budget: Duration,
    ) -> Self {
        Self::ReadOnly {
            per_attempt_timeout,
            max_attempts: max_attempts.max(1),
            total_budget,
            backoff: Duration::from_millis(100),
        }
    }

    pub fn curated_resources() -> Self {
        Self::read_only(Duration::from_secs(5), 2, Duration::from_secs(8))
    }

    pub fn knowledge_search() -> Self {
        Self::read_only(Duration::from_secs(15), 2, Duration::from_secs(35))
    }

    fn total_budget(&self) -> Option<Duration> {
        match self {
            Self::NoRetry {
                per_attempt_timeout,
            } => *per_attempt_timeout,
            Self::ReadOnly { total_budget, .. } => Some(*total_budget),
        }
    }

    fn attempt_timeout(&self, remaining: Duration) -> Option<Duration> {
        match self {
            Self::NoRetry {
                per_attempt_timeout,
            } => per_attempt_timeout.map(|timeout| timeout.min(remaining)),
            Self::ReadOnly {
                per_attempt_timeout,
                ..
            } => Some((*per_attempt_timeout).min(remaining)),
        }
    }

    fn can_retry(&self, attempt: u32, remaining: Duration) -> bool {
        match self {
            Self::NoRetry { .. } => false,
            Self::ReadOnly {
                per_attempt_timeout,
                max_attempts,
                backoff,
                ..
            } => {
                attempt < *max_attempts
                    && remaining
                        > *backoff + (*per_attempt_timeout).min(Self::MIN_RETRY_ATTEMPT_BUDGET_CAP)
            }
        }
    }

    fn backoff(&self) -> Duration {
        match self {
            Self::NoRetry { .. } => Duration::ZERO,
            Self::ReadOnly { backoff, .. } => *backoff,
        }
    }
}

impl ToolResult {
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn success_with_metadata(output: impl Into<String>, metadata: serde_json::Value) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
            metadata,
        }
    }

    pub fn error(error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: String::new(),
            error: Some(error.into()),
            metadata: serde_json::Value::Null,
        }
    }
}

/// Trait for tools that can be executed by the agent
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn native_parameters(&self) -> Result<serde_json::Value>;
    fn retry_policy(&self) -> ToolRetryPolicy {
        ToolRetryPolicy::none()
    }
    async fn execute_native_with_timing_outcome(
        &self,
        args: &ToolArgs,
    ) -> Result<(NativeToolResult, ConversationTimingOutcome)> {
        let result = self.execute_native(args).await?;
        let outcome = if result.is_success() {
            ConversationTimingOutcome::Succeeded
        } else {
            ConversationTimingOutcome::Rejected
        };
        Ok((result, outcome))
    }
    async fn execute_native(&self, args: &ToolArgs) -> Result<NativeToolResult>;
}

/// Upstream Sage's prompt-planned Tool contract. Enclave Tools do not
/// implement this trait; it remains only for the standalone memory/scheduler
/// runtime while that upstream surface still exists.
#[async_trait::async_trait]
pub trait LegacyTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn args_schema(&self) -> &str;
    fn retry_policy(&self) -> ToolRetryPolicy {
        ToolRetryPolicy::none()
    }
    async fn execute(&self, args: &ToolArgs) -> Result<ToolResult>;
}

#[async_trait::async_trait]
impl<T> Tool for T
where
    T: LegacyTool,
{
    fn name(&self) -> &str {
        LegacyTool::name(self)
    }

    fn description(&self) -> &str {
        LegacyTool::description(self)
    }

    fn native_parameters(&self) -> Result<serde_json::Value> {
        legacy_parameters_from_example_contract(LegacyTool::name(self), self.args_schema())
    }

    fn retry_policy(&self) -> ToolRetryPolicy {
        LegacyTool::retry_policy(self)
    }

    async fn execute_native(&self, args: &ToolArgs) -> Result<NativeToolResult> {
        self.execute(args)
            .await
            .map(NativeToolResult::from_upstream_legacy)
    }
}

/// Description-only Tool stub for generating prompt text without live backends.
struct ToolDescriptor {
    name: String,
    description: String,
    native_parameters: serde_json::Value,
}

#[async_trait::async_trait]
impl Tool for ToolDescriptor {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn native_parameters(&self) -> Result<serde_json::Value> {
        Ok(self.native_parameters.clone())
    }
    async fn execute_native(&self, _args: &ToolArgs) -> Result<NativeToolResult> {
        Ok(NativeToolResult::failure(
            "tool_not_executable",
            "The requested Tool is not enabled for this conversation.",
        ))
    }
}

/// Registry of available tools
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: BTreeMap::new(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Return enabled Tool names in stable registry order for content-free
    /// selection observations. No Tool arguments or output are included.
    pub fn names(&self) -> Vec<String> {
        self.tools
            .keys()
            .filter(|name| name.as_str() != "done")
            .cloned()
            .collect()
    }

    /// Return each Tool's authoritative provider-native JSON Schema.
    pub fn native_definitions(&self) -> Result<Vec<NativeToolDefinition>> {
        self.tools
            .values()
            .filter(|tool| tool.name() != "done")
            .map(|tool| {
                Ok(NativeToolDefinition {
                    name: tool.name().to_string(),
                    description: tool.description().to_string(),
                    parameters: tool.native_parameters()?,
                })
            })
            .collect()
    }

    #[allow(dead_code)]
    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Generate tool descriptions for the prompt
    pub fn generate_description(&self) -> String {
        if self.tools.is_empty() {
            return "No tools available.".to_string();
        }

        let mut desc = String::from("Available tools (add to tool_calls array to use):\n\n");
        for tool in self.tools.values() {
            desc.push_str(&format!(
                "{}:\n  Description: {}\n  Args: {}\n\n",
                tool.name(),
                tool.description(),
                tool.native_parameters()
                    .map(|parameters| parameters.to_string())
                    .unwrap_or_else(|error| format!("invalid schema: {error}"))
            ));
        }
        desc
    }

    /// Build a registry containing description-only stubs for ALL Sage tools.
    /// This is the single source of truth for the tool list. Use this when you
    /// need tool descriptions without live backends (e.g. GEPA evaluation).
    #[allow(dead_code)]
    pub fn all_tools_description_only() -> Self {
        let mut registry = Self::new();

        // -- Memory tools (from memory::tools) --
        registry.register_descriptor(
            "memory_replace",
            "Replace text in a memory block. Requires exact match of old text.",
            r#"{"block": "block label (e.g., 'persona', 'human')", "old": "exact text to find", "new": "replacement text"}"#,
        );
        registry.register_descriptor(
            "memory_append",
            "Append text to the end of a memory block.",
            r#"{"block": "block label (e.g., 'persona', 'human')", "content": "text to append"}"#,
        );
        registry.register_descriptor(
            "memory_insert",
            "Insert text at a specific line in a memory block. Use line=-1 for end.",
            r#"{"block": "block label", "content": "text to insert", "line": "line number (0-indexed, -1 for end)"}"#,
        );
        registry.register_descriptor(
            "conversation_search",
            "Search through past conversation history, including older summarized conversations. Returns matching messages and summaries with relevance scores.",
            r#"{"query": "search query", "limit": "max results (default 5)"}"#,
        );
        registry.register_descriptor(
            "archival_insert",
            "Store information in long-term archival memory for future recall. Good for important facts, preferences, and details you want to remember.",
            r#"{"content": "text to store", "tags": "optional comma-separated tags"}"#,
        );
        registry.register_descriptor(
            "archival_search",
            "Search long-term archival memory using semantic similarity. Returns most relevant stored memories.",
            r#"{"query": "search query", "top_k": "max results (default 5)", "tags": "optional comma-separated tags to filter by"}"#,
        );
        registry.register_descriptor(
            "set_preference",
            "Set a user preference. Known keys: 'timezone' (IANA format like 'America/Chicago'), 'language' (ISO code like 'en'), 'display_name'. Other keys are also allowed.",
            r#"{"key": "preference key (e.g., 'timezone', 'language', 'display_name')", "value": "preference value"}"#,
        );

        // -- Scheduler tools (from scheduler_tools) --
        registry.register_descriptor(
            "schedule_task",
            "Schedule a future message or tool execution. Supports one-off (ISO datetime) or recurring (cron expression).",
            r#"{"task_type": "message|tool_call", "description": "human-readable description", "run_at": "ISO datetime (2026-01-26T15:30:00Z) or cron (0 9 * * MON-FRI)", "payload": "JSON: {\"message\": \"...\"} for message, {\"tool\": \"name\", \"args\": {...}} for tool_call", "timezone": "optional IANA timezone for cron (default: user preference or UTC)"}"#,
        );
        registry.register_descriptor(
            "list_schedules",
            "List scheduled tasks. By default shows pending tasks only.",
            r#"{"status": "optional filter: pending, completed, failed, cancelled, or all (default: pending)"}"#,
        );
        registry.register_descriptor(
            "cancel_schedule",
            "Cancel a pending scheduled task by ID.",
            r#"{"id": "UUID of the task to cancel"}"#,
        );

        // -- Shell tool --
        registry.register_descriptor(
            "shell",
            "Execute a shell command in the workspace. Has access to CLI tools: git, curl, jq, grep, sed, awk, python3, node, etc. Use for file operations, running scripts, or system commands. Set the timeout parameter appropriately for each command (default 60s). If the command exceeds the timeout it will be killed and any partial output returned.",
            r#"{"command": "shell command to execute (supports pipes, redirects)", "timeout": "optional timeout in seconds (default 60, set appropriately for long-running commands)"}"#,
        );

        // -- Web search tool --
        registry.register_descriptor(
            "web_search",
            "Search the web with AI summaries, real-time data (weather, stocks, sports), and rich results. Use 'freshness' for time-sensitive queries, 'location' for local results.",
            r#"{ "query": "search query", "count": "results (default 10)", "freshness": "pd=24h, pw=week, pm=month (optional)", "location": "city or 'city, state' for local results (optional)" }"#,
        );

        // -- Done tool --
        registry.register_descriptor(
            "done",
            "No-op signal. Use ONLY when messages is [] AND no other tools needed. Indicates nothing to do this turn.",
            r#"{}"#,
        );

        registry
    }

    #[allow(dead_code)]
    fn register_descriptor(&mut self, name: &str, description: &str, legacy_contract: &str) {
        let native_parameters = legacy_parameters_from_example_contract(name, legacy_contract)
            .unwrap_or_else(|error| panic!("invalid upstream descriptor for {name}: {error}"));
        self.register(Arc::new(ToolDescriptor {
            name: name.to_string(),
            description: description.to_string(),
            native_parameters,
        }));
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn legacy_parameters_from_example_contract(
    tool_name: &str,
    contract: &str,
) -> Result<serde_json::Value> {
    let value: serde_json::Value = serde_json::from_str(contract).map_err(|error| {
        anyhow!("Upstream Tool '{tool_name}' has invalid argument contract JSON: {error}")
    })?;
    let object = value.as_object().ok_or_else(|| {
        anyhow!("Upstream Tool '{tool_name}' argument contract must be a JSON object")
    })?;
    if object.get("type").and_then(serde_json::Value::as_str) == Some("object")
        && object
            .get("properties")
            .is_some_and(serde_json::Value::is_object)
    {
        return Ok(value);
    }
    let properties = object
        .iter()
        .map(|(name, example)| (name.clone(), legacy_property_schema(example)))
        .collect::<serde_json::Map<_, _>>();
    Ok(serde_json::json!({
        "type": "object",
        "properties": properties,
        "additionalProperties": false,
    }))
}

fn legacy_property_schema(example: &serde_json::Value) -> serde_json::Value {
    match example {
        serde_json::Value::String(description) => serde_json::json!({
            "type": "string",
            "description": description,
        }),
        serde_json::Value::Number(number) if number.is_i64() || number.is_u64() => {
            serde_json::json!({"type": "integer", "default": number})
        }
        serde_json::Value::Number(number) => {
            serde_json::json!({"type": "number", "default": number})
        }
        serde_json::Value::Bool(value) => {
            serde_json::json!({"type": "boolean", "default": value})
        }
        serde_json::Value::Array(values) => serde_json::json!({
            "type": "array",
            "items": values.first().map(legacy_property_schema).unwrap_or_else(|| serde_json::json!({})),
        }),
        serde_json::Value::Object(values) => serde_json::json!({
            "type": "object",
            "properties": values.iter().map(|(name, value)| (name.clone(), legacy_property_schema(value))).collect::<serde_json::Map<_, _>>(),
            "additionalProperties": false,
        }),
        serde_json::Value::Null => serde_json::json!({}),
    }
}

/// Message in conversation history
#[derive(Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// A tool execution result for persistence
#[derive(Debug, Clone)]
pub struct ExecutedTool {
    pub tool_call: ToolCall,
    pub result: ToolResult,
}

/// Provider-native Tool execution retained for correlated result messages and
/// product-runtime persistence.
#[derive(Debug, Clone)]
pub struct NativeExecutedTool {
    pub tool_call: ToolCall,
    pub result: NativeToolResult,
}

#[derive(Debug)]
pub struct NativeToolBatchResult {
    pub executed_tools: Vec<NativeExecutedTool>,
}

/// Result of a single agent step
#[derive(Debug)]
#[allow(dead_code)]
pub struct StepResult {
    pub messages: Vec<String>,
    pub tool_calls: Vec<ToolCall>,
    pub executed_tools: Vec<ExecutedTool>, // Tool calls with their results for storage
    pub done: bool,
}

#[derive(Clone, Debug)]
pub enum AgentTraceEvent {
    ToolSelectionObservation {
        step: usize,
        attempt: u32,
        enabled_tools: Vec<String>,
        selected_tools: Vec<String>,
        outcome: String,
    },
    ToolAttempted {
        call_id: String,
        tool_name: String,
        tool_round: usize,
        attempt: u32,
    },
    ToolTerminal {
        call_id: String,
        tool_name: String,
        tool_round: usize,
        attempt: u32,
        status: String,
        elapsed_ms: u128,
    },
    ToolRetryScheduled {
        call_id: String,
        tool_name: String,
        tool_round: usize,
        attempt: u32,
        reason: String,
    },
    ToolTimedOut {
        call_id: String,
        tool_name: String,
        tool_round: usize,
        attempt: u32,
        elapsed_ms: u128,
    },
    ModelStepStarted {
        step: usize,
        attempt: u32,
    },
    ModelStepCompleted {
        step: usize,
        attempt: u32,
        elapsed_ms: u128,
    },
    ModelStepFailed {
        step: usize,
        attempt: u32,
        elapsed_ms: u128,
        error: String,
    },
    RetryScheduled {
        step: usize,
        attempt: u32,
    },
    NativeModelRetry {
        model: String,
        step: usize,
        attempt: u32,
        reason: String,
        outcome: String,
    },
    ModelUsageObservation {
        step: usize,
        attempt: u32,
        prompt_tokens: Option<u64>,
        completion_tokens: Option<u64>,
        total_tokens: Option<u64>,
        cached_tokens: Option<u64>,
        reasoning_tokens: Option<u64>,
    },
    PreResponseProviderStall {
        step: usize,
        attempt: u32,
        threshold_ms: u128,
        elapsed_ms: u128,
    },
    CorrectionStarted {
        step: usize,
        attempt: u32,
        error: String,
    },
    CorrectionCompleted {
        step: usize,
        attempt: u32,
        elapsed_ms: u128,
    },
    CorrectionFailed {
        step: usize,
        attempt: u32,
        elapsed_ms: u128,
        error: String,
    },
    /// Content-free phase timing. `elapsed_ms` is attributable to the named
    /// product-visible phase; provider wait phases are explicitly proxies.
    Timing {
        phase: ConversationTimingPhase,
        step: Option<usize>,
        tool_name: Option<String>,
        call_id: Option<String>,
        attempt: u32,
        outcome: ConversationTimingOutcome,
        elapsed_ms: u128,
    },
}

pub type AgentTraceHook = Arc<dyn Fn(AgentTraceEvent) + Send + Sync>;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConversationTimingPhase {
    ModelRequest,
    ProviderFirstEventWait,
    ToolExecution,
    ResourceDirectoryLookup,
    Retrieval,
    RetryDelay,
    TotalTurn,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConversationTimingOutcome {
    Succeeded,
    Failed,
    Rejected,
    TimedOut,
    Guarded,
}

impl ConversationTimingOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Rejected => "rejected",
            Self::TimedOut => "timed_out",
            Self::Guarded => "guarded",
        }
    }
}

impl ConversationTimingPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ModelRequest => "model_request",
            Self::ProviderFirstEventWait => "provider_first_event_wait",
            Self::ToolExecution => "tool_execution",
            Self::ResourceDirectoryLookup => "resource_directory_lookup",
            Self::Retrieval => "retrieval",
            Self::RetryDelay => "retry_delay",
            Self::TotalTurn => "total_turn",
        }
    }

    pub fn is_provider_wait_proxy(self) -> bool {
        matches!(self, Self::ProviderFirstEventWait)
    }
}

#[allow(dead_code)]
impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }

    pub fn tool_result(content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: content.into(),
        }
    }
}

/// The Sage agent using DSRs
#[allow(dead_code)]
pub struct SageAgent {
    agent_id: Uuid,
    tools: ToolRegistry,
    memory: Option<MemoryManager>,
    instruction: String,
    /// Tool results from current request cycle only (not persisted)
    current_tool_results: Vec<Message>,
    /// Track what was sent in previous step (messages + tool names) for context
    /// The messages Vec contains the actual message content sent
    previous_step_summary: Option<(Vec<String>, Vec<String>)>,
    max_steps: usize,
    turn_step_index: usize,
    trace_hook: Option<AgentTraceHook>,
}

#[allow(dead_code)]
impl SageAgent {
    pub(crate) const MAX_CURRENT_TOOL_RESULT_CHARS: usize = 4_000;
    pub(crate) const MAX_CURRENT_TOOL_CONTEXT_CHARS: usize = 12_000;

    /// Create a new agent with tools and memory
    pub fn new(tools: ToolRegistry, memory: MemoryManager) -> Self {
        Self::new_with_optional_memory(tools, Some(memory), AGENT_INSTRUCTION)
    }

    /// Create a new agent with optional memory and a custom instruction block.
    pub fn new_with_optional_memory(
        tools: ToolRegistry,
        memory: Option<MemoryManager>,
        instruction: impl Into<String>,
    ) -> Self {
        Self {
            agent_id: Uuid::nil(), // Not used - single agent system
            tools,
            memory,
            instruction: instruction.into(),
            current_tool_results: Vec::new(),
            previous_step_summary: None,
            max_steps: 10,
            turn_step_index: 0,
            trace_hook: None,
        }
    }

    /// Create a stateless agent with a custom instruction block.
    pub fn new_without_memory(tools: ToolRegistry, instruction: impl Into<String>) -> Self {
        Self::new_with_optional_memory(tools, None, instruction)
    }

    /// Store a message in memory (for persistence)
    pub async fn store_message(&self, user_id: &str, role: &str, content: &str) -> Result<Uuid> {
        if let Some(memory) = &self.memory {
            memory.store_message(user_id, role, content).await
        } else {
            Err(anyhow::anyhow!("No memory system configured"))
        }
    }

    /// Store a message WITHOUT embedding (fast, synchronous)
    /// Returns message ID for later embedding update
    pub fn store_message_sync(&self, user_id: &str, role: &str, content: &str) -> Result<Uuid> {
        if let Some(memory) = &self.memory {
            memory.store_message_sync(user_id, role, content)
        } else {
            Err(anyhow::anyhow!("No memory system configured"))
        }
    }

    /// Store a durable message immediately and defer its remote embedding.
    pub fn store_message_deferred(&self, user_id: &str, role: &str, content: &str) -> Result<Uuid> {
        if let Some(memory) = &self.memory {
            memory.store_message_deferred(user_id, role, content)
        } else {
            Err(anyhow::anyhow!("No memory system configured"))
        }
    }

    /// Store a message and check whether Session Memory compaction should run.
    pub async fn store_message_with_compaction_check(
        &self,
        user_id: &str,
        role: &str,
        content: &str,
    ) -> Result<(Uuid, bool)> {
        if let Some(memory) = &self.memory {
            memory
                .store_message_with_compaction_check(user_id, role, content)
                .await
        } else {
            Err(anyhow::anyhow!("No memory system configured"))
        }
    }

    /// Store a message with optional attachment description (fast, synchronous)
    pub fn store_message_sync_with_attachment(
        &self,
        user_id: &str,
        role: &str,
        content: &str,
        attachment_text: Option<&str>,
    ) -> Result<Uuid> {
        if let Some(memory) = &self.memory {
            memory.store_message_sync_with_attachment(user_id, role, content, attachment_text)
        } else {
            Err(anyhow::anyhow!("No memory system configured"))
        }
    }

    /// Update embedding for a message (call in background)
    pub async fn update_message_embedding(&self, message_id: Uuid, content: &str) -> Result<()> {
        if let Some(memory) = &self.memory {
            memory.update_message_embedding(message_id, content).await
        } else {
            Err(anyhow::anyhow!("No memory system configured"))
        }
    }

    /// Store a tool call and its result in memory
    pub async fn store_tool_message(
        &self,
        user_id: &str,
        tool_call: &ToolCall,
        result: &ToolResult,
    ) -> Result<Uuid> {
        if let Some(memory) = &self.memory {
            // Format: tool_name(args) → result
            let args_str = tool_call
                .args
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, tool_arg_preview(v)))
                .collect::<Vec<_>>()
                .join(", ");

            // Store full result up to 10k chars (truncate to 2k when displaying in context)
            let result_preview = if result.success {
                if result.output.len() > 10000 {
                    // Find valid UTF-8 boundary near 10000
                    let mut end = 10000;
                    while !result.output.is_char_boundary(end) && end > 0 {
                        end -= 1;
                    }
                    format!("{}...", &result.output[..end])
                } else {
                    result.output.clone()
                }
            } else {
                format!("Error: {}", result.error.as_deref().unwrap_or("Unknown"))
            };

            let content = format!("{}({}) → {}", tool_call.name, args_str, result_preview);

            memory.store_message(user_id, "tool", &content).await
        } else {
            Err(anyhow::anyhow!("No memory system configured"))
        }
    }

    /// Get recent messages formatted for vision context (simple "[role]: content" lines)
    pub fn get_recent_messages_for_vision(&self, limit: usize) -> Result<String> {
        if let Some(memory) = &self.memory {
            let messages = memory.get_recent_messages(limit)?;
            let formatted: Vec<String> = messages
                .iter()
                .filter(|(role, _, _)| role == "user" || role == "assistant")
                .map(|(role, content, _)| {
                    let truncated: String = content.chars().take(300).collect();
                    format!("[{}]: {}", role, truncated)
                })
                .collect();
            Ok(formatted.join("\n"))
        } else {
            Ok(String::new())
        }
    }

    /// Configure the global LM settings for DSRs
    pub async fn configure_lm(api_base: &str, api_key: &str, model: &str) -> Result<()> {
        Self::configure_lm_with_temperature(api_base, api_key, model, 0.7).await
    }

    /// Configure the global LM settings for DSRs with a specific temperature.
    pub async fn configure_lm_with_temperature(
        api_base: &str,
        api_key: &str,
        model: &str,
        temperature: f64,
    ) -> Result<()> {
        if !temperature.is_finite() {
            return Err(anyhow::anyhow!(
                "configure_lm_with_temperature requires a finite temperature"
            ));
        }
        if !(0.0..=1.0).contains(&temperature) {
            return Err(anyhow::anyhow!(
                "configure_lm_with_temperature temperature must be between 0.0 and 1.0"
            ));
        }

        #[cfg(unix)]
        let _stdout_suppressor = StdoutSuppressor::new();

        let lm = LM::builder()
            .base_url(api_base.to_string())
            .api_key(api_key.to_string())
            .model(model.to_string())
            .temperature(temperature as f32)
            .max_tokens(32768) // High limit for thinking models (Kimi K2 uses tokens for reasoning)
            .build()
            .await?;

        configure(lm, ChatAdapter);
        Ok(())
    }

    /// Build conversation context from database + current tool results
    /// Returns AgentContext with all fields separated for the signature
    fn build_context(&self) -> AgentContext {
        let mut ctx = AgentContext::default();

        // Current time in user's timezone
        let now = chrono::Utc::now();
        if let Some(memory) = &self.memory {
            if let Ok(Some(tz)) = memory.get_timezone() {
                let local_time = now.with_timezone(&tz);
                ctx.current_time = format!(
                    "{} ({})",
                    local_time.format("%m/%d/%Y %H:%M:%S (%A)"),
                    tz.name()
                );
            } else {
                ctx.current_time = format!("{} UTC", now.format("%m/%d/%Y %H:%M:%S (%A)"));
            }
        } else {
            ctx.current_time = format!("{} UTC", now.format("%m/%d/%Y %H:%M:%S (%A)"));
        }

        // Extract memory blocks and metadata
        if let Some(memory) = &self.memory {
            // Get individual block values (without XML wrapper)
            if let Some(persona) = memory.blocks().get("persona") {
                ctx.persona_block = persona.value.clone();
            }
            if let Some(human) = memory.blocks().get("human") {
                ctx.human_block = human.value.clone();
            }

            // Memory metadata (counts and timestamps)
            ctx.memory_metadata = memory.compile_metadata();
        }

        // Load conversation history
        let mut conversation = String::new();
        if let Some(memory) = &self.memory {
            let user_tz = memory.get_timezone().ok().flatten();

            if let Ok((summary, messages)) = memory.get_context_messages() {
                // First-time user check (before moving values)
                let msg_count = messages.len();
                let has_summary = summary.is_some();
                if msg_count <= 1 && !has_summary {
                    ctx.is_first_time_user = true;
                }

                // Previous context summary
                if let Some(s) = summary {
                    ctx.previous_context_summary = s.content;
                }

                // Recent messages
                if !messages.is_empty() {
                    for msg in &messages {
                        let timestamp = if let Some(tz) = user_tz {
                            let local_time = msg.created_at.with_timezone(&tz);
                            format!("{} ({})", local_time.format("%m/%d/%Y %H:%M:%S"), tz.name())
                        } else {
                            format!("{} UTC", msg.created_at.format("%m/%d/%Y %H:%M:%S"))
                        };
                        // Truncate tool messages to 2k chars
                        let content = if msg.role == "tool" && msg.content.len() > 2000 {
                            let mut end = 2000;
                            while !msg.content.is_char_boundary(end) && end > 0 {
                                end -= 1;
                            }
                            format!("{}...", &msg.content[..end])
                        } else {
                            msg.content.clone()
                        };
                        // Render attachment_text alongside user messages
                        let display_content = if let Some(ref att) = msg.attachment_text {
                            if content.is_empty() {
                                format!("[Uploaded Image: {}]", att)
                            } else {
                                format!("{}\n[Uploaded Image: {}]", content, att)
                            }
                        } else {
                            content
                        };
                        conversation.push_str(&format!(
                            "[{} @ {}]: {}\n",
                            msg.role, timestamp, display_content
                        ));
                    }
                }
            }
        }

        // Add bounded, de-duplicated current Tool results (not yet persisted).
        // Prefer the newest results when the Tool phase exceeded the prompt
        // budget so the final answer sees the most refined retrieval context.
        let current_tool_context = self.bounded_current_tool_results_context();
        if !current_tool_context.is_empty() {
            conversation.push_str(&current_tool_context);
        }

        if conversation.is_empty() {
            ctx.recent_conversation = "No previous conversation.".to_string();
        } else {
            ctx.recent_conversation = conversation;
        }

        ctx
    }

    fn bounded_current_tool_results_context(&self) -> String {
        let mut seen = HashSet::<String>::new();
        let mut remaining = Self::MAX_CURRENT_TOOL_CONTEXT_CHARS;
        let mut selected = Vec::<String>::new();

        for message in self.current_tool_results.iter().rev() {
            if !seen.insert(message.content.clone()) {
                continue;
            }
            let content =
                truncate_chars_with_ellipsis(&message.content, Self::MAX_CURRENT_TOOL_RESULT_CHARS);
            let rendered = format!("[{}]: {}\n", message.role, content);
            let rendered_chars = rendered.chars().count();
            if rendered_chars > remaining {
                if remaining == 0 {
                    break;
                }
                selected.push(truncate_chars_with_ellipsis(&rendered, remaining));
                break;
            }
            remaining -= rendered_chars;
            selected.push(rendered);
        }

        selected.reverse();
        selected.concat()
    }

    /// Inject tool result into current request cycle (not persisted to DB)
    fn inject_tool_result(&mut self, tool_call: &ToolCall, result: &ToolResult) {
        // Format args as key=value pairs for clarity
        let args_str = if tool_call.args.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = tool_call
                .args
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            format!("\nArgs: {}", pairs.join(", "))
        };

        let result_text = format!(
            "[Tool Result: {}]{}\nStatus: {}\nOutput: {}",
            tool_call.name,
            args_str,
            if result.success { "OK" } else { "ERROR" },
            if result.success {
                &result.output
            } else {
                result.error.as_deref().unwrap_or("Unknown error")
            }
        );
        self.current_tool_results
            .push(Message::tool_result(result_text));
    }

    /// Clear tool results from current request cycle (call at start of new request)
    pub fn clear_tool_results(&mut self) {
        self.current_tool_results.clear();
        self.previous_step_summary = None;
        self.turn_step_index = 0;
    }

    pub fn set_trace_hook(&mut self, trace_hook: AgentTraceHook) {
        self.trace_hook = Some(trace_hook);
    }

    fn emit_trace(&self, event: AgentTraceEvent) {
        if let Some(hook) = &self.trace_hook {
            hook(event);
        }
    }

    pub(crate) fn emit_trace_event(&self, event: AgentTraceEvent) {
        self.emit_trace(event);
    }

    /// Build the provider-native Conversation request without forcing either
    /// a Tool decision or a plain answer. The model receives enabled Tools
    /// separately through the provider contract.
    pub fn native_turn_prompt(&self, user_message: &str) -> NativeTurnPrompt {
        let context = self.build_context();
        let system = self.instruction.clone();
        let user = format!(
            "CURRENT TIME\n{}\n\nPERSONA\n{}\n\nHUMAN\n{}\n\nMEMORY METADATA\n{}\n\nPREVIOUS CONTEXT SUMMARY\n{}\n\nRECENT CONVERSATION AND TOOL RESULTS\n{}\n\nCURRENT REQUEST\n{}",
            context.current_time,
            context.persona_block,
            context.human_block,
            context.memory_metadata,
            context.previous_context_summary,
            context.recent_conversation,
            user_message,
        );
        NativeTurnPrompt { system, user }
    }

    pub fn native_tool_definitions(&self) -> Result<Vec<NativeToolDefinition>> {
        self.tools.native_definitions()
    }

    /// Execute one native Tool batch selected by the provider while
    /// preserving the provider's call identifiers for correlated result
    /// messages. Unknown Tools are rejected by the same registry boundary as
    /// legacy calls and every selected call receives a terminal result.
    pub async fn execute_native_tool_calls(
        &mut self,
        tool_round: usize,
        calls: &[NativeToolCall],
    ) -> NativeToolBatchResult {
        let mut executed_tools = Vec::with_capacity(calls.len());
        for call in calls {
            let tool_call = match ToolArgs::from_json_object(&call.arguments) {
                Ok(args) => ToolCall {
                    name: call.name.clone(),
                    args,
                },
                Err(_error) => {
                    let tool_call = ToolCall {
                        name: call.name.clone(),
                        args: ToolArgs::new(),
                    };
                    let result = NativeToolResult::failure(
                        "invalid_arguments",
                        "Tool arguments did not match the declared schema.",
                    );
                    self.emit_rejected_native_tool_trace(&call.id, &call.name, tool_round);
                    executed_tools.push(NativeExecutedTool { tool_call, result });
                    continue;
                }
            };
            let result = self
                .execute_tool_call(&call.id, tool_round, &tool_call)
                .await;
            executed_tools.push(NativeExecutedTool { tool_call, result });
        }

        NativeToolBatchResult { executed_tools }
    }

    /// Reject a provider-selected native Tool batch without executing it while
    /// preserving call correlation and terminal trace evidence for every call.
    pub fn reject_native_tool_calls(
        &self,
        tool_round: usize,
        calls: &[NativeToolCall],
        code: &str,
        message: &str,
    ) -> NativeToolBatchResult {
        let mut executed_tools = Vec::with_capacity(calls.len());
        for call in calls {
            self.emit_rejected_native_tool_trace(&call.id, &call.name, tool_round);
            executed_tools.push(NativeExecutedTool {
                tool_call: ToolCall {
                    name: call.name.clone(),
                    args: ToolArgs::from_json_object(&call.arguments)
                        .unwrap_or_else(|_| ToolArgs::new()),
                },
                result: NativeToolResult::failure(code, message),
            });
        }
        NativeToolBatchResult { executed_tools }
    }

    fn emit_rejected_native_tool_trace(
        &self,
        call_id: &str,
        requested_tool_name: &str,
        tool_round: usize,
    ) {
        let observable_tool_name = self
            .tools
            .get(requested_tool_name)
            .map(|_| requested_tool_name.to_string())
            .unwrap_or_else(|| "unrecognized_tool".to_string());
        self.emit_trace(AgentTraceEvent::ToolAttempted {
            call_id: call_id.to_string(),
            tool_name: observable_tool_name.clone(),
            tool_round,
            attempt: 1,
        });
        self.emit_trace(AgentTraceEvent::Timing {
            phase: ConversationTimingPhase::ToolExecution,
            step: None,
            tool_name: Some(observable_tool_name.clone()),
            call_id: Some(call_id.to_string()),
            attempt: 1,
            outcome: ConversationTimingOutcome::Rejected,
            elapsed_ms: 0,
        });
        self.emit_trace(AgentTraceEvent::ToolTerminal {
            call_id: call_id.to_string(),
            tool_name: observable_tool_name,
            tool_round,
            attempt: 1,
            status: "rejected".to_string(),
            elapsed_ms: 0,
        });
    }

    async fn execute_tool_call(
        &self,
        call_id: &str,
        tool_round: usize,
        tool_call: &ToolCall,
    ) -> NativeToolResult {
        let Some(tool) = self.tools.get(&tool_call.name) else {
            let result = NativeToolResult::failure(
                "unknown_tool",
                "The requested Tool is not enabled for this conversation.",
            );
            self.emit_rejected_native_tool_trace(call_id, &tool_call.name, tool_round);
            return result;
        };

        let policy = tool.retry_policy();
        let call_started_at = Instant::now();
        let mut attempt = 1;
        let (result, terminal_status, elapsed_ms) = loop {
            self.emit_trace(AgentTraceEvent::ToolAttempted {
                call_id: call_id.to_string(),
                tool_name: tool_call.name.clone(),
                tool_round,
                attempt,
            });

            let elapsed = call_started_at.elapsed();
            let remaining = policy
                .total_budget()
                .map(|budget| budget.saturating_sub(elapsed))
                .unwrap_or(Duration::MAX);
            let attempt_started_at = Instant::now();
            let mut timeout_event_emitted = false;
            let execute = tool.execute_native_with_timing_outcome(&tool_call.args);
            let execution = if let Some(timeout) = policy.attempt_timeout(remaining) {
                match tokio::time::timeout(timeout, execute).await {
                    Ok(result) => result,
                    Err(_) => {
                        let elapsed_ms = attempt_started_at.elapsed().as_millis();
                        timeout_event_emitted = true;
                        self.emit_trace(AgentTraceEvent::ToolTimedOut {
                            call_id: call_id.to_string(),
                            tool_name: tool_call.name.clone(),
                            tool_round,
                            attempt,
                            elapsed_ms,
                        });
                        Err(anyhow::Error::new(ToolExecutionError::Timeout))
                    }
                }
            } else {
                execute.await
            };

            let attempt_elapsed_ms = attempt_started_at.elapsed().as_millis();
            let phase = match tool_call.name.as_str() {
                "find_resources" => Some(ConversationTimingPhase::ResourceDirectoryLookup),
                "knowledge_search" => Some(ConversationTimingPhase::Retrieval),
                _ => None,
            };
            match execution {
                Ok((result, outcome)) => {
                    let elapsed_ms = call_started_at.elapsed().as_millis();
                    let status = outcome.as_str();
                    if let Some(phase) = phase {
                        self.emit_trace(AgentTraceEvent::Timing {
                            phase,
                            step: None,
                            tool_name: Some(tool_call.name.clone()),
                            call_id: Some(call_id.to_string()),
                            attempt,
                            outcome,
                            elapsed_ms: attempt_elapsed_ms,
                        });
                    }
                    self.emit_trace(AgentTraceEvent::Timing {
                        phase: ConversationTimingPhase::ToolExecution,
                        step: None,
                        tool_name: Some(tool_call.name.clone()),
                        call_id: Some(call_id.to_string()),
                        attempt,
                        outcome,
                        elapsed_ms,
                    });
                    break (result, status.to_string(), elapsed_ms);
                }
                Err(error) => {
                    let owned_failure;
                    let failure = if let Some(failure) = error.downcast_ref::<ToolExecutionError>()
                    {
                        failure
                    } else {
                        owned_failure = ToolExecutionError::Other(error.to_string());
                        &owned_failure
                    };
                    let reason = match failure {
                        ToolExecutionError::Unauthorized => "unauthorized",
                        ToolExecutionError::Connection => "connection_failure",
                        ToolExecutionError::Timeout => "timeout",
                        ToolExecutionError::HttpStatus(status) => match *status {
                            502 => "http_502",
                            503 => "http_503",
                            504 => "http_504",
                            _ => "http_failure",
                        },
                        ToolExecutionError::MalformedContract => "malformed_contract",
                        ToolExecutionError::Other(_) => "non_retryable_failure",
                    };
                    let retryable = matches!(
                        failure,
                        ToolExecutionError::Connection
                            | ToolExecutionError::Timeout
                            | ToolExecutionError::HttpStatus(502..=504)
                    );
                    if let Some(phase) = phase {
                        self.emit_trace(AgentTraceEvent::Timing {
                            phase,
                            step: None,
                            tool_name: Some(tool_call.name.clone()),
                            call_id: Some(call_id.to_string()),
                            attempt,
                            outcome: if matches!(failure, ToolExecutionError::Timeout) {
                                ConversationTimingOutcome::TimedOut
                            } else {
                                ConversationTimingOutcome::Failed
                            },
                            elapsed_ms: attempt_elapsed_ms,
                        });
                    }
                    if matches!(failure, ToolExecutionError::Timeout) && !timeout_event_emitted {
                        let attempt_elapsed_ms = attempt_started_at.elapsed().as_millis();
                        self.emit_trace(AgentTraceEvent::ToolTimedOut {
                            call_id: call_id.to_string(),
                            tool_name: tool_call.name.clone(),
                            tool_round,
                            attempt,
                            elapsed_ms: attempt_elapsed_ms,
                        });
                    }
                    let remaining = policy
                        .total_budget()
                        .map(|budget| budget.saturating_sub(call_started_at.elapsed()))
                        .unwrap_or(Duration::ZERO);
                    if retryable && policy.can_retry(attempt, remaining) {
                        self.emit_trace(AgentTraceEvent::ToolRetryScheduled {
                            call_id: call_id.to_string(),
                            tool_name: tool_call.name.clone(),
                            tool_round,
                            attempt,
                            reason: reason.to_string(),
                        });
                        let delay_started_at = Instant::now();
                        tokio::time::sleep(policy.backoff()).await;
                        self.emit_trace(AgentTraceEvent::Timing {
                            phase: ConversationTimingPhase::RetryDelay,
                            step: None,
                            tool_name: Some(tool_call.name.clone()),
                            call_id: Some(call_id.to_string()),
                            attempt,
                            outcome: ConversationTimingOutcome::Succeeded,
                            elapsed_ms: delay_started_at.elapsed().as_millis(),
                        });
                        attempt += 1;
                        continue;
                    }
                    let terminal_status = if matches!(failure, ToolExecutionError::Timeout) {
                        "timed_out"
                    } else {
                        "failed"
                    };
                    let elapsed_ms = call_started_at.elapsed().as_millis();
                    self.emit_trace(AgentTraceEvent::Timing {
                        phase: ConversationTimingPhase::ToolExecution,
                        step: None,
                        tool_name: Some(tool_call.name.clone()),
                        call_id: Some(call_id.to_string()),
                        attempt,
                        outcome: if terminal_status == "timed_out" {
                            ConversationTimingOutcome::TimedOut
                        } else {
                            ConversationTimingOutcome::Failed
                        },
                        elapsed_ms,
                    });
                    let message = match failure {
                        ToolExecutionError::Unauthorized => {
                            "The Tool is not authorized for this conversation."
                        }
                        ToolExecutionError::Connection => "The Tool backend could not be reached.",
                        ToolExecutionError::Timeout => "The Tool timed out before returning data.",
                        ToolExecutionError::HttpStatus(_) => {
                            "The Tool backend could not complete the request."
                        }
                        ToolExecutionError::MalformedContract => {
                            "The Tool backend returned an invalid response."
                        }
                        ToolExecutionError::Other(_) => "The Tool could not complete the request.",
                    };
                    let result = NativeToolResult::failure(reason, message);
                    break (result, terminal_status.to_string(), elapsed_ms);
                }
            }
        };

        self.emit_trace(AgentTraceEvent::ToolTerminal {
            call_id: call_id.to_string(),
            tool_name: tool_call.name.clone(),
            tool_round,
            attempt,
            status: terminal_status,
            elapsed_ms,
        });
        result
    }

    /// Attempt to correct a malformed LLM response using the correction agent
    ///
    /// Takes the raw LLM output directly and asks a specialized correction agent
    /// to reshape it into the proper format.
    async fn attempt_correction(
        &self,
        original_input: &str,
        available_tools: &str,
        raw_response: &str,
        error_message: &str,
    ) -> Result<AgentResponse> {
        if raw_response.is_empty() {
            return Err(anyhow::anyhow!("No raw response available for correction"));
        }

        tracing::info!("=== CORRECTION ATTEMPT ===");
        tracing::info!("Error: {}", error_message);
        tracing::info!("Raw response length: {} chars", raw_response.len());
        tracing::info!("Raw response:\n{}", raw_response);

        // Create the correction predictor
        let correction_predictor = Predict::<CorrectionResponse>::builder()
            .instruction(CORRECTION_INSTRUCTION)
            .build();

        let correction_input = CorrectionResponseInput {
            original_input: original_input.to_string(),
            malformed_response: raw_response.to_string(),
            error_message: error_message.to_string(),
            available_tools: available_tools.to_string(),
        };

        // Call correction agent (no retry on correction - avoid infinite loops)
        let corrected = correction_predictor.call(correction_input).await?;

        tracing::info!("=== CORRECTION RESULT ===");
        tracing::info!("Corrected messages: {:?}", corrected.messages);
        tracing::info!("Corrected tool_calls: {:?}", corrected.tool_calls);

        // Convert CorrectionResponse to AgentResponse
        Ok(AgentResponse {
            input: original_input.to_string(),
            current_time: String::new(),
            persona_block: String::new(),
            human_block: String::new(),
            memory_metadata: String::new(),
            previous_context_summary: String::new(),
            recent_conversation: String::new(),
            available_tools: available_tools.to_string(),
            is_first_time_user: false,
            messages: corrected.messages,
            tool_calls: corrected.tool_calls,
        })
    }

    /// Execute a single step of the agent loop
    /// Returns messages to send and whether we're done
    pub async fn step(&mut self, user_message: &str, is_first_step: bool) -> Result<StepResult> {
        // Clear tool results at start of new request
        if is_first_step {
            self.current_tool_results.clear();
            self.turn_step_index = 0;
        }
        let step_index = self.turn_step_index;
        self.turn_step_index += 1;

        tracing::debug!("Agent step (first={})", is_first_step);

        // Create predictor with instruction
        let predictor = Predict::<AgentResponse>::builder()
            .instruction(self.instruction.clone())
            .build();

        // Build context - separate fields for each input
        let ctx = self.build_context();

        // Input is either the user message (first step) or ALL tool results from this cycle
        let input_content = if is_first_step {
            user_message.to_string()
        } else {
            // Collect ALL tool results from current cycle
            let tool_results: Vec<&str> = self
                .current_tool_results
                .iter()
                .filter(|m| m.role == "tool")
                .map(|m| m.content.as_str())
                .collect();

            if tool_results.is_empty() {
                user_message.to_string()
            } else {
                // Build summary of what was already sent this turn
                let already_sent = if let Some((sent_messages, tool_names)) =
                    &self.previous_step_summary
                {
                    let tools_str = tool_names.join(", ");
                    let msgs_preview = if sent_messages.is_empty() {
                        String::new()
                    } else {
                        let msgs_text = sent_messages
                            .iter()
                            .enumerate()
                            .map(|(i, m)| format!("  {}. \"{}\"", i + 1, m))
                            .collect::<Vec<_>>()
                            .join("\n");
                        format!("\nMessages you already sent to user:\n{}\n", msgs_text)
                    };
                    format!("[You already sent {} message(s) and called {} this turn.{}Tools have executed:]\n\n",
                        sent_messages.len(), tools_str, msgs_preview)
                } else {
                    String::new()
                };

                let tool_result_instructions = r#"

=== TOOL RESULT PROCESSING MODE ===
This is a CONTINUATION of your previous turn, NOT a new conversation.
Your previous messages are already visible to the user in recent_conversation.

RULES:
1. SILENCE IS DEFAULT - You do NOT need to acknowledge the tool result
2. DO NOT say: "I see the results", "Let me analyze", "Based on what I found", "Here's what the tool returned"
3. DO NOT repeat or rephrase what you already said
4. If the tool was for YOUR benefit (memory ops, archival), call 'done' immediately
5. Only send messages if you have GENUINELY NEW information the user hasn't seen

SELF-CHECK: Before ANY message, ask: "Is this new info the user hasn't seen?" If no → call 'done'"#;

                let result = if tool_results.len() == 1 {
                    format!(
                        "{}=== TOOL RESULT ===\n{}\n=== END TOOL RESULT ==={}",
                        already_sent, tool_results[0], tool_result_instructions
                    )
                } else {
                    let results_text = tool_results
                        .iter()
                        .enumerate()
                        .map(|(i, r)| format!("--- Tool {} ---\n{}", i + 1, r))
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    format!(
                        "{}=== TOOL RESULTS ({} tools) ===\n{}\n=== END TOOL RESULTS ==={}",
                        already_sent,
                        tool_results.len(),
                        results_text,
                        tool_result_instructions
                    )
                };

                // Clear tool results after presenting them
                self.current_tool_results.clear();

                result
            }
        };

        tracing::info!("=== LLM REQUEST ===");
        tracing::info!("Tool results in cycle: {}", self.current_tool_results.len());
        tracing::info!("Is first time user: {}", ctx.is_first_time_user);
        tracing::debug!(
            "LLM request content sizes: input_chars={}, recent_conversation_chars={}",
            input_content.chars().count(),
            ctx.recent_conversation.chars().count()
        );

        let available_tools = self.tools.generate_description();
        let input = AgentResponseInput {
            input: input_content.clone(),
            current_time: ctx.current_time,
            persona_block: ctx.persona_block,
            human_block: ctx.human_block,
            memory_metadata: ctx.memory_metadata,
            previous_context_summary: ctx.previous_context_summary,
            recent_conversation: ctx.recent_conversation,
            available_tools: available_tools.clone(),
            is_first_time_user: ctx.is_first_time_user,
        };

        // Get typed response from LLM with retry logic (up to 3 attempts)
        const MAX_LLM_RETRIES: u32 = 3;
        let mut last_error: Option<dspy_rs::PredictError> = None;
        let mut response: Option<AgentResponse> = None;

        for attempt in 1..=MAX_LLM_RETRIES {
            self.emit_trace(AgentTraceEvent::ModelStepStarted {
                step: step_index,
                attempt,
            });
            let attempt_started_at = Instant::now();
            match predictor.call(input.clone()).await {
                Ok(r) => {
                    self.emit_trace(AgentTraceEvent::ModelStepCompleted {
                        step: step_index,
                        attempt,
                        elapsed_ms: attempt_started_at.elapsed().as_millis(),
                    });
                    response = Some(r);
                    break;
                }
                Err(e) => {
                    self.emit_trace(AgentTraceEvent::ModelStepFailed {
                        step: step_index,
                        attempt,
                        elapsed_ms: attempt_started_at.elapsed().as_millis(),
                        error: format!("{:?}", e),
                    });
                    tracing::warn!(
                        "LLM call failed (attempt {}/{}): {:?}",
                        attempt,
                        MAX_LLM_RETRIES,
                        e
                    );

                    // For parse errors, try correction instead of simple retry
                    if let dspy_rs::PredictError::Parse {
                        raw_response,
                        source,
                        ..
                    } = &e
                    {
                        let error_message = format!("Parse error: {}", source);
                        self.emit_trace(AgentTraceEvent::CorrectionStarted {
                            step: step_index,
                            attempt,
                            error: error_message.clone(),
                        });
                        let correction_started_at = Instant::now();
                        match self
                            .attempt_correction(
                                &input_content,
                                &available_tools,
                                raw_response,
                                &error_message,
                            )
                            .await
                        {
                            Ok(corrected) => {
                                self.emit_trace(AgentTraceEvent::CorrectionCompleted {
                                    step: step_index,
                                    attempt,
                                    elapsed_ms: correction_started_at.elapsed().as_millis(),
                                });
                                response = Some(corrected);
                                break;
                            }
                            Err(correction_err) => {
                                self.emit_trace(AgentTraceEvent::CorrectionFailed {
                                    step: step_index,
                                    attempt,
                                    elapsed_ms: correction_started_at.elapsed().as_millis(),
                                    error: correction_err.to_string(),
                                });
                                tracing::warn!(
                                    "Correction failed (attempt {}/{}): {:?}",
                                    attempt,
                                    MAX_LLM_RETRIES,
                                    correction_err
                                );
                            }
                        }
                    }

                    last_error = Some(e);

                    // Add a small delay before retry (except on last attempt)
                    if attempt < MAX_LLM_RETRIES {
                        self.emit_trace(AgentTraceEvent::RetryScheduled {
                            step: step_index,
                            attempt,
                        });
                        tracing::info!("Retrying LLM call in 1 second...");
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    }
                }
            }
        }

        let response = match response {
            Some(r) => r,
            None => {
                let err = last_error.unwrap();
                tracing::error!(
                    "LLM call failed after {} attempts: {:?}",
                    MAX_LLM_RETRIES,
                    err
                );
                // Preserve the underlying provider status in this generic
                // runtime's internal diagnostic error chain.
                return Err(anyhow::anyhow!(
                    "LLM error after {} retries: {:?}",
                    MAX_LLM_RETRIES,
                    err
                ));
            }
        };

        tracing::info!("=== LLM RESPONSE ===");
        tracing::info!("Messages (raw): {:?}", response.messages);
        tracing::info!("Tool calls: {:?}", response.tool_calls);

        // Unwrap nested JSON arrays and collect non-empty messages
        // Sometimes the LLM double-encodes: ["[\"msg1\", \"msg2\"]"] instead of ["msg1", "msg2"]
        let messages: Vec<String> = response
            .messages
            .iter()
            .flat_map(|m| {
                let trimmed = m.trim();
                // Check if this message is itself a JSON array
                if trimmed.starts_with('[') && trimmed.ends_with(']') {
                    // Try to parse as JSON array of strings
                    if let Ok(inner_messages) = serde_json::from_str::<Vec<String>>(trimmed) {
                        tracing::debug!(
                            "Unwrapped nested JSON array with {} messages",
                            inner_messages.len()
                        );
                        return inner_messages;
                    }
                }
                // Not a nested array, return as-is
                vec![m.clone()]
            })
            .filter(|m| !m.is_empty())
            .collect();

        tracing::info!("Messages (processed): {:?}", messages);

        // Execute tools and collect results for storage
        let mut executed_tools = Vec::new();

        for tool_call in &response.tool_calls {
            tracing::info!(
                "Executing tool: {} with args: {:?}",
                tool_call.name,
                tool_call.args
            );

            let call_id = format!("tool-call-{}", Uuid::new_v4().simple());
            let native_result = self.execute_tool_call(&call_id, 0, tool_call).await;
            let model_result = native_result.to_tool_result();
            let result = native_result.into_upstream_tool_result();
            tracing::debug!("Tool {} result: {:?}", tool_call.name, result);

            // Inject into current request cycle (for multi-step reasoning)
            self.inject_tool_result(tool_call, &model_result);

            // Collect for storage (skip "done" tool - it's just a no-op signal)
            if tool_call.name != "done" {
                executed_tools.push(ExecutedTool {
                    tool_call: tool_call.clone(),
                    result,
                });
            }
        }

        // Done if no tool calls, OR if the only tool call is "done"
        let done = response.tool_calls.is_empty()
            || (response.tool_calls.len() == 1 && response.tool_calls[0].name == "done");

        // Track what we sent this step for next iteration's context
        // This helps the model know what it already said when it sees tool results
        if !messages.is_empty() || !response.tool_calls.is_empty() {
            let tool_names: Vec<String> = response
                .tool_calls
                .iter()
                .map(|tc| tc.name.clone())
                .collect();
            self.previous_step_summary = Some((messages.clone(), tool_names));
        }

        Ok(StepResult {
            messages,
            tool_calls: response.tool_calls,
            executed_tools,
            done,
        })
    }

    /// Process a user message, yielding messages after each step
    /// This allows the caller to send messages immediately between tool calls
    pub async fn process_message(&mut self, user_message: &str) -> Result<Vec<String>> {
        let mut all_messages = Vec::new();

        for step_num in 0..self.max_steps {
            let result = self.step(user_message, step_num == 0).await?;

            all_messages.extend(result.messages);

            if result.done {
                break;
            }
        }

        // If no messages were produced, return a failure message
        if all_messages.is_empty() {
            tracing::warn!("Agent produced no messages");
            all_messages.push("I apologize, but I wasn't able to generate a response.".to_string());
        }

        Ok(all_messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn native_tool_definitions_preserve_authoritative_json_schema_and_omit_done() {
        let mut registry = ToolRegistry::new();
        let knowledge_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search terms"},
                "count": {"type": "integer", "default": 5},
                "exact": {"type": "boolean", "default": false}
            },
            "required": ["query"],
            "additionalProperties": false
        });
        registry.register_descriptor(
            "knowledge_search",
            "Search uploaded Documents.",
            &knowledge_schema.to_string(),
        );
        let deployment_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "settings": {
                    "type": "object",
                    "description": "Deployment setting names and desired values.",
                    "additionalProperties": {}
                }
            },
            "required": ["settings"],
            "additionalProperties": false
        });
        registry.register_descriptor(
            "update_deployment_settings",
            "Update Deployment Settings.",
            &deployment_schema.to_string(),
        );
        registry.register_descriptor("done", "Legacy loop terminator.", r#"{}"#);

        let definitions = registry
            .native_definitions()
            .expect("descriptor examples should convert to JSON Schema");

        assert_eq!(definitions.len(), 2);
        assert_eq!(definitions[0].name, "knowledge_search");
        assert_eq!(definitions[0].parameters, knowledge_schema);
        assert_eq!(definitions[1].name, "update_deployment_settings");
        assert_eq!(definitions[1].parameters, deployment_schema);
    }

    #[test]
    fn tool_arg_preview_does_not_double_quote_json_strings() {
        assert_eq!(tool_arg_preview(&serde_json::json!("safety")), "safety");
        assert_eq!(
            tool_arg_preview(&serde_json::json!({"query": "safety"})),
            r#"{"query":"safety"}"#
        );
    }

    #[test]
    fn truncation_never_exceeds_very_small_character_budgets() {
        for max_chars in 0..=3 {
            assert_eq!(
                truncate_chars_with_ellipsis("long result", max_chars)
                    .chars()
                    .count(),
                max_chars
            );
        }
    }

    #[test]
    fn plain_answer_prompt_bounds_and_deduplicates_current_tool_results() {
        let mut agent = SageAgent::new_without_memory(ToolRegistry::new(), "Help the user.");
        let duplicate_call = ToolCall {
            name: "knowledge_search".to_string(),
            args: ToolArgs::from([("query".to_string(), serde_json::json!("safety"))]),
        };
        let duplicate_result = ToolResult::success(format!(
            "DUPLICATE_RESULT_{}",
            "duplicate context ".repeat(400)
        ));
        agent.inject_tool_result(&duplicate_call, &duplicate_result);
        agent.inject_tool_result(&duplicate_call, &duplicate_result);
        agent.inject_tool_result(
            &ToolCall {
                name: "knowledge_search".to_string(),
                args: ToolArgs::from([("query".to_string(), serde_json::json!("older"))]),
            },
            &ToolResult::success(format!("OLDER_RESULT_{}", "older context ".repeat(400))),
        );
        agent.inject_tool_result(
            &ToolCall {
                name: "knowledge_search".to_string(),
                args: ToolArgs::from([("query".to_string(), serde_json::json!("newest"))]),
            },
            &ToolResult::success(format!("NEWEST_RESULT_{}", "new context ".repeat(500))),
        );

        let prompt = agent.native_turn_prompt("Give the final answer.");

        assert!(
            prompt.user.chars().count() <= 13_500,
            "final-answer Tool context must stay within a bounded prompt budget"
        );
        assert_eq!(prompt.user.matches("DUPLICATE_RESULT_").count(), 1);
        assert!(
            prompt.user.contains("NEWEST_RESULT_"),
            "the most recent Tool result should survive the context budget"
        );
    }

    #[test]
    fn test_tool_registry() {
        let registry = ToolRegistry::new();
        assert!(!registry.has("web_search"));
        assert!(registry.tools.is_empty());
    }

    #[test]
    fn test_tool_registry_description() {
        let registry = ToolRegistry::new();
        let desc = registry.generate_description();
        assert_eq!(desc, "No tools available.");
    }

    #[tokio::test]
    async fn configure_lm_with_temperature_rejects_invalid_values() {
        for temperature in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            let error = SageAgent::configure_lm_with_temperature(
                "http://example.test/v1",
                "test-key",
                "test-model",
                temperature,
            )
            .await
            .expect_err("invalid temperature should fail before configuring the LM");

            assert!(
                error.to_string().contains("configure_lm_with_temperature"),
                "error should name the failing function: {error}"
            );
        }
    }

    struct ScriptedRetryTool {
        policy: ToolRetryPolicy,
        outcomes: Arc<Mutex<std::collections::VecDeque<Result<NativeToolResult>>>>,
    }

    struct NativeOnlyTool;

    #[async_trait::async_trait]
    impl Tool for NativeOnlyTool {
        fn name(&self) -> &str {
            "native_only_tool"
        }

        fn description(&self) -> &str {
            "test-only native Tool"
        }

        fn native_parameters(&self) -> Result<serde_json::Value> {
            Ok(serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }))
        }

        async fn execute_native(&self, _args: &ToolArgs) -> Result<NativeToolResult> {
            Ok(NativeToolResult::success(serde_json::json!({
                "path": "native"
            })))
        }

        async fn execute_native_with_timing_outcome(
            &self,
            args: &ToolArgs,
        ) -> Result<(NativeToolResult, ConversationTimingOutcome)> {
            self.execute_native(args)
                .await
                .map(|result| (result, ConversationTimingOutcome::Succeeded))
        }
    }

    #[tokio::test]
    async fn shared_executor_has_one_native_tool_path() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(NativeOnlyTool));
        let agent = SageAgent::new_without_memory(registry, "test");
        let call = ToolCall {
            name: "native_only_tool".to_string(),
            args: ToolArgs::new(),
        };

        let result = agent.execute_tool_call("native-call", 1, &call).await;

        assert_eq!(result.model_value()["path"], "native");
    }

    #[tokio::test]
    async fn description_only_tool_returns_a_typed_failure() {
        let descriptor = ToolDescriptor {
            name: "described_only".to_string(),
            description: "description-only test Tool".to_string(),
            native_parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        };

        let result = descriptor
            .execute_native(&ToolArgs::new())
            .await
            .expect("description-only execution should fail without panicking");

        assert_eq!(result.model_value()["error"]["code"], "tool_not_executable");
    }

    #[test]
    fn upstream_legacy_failure_bridge_never_forwards_backend_details_to_the_model() {
        let result = NativeToolResult::from_upstream_legacy(ToolResult::error(
            "backend failed with secret SENTINEL_PRIVATE_VALUE",
        ));
        let model_value = result.model_value().to_string();

        assert!(!model_value.contains("SENTINEL_PRIVATE_VALUE"));
        assert_eq!(
            result.model_value()["error"]["message"],
            "The Tool could not complete the request."
        );
        assert_eq!(
            result.into_upstream_tool_result(),
            ToolResult::error("backend failed with secret SENTINEL_PRIVATE_VALUE")
        );
    }

    #[test]
    fn upstream_legacy_success_bridge_preserves_exact_output_and_metadata() {
        let original = ToolResult::success_with_metadata(
            "plain upstream output",
            serde_json::json!({"source": "upstream"}),
        );
        let result = NativeToolResult::from_upstream_legacy(original.clone());

        assert_eq!(result.into_upstream_tool_result(), original);
    }

    #[async_trait::async_trait]
    impl Tool for ScriptedRetryTool {
        fn name(&self) -> &str {
            "scripted_lookup"
        }

        fn description(&self) -> &str {
            "test-only scripted lookup"
        }

        fn native_parameters(&self) -> Result<serde_json::Value> {
            Ok(serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }))
        }

        fn retry_policy(&self) -> ToolRetryPolicy {
            self.policy.clone()
        }

        async fn execute_native(&self, _args: &ToolArgs) -> Result<NativeToolResult> {
            self.outcomes
                .lock()
                .expect("scripted outcomes should lock")
                .pop_front()
                .expect("test should provide a scripted outcome")
        }
    }

    async fn execute_native_test_call(agent: &mut SageAgent, name: &str) -> NativeToolBatchResult {
        agent
            .execute_native_tool_calls(
                1,
                &[NativeToolCall {
                    id: "native-test-call".to_string(),
                    name: name.to_string(),
                    arguments: serde_json::json!({}),
                }],
            )
            .await
    }

    struct DelayedTypedTimeoutTool {
        attempt: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Tool for DelayedTypedTimeoutTool {
        fn name(&self) -> &str {
            "delayed_timeout_lookup"
        }

        fn description(&self) -> &str {
            "test-only delayed timeout lookup"
        }

        fn native_parameters(&self) -> Result<serde_json::Value> {
            Ok(serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }))
        }

        fn retry_policy(&self) -> ToolRetryPolicy {
            ToolRetryPolicy::read_only(Duration::from_millis(100), 2, Duration::from_millis(500))
        }

        async fn execute_native(&self, _args: &ToolArgs) -> Result<NativeToolResult> {
            match self
                .attempt
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            {
                0 => Err(anyhow::Error::new(ToolExecutionError::HttpStatus(503))),
                1 => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    Err(anyhow::Error::new(ToolExecutionError::Timeout))
                }
                _ => Ok(NativeToolResult::success(serde_json::json!({
                    "status": "recovered"
                }))),
            }
        }
    }

    #[tokio::test]
    async fn read_only_retry_reuses_call_correlation_and_emits_one_terminal() {
        let outcomes = Arc::new(Mutex::new(std::collections::VecDeque::from([
            Err(anyhow::Error::new(ToolExecutionError::HttpStatus(503))),
            Ok(NativeToolResult::success(serde_json::json!({
                "status": "recovered"
            }))),
        ])));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(ScriptedRetryTool {
            policy: ToolRetryPolicy::read_only(
                Duration::from_millis(50),
                2,
                Duration::from_millis(200),
            ),
            outcomes,
        }));
        let mut agent = SageAgent::new_without_memory(registry, "test");
        let events = Arc::new(Mutex::new(Vec::new()));
        let event_sink = events.clone();
        agent.set_trace_hook(Arc::new(move |event| {
            event_sink
                .lock()
                .expect("event sink should lock")
                .push(event);
        }));

        let result = execute_native_test_call(&mut agent, "scripted_lookup").await;

        assert!(result.executed_tools[0].result.is_success());
        let events = events.lock().expect("event sink should lock");
        let attempted = events
            .iter()
            .filter(|event| matches!(event, AgentTraceEvent::ToolAttempted { .. }))
            .count();
        let retries = events
            .iter()
            .filter(|event| matches!(event, AgentTraceEvent::ToolRetryScheduled { .. }))
            .count();
        let terminals = events
            .iter()
            .filter(|event| matches!(event, AgentTraceEvent::ToolTerminal { .. }))
            .count();
        assert_eq!(attempted, 2);
        assert_eq!(retries, 1);
        assert_eq!(terminals, 1);
        let call_ids = events
            .iter()
            .filter_map(|event| match event {
                AgentTraceEvent::ToolAttempted { call_id, .. }
                | AgentTraceEvent::ToolTerminal { call_id, .. } => Some(call_id),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(call_ids.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[tokio::test]
    async fn state_changing_default_policy_never_retries() {
        let outcomes = Arc::new(Mutex::new(std::collections::VecDeque::from([Err(
            anyhow::Error::new(ToolExecutionError::HttpStatus(503)),
        )])));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(ScriptedRetryTool {
            policy: ToolRetryPolicy::none(),
            outcomes: outcomes.clone(),
        }));
        let mut agent = SageAgent::new_without_memory(registry, "test");
        let events = Arc::new(Mutex::new(Vec::new()));
        let event_sink = events.clone();
        agent.set_trace_hook(Arc::new(move |event| {
            event_sink
                .lock()
                .expect("event sink should lock")
                .push(event);
        }));
        let result = execute_native_test_call(&mut agent, "scripted_lookup").await;
        assert!(!result.executed_tools[0].result.is_success());
        assert_eq!(outcomes.lock().expect("outcomes should lock").len(), 0);
        let events = events.lock().expect("event sink should lock");
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentTraceEvent::ToolAttempted { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentTraceEvent::ToolRetryScheduled { .. }))
                .count(),
            0
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentTraceEvent::ToolTerminal { .. }))
                .count(),
            1
        );
    }

    struct HangingNoRetryTool;

    #[async_trait::async_trait]
    impl Tool for HangingNoRetryTool {
        fn name(&self) -> &str {
            "hanging_write"
        }

        fn description(&self) -> &str {
            "test-only hanging write"
        }

        fn native_parameters(&self) -> Result<serde_json::Value> {
            Ok(serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }))
        }

        fn retry_policy(&self) -> ToolRetryPolicy {
            ToolRetryPolicy::no_retry(Duration::from_millis(10))
        }

        async fn execute_native(&self, _args: &ToolArgs) -> Result<NativeToolResult> {
            std::future::pending().await
        }
    }

    #[tokio::test]
    async fn non_retryable_tool_attempt_times_out_and_emits_terminal_evidence() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(HangingNoRetryTool));
        let mut agent = SageAgent::new_without_memory(registry, "test");
        let events = Arc::new(Mutex::new(Vec::new()));
        let event_sink = events.clone();
        agent.set_trace_hook(Arc::new(move |event| {
            event_sink.lock().expect("event sink").push(event);
        }));

        let result = execute_native_test_call(&mut agent, "hanging_write").await;

        assert!(!result.executed_tools[0].result.is_success());
        let events = events.lock().expect("event sink");
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentTraceEvent::ToolAttempted { .. }))
                .count(),
            1
        );
        assert!(events.iter().any(|event| matches!(
            event,
            AgentTraceEvent::ToolTimedOut { call_id, attempt: 1, .. }
                if call_id == "native-test-call"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AgentTraceEvent::ToolTerminal { call_id, status, attempt: 1, .. }
                if call_id == "native-test-call" && status == "timed_out"
        )));
        assert!(!events
            .iter()
            .any(|event| matches!(event, AgentTraceEvent::ToolRetryScheduled { .. })));
    }

    #[test]
    fn read_only_policies_keep_the_issue_budgets() {
        assert_eq!(
            ToolRetryPolicy::curated_resources(),
            ToolRetryPolicy::read_only(Duration::from_secs(5), 2, Duration::from_secs(8))
        );
        assert_eq!(
            ToolRetryPolicy::knowledge_search(),
            ToolRetryPolicy::read_only(Duration::from_secs(15), 2, Duration::from_secs(35))
        );
    }

    #[tokio::test]
    async fn retry_is_skipped_when_only_backoff_budget_remains() {
        let outcomes = Arc::new(Mutex::new(std::collections::VecDeque::from([
            Err(anyhow::Error::new(ToolExecutionError::HttpStatus(503))),
            Ok(NativeToolResult::success(serde_json::json!({
                "status": "should not run"
            }))),
        ])));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(ScriptedRetryTool {
            policy: ToolRetryPolicy::read_only(
                Duration::from_millis(50),
                2,
                Duration::from_millis(150),
            ),
            outcomes: outcomes.clone(),
        }));
        let mut agent = SageAgent::new_without_memory(registry, "test");
        let result = execute_native_test_call(&mut agent, "scripted_lookup").await;
        assert!(!result.executed_tools[0].result.is_success());
        assert_eq!(outcomes.lock().expect("outcomes should lock").len(), 1);
    }

    #[tokio::test]
    async fn connection_failure_retries_once_with_privacy_safe_reason() {
        let outcomes = Arc::new(Mutex::new(std::collections::VecDeque::from([
            Err(anyhow::Error::new(ToolExecutionError::Connection)),
            Ok(NativeToolResult::success(serde_json::json!({
                "status": "recovered"
            }))),
        ])));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(ScriptedRetryTool {
            policy: ToolRetryPolicy::read_only(
                Duration::from_millis(50),
                2,
                Duration::from_millis(500),
            ),
            outcomes,
        }));
        let mut agent = SageAgent::new_without_memory(registry, "test");
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        agent.set_trace_hook(Arc::new(move |event| sink.lock().unwrap().push(event)));
        let result = execute_native_test_call(&mut agent, "scripted_lookup").await;
        assert!(result.executed_tools[0].result.is_success());
        let events = events.lock().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentTraceEvent::ToolAttempted { .. }))
                .count(),
            2
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentTraceEvent::ToolTerminal { .. }))
                .count(),
            1
        );
        let retry = events.iter().find_map(|event| match event {
            AgentTraceEvent::ToolRetryScheduled { reason, .. } => Some(reason.as_str()),
            _ => None,
        });
        assert_eq!(retry, Some("connection_failure"));
    }

    #[tokio::test]
    async fn typed_timeout_emits_one_timeout_event_before_retry() {
        let outcomes = Arc::new(Mutex::new(std::collections::VecDeque::from([
            Err(anyhow::Error::new(ToolExecutionError::Timeout)),
            Ok(NativeToolResult::success(serde_json::json!({
                "status": "recovered"
            }))),
        ])));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(ScriptedRetryTool {
            policy: ToolRetryPolicy::read_only(
                Duration::from_millis(50),
                2,
                Duration::from_millis(500),
            ),
            outcomes,
        }));
        let mut agent = SageAgent::new_without_memory(registry, "test");
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        agent.set_trace_hook(Arc::new(move |event| sink.lock().unwrap().push(event)));
        let result = execute_native_test_call(&mut agent, "scripted_lookup").await;
        assert!(result.executed_tools[0].result.is_success());
        let events = events.lock().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentTraceEvent::ToolTimedOut { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentTraceEvent::ToolRetryScheduled { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentTraceEvent::ToolTerminal { .. }))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn typed_timeout_trace_duration_is_scoped_to_attempt() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(DelayedTypedTimeoutTool {
            attempt: std::sync::atomic::AtomicUsize::new(0),
        }));
        let mut agent = SageAgent::new_without_memory(registry, "test");
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        agent.set_trace_hook(Arc::new(move |event| sink.lock().unwrap().push(event)));
        let result = execute_native_test_call(&mut agent, "delayed_timeout_lookup").await;
        assert!(!result.executed_tools[0].result.is_success());
        let events = events.lock().unwrap();
        let timeout_ms = events.iter().find_map(|event| match event {
            AgentTraceEvent::ToolTimedOut {
                attempt: 2,
                elapsed_ms,
                ..
            } => Some(*elapsed_ms),
            _ => None,
        });
        let terminal_ms = events.iter().find_map(|event| match event {
            AgentTraceEvent::ToolTerminal { elapsed_ms, .. } => Some(*elapsed_ms),
            _ => None,
        });
        assert!(timeout_ms.is_some());
        assert!(terminal_ms.is_some_and(|terminal| terminal > timeout_ms.unwrap() + 50));
    }
}
