//! Sage Agent using DSRs signatures and BAML parsing
//!
//! This module implements the core agent using dspy-rs for:
//! - Typed input/output signatures
//! - BAML-based response parsing
//! - GEPA-compatible instruction optimization

use anyhow::Result;
use baml_bridge::{
    baml_types::{type_meta, BamlValue, TypeIR},
    BamlAdapter, BamlConvertError,
};
use dspy_rs::{configure, BamlType, ChatAdapter, Predict, LM};
use isocountry::CountryCode;
use std::collections::{BTreeMap, HashMap, HashSet};
#[cfg(unix)]
use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::memory::MemoryManager;

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

/// Typed model contract used only to decide which Tools should run.
/// User-visible prose intentionally has no output field here.
#[derive(dspy_rs::Signature, Clone, Debug)]
struct ToolDecisionResponse {
    #[input(desc = "The user request or Tool results to plan against")]
    pub input: String,

    #[input(desc = "Current date and time in user's timezone")]
    pub current_time: String,

    #[input(desc = "Your persona and Agent Settings profile")]
    pub persona_block: String,

    #[input(desc = "What you know about this human")]
    pub human_block: String,

    #[input(desc = "Session Memory statistics")]
    pub memory_metadata: String,

    #[input(desc = "Summary of older conversation; ignore when empty")]
    pub previous_context_summary: String,

    #[input(desc = "Recent conversation and any completed Tool results")]
    pub recent_conversation: String,

    #[input(desc = "Enabled Tools and their argument contracts")]
    pub available_tools: String,

    #[input(desc = "Whether this is the first conversation with this user")]
    pub is_first_time_user: bool,

    #[output(desc = "All immediately useful Tool calls; never include user messages")]
    pub tool_calls: Vec<ToolCall>,

    #[output(
        desc = "Optional hint. True only when Tool results are needed to decide whether another Tool round is required; omit or use false otherwise"
    )]
    pub replan_after_results: Option<bool>,
}

/// Provider-neutral Tool plan returned to the web conversation state machine.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct ToolDecision {
    pub tool_calls: Vec<ToolCall>,
    pub replan_after_results: bool,
    pub planning_round: usize,
}

impl ToolDecision {
    pub fn new(tool_calls: Vec<ToolCall>, replan_after_results: bool) -> Self {
        Self {
            tool_calls: tool_calls
                .into_iter()
                .filter(|tool_call| tool_call.name != "done")
                .collect(),
            replan_after_results,
            planning_round: 0,
        }
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum ToolPlanningOutcome {
    Decision(ToolDecision),
    RecoveredTerminalProse(String),
}

/// Provider-neutral seam for the bounded, typed Tool-planning phase.
#[async_trait::async_trait]
#[allow(dead_code)]
pub trait ToolPlanner: Send {
    fn has_actionable_tools(&self) -> bool;

    /// Supply the current raw User message's Curated Resources requirement.
    /// This validates the model's Tool plan; it never authorizes or executes a
    /// Tool call by itself.
    fn set_curated_resource_lookup_expectation(
        &mut self,
        _expectation: CuratedResourceLookupExpectation,
    ) {
    }

    async fn plan_tools(
        &mut self,
        user_message: &str,
        is_first_plan: bool,
    ) -> Result<ToolPlanningOutcome>;

    async fn execute_tool_decision(&mut self, decision: &ToolDecision) -> StepResult;

    fn plain_answer_prompt(&self, user_message: &str) -> PlainAnswerPrompt;

    fn plain_answer_trace_started(&mut self) -> usize {
        0
    }

    fn plain_answer_reasoning_trace_hook(
        &self,
        _step: usize,
    ) -> Option<ProviderReasoningTraceHook> {
        None
    }

    fn plain_answer_provider_timing_hook(&self, _step: usize) -> Option<ProviderTimingTraceHook> {
        None
    }

    fn plain_answer_trace_completed(&self, _step: usize, _elapsed_ms: u128) {}

    fn plain_answer_trace_failed(&self, _step: usize, _elapsed_ms: u128, _error: &str) {}
}

/// Provider-neutral prompt passed to plain answer generation.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct PlainAnswerPrompt {
    pub system: String,
    pub user: String,
    /// Trusted runtime state from the executed Curated Resources Tool. This
    /// must not be reconstructed from rendered prompt text, which can contain
    /// user-controlled strings that resemble Tool-result markers.
    pub incomplete_curated_resource_page: bool,
}

const TOOL_PLANNING_INSTRUCTION: &str = r#"

=== TOOL PLANNING MODE ===
You are deciding Tools, not writing the answer.
- Never emit user-visible prose or messages.
- Request every immediately useful enabled Tool in this planning round.
- Do not call `done`; an empty tool_calls array means no Tool is needed.
- Set replan_after_results=true only when the results are required to decide whether another Tool call is needed.
- Otherwise set replan_after_results=false so the runtime can write the final answer directly.
- Follow every Tool's authorization, argument, and safety contract exactly.
"#;

const PLAIN_ANSWER_INSTRUCTION: &str = r#"

=== FINAL ANSWER MODE ===
Write only the final user-visible answer as plain text.
Do not emit JSON, schema field markers, tool_calls, function calls, or internal reasoning.
The Tool phase is complete. Use the supplied Tool results as facts, respect their warnings and failures, and do not claim that a failed Tool succeeded.
"#;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CuratedResourceLookupExpectation {
    required: bool,
    query_required: bool,
    positive_offset_required: bool,
    expected_query: Option<String>,
    expected_region: Option<String>,
    expected_help_type: Option<String>,
    expected_language: Option<String>,
    expected_offset: Option<usize>,
    query_must_be_absent: bool,
    region_must_be_absent: bool,
    help_type_must_be_absent: bool,
    language_must_be_absent: bool,
    continuation_cursor_missing: bool,
    exact_filter_missing: bool,
    initial_offset_required: bool,
    expected_lookup_mode: Option<&'static str>,
    lookup_mode_must_be_absent: bool,
    contact_query_from_context: bool,
    context_grounded_resources: Vec<(String, Option<String>)>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct CuratedResourceQueryContext {
    resources: Vec<(String, Option<String>)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CuratedResourceContinuation {
    pub query: Option<String>,
    pub region: Option<String>,
    pub help_type: Option<String>,
    pub language: Option<String>,
    pub lookup_mode: Option<String>,
    pub next_offset: usize,
}

impl CuratedResourceLookupExpectation {
    pub(crate) fn with_query_context(mut self, context: &CuratedResourceQueryContext) -> Self {
        if self.contact_query_from_context {
            self.context_grounded_resources = context.resources.clone();
        }
        self
    }

    fn retry_instruction(&self) -> String {
        if self.continuation_cursor_missing {
            return "The current request asks for another Curated Resources page, but the exact prior query and next_offset are unavailable. Do not invent a query or offset.".to_string();
        }
        if self.exact_filter_missing {
            return "The current request asks for a filtered Curated Resources inventory, but the exact filter could not be derived safely. Do not invent or broaden a query.".to_string();
        }
        if let Some(offset) = self.expected_offset {
            return format!(
                "The current request explicitly asks for the next Curated Resources page. Return exactly one find_resources Tool call using next_offset {offset} and preserving every prior query, region, help_type, and language filter, including fields that were absent."
            );
        }
        if self.positive_offset_required {
            "The current request explicitly asks for the next Curated Resources page. Return a find_resources Tool call with the non-empty prior query and a positive continuation offset derived from the recent conversation.".to_string()
        } else if let Some(query) = &self.expected_query {
            format!(
                "The current request explicitly asks for a filtered Curated Resources lookup. Return a find_resources Tool call preserving query={query:?}."
            )
        } else if self.query_required {
            "The current request explicitly requires a targeted Curated Resources lookup. Return a find_resources Tool call with the organization/name from the current or recent-conversation context in a non-empty query argument.".to_string()
        } else {
            "The current request explicitly requires a fresh Curated Resources lookup. Return a find_resources Tool call using the relevant current and recent-conversation context.".to_string()
        }
    }

    fn retry_instruction_for_violation(&self, violation: &str) -> String {
        if violation == "additional find_resources call was not allowed after turn success" {
            return "A Curated Resources lookup already succeeded for this turn. Do not call find_resources again; use the completed Tool results to write the final answer."
                .to_string();
        }
        if violation == "find_resources query was not grounded in recent conversation" {
            return "Use the organization or name established in the current or recent Conversation as the find_resources query. Do not invent or substitute a name."
                .to_string();
        }
        if violation == "find_resources region did not preserve the requested location" {
            return "Preserve both the requested organization/name and location in the find_resources query and region arguments. Do not omit, invent, or substitute either value."
                .to_string();
        }
        if violation == "find_resources lookup mode did not match the current request" {
            return match self.expected_lookup_mode {
                Some("contact") => "This is a contact-detail follow-up. Return one find_resources Tool call with lookup_mode=contact so the lookup preserves the user's jurisdiction even when help_type is unavailable.".to_string(),
                Some("inventory") => "This is a resource inventory request. Return one find_resources Tool call with lookup_mode=inventory so an omitted region remains global.".to_string(),
                _ => self.retry_instruction(),
            };
        }
        self.retry_instruction()
    }
}

pub(crate) fn normalized_lookup_text(input: &str) -> String {
    input
        .to_lowercase()
        .chars()
        .map(|character| match character {
            'á' | 'à' | 'ä' | 'â' => 'a',
            'é' | 'è' | 'ë' | 'ê' => 'e',
            'í' | 'ì' | 'ï' | 'î' => 'i',
            'ó' | 'ò' | 'ö' | 'ô' => 'o',
            'ú' | 'ù' | 'ü' | 'û' => 'u',
            'ñ' => 'n',
            character if character.is_alphanumeric() => character,
            _ => ' ',
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn contains_phrase(normalized: &str, phrase: &str) -> bool {
    normalized == phrase
        || normalized.starts_with(&format!("{} ", phrase))
        || normalized.ends_with(&format!(" {}", phrase))
        || normalized.contains(&format!(" {} ", phrase))
}

fn has_contact_lookup_cue(normalized: &str) -> bool {
    [
        "email",
        "e mail",
        "correo",
        "correo electronico",
        "phone",
        "phone number",
        "telephone",
        "telefono",
        "celular",
        "website",
        "web site",
        "url",
        "sitio web",
        "pagina web",
        "what is the address",
        "where is the address",
        "address is",
        "address",
        "postal address",
        "direccion",
        "domicilio",
        "secure channel",
        "secure contact",
        "encrypted channel",
        "contact information",
        "contact info",
        "contact details",
        "canal seguro",
        "contacto seguro",
        "canal cifrado",
        "informacion de contacto",
        "datos de contacto",
    ]
    .iter()
    .any(|cue| {
        let present = contains_phrase(normalized, cue);
        if !present {
            return false;
        }
        if *cue == "address" {
            if normalized.contains("physical address")
                || normalized.contains("mailing address")
                || normalized.contains("postal address")
            {
                return true;
            }
            // “Address this/the/my concern” is ordinary prose, not a request
            // for a physical contact address. Other contact cues in a mixed
            // sentence are still allowed to establish the expectation.
            for non_contact in [
                "address this",
                "address the",
                "address my",
                "address your",
                "address our",
                "address a",
                "address an",
            ] {
                if normalized.contains(non_contact) {
                    return false;
                }
            }
        }
        true
    })
}

fn contains_only_contact_methods(value: &str) -> bool {
    let mut remainder = value.to_string();
    let mut removed_method = false;
    for method in [
        "contact information",
        "informacion de contacto",
        "contact details",
        "contact info",
        "datos de contacto",
        "physical address",
        "mailing address",
        "postal address",
        "email address",
        "e mail",
        "phone number",
        "correo electronico",
        "secure channel",
        "canal seguro",
        "web site",
        "sitio web",
        "telephone",
        "telefono",
        "celular",
        "website",
        "direccion",
        "address",
        "correo",
        "phone",
        "email",
        "url",
    ] {
        if contains_phrase(&remainder, method) {
            removed_method = true;
            remainder = remainder.replace(method, " ");
        }
    }
    removed_method
        && remainder.split_whitespace().all(|word| {
            [
                "the", "their", "its", "and", "or", "el", "la", "los", "las", "su", "sus", "y", "o",
            ]
            .contains(&word)
        })
}

fn is_contact_lookup_request(normalized: &str) -> bool {
    if !has_contact_lookup_cue(normalized) {
        return false;
    }
    if contact_lookup_subject(normalized).is_some() {
        return true;
    }
    let methods = [
        "email",
        "e mail",
        "email address",
        "correo",
        "correo electronico",
        "phone",
        "phone number",
        "telephone",
        "telefono",
        "celular",
        "website",
        "web site",
        "url",
        "sitio web",
        "address",
        "physical address",
        "mailing address",
        "postal address",
        "direccion",
        "secure channel",
        "canal seguro",
        "contact information",
        "contact info",
        "contact details",
        "informacion de contacto",
        "datos de contacto",
    ];
    if methods.contains(&normalized) {
        return true;
    }
    if ["need the", "necesito"]
        .iter()
        .any(|phrase| contains_phrase(normalized, phrase))
        && methods.iter().any(|method| normalized.ends_with(method))
    {
        return true;
    }
    let requests_context_value = [
        "what is",
        "what s",
        "what are",
        "where is",
        "give me",
        "can you give me",
        "could you give me",
        "can you share",
        "could you share",
        "can i have",
        "can i get",
        "could i have",
        "could i get",
        "need the",
        "do you have",
        "share",
        "share the",
        "share their",
        "please share",
        "please send",
        "send me",
        "is listed",
        "are listed",
        "cual es",
        "cuales son",
        "donde esta",
        "me das",
        "me puedes dar",
        "puedes darme",
        "puedes dar",
        "necesito",
        "tienes",
    ];
    if requests_context_value.iter().any(|request| {
        normalized.match_indices(request).any(|(start, matched)| {
            let starts_at_word = start == 0
                || normalized[..start]
                    .chars()
                    .next_back()
                    .is_some_and(|character| !character.is_alphanumeric());
            let remainder = normalized[start + matched.len()..].trim_start();
            starts_at_word && contains_only_contact_methods(remainder)
        })
    }) {
        return true;
    }
    methods.iter().any(|method| {
        ["the", "their", "its", "el", "la", "su", "sus"]
            .iter()
            .any(|reference| {
                normalized
                    .strip_suffix(&format!(" {reference} {method}"))
                    .is_some_and(|prefix| requests_context_value.contains(&prefix.trim()))
            })
    })
}

const NON_COUNTRY_RESOURCE_REGIONS: &[&str] = &[
    "mexico city",
    "england",
    "europe",
    "latin america",
    "central america",
    "south america",
];

fn iso_country(value: &str) -> Option<CountryCode> {
    let normalized = normalized_lookup_text(value);
    match normalized.as_str() {
        "usa" | "united states" | "united states of america" => {
            return CountryCode::for_alpha2("US").ok()
        }
        "uk" | "england" => return CountryCode::for_alpha2("GB").ok(),
        "bolivia" => return CountryCode::for_alpha2("BO").ok(),
        "brunei" => return CountryCode::for_alpha2("BN").ok(),
        "iran" => return CountryCode::for_alpha2("IR").ok(),
        "laos" => return CountryCode::for_alpha2("LA").ok(),
        "moldova" => return CountryCode::for_alpha2("MD").ok(),
        "north korea" => return CountryCode::for_alpha2("KP").ok(),
        "palestine" => return CountryCode::for_alpha2("PS").ok(),
        "russia" => return CountryCode::for_alpha2("RU").ok(),
        "south korea" => return CountryCode::for_alpha2("KR").ok(),
        "syria" => return CountryCode::for_alpha2("SY").ok(),
        "tanzania" => return CountryCode::for_alpha2("TZ").ok(),
        "venezuela" => return CountryCode::for_alpha2("VE").ok(),
        "vietnam" => return CountryCode::for_alpha2("VN").ok(),
        value if value.len() == 2 => return CountryCode::for_alpha2_caseless(value).ok(),
        value if value.len() == 3 => return CountryCode::for_alpha3_caseless(value).ok(),
        _ => {}
    }
    CountryCode::iter()
        .find(|country| normalized_lookup_text(country.name()) == normalized)
        .copied()
}

fn canonical_resource_region(value: &str) -> Option<String> {
    let normalized = normalized_lookup_text(value);
    if let Some(country) = iso_country(&normalized) {
        return Some(country.alpha2().to_string());
    }
    NON_COUNTRY_RESOURCE_REGIONS
        .contains(&normalized.as_str())
        .then_some(normalized)
}

fn split_resource_geographic_qualifier(query: &str) -> Option<(String, String)> {
    for separator in [" in ", " en "] {
        let Some((subject, region)) = query.rsplit_once(separator) else {
            continue;
        };
        let subject = subject.trim();
        let region = region.trim();
        let unambiguous_alpha2 = region.len() != 2
            || region
                .chars()
                .filter(|character| character.is_alphabetic())
                .all(char::is_uppercase);
        if !subject.is_empty() && unambiguous_alpha2 && canonical_resource_region(region).is_some()
        {
            return Some((subject.to_string(), region.to_string()));
        }
    }
    None
}

fn split_resource_geographic_qualifier_from_input(
    query: &str,
    source_input: &str,
) -> Option<(String, String)> {
    if let Some(qualifier) = split_resource_geographic_qualifier(query) {
        return Some(qualifier);
    }
    for separator in [" in ", " en "] {
        let Some((subject, region)) = query.rsplit_once(separator) else {
            continue;
        };
        let region = region.trim();
        if subject.trim().is_empty() || region.len() != 2 {
            continue;
        }
        let uppercase_region = region.to_uppercase();
        if source_input.contains(&format!("{separator}{uppercase_region}"))
            && canonical_resource_region(&uppercase_region).is_some()
        {
            return Some((subject.trim().to_string(), uppercase_region));
        }
    }
    None
}

fn context_entity_token(value: &str) -> &str {
    value.trim_matches(|character: char| {
        !character.is_alphanumeric() && character != '\'' && character != '’' && character != '-'
    })
}

fn is_capitalized_entity_token(value: &str) -> bool {
    context_entity_token(value)
        .chars()
        .find(|character| character.is_alphabetic())
        .is_some_and(char::is_uppercase)
}

fn push_context_resource(
    resources: &mut Vec<(String, Option<String>)>,
    candidate: &str,
    explicit_region: Option<&str>,
) {
    let candidate = candidate.trim();
    if candidate.is_empty() {
        return;
    }
    let (query, region) =
        if let Some((query, region)) = split_resource_geographic_qualifier(candidate) {
            (query, Some(region))
        } else {
            (candidate.to_string(), explicit_region.map(str::to_string))
        };
    let query = normalized_lookup_text(&query);
    let token_count = query.split_whitespace().count();
    let acronym = candidate
        .chars()
        .filter(|character| character.is_alphabetic())
        .count()
        >= 2
        && candidate
            .chars()
            .filter(|character| character.is_alphabetic())
            .all(char::is_uppercase);
    let organization_cue = query.split_whitespace().any(|token| {
        [
            "aid",
            "network",
            "center",
            "centre",
            "foundation",
            "organization",
            "association",
            "clinic",
            "shelter",
            "services",
            "service",
            "council",
            "coalition",
            "project",
            "initiative",
            "support",
            "help",
            "need",
            "resource",
            "resources",
            "relief",
            "alliance",
            "society",
            "fund",
        ]
        .contains(&token)
    });
    if query.is_empty() || !acronym && (token_count < 2 || !organization_cue) {
        return;
    }
    let region = region.and_then(|region| canonical_resource_region(&region));
    resources.retain(|(existing, _)| existing != &query);
    resources.push((query, region));
}

fn context_resources_from_text(text: &str) -> Vec<(String, Option<String>)> {
    let mut resources = Vec::new();
    for segment in text.split(['\n', '.', '?', '!', ';', ',']) {
        let mut sequence = Vec::new();
        let flush = |sequence: &mut Vec<&str>, resources: &mut Vec<(String, Option<String>)>| {
            if !sequence.is_empty() {
                push_context_resource(resources, &sequence.join(" "), None);
                sequence.clear();
            }
        };
        for raw_token in segment.split_whitespace() {
            let token = context_entity_token(raw_token);
            let normalized_token = normalized_lookup_text(token);
            if sequence.is_empty()
                && [
                    "a",
                    "an",
                    "the",
                    "then",
                    "earlier",
                    "previously",
                    "user",
                    "assistant",
                    "i",
                    "we",
                ]
                .contains(&normalized_token.as_str())
            {
                continue;
            }
            let connector = ["and", "of", "the", "in", "for", "de", "del", "la", "en"]
                .contains(&normalized_token.as_str());
            if is_capitalized_entity_token(token) || (connector && !sequence.is_empty()) {
                sequence.push(token);
            } else {
                flush(&mut sequence, &mut resources);
            }
        }
        flush(&mut sequence, &mut resources);
    }
    resources
}

impl CuratedResourceQueryContext {
    pub(crate) fn from_trusted_text(text: &str) -> Self {
        Self {
            resources: context_resources_from_text(text),
        }
    }

    fn push_structured_resource(&mut self, query: &str, region: Option<&str>) {
        let query = query.trim();
        let normalized = normalized_lookup_text(query);
        if normalized.is_empty()
            || canonical_resource_region(query).is_some()
            || contains_only_contact_methods(&normalized)
        {
            return;
        }
        let region = region.and_then(canonical_resource_region);
        self.resources
            .retain(|(existing, _)| existing != &normalized);
        self.resources.push((normalized, region));
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }
}

fn prefer_structured_resource_context(
    prose_context: CuratedResourceQueryContext,
    structured_context: CuratedResourceQueryContext,
) -> CuratedResourceQueryContext {
    if structured_context.resources.is_empty() {
        prose_context
    } else {
        structured_context
    }
}

fn resource_regions_match(expected: &str, actual: &str) -> bool {
    match (
        canonical_resource_region(expected),
        canonical_resource_region(actual),
    ) {
        (Some(expected), Some(actual)) => expected == actual,
        _ => normalized_lookup_text(expected) == normalized_lookup_text(actual),
    }
}

fn strip_contact_request_prefixes(mut candidate: &str) -> &str {
    for request_prefix in [
        "what is the ",
        "what s the ",
        "what is ",
        "what s ",
        "do you have the ",
        "do you have ",
        "where is the ",
        "where is ",
        "give me the ",
        "give me ",
        "please give me the ",
        "please give me ",
        "can you give me the ",
        "can you give me ",
        "could you give me the ",
        "could you give me ",
        "can i have the ",
        "can i have ",
        "can i get the ",
        "can i get ",
        "could i have the ",
        "could i have ",
        "could i get the ",
        "could i get ",
        "i need the ",
        "i need ",
        "can you share the ",
        "can you share ",
        "could you share the ",
        "could you share ",
        "please share the ",
        "please share ",
        "share the ",
        "share ",
        "send me the ",
        "send me ",
        "cual es el ",
        "cual es la ",
        "cual es ",
        "cuales son los ",
        "cuales son las ",
        "cuales son ",
        "donde esta el ",
        "donde esta la ",
        "donde esta ",
        "me das el ",
        "me das la ",
        "me das ",
        "me puedes dar el ",
        "me puedes dar la ",
        "me puedes dar ",
        "puedes darme el ",
        "puedes darme la ",
        "puedes darme ",
        "dame el ",
        "dame la ",
        "dame ",
        "necesito el ",
        "necesito la ",
        "necesito ",
        "tienes el ",
        "tienes la ",
        "tienes ",
    ] {
        candidate = candidate.strip_prefix(request_prefix).unwrap_or(candidate);
    }
    candidate.trim()
}

fn contact_lookup_subject(normalized: &str) -> Option<String> {
    let is_context_reference = |candidate: &str| {
        [
            "",
            "a",
            "an",
            "the",
            "their",
            "its",
            "my",
            "your",
            "this",
            "that",
            "it",
            "the organization",
            "this organization",
            "that organization",
            "un",
            "una",
            "el",
            "la",
            "los",
            "las",
            "su",
            "sus",
            "esta",
            "este",
            "esa",
            "ese",
            "la organizacion",
            "esta organizacion",
            "esa organizacion",
        ]
        .contains(&candidate.trim())
    };
    let trim_context = |candidate: &str| {
        let mut candidate = candidate.trim();
        for suffix in [" please", " por favor"] {
            candidate = candidate.strip_suffix(suffix).unwrap_or(candidate).trim();
        }
        (!is_context_reference(candidate)).then(|| candidate.to_string())
    };

    // Method-first requests may combine several contact fields before the
    // organization: “Can I get the email and phone number for WLC?”
    for separator in [" for ", " of ", " de ", " para "] {
        let Some((methods, candidate)) = normalized.split_once(separator) else {
            continue;
        };
        if contains_only_contact_methods(strip_contact_request_prefixes(methods)) {
            if let Some(candidate) = trim_context(candidate) {
                return Some(candidate);
            }
        }
    }

    let marker_request_prefixes = [
        "",
        "what is the",
        "what s the",
        "do you have the",
        "where is the",
        "give me the",
        "please give me the",
        "can you give me the",
        "could you give me the",
        "can you share the",
        "could you share the",
        "can i have the",
        "can i get the",
        "could i have the",
        "could i get the",
        "i need the",
        "please share the",
        "share the",
        "send me the",
        "cual es el",
        "cual es la",
        "cuales son los",
        "cuales son las",
        "donde esta el",
        "donde esta la",
        "me das el",
        "me das la",
        "me puedes dar el",
        "me puedes dar la",
        "puedes darme el",
        "puedes darme la",
        "dame el",
        "dame la",
        "necesito el",
        "necesito la",
        "tienes el",
        "tienes la",
    ];
    for marker in [
        "contact information for ",
        "contact details for ",
        "contact info for ",
        "email address for ",
        "email address of ",
        "e mail for ",
        "e mail of ",
        "email for ",
        "email of ",
        "phone number for ",
        "phone number of ",
        "phone for ",
        "phone of ",
        "website for ",
        "website of ",
        "url for ",
        "url of ",
        "correo electronico de ",
        "correo de ",
        "email de ",
        "telefono de ",
        "celular de ",
        "sitio web de ",
        "direccion de ",
        "canal seguro de ",
        "informacion de contacto de ",
        "datos de contacto de ",
        "email para ",
        "correo electronico para ",
        "correo para ",
        "telefono para ",
        "celular para ",
    ] {
        if let Some((prefix, candidate)) = normalized.rsplit_once(marker) {
            if !marker_request_prefixes.contains(&prefix.trim()) {
                continue;
            }
            return trim_context(candidate);
        }
    }

    for request_prefix in [
        "how can i contact ",
        "how do i contact ",
        "como puedo contactar a ",
        "como contacto a ",
    ] {
        let Some(mut candidate) = normalized.strip_prefix(request_prefix) else {
            continue;
        };
        for method_suffix in [
            " by email",
            " via email",
            " by phone",
            " by telephone",
            " through their website",
            " por correo",
            " por telefono",
            " mediante su sitio web",
        ] {
            candidate = candidate.strip_suffix(method_suffix).unwrap_or(candidate);
        }
        return trim_context(candidate);
    }

    if let Some((prefix, methods)) = normalized.rsplit_once(" s ") {
        if contains_only_contact_methods(methods) {
            let mut candidate = prefix.trim();
            for request_prefix in [
                "what is ",
                "what s ",
                "where is ",
                "give me ",
                "please give me ",
                "can you give me ",
                "could you give me ",
                "can i have ",
                "can i get ",
                "could i have ",
                "could i get ",
                "i need ",
                "can you share ",
                "could you share ",
                "please share ",
                "share ",
                "cual es ",
                "donde esta ",
                "me das ",
                "me puedes dar ",
                "puedes darme ",
                "dame ",
                "necesito ",
            ] {
                candidate = candidate.strip_prefix(request_prefix).unwrap_or(candidate);
            }
            if !is_context_reference(candidate) {
                return Some(candidate.to_string());
            }
        }
    }

    // Terse organization-first requests do not need a possessive marker:
    // “WLC contact information” and “WLC phone number”. Try every word
    // boundary and accept only a suffix made entirely of contact methods.
    for (split, _) in normalized.match_indices(' ') {
        let candidate = strip_contact_request_prefixes(&normalized[..split]);
        let methods = normalized[split + 1..].trim();
        let prose_subject = [
            "what is",
            "what s",
            "what are",
            "where is",
            "summarize",
            "share",
            "send",
            "give",
            "draft",
            "review",
            "feedback",
            "wrong",
            "can you share",
            "could you share",
            "can you give me",
            "could you give me",
            "can i get",
            "can i have",
            "could i get",
            "could i have",
            "i need",
            "their",
            "its",
            "this",
            "that",
            "my",
            "your",
            "me das",
            "me puedes dar",
            "puedes darme",
            "dame",
            "necesito",
            "cual es",
            "cuales son",
            "donde esta",
        ]
        .iter()
        .any(|stem| candidate == *stem || candidate.starts_with(&format!("{stem} ")));
        if !prose_subject && contains_only_contact_methods(methods) {
            if let Some(candidate) = trim_context(candidate.strip_suffix(" s").unwrap_or(candidate))
            {
                return Some(candidate);
            }
        }
    }
    None
}

fn quoted_lookup_filter(input: &str) -> Option<&str> {
    let input = input.trim_start();
    for (open, close) in [('\'', '\''), ('"', '"'), ('‘', '’'), ('“', '”')] {
        let Some(remainder) = input.strip_prefix(open) else {
            continue;
        };
        let Some(end) = remainder.find(close) else {
            continue;
        };
        let value = remainder[..end].trim();
        if !value.is_empty() {
            return Some(value);
        }
    }
    None
}

fn explicit_lookup_filter(input: &str) -> Option<String> {
    let lowercase = input.to_ascii_lowercase();
    let markers = [
        "names start with ",
        "name starts with ",
        "matching ",
        "matches ",
        "named ",
        "nombres empiezan con ",
        "nombre empieza con ",
        "coincidentes con ",
        "coincidente con ",
        "llamados ",
        "llamado ",
    ];
    if let Some((start, marker)) = markers
        .iter()
        .filter_map(|marker| lowercase.find(marker).map(|start| (start, *marker)))
        .min_by_key(|(start, _)| *start)
    {
        let start = start + marker.len();
        let remainder = input[start..].trim_start();
        if let Some(filter) = quoted_lookup_filter(remainder) {
            return Some(filter.to_string());
        }
        let punctuation_end = remainder
            .find(['.', ';', '!', '?', '\n'])
            .unwrap_or(remainder.len());
        let mut candidate = remainder[..punctuation_end].trim();
        let candidate_lower = candidate.to_ascii_lowercase();
        if let Some(instruction_start) = [
            " and summarize",
            " and answer",
            " and describe",
            " then summarize",
            " then answer",
            " y resume",
            " y responde",
            " luego resume",
            " luego responde",
        ]
        .iter()
        .filter_map(|boundary| candidate_lower.find(boundary))
        .min()
        {
            candidate = candidate[..instruction_start].trim();
        }
        let candidate = candidate
            .strip_suffix(" please")
            .or_else(|| candidate.strip_suffix(" por favor"))
            .unwrap_or(candidate)
            .trim();
        if !candidate.is_empty() {
            return Some(candidate.to_string());
        }
    }
    None
}

fn clean_inventory_subject(candidate: &str) -> Option<String> {
    let punctuation_end = candidate
        .find(['.', ';', '!', '?', '\n'])
        .unwrap_or(candidate.len());
    let mut candidate = candidate[..punctuation_end].trim();
    if let Some(quoted) = quoted_lookup_filter(candidate) {
        return Some(quoted.to_string());
    }
    let candidate_lower = candidate.to_ascii_lowercase();
    if let Some(instruction_start) = [
        " and summarize",
        " and answer",
        " and describe",
        " then summarize",
        " then answer",
        " y resume",
        " y responde",
        " luego resume",
        " luego responde",
    ]
    .iter()
    .filter_map(|boundary| candidate_lower.find(boundary))
    .min()
    {
        candidate = candidate[..instruction_start].trim();
    }
    let candidate = candidate
        .strip_suffix(" please")
        .or_else(|| candidate.strip_suffix(" por favor"))
        .unwrap_or(candidate)
        .trim();
    let normalized = normalized_lookup_text(candidate);
    let unambiguous_alpha2 = candidate.len() != 2
        || candidate
            .chars()
            .filter(|character| character.is_alphabetic())
            .all(char::is_uppercase);
    if unambiguous_alpha2 && canonical_resource_region(candidate).is_some() {
        return Some(candidate.to_string());
    }
    let generic_recipient_or_help_type = [
        "me",
        "us",
        "them",
        "everyone",
        "anyone",
        "people",
        "users",
        "myself",
        "my family",
        "all",
        "all available",
        "all ready",
        "all curated",
        "the",
        "ready",
        "available",
        "curated",
        "matching",
        "help",
        "support",
        "assistance",
        "legal help",
        "medical help",
        "humanitarian help",
        "food help",
        "shelter help",
        "financial help",
        "psychosocial help",
        "legal",
        "medical",
        "humanitarian",
        "food",
        "shelter",
        "financial",
        "psychosocial",
        "mi",
        "nosotros",
        "ellos",
        "todos",
        "todas",
        "todos disponibles",
        "todas disponibles",
        "todos listos",
        "todas listas",
        "los",
        "las",
        "listos",
        "listas",
        "disponibles",
        "curados",
        "curadas",
        "coincidentes",
        "todas las personas",
        "ayuda",
        "apoyo",
        "asistencia",
        "ayuda legal",
        "ayuda medica",
        "ayuda humanitaria",
        "ayuda financiera",
        "medica",
        "humanitaria",
        "alimentos",
        "refugio",
        "financiera",
        "psicosocial",
    ]
    .contains(&normalized.as_str())
        || [
            "people who ",
            "users who ",
            "someone who ",
            "those who ",
            "personas que ",
            "usuarios que ",
            "alguien que ",
        ]
        .iter()
        .any(|prefix| normalized.starts_with(prefix));
    let token_count = normalized.split_whitespace().count();
    let is_acronym = candidate.chars().any(char::is_alphabetic)
        && candidate
            .chars()
            .filter(|character| character.is_alphabetic())
            .all(char::is_uppercase);
    (!normalized.is_empty() && !generic_recipient_or_help_type && (token_count > 1 || is_acronym))
        .then(|| candidate.to_string())
}

fn explicit_inventory_subject(input: &str) -> Option<String> {
    let lowercase = input.to_ascii_lowercase();
    let relation_markers = [
        "resources for ",
        "resource for ",
        "organizations for ",
        "organization for ",
        "recursos para ",
        "recurso para ",
        "organizaciones para ",
        "organizacion para ",
        "recursos de ",
        "organizaciones de ",
    ];
    if let Some((start, marker)) = relation_markers
        .iter()
        .filter_map(|marker| lowercase.find(marker).map(|start| (start, *marker)))
        .min_by_key(|(start, _)| *start)
    {
        let candidate = &input[start + marker.len()..];
        if let Some(subject) = clean_inventory_subject(candidate) {
            return Some(subject);
        }
    }

    let trimmed = input.trim().trim_end_matches(['.', ';', '!', '?']);
    let lowercase = trimmed.to_ascii_lowercase();
    for prefix in [
        "show me the ",
        "give me the ",
        "list the ",
        "show me ",
        "give me ",
        "enumerate ",
        "list ",
        "show ",
        "muestrame los ",
        "muestrame las ",
        "muéstrame los ",
        "muéstrame las ",
        "dame los ",
        "dame las ",
        "lista los ",
        "lista las ",
        "muestrame ",
        "muéstrame ",
        "enumera ",
        "lista ",
        "muestra ",
    ] {
        let Some(body) = lowercase.strip_prefix(prefix) else {
            continue;
        };
        for suffix in [
            " curated resources",
            " resource directory entries",
            " ready resources",
            " resources",
            " organizations",
            " recursos curados",
            " recursos listos",
            " recursos",
            " organizaciones",
        ] {
            let Some(subject) = body.strip_suffix(suffix) else {
                continue;
            };
            let start = prefix.len();
            let end = start + subject.len();
            if let Some(subject) = clean_inventory_subject(&trimmed[start..end]) {
                return Some(subject);
            }
        }
    }
    None
}

fn explicit_inventory_region(input: &str) -> Option<String> {
    let lowercase = input.to_ascii_lowercase();
    for marker in [
        " available in ",
        " disponibles en ",
        "resources are available in ",
        "organizations are available in ",
        "resources available in ",
        "organizations available in ",
        "resources for ",
        "organizations for ",
        "resources in ",
        "organizations in ",
        "recursos están disponibles en ",
        "organizaciones están disponibles en ",
        "recursos estan disponibles en ",
        "organizaciones estan disponibles en ",
        "recursos disponibles en ",
        "organizaciones disponibles en ",
        "recursos para ",
        "organizaciones para ",
        "recursos en ",
        "organizaciones en ",
    ] {
        let Some(start) = lowercase.find(marker) else {
            continue;
        };
        let candidate = input[start + marker.len()..]
            .split(['.', ';', '!', '?', '\n'])
            .next()
            .unwrap_or_default()
            .trim();
        if let Some(region) = canonical_resource_region(candidate) {
            return Some(region);
        }
    }
    None
}

fn is_resource_explanation_request(normalized: &str) -> bool {
    if contains_phrase(normalized, "how many") || !contains_phrase(normalized, "how") {
        return false;
    }
    [
        "work",
        "works",
        "operate",
        "operates",
        "are selected",
        "are curated",
        "can help",
        "provide help",
    ]
    .iter()
    .any(|phrase| contains_phrase(normalized, phrase))
}

/// Derive a narrow validation requirement from the raw current User message.
/// The requirement can reject an incomplete model-generated plan, but it does
/// not create, authorize, or execute a Tool call.
pub(crate) fn curated_resource_lookup_expectation(
    input: &str,
    continuation_cursor: Option<&CuratedResourceContinuation>,
) -> CuratedResourceLookupExpectation {
    let normalized = normalized_lookup_text(input);
    let contact = is_contact_lookup_request(&normalized);
    let resource_noun = [
        "resource",
        "resources",
        "curated resource",
        "curated resources",
        "recurso",
        "recursos",
        "recursos curados",
        "organization",
        "organizations",
        "organizacion",
        "organizaciones",
        "resource directory",
        "directory",
        "directorio de recursos",
        "directorio",
    ]
    .iter()
    .any(|phrase| contains_phrase(&normalized, phrase));
    let inventory_action = [
        "list",
        "show",
        "see",
        "give me",
        "all",
        "inventory",
        "enumerate",
        "how many",
        "available",
        "ready",
        "what resources",
        "which resources",
        "are there",
        "next page",
        "following page",
        "more resources",
        "continue",
        "lista",
        "listar",
        "muestra",
        "mostrar",
        "ver",
        "dame",
        "todos",
        "todas",
        "inventario",
        "enumera",
        "cuantos",
        "cuantas",
        "disponibles",
        "listos",
        "siguiente pagina",
        "pagina siguiente",
        "mas recursos",
        "continua",
        "hay",
    ]
    .iter()
    .any(|phrase| contains_phrase(&normalized, phrase));
    let continuation_action = [
        "next page",
        "following page",
        "show me more",
        "show more resources",
        "more resources",
        "continue",
        "continue listing resources",
        "siguiente pagina",
        "pagina siguiente",
        "muestra mas",
        "muestra mas recursos",
        "mas recursos",
        "continua",
        "continua listando recursos",
    ]
    .iter()
    .any(|phrase| contains_phrase(&normalized, phrase));
    let is_continuation = continuation_action && (resource_noun || continuation_cursor.is_some());
    let inventory = ((resource_noun && inventory_action) || is_continuation)
        && !is_resource_explanation_request(&normalized);
    let inventory_subject = inventory
        .then(|| explicit_inventory_subject(input))
        .flatten();
    let subject_region = inventory_subject
        .as_deref()
        .and_then(canonical_resource_region);
    let inventory_region = inventory
        .then(|| explicit_inventory_region(input))
        .flatten()
        .or_else(|| subject_region.clone());
    let inventory_subject_is_region = subject_region.is_some();
    let filtered_inventory = inventory
        && ([
            "names start",
            "name starts",
            "named",
            "matching",
            "matches",
            "nombres empiezan",
            "nombre empieza",
            "llamado",
            "llamados",
            "coincidente",
            "coincidentes",
        ]
        .iter()
        .any(|phrase| contains_phrase(&normalized, phrase))
            || inventory_subject.is_some())
        && !inventory_subject_is_region;
    let expected_filter = filtered_inventory
        .then(|| explicit_lookup_filter(input).or(inventory_subject))
        .flatten();
    let mut expected_query = if is_continuation {
        continuation_cursor.and_then(|cursor| cursor.query.clone())
    } else if filtered_inventory {
        expected_filter.clone()
    } else if contact {
        contact_lookup_subject(&normalized)
    } else {
        None
    };
    let geographic_qualifier = (!is_continuation)
        .then(|| {
            expected_query
                .as_deref()
                .and_then(|query| split_resource_geographic_qualifier_from_input(query, input))
        })
        .flatten();
    let expected_region = if is_continuation {
        continuation_cursor.and_then(|cursor| cursor.region.clone())
    } else {
        geographic_qualifier
            .map(|(query, region)| {
                expected_query = Some(query);
                region
            })
            .or(inventory_region)
    };
    let continuation = is_continuation.then_some(continuation_cursor).flatten();
    let unfiltered_inventory = inventory && !filtered_inventory && !is_continuation;
    let fresh_region_must_be_absent = unfiltered_inventory && expected_region.is_none();
    CuratedResourceLookupExpectation {
        required: contact || inventory,
        query_required: contact
            || filtered_inventory
            || (is_continuation
                && continuation_cursor.is_some_and(|cursor| cursor.query.is_some())),
        positive_offset_required: is_continuation,
        expected_query,
        expected_region,
        expected_help_type: continuation.and_then(|cursor| cursor.help_type.clone()),
        expected_language: continuation.and_then(|cursor| cursor.language.clone()),
        expected_offset: if is_continuation {
            continuation_cursor.map(|cursor| cursor.next_offset)
        } else {
            None
        },
        query_must_be_absent: is_continuation
            && continuation.is_some_and(|cursor| cursor.query.is_none())
            || unfiltered_inventory,
        region_must_be_absent: is_continuation
            && continuation.is_some_and(|cursor| cursor.region.is_none())
            || fresh_region_must_be_absent,
        help_type_must_be_absent: inventory
            && (!is_continuation || continuation.is_some_and(|cursor| cursor.help_type.is_none())),
        language_must_be_absent: is_continuation
            && continuation.is_some_and(|cursor| cursor.language.is_none())
            || unfiltered_inventory,
        continuation_cursor_missing: is_continuation && continuation_cursor.is_none(),
        exact_filter_missing: filtered_inventory && !is_continuation && expected_filter.is_none(),
        initial_offset_required: (contact || inventory) && !is_continuation,
        expected_lookup_mode: if is_continuation {
            continuation_cursor
                .and_then(|cursor| cursor.lookup_mode.as_deref())
                .and_then(|mode| match mode {
                    "contact" => Some("contact"),
                    "inventory" => Some("inventory"),
                    _ => None,
                })
        } else if contact {
            Some("contact")
        } else if inventory {
            Some("inventory")
        } else {
            None
        },
        lookup_mode_must_be_absent: is_continuation
            && continuation_cursor.is_some_and(|cursor| cursor.lookup_mode.is_none()),
        contact_query_from_context: contact
            && !is_continuation
            && contact_lookup_subject(&normalized).is_none(),
        context_grounded_resources: Vec::new(),
    }
}

#[cfg(test)]
pub(crate) fn expects_curated_resource_lookup(input: &str) -> bool {
    curated_resource_lookup_expectation(input, None).required
}

/// Correction agent signature for fixing malformed responses
///
/// This agent takes a malformed response and reshapes it into the correct format.
/// It should preserve the intent/content of the original response, not generate new content.
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
#[derive(Clone, Debug)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    /// Structured, Tool-owned execution facts used by runtime policy. Raw
    /// prompt text is never authoritative for these values.
    pub metadata: serde_json::Value,
    /// Optional Tool-owned text that is already safe to show without another
    /// model pass. This is intentionally separate from the internal Tool
    /// output, which can contain instructions for the final-answer model.
    pub user_safe_fallback: Option<UserSafeToolFallback>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UserSafeToolFallbackKind {
    CuratedResourceInventory,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UserSafeToolFallback {
    pub kind: UserSafeToolFallbackKind,
    pub output: String,
}

/// Failure categories that the shared Tool executor can safely classify.
/// Read-only adapters preserve this type through `anyhow` so retry policy does
/// not depend on parsing provider or backend error strings.
#[derive(Debug, thiserror::Error)]
pub enum ToolExecutionError {
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

/// Explicit retry/timeout contract for a Tool. The default is no retry; only
/// read-only Tools opt into the bounded policy constructors below.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolRetryPolicy {
    None,
    ReadOnly {
        per_attempt_timeout: Duration,
        max_attempts: u32,
        total_budget: Duration,
        backoff: Duration,
    },
}

impl ToolRetryPolicy {
    const MIN_RETRY_ATTEMPT_BUDGET_CAP: Duration = Duration::from_secs(1);
    pub fn none() -> Self {
        Self::None
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

    fn attempt_timeout(&self, remaining: Duration) -> Option<Duration> {
        match self {
            Self::None => None,
            Self::ReadOnly {
                per_attempt_timeout,
                ..
            } => Some((*per_attempt_timeout).min(remaining)),
        }
    }

    fn can_retry(&self, attempt: u32, remaining: Duration) -> bool {
        match self {
            Self::None => false,
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
            Self::None => Duration::ZERO,
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
            user_safe_fallback: None,
        }
    }

    pub fn success_with_metadata(output: impl Into<String>, metadata: serde_json::Value) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
            metadata,
            user_safe_fallback: None,
        }
    }

    pub fn success_with_user_safe_fallback(
        output: impl Into<String>,
        metadata: serde_json::Value,
        kind: UserSafeToolFallbackKind,
        user_safe_output: impl Into<String>,
    ) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
            metadata,
            user_safe_fallback: Some(UserSafeToolFallback {
                kind,
                output: user_safe_output.into(),
            }),
        }
    }

    pub fn error(error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: String::new(),
            error: Some(error.into()),
            metadata: serde_json::Value::Null,
            user_safe_fallback: None,
        }
    }
}

/// Trait for tools that can be executed by the agent
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn args_schema(&self) -> &str;
    fn retry_policy(&self) -> ToolRetryPolicy {
        ToolRetryPolicy::None
    }
    async fn execute_with_timing_outcome(
        &self,
        args: &ToolArgs,
    ) -> Result<(ToolResult, ConversationTimingOutcome)> {
        let result = self.execute(args).await?;
        let outcome = if result.success {
            ConversationTimingOutcome::Succeeded
        } else {
            ConversationTimingOutcome::Failed
        };
        Ok((result, outcome))
    }
    async fn execute(&self, args: &ToolArgs) -> Result<ToolResult>;
}

/// Description-only Tool stub for generating prompt text without live backends.
struct ToolDescriptor {
    name: String,
    description: String,
    args_schema: String,
}

#[async_trait::async_trait]
impl Tool for ToolDescriptor {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn args_schema(&self) -> &str {
        &self.args_schema
    }
    async fn execute(&self, _args: &ToolArgs) -> Result<ToolResult> {
        unreachable!("ToolDescriptor is description-only and should never be executed")
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

    /// Whether planning can produce a Tool call with an observable effect.
    /// `done` is only the legacy loop terminator and must not force a planning
    /// model call for an otherwise tool-free conversation.
    pub fn has_actionable_tools(&self) -> bool {
        self.tools.keys().any(|name| name != "done")
    }

    /// Return enabled Tool names in stable registry order for content-free
    /// planning observations. No Tool arguments or output are included.
    pub fn names(&self) -> Vec<String> {
        self.tools
            .keys()
            .filter(|name| name.as_str() != "done")
            .cloned()
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
                tool.args_schema()
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
    fn register_descriptor(&mut self, name: &str, description: &str, args_schema: &str) {
        self.register(Arc::new(ToolDescriptor {
            name: name.to_string(),
            description: description.to_string(),
            args_schema: args_schema.to_string(),
        }));
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
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
        round: usize,
        attempt: u32,
        enabled_tools: Vec<String>,
        raw_selected_tools: Vec<String>,
        selected_tools: Vec<String>,
        expected_curated_resources: bool,
        curated_resources_available: bool,
        missed_expected_curated_resources: bool,
        violation_reason: Option<String>,
        outcome: String,
    },
    ToolAttempted {
        call_id: String,
        tool_name: String,
        planning_round: usize,
        attempt: u32,
    },
    ToolTerminal {
        call_id: String,
        tool_name: String,
        planning_round: usize,
        attempt: u32,
        status: String,
        elapsed_ms: u128,
    },
    ToolRetryScheduled {
        call_id: String,
        tool_name: String,
        planning_round: usize,
        attempt: u32,
        reason: String,
    },
    ToolTimedOut {
        call_id: String,
        tool_name: String,
        planning_round: usize,
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
    ProviderReasoning {
        step: usize,
        content: String,
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
        planning_round: Option<usize>,
        tool_name: Option<String>,
        call_id: Option<String>,
        attempt: u32,
        outcome: ConversationTimingOutcome,
        elapsed_ms: u128,
    },
    TimingUnavailable {
        phase: ConversationTimingPhase,
        planning_round: Option<usize>,
        attempt: u32,
        reason: &'static str,
    },
}

pub type AgentTraceHook = Arc<dyn Fn(AgentTraceEvent) + Send + Sync>;
pub type ProviderReasoningTraceHook = Arc<dyn Fn(String) + Send + Sync>;

#[derive(Clone, Debug)]
pub enum ProviderTimingEvent {
    ResponseHeaders {
        attempt: u32,
        elapsed_ms: u128,
        outcome: ConversationTimingOutcome,
    },
    FirstProviderEvent {
        attempt: u32,
        elapsed_ms: u128,
        outcome: ConversationTimingOutcome,
    },
}

pub type ProviderTimingTraceHook = Arc<dyn Fn(ProviderTimingEvent) + Send + Sync>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConversationTimingPhase {
    ToolPlanningModelDuration,
    ToolPlanningClusterScheduling,
    ToolPlanningInference,
    FinalAnswerModelDuration,
    FinalAnswerResponseHeaderWait,
    FinalAnswerFirstProviderEventWait,
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
    TimedOut,
    Guarded,
}

impl ConversationTimingOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::TimedOut => "timed_out",
            Self::Guarded => "guarded",
        }
    }
}

impl ConversationTimingPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ToolPlanningModelDuration => "tool_planning_model_duration",
            Self::ToolPlanningClusterScheduling => "tool_planning_cluster_scheduling",
            Self::ToolPlanningInference => "tool_planning_inference",
            Self::FinalAnswerModelDuration => "final_answer_model_duration",
            Self::FinalAnswerResponseHeaderWait => "final_answer_response_header_wait",
            Self::FinalAnswerFirstProviderEventWait => "final_answer_first_provider_event_wait",
            Self::ToolExecution => "tool_execution",
            Self::ResourceDirectoryLookup => "resource_directory_lookup",
            Self::Retrieval => "retrieval",
            Self::RetryDelay => "retry_delay",
            Self::TotalTurn => "total_turn",
        }
    }

    pub fn is_provider_wait_proxy(self) -> bool {
        matches!(
            self,
            Self::FinalAnswerResponseHeaderWait | Self::FinalAnswerFirstProviderEventWait
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MultilineToolCallHeaderMatch {
    None,
    Partial,
    Complete,
}

pub(crate) fn multiline_tool_call_header_match(candidate: &str) -> MultilineToolCallHeaderMatch {
    let lowercase = candidate.trim_start().to_ascii_lowercase();
    let first_line = lowercase.lines().next().unwrap_or(&lowercase).trim_end();
    if first_line.is_empty() {
        return MultilineToolCallHeaderMatch::None;
    }

    const HEADER: &str = "tool calls";
    const COLON_HEADER: &str = "tool calls:";
    if matches!(first_line, HEADER | COLON_HEADER) {
        MultilineToolCallHeaderMatch::Complete
    } else if HEADER.starts_with(first_line) || COLON_HEADER.starts_with(first_line) {
        MultilineToolCallHeaderMatch::Partial
    } else {
        MultilineToolCallHeaderMatch::None
    }
}

pub(crate) fn has_syntactic_tool_intent(candidate: &str) -> bool {
    let candidate = candidate.trim();
    if candidate.is_empty() {
        return false;
    }
    if serde_json::from_str::<serde_json::Value>(candidate)
        .ok()
        .is_some_and(|value| json_has_tool_intent(&value))
    {
        return true;
    }

    // Use the same apostrophe normalization as the opening classifier's
    // process-narration vocabulary so held text cannot become releasable only
    // because the provider used a typographic apostrophe.
    let lowercase = candidate.to_ascii_lowercase().replace(['’', '‘'], "'");
    let mut nonempty_lines = lowercase
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let has_multiline_header = nonempty_lines.next().is_some_and(|line| {
        multiline_tool_call_header_match(line) == MultilineToolCallHeaderMatch::Complete
    });
    let mut has_function_call = false;
    let mut has_arguments = false;
    for line in nonempty_lines {
        if let Some(name) = line.strip_prefix("function call:") {
            let name = name.trim();
            has_function_call = !name.is_empty()
                && name.chars().all(|character| {
                    character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
                });
        } else if has_function_call && matches!(line, "arguments" | "arguments:") {
            has_arguments = true;
        }
    }
    if has_multiline_header && has_function_call && has_arguments {
        return true;
    }
    if let Some(tool_calls_start) = lowercase.find("tool calls:") {
        let preamble = &lowercase[..tool_calls_start];
        let transcript = &lowercase[tool_calls_start + "tool calls:".len()..];
        let first_line = transcript.lines().next().unwrap_or(transcript);
        let has_invocation = first_line.contains('(') || first_line.contains('{');
        let has_deliberation_preamble = [
            "let me ",
            "i'll search",
            "i will search",
            "i need to ",
            "i should ",
        ]
        .iter()
        .any(|marker| preamble.contains(marker));
        let starts_with_tool_calls = preamble.trim().is_empty();
        if has_invocation
            && (starts_with_tool_calls
                || has_deliberation_preamble
                || transcript.contains("tool result:"))
        {
            return true;
        }
    }
    if provider_neutral_labels_have_tool_intent(candidate) {
        return true;
    }
    if lowercase.starts_with("```") {
        let after_open = candidate.strip_prefix("```").unwrap_or(candidate);
        let fenced = after_open
            .split_once('\n')
            .map(|(_, body)| body)
            .unwrap_or(after_open)
            .strip_suffix("```")
            .unwrap_or(after_open);
        return has_syntactic_tool_intent(fenced);
    }

    if lowercase.starts_with("[[ ##")
        || lowercase.starts_with("<tool_call")
        || lowercase.starts_with("</tool_call")
        || lowercase.starts_with("<|tool_call")
    {
        return true;
    }

    let starts_structured = matches!(candidate.chars().next(), Some('{') | Some('['))
        || [
            "tool_calls:",
            "function_call:",
            "name:",
            "args:",
            "arguments:",
        ]
        .iter()
        .any(|prefix| candidate.starts_with(prefix));
    if !starts_structured {
        return false;
    }

    if [
        "\"tool_calls\"",
        "'tool_calls'",
        "tool_calls:",
        "\"function_call\"",
        "'function_call'",
        "function_call:",
    ]
    .iter()
    .any(|marker| lowercase.contains(marker))
    {
        return true;
    }

    let has_name_field = lowercase.contains("\"name\"")
        || lowercase.contains("'name'")
        || lowercase.starts_with("name:")
        || lowercase.contains("\nname:");
    let has_args_field = lowercase.contains("\"args\"")
        || lowercase.contains("\"arguments\"")
        || lowercase.contains("'args'")
        || lowercase.contains("'arguments'")
        || lowercase.starts_with("args:")
        || lowercase.starts_with("arguments:")
        || lowercase.contains("\nargs:")
        || lowercase.contains("\narguments:");
    has_name_field && has_args_field
}

const LOOKUP_PROCESS_NARRATION_OPENERS: [&str; 2] = ["looking up", "buscando"];

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProcessNarrationOpeningMatch {
    None,
    Partial,
    Complete,
}

#[allow(dead_code)]
pub(crate) fn lookup_process_narration_opening(value: &str) -> ProcessNarrationOpeningMatch {
    let opening = value.trim_start().to_lowercase();
    if LOOKUP_PROCESS_NARRATION_OPENERS
        .iter()
        .any(|candidate| opening.starts_with(candidate))
    {
        return ProcessNarrationOpeningMatch::Complete;
    }
    if !opening.is_empty()
        && LOOKUP_PROCESS_NARRATION_OPENERS
            .iter()
            .any(|candidate| candidate.starts_with(&opening))
    {
        return ProcessNarrationOpeningMatch::Partial;
    }
    ProcessNarrationOpeningMatch::None
}

fn provider_neutral_tool_label_has_invocation(value: &str) -> bool {
    let value = value.trim_start();
    let (first_line, remaining) = value.split_once('\n').unwrap_or((value, ""));
    let first_line = first_line.trim();
    let name_end = first_line
        .find(|character: char| {
            !(character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.'))
        })
        .unwrap_or(first_line.len());
    if name_end == 0 {
        return false;
    }
    let name = &first_line[..name_end];
    if !name
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.'))
    {
        return false;
    }
    let remaining = remaining.trim_start();
    if remaining.starts_with('{') || remaining.starts_with('[') || remaining.starts_with("```json")
    {
        return true;
    }
    if remaining
        .strip_prefix("args:")
        .or_else(|| remaining.strip_prefix("arguments:"))
        .is_some()
    {
        // A valid Tool label/name followed by an explicit argument label is
        // itself a serialized invocation envelope. The provider may render
        // those arguments as JSON, bullets, or key=value prose.
        return true;
    }

    let same_line_suffix = first_line[name_end..].trim_start();
    same_line_suffix.starts_with('(')
        || same_line_suffix.starts_with('{')
        || same_line_suffix
            .strip_prefix("with ")
            .is_some_and(|arguments| {
                // This is fail-closed intent classification, not Tool argument
                // validation. One credible named argument is sufficient even
                // when the provider appends malformed or prose-like items.
                arguments.split(',').any(|argument| {
                    let Some((name, value)) = argument.trim().split_once('=') else {
                        return false;
                    };
                    let name = name.trim();
                    let value = value.trim();
                    !name.is_empty()
                        && name
                            .chars()
                            .all(|character| character.is_ascii_alphanumeric() || character == '_')
                        && !value.is_empty()
                })
            })
}

fn provider_neutral_tool_label_is_bare_name(value: &str) -> bool {
    let name = value.trim();
    !name.is_empty()
        && name.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
        })
}

fn provider_neutral_tool_label_is_explanatory(candidate: &str, start: usize, label: &str) -> bool {
    if !matches!(label, "tool:" | "tool decision:") {
        return false;
    }
    let same_line_preamble = candidate[..start]
        .rsplit('\n')
        .next()
        .unwrap_or_default()
        .trim_end();
    if same_line_preamble.ends_with("curated resources") {
        return true;
    }

    // A reporting/copular predicate directly before `Tool:` makes the label
    // part of documentation prose ("the panel shows Tool: done"). Selection
    // adjectives and process narration ("Selected Tool: ...") do not.
    let normalized_preamble = same_line_preamble
        .trim_end_matches(|character: char| !character.is_alphanumeric() && character != '_');
    let mut words = normalized_preamble.split_whitespace().rev();
    let predicate = words.next().unwrap_or_default();
    let has_subject = words.next().is_some();
    has_subject
        && [
            "is", "are", "reads", "shows", "says", "displays", "uses", "prints", "renders",
            "appears", "means", "as",
        ]
        .contains(&predicate)
}

fn provider_neutral_tool_label_has_lexical_boundary(candidate: &str, start: usize) -> bool {
    if start == 0 {
        return true;
    }
    let Some(previous) = candidate[..start].chars().next_back() else {
        return true;
    };
    if !previous.is_alphanumeric() && !matches!(previous, '_' | '-' | '.') {
        return true;
    }

    // Providers sometimes join a new, capitalized Tool label directly to the
    // preceding sentence. Keep lower-case code identifiers such as
    // `namespace.tool:build` embedded, while treating `.Tool:` and
    // `.Tool decision:` as sentence-level labels.
    previous == '.'
        && candidate[start..]
            .strip_prefix("Tool")
            .is_some_and(|suffix| suffix.starts_with(':') || suffix.starts_with(" decision:"))
}

pub(crate) fn provider_neutral_tool_label_start_at_or_after(
    candidate: &str,
    minimum_start: usize,
) -> Option<usize> {
    let lowercase = candidate.to_ascii_lowercase();
    ["tool:", "tool decision:"]
        .iter()
        .flat_map(|label| lowercase.match_indices(label).map(|(start, _)| start))
        .filter(|start| {
            *start >= minimum_start
                && provider_neutral_tool_label_has_lexical_boundary(candidate, *start)
        })
        .min()
}

fn provider_neutral_tool_label_has_argument_section(value: &str) -> bool {
    let Some((_, remaining)) = value.split_once('\n') else {
        return false;
    };
    let remaining = remaining.trim_start();
    remaining.starts_with("args:") || remaining.starts_with("arguments:")
}

fn provider_neutral_labels_have_tool_intent(candidate: &str) -> bool {
    let lowercase = candidate.to_ascii_lowercase();
    let mut labels = Vec::new();
    for label in ["tool:", "tool decision:"] {
        labels.extend(
            lowercase
                .match_indices(label)
                .map(|(start, _)| (start, label)),
        );
    }
    labels.sort_unstable_by_key(|(start, _)| *start);

    for (start, label) in labels {
        let invocation = &lowercase[start + label.len()..];
        if !provider_neutral_tool_label_has_lexical_boundary(candidate, start)
            && !provider_neutral_tool_label_has_argument_section(invocation)
        {
            continue;
        }
        if provider_neutral_tool_label_has_invocation(invocation) {
            return true;
        }
        if provider_neutral_tool_label_is_bare_name(invocation) {
            if provider_neutral_tool_label_is_explanatory(&lowercase, start, label) {
                continue;
            }
            return true;
        }
    }
    false
}

fn json_has_tool_intent(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Array(values) => values.iter().any(json_has_tool_intent),
        serde_json::Value::Object(object) => {
            object.contains_key("tool_calls")
                || object.contains_key("function_call")
                || (object.contains_key("name")
                    && (object.contains_key("args") || object.contains_key("arguments")))
                || object.values().any(json_has_tool_intent)
        }
        _ => false,
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
    curated_resource_lookup_expectation: CuratedResourceLookupExpectation,
    curated_resource_lookup_succeeded: bool,
    curated_resource_page_incomplete: bool,
    max_steps: usize,
    turn_step_index: usize,
    trace_hook: Option<AgentTraceHook>,
    final_answer_attempt: Arc<AtomicU32>,
}

#[allow(dead_code)]
impl SageAgent {
    const MAX_CURRENT_TOOL_RESULT_CHARS: usize = 4_000;
    const MAX_CURRENT_TOOL_CONTEXT_CHARS: usize = 12_000;

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
            curated_resource_lookup_expectation: CuratedResourceLookupExpectation::default(),
            curated_resource_lookup_succeeded: false,
            curated_resource_page_incomplete: false,
            max_steps: 10,
            turn_step_index: 0,
            trace_hook: None,
            final_answer_attempt: Arc::new(AtomicU32::new(1)),
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
        self.curated_resource_lookup_succeeded = false;
        self.curated_resource_page_incomplete = false;
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

    /// Trusted, server-built text in which a model-selected organization query
    /// must be grounded for a context-only contact follow-up. This mirrors the
    /// memory fields shown to Tool planning without making the runtime choose
    /// a query or a Tool call.
    pub(crate) fn curated_resource_query_context(&self) -> CuratedResourceQueryContext {
        let Some(memory) = &self.memory else {
            return CuratedResourceQueryContext::default();
        };
        let Ok((summary, mut messages)) = memory.get_context_messages() else {
            return CuratedResourceQueryContext::default();
        };
        if messages.last().is_none_or(|message| message.role != "user") {
            // The current user message is persisted immediately before Tool
            // planning. If that positional invariant is absent, do not risk
            // grounding a query in the current untrusted request.
            return CuratedResourceQueryContext::default();
        }
        messages.pop();

        let mut trusted_text = summary
            .as_ref()
            .map(|summary| summary.content.clone())
            .unwrap_or_default();
        for message in &messages {
            trusted_text.push('\n');
            trusted_text.push_str(&message.content);
        }
        let prose_context = CuratedResourceQueryContext::from_trusted_text(&trusted_text);
        let mut structured_context = CuratedResourceQueryContext::default();
        for message in messages
            .iter()
            .filter(|message| message.role == "assistant")
        {
            let Some(tools) = message
                .tool_results
                .as_ref()
                .and_then(|metadata| metadata.pointer("/conversation_trace/tools"))
                .and_then(serde_json::Value::as_array)
            else {
                continue;
            };
            for tool in tools.iter().filter(|tool| {
                tool.get("id").and_then(serde_json::Value::as_str) == Some("curated-resources")
            }) {
                let Some(metadata) = tool.get("metadata") else {
                    continue;
                };
                let region = metadata
                    .get("resolved_region")
                    .and_then(serde_json::Value::as_str);
                let resource_names = metadata
                    .get("resource_names")
                    .and_then(serde_json::Value::as_array);
                if let Some(resource_names) = resource_names {
                    for name in resource_names.iter().filter_map(serde_json::Value::as_str) {
                        structured_context.push_structured_resource(name, region);
                    }
                }
                if resource_names.is_none_or(Vec::is_empty) {
                    if let Some(query) = metadata
                        .get("continuation_query")
                        .and_then(serde_json::Value::as_str)
                    {
                        structured_context.push_structured_resource(query, region);
                    }
                }
            }
        }
        prefer_structured_resource_context(prose_context, structured_context)
    }

    /// Build the plain final-answer prompt after the bounded Tool phase.
    pub fn plain_answer_prompt(&self, user_message: &str) -> PlainAnswerPrompt {
        let context = self.build_context();
        let system = format!("{}{}", self.instruction, PLAIN_ANSWER_INSTRUCTION);
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
        PlainAnswerPrompt {
            system,
            user,
            incomplete_curated_resource_page: self.curated_resource_page_incomplete,
        }
    }

    /// Execute one provider-neutral Tool decision and retain its results for
    /// either an explicitly requested replan or plain final-answer generation.
    pub async fn execute_tool_decision(&mut self, decision: &ToolDecision) -> StepResult {
        let mut executed_tools = Vec::new();
        for tool_call in &decision.tool_calls {
            let call_id = format!("tool-call-{}", Uuid::new_v4().simple());
            let planning_round = decision.planning_round;
            let result = self
                .execute_tool_call(&call_id, planning_round, tool_call)
                .await;
            self.inject_tool_result(tool_call, &result);
            if tool_call.name == "find_resources" && result.success {
                self.curated_resource_lookup_succeeded = true;
                self.curated_resource_page_incomplete = result
                    .metadata
                    .get("has_more")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true);
            }
            executed_tools.push(ExecutedTool {
                tool_call: tool_call.clone(),
                result,
            });
        }

        StepResult {
            messages: Vec::new(),
            tool_calls: decision.tool_calls.clone(),
            executed_tools,
            done: decision.tool_calls.is_empty(),
        }
    }

    async fn execute_tool_call(
        &self,
        call_id: &str,
        planning_round: usize,
        tool_call: &ToolCall,
    ) -> ToolResult {
        let Some(tool) = self.tools.get(&tool_call.name) else {
            self.emit_trace(AgentTraceEvent::ToolAttempted {
                call_id: call_id.to_string(),
                tool_name: tool_call.name.clone(),
                planning_round,
                attempt: 1,
            });
            let result = ToolResult::error(format!("Unknown tool: {}", tool_call.name));
            self.emit_trace(AgentTraceEvent::Timing {
                phase: ConversationTimingPhase::ToolExecution,
                planning_round: Some(planning_round),
                tool_name: Some(tool_call.name.clone()),
                call_id: Some(call_id.to_string()),
                attempt: 1,
                outcome: ConversationTimingOutcome::Failed,
                elapsed_ms: 0,
            });
            self.emit_trace(AgentTraceEvent::ToolTerminal {
                call_id: call_id.to_string(),
                tool_name: tool_call.name.clone(),
                planning_round,
                attempt: 1,
                status: "failed".to_string(),
                elapsed_ms: 0,
            });
            return result;
        };

        let policy = tool.retry_policy();
        let call_started_at = Instant::now();
        let mut attempt = 1;
        let (result, terminal_status, elapsed_ms) = loop {
            self.emit_trace(AgentTraceEvent::ToolAttempted {
                call_id: call_id.to_string(),
                tool_name: tool_call.name.clone(),
                planning_round,
                attempt,
            });

            let elapsed = call_started_at.elapsed();
            let remaining = match &policy {
                ToolRetryPolicy::None => Duration::from_secs(365 * 24 * 60 * 60),
                ToolRetryPolicy::ReadOnly { total_budget, .. } => {
                    total_budget.saturating_sub(elapsed)
                }
            };
            let attempt_started_at = Instant::now();
            let mut timeout_event_emitted = false;
            let execution = if let Some(timeout) = policy.attempt_timeout(remaining) {
                match tokio::time::timeout(
                    timeout,
                    tool.execute_with_timing_outcome(&tool_call.args),
                )
                .await
                {
                    Ok(result) => result,
                    Err(_) => {
                        let elapsed_ms = attempt_started_at.elapsed().as_millis();
                        timeout_event_emitted = true;
                        self.emit_trace(AgentTraceEvent::ToolTimedOut {
                            call_id: call_id.to_string(),
                            tool_name: tool_call.name.clone(),
                            planning_round,
                            attempt,
                            elapsed_ms,
                        });
                        Err(anyhow::Error::new(ToolExecutionError::Timeout))
                    }
                }
            } else {
                tool.execute_with_timing_outcome(&tool_call.args).await
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
                            planning_round: Some(planning_round),
                            tool_name: Some(tool_call.name.clone()),
                            call_id: Some(call_id.to_string()),
                            attempt,
                            outcome,
                            elapsed_ms: attempt_elapsed_ms,
                        });
                    }
                    self.emit_trace(AgentTraceEvent::Timing {
                        phase: ConversationTimingPhase::ToolExecution,
                        planning_round: Some(planning_round),
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
                            planning_round: Some(planning_round),
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
                            planning_round,
                            attempt,
                            elapsed_ms: attempt_elapsed_ms,
                        });
                    }
                    let remaining = match &policy {
                        ToolRetryPolicy::None => Duration::ZERO,
                        ToolRetryPolicy::ReadOnly { total_budget, .. } => {
                            total_budget.saturating_sub(call_started_at.elapsed())
                        }
                    };
                    if retryable && policy.can_retry(attempt, remaining) {
                        self.emit_trace(AgentTraceEvent::ToolRetryScheduled {
                            call_id: call_id.to_string(),
                            tool_name: tool_call.name.clone(),
                            planning_round,
                            attempt,
                            reason: reason.to_string(),
                        });
                        let delay_started_at = Instant::now();
                        tokio::time::sleep(policy.backoff()).await;
                        self.emit_trace(AgentTraceEvent::Timing {
                            phase: ConversationTimingPhase::RetryDelay,
                            planning_round: Some(planning_round),
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
                        planning_round: Some(planning_round),
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
                    break (
                        ToolResult::error(error.to_string()),
                        terminal_status.to_string(),
                        elapsed_ms,
                    );
                }
            }
        };

        self.emit_trace(AgentTraceEvent::ToolTerminal {
            call_id: call_id.to_string(),
            tool_name: tool_call.name.clone(),
            planning_round,
            attempt,
            status: terminal_status,
            elapsed_ms,
        });
        result
    }

    fn planning_input_content(&self, user_message: &str, is_first_plan: bool) -> String {
        if is_first_plan || self.current_tool_results.is_empty() {
            return user_message.to_string();
        }

        let results = self.bounded_current_tool_results_context();
        format!(
            "ORIGINAL REQUEST\n{}\n\nCOMPLETED TOOL RESULTS\n{}\n\nDecide only whether another Tool round is required.",
            user_message, results
        )
    }

    fn expected_curated_resources(&self, enabled_tools: &[String]) -> bool {
        self.curated_resources_requested()
            && enabled_tools.iter().any(|name| name == "find_resources")
    }

    fn curated_resources_requested(&self) -> bool {
        self.curated_resource_lookup_expectation.required && !self.curated_resource_lookup_succeeded
    }

    fn curated_resource_plan_violation(
        &self,
        decision: &ToolDecision,
        expected_curated_resources: bool,
    ) -> Option<&'static str> {
        if !expected_curated_resources {
            if self.curated_resource_lookup_expectation.required
                && self.curated_resource_lookup_succeeded
                && decision
                    .tool_calls
                    .iter()
                    .any(|tool_call| tool_call.name == "find_resources")
            {
                return Some("additional find_resources call was not allowed after turn success");
            }
            return None;
        }
        if self
            .curated_resource_lookup_expectation
            .continuation_cursor_missing
        {
            return Some("find_resources continuation cursor was unavailable");
        }
        if self
            .curated_resource_lookup_expectation
            .exact_filter_missing
        {
            return Some("exact find_resources inventory filter was unavailable");
        }
        let resource_calls = decision
            .tool_calls
            .iter()
            .filter(|tool_call| tool_call.name == "find_resources")
            .collect::<Vec<_>>();
        if resource_calls.is_empty() {
            return Some("required find_resources Tool call was omitted");
        }
        if resource_calls.len() != 1 {
            return Some("multiple find_resources calls were not allowed in one plan");
        }
        let call = resource_calls[0];
        if self.curated_resource_lookup_expectation.query_required
            && tool_string_arg(&call.args, "query").is_none_or(|query| query.trim().is_empty())
        {
            return Some("required find_resources query was omitted");
        }
        if let Some(expected_query) = &self.curated_resource_lookup_expectation.expected_query {
            let expected_query = normalized_lookup_text(expected_query);
            let actual_query = tool_string_arg(&call.args, "query").map(normalized_lookup_text);
            let actual_region = tool_string_arg(&call.args, "region").map(normalized_lookup_text);
            let primary_query_matches = actual_query.as_deref() == Some(expected_query.as_str());
            let primary_region_matches = self
                .curated_resource_lookup_expectation
                .expected_region
                .as_deref()
                .is_none_or(|expected_region| {
                    actual_region.as_deref().is_some_and(|actual_region| {
                        resource_regions_match(expected_region, actual_region)
                    })
                });
            if !primary_query_matches {
                return Some("find_resources query did not preserve the requested filter");
            }
            if primary_query_matches && !primary_region_matches {
                return Some("find_resources region did not preserve the requested location");
            }
        }
        if self
            .curated_resource_lookup_expectation
            .expected_query
            .is_none()
        {
            if let Some(expected_region) = &self.curated_resource_lookup_expectation.expected_region
            {
                let actual_region = tool_string_arg(&call.args, "region");
                if actual_region.is_none_or(|actual_region| {
                    !resource_regions_match(expected_region, actual_region)
                }) {
                    return Some("find_resources region did not preserve the requested location");
                }
            }
        }
        for (field, expected, must_be_absent, mismatch) in [
            (
                "help_type",
                self.curated_resource_lookup_expectation
                    .expected_help_type
                    .as_deref(),
                self.curated_resource_lookup_expectation
                    .help_type_must_be_absent,
                "find_resources help_type did not preserve the prior filter",
            ),
            (
                "language",
                self.curated_resource_lookup_expectation
                    .expected_language
                    .as_deref(),
                self.curated_resource_lookup_expectation
                    .language_must_be_absent,
                "find_resources language did not preserve the prior filter",
            ),
        ] {
            let actual =
                tool_string_arg(&call.args, field).filter(|value| !value.trim().is_empty());
            if expected.is_some_and(|expected| {
                actual.is_none_or(|actual| {
                    normalized_lookup_text(actual) != normalized_lookup_text(expected)
                })
            }) {
                return Some(mismatch);
            }
            if must_be_absent && actual.is_some() {
                return Some(mismatch);
            }
        }
        if self
            .curated_resource_lookup_expectation
            .contact_query_from_context
        {
            let actual_query = tool_string_arg(&call.args, "query").map(normalized_lookup_text);
            let actual_region = tool_string_arg(&call.args, "region");
            let grounded_resource = actual_query.as_deref().and_then(|actual_query| {
                self.curated_resource_lookup_expectation
                    .context_grounded_resources
                    .iter()
                    .rev()
                    .find(|(query, _)| query == actual_query)
            });
            let Some((_, expected_region)) = grounded_resource else {
                return Some("find_resources query was not grounded in recent conversation");
            };
            if expected_region.as_deref().is_some_and(|expected_region| {
                actual_region.is_none_or(|actual_region| {
                    !resource_regions_match(expected_region, actual_region)
                })
            }) {
                return Some("find_resources region did not preserve the requested location");
            }
        }
        if self
            .curated_resource_lookup_expectation
            .query_must_be_absent
            && tool_string_arg(&call.args, "query").is_some_and(|query| !query.trim().is_empty())
        {
            return Some("find_resources added a query that was absent from the prior page");
        }
        if self
            .curated_resource_lookup_expectation
            .region_must_be_absent
            && tool_string_arg(&call.args, "region").is_some_and(|region| !region.trim().is_empty())
        {
            return Some("find_resources added a region that was absent from the prior page");
        }
        if self
            .curated_resource_lookup_expectation
            .expected_lookup_mode
            .is_some_and(|expected_mode| {
                tool_string_arg(&call.args, "lookup_mode")
                    .is_none_or(|actual_mode| actual_mode.trim() != expected_mode)
            })
        {
            return Some("find_resources lookup mode did not match the current request");
        }
        if self
            .curated_resource_lookup_expectation
            .lookup_mode_must_be_absent
            && tool_string_arg(&call.args, "lookup_mode")
                .is_some_and(|mode| !mode.trim().is_empty())
        {
            return Some("find_resources lookup mode did not match the current request");
        }
        if self
            .curated_resource_lookup_expectation
            .positive_offset_required
            && tool_parse_arg::<usize>(&call.args, "offset").is_none_or(|offset| offset == 0)
        {
            return Some("required positive find_resources continuation offset was omitted");
        }
        if let Some(expected_offset) = self.curated_resource_lookup_expectation.expected_offset {
            if tool_parse_arg::<usize>(&call.args, "offset") != Some(expected_offset) {
                return Some("find_resources offset did not match the prior next_offset");
            }
        } else if self
            .curated_resource_lookup_expectation
            .initial_offset_required
            && tool_parse_arg::<usize>(&call.args, "offset").is_some_and(|offset| offset > 0)
        {
            return Some("fresh find_resources lookup used a stale positive offset");
        }
        None
    }

    /// Remove redundant resource calls only when doing so cannot broaden or
    /// invent a plan. Invalid calls remain untouched so normal validation can
    /// reject the model output and request a corrected plan.
    fn sanitize_curated_resource_calls(
        &self,
        decision: &mut ToolDecision,
        expected_curated_resources: bool,
    ) {
        if self.curated_resource_lookup_succeeded {
            decision
                .tool_calls
                .retain(|tool_call| tool_call.name != "find_resources");
            return;
        }

        let resource_call_indexes = decision
            .tool_calls
            .iter()
            .enumerate()
            .filter_map(|(index, tool_call)| (tool_call.name == "find_resources").then_some(index))
            .collect::<Vec<_>>();
        if resource_call_indexes.len() <= 1 {
            return;
        }

        let valid_index = resource_call_indexes.into_iter().find(|candidate_index| {
            let mut candidate = decision.clone();
            candidate.tool_calls = candidate
                .tool_calls
                .into_iter()
                .enumerate()
                .filter_map(|(index, tool_call)| {
                    (tool_call.name != "find_resources" || index == *candidate_index)
                        .then_some(tool_call)
                })
                .collect();
            self.curated_resource_plan_violation(&candidate, expected_curated_resources)
                .is_none()
        });
        if let Some(valid_index) = valid_index {
            decision.tool_calls = decision
                .tool_calls
                .drain(..)
                .enumerate()
                .filter_map(|(index, tool_call)| {
                    (tool_call.name != "find_resources" || index == valid_index)
                        .then_some(tool_call)
                })
                .collect();
        }
    }

    fn emit_unavailable_tool_planning_latency(&self, step: usize, attempt: u32) {
        for phase in [
            ConversationTimingPhase::ToolPlanningClusterScheduling,
            ConversationTimingPhase::ToolPlanningInference,
        ] {
            self.emit_trace(AgentTraceEvent::TimingUnavailable {
                phase,
                planning_round: Some(step),
                attempt,
                reason: "provider_contract_does_not_expose_phase_timing",
            });
        }
    }

    async fn plan_tools_with_dspy(
        &mut self,
        user_message: &str,
        is_first_plan: bool,
    ) -> Result<ToolPlanningOutcome> {
        if is_first_plan {
            self.clear_tool_results();
        }
        let step_index = self.turn_step_index;
        self.turn_step_index += 1;
        let input_content = self.planning_input_content(user_message, is_first_plan);
        let context = self.build_context();
        let available_tools = self.tools.generate_description();
        let predictor = Predict::<ToolDecisionResponse>::builder()
            .instruction(format!("{}{}", self.instruction, TOOL_PLANNING_INSTRUCTION))
            .build();
        let mut input = ToolDecisionResponseInput {
            input: input_content.clone(),
            current_time: context.current_time,
            persona_block: context.persona_block,
            human_block: context.human_block,
            memory_metadata: context.memory_metadata,
            previous_context_summary: context.previous_context_summary,
            recent_conversation: context.recent_conversation,
            available_tools,
            is_first_time_user: context.is_first_time_user,
        };

        const MAX_TOOL_PLAN_ATTEMPTS: u32 = 3;
        let mut last_error = None;
        let mut terminal_selection_rejection_emitted = false;
        for attempt in 1..=MAX_TOOL_PLAN_ATTEMPTS {
            self.emit_trace(AgentTraceEvent::ModelStepStarted {
                step: step_index,
                attempt,
            });
            let started_at = Instant::now();
            match predictor.call(input.clone()).await {
                Ok(response) => {
                    let elapsed_ms = started_at.elapsed().as_millis();
                    self.emit_unavailable_tool_planning_latency(step_index, attempt);
                    self.emit_trace(AgentTraceEvent::ModelStepCompleted {
                        step: step_index,
                        attempt,
                        elapsed_ms,
                    });
                    self.emit_trace(AgentTraceEvent::Timing {
                        phase: ConversationTimingPhase::ToolPlanningModelDuration,
                        planning_round: Some(step_index),
                        tool_name: None,
                        call_id: None,
                        attempt,
                        outcome: ConversationTimingOutcome::Succeeded,
                        elapsed_ms,
                    });
                    let decision = ToolDecision::new(
                        response.tool_calls,
                        response.replan_after_results.unwrap_or(false),
                    );
                    let mut decision = decision;
                    decision.planning_round = step_index;
                    let enabled_tools = self.tools.names();
                    let curated_resources_available =
                        enabled_tools.iter().any(|name| name == "find_resources");
                    let raw_selected_tools = decision
                        .tool_calls
                        .iter()
                        .map(|tool_call| tool_call.name.clone())
                        .collect::<Vec<_>>();
                    let expected_curated_resources =
                        self.expected_curated_resources(&enabled_tools);
                    self.sanitize_curated_resource_calls(&mut decision, expected_curated_resources);
                    let selected_tools = decision
                        .tool_calls
                        .iter()
                        .map(|tool_call| tool_call.name.clone())
                        .collect::<Vec<_>>();
                    let violation =
                        self.curated_resource_plan_violation(&decision, expected_curated_resources);
                    let missed_expected_curated_resources = self.curated_resources_requested()
                        && (!expected_curated_resources || violation.is_some());
                    self.emit_trace(AgentTraceEvent::ToolSelectionObservation {
                        round: step_index,
                        attempt,
                        enabled_tools,
                        raw_selected_tools,
                        selected_tools,
                        expected_curated_resources: self.curated_resources_requested(),
                        curated_resources_available,
                        missed_expected_curated_resources,
                        violation_reason: violation.map(str::to_string),
                        outcome: if violation.is_some() {
                            "rejected".to_string()
                        } else {
                            "planned".to_string()
                        },
                    });
                    if let Some(violation) = violation {
                        last_error = Some(anyhow::anyhow!(violation));
                        terminal_selection_rejection_emitted = attempt == MAX_TOOL_PLAN_ATTEMPTS;
                        if attempt < MAX_TOOL_PLAN_ATTEMPTS {
                            self.emit_trace(AgentTraceEvent::RetryScheduled {
                                step: step_index,
                                attempt,
                            });
                            input.input = format!(
                                "{}\n\nRUNTIME TOOL-PLAN VALIDATION\n{}",
                                input_content,
                                self.curated_resource_lookup_expectation
                                    .retry_instruction_for_violation(violation),
                            );
                            let delay_started_at = Instant::now();
                            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                            self.emit_trace(AgentTraceEvent::Timing {
                                phase: ConversationTimingPhase::RetryDelay,
                                planning_round: Some(step_index),
                                tool_name: None,
                                call_id: None,
                                attempt,
                                outcome: ConversationTimingOutcome::Succeeded,
                                elapsed_ms: delay_started_at.elapsed().as_millis(),
                            });
                        }
                        continue;
                    }
                    return Ok(ToolPlanningOutcome::Decision(decision));
                }
                Err(error) => {
                    let elapsed_ms = started_at.elapsed().as_millis();
                    self.emit_unavailable_tool_planning_latency(step_index, attempt);
                    self.emit_trace(AgentTraceEvent::ModelStepFailed {
                        step: step_index,
                        attempt,
                        elapsed_ms,
                        error: format!("{:?}", error),
                    });
                    self.emit_trace(AgentTraceEvent::Timing {
                        phase: ConversationTimingPhase::ToolPlanningModelDuration,
                        planning_round: Some(step_index),
                        tool_name: None,
                        call_id: None,
                        attempt,
                        outcome: ConversationTimingOutcome::Failed,
                        elapsed_ms,
                    });
                    // This planner is only entered when actionable tools are available.
                    // A bare-prose parse failure is therefore not a trustworthy terminal
                    // answer: accepting it would let the model bypass the required tool
                    // decision and fabricate an unverified admin response. Keep retrying
                    // the typed contract instead.
                    last_error = Some(error.into());
                    if attempt < MAX_TOOL_PLAN_ATTEMPTS {
                        self.emit_trace(AgentTraceEvent::RetryScheduled {
                            step: step_index,
                            attempt,
                        });
                        let delay_started_at = Instant::now();
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        self.emit_trace(AgentTraceEvent::Timing {
                            phase: ConversationTimingPhase::RetryDelay,
                            planning_round: Some(step_index),
                            tool_name: None,
                            call_id: None,
                            attempt,
                            outcome: ConversationTimingOutcome::Succeeded,
                            elapsed_ms: delay_started_at.elapsed().as_millis(),
                        });
                    }
                }
            }
        }

        if !terminal_selection_rejection_emitted {
            let enabled_tools = self.tools.names();
            let curated_resources_available =
                enabled_tools.iter().any(|name| name == "find_resources");
            self.emit_trace(AgentTraceEvent::ToolSelectionObservation {
                round: step_index,
                attempt: MAX_TOOL_PLAN_ATTEMPTS,
                enabled_tools,
                raw_selected_tools: Vec::new(),
                selected_tools: Vec::new(),
                expected_curated_resources: self.curated_resources_requested(),
                curated_resources_available,
                missed_expected_curated_resources: self.curated_resources_requested(),
                violation_reason: last_error.as_ref().map(ToString::to_string),
                outcome: "failed".to_string(),
            });
        }
        Err(anyhow::anyhow!(
            "Tool planning failed after {} attempts: {:?}",
            MAX_TOOL_PLAN_ATTEMPTS,
            last_error.expect("planner records every failed attempt")
        ))
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
        tracing::info!("Input: {}", input_content);
        tracing::info!("Recent conversation:\n{}", ctx.recent_conversation);

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
                // Use Debug ({:?}) so the underlying provider status (e.g. the
                // upstream "503 Service Unavailable") is preserved in the error
                // message. The provider error's Display is only "LLM call failed",
                // which hides the status that model-fallback classification needs.
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
            let result = self.execute_tool_call(&call_id, 0, tool_call).await;
            tracing::debug!("Tool {} result: {:?}", tool_call.name, result);

            // Inject into current request cycle (for multi-step reasoning)
            self.inject_tool_result(tool_call, &result);

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

#[async_trait::async_trait]
impl ToolPlanner for SageAgent {
    fn has_actionable_tools(&self) -> bool {
        self.tools.has_actionable_tools()
    }

    fn set_curated_resource_lookup_expectation(
        &mut self,
        expectation: CuratedResourceLookupExpectation,
    ) {
        self.curated_resource_lookup_expectation = expectation;
    }

    async fn plan_tools(
        &mut self,
        user_message: &str,
        is_first_plan: bool,
    ) -> Result<ToolPlanningOutcome> {
        self.plan_tools_with_dspy(user_message, is_first_plan).await
    }

    async fn execute_tool_decision(&mut self, decision: &ToolDecision) -> StepResult {
        SageAgent::execute_tool_decision(self, decision).await
    }

    fn plain_answer_prompt(&self, user_message: &str) -> PlainAnswerPrompt {
        SageAgent::plain_answer_prompt(self, user_message)
    }

    fn plain_answer_trace_started(&mut self) -> usize {
        self.final_answer_attempt.store(1, Ordering::Relaxed);
        let step = self.turn_step_index;
        self.turn_step_index += 1;
        self.emit_trace(AgentTraceEvent::ModelStepStarted { step, attempt: 1 });
        step
    }

    fn plain_answer_reasoning_trace_hook(&self, step: usize) -> Option<ProviderReasoningTraceHook> {
        let trace_hook = self.trace_hook.clone()?;
        Some(Arc::new(move |content| {
            trace_hook(AgentTraceEvent::ProviderReasoning { step, content });
        }))
    }

    fn plain_answer_provider_timing_hook(&self, step: usize) -> Option<ProviderTimingTraceHook> {
        let trace_hook = self.trace_hook.clone()?;
        let final_answer_attempt = self.final_answer_attempt.clone();
        Some(Arc::new(move |timing| {
            let (phase, attempt, elapsed_ms, outcome) = match timing {
                ProviderTimingEvent::ResponseHeaders {
                    attempt,
                    elapsed_ms,
                    outcome,
                } => {
                    final_answer_attempt.fetch_max(attempt, Ordering::Relaxed);
                    (
                        ConversationTimingPhase::FinalAnswerResponseHeaderWait,
                        attempt,
                        elapsed_ms,
                        outcome,
                    )
                }
                ProviderTimingEvent::FirstProviderEvent {
                    attempt,
                    elapsed_ms,
                    outcome,
                } => {
                    final_answer_attempt.fetch_max(attempt, Ordering::Relaxed);
                    (
                        ConversationTimingPhase::FinalAnswerFirstProviderEventWait,
                        attempt,
                        elapsed_ms,
                        outcome,
                    )
                }
            };
            trace_hook(AgentTraceEvent::Timing {
                phase,
                planning_round: Some(step),
                tool_name: None,
                call_id: None,
                attempt,
                outcome,
                elapsed_ms,
            });
        }))
    }

    fn plain_answer_trace_completed(&self, step: usize, elapsed_ms: u128) {
        let attempt = self.final_answer_attempt.load(Ordering::Relaxed);
        self.emit_trace(AgentTraceEvent::ModelStepCompleted {
            step,
            attempt,
            elapsed_ms,
        });
        self.emit_trace(AgentTraceEvent::Timing {
            phase: ConversationTimingPhase::FinalAnswerModelDuration,
            planning_round: Some(step),
            tool_name: None,
            call_id: None,
            attempt,
            outcome: ConversationTimingOutcome::Succeeded,
            elapsed_ms,
        });
    }

    fn plain_answer_trace_failed(&self, step: usize, elapsed_ms: u128, error: &str) {
        let attempt = self.final_answer_attempt.load(Ordering::Relaxed);
        self.emit_trace(AgentTraceEvent::ModelStepFailed {
            step,
            attempt,
            elapsed_ms,
            error: error.to_string(),
        });
        self.emit_trace(AgentTraceEvent::Timing {
            phase: ConversationTimingPhase::FinalAnswerModelDuration,
            planning_round: Some(step),
            tool_name: None,
            call_id: None,
            attempt,
            outcome: ConversationTimingOutcome::Failed,
            elapsed_ms,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn curated_resource_expectation_is_conservative_and_bilingual() {
        assert!(expects_curated_resource_lookup("Can you share the email?"));
        assert!(expects_curated_resource_lookup(
            "¿Me das el teléfono y la dirección?"
        ));
        assert!(expects_curated_resource_lookup(
            "Necesito el sitio web o canal seguro."
        ));
        assert!(!expects_curated_resource_lookup(
            "Tell me about this organization."
        ));
        assert!(!expects_curated_resource_lookup("What help is available?"));
        assert!(!expects_curated_resource_lookup(
            "Please address this concern."
        ));
        assert!(expects_curated_resource_lookup(
            "Please address this concern and give me the email."
        ));
        assert!(expects_curated_resource_lookup(
            "Please address my concern and share their physical address."
        ));
        assert!(!expects_curated_resource_lookup("Show me a curl example."));
        assert!(!expects_curated_resource_lookup("Summarize this email."));
        assert!(!expects_curated_resource_lookup(
            "Draft an email to my boss."
        ));
        assert!(!expects_curated_resource_lookup(
            "Prepare me for a phone interview."
        ));
        assert!(!expects_curated_resource_lookup("What is an email?"));
        assert!(!expects_curated_resource_lookup(
            "Give me an email example."
        ));
        for prompt in [
            "What is email authentication?",
            "Can you give me feedback on this email?",
            "Can you give me feedback on their email?",
            "What is wrong with their email?",
            "Give me advice about email security.",
            "What is phone banking?",
            "Draft an email for Acme Legal Aid.",
            "Review the website for Acme Legal Aid.",
            "Show me how organizations work.",
            "Show me how the Resource Directory works.",
            "Can I see how available organizations are selected?",
        ] {
            assert!(!expects_curated_resource_lookup(prompt), "{prompt}");
        }
        for (prompt, expected_query) in [
            (
                "How can I contact Acme Legal Aid by email?",
                "acme legal aid",
            ),
            (
                "Please share Acme Legal Aid's phone number.",
                "acme legal aid",
            ),
            ("What is the email for Women in Need?", "women in need"),
            (
                "Give me the contact information for Acme Legal Aid.",
                "acme legal aid",
            ),
            ("Can you share Acme Legal Aid's email?", "acme legal aid"),
            (
                "Could you give me Acme Legal Aid's phone number?",
                "acme legal aid",
            ),
            ("What's Acme Legal Aid's email?", "acme legal aid"),
            (
                "¿Me puedes dar el email de Acme Legal Aid?",
                "acme legal aid",
            ),
            (
                "¿Me puedes dar el correo electrónico para Acme Legal Aid?",
                "acme legal aid",
            ),
            (
                "What is The Women's Law Center's email?",
                "the women s law center",
            ),
            (
                "What is the email for The Women's Law Center?",
                "the women s law center",
            ),
            ("Can I get Acme Legal Aid's email?", "acme legal aid"),
            ("Can I get the email for Acme Legal Aid?", "acme legal aid"),
            ("What is Acme Legal Aid's e-mail?", "acme legal aid"),
            ("¿Cuál es el celular de Acme Legal Aid?", "acme legal aid"),
            ("What is the e-mail of Acme Legal Aid?", "acme legal aid"),
            (
                "Give me the phone number of Acme Legal Aid.",
                "acme legal aid",
            ),
            (
                "Could I get the email for Acme Legal Aid?",
                "acme legal aid",
            ),
            ("Could I have Acme Legal Aid's email?", "acme legal aid"),
            ("I need Acme Legal Aid's email.", "acme legal aid"),
            ("I need the email for Acme Legal Aid.", "acme legal aid"),
            ("Dame el correo de Acme Legal Aid.", "acme legal aid"),
            (
                "Share Acme Legal Aid's email and phone number.",
                "acme legal aid",
            ),
        ] {
            let expectation = curated_resource_lookup_expectation(prompt, None);
            assert!(expectation.required, "{prompt}");
            assert_eq!(
                expectation.expected_query.as_deref(),
                Some(expected_query),
                "{prompt}"
            );
        }
        assert!(expects_curated_resource_lookup(
            "What is Acme Legal Aid's email?"
        ));
        assert_eq!(
            curated_resource_lookup_expectation(
                "What is the email address for Acme Legal Aid in Mexico?",
                None,
            )
            .expected_query
            .as_deref(),
            Some("acme legal aid")
        );
        assert_eq!(
            curated_resource_lookup_expectation(
                "What is the email address for Acme Legal Aid in Mexico?",
                None,
            )
            .expected_region
            .as_deref(),
            Some("mexico")
        );
        let uppercase_alpha2 = curated_resource_lookup_expectation(
            "What is the email address for Acme Legal Aid in US?",
            None,
        );
        assert_eq!(
            uppercase_alpha2.expected_query.as_deref(),
            Some("acme legal aid")
        );
        assert_eq!(uppercase_alpha2.expected_region.as_deref(), Some("US"));
        let lowercase_pronoun =
            curated_resource_lookup_expectation("What is the email address for Women in us?", None);
        assert_eq!(
            lowercase_pronoun.expected_query.as_deref(),
            Some("women in us")
        );
        assert_eq!(lowercase_pronoun.expected_region, None);
        assert_eq!(
            curated_resource_lookup_expectation(
                "¿Cuál es el correo electrónico de Acme Legal Aid en México?",
                None,
            )
            .expected_query
            .as_deref(),
            Some("acme legal aid")
        );
        for (prompt, expected_query) in [
            ("WLC contact information", "wlc"),
            ("WLC phone number", "wlc"),
            ("I need contact information for WLC", "wlc"),
            ("Can I get the email and phone number for WLC?", "wlc"),
            (
                "Do you have the email for Acme Legal Aid?",
                "acme legal aid",
            ),
            ("Dame el correo y teléfono de WLC", "wlc"),
            ("¿Tienes el correo de Acme Legal Aid?", "acme legal aid"),
        ] {
            let expectation = curated_resource_lookup_expectation(prompt, None);
            assert!(expectation.required, "{prompt}");
            assert_eq!(
                expectation.expected_query.as_deref(),
                Some(expected_query),
                "{prompt}"
            );
        }
        for prompt in [
            "Share their physical address.",
            "What is the address?",
            "Where is the address?",
            "Can you give me the phone number?",
            "Could I get the email?",
            "Could I have the phone number?",
            "What are their contact details?",
            "Me puedes dar el email?",
            "¿Cuál es el correo?",
            "¿Dónde está la dirección?",
            "¿Cuáles son sus datos de contacto?",
        ] {
            let expectation = curated_resource_lookup_expectation(prompt, None);
            assert!(expectation.required, "{prompt}");
            assert_eq!(expectation.expected_query, None, "{prompt}");
        }
        assert!(expects_curated_resource_lookup(
            "List the ready Curated Resources whose names start with 'Issue 539 Inventory'."
        ));
        assert!(expects_curated_resource_lookup(
            "Show the next page of those matching resources."
        ));
        assert!(expects_curated_resource_lookup(
            "Lista los recursos curados listos cuyos nombres empiezan con 'Issue 539 Inventory'."
        ));
        assert!(expects_curated_resource_lookup(
            "Muestra la siguiente página de esos recursos coincidentes."
        ));
        assert!(expects_curated_resource_lookup(
            "Give me all Curated Resources."
        ));
        assert!(expects_curated_resource_lookup(
            "Can I see the available organizations?"
        ));
        assert!(expects_curated_resource_lookup(
            "How many Curated Resources are ready?"
        ));
        assert!(expects_curated_resource_lookup(
            "Enumerate the resource directory."
        ));
        assert!(expects_curated_resource_lookup(
            "Are there any curated resources?"
        ));
        assert!(expects_curated_resource_lookup("¿Hay recursos curados?"));
        for (prompt, expected_region) in [
            ("List resources for Mexico.", "MX"),
            ("What resources are available in Mexico?", "MX"),
            ("Which resources are currently available in Mexico?", "MX"),
            ("List resources available in Mexico.", "MX"),
            ("Lista recursos para México.", "MX"),
            ("¿Qué recursos están disponibles en México?", "MX"),
            (
                "¿Qué recursos están actualmente disponibles en México?",
                "MX",
            ),
            ("Lista recursos disponibles en México.", "MX"),
            ("List resources for United States.", "US"),
            ("List organizations in Latin America.", "latin america"),
            ("List Mexico resources.", "MX"),
            ("Lista recursos de México.", "MX"),
        ] {
            let expectation = curated_resource_lookup_expectation(prompt, None);
            assert!(expectation.required, "{prompt}");
            assert_eq!(expectation.expected_query, None, "{prompt}");
            assert_eq!(
                expectation.expected_region.as_deref(),
                Some(expected_region),
                "{prompt}"
            );
        }
        assert_eq!(
            curated_resource_lookup_expectation(
                "List ready Curated Resources matching Acme Legal Aid.",
                None,
            )
            .expected_query
            .as_deref(),
            Some("Acme Legal Aid")
        );
        for (prompt, expected_query) in [
            ("List resources for Acme Legal Aid.", "Acme Legal Aid"),
            ("List organizations for Acme Legal Aid.", "Acme Legal Aid"),
            ("List Acme Legal Aid resources.", "Acme Legal Aid"),
            ("Lista recursos para Acme Legal Aid.", "Acme Legal Aid"),
        ] {
            let expectation = curated_resource_lookup_expectation(prompt, None);
            assert!(expectation.required, "{prompt}");
            assert_eq!(
                expectation.expected_query.as_deref(),
                Some(expected_query),
                "{prompt}"
            );
        }
        for prompt in [
            "List resources for legal help.",
            "List resources for me.",
            "List all resources.",
            "List available organizations.",
        ] {
            let expectation = curated_resource_lookup_expectation(prompt, None);
            assert!(expectation.required, "{prompt}");
            assert_eq!(expectation.expected_query, None, "{prompt}");
        }

        let continuation = CuratedResourceContinuation {
            query: Some("Issue 539 Inventory".to_string()),
            region: None,
            help_type: None,
            language: None,
            lookup_mode: Some("inventory".to_string()),
            next_offset: 10,
        };
        for prompt in [
            "Next page please.",
            "Show me more.",
            "More resources.",
            "Show more resources.",
            "Continue listing resources.",
            "Siguiente página.",
            "Muestra más recursos.",
            "Continúa listando recursos.",
        ] {
            let expectation = curated_resource_lookup_expectation(prompt, Some(&continuation));
            assert!(expectation.required, "{prompt}");
            assert!(expectation.positive_offset_required, "{prompt}");
            assert_eq!(expectation.expected_offset, Some(10), "{prompt}");
        }
        let explicit_without_cursor = curated_resource_lookup_expectation(
            "Show the next page of those matching resources.",
            None,
        );
        assert!(explicit_without_cursor.required);
        assert!(explicit_without_cursor.positive_offset_required);
        assert_eq!(explicit_without_cursor.expected_offset, None);
        assert_eq!(
            curated_resource_lookup_expectation(
                "List resources matching Acme Legal Aid and summarize them briefly.",
                None,
            )
            .expected_query
            .as_deref(),
            Some("Acme Legal Aid")
        );
        assert_eq!(
            curated_resource_lookup_expectation(
                "List resources matching Acme Legal Aid; answer \"briefly\".",
                None,
            )
            .expected_query
            .as_deref(),
            Some("Acme Legal Aid")
        );
        assert_eq!(
            curated_resource_lookup_expectation(
                "List resources named \"Acme Matching Center\".",
                None,
            )
            .expected_query
            .as_deref(),
            Some("Acme Matching Center")
        );
    }

    #[test]
    fn trusted_contact_context_requires_an_exact_entity_and_preserves_country() {
        let context = CuratedResourceQueryContext::from_trusted_text(
            "The prior referrals were Atlas Aid in Mexico, Women in Need, and WLC.",
        );
        assert_eq!(
            context.resources,
            vec![
                ("atlas aid".to_string(), Some("MX".to_string())),
                ("women in need".to_string(), None),
                ("wlc".to_string(), None),
            ]
        );

        assert_eq!(
            split_resource_geographic_qualifier("Atlas Aid in FRA"),
            Some(("Atlas Aid".to_string(), "FRA".to_string()))
        );
        assert_eq!(
            split_resource_geographic_qualifier("Atlas Aid in Russia"),
            Some(("Atlas Aid".to_string(), "Russia".to_string()))
        );
        assert_eq!(split_resource_geographic_qualifier("Women in us"), None);
        assert_eq!(split_resource_geographic_qualifier("Women in Need"), None);

        let false_names = CuratedResourceQueryContext::from_trusted_text(
            "The people mentioned were John Smith and New York.",
        );
        assert!(false_names.resources.is_empty());
        let mut structured = CuratedResourceQueryContext::default();
        structured.push_structured_resource("Amnesty", Some("Mexico"));
        assert_eq!(
            structured.resources,
            vec![("amnesty".to_string(), Some("MX".to_string()))]
        );

        let prose = CuratedResourceQueryContext::from_trusted_text(
            "We discussed Horizon Foundation and an unrelated support project.",
        );
        let mut returned_resources = CuratedResourceQueryContext::default();
        returned_resources.push_structured_resource("Acme", Some("Mexico"));
        assert_eq!(
            prefer_structured_resource_context(prose, returned_resources).resources,
            vec![("acme".to_string(), Some("MX".to_string()))],
            "structured returned resources must replace ambiguous prose candidates"
        );
    }

    #[test]
    fn context_followup_plan_rejects_partial_entities_and_missing_regions() {
        let mut registry = ToolRegistry::new();
        registry.register_descriptor("find_resources", "lookup", r#"{"query":"text"}"#);
        let mut agent = SageAgent::new_without_memory(registry, "test");
        let context = CuratedResourceQueryContext::from_trusted_text(
            "Earlier we discussed Acme Legal Aid Network in Mexico. Then Atlas Aid in France.",
        );
        agent.curated_resource_lookup_expectation =
            curated_resource_lookup_expectation("What is their email?", None)
                .with_query_context(&context);
        let expected = agent.expected_curated_resources(&agent.tools.names());

        for query in ["Acme Legal Aid", "Atlas", "Invented Organization"] {
            let decision = ToolDecision::new(
                vec![ToolCall {
                    name: "find_resources".to_string(),
                    args: [
                        ("query".to_string(), serde_json::json!(query)),
                        ("lookup_mode".to_string(), serde_json::json!("contact")),
                    ]
                    .into(),
                }],
                false,
            );
            assert_eq!(
                agent.curated_resource_plan_violation(&decision, expected),
                Some("find_resources query was not grounded in recent conversation"),
                "{query}"
            );
        }

        let exact_without_region = ToolDecision::new(
            vec![ToolCall {
                name: "find_resources".to_string(),
                args: [
                    ("query".to_string(), serde_json::json!("Atlas Aid")),
                    ("lookup_mode".to_string(), serde_json::json!("contact")),
                ]
                .into(),
            }],
            false,
        );
        assert_eq!(
            agent.curated_resource_plan_violation(&exact_without_region, expected),
            Some("find_resources region did not preserve the requested location")
        );
        let exact = ToolDecision::new(
            vec![ToolCall {
                name: "find_resources".to_string(),
                args: [
                    ("query".to_string(), serde_json::json!("Atlas Aid")),
                    ("region".to_string(), serde_json::json!("FR")),
                    ("lookup_mode".to_string(), serde_json::json!("contact")),
                ]
                .into(),
            }],
            false,
        );
        assert!(agent
            .curated_resource_plan_violation(&exact, expected)
            .is_none());
    }

    #[test]
    fn curated_resource_plan_validation_preserves_inventory_filter_and_continuation() {
        let mut registry = ToolRegistry::new();
        registry.register_descriptor("find_resources", "lookup", r#"{"query":"text"}"#);
        let mut agent = SageAgent::new_without_memory(registry, "test");
        let enabled = agent.tools.names();

        agent.curated_resource_lookup_expectation = curated_resource_lookup_expectation(
            "List ready Curated Resources whose names start with 'Issue 539 Inventory'.",
            None,
        );
        let expected = agent.expected_curated_resources(&enabled);
        assert!(expected);
        assert_eq!(
            agent.curated_resource_plan_violation(&ToolDecision::new(Vec::new(), false), expected),
            Some("required find_resources Tool call was omitted")
        );
        let mut filtered_call = ToolCall {
            name: "find_resources".to_string(),
            args: [("lookup_mode".to_string(), serde_json::json!("inventory"))].into(),
        };
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![filtered_call.clone()], false),
                expected,
            ),
            Some("required find_resources query was omitted")
        );
        filtered_call
            .args
            .insert("query".to_string(), serde_json::json!("aid"));
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![filtered_call.clone()], false),
                expected,
            ),
            Some("find_resources query did not preserve the requested filter")
        );
        filtered_call.args.insert(
            "query".to_string(),
            serde_json::json!("Issue 539 Inventory unrelated"),
        );
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![filtered_call.clone()], false),
                expected,
            ),
            Some("find_resources query did not preserve the requested filter")
        );
        filtered_call.args.insert(
            "query".to_string(),
            serde_json::Value::String("Issue 539 Inventory".to_string()),
        );
        assert!(agent
            .curated_resource_plan_violation(
                &ToolDecision::new(vec![filtered_call], false),
                expected,
            )
            .is_none());

        let continuation = CuratedResourceContinuation {
            query: Some("Issue 539 Inventory".to_string()),
            region: None,
            help_type: None,
            language: None,
            lookup_mode: Some("inventory".to_string()),
            next_offset: 10,
        };
        agent.curated_resource_lookup_expectation = curated_resource_lookup_expectation(
            "Show the next page of those matching resources.",
            Some(&continuation),
        );
        let expected = agent.expected_curated_resources(&enabled);
        let mut continuation_call = ToolCall {
            name: "find_resources".to_string(),
            args: [
                (
                    "query".to_string(),
                    serde_json::json!("Issue 539 Inventory"),
                ),
                ("lookup_mode".to_string(), serde_json::json!("inventory")),
            ]
            .into(),
        };
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![continuation_call.clone()], false),
                expected,
            ),
            Some("required positive find_resources continuation offset was omitted")
        );
        continuation_call
            .args
            .insert("offset".to_string(), serde_json::json!(1));
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![continuation_call.clone()], false),
                expected,
            ),
            Some("find_resources offset did not match the prior next_offset")
        );
        continuation_call
            .args
            .insert("offset".to_string(), serde_json::json!(10));
        let mut wrong_continuation_mode = continuation_call.clone();
        wrong_continuation_mode
            .args
            .insert("lookup_mode".to_string(), serde_json::json!("contact"));
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![wrong_continuation_mode], false),
                expected,
            ),
            Some("find_resources lookup mode did not match the current request")
        );
        assert!(agent
            .curated_resource_plan_violation(
                &ToolDecision::new(vec![continuation_call], false),
                expected,
            )
            .is_none());

        let full_filter_continuation = CuratedResourceContinuation {
            query: Some("Atlas Aid".to_string()),
            region: Some("MX".to_string()),
            help_type: Some("legal".to_string()),
            language: Some("es".to_string()),
            lookup_mode: None,
            next_offset: 5,
        };
        agent.curated_resource_lookup_expectation = curated_resource_lookup_expectation(
            "Show the next page of those resources.",
            Some(&full_filter_continuation),
        );
        let expected = agent.expected_curated_resources(&enabled);
        let full_filter_call = ToolCall {
            name: "find_resources".to_string(),
            args: [
                ("query".to_string(), serde_json::json!("Atlas Aid")),
                ("region".to_string(), serde_json::json!("Mexico")),
                ("help_type".to_string(), serde_json::json!("legal")),
                ("language".to_string(), serde_json::json!("es")),
                ("offset".to_string(), serde_json::json!(5)),
            ]
            .into(),
        };
        assert!(agent
            .curated_resource_plan_violation(
                &ToolDecision::new(vec![full_filter_call.clone()], false),
                expected,
            )
            .is_none());
        let mut dropped_language = full_filter_call.clone();
        dropped_language.args.remove("language");
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![dropped_language], false),
                expected,
            ),
            Some("find_resources language did not preserve the prior filter")
        );

        agent.curated_resource_lookup_expectation =
            curated_resource_lookup_expectation("List all available resources.", None);
        let expected = agent.expected_curated_resources(&enabled);
        let narrowed_inventory = ToolCall {
            name: "find_resources".to_string(),
            args: [("help_type".to_string(), serde_json::json!("legal"))].into(),
        };
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![narrowed_inventory], false),
                expected,
            ),
            Some("find_resources help_type did not preserve the prior filter")
        );
        let wrong_inventory_mode = ToolCall {
            name: "find_resources".to_string(),
            args: [("lookup_mode".to_string(), serde_json::json!("contact"))].into(),
        };
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![wrong_inventory_mode], false),
                expected,
            ),
            Some("find_resources lookup mode did not match the current request")
        );
        let inventory_call = ToolCall {
            name: "find_resources".to_string(),
            args: [("lookup_mode".to_string(), serde_json::json!("inventory"))].into(),
        };
        assert!(agent
            .curated_resource_plan_violation(
                &ToolDecision::new(vec![inventory_call], false),
                expected,
            )
            .is_none());

        agent.curated_resource_lookup_expectation =
            curated_resource_lookup_expectation("What is Acme Legal Aid's email?", None);
        let expected = agent.expected_curated_resources(&enabled);
        let mut wrong_contact_call = ToolCall {
            name: "find_resources".to_string(),
            args: [
                ("query".to_string(), serde_json::json!("aid")),
                ("lookup_mode".to_string(), serde_json::json!("contact")),
            ]
            .into(),
        };
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![wrong_contact_call.clone()], false),
                expected,
            ),
            Some("find_resources query did not preserve the requested filter")
        );
        wrong_contact_call
            .args
            .insert("query".to_string(), serde_json::json!("Acme Legal Aid"));
        assert!(agent
            .curated_resource_plan_violation(
                &ToolDecision::new(vec![wrong_contact_call], false),
                expected,
            )
            .is_none());

        agent.curated_resource_lookup_expectation = curated_resource_lookup_expectation(
            "What is the email address for Acme Legal Aid in Mexico?",
            None,
        );
        let expected = agent.expected_curated_resources(&enabled);
        let mut regional_contact_call = ToolCall {
            name: "find_resources".to_string(),
            args: [
                ("query".to_string(), serde_json::json!("Acme Legal Aid")),
                ("lookup_mode".to_string(), serde_json::json!("contact")),
            ]
            .into(),
        };
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![regional_contact_call.clone()], false),
                expected,
            ),
            Some("find_resources region did not preserve the requested location")
        );
        regional_contact_call
            .args
            .insert("region".to_string(), serde_json::json!("Canada"));
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![regional_contact_call.clone()], false),
                expected,
            ),
            Some("find_resources region did not preserve the requested location")
        );
        regional_contact_call
            .args
            .insert("region".to_string(), serde_json::json!("MX"));
        assert!(agent
            .curated_resource_plan_violation(
                &ToolDecision::new(vec![regional_contact_call], false),
                expected,
            )
            .is_none());
        assert!(agent
            .curated_resource_lookup_expectation
            .retry_instruction_for_violation(
                "find_resources region did not preserve the requested location"
            )
            .contains("both the requested organization/name and location"));

        agent.curated_resource_lookup_expectation = curated_resource_lookup_expectation(
            "What is the email address for Acme Legal Aid in France?",
            None,
        );
        let expected = agent.expected_curated_resources(&enabled);
        let france_contact_call = ToolCall {
            name: "find_resources".to_string(),
            args: [
                ("query".to_string(), serde_json::json!("Acme Legal Aid")),
                ("region".to_string(), serde_json::json!("France")),
                ("lookup_mode".to_string(), serde_json::json!("contact")),
            ]
            .into(),
        };
        assert!(agent
            .curated_resource_plan_violation(
                &ToolDecision::new(vec![france_contact_call], false),
                expected,
            )
            .is_none());
        let france_in_query = ToolCall {
            name: "find_resources".to_string(),
            args: [
                (
                    "query".to_string(),
                    serde_json::json!("Acme Legal Aid in France"),
                ),
                ("region".to_string(), serde_json::json!("Canada")),
            ]
            .into(),
        };
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![france_in_query], false),
                expected,
            ),
            Some("find_resources query did not preserve the requested filter")
        );

        agent.curated_resource_lookup_expectation =
            curated_resource_lookup_expectation("What is the email for Women in Need?", None);
        let expected = agent.expected_curated_resources(&enabled);
        let organization_with_in = ToolCall {
            name: "find_resources".to_string(),
            args: [
                ("query".to_string(), serde_json::json!("Women in Need")),
                ("lookup_mode".to_string(), serde_json::json!("contact")),
            ]
            .into(),
        };
        assert!(agent
            .curated_resource_plan_violation(
                &ToolDecision::new(vec![organization_with_in], false),
                expected,
            )
            .is_none());
        let corrupted_organization_with_in = ToolCall {
            name: "find_resources".to_string(),
            args: [
                ("query".to_string(), serde_json::json!("Women")),
                ("region".to_string(), serde_json::json!("Need")),
            ]
            .into(),
        };
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![corrupted_organization_with_in], false),
                expected,
            ),
            Some("find_resources query did not preserve the requested filter")
        );
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(
                    vec![ToolCall {
                        name: "find_resources".to_string(),
                        args: ToolArgs::new(),
                    }],
                    false,
                ),
                expected,
            ),
            Some("required find_resources query was omitted")
        );

        let query_context = CuratedResourceQueryContext::from_trusted_text(
            "RECENT CONVERSATION\nuser: I need Acme Legal Aid in Mexico.",
        );
        agent.curated_resource_lookup_expectation =
            curated_resource_lookup_expectation("What is the address?", None)
                .with_query_context(&query_context);
        let expected = agent.expected_curated_resources(&enabled);
        let mut contextual_contact_call = ToolCall {
            name: "find_resources".to_string(),
            args: [
                ("query".to_string(), serde_json::json!("Invented Org")),
                ("lookup_mode".to_string(), serde_json::json!("contact")),
            ]
            .into(),
        };
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![contextual_contact_call.clone()], false),
                expected,
            ),
            Some("find_resources query was not grounded in recent conversation")
        );
        for invalid_query in [
            "email",
            "the",
            "organization",
            "legal help",
            "Mexico",
            "Acme",
            "Women",
            "Acme Legal",
        ] {
            contextual_contact_call
                .args
                .insert("query".to_string(), serde_json::json!(invalid_query));
            assert_eq!(
                agent.curated_resource_plan_violation(
                    &ToolDecision::new(vec![contextual_contact_call.clone()], false),
                    expected,
                ),
                Some("find_resources query was not grounded in recent conversation"),
                "{invalid_query}"
            );
        }
        contextual_contact_call
            .args
            .insert("query".to_string(), serde_json::json!("Acme Legal Aid"));
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![contextual_contact_call.clone()], false),
                expected,
            ),
            Some("find_resources region did not preserve the requested location")
        );
        contextual_contact_call
            .args
            .insert("region".to_string(), serde_json::json!("MX"));
        assert!(agent
            .curated_resource_plan_violation(
                &ToolDecision::new(vec![contextual_contact_call], false),
                expected,
            )
            .is_none());
        assert!(agent
            .curated_resource_lookup_expectation
            .retry_instruction_for_violation(
                "find_resources query was not grounded in recent conversation"
            )
            .contains("Do not invent"));

        agent.curated_resource_lookup_expectation = curated_resource_lookup_expectation(
            "What is Acme Legal Aid's email?",
            Some(&continuation),
        );
        let expected = agent.expected_curated_resources(&enabled);
        let fresh_contact_call = ToolCall {
            name: "find_resources".to_string(),
            args: [
                ("query".to_string(), serde_json::json!("Acme Legal Aid")),
                ("lookup_mode".to_string(), serde_json::json!("contact")),
            ]
            .into(),
        };
        assert!(agent
            .curated_resource_plan_violation(
                &ToolDecision::new(vec![fresh_contact_call], false),
                expected,
            )
            .is_none());
        let stale_offset_contact_call = ToolCall {
            name: "find_resources".to_string(),
            args: [
                ("query".to_string(), serde_json::json!("Acme Legal Aid")),
                ("lookup_mode".to_string(), serde_json::json!("contact")),
                ("offset".to_string(), serde_json::json!(10)),
            ]
            .into(),
        };
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![stale_offset_contact_call], false),
                expected,
            ),
            Some("fresh find_resources lookup used a stale positive offset")
        );

        let duplicate_resource_calls = vec![
            ToolCall {
                name: "find_resources".to_string(),
                args: [("query".to_string(), serde_json::json!("Acme Legal Aid"))].into(),
            },
            ToolCall {
                name: "find_resources".to_string(),
                args: [("query".to_string(), serde_json::json!("wrong"))].into(),
            },
        ];
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(duplicate_resource_calls, false),
                expected,
            ),
            Some("multiple find_resources calls were not allowed in one plan")
        );
        agent.curated_resource_lookup_succeeded = true;
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(
                    vec![ToolCall {
                        name: "find_resources".to_string(),
                        args: [("query".to_string(), serde_json::json!("Acme Legal Aid"))].into(),
                    }],
                    false,
                ),
                false,
            ),
            Some("additional find_resources call was not allowed after turn success")
        );
        assert!(agent
            .curated_resource_lookup_expectation
            .retry_instruction_for_violation(
                "additional find_resources call was not allowed after turn success"
            )
            .contains("Do not call find_resources again"));
        agent.curated_resource_lookup_succeeded = false;

        agent.curated_resource_lookup_expectation = curated_resource_lookup_expectation(
            "Show the next page of those matching resources.",
            None,
        );
        let expected = agent.expected_curated_resources(&enabled);
        let invented_continuation = ToolCall {
            name: "find_resources".to_string(),
            args: [("offset".to_string(), serde_json::json!(1))].into(),
        };
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![invented_continuation], false),
                expected,
            ),
            Some("find_resources continuation cursor was unavailable")
        );

        agent.curated_resource_lookup_expectation = curated_resource_lookup_expectation(
            "List ready Curated Resources matching 'Acme Legal Aid'.",
            None,
        );
        let expected = agent.expected_curated_resources(&enabled);
        let wrong_matching_filter = ToolCall {
            name: "find_resources".to_string(),
            args: [("query".to_string(), serde_json::json!("aid"))].into(),
        };
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(vec![wrong_matching_filter], false),
                expected,
            ),
            Some("find_resources query did not preserve the requested filter")
        );
        agent.curated_resource_lookup_expectation =
            curated_resource_lookup_expectation("List resources for Acme Legal Aid.", None);
        let expected = agent.expected_curated_resources(&enabled);
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(
                    vec![ToolCall {
                        name: "find_resources".to_string(),
                        args: ToolArgs::new(),
                    }],
                    false,
                ),
                expected,
            ),
            Some("required find_resources query was omitted")
        );
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(
                    vec![ToolCall {
                        name: "find_resources".to_string(),
                        args: [("query".to_string(), serde_json::json!("Other Org"))].into(),
                    }],
                    false,
                ),
                expected,
            ),
            Some("find_resources query did not preserve the requested filter")
        );
        agent.curated_resource_lookup_expectation = curated_resource_lookup_expectation(
            "List resources for Acme Legal Aid in France.",
            None,
        );
        let expected = agent.expected_curated_resources(&enabled);
        assert!(agent
            .curated_resource_plan_violation(
                &ToolDecision::new(
                    vec![ToolCall {
                        name: "find_resources".to_string(),
                        args: [
                            ("query".to_string(), serde_json::json!("Acme Legal Aid")),
                            ("region".to_string(), serde_json::json!("France")),
                            ("lookup_mode".to_string(), serde_json::json!("inventory")),
                        ]
                        .into(),
                    }],
                    false,
                ),
                expected,
            )
            .is_none());
        agent.curated_resource_lookup_expectation = curated_resource_lookup_expectation(
            "List ready Curated Resources matching Acme Legal Aid.",
            None,
        );
        let expected = agent.expected_curated_resources(&enabled);
        assert_eq!(
            agent.curated_resource_plan_violation(
                &ToolDecision::new(
                    vec![ToolCall {
                        name: "find_resources".to_string(),
                        args: [("query".to_string(), serde_json::json!("aid"))].into(),
                    }],
                    false,
                ),
                expected,
            ),
            Some("find_resources query did not preserve the requested filter")
        );
    }

    #[test]
    fn resource_plan_sanitizer_keeps_one_valid_call_and_drops_post_success_calls() {
        let mut registry = ToolRegistry::new();
        registry.register_descriptor("find_resources", "lookup", r#"{"query":"text"}"#);
        registry.register_descriptor("knowledge_search", "search", r#"{"query":"text"}"#);
        let mut agent = SageAgent::new_without_memory(registry, "test");
        agent.curated_resource_lookup_expectation =
            curated_resource_lookup_expectation("What is Acme Legal Aid's email?", None);
        let expected = agent.expected_curated_resources(&agent.tools.names());
        let knowledge_call = ToolCall {
            name: "knowledge_search".to_string(),
            args: [("query".to_string(), serde_json::json!("manual policy"))].into(),
        };
        let wrong_resource_call = ToolCall {
            name: "find_resources".to_string(),
            args: [
                ("query".to_string(), serde_json::json!("wrong")),
                ("lookup_mode".to_string(), serde_json::json!("contact")),
            ]
            .into(),
        };
        let valid_resource_call = ToolCall {
            name: "find_resources".to_string(),
            args: [
                ("query".to_string(), serde_json::json!("Acme Legal Aid")),
                ("lookup_mode".to_string(), serde_json::json!("contact")),
            ]
            .into(),
        };
        let mut decision = ToolDecision::new(
            vec![
                knowledge_call.clone(),
                wrong_resource_call.clone(),
                valid_resource_call.clone(),
            ],
            false,
        );
        agent.sanitize_curated_resource_calls(&mut decision, expected);
        assert_eq!(decision.tool_calls.len(), 2);
        assert_eq!(decision.tool_calls[0].name, "knowledge_search");
        assert_eq!(
            tool_string_arg(&decision.tool_calls[1].args, "query"),
            Some("Acme Legal Aid")
        );
        assert!(agent
            .curated_resource_plan_violation(&decision, expected)
            .is_none());

        let mut all_invalid = ToolDecision::new(
            vec![wrong_resource_call.clone(), wrong_resource_call],
            false,
        );
        agent.sanitize_curated_resource_calls(&mut all_invalid, expected);
        assert_eq!(all_invalid.tool_calls.len(), 2);
        assert_eq!(
            agent.curated_resource_plan_violation(&all_invalid, expected),
            Some("multiple find_resources calls were not allowed in one plan")
        );

        agent.curated_resource_lookup_succeeded = true;
        let mut post_success = ToolDecision::new(vec![valid_resource_call, knowledge_call], false);
        agent.sanitize_curated_resource_calls(&mut post_success, false);
        assert_eq!(post_success.tool_calls.len(), 1);
        assert_eq!(post_success.tool_calls[0].name, "knowledge_search");
        assert!(agent
            .curated_resource_plan_violation(&post_success, false)
            .is_none());
    }

    #[test]
    fn done_does_not_make_a_registry_actionable() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(crate::tools::DoneTool));

        assert!(!registry.has_actionable_tools());

        registry.register_descriptor("lookup", "Look up a fact", r#"{"query":"text"}"#);
        assert!(registry.has_actionable_tools());
    }

    #[test]
    fn textual_tool_transcripts_are_distinct_from_explanatory_prose() {
        assert!(has_syntactic_tool_intent(
            "Tool calls: find_resources(lookup_mode=\"inventory\", query=\"Issue 539 Inventory\", offset=10)"
        ));
        assert!(has_syntactic_tool_intent(
            "Tool calls\n\nFunction call: find_resources\n\nArguments\n\n{\"query\":\"Issue 539 Inventory\",\"lookup_mode\":\"inventory\",\"offset\":20}"
        ));
        assert!(has_syntactic_tool_intent(
            "Tool calls \r\n\r\nFunction call: find_resources\r\n\r\nArguments:\r\n\r\n{\"offset\":20}"
        ));
        assert!(!has_syntactic_tool_intent(
            "Tool calls\n\nFunction call: a named section in the Activity panel.\n\nThis page explains the transcript format."
        ));
        assert!(!has_syntactic_tool_intent(
            "Tool calls\n\nFunction call: find_resources\n\nThis page explains how the Activity panel is formatted."
        ));
        assert!(has_syntactic_tool_intent(
            "I will search now. Tool calls: knowledge_search(query=\"referral\")\nTool Result: found one"
        ));
        assert!(has_syntactic_tool_intent(
            "I'll look up the current contact details. Tool: find_resources(help_type=\"legal\")"
        ));
        assert!(has_syntactic_tool_intent(
            "Tool decision: find_resources\n\nArgs:\n```json\n{\"offset\":30}\n```"
        ));
        assert!(has_syntactic_tool_intent("Tool decision: find_resources"));
        assert!(has_syntactic_tool_intent("Tool: find_resources"));
        assert!(has_syntactic_tool_intent("Tool: done"));
        assert!(has_syntactic_tool_intent(
            "I will search. Tool: find_resources"
        ));
        assert!(has_syntactic_tool_intent(
            "Internal choice: Tool decision: find_resources"
        ));
        assert!(has_syntactic_tool_intent(
            "Tool: find_resources\n\nArgs:\n{\"query\":\"Issue 539 Inventory\"}"
        ));
        assert!(has_syntactic_tool_intent(
            "Tool decision: find_resources\n\nArgs:\n- region: \"Mexico\"\n- language: \"es\""
        ));
        assert!(has_syntactic_tool_intent(
            "Tool: find_resources\nArgs: help_type=\"legal\", region=\"Mexico\""
        ));
        assert!(has_syntactic_tool_intent(
            "Tool: find_resources\n```json\n{\"query\":\"Issue 539 Inventory\",\"offset\":30}\n```"
        ));
        assert!(!has_syntactic_tool_intent(
            "The Activity panel labels these sections Tool calls: and Tool Result: so you can audit the turn."
        ));
        assert!(!has_syntactic_tool_intent(
            "For example, the Activity panel may show Tool calls: knowledge_search(query=\"referral\")."
        ));
        assert!(!has_syntactic_tool_intent(
            "The Curated Resources Tool: finds vetted organizations when contact details are requested."
        ));
        assert!(!has_syntactic_tool_intent(
            "The Tool decision: section in Activity explains which lookup ran."
        ));
        assert!(!has_syntactic_tool_intent(
            "The Activity label is Tool: done"
        ));
        assert!(has_syntactic_tool_intent(
            "The Curated Resources Tool: finds vetted organizations. Tool: find_resources(query=\"legal aid\")"
        ));
        assert!(has_syntactic_tool_intent(
            "The Tool decision: section is explanatory.\nTool decision: find_resources\nArgs: {\"offset\":10}"
        ));
        assert!(has_syntactic_tool_intent(
            "Tool: find_resources will run\nArgs: {\"query\":\"legal aid\"}"
        ));
        assert!(has_syntactic_tool_intent(
            "I’m going to search. Tool: find_resources(query=\"legal aid\")"
        ));
        assert!(has_syntactic_tool_intent(
            "I'm going to search. Tool: find_resources(query=\"legal aid\")"
        ));
        assert!(has_syntactic_tool_intent(
            "I need to fetch fresh contact details for this resource before sharing them.Tool decision: find_resources with language=\"es\", query=\"Issue 539 Legal Aid\", help_type=\"legal\", region=\"Mexico\""
        ));
    }

    #[test]
    fn tool_decision_ignores_done_when_actionable_calls_are_present() {
        let decision = ToolDecision::new(
            vec![
                ToolCall {
                    name: "done".to_string(),
                    args: ToolArgs::new(),
                },
                ToolCall {
                    name: "knowledge_search".to_string(),
                    args: ToolArgs::from([("query".to_string(), serde_json::json!("safety plan"))]),
                },
            ],
            true,
        );

        assert_eq!(decision.tool_calls.len(), 1);
        assert_eq!(decision.tool_calls[0].name, "knowledge_search");
        assert!(decision.replan_after_results);
    }

    #[test]
    fn tool_decision_accepts_native_object_and_array_arguments() {
        let parsed = baml_bridge::parse_llm_output::<__ToolDecisionResponseOutput>(
            r#"{
                "tool_calls": [{
                    "name": "configure_instance",
                    "args": {
                        "settings": {
                            "instance_name": "FreeThem",
                            "auto_approve_users": false
                        },
                        "user_types": [{
                            "reference": "families",
                            "name": "Families",
                            "display_order": 1
                        }],
                        "behavior_rules": ["Be direct", "Protect privacy"]
                        ,"forbidden_topics": null
                    }
                }],
                "replan_after_results": false
            }"#,
            true,
        )
        .expect("native Tool arguments should parse");

        let args = &parsed.value.tool_calls[0].args;
        assert_eq!(
            args["settings"],
            serde_json::json!({
                "instance_name": "FreeThem",
                "auto_approve_users": false
            })
        );
        assert_eq!(
            args["user_types"],
            serde_json::json!([{
                "reference": "families",
                "name": "Families",
                "display_order": 1
            }])
        );
        assert_eq!(
            args["behavior_rules"],
            serde_json::json!(["Be direct", "Protect privacy"])
        );
        assert!(!args.contains_key("forbidden_topics"));
    }

    #[test]
    fn tool_decision_allows_omitted_replan_flag() {
        let parsed = baml_bridge::parse_llm_output::<__ToolDecisionResponseOutput>(
            r#"{
                "tool_calls": [{
                    "name": "knowledge_search",
                    "args": {"query": "release safety"}
                }]
            }"#,
            true,
        );

        assert!(
            parsed.is_ok(),
            "omitting the optional replan hint should not invalidate an otherwise usable Tool plan: {parsed:?}"
        );
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

        let prompt = agent.plain_answer_prompt("Give the final answer.");

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
        outcomes: Arc<Mutex<std::collections::VecDeque<Result<ToolResult>>>>,
    }

    struct SuccessfulResourceTool {
        metadata: serde_json::Value,
    }

    #[async_trait::async_trait]
    impl Tool for SuccessfulResourceTool {
        fn name(&self) -> &str {
            "find_resources"
        }

        fn description(&self) -> &str {
            "test-only resource lookup"
        }

        fn args_schema(&self) -> &str {
            "{}"
        }

        async fn execute(&self, _args: &ToolArgs) -> Result<ToolResult> {
            Ok(ToolResult::success_with_metadata(
                "resource result",
                self.metadata.clone(),
            ))
        }
    }

    #[async_trait::async_trait]
    impl Tool for ScriptedRetryTool {
        fn name(&self) -> &str {
            "scripted_lookup"
        }

        fn description(&self) -> &str {
            "test-only scripted lookup"
        }

        fn args_schema(&self) -> &str {
            "{}"
        }

        fn retry_policy(&self) -> ToolRetryPolicy {
            self.policy.clone()
        }

        async fn execute(&self, _args: &ToolArgs) -> Result<ToolResult> {
            self.outcomes
                .lock()
                .expect("scripted outcomes should lock")
                .pop_front()
                .expect("test should provide a scripted outcome")
        }
    }

    #[tokio::test]
    async fn successful_resource_metadata_fails_closed_when_pagination_is_unknown() {
        for (metadata, expected_incomplete) in [
            (serde_json::Value::Null, true),
            (serde_json::json!({"has_more": "unknown"}), true),
            (serde_json::json!({"has_more": true}), true),
            (serde_json::json!({"has_more": false}), false),
        ] {
            let mut registry = ToolRegistry::new();
            registry.register(Arc::new(SuccessfulResourceTool { metadata }));
            let mut agent = SageAgent::new_without_memory(registry, "test");
            let result = agent
                .execute_tool_decision(&ToolDecision::new(
                    vec![ToolCall {
                        name: "find_resources".to_string(),
                        args: ToolArgs::new(),
                    }],
                    false,
                ))
                .await;
            assert!(result.executed_tools[0].result.success);
            assert_eq!(
                agent.curated_resource_page_incomplete, expected_incomplete,
                "metadata: {:?}",
                result.executed_tools[0].result.metadata
            );
        }
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

        fn args_schema(&self) -> &str {
            "{}"
        }

        fn retry_policy(&self) -> ToolRetryPolicy {
            ToolRetryPolicy::read_only(Duration::from_millis(100), 2, Duration::from_millis(500))
        }

        async fn execute(&self, _args: &ToolArgs) -> Result<ToolResult> {
            match self
                .attempt
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            {
                0 => Err(anyhow::Error::new(ToolExecutionError::HttpStatus(503))),
                1 => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    Err(anyhow::Error::new(ToolExecutionError::Timeout))
                }
                _ => Ok(ToolResult::success("recovered")),
            }
        }
    }

    #[tokio::test]
    async fn read_only_retry_reuses_call_correlation_and_emits_one_terminal() {
        let outcomes = Arc::new(Mutex::new(std::collections::VecDeque::from([
            Err(anyhow::Error::new(ToolExecutionError::HttpStatus(503))),
            Ok(ToolResult::success("recovered")),
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

        let result = agent
            .execute_tool_decision(&ToolDecision {
                tool_calls: vec![ToolCall {
                    name: "scripted_lookup".to_string(),
                    args: ToolArgs::new(),
                }],
                replan_after_results: false,
                planning_round: 4,
            })
            .await;

        assert!(result.executed_tools[0].result.success);
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
        let result = agent
            .execute_tool_decision(&ToolDecision {
                tool_calls: vec![ToolCall {
                    name: "scripted_lookup".to_string(),
                    args: ToolArgs::new(),
                }],
                replan_after_results: false,
                planning_round: 1,
            })
            .await;
        assert!(!result.executed_tools[0].result.success);
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
            Ok(ToolResult::success("should not run")),
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
        let result = agent
            .execute_tool_decision(&ToolDecision {
                tool_calls: vec![ToolCall {
                    name: "scripted_lookup".to_string(),
                    args: ToolArgs::new(),
                }],
                replan_after_results: false,
                planning_round: 1,
            })
            .await;
        assert!(!result.executed_tools[0].result.success);
        assert_eq!(outcomes.lock().expect("outcomes should lock").len(), 1);
    }

    #[tokio::test]
    async fn connection_failure_retries_once_with_privacy_safe_reason() {
        let outcomes = Arc::new(Mutex::new(std::collections::VecDeque::from([
            Err(anyhow::Error::new(ToolExecutionError::Connection)),
            Ok(ToolResult::success("recovered")),
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
        let result = agent
            .execute_tool_decision(&ToolDecision {
                tool_calls: vec![ToolCall {
                    name: "scripted_lookup".to_string(),
                    args: ToolArgs::new(),
                }],
                replan_after_results: false,
                planning_round: 1,
            })
            .await;
        assert!(result.executed_tools[0].result.success);
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
            Ok(ToolResult::success("recovered")),
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
        let result = agent
            .execute_tool_decision(&ToolDecision {
                tool_calls: vec![ToolCall {
                    name: "scripted_lookup".to_string(),
                    args: ToolArgs::new(),
                }],
                replan_after_results: false,
                planning_round: 1,
            })
            .await;
        assert!(result.executed_tools[0].result.success);
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
        let result = agent
            .execute_tool_decision(&ToolDecision {
                tool_calls: vec![ToolCall {
                    name: "delayed_timeout_lookup".to_string(),
                    args: ToolArgs::new(),
                }],
                replan_after_results: false,
                planning_round: 1,
            })
            .await;
        assert!(!result.executed_tools[0].result.success);
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
