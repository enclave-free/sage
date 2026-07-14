//! Sage Memory System
//!
//! A 4-tier memory architecture inspired by Letta/MemGPT:
//! 1. Core Memory - Editable blocks in system prompt (persona, human, etc.)
//! 2. Recall Memory - Full conversation history, searchable with embeddings
//! 3. Archival Memory - Long-term semantic storage with embeddings
//! 4. Summary Memory - Compaction when context overflows
//!
//! All conversation history is stored with embeddings for semantic search.
//! Uses PostgreSQL with pgvector for efficient similarity queries.

mod archival_new;
mod block;
mod compaction;
mod context;
mod db;
mod embedding;
mod recall_new;
mod tools;

pub use block::BlockManager;
// Use new database-backed managers
pub use archival_new::ArchivalManager;
pub use compaction::{CompactionManager, SummaryResult};
pub use context::ContextManager;
pub use db::{preference_keys, MemoryDb};
pub use embedding::EmbeddingService;
pub use recall_new::RecallManager;
pub use tools::{
    ArchivalInsertTool, ArchivalSearchTool, ConversationSearchTool, MemoryAppendTool,
    MemoryInsertTool, MemoryReplaceTool, SetPreferenceTool,
};

use anyhow::Result;
use std::{
    collections::HashMap,
    future::Future,
    sync::{Arc, Mutex as StdMutex, OnceLock, Weak},
};
use tokio::sync::Mutex as TokioMutex;
use uuid::Uuid;

use crate::sage_agent::Tool;
use db::{MessageRow, SummaryRow};

/// Default descriptions for memory blocks (from Letta)
pub const DEFAULT_PERSONA_DESCRIPTION: &str = "The persona block: Stores details about your current persona, guiding how you behave and respond. This helps you to maintain consistency and personality in your interactions.";

pub const DEFAULT_HUMAN_DESCRIPTION: &str = "The human block: Stores key details about the person you are conversing with, allowing for more personalized and friend-like conversation.";

/// Constants for context management
/// Note: Kimi K2 supports 256k tokens, but using 100k for faster compaction testing
pub const DEFAULT_CONTEXT_WINDOW: usize = 100_000;
#[allow(dead_code)]
pub const COMPACTION_THRESHOLD: f32 = 0.80; // 80% threshold (80k tokens triggers compaction)
pub const MIN_MESSAGES_IN_CONTEXT: usize = 20; // Always show at least 20 messages after compaction

#[async_trait::async_trait]
trait DeferredMessageBackend: Clone + Send + Sync + 'static {
    fn insert_pending_message(&self, user_id: &str, role: &str, content: &str) -> Result<Uuid>;

    async fn update_message_embedding(&self, message_id: Uuid, content: &str) -> Result<()>;
}

#[async_trait::async_trait]
impl DeferredMessageBackend for RecallManager {
    fn insert_pending_message(&self, user_id: &str, role: &str, content: &str) -> Result<Uuid> {
        self.add_message_sync(user_id, role, content)
    }

    async fn update_message_embedding(&self, message_id: Uuid, content: &str) -> Result<()> {
        self.update_embedding(message_id, content).await
    }
}

/// Persists a durable message row first, then fills its remote embedding in a
/// detached task. A failed embedding intentionally leaves the row with a NULL
/// embedding so a repair worker can find and retry it later.
struct DeferredMessageStore<B> {
    backend: B,
}

/// Serializes threshold compactions and rejects a stale threshold observation
/// after another task has already advanced the summary boundary.
#[derive(Clone)]
struct CompactionCoordinator {
    lock: Arc<TokioMutex<()>>,
}

impl CompactionCoordinator {
    fn for_agent(agent_id: Uuid) -> Self {
        static REGISTRY: OnceLock<StdMutex<HashMap<Uuid, Weak<TokioMutex<()>>>>> = OnceLock::new();

        let registry = REGISTRY.get_or_init(|| StdMutex::new(HashMap::new()));
        let mut coordinators = registry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        coordinators.retain(|_, coordinator| coordinator.strong_count() > 0);

        if let Some(lock) = coordinators.get(&agent_id).and_then(Weak::upgrade) {
            return Self { lock };
        }

        let lock = Arc::new(TokioMutex::new(()));
        coordinators.insert(agent_id, Arc::downgrade(&lock));
        Self { lock }
    }

    async fn run_if_unchanged_and_needed<Check, Compact, CompactFuture>(
        &self,
        observed_generation: i64,
        check: Check,
        compact: Compact,
    ) -> Result<bool>
    where
        Check: FnOnce() -> Result<(i64, bool)>,
        Compact: FnOnce() -> CompactFuture,
        CompactFuture: Future<Output = Result<()>>,
    {
        let _guard = self.lock.lock().await;
        let (current_generation, still_needed) = check()?;
        if current_generation != observed_generation || !still_needed {
            return Ok(false);
        }

        compact().await?;
        Ok(true)
    }
}

async fn finish_persisted_message<CompactionFuture>(
    message_id: Uuid,
    compaction: CompactionFuture,
) -> (Uuid, bool)
where
    CompactionFuture: Future<Output = Result<bool>>,
{
    let compacted = match compaction.await {
        Ok(compacted) => compacted,
        Err(error) => {
            tracing::warn!(
                message_id = %message_id,
                error = %error,
                "message persisted, but Session Memory compaction maintenance failed"
            );
            false
        }
    };
    (message_id, compacted)
}

impl<B> DeferredMessageStore<B>
where
    B: DeferredMessageBackend,
{
    fn new(backend: B) -> Self {
        Self { backend }
    }

    fn store(&self, user_id: &str, role: &str, content: &str) -> Result<Uuid> {
        let message_id = self
            .backend
            .insert_pending_message(user_id, role, content)?;
        let backend = self.backend.clone();
        let embedding_content = content.to_string();

        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => {
                runtime.spawn(async move {
                    if let Err(error) = backend
                        .update_message_embedding(message_id, &embedding_content)
                        .await
                    {
                        tracing::warn!(
                            message_id = %message_id,
                            error = %error,
                            "failed to update deferred message embedding; row remains pending"
                        );
                    }
                });
            }
            Err(error) => {
                tracing::warn!(
                    message_id = %message_id,
                    error = %error,
                    "could not schedule deferred message embedding; row remains pending"
                );
            }
        }

        Ok(message_id)
    }
}

/// Main memory manager that coordinates all memory tiers
#[allow(dead_code)]
pub struct MemoryManager {
    agent_id: Uuid,
    db: MemoryDb,
    embedding: EmbeddingService,
    blocks: BlockManager,
    recall: RecallManager,
    deferred_messages: DeferredMessageStore<RecallManager>,
    archival: ArchivalManager,
    compaction: CompactionManager,
    context: ContextManager,
    /// Coordinates compaction and rejects stale concurrent threshold checks.
    compaction_coordinator: CompactionCoordinator,
}

#[allow(dead_code)]
impl MemoryManager {
    /// Create a new memory manager for an agent
    pub async fn new(
        agent_id: Uuid,
        db_url: &str,
        embedding_api_url: &str,
        embedding_api_key: &str,
        embedding_model: &str,
    ) -> Result<Self> {
        // Create shared database connection
        let db = MemoryDb::new(db_url)?;

        // Ensure the agent exists in the database (needed for foreign key constraints)
        db.agents().ensure_agent_exists(agent_id, "sage")?;

        // Create shared embedding service
        let embedding =
            EmbeddingService::new(embedding_api_url, embedding_api_key, embedding_model);

        // Initialize memory tiers - BlockManager now uses database
        let blocks = BlockManager::new(agent_id, db.clone())?;
        let recall = RecallManager::new(agent_id, db.clone(), embedding.clone());
        let deferred_messages = DeferredMessageStore::new(recall.clone());
        let archival = ArchivalManager::new(agent_id, db.clone(), embedding.clone());
        let compaction = CompactionManager::new();
        let context = ContextManager::new(DEFAULT_CONTEXT_WINDOW);

        Ok(Self {
            agent_id,
            db,
            embedding,
            blocks,
            recall,
            deferred_messages,
            archival,
            compaction,
            context,
            compaction_coordinator: CompactionCoordinator::for_agent(agent_id),
        })
    }

    /// Get the agent ID
    pub fn agent_id(&self) -> Uuid {
        self.agent_id
    }

    /// Store a message in recall memory with embedding
    pub async fn store_message(&self, user_id: &str, role: &str, content: &str) -> Result<Uuid> {
        self.recall.add_message(user_id, role, content).await
    }

    /// Store a message WITHOUT embedding (fast, synchronous)
    /// Use update_message_embedding() in background to add embedding later
    pub fn store_message_sync(&self, user_id: &str, role: &str, content: &str) -> Result<Uuid> {
        self.recall.add_message_sync(user_id, role, content)
    }

    /// Store a durable message immediately and populate its embedding outside
    /// the caller's response critical path.
    pub fn store_message_deferred(&self, user_id: &str, role: &str, content: &str) -> Result<Uuid> {
        self.deferred_messages.store(user_id, role, content)
    }

    /// Store a message with optional image attachment description (fast, synchronous)
    pub fn store_message_sync_with_attachment(
        &self,
        user_id: &str,
        role: &str,
        content: &str,
        attachment_text: Option<&str>,
    ) -> Result<Uuid> {
        self.recall
            .add_message_sync_with_attachment(user_id, role, content, attachment_text)
    }

    /// Update embedding for a message (call in background after store_message_sync)
    pub async fn update_message_embedding(&self, message_id: Uuid, content: &str) -> Result<()> {
        self.recall.update_embedding(message_id, content).await
    }

    /// Get recent messages from recall memory with timestamps
    /// Returns (role, content, created_at)
    pub fn get_recent_messages(
        &self,
        limit: usize,
    ) -> Result<Vec<(String, String, chrono::DateTime<chrono::Utc>)>> {
        let messages = self.recall.get_recent(limit)?;
        Ok(messages
            .into_iter()
            .map(|m| (m.role, m.content, m.created_at))
            .collect())
    }

    /// Compile memory blocks into XML for system prompt injection
    pub fn compile(&self) -> String {
        self.blocks.compile()
    }

    /// Compile memory metadata (counts, timestamps)
    pub fn compile_metadata(&self) -> String {
        let recall_count = self.recall.message_count();
        let archival_count = self.archival.passage_count();
        let last_modified = self.blocks.last_modified();

        let mut s = String::new();

        if let Some(modified) = last_modified {
            s.push_str(&format!(
                "- Memory blocks last modified: {}\n",
                modified.format("%Y-%m-%d %H:%M:%S %Z")
            ));
        }

        s.push_str(&format!(
            "- {} messages in recall memory (use conversation_search to access)\n",
            recall_count
        ));
        s.push_str(&format!(
            "- {} passages in archival memory (use archival_search to access)",
            archival_count
        ));

        s
    }

    /// Get all memory tools for the agent
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![
            Arc::new(MemoryReplaceTool::new(self.blocks.clone())),
            Arc::new(MemoryAppendTool::new(self.blocks.clone())),
            Arc::new(MemoryInsertTool::new(self.blocks.clone())),
            Arc::new(ConversationSearchTool::new(self.recall.clone())),
            Arc::new(ArchivalInsertTool::new(self.archival.clone())),
            Arc::new(ArchivalSearchTool::new(self.archival.clone())),
            Arc::new(SetPreferenceTool::new(self.db.clone(), self.agent_id)),
        ]
    }

    /// Get a user preference by key
    pub fn get_preference(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .db
            .preferences()
            .get(self.agent_id, key)?
            .map(|p| p.value))
    }

    /// Get the user's timezone preference (if set)
    pub fn get_timezone(&self) -> Result<Option<chrono_tz::Tz>> {
        if let Some(tz_str) = self.get_preference(preference_keys::TIMEZONE)? {
            Ok(Some(tz_str.parse::<chrono_tz::Tz>().map_err(|_| {
                anyhow::anyhow!("Invalid timezone stored: {}", tz_str)
            })?))
        } else {
            Ok(None)
        }
    }

    /// Get the latest summary for this agent (if any)
    pub fn get_latest_summary(&self) -> Result<Option<SummaryRow>> {
        self.db.summaries().get_latest(self.agent_id)
    }

    /// Get messages for context building
    /// - No summary yet: Load ALL messages (need to build up to hit compaction threshold)
    /// - Has summary: Load messages after summary boundary, with minimum of MIN_MESSAGES_IN_CONTEXT
    pub fn get_context_messages(&self) -> Result<(Option<SummaryRow>, Vec<MessageRow>)> {
        let summary = self.get_latest_summary()?;

        let messages = if let Some(ref s) = summary {
            // Has summary - get messages after summary boundary
            let after_summary = self.db.summaries().get_messages_after_sequence(
                self.agent_id,
                s.to_sequence_id,
                10000, // High limit
            )?;

            // Ensure minimum messages for context continuity (some may overlap with summary)
            if after_summary.len() < MIN_MESSAGES_IN_CONTEXT {
                self.db
                    .messages()
                    .get_recent(self.agent_id, MIN_MESSAGES_IN_CONTEXT as i64)?
            } else {
                after_summary
            }
        } else {
            // No summary yet - load ALL messages so we can build up to compaction threshold
            // Without this, we'd never accumulate enough context to trigger compaction
            self.db.messages().get_recent(self.agent_id, 100000)? // Effectively unlimited
        };

        Ok((summary, messages))
    }

    /// Store a message and check if compaction is needed
    /// Returns the message ID and whether compaction was triggered
    pub async fn store_message_with_compaction_check(
        &self,
        user_id: &str,
        role: &str,
        content: &str,
    ) -> Result<(Uuid, bool)> {
        // The durable row is committed before any remote embedding or
        // compaction work. Embedding completion is deliberately detached.
        let message_id = self.store_message_deferred(user_id, role, content)?;
        Ok(finish_persisted_message(message_id, self.compact_if_needed()).await)
    }

    fn compaction_state(&self) -> Result<(i64, bool, usize)> {
        let (summary, messages) = self.get_context_messages()?;
        let generation = summary
            .as_ref()
            .map(|summary| summary.to_sequence_id)
            .unwrap_or(0);
        let current_tokens = self.estimate_context_tokens(&summary, &messages);
        let needed = self.compaction.should_compact(
            current_tokens,
            DEFAULT_CONTEXT_WINDOW,
            COMPACTION_THRESHOLD,
        );
        Ok((generation, needed, current_tokens))
    }

    async fn compact_if_needed(&self) -> Result<bool> {
        let (observed_generation, needed, current_tokens) = self.compaction_state()?;
        if !needed {
            return Ok(false);
        }

        tracing::info!(
            "Context tokens ({}) exceed threshold ({}), checking compaction generation",
            current_tokens,
            (DEFAULT_CONTEXT_WINDOW as f32 * COMPACTION_THRESHOLD) as usize
        );

        self.compaction_coordinator
            .run_if_unchanged_and_needed(
                observed_generation,
                || {
                    let (generation, needed, _) = self.compaction_state()?;
                    Ok((generation, needed))
                },
                || async { self.run_compaction_unlocked().await.map(|_| ()) },
            )
            .await
    }

    /// Run compaction with mutex lock to prevent concurrent compaction
    pub async fn run_compaction(&self) -> Result<SummaryResult> {
        let _lock = self.compaction_coordinator.lock.lock().await;
        tracing::info!("Acquired compaction lock, starting compaction");
        self.run_compaction_unlocked().await
    }

    async fn run_compaction_unlocked(&self) -> Result<SummaryResult> {
        // Get current state
        let current_summary = self.get_latest_summary()?;
        let summary_boundary = current_summary
            .as_ref()
            .map(|s| s.to_sequence_id)
            .unwrap_or(0);

        // Get messages after the current summary boundary
        let messages = self.db.summaries().get_messages_after_sequence(
            self.agent_id,
            summary_boundary,
            1000, // Get all messages after summary
        )?;

        if messages.is_empty() {
            anyhow::bail!("No messages to compact");
        }

        // Decide what to summarize: keep ~50% of messages in context
        let keep_count = (messages.len() / 2).max(MIN_MESSAGES_IN_CONTEXT);
        let to_summarize_count = messages.len().saturating_sub(keep_count);

        if to_summarize_count == 0 {
            anyhow::bail!(
                "Not enough messages to compact (need to keep {} minimum)",
                MIN_MESSAGES_IN_CONTEXT
            );
        }

        let messages_to_summarize = &messages[..to_summarize_count];
        let from_sequence_id = messages_to_summarize.first().unwrap().sequence_id;
        let to_sequence_id = messages_to_summarize.last().unwrap().sequence_id;

        tracing::info!(
            "Compacting {} messages (sequence {} to {}), keeping {} in context",
            to_summarize_count,
            from_sequence_id,
            to_sequence_id,
            messages.len() - to_summarize_count
        );

        // Format messages for summarization
        let new_messages = messages_to_summarize
            .iter()
            .map(|m| format!("[{}]: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n---\n");

        // Get previous summary content
        let previous_summary = current_summary
            .as_ref()
            .map(|s| s.content.as_str())
            .unwrap_or("");
        let previous_summary_id = current_summary.as_ref().map(|s| s.id);

        // Run summarization with retry
        let result = self
            .compaction
            .summarize(
                previous_summary,
                &new_messages,
                from_sequence_id,
                to_sequence_id,
                previous_summary_id,
            )
            .await?;

        // Generate embedding for the summary
        let embedding = self.embedding.embed(&result.summary).await?;

        // Store the summary in the database
        self.db.summaries().insert_summary(
            self.agent_id,
            result.from_sequence_id,
            result.to_sequence_id,
            &result.summary,
            &embedding,
            result.previous_summary_id,
        )?;

        tracing::info!(
            "Compaction complete, created summary covering sequence {} to {}",
            result.from_sequence_id,
            result.to_sequence_id
        );

        Ok(result)
    }

    /// Estimate token count for context (summary + messages)
    fn estimate_context_tokens(
        &self,
        summary: &Option<SummaryRow>,
        messages: &[MessageRow],
    ) -> usize {
        // Rough estimate: ~4 chars per token
        let summary_chars = summary.as_ref().map(|s| s.content.len()).unwrap_or(0);
        let message_chars: usize = messages
            .iter()
            .map(|m| m.content.len() + m.role.len() + 10)
            .sum();
        (summary_chars + message_chars) / 4
    }

    /// Search summaries by semantic similarity
    pub async fn search_summaries(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<db::SummarySearchResult>> {
        let embedding = self.embedding.embed(query).await?;
        self.db
            .summaries()
            .search_by_embedding(self.agent_id, &embedding, limit as i64)
    }

    /// Get a mutable reference to the block manager
    pub fn blocks_mut(&mut self) -> &mut BlockManager {
        &mut self.blocks
    }

    /// Get a reference to the block manager
    pub fn blocks(&self) -> &BlockManager {
        &self.blocks
    }

    /// Get a reference to the recall manager
    pub fn recall(&self) -> &RecallManager {
        &self.recall
    }

    /// Get a reference to the archival manager
    pub fn archival(&self) -> &ArchivalManager {
        &self.archival
    }

    /// Get a reference to the database
    pub fn db(&self) -> &MemoryDb {
        &self.db
    }
}

#[cfg(test)]
mod deferred_message_tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicI64, AtomicUsize, Ordering},
        Arc, Mutex,
    };
    use tokio::sync::Notify;

    #[derive(Clone, Default)]
    struct TestMessageBackend {
        inserted: Arc<Mutex<Vec<(Uuid, String)>>>,
        embedding_started: Arc<Notify>,
        release_embedding: Arc<Notify>,
        embedded: Arc<Mutex<Vec<Uuid>>>,
    }

    #[derive(Clone, Default)]
    struct FailingMessageBackend {
        inserted: Arc<Mutex<Vec<Uuid>>>,
        embedding_attempted: Arc<Notify>,
    }

    #[async_trait::async_trait]
    impl DeferredMessageBackend for FailingMessageBackend {
        fn insert_pending_message(
            &self,
            _user_id: &str,
            _role: &str,
            _content: &str,
        ) -> Result<Uuid> {
            let id = Uuid::new_v4();
            self.inserted
                .lock()
                .expect("inserted messages should lock")
                .push(id);
            Ok(id)
        }

        async fn update_message_embedding(&self, _message_id: Uuid, _content: &str) -> Result<()> {
            self.embedding_attempted.notify_one();
            anyhow::bail!("embedding service unavailable")
        }
    }

    #[derive(Clone, Default)]
    struct OutOfOrderMessageBackend {
        inserted: Arc<Mutex<Vec<(Uuid, String)>>>,
        release_first: Arc<Notify>,
        second_embedded: Arc<Notify>,
        embedded: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl DeferredMessageBackend for OutOfOrderMessageBackend {
        fn insert_pending_message(
            &self,
            _user_id: &str,
            _role: &str,
            content: &str,
        ) -> Result<Uuid> {
            let id = Uuid::new_v4();
            self.inserted
                .lock()
                .expect("inserted messages should lock")
                .push((id, content.to_string()));
            Ok(id)
        }

        async fn update_message_embedding(&self, _message_id: Uuid, content: &str) -> Result<()> {
            if content == "first" {
                self.release_first.notified().await;
            }
            self.embedded
                .lock()
                .expect("embedded messages should lock")
                .push(content.to_string());
            if content == "second" {
                self.second_embedded.notify_one();
            }
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl DeferredMessageBackend for TestMessageBackend {
        fn insert_pending_message(
            &self,
            _user_id: &str,
            _role: &str,
            content: &str,
        ) -> Result<Uuid> {
            let id = Uuid::new_v4();
            self.inserted
                .lock()
                .expect("inserted messages should lock")
                .push((id, content.to_string()));
            Ok(id)
        }

        async fn update_message_embedding(&self, message_id: Uuid, _content: &str) -> Result<()> {
            self.embedding_started.notify_one();
            self.release_embedding.notified().await;
            self.embedded
                .lock()
                .expect("embedded messages should lock")
                .push(message_id);
            Ok(())
        }
    }

    #[tokio::test]
    async fn deferred_store_returns_durable_id_before_embedding_and_eventually_updates_it() {
        let backend = TestMessageBackend::default();
        let store = DeferredMessageStore::new(backend.clone());

        let message_id = store
            .store("admin:1", "user", "Show the current deployment config")
            .expect("pending message should be inserted");

        assert_eq!(
            backend
                .inserted
                .lock()
                .expect("inserted messages should lock")
                .as_slice(),
            &[(message_id, "Show the current deployment config".to_string())]
        );
        assert!(backend
            .embedded
            .lock()
            .expect("embedded messages should lock")
            .is_empty());

        backend.embedding_started.notified().await;
        backend.release_embedding.notify_one();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if backend
                    .embedded
                    .lock()
                    .expect("embedded messages should lock")
                    .as_slice()
                    == [message_id]
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("embedding should eventually update the inserted row");
    }

    #[tokio::test]
    async fn embedding_failure_does_not_fail_or_remove_the_pending_message() {
        let backend = FailingMessageBackend::default();
        let store = DeferredMessageStore::new(backend.clone());

        let message_id = store
            .store("admin:1", "assistant", "The deployment is ready")
            .expect("durable insertion should not depend on embedding success");
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            backend.embedding_attempted.notified(),
        )
        .await
        .expect("background embedding should be attempted");

        assert_eq!(
            backend
                .inserted
                .lock()
                .expect("inserted messages should lock")
                .as_slice(),
            &[message_id]
        );
    }

    #[tokio::test]
    async fn durable_message_order_does_not_depend_on_embedding_completion_order() {
        let backend = OutOfOrderMessageBackend::default();
        let store = DeferredMessageStore::new(backend.clone());

        let first_id = store
            .store("admin:1", "user", "first")
            .expect("first message should persist");
        let second_id = store
            .store("admin:1", "assistant", "second")
            .expect("second message should persist");
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            backend.second_embedded.notified(),
        )
        .await
        .expect("second embedding should finish while first is pending");

        assert_eq!(
            backend
                .inserted
                .lock()
                .expect("inserted messages should lock")
                .as_slice(),
            &[
                (first_id, "first".to_string()),
                (second_id, "second".to_string()),
            ]
        );
        assert_eq!(
            backend
                .embedded
                .lock()
                .expect("embedded messages should lock")
                .as_slice(),
            &["second".to_string()]
        );

        backend.release_first.notify_one();
    }

    #[tokio::test]
    async fn separate_managers_for_one_agent_compact_once_for_the_observed_summary_generation() {
        let agent_id = Uuid::new_v4();
        let first_coordinator = CompactionCoordinator::for_agent(agent_id);
        let second_coordinator = CompactionCoordinator::for_agent(agent_id);
        let summary_generation = Arc::new(AtomicI64::new(0));
        let compaction_count = Arc::new(AtomicUsize::new(0));

        let compact = |coordinator: CompactionCoordinator| {
            let summary_generation = summary_generation.clone();
            let compaction_count = compaction_count.clone();
            async move {
                coordinator
                    .run_if_unchanged_and_needed(
                        0,
                        || {
                            Ok::<_, anyhow::Error>((
                                summary_generation.load(Ordering::SeqCst),
                                true,
                            ))
                        },
                        || async {
                            tokio::task::yield_now().await;
                            compaction_count.fetch_add(1, Ordering::SeqCst);
                            summary_generation.fetch_add(1, Ordering::SeqCst);
                            Ok::<_, anyhow::Error>(())
                        },
                    )
                    .await
            }
        };

        let (first, second) = tokio::join!(compact(first_coordinator), compact(second_coordinator));
        let compacted = [
            first.expect("first threshold check should succeed"),
            second.expect("second threshold check should succeed"),
        ];

        assert_eq!(
            compacted.iter().filter(|&&did_compact| did_compact).count(),
            1
        );
        assert_eq!(compaction_count.load(Ordering::SeqCst), 1);
        assert_eq!(summary_generation.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn compaction_failure_after_insert_preserves_durable_id_for_trace_attachment() {
        let message_id = Uuid::new_v4();

        let (returned_id, compacted) = finish_persisted_message(message_id, async {
            anyhow::bail!("summary provider unavailable")
        })
        .await;

        assert_eq!(returned_id, message_id);
        assert!(!compacted);
    }
}
