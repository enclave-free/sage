# AGENTS.md - Sage

## Project Overview

Sage is a privacy-first personal AI agent built in Rust. It stores data locally in PostgreSQL with pgvector, uses a 4-tier memory architecture inspired by Letta/MemGPT, communicates over Signal or Marmot, and runs inference through a local verified Tinfoil proxy instead of provider-native tool calling.

The core design constraints are:

- Local-first state and memory
- Typed DSR signatures instead of provider-native function calling
- Multi-user isolation by `agent_id`
- A messenger abstraction that can target Signal or Marmot
- TEE-backed inference via Tinfoil's hosted router behind the local proxy

## Repository Structure

```text
sage/
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml
├── Dockerfile
├── docker-compose.yml
├── flake.nix
├── justfile
├── .env.example
├── .githooks/
├── docs/
├── examples/
│   └── gepa/
├── optimized_instructions/
├── scripts/
│   └── smoke_tinfoil.sh
└── crates/
    ├── sage-core/
    │   ├── migrations/
    │   └── src/
    │       ├── main.rs
    │       ├── lib.rs
    │       ├── config.rs
    │       ├── agent_manager.rs
    │       ├── sage_agent.rs
    │       ├── messenger.rs
    │       ├── signal.rs
    │       ├── marmot.rs
    │       ├── tools.rs
    │       ├── shell_tool.rs
    │       ├── scheduler.rs
    │       ├── scheduler_tools.rs
    │       ├── storage.rs
    │       ├── schema.rs
    │       ├── vision.rs
    │       ├── memory/
    │       └── bin/
    │           └── gepa_optimize.rs
    └── sage-tools/
        └── src/
            ├── lib.rs
            ├── brave.rs
            └── web_search.rs
```

Key directories:

- `crates/sage-core/`: main application crate
- `crates/sage-tools/`: external tool integrations
- `crates/sage-core/migrations/`: Diesel migrations
- `docs/`: architecture and product notes
- `examples/gepa/`: GEPA training and validation inputs
- `optimized_instructions/`: optimized agent prompts
- `scripts/smoke_tinfoil.sh`: isolated pre-push Tinfoil + pgvector smoke gate

## Development Environment

### Preferred Setup

Prefer `nix develop` when possible. The Nix shell provides the Rust toolchain, PostgreSQL helpers, container tools, `just`, and other repo dependencies in one place.

If not using Nix, install at least:

- Rust stable toolchain
- Podman or Docker
- PostgreSQL tooling / `libpq`
- `signal-cli` for Signal work
- `just`

### Environment Variables

Copy `.env.example` to `.env` and configure the values you need.

Important runtime variables:

```bash
# Tinfoil / inference
TINFOIL_API_URL=http://localhost:8089/v1
TINFOIL_API_KEY=your-api-key
TINFOIL_MODEL=kimi-k2-5
TINFOIL_EMBEDDING_MODEL=nomic-embed-text
TINFOIL_VISION_MODEL=kimi-k2-5
TINFOIL_PROXY_PORT=8089
TINFOIL_ROUTER_HOST=inference.tinfoil.sh
TINFOIL_ROUTER_REPO=tinfoilsh/confidential-model-router

# Messenger selection
MESSENGER=signal   # or marmot

# Signal
SIGNAL_PHONE_NUMBER=+1234567890
SIGNAL_ALLOWED_USERS=uuid1,uuid2
SIGNAL_CLI_HOST=localhost
SIGNAL_CLI_PORT=7583

# Marmot
MARMOT_RELAYS=wss://relay.example
MARMOT_STATE_DIR=/data/marmot-state
MARMOT_ALLOWED_PUBKEYS=npub1...,hexpubkey,...
MARMOT_AUTO_ACCEPT_WELCOMES=true

# Database
DATABASE_URL=postgres://sage:sage@localhost:5434/sage

# Optional tools and tuning
BRAVE_API_KEY=
ANTHROPIC_API_KEY=
RUST_LOG=info
HEALTH_PORT=8080
SAGE_WORKSPACE=/workspace
```

Notes:

- Local default inference target is `http://localhost:8089/v1`.
- Compose uses `http://tinfoil-proxy:8089/v1` internally.
- `TINFOIL_API_KEY` is still passed to Sage for now, even though the long-term trust boundary is "Sage -> local proxy -> Tinfoil router".

## Build and Run

### Primary Workflow: `just`

Container and dev tasks are exposed in `justfile`.

```bash
just build
just start
just restart
just stop
just logs
just logs-all
just status
just shell
just psql
```

Useful variants:

- `MESSENGER=marmot just start`
- `just tinfoil-proxy-start`
- `just tinfoil-proxy-stop`
- `just tinfoil-proxy-logs`
- `just signal-init`
- `just smoke-tinfoil`

### Docker Compose

`docker-compose.yml` defines the full stack:

- `postgres`
- `signal-cli`
- `signal-cli-perms`
- `tinfoil-proxy`
- `sage`

Use it when you want the full compose topology rather than the host-network Podman workflow in `justfile`.

### Local Rust Development

```bash
just build-local
just run
just run-debug
just check
just test
just fmt
just lint
just ci-check
```

## Testing and Verification

### Core Checks

Use these before pushing:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

`just ci-check` runs the equivalent repo gate.

### Smoke Testing

`just smoke-tinfoil` runs the isolated messenger-free smoke gate. It validates:

- containerized `cargo check`, `cargo test`, and `cargo clippy`
- direct proxy chat completions
- embeddings shape and connectivity
- vision/image path
- invalid-model rejection
- recall-memory `NULL -> embedded -> searchable`
- archival pgvector search

This smoke test intentionally excludes Signal and Marmot delivery. It is a data-plane and inference-plane gate.

### Git Hooks

`just setup-hooks` configures `.githooks/pre-commit`, which runs repo checks before commit.

## Architecture and Design Patterns

### No Provider-Native Tool Calling

Sage does not use provider-native function calling. The agent uses DSRs and BAML-style parsing, so structured agent output is provider-agnostic and expressed through typed Rust signatures.

### Tinfoil Proxy Architecture

Sage talks to a local verified Tinfoil proxy over OpenAI-compatible HTTP:

- chat: `/chat/completions`
- embeddings: `/embeddings`
- vision preprocessing: `/chat/completions` with image content

Current target architecture:

- Sage -> local Tinfoil proxy
- proxy -> `inference.tinfoil.sh`
- router repo -> `tinfoilsh/confidential-model-router`

This keeps Sage simple while still benefiting from the Tinfoil verification path.

### Agent Loop

`SageAgent::step()` is the core loop:

1. Build context from memory blocks, recall, archival results, and summaries
2. Run the typed DSR signature
3. Execute requested tools
4. Feed tool results back into the next step until done or step limit

The main event loop in `main.rs` handles message intake, memory updates, tool execution, and messenger replies.

### DSR Signatures

Primary signatures live in `sage_agent.rs`, including the main agent response and correction/self-healing paths. If you change output structure or tool usage conventions, update both the signatures and the supporting examples/eval inputs.

### Memory System

Sage uses four coordinated memory tiers:

| Tier | Module | Storage | Purpose |
|------|--------|---------|---------|
| Core | `memory/block.rs` | `blocks` | persona / human memory always in context |
| Recall | `memory/recall_new.rs` | `messages` + embeddings | conversation history and semantic recall |
| Archival | `memory/archival_new.rs` | `passages` + pgvector | long-term semantic storage |
| Summary | `memory/compaction.rs` | `summaries` | compacted historical context |

Important current behavior:

- Messages are inserted synchronously for durability.
- Pending message embeddings start as `NULL`, not zero vectors.
- Background embedding fill updates the row later.
- Semantic recall only begins after real embeddings exist.

### Multi-User Isolation

`AgentManager` isolates state per agent/user:

- unique `agent_id` per chat context
- isolated memory and preferences
- isolated scheduled tasks
- isolated workspace under `SAGE_WORKSPACE/<agent_id>/`

Do not introduce shortcuts that let one agent read or mutate another agent's memory or workspace.

### Messenger Abstraction

Sage supports two messenger backends:

- `Signal`: production-oriented JSON-RPC via `signal-cli`
- `Marmot`: MLS-over-Nostr workflow via `marmotd`

`Config::from_env()` selects the backend with `MESSENGER`, defaulting to `signal`.

### Vision Pipeline

Image attachments are preprocessed by a vision-capable model in `vision.rs`, and the resulting description is injected into the text context the agent sees. Keep this path aligned with the configured `TINFOIL_VISION_MODEL`.

### Scheduler

Scheduled tasks live in PostgreSQL and are exposed through scheduler tools such as:

- `schedule_task`
- `list_schedules`
- `cancel_schedule`

If you change scheduling semantics, update the DB layer, tool layer, and any user-facing descriptions together.

## Tooling Conventions

Tools implement the shared trait in `sage_agent.rs` and are registered through the agent/tool registry flow. The descriptions shown to the model must stay aligned with the actual registered implementations.

Current major tool areas include:

- memory editing and search
- archival insert/search
- preferences
- scheduling
- shell execution
- web search
- `done`

When adding a new tool:

1. Implement the tool
2. Register it in agent creation / registry wiring
3. Update model-facing descriptions
4. Add or update GEPA examples if the new behavior matters for prompting

## Database and Migrations

PostgreSQL with pgvector backs all persistent storage.

Conventions:

- UUID primary keys for most entities
- pgvector for embeddings
- raw SQL where Diesel support is awkward
- schema and migrations must stay in sync

When touching migrations:

- update related schema expectations
- verify fresh-database behavior, not just upgraded databases
- keep docs in sync if the operational contract changes

## Coding Conventions

### Rust

- Edition 2021
- `anyhow::Result` in app code
- `thiserror` where typed library errors help
- Tokio async runtime
- `tracing` for logging
- string-typed tool arguments for LLM compatibility

### Organization

- `sage-core` contains the application and binaries
- `sage-tools` is intentionally narrower and focused on external integrations
- `memory/` is the main subsystem directory with multiple focused modules

### Prompt and Tool Changes

If you materially change `AGENT_INSTRUCTION`, tool semantics, or output structure:

1. Run `just gepa-eval`
2. Consider `just gepa-optimize`
3. Update examples/eval inputs if needed

## Security Considerations

- Never commit `.env` files or API keys
- Treat Signal allowlists and Marmot allowlists as security-sensitive
- Treat shell tool restrictions as security-sensitive
- Tinfoil and proxy configuration are part of the trust boundary
- Database data is private user memory; do not casually expose or export it

The shell tool is explicitly sensitive and must retain its guardrails around destructive or dangerous commands.

## Common Tasks

### Add a Migration

```bash
diesel migration generate your_migration_name --database-url postgres://sage:sage@localhost:5434/sage
```

Then edit `up.sql` / `down.sql`, validate the schema impact, and test both fresh and incremental paths when relevant.

### Add a Tool

Implement the tool, register it, expose the correct model-facing description, and update any prompt examples that depend on the new capability.

### Work on Prompts

The primary agent instruction lives in `sage_agent.rs`. Prompt work should be evaluated, not just edited.

### Smoke-Test Direct Tinfoil

Run:

```bash
just smoke-tinfoil
```

Use this before pushing changes that affect:

- Tinfoil config
- proxy wiring
- embeddings
- vision
- memory recall / archival behavior
- migrations

## Workflow

- Prefer `nix develop` for a consistent environment.
- Use the existing `just` and compose workflows instead of inventing new ad hoc startup flows.
- Update docs when operational behavior changes.
- Keep migration behavior, schema expectations, and runtime code aligned.
- Unless instructed otherwise, attempt to run the CodeRabbit CLI on unstaged changes before committing and pushing.
