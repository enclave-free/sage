# AGENTS.md - Sage Enclave Web Runtime Branch

## Project Overview

This branch turns Sage into the Enclave web AI runtime.

Treat this branch as:

- Sage owning the public AI routes
- Python remaining the Enclave control plane
- Tinfoil providing model inference
- Postgres storing Sage sessions and memory

Do not assume the primary runtime story is Signal or Marmot. Those modules still exist in the repo, but on this branch the main product path is `enclave_web`.

## Branch Note

- `master`: generic Sage runtime branch
- `enclave-web-prototype`: Enclave-specific web runtime branch
- `enclave-web-native-auth`: local alias in this checkout that currently points at the same commit

The operator entrypoint for the full system is the parent `enclave.free-prototype` repo. This repo is the pinned Sage fork used inside that stack.

## Repository Structure

```text
sage/
├── Cargo.toml
├── Dockerfile
├── justfile
├── .env.example
├── AGENTS.md
├── README.md
├── TODO.md
├── docs/
└── crates/
    ├── sage-core/
    │   ├── migrations/
    │   └── src/
    │       ├── bin/
    │       │   ├── enclave_web.rs
    │       │   └── gepa_optimize.rs
    │       ├── web_runtime.rs
    │       ├── sage_agent.rs
    │       ├── config.rs
    │       ├── schema.rs
    │       ├── memory/
    │       ├── main.rs
    │       ├── signal.rs
    │       └── marmot.rs
    └── sage-tools/
```

Key files for this branch:

- `crates/sage-core/src/bin/enclave_web.rs`
- `crates/sage-core/src/web_runtime.rs`
- `crates/sage-core/src/sage_agent.rs`
- `crates/sage-core/src/memory/`
- `crates/sage-core/src/schema.rs`
- `crates/sage-core/migrations/`

## Development Environment

Prefer `nix develop` when possible.

If not using Nix, install at least:

- Rust stable toolchain
- Docker or Podman
- PostgreSQL client tooling / `libpq`
- `just`

### Branch-critical env vars

```bash
DATABASE_URL=postgres://sage:sage@localhost:5434/sage
TINFOIL_API_URL=http://localhost:8089/v1
TINFOIL_API_KEY=your-api-key
TINFOIL_MODEL=gemma4-31b
TINFOIL_EMBEDDING_MODEL=nomic-embed-text
ENCLAVE_WEB_PORT=3000
ENCLAVE_BACKEND_URL=http://core-backend:8000
INTERNAL_AGENT_TOKEN=dev-internal-agent-token
SEARXNG_URL=http://searxng:8080
FRONTEND_URL=http://localhost:5173
CORS_ORIGINS=http://localhost:5173
USER_SESSION_COOKIE_NAME=enclave_session
ADMIN_SESSION_COOKIE_NAME=enclave_admin_session
CSRF_COOKIE_NAME=enclave_csrf
```

Messenger-specific env vars still matter only if you are working on the generic runtime modules rather than the Enclave web runtime path.

## Build And Run

### Recommended system run path

Run this branch through `enclave.free-prototype`, not through this repo's standalone compose, when you want the real product topology.

### Local branch checks

```bash
cargo check -p sage-core --bin enclave_web
cargo run -p sage-core --bin enclave_web
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

On macOS, Diesel/Postgres tests need Homebrew `libpq` on the linker search path. Use the helper task for streamed chat checks:

```bash
./scripts/test_chat_stream.sh
# or, when just is installed:
just test-chat-stream
```

This runs `cargo test -p sage-core chat_stream` after exporting `LIBRARY_PATH="$(brew --prefix libpq)/lib:${LIBRARY_PATH:-}"` on macOS.

### Useful `just` tasks

```bash
just build-local
just test
./scripts/test_chat_stream.sh
just test-chat-stream
just lint
just ci-check
just smoke-tinfoil
```

Use `just smoke-tinfoil` before pushing changes that affect:

- Tinfoil config
- embeddings
- migrations
- recall/archival behavior
- `enclave_web` runtime startup

## Testing And Verification

Core checks:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

Branch-specific check:

```bash
cargo check -p sage-core --bin enclave_web
```

## Architecture And Design Patterns

### Current runtime shape

The main branch-specific flow is:

1. gateway forwards a public AI route to Sage
2. Sage resolves auth through Enclave Python
3. Sage fetches effective config and retrieval context from Python as needed
4. Sage runs the agent loop against Tinfoil
5. Sage persists session/memory state in Postgres

### Public routes on this branch

- `POST /llm/chat`
- `POST /query`
- `GET /query/session/{session_id}`
- `DELETE /query/session/{session_id}`
- `GET /session-defaults`
- `POST /admin/tools/execute`
- `/admin/ai-config/*`

### Internal support contract

`web_runtime.rs` depends on private `/internal/agent/*` routes for:

- auth context
- document search
- session defaults
- user profile context
- admin DB query
- effective AI config

If you change the request/response shape of those routes, update both repos together.

### Tool model

Branch-relevant tools:

- `knowledge_search`
- `web_search`
- `db_query`
- `done`

Important rules:

- `db_query` is admin-only
- `tool_context` is admin-only
- `/query` is session-backed and retrieval-first
- `/llm/chat` is the assistant-style route

### Memory model

Sage still uses the shared memory system:

- blocks
- messages
- passages
- summaries

This branch adds Enclave web session ownership on top through `web_sessions` and related records.

### Generic runtime modules

`main.rs`, `signal.rs`, and `marmot.rs` remain in the repo and still matter for upstream/generic Sage work. On this branch they are background context, not the primary Enclave runtime entrypoint.

## Database And Migrations

Postgres with pgvector backs all persistent Sage state.

When touching migrations:

- keep Diesel schema and migrations in sync
- verify fresh-database behavior
- verify upgraded-database behavior
- update docs if route/session/config behavior changes

## Security Considerations

- `INTERNAL_AGENT_TOKEN` is security-sensitive
- cookie-name and CSRF assumptions are security-sensitive
- gateway auth/cookie forwarding is part of the trust boundary
- Postgres data contains private AI session and memory state
- `db_query` and any shell-like tooling are security-sensitive

## Common Tasks

### Touching `enclave_web`

When changing `web_runtime.rs` or `enclave_web.rs`:

1. run `cargo check -p sage-core --bin enclave_web`
2. verify the route list still matches the gateway
3. verify docs stay aligned with actual route ownership

### Touching the Python contract

If you change a private `/internal/agent/*` call:

1. update the Sage caller
2. update the Enclave Python handler
3. update branch docs in both repos

### Touching prompts or tool semantics

If you materially change instruction blocks, tool behavior, or response shapes:

1. run tests and checks
2. update examples or eval assets if relevant
3. update docs that describe `/llm/chat` vs `/query`

## Workflow

- Prefer the existing compose and `just` flows over ad hoc startup scripts.
- Keep route ownership, config ownership, and docs aligned.
- Treat this branch as `enclave_web`-first unless you are explicitly doing upstream generic runtime work.
