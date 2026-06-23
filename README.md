# Sage Enclave Web Runtime Branch

This repo still contains the upstream generic Sage runtime, but on the Enclave prototype branch the primary product entrypoint is the `enclave_web` binary, not the messenger runtime.

## Branch Note

- `master`: generic Sage messenger runtime
- `enclave-web-prototype`: original Enclave hard-cut branch
- `enclave-web-native-auth`: current native-auth / dumb-gateway branch in this checkout

If you are working on the prototype in `enclave.free-prototype`, read this branch as `enclave_web`-first.

## What This Branch Does

- adds `crates/sage-core/src/bin/enclave_web.rs`
- adds `crates/sage-core/src/web_runtime.rs`
- serves Enclave-facing AI routes over Axum
- persists query sessions and memory in Postgres
- calls Tinfoil directly for chat and embeddings
- verifies Enclave bearer and cookie auth natively
- serves CORS and enforces CSRF for Sage-owned routes
- stores runtime AI config in Postgres
- calls Enclave Python over private `/internal/agent/*` endpoints for retrieval and control-plane context

## Public Routes Owned On This Branch

| Route family | Behavior |
| --- | --- |
| `/llm/chat` | stateless assistant-style route |
| `/query` | retrieval-first, session-backed route |
| `/query/session/*` | session inspection and delete |
| `/session-defaults` | local AI defaults plus Enclave document defaults |
| `/admin/tools/execute` | public Sage entry, execution delegated to Python |
| `/admin/ai-config/*` | public Sage route family and Sage-backed storage |

## What Sage Still Delegates To Enclave Python

Sage does not currently own:

- auth issuance
- Enclave document retrieval and access filtering
- user/admin record lookup after token verification
- user type lookup
- user profile context lookup
- admin SQLite query safety logic

Those come through the private `/internal/agent/*` contract defined in `web_runtime.rs`.

## What Sage Owns Directly

- public AI route handling
- Tinfoil chat and embedding calls
- Postgres-backed memory tables
- `web_sessions`
- `ai_config`
- `ai_config_user_type_overrides`
- session ownership checks after auth revalidation
- server-side `web_search`
- internal `knowledge_search`

## Key Files

- `crates/sage-core/src/bin/enclave_web.rs`: web runtime startup, migrations, and Axum server boot
- `crates/sage-core/src/web_runtime.rs`: route layer, Enclave contract client, session logic, tool gating
- `crates/sage-core/src/sage_agent.rs`: shared agent core, including custom-instruction and optional-memory support used by the web runtime
- `crates/sage-core/src/schema.rs`: memory tables plus Enclave web tables
- `Dockerfile`: ships both `/app/sage` and `/app/enclave_web`

## Running And Verifying

Recommended full-system path:

- run this repo through the parent `enclave.free-prototype` compose stack

Useful local checks:

```bash
cargo check -p sage-core --bin enclave_web
cargo run -p sage-core --bin enclave_web
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

On macOS, local Diesel/Postgres test binaries link against `libpq`. If `cargo test -p sage-core chat_stream` fails with `ld: library 'pq' not found`, install Homebrew `libpq` and run the helper:

```bash
brew install libpq
./scripts/test_chat_stream.sh
# or, when just is installed:
just test-chat-stream
```

The helper exports `LIBRARY_PATH="$(brew --prefix libpq)/lib:${LIBRARY_PATH:-}"` before running `cargo test -p sage-core chat_stream` on macOS.

Useful env for the web runtime path:

```bash
DATABASE_URL=postgres://sage:sage@localhost:5432/sage
TINFOIL_API_URL=http://localhost:8089/v1
TINFOIL_API_KEY=your-key
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

## Background And Historical Docs

Messenger-specific code still exists in this repo, and `main.rs`, `signal.rs`, and `marmot.rs` still matter for upstream Sage work. They are not the primary Enclave product path on this branch.

For the current branch story, start with:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/decisions.md`](docs/decisions.md)
- [`AGENTS.md`](AGENTS.md)
- [`TODO.md`](TODO.md)

Older design docs such as `docs/brainstorm.md`, `docs/roadmap.md`, and `docs/SAGE_V2_DESIGN.md` are now historical background only.
