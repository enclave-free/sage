# Sage Architecture For The Enclave Web Runtime Branch

This document describes the active `enclave_web` architecture used by `enclave.free-prototype` on `enclave-web-native-auth`.

It does not describe the older Signal-first or Letta-era design notes elsewhere in this repo.

## Runtime Shape

```text
frontend
  -> gateway (:8000)
      -> sage / enclave_web (:3000)
          -> Tinfoil proxy
          -> Postgres
          -> Enclave Python /internal/agent/*
      -> core-backend (:8000 internal)
          -> SQLite
          -> Qdrant
```

The main idea is:

- Sage owns public AI-route correctness
- Python remains the Enclave control plane
- the gateway keeps the public API stable without taking on application logic

## Entry Points

### `crates/sage-core/src/bin/enclave_web.rs`

Startup behavior:

1. load config from env
2. run embedded Diesel migrations
3. seed default AI config if the Sage config tables are empty
4. configure the model against Tinfoil with a low default temperature
5. build the Axum router from `web_runtime.rs`
6. listen on `ENCLAVE_WEB_PORT`

### `crates/sage-core/src/web_runtime.rs`

This file contains the branch-specific integration layer:

- public route definitions
- native Enclave bearer/cookie auth verification
- CSRF and origin validation
- CORS layer for Sage-owned routes
- `InternalAgentClient`
- session ownership checks
- AI config CRUD and prompt preview
- tool registration for `/llm/chat` and `/query`
- prompt assembly helpers

## Public Routes

| Route | Ownership | Notes |
| --- | --- | --- |
| `GET /health` | Sage service health | direct Sage runtime health, usually consumed internally |
| `POST /llm/chat` | Sage | stateless; optional server-side tools |
| `POST /query` | Sage | retrieval-first; stateful; memory-backed |
| `GET /query/session/{session_id}` | Sage | session inspection |
| `DELETE /query/session/{session_id}` | Sage | deletes session record |
| `GET /session-defaults` | Sage | local AI defaults plus Python document defaults |
| `POST /admin/tools/execute` | Sage | public admin route; execution delegated to Python |
| `/admin/ai-config/*` | Sage | public route family and storage both live in Sage |

## InternalAgentClient Contract

`InternalAgentClient` is the main coupling point between Sage and Enclave Python.

Active calls:

- `GET /internal/agent/users/{user_id}`
- `GET /internal/agent/admins/by-pubkey/{pubkey}`
- `GET /internal/agent/user-types/{user_type_id}`
- `GET /internal/agent/document-access`
- `GET /internal/agent/user-profile-context/{user_id}`
- `POST /internal/agent/document-search`
- `POST /internal/agent/admin-db-query`

Compatibility endpoints may still exist in Python, but they are no longer part of the primary Sage call graph:

- `POST /internal/agent/auth-context`
- `GET /internal/agent/session-defaults`
- `GET /internal/agent/ai-config/effective`

This is the real integration boundary. If request or response shapes change, both repos must change together.

## `/llm/chat` Flow

`/llm/chat` is the stateless route.

1. enforce CSRF for cookie-authenticated unsafe requests
2. verify auth natively in Sage
3. hydrate user/admin identity from Python if needed
4. load effective AI config and request temperature from Sage Postgres
5. register route-appropriate tools
6. create `SageAgent::new_without_memory(...)`
7. run the agent loop against Tinfoil
8. return the assistant message plus `tools_used`

Registered tools on this route:

- `web_search` when `web-search` is selected
- `db_query` when `db-query` is selected by an admin
- `done`

`tool_context` is admin-only and is intended for trusted client-side context injection such as decrypted DB output or admin config snapshots.

## `/query` Flow

`/query` is the stateful route.

1. enforce CSRF
2. verify auth natively in Sage
3. hydrate user/admin identity from Python if needed
4. load effective AI config from Sage Postgres
5. load or create a `web_session`
6. fetch document access and initial document context through Python
7. create/update memory blocks:
   - persona block from compiled Enclave prompt profile
   - human block from auth + profile context
8. persist the user turn
9. run the agent with optional memory enabled
10. persist the assistant turn
11. return `session_id`, `sources`, `context_used`, and answer

Registered tools on this route:

- `knowledge_search` always
- `web_search` when selected
- `db_query` when selected by an admin
- `done`

This route uses the shared Sage memory system plus Enclave-specific `web_sessions`.

## Persistence Model

Important tables in `schema.rs` on this branch:

- shared Sage memory: `messages`, `blocks`, `passages`, `summaries`
- Enclave web runtime: `web_sessions`, `external_identities`
- runtime AI config: `ai_config`, `ai_config_user_type_overrides`

Current reality:

- `web_sessions` and `external_identities` are actively used
- AI config CRUD is now Sage-backed
- query-session deletion still deletes the session record only, not the full memory graph

## Tool Gating And Security

Current protections in `web_runtime.rs`:

- native bearer and cookie auth verification
- cookie-origin CSRF checks for unsafe browser requests
- `tool_context` restricted to admins
- `db_query` restricted to admins
- session ownership enforced in `ensure_session_access`
- private contract protected by `INTERNAL_AGENT_TOKEN`

## Why `sage_agent.rs` Still Matters On This Branch

The web runtime depends on shared agent-core changes that let the same engine support both routes:

- custom instruction blocks
- optional memory
- no-memory mode
- configurable per-request temperature

Without those changes, Sage could not cleanly support stateless `/llm/chat` and stateful `/query` in one runtime.

## Temporary Architecture Choices

- Python still issues the auth tokens and cookies Sage verifies
- the internal `/internal/agent/*` contract is still the main cross-repo coupling point
- deployment/runtime config is still split between Python deployment config, Sage env, and gateway config
- legacy messenger runtime code remains in-repo as upstream background, not as the main Enclave path
