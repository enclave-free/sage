# Sage Architecture For The Enclave Web Runtime Branch

This document describes the active `enclave_web` architecture used by the current
`enclave.free-prototype` staging snapshot.

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
- current prepared-tool context logic plus the ADR-0023 target Tool Set expansion
  and model-driven Tool loop execution for Conversation routes
- prompt assembly helpers

## Public Routes

| Route | Ownership | Notes |
| --- | --- | --- |
| `GET /health` | Sage service health | direct Sage runtime health, usually consumed internally |
| `POST /llm/chat` | Sage | Conversation transport with model-driven Tool loop |
| `POST /query` | Sage | stateful Conversation API compatibility shape |
| `GET /query/session/{session_id}` | Sage | session inspection |
| `DELETE /query/session/{session_id}` | Sage | deletes session record |
| `GET /session-defaults` | Sage | local AI defaults plus Python document defaults |
| `POST /admin/tools/execute` | Sage | public admin route; execution delegated to Python |
| `/admin/ai-config/*` | Sage | public route family and storage both live in Sage |

### Conversation final-answer safety

`POST /llm/chat` plans and runs selected Tools before requesting a separate
plain final answer. Current-turn Tool results are de-duplicated, limited to
4,000 characters each and 12,000 characters total, with the newest results
preferred when the budget is full. The planner's `replan_after_results` field
is an optional hint; omitting it means no requested replan.

The final-answer stream briefly holds ambiguous planning/search openings and
structured Tool-like output. Repeated process narration, Tool intent, provider
token-limit termination, and unsupported finish reasons fail the answer rather
than being persisted as success. A quarantined Tool, repetition, or token-limit
failure may retry once only when no answer text has reached the client.

One narrow deterministic terminal fallback applies after that retry is
exhausted: when the turn executed exactly one successful Curated Resources
inventory lookup and exposed no answer text, Sage may return the Tool adapter's
separately marked user-safe inventory rendering. Internal Tool output is never
used for this fallback. Contact lookups, multiple-Tool turns, partial answer
streams, and ordinary provider or transport failures retain the fail-closed
behavior above.

Conversation traces time total Tool planning, retrieval, Resource Directory
lookup, Tool execution, retry delay, final-answer generation, and total turn
duration separately. Final-answer response-header and first-event waits are
explicit provider-wait proxies: network transit, provider queueing, and model
startup may all contribute. The current typed Tool-planning provider contract
does not expose cluster-scheduling or inference-only timings, so Sage emits
those two phases as `unavailable` instead of fabricating durations or treating
the combined planning duration as either metric. This preserves an honest
correlation between slow planning attempts and omitted/rejected Tool selections
while making the provider instrumentation gap visible.

## InternalAgentClient Contract

`InternalAgentClient` is the main coupling point between Sage and Enclave Python.

Active calls:

- `GET /internal/agent/users/{user_id}`
- `GET /internal/agent/admins/by-pubkey/{pubkey}`
- `GET /internal/agent/user-types/{user_type_id}`
- `GET /internal/agent/document-access`
- `GET /internal/agent/user-profile-context/{user_id}`
- `POST /internal/agent/document-search`
- `POST /internal/agent/resources/search` — accepts optional `query`, `limit`, and `offset`; returns normalized query plus `total_count`, `returned_count`, `limit`, `offset`, `has_more`, and `next_offset` metadata for ready Curated Resources. Query relevance (exact normalized ID/name/contact, then partial name/contact, then description) precedes existing scope, verification, language, display-order, and name ranking.
- Explicit contact, Curated Resource inventory, and inventory-continuation requests are validated at the model-planning boundary; ordinary questions about how organizations or the directory work are not inventory requests. Each such user turn requires exactly one successful `find_resources` execution: an initial lookup runs on the current turn, while a continuation can run only on a later user turn using the immediately preceding open page's structured query, region, help type, language, and `next_offset`. Sage does not fetch multiple pages within one turn. Rejected model selections are never executed and do not count toward that one-success limit; Sage retries a plan that omits the required call, changes an explicit inventory/contact query, invents a context-only contact query outside the structured Curated Resources results (with legacy prose grounding only when structured state is unavailable), uses a stale positive offset for a fresh lookup, or changes an exact continuation filter or offset. Redundant `find_resources` calls after a successful turn lookup are removed before execution; selection traces retain both the raw model choice and sanitized executable choice. Rejected selections include their privacy-safe validation reason in structured trace metadata. A missing or zero continuation cursor and bounded planning exhaustion fail closed without an ungrounded answer; an intervening assistant turn expires an older cursor. Conservative pagination reconciles backend counts and cursor state; when more results are reported without a safe cursor, Sage says so instead of inventing one. The Tool’s structured `has_more` result—not rendered prompt text—controls the incomplete-page final-answer guard. For incomplete pages, final-answer text is held until completion and retried before exposure if it falsely claims all, every, or a complete list, while explicit limitations such as “not a complete list” remain valid. The validator never constructs or executes a Tool call itself.
- `POST /internal/agent/admin-db-query`

ADR-0023 target calls:

- `POST /internal/agent/admin-config/*`

Compatibility endpoints may still exist in Python, but they are no longer part of the primary Sage call graph:

- `POST /internal/agent/auth-context`
- `GET /internal/agent/session-defaults`
- `GET /internal/agent/ai-config/effective`

This is the real integration boundary. If request or response shapes change, both repos must change together.

## Conversation Flow Target

Sage owns the model-driven Tool loop for Conversation routes.

1. enforce CSRF for cookie-authenticated unsafe requests
2. verify auth natively in Sage
3. hydrate user/admin identity from Python if needed
4. load effective AI config and request temperature from Sage Postgres
5. expand enabled Tool Sets into concrete Tool contracts
6. run the model-driven Tool loop against the configured Model Provider
7. execute authorized Tool calls, inject results, and continue until answer or Executable Change Set
8. return the assistant message plus Activity/Trace metadata and Tool summaries

Tool Sets:

- `knowledge-search` exposes `knowledge_search` with Document Access and selected Document constraints
- `web-search` exposes `web_search`
- `admin-config` exposes admin-only configuration read/proposal Tools
- `db-query` exposes admin-only read-only database inspection Tools
- `done` or the equivalent final-answer signal completes the loop

The frontend should not pre-run Tools or inject admin configuration snapshots through `tool_context`.

## Stateful Conversation Compatibility

`/query` remains a stateful public API shape while the product converges on one Conversation runtime.

1. enforce CSRF
2. verify auth natively in Sage
3. hydrate user/admin identity from Python if needed
4. load effective AI config from Sage Postgres
5. load or create a `web_session`
6. load available Document metadata and Tool constraints as needed
7. create/update memory blocks:
   - persona block from compiled Enclave prompt profile
   - human block from auth + profile context
8. persist the user turn
9. run the same model-driven Tool loop with memory enabled
10. persist the assistant turn
11. return `session_id`, `sources`, `context_used`, and answer

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

Without those changes, Sage could not cleanly support stateless and stateful Conversation API shapes in one runtime.

## Temporary Architecture Choices

- Python still issues the auth tokens and cookies Sage verifies
- the internal `/internal/agent/*` contract is still the main cross-repo coupling point
- deployment/runtime config is still split between Python deployment config, Sage env, and gateway config
- legacy messenger runtime code remains in-repo as upstream background, not as the main Enclave path
