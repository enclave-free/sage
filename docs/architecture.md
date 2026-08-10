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
- current prepared-tool context logic plus Tool Set expansion and the bounded
  native Tool loop for Conversation routes
- prompt assembly helpers

## Public Routes

| Route | Ownership | Notes |
| --- | --- | --- |
| `GET /health` | Sage service health | direct Sage runtime health, usually consumed internally |
| `POST /llm/chat` | Sage | Conversation transport with at most six native Tool batches |
| `POST /query` | Sage | stateful Conversation API compatibility shape |
| `GET /query/session/{session_id}` | Sage | session inspection |
| `DELETE /query/session/{session_id}` | Sage | deletes session record |
| `GET /session-defaults` | Sage | local AI defaults plus Python document defaults |
| `POST /admin/tools/execute` | Sage | public admin route; execution delegated to Python |
| `/admin/ai-config/*` | Sage | public route family and storage both live in Sage |

### Native Conversation trust boundary

`POST /llm/chat` sends enabled, authorized native Tool definitions to the one
configured Conversation model. The model either answers directly or selects
a bounded Tool batch. After a Tool batch, correlated structured Tool results
and the same enabled Tool definitions return to the same model. The model may
continue selecting batches within a six-batch safety ceiling; Sage rejects a
seventh selected batch before execution, returns correlated
`tool_budget_exhausted` failures for those unexecuted calls, and makes one final
same-model request with Tools disabled so the model can answer from the accumulated
evidence. Each executed batch's Tool results are limited to 4,000 characters per
result and 12,000 characters total.

Native assistant content streams in provider order without semantic scanning,
rewriting, or deterministic answer fallback. When a model request has Tools
available, Sage buffers that request's visible content until the provider has
finished Tool selection: content from a Tool-selection turn remains private,
while a direct answer is released intact. Requests with no Tools available keep
incremental answer streaming. Provider reasoning is discarded. Before the first
provider event, structural protocol failures and eligible connection, timeout,
429, or 502/503/504 failures share a bounded recovery budget of three identical
attempts against the identical model. Complete provider silence for 30 seconds is
also eligible within that same budget. Eligible retries wait through a short
capped exponential delay with jitter; valid `Retry-After` guidance is honored up
to the same internal cap. Once any provider stream event has arrived, a later
failure does not retry. A final-request retry reuses existing Tool-result messages
and cannot execute Tools again. No other Conversation model is substituted after
failure.

Conversation traces record each native loop step, provider first-event wait,
Retrieval or Resource lookup, Tool execution, content-free retry delay, retry
outcome, and total-turn timing where those stages are measurable. Provider
first-event wait is a combined proxy:
network transit, provider queueing, and model startup may all contribute. Sage
does not emit fabricated `cluster_scheduling` or `inference_only` phases when
the provider does not supply those measurements. A model retry keeps one stable
per-step Trace identity so scheduled state is replaced by recovered or exhausted
state instead of leaving a completed retry visibly running.

## InternalAgentClient Contract

`InternalAgentClient` is the main coupling point between Sage and Enclave Python.

Active calls:

- `GET /internal/agent/users/{user_id}`
- `GET /internal/agent/admins/by-pubkey/{pubkey}`
- `GET /internal/agent/user-types/{user_type_id}`
- `GET /internal/agent/document-access`
- `GET /internal/agent/user-profile-context/{user_id}`
- `POST /internal/agent/document-search`
- `POST /internal/agent/resources/search` — accepts optional generic `query`, `kind`, `tags`, `region`, `language`, and pagination fields. It returns relevance-ranked generic Resources plus `total_count`, `returned_count`, `limit`, `offset`, `has_more`, and `next_offset` metadata. Exact normalized IDs, names, and pointers rank ahead of partial names, pointers, and descriptions.
- `find_resources` exposes a smaller model-facing contract: optional `exact_resource`, `region`, `language`, and continuation `offset`. Sage maps `exact_resource` to the private endpoint's `query` field and leaves private `kind` and `tags` unset, removing guess-prone facets from the model without adding a classifier or forced lookup. The Conversation model decides whether to call the Tool and may use returned pagination metadata on a later turn. Sage validates arguments and backend page consistency, but does not force a lookup, rewrite the selected batch, or quarantine final prose for completeness claims.
- `POST /internal/agent/admin-db-query`

ADR-0023 target calls:

- `POST /internal/agent/admin-config/*`

Compatibility endpoints may still exist in Python, but they are no longer part of the primary Sage call graph:

- `POST /internal/agent/auth-context`
- `GET /internal/agent/session-defaults`
- `GET /internal/agent/ai-config/effective`

This is the real integration boundary. If request or response shapes change, both repos must change together.

## Conversation Flow Target

Sage owns the bounded native Tool loop for Conversation routes.

1. enforce CSRF for cookie-authenticated unsafe requests
2. verify auth natively in Sage
3. hydrate user/admin identity from Python if needed
4. load effective AI config and request temperature from Sage Postgres
5. expand enabled Tool Sets into concrete Tool contracts
6. ask the configured Conversation model to answer directly or select a bounded
   batch of authorized Tool calls
7. if Tools were selected, execute that batch and return its correlated,
   structured results plus the same enabled Tool definitions to the same model
8. allow model-selected continuation within the six-batch safety ceiling;
   reject a seventh selected batch before execution, correlate bounded failures,
   and request one final same-model answer with Tools disabled
9. return the assistant message plus Activity/Trace metadata and Tool summaries;
   separate prose from distinct model requests when their boundary contains no
   whitespace

Tool Sets:

- `knowledge-search` exposes `knowledge_search` with Document Access and selected Document constraints
- `curated-resources` exposes `find_resources` for relevance-ranked, operator-curated referrals and exact pointers
- `web-search` exposes `web_search`
- `admin-config` exposes admin-only configuration read/proposal Tools
- `db-query` exposes admin-only read-only database inspection Tools

The frontend should not pre-run Tools or inject admin configuration snapshots through `tool_context`.

The shared Conversation instruction leaves judgment with the model while
requiring it to preserve explicit consent and personal autonomy without indirect
or observer-record workarounds, default the complete User turn to one current
action in no more than three short paragraphs without headings or lists, emit a
Tool call without preparatory prose, and stop selecting Tools once the evidence
is enough. Broad Curated Resource discovery sends only region and language when
applicable. The optional model-facing `exact_resource` field is reserved for a
Resource name or pointer the User is asking for directly, not a place or subject
mentioned as context. Private directory query, kind, and tag facets remain
outside the native Conversation Tool schema.
The configurable Agent Settings profile is assembled before these immutable
runtime requirements in the same system instruction, so personalization remains
available without becoming a later override of the platform contract.
Sage does not implement those behaviors with an intent classifier, semantic
answer rewrite, or hard output truncation.

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
9. run the same bounded native Tool loop with memory enabled
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
