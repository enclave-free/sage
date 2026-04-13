# Architecture Decisions For The Enclave Web Runtime Branch

This file records the active decisions for the Enclave web runtime branch.

The branch documented here is `enclave-web-native-auth`, which builds on the original hard-cut branch by making the public gateway boring and moving route correctness into Sage.

## ADR-001: Use Direct Tinfoil Through A Local Verified Proxy

**Status**: Accepted

**Context**

The Enclave web runtime needs a stable OpenAI-compatible backend for chat, embeddings, and vision while keeping the privacy posture of confidential compute.

**Decision**

Use Tinfoil through a local verified proxy, with Sage calling the proxy directly.

**Consequences**

- Sage owns chat, embeddings, and vision calls against `TINFOIL_API_URL`
- model transport stays simple and OpenAI-compatible
- the branch depends on `TINFOIL_*` runtime env

## ADR-002: Add An Enclave-Specific Web Runtime Instead Of Reusing The Messenger Runtime

**Status**: Accepted

**Context**

The upstream Sage runtime is oriented around messenger-style interaction. Enclave needs a web/backend runtime with HTTP routes, auth re-validation, session ownership, and document-grounded query flows.

**Decision**

Add a dedicated `enclave_web` binary and `web_runtime.rs` rather than trying to bend the messenger entrypoint into the product API shape.

**Consequences**

- branch-specific entrypoint: `crates/sage-core/src/bin/enclave_web.rs`
- branch-specific route layer: `crates/sage-core/src/web_runtime.rs`
- clearer separation between upstream generic runtime and Enclave-specific integration logic

## ADR-003: Sage Owns The Public AI Routes

**Status**: Accepted

**Context**

The prototype aims to move answer generation and AI orchestration out of the Python backend while keeping the same public API origin.

**Decision**

Sage owns:

- `/llm/chat`
- `/query`
- `/query/session/*`
- `/session-defaults`
- `/admin/tools/execute`
- `/admin/ai-config/*`

with the gateway routing those paths to Sage.

**Consequences**

- answer generation and tool orchestration live in Sage
- the gateway remains the compatibility layer for the public API surface
- legacy Python implementations of some AI routes can remain in-repo without serving public traffic

## ADR-004: Python Remains The Control Plane

**Status**: Accepted

**Context**

Enclave already has product logic in Python for auth, admin data, ingest, document access, and AI config assembly. Rebuilding all of that in Sage would expand the cutover far beyond prototype scope.

**Decision**

Keep Python as the control plane and have Sage call it over private support endpoints.

**Consequences**

- Python remains the source of truth for auth and approval
- Python remains the source of truth for document access and effective AI config
- Sage depends on a stable private contract instead of duplicating control-plane logic

## ADR-005: Use A Private `/internal/agent/*` Contract Between Sage And Python

**Status**: Accepted

**Context**

Sage needs a narrow, explicit way to ask Enclave for auth context, retrieval results, user profile data, and admin DB access.

**Decision**

Define the Sage <-> Python boundary through private `/internal/agent/*` endpoints protected by `INTERNAL_AGENT_TOKEN`.

**Consequences**

- Sage can stay focused on AI orchestration
- the trust boundary is explicit in code and docs
- the private contract becomes a major integration surface that must remain versioned and consistent

## ADR-006: Sage Owns Query Sessions And Memory In Postgres

**Status**: Accepted

**Context**

The prototype needs durable query continuity and Sage-native memory behavior for RAG turns.

**Decision**

Persist query sessions and runtime memory in Sage Postgres instead of keeping session continuity in Python.

**Consequences**

- `web_sessions` and Sage memory tables become the source of truth for `/query` continuity
- sessions survive Sage restarts if Postgres persists
- session ownership checks happen inside Sage after auth re-validation

## ADR-007: Keep Admin AI Config CRUD Publicly Served By Sage But Stored/Mutated Via Python For Now

**Status**: Superseded on `enclave-web-native-auth`

**Context**

The public AI config surface needs to appear Sage-owned because it is part of the AI runtime API, but Python already holds the current storage and mutation logic.

**Decision**

Serve `/admin/ai-config/*` through Sage while proxying CRUD operations to Python on the original hard-cut branch.

**Consequences**

- public route ownership stayed consistent during the initial cutover
- storage ownership remained transitional during the first prototype pass
- this decision was later replaced by Sage-backed AI config storage on `enclave-web-native-auth`
