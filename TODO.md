# Sage Branch Checkpoint

> Last updated: 2026-04-13

This file tracks the current state of `enclave-web-native-auth`, the branch that moves Enclave web mode toward a dumb public gateway and native Sage auth.

## Done

- Added `enclave_web` as a dedicated Enclave web runtime binary
- Added `web_runtime.rs` with public AI routes and Enclave-specific handlers
- Cut public `/llm/chat` and `/query` traffic over to Sage through the gateway
- Replaced Python `auth-context` dependency with native Sage bearer/cookie verification
- Added private Python support-route integration for retrieval, user profile context, user/admin lookups, user-type lookups, and admin DB access
- Added Sage-owned Postgres query session persistence
- Added Sage-owned AI config storage and user-type overrides
- Added Enclave web tables and schema wiring
- Adapted `SageAgent` to support custom instructions and optional/stateless memory modes for web flows
- Verified `cargo check -p sage-core --bin enclave_web`

## Intentionally Temporary On This Branch

- Python still issues the auth tokens and cookies Sage verifies
- deployment/runtime config is still split between Python deployment config, Sage env, and gateway config
- legacy Python AI route implementations still exist in-repo even though the gateway bypasses them
- query-session delete still removes the session record, not the full memory graph

## What To Tighten If Productizing

- formalize and version the `/internal/agent/*` contract
- decide whether query-session delete should remain a session-record delete or become a full Sage memory purge contract
- reduce duplicated shared config across Python deployment config and Sage env
- add stronger end-to-end tests around route ownership, auth forwarding, and session ownership
- add browser-level automated tests for cookie auth + CSRF on Sage-owned routes

## High-Value Checks

```bash
cargo check -p sage-core --bin enclave_web
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
just smoke-tinfoil
```

## Current Mental Model

On this branch:

- Sage is the AI runtime
- Python is the control plane
- gateway keeps the public API stable without owning app behavior

If a change does not fit that model cleanly, it probably needs either a boundary clarification or a productization decision first.
