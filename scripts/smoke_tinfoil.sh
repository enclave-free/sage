#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() {
    printf '[smoke] %s\n' "$*"
}

fail() {
    printf '[smoke] ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

detect_engine() {
    if command -v docker >/dev/null 2>&1; then
        printf 'docker\n'
        return
    fi

    if command -v podman >/dev/null 2>&1; then
        printf 'podman\n'
        return
    fi

    fail "Neither docker nor podman is available"
}

ENGINE="${CONTAINER_ENGINE:-$(detect_engine)}"
require_command "$ENGINE"

SMOKE_ID="${SMOKE_ID:-$(date +%Y%m%d%H%M%S)-$$}"
NETWORK_NAME="sage-smoke-${SMOKE_ID}"
POSTGRES_CONTAINER="sage-smoke-postgres-${SMOKE_ID}"
PROXY_CONTAINER="sage-smoke-tinfoil-proxy-${SMOKE_ID}"
POSTGRES_VOLUME="sage-smoke-pgdata-${SMOKE_ID}"
CARGO_HOME_VOLUME="sage-smoke-cargo-${SMOKE_ID}"
RUNNER_IMAGE="${RUNNER_IMAGE:-sage:smoke}"
KEEP_IMAGE="${KEEP_IMAGE:-0}"
KEEP_STACK_ON_FAILURE="${KEEP_STACK_ON_FAILURE:-0}"

export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-sage-smoke}"
export TINFOIL_MODEL="${TINFOIL_MODEL:-kimi-k2-5}"
export TINFOIL_EMBEDDING_MODEL="${TINFOIL_EMBEDDING_MODEL:-nomic-embed-text}"
export TINFOIL_VISION_MODEL="${TINFOIL_VISION_MODEL:-kimi-k2-5}"
export TINFOIL_ROUTER_HOST="${TINFOIL_ROUTER_HOST:-inference.tinfoil.sh}"
export TINFOIL_ROUTER_REPO="${TINFOIL_ROUTER_REPO:-tinfoilsh/confidential-model-router}"
export TINFOIL_PROXY_PORT="${TINFOIL_PROXY_PORT:-8089}"

if [[ -z "${TINFOIL_API_KEY:-}" ]]; then
    read -rsp "TINFOIL_API_KEY: " TINFOIL_API_KEY
    echo
    export TINFOIL_API_KEY
fi

if [[ -z "${TINFOIL_API_KEY:-}" ]]; then
    fail "TINFOIL_API_KEY is required"
fi

run_engine() {
    "$ENGINE" "$@"
}

run_compose_config_check() {
    if run_engine compose version >/dev/null 2>&1; then
        log "Validating docker-compose.yml"
        run_engine compose -f "${ROOT_DIR}/docker-compose.yml" config >/dev/null
        return
    fi

    log "Skipping docker compose config validation because '${ENGINE} compose' is unavailable"
}

cleanup() {
    local status=$?

    if [[ $status -ne 0 ]]; then
        log "Smoke test failed; collecting recent logs"
        run_engine logs --tail=200 "$PROXY_CONTAINER" 2>/dev/null || true
        run_engine logs --tail=200 "$POSTGRES_CONTAINER" 2>/dev/null || true
    fi

    if [[ $status -eq 0 || "${KEEP_STACK_ON_FAILURE}" != "1" ]]; then
        run_engine rm -f "$PROXY_CONTAINER" "$POSTGRES_CONTAINER" >/dev/null 2>&1 || true
        run_engine network rm "$NETWORK_NAME" >/dev/null 2>&1 || true
        run_engine volume rm "$POSTGRES_VOLUME" >/dev/null 2>&1 || true
        run_engine volume rm "$CARGO_HOME_VOLUME" >/dev/null 2>&1 || true
    else
        log "Keeping failed smoke stack for inspection"
    fi

    if [[ "${KEEP_IMAGE}" != "1" ]]; then
        run_engine image rm "$RUNNER_IMAGE" >/dev/null 2>&1 || true
    fi

    unset TINFOIL_API_KEY || true
    exit $status
}

trap cleanup EXIT

wait_for_postgres() {
    local attempts=0
    until run_engine exec "$POSTGRES_CONTAINER" pg_isready -U sage -d sage >/dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [[ $attempts -ge 30 ]]; then
            fail "Postgres did not become ready in time"
        fi
        sleep 2
    done
}

apply_migrations() {
    log "Applying database migrations"

    run_engine exec -i "$POSTGRES_CONTAINER" psql -v ON_ERROR_STOP=1 -U sage -d sage \
        -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"

    while IFS= read -r migration; do
        log "Applying $(basename "$(dirname "$migration")")"
        run_engine exec -i "$POSTGRES_CONTAINER" psql -v ON_ERROR_STOP=1 -U sage -d sage <"$migration"
    done < <(find "${ROOT_DIR}/crates/sage-core/migrations" -mindepth 2 -maxdepth 2 -name up.sql | sort)
}

run_in_runner() {
    run_engine run --rm \
        --network "$NETWORK_NAME" \
        --user root \
        -e CARGO_HOME=/cargo-home \
        -e DATABASE_URL="postgres://sage:sage@postgres:5432/sage" \
        -e TINFOIL_API_URL="http://tinfoil-proxy:${TINFOIL_PROXY_PORT}/v1" \
        -e TINFOIL_API_KEY \
        -e TINFOIL_MODEL \
        -e TINFOIL_EMBEDDING_MODEL \
        -e TINFOIL_VISION_MODEL \
        -v "${CARGO_HOME_VOLUME}:/cargo-home" \
        -v "${ROOT_DIR}:/repo" \
        -w /repo \
        "$RUNNER_IMAGE" "$@"
}

run_proxy_checks() {
    log "Running chat, embeddings, and vision smoke checks"

    run_in_runner python3 - <<'PY'
import base64
import os
import subprocess
import time

import requests

api_url = os.environ["TINFOIL_API_URL"]
api_key = os.environ["TINFOIL_API_KEY"]
chat_model = os.environ["TINFOIL_MODEL"]
embedding_model = os.environ["TINFOIL_EMBEDDING_MODEL"]
vision_model = os.environ["TINFOIL_VISION_MODEL"]
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}


def post(path: str, payload: dict):
    return requests.post(f"{api_url}{path}", headers=headers, json=payload, timeout=120)


chat_payload = {
    "model": chat_model,
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "max_tokens": 8,
}

last_error = None
for _ in range(30):
    try:
        response = post("/chat/completions", chat_payload)
        if response.status_code == 200:
            break
        last_error = f"chat status {response.status_code}: {response.text}"
    except Exception as exc:  # noqa: BLE001
        last_error = str(exc)
    time.sleep(2)
else:
    raise SystemExit(f"FAIL chat readiness: {last_error}")

chat_json = response.json()
chat_content = chat_json["choices"][0]["message"]["content"]
if not chat_content or "ok" not in chat_content.lower():
    raise SystemExit(f"FAIL chat content: {chat_content!r}")
print("PASS chat completion")

embedding_response = post(
    "/embeddings",
    {
        "model": embedding_model,
        "input": "hello from sage smoke test",
        "encoding_format": "float",
    },
)
if embedding_response.status_code != 200:
    raise SystemExit(
        f"FAIL embeddings status {embedding_response.status_code}: {embedding_response.text}"
    )

embedding = embedding_response.json()["data"][0]["embedding"]
if len(embedding) != 768:
    raise SystemExit(f"FAIL embeddings dimension: {len(embedding)}")
if not all(isinstance(value, (int, float)) for value in embedding):
    raise SystemExit("FAIL embeddings contain non-numeric values")
print("PASS embeddings shape")

subprocess.run(["mkdir", "-p", "/tmp/sage-smoke"], check=True)
subprocess.run(
    ["convert", "-size", "64x64", "xc:red", "/tmp/sage-smoke/red.png"],
    check=True,
)
with open("/tmp/sage-smoke/red.png", "rb") as handle:
    image_b64 = base64.b64encode(handle.read()).decode("ascii")

vision_response = post(
    "/chat/completions",
    {
        "model": vision_model,
        "messages": [
            {"role": "system", "content": "Describe the user image in one sentence."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in one sentence.",
                    },
                ],
            },
        ],
        "max_tokens": 64,
    },
)
if vision_response.status_code != 200:
    raise SystemExit(
        f"FAIL vision status {vision_response.status_code}: {vision_response.text}"
    )

vision_content = vision_response.json()["choices"][0]["message"]["content"]
vision_lower = vision_content.lower()
if not vision_content or not any(token in vision_lower for token in ("red", "square", "solid", "color")):
    raise SystemExit(f"FAIL vision content: {vision_content!r}")
print("PASS vision completion")

invalid_model_response = post(
    "/chat/completions",
    {
        "model": "definitely-not-a-real-model",
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 8,
    },
)
if 200 <= invalid_model_response.status_code < 300:
    raise SystemExit("FAIL invalid model unexpectedly succeeded")
print("PASS invalid-model preflight")
PY
}

run_memory_harness() {
    log "Running recall and archival pgvector harness"

    local harness_dir
    harness_dir="$(mktemp -d)"
    cp "${ROOT_DIR}/Cargo.lock" "${harness_dir}/Cargo.lock"

    cat >"${harness_dir}/Cargo.toml" <<'EOF'
[package]
name = "sage-smoke-harness"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
tokio = { version = "1", features = ["full"] }
uuid = "1"
sage-core = { path = "/repo/crates/sage-core" }
EOF

    mkdir -p "${harness_dir}/src"
    cat >"${harness_dir}/src/main.rs" <<'EOF'
use anyhow::{ensure, Context, Result};
use sage_core::memory::{EmbeddingService, MemoryDb};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    let database_url = std::env::var("DATABASE_URL").context("DATABASE_URL not set")?;
    let api_url = std::env::var("TINFOIL_API_URL").context("TINFOIL_API_URL not set")?;
    let api_key = std::env::var("TINFOIL_API_KEY").context("TINFOIL_API_KEY not set")?;
    let embedding_model =
        std::env::var("TINFOIL_EMBEDDING_MODEL").context("TINFOIL_EMBEDDING_MODEL not set")?;

    let agent_id = Uuid::new_v4();
    let agent_id_str = agent_id.to_string();
    let db = MemoryDb::new(&database_url)?;
    db.agents().ensure_agent_exists(agent_id, "smoke-agent")?;
    let embedding = EmbeddingService::new(&api_url, &api_key, &embedding_model);

    let message_content = "The blue notebook is in the second desk drawer.";
    let message_id = db.messages().insert_message(
        agent_id,
        "smoke-user",
        "user",
        message_content,
        None,
        None,
        None,
        None,
    )?;

    let query_embedding = embedding.embed("Where is the blue notebook?").await?;
    let pending_results = db.messages().search_by_embedding(agent_id, &query_embedding, 5)?;
    ensure!(
        pending_results.iter().all(|result| result.message.id != message_id),
        "pending message embedding was unexpectedly searchable"
    );
    println!("PASS recall pending-embedding miss");

    let message_embedding = embedding.embed(message_content).await?;
    db.messages().update_embedding(message_id, &message_embedding)?;
    let filled_results = db.messages().search_by_embedding(agent_id, &query_embedding, 5)?;
    ensure!(
        filled_results.iter().any(|result| result.message.id == message_id),
        "filled message embedding did not become searchable"
    );
    println!("PASS recall post-fill hit");

    let passage_content = "Server maintenance window starts Friday at 22:00 UTC.";
    let passage_embedding = embedding.embed(passage_content).await?;
    let tags = vec!["ops".to_string(), "maintenance".to_string()];
    db.passages()
        .insert_passage_with_embedding(&agent_id_str, passage_content, &passage_embedding, &tags)?;

    let archival_query = embedding.embed("When is the maintenance window?").await?;
    let maintenance_filter = vec!["maintenance".to_string()];
    let archival_results = db.passages().search_passages_by_embedding(
        &agent_id_str,
        &archival_query,
        5,
        Some(&maintenance_filter),
    )?;
    ensure!(!archival_results.is_empty(), "archival search returned no results");
    ensure!(
        archival_results
            .iter()
            .any(|(row, _distance)| row.content == passage_content),
        "archival search did not return the inserted passage"
    );
    println!("PASS archival tagged hit");

    Ok(())
}
EOF

    run_engine run --rm \
        --network "$NETWORK_NAME" \
        --user root \
        -e CARGO_HOME=/cargo-home \
        -e DATABASE_URL="postgres://sage:sage@postgres:5432/sage" \
        -e TINFOIL_API_URL="http://tinfoil-proxy:${TINFOIL_PROXY_PORT}/v1" \
        -e TINFOIL_API_KEY \
        -e TINFOIL_EMBEDDING_MODEL \
        -v "${CARGO_HOME_VOLUME}:/cargo-home" \
        -v "${ROOT_DIR}:/repo" \
        -v "${harness_dir}:/tmp/sage-smoke-harness" \
        -w /tmp/sage-smoke-harness \
        "$RUNNER_IMAGE" \
        cargo run --quiet

    rm -rf "${harness_dir}"
}

log "Using container engine: ${ENGINE}"
run_compose_config_check

log "Creating isolated smoke network and postgres volume"
run_engine network create "$NETWORK_NAME" >/dev/null
run_engine volume create "$POSTGRES_VOLUME" >/dev/null
run_engine volume create "$CARGO_HOME_VOLUME" >/dev/null

log "Starting postgres container"
run_engine run -d \
    --name "$POSTGRES_CONTAINER" \
    --network "$NETWORK_NAME" \
    --network-alias postgres \
    -e POSTGRES_USER=sage \
    -e POSTGRES_PASSWORD=sage \
    -e POSTGRES_DB=sage \
    -v "${POSTGRES_VOLUME}:/var/lib/postgresql/data" \
    pgvector/pgvector:pg17 >/dev/null

log "Starting Tinfoil proxy container"
run_engine run -d \
    --name "$PROXY_CONTAINER" \
    --network "$NETWORK_NAME" \
    --network-alias tinfoil-proxy \
    -e TINFOIL_API_KEY \
    ghcr.io/tinfoilsh/tinfoil-cli:latest \
    proxy -e "$TINFOIL_ROUTER_HOST" -r "$TINFOIL_ROUTER_REPO" -p "$TINFOIL_PROXY_PORT" -b 0.0.0.0 >/dev/null

log "Waiting for postgres readiness"
wait_for_postgres
apply_migrations

log "Building smoke runner image"
run_engine build --target smoke-runner -t "$RUNNER_IMAGE" "$ROOT_DIR"

log "Running containerized workspace checks"
run_in_runner cargo check --workspace
run_in_runner cargo test --workspace
run_in_runner cargo clippy --workspace --all-targets --all-features -- -D warnings

run_proxy_checks
run_memory_harness

log "Smoke test passed"
