#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" == "Darwin" ]]; then
    if ! command -v brew >/dev/null 2>&1; then
        echo "Homebrew is required on macOS to locate libpq. Install libpq, then rerun:" >&2
        echo "  brew install libpq" >&2
        exit 1
    fi
    if ! brew --prefix libpq >/dev/null 2>&1; then
        echo "libpq is required for Diesel/Postgres linking. Install it, then rerun:" >&2
        echo "  brew install libpq" >&2
        exit 1
    fi
    export LIBRARY_PATH
    LIBRARY_PATH="$(brew --prefix libpq)/lib:${LIBRARY_PATH:-}"
fi

cargo test -p sage-core chat_stream
