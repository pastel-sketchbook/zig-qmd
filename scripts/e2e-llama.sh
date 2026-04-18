#!/usr/bin/env bash
set -euo pipefail

# E2E test with native llama.cpp dual models (embed + generation)
# Defaults can be overridden via env vars.

export QMD_EMBED_MODEL="${QMD_EMBED_MODEL:-$HOME/tools/llama/models/nomic-embed-text-v1.5.Q8_0.gguf}"
export QMD_MODEL="${QMD_MODEL:-$HOME/tools/llama/models/gemma-4-E2B-it-Q8_0.gguf}"

echo "EMBED_MODEL: $QMD_EMBED_MODEL"
echo "MODEL:       $QMD_MODEL"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ZMD="$ROOT_DIR/zig-out/bin/zmd"

# llama.cpp logs go to stderr — filter them for clean output
FILTER='ggml\|^llama_\|^print_info\|^load\|^create_tensor\|^sched\|^graph\|^set_abort\|^done_getting\|^\.\.\|^~llama\|^kernel\|^init_token\|compiling pipeline\|^loaded kernel'

DIR=$(mktemp -d)
trap "rm -rf $DIR" EXIT
mkdir -p "$DIR/notes"

printf '# Authentication Flow\n\nJWT tokens for session management. OAuth2 with Google or GitHub.\nTokens expire after 24 hours. Refresh tokens rotated on each use.\n' > "$DIR/notes/auth.md"
printf '# Deployment Guide\n\nDocker build, push to registry, Helm upgrade on Kubernetes.\nRollback with helm rollback. Prometheus metrics on /metrics.\n' > "$DIR/notes/deploy.md"
printf '# Database Schema\n\nUsers table with UUID primary key, email, name.\nSessions table references users. Indexes on email and expires_at.\n' > "$DIR/notes/schema.md"

cd "$DIR"

echo "=== Init ==="
$ZMD update
$ZMD collection add notes ./notes
$ZMD update
$ZMD status

echo ""
echo "=== FTS Search ==="
$ZMD search "authentication JWT"

echo ""
echo "=== Embed ==="
$ZMD embed "hello world" 2>&1 | grep -E 'Embedding|error'

echo ""
echo "=== Vector Search ==="
$ZMD vsearch "how does user login work" 2>&1 | grep -v "$FILTER"

echo ""
echo "=== Hybrid Query (dual model) ==="
$ZMD query "how to deploy the application" 2>&1 | grep -v "$FILTER"

echo ""
echo "=== Context (RAG answer) ==="
$ZMD context "how does authentication work" 2>&1 | grep -v "$FILTER"

echo ""
echo "E2E llama test complete!"
