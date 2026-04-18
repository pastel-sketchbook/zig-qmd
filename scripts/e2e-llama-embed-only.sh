#!/usr/bin/env bash
set -euo pipefail

# E2E test with embed model only (no generation model)

export QMD_EMBED_MODEL="${QMD_EMBED_MODEL:-$HOME/tools/llama/models/nomic-embed-text-v1.5.Q8_0.gguf}"
unset QMD_MODEL 2>/dev/null || true

echo "EMBED_MODEL: $QMD_EMBED_MODEL"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ZMD="$ROOT_DIR/zig-out/bin/zmd"

FILTER='ggml\|^llama_\|^print_info\|^load\|^create_tensor\|^sched\|^graph\|^set_abort\|^done_getting\|^\.\.\|^~llama\|^kernel\|^init_token\|compiling pipeline\|^loaded kernel'

DIR=$(mktemp -d)
trap "rm -rf $DIR" EXIT
mkdir -p "$DIR/notes"

printf '# Auth Guide\n\nAuthentication details here.\n' > "$DIR/notes/auth.md"
printf '# Deploy Guide\n\nDeployment instructions.\n' > "$DIR/notes/deploy.md"

cd "$DIR"

$ZMD update
$ZMD collection add notes ./notes
$ZMD update

echo "=== Embed ==="
$ZMD embed "test embedding" 2>&1 | grep -E 'Embedding|error'

echo ""
echo "=== Vector Search ==="
$ZMD vsearch "how to authenticate" 2>&1 | grep -v "$FILTER"

echo ""
echo "E2E embed-only test complete!"
