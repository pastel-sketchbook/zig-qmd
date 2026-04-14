#!/usr/bin/env bash
set -euo pipefail

EMBED_BIN="${EMBED_BIN:-deps/llama.cpp/build/bin/llama-embedding}"
MODEL="${MODEL:-hf://ggml-org/embeddinggemma-300M-qat-q4_0-GGUF:Q4_0}"
TEST_ROOT="/tmp/qmd-test-real"

rm -rf .qmd "$TEST_ROOT"
mkdir -p .qmd "$TEST_ROOT/notes"

cat > "$TEST_ROOT/notes/auth.md" <<'EOF'
# Authentication Flow

Users sign in with OAuth and receive a JWT access token.
EOF

cat > "$TEST_ROOT/notes/cooking.md" <<'EOF'
# Pasta Recipe

Boil salted water, cook pasta, and finish with olive oil.
EOF

./zig-out/bin/zmd update
./zig-out/bin/zmd collection add notes "$TEST_ROOT/notes"

QMD_LLAMA_EMBED_BIN="$EMBED_BIN" QMD_LLAMA_MODEL="$MODEL" ./zig-out/bin/zmd update

echo
echo "=== Vector Search (real model) ==="
QMD_LLAMA_EMBED_BIN="$EMBED_BIN" QMD_LLAMA_MODEL="$MODEL" ./zig-out/bin/zmd vsearch "oauth token sign in"

echo
echo "=== Hybrid Query (real model) ==="
QMD_LLAMA_EMBED_BIN="$EMBED_BIN" QMD_LLAMA_MODEL="$MODEL" ./zig-out/bin/zmd query "oauth token sign in"

echo
echo "=== Stored vectors ==="
sqlite3 .qmd/data.db "select count(*) as vectors from content_vectors;"

rm -rf .qmd "$TEST_ROOT"
echo
echo "Real-model E2E test complete!"
