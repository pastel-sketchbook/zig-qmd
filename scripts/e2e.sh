#!/usr/bin/env bash
set -euo pipefail

# E2E test with fresh database and test files (no llama, FTS + FNV fallback)
ZMD="${ZMD:-./zig-out/bin/zmd}"

rm -rf .qmd /tmp/qmd-test
mkdir -p .qmd /tmp/qmd-test/notes

printf '# Auth Guide\n\nAuthentication details here.\n' > /tmp/qmd-test/notes/auth.md
printf '# Test Doc\n\nThis is about authentication flow.\n' > /tmp/qmd-test/notes/test.md

$ZMD update
$ZMD collection add notes /tmp/qmd-test/notes
$ZMD update

echo ""
echo "=== FTS Search ==="
$ZMD search auth
echo ""
echo "=== Vector Search ==="
$ZMD vsearch auth
echo ""
echo "=== Hybrid Query ==="
$ZMD query auth
echo ""
echo "=== Status ==="
$ZMD status

rm -rf .qmd /tmp/qmd-test
echo ""
echo "E2E test complete!"
