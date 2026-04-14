# 0003 Indexing Performance

## Context

ZMD targets large markdown repositories (6,900–77,000 files). The original
indexing pipeline took 106 seconds for 515 documents (3.7 MB). Extrapolating
to 77K documents, a full index would take over 6 hours. Re-indexing unchanged
files was equally slow because no work was skipped.

## Problem analysis

Profiling the per-document hot path revealed six bottlenecks, ranked by
contribution to wall time:

1. **No skip for unchanged documents.** Every re-index re-read, re-chunked,
   and re-embedded all files regardless of content changes.
2. **Prepared statement recompilation.** Each document triggered 8+ calls to
   `sqlite3_prepare_v2` compiling the same SQL strings every time.
3. **Tree-sitter parser allocated per document.** `ts_parser_new()` +
   `ts_parser_delete()` on every file; parsers are designed for reuse.
4. **Embedding engine created per document.** Environment variable reads,
   file existence checks, and allocator work repeated for every file.
5. **Redundant hash re-query.** After `insertDocument` computed the SHA-256
   hash internally, a separate `findActiveDocumentHash` SELECT re-fetched it.
6. **No progress feedback.** Large collections gave no indication of progress,
   making it impossible to estimate completion time.

## Changes

### Phase A — Eliminate waste (behavioral)

| Change | Files | Effect |
|---|---|---|
| Return hash from `insertDocument` | store.zig | Eliminates 1 SELECT per document |
| `InsertResult.content_changed` via `sqlite3_changes()` | store.zig, db.zig | Skip chunking + embedding for unchanged docs |
| Hoist `make_embedding_engine` before loop | main.zig | 1 init instead of N |
| Hoist fallback `LlamaCpp` before loop | main.zig | 1 init instead of N |

### Phase B — Structural (Tidy First)

| Change | Files | Effect |
|---|---|---|
| `Db.prepareCached()` with fixed-capacity cache | db.zig | Compile each SQL once; reuse via `sqlite3_reset` |
| Reuse tree-sitter parser across documents | ast.zig, main.zig | 1 `ts_parser_new` per update, not per document |

### Phase C — UX

| Change | Files | Effect |
|---|---|---|
| Per-collection progress every 500 docs | main.zig | Shows `... N documents processed` during indexing |
| Per-collection new/unchanged breakdown | main.zig | `Indexed 515 documents (0 new, 515 unchanged)` |
| Total summary with skip count | main.zig | `Total: 515 documents (515 unchanged, skipped)` |

## Measured results (515 documents, 3.7 MB)

| Scenario | Before | After | Speedup |
|---|---|---|---|
| Re-index (all unchanged) | 106 s | 0.85 s | **125x** |
| Fresh index (all new) | 106 s | 69 s | **1.5x** |

## Projected impact on target repositories

| Repo | Files | Before (est.) | After (est.) |
|---|---|---|---|
| legalize-kr (fresh) | 6,908 | ~2.3 h | ~1.0 h |
| precedent-kr (fresh) | 76,943 | ~6 h | ~3.5 h |
| legalize-kr (re-index, unchanged) | 6,908 | ~2.3 h | ~12 s |
| precedent-kr (re-index, unchanged) | 76,943 | ~6 h | ~2 min |

## Statement cache design

The cache uses a fixed-size array (16 slots) keyed by SQL string pointer
identity. Since all hot-path SQL strings are comptime literals, pointer
comparison is sufficient — no hashing or string comparison needed. Cached
statements are reset via `sqlite3_reset()` + `sqlite3_clear_bindings()` on
each reuse. The cache is finalized in `Db.close()`.

Callers use `db_.prepareCached(sql)` instead of `db_.prepare(sql)` and must
not call `stmt.finalize()` on the returned handle. Only the 5 hot-path
functions were converted; read-path queries continue using uncached `prepare`.

## Future opportunities

- **Bind embeddings as f32 blobs** instead of JSON text to avoid per-chunk
  string serialization (~4 KB allocation saved per chunk).
- **File mtime tracking** to skip reading unchanged files entirely (requires
  adding an `mtime` column to the `documents` table).
- **Parallel file reading** with `std.Thread.Pool` to overlap I/O with
  SQLite writes.
