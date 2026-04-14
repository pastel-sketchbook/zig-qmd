# 0001 — Sort search results by score by default

**Date:** 2026-04-14
**Status:** Accepted

## Architecture overview

ZMD is a local hybrid search engine for markdown notes. It compiles to a
single static binary that bundles SQLite (with FTS5 + sqlite-vec), tree-sitter,
and optionally llama.cpp — no runtime dependencies.

```
User query
    │
    ├─→ FTS5 (BM25)  ───────────────────┐
    │    src/search.zig:searchFTS()     │
    │                                   │
    ├─→ Query Expansion (llama.cpp) ────┤
    │    src/llm.zig:expandQuery()      │
    │         │                         │
    │         └─→ Vector Search ────────┤
    │              src/search.zig       │
    │              :searchVec()         │
    │              (sqlite-vec cosine)  │
    │                                   │
    └───────────────────────────────────┤
                                        ▼
                                RRF Fusion (k=60)
                                src/search.zig
                                :reciprocalRankFusion()
                                        │
                                        ▼
                                LLM Reranking
                                src/search.zig
                                :rerankByEmbedding()
                                        │
                                        ▼
                                CLI Sort Layer     ← THIS DECISION
                                src/main.zig
                                :sortSearchResults()
                                        │
                                        ▼
                                Formatted Output
                                (text / json / csv / md)
```

### Module structure

```
src/
├── root.zig      Public library API (SDK entry point)
├── main.zig      CLI: arg parsing, output formatting, sort logic
├── db.zig        SQLite wrapper: open, pragma WAL, FTS5, triggers
├── store.zig     Content-addressable doc storage: CRUD, hash, upsert
├── chunker.zig   Smart chunking: break-points, code fences, overlap
├── search.zig    Search pipelines: FTS, vector, hybrid, RRF fusion
├── llm.zig       Embedding + generation: llama.cpp FFI, FNV fallback
├── config.zig    Collection management: add/list/remove in SQLite
├── ast.zig       Tree-sitter AST chunking with regex fallback
└── mcp.zig       MCP JSON-RPC server over stdio
```

### How scores flow through the system

Each search path produces its own score type:

| Path | Score source | Range | Meaning |
|------|-------------|-------|---------|
| FTS (BM25) | `bm25()` SQL function | 0–1 (normalized) | Higher = more term relevance |
| Vector | Cosine similarity | -1–1 (raw), 0–1 (normalized) | Higher = more semantic similarity |
| RRF fusion | `1/(k + rank)` sum | 0–1 | Higher = appears high in more lists |
| Reranking | Confidence-weighted blend of dense + generative scores | 0–1 | Higher = LLM agrees it's relevant |

The CLI sort layer (`main.zig`) operates **after** the search pipeline has
already computed final scores. The pipeline itself sorts by score internally
(FTS via SQL `ORDER BY`, vector/RRF/rerank via `std.sort.heap`). The CLI
sort layer provides a user-facing control point to override that order.

### Where the sort happens

```
search.zig                        main.zig
──────────                        ────────
searchFTS()  ─→ ORDER BY score ─→ sortSearchResults(.score)  ← default
searchVec()  ─→ heap sort desc ─→ sortScoredResults(.score)  ← default
hybridSearch ─→ RRF + rerank   ─→ sortSearchResults(.score)  ← default
                                  sortSearchResults(.index)   ← --sort=index
```

The sort in `main.zig` is intentionally redundant with the pipeline's own
sort for the `.score` case. This is by design:

1. The pipeline's sort is an internal optimization (top-k truncation needs
   score order).
2. The CLI's sort is a user-facing contract that remains correct even if the
   pipeline changes its internal ordering.
3. `--sort=index` flips to ascending row-id, which no pipeline stage provides.

### C dependencies compiled by build.zig

| Dependency | Purpose | Flags |
|-----------|---------|-------|
| `sqlite3.c` | Database + FTS5 | `-DSQLITE_ENABLE_FTS5`, `-fno-sanitize=undefined` |
| `sqlite-vec.c` | Vector similarity via `vec0` virtual tables | `-fno-sanitize=undefined` |
| tree-sitter + `tree-sitter-md` | AST-aware markdown chunking | Standard C flags |
| llama.cpp (optional) | Embeddings, generation, reranking | `-Dllama` build flag |

The `-fno-sanitize=undefined` flags are required because SQLite's FTS5
Porter stemmer contains known-benign signed integer overflow that triggers
Zig's UB sanitizer.

## Context

All search commands (`search`, `vsearch`, `query`, `context`) return ranked
results. The underlying search layer (FTS BM25, vector cosine, RRF fusion)
already computes relevance scores for every result. However, the CLI output
order was inconsistent — some commands showed results in score order while
others preserved the database row-id order. Users expect a search tool to show
the *best* results first, not the oldest.

## Decision

1. **Sort by score descending is the default** for all search commands.
2. A `--sort=index` flag is available to sort by database row-id (insertion
   order) instead, for workflows that need chronological or stable ordering.
3. The text output format is unified across all search commands to:
   ```
   1. Title (zmd://collection/path) score=0.9055
   ```
   This puts the most useful information — rank, title, virtual path, and
   score — on a single scannable line.

## Consequences

- The most relevant result is always position 1 unless `--sort=index` is used.
- Machine-readable formats (`--json`, `--csv`, `--md`) also respect the sort
  flag so piped workflows remain predictable.
- Score-first ordering makes `zmd search` / `zmd context` immediately useful
  as a "what matches best?" tool without extra flags.

## Alternatives considered

- **No flag, always score-order.** Rejected because some users pipe results
  into other tools and need stable id-based ordering.
- **`--sort=path` or `--sort=title` alphabetical sorts.** Deferred — easy to
  add later without breaking the current flag contract.
