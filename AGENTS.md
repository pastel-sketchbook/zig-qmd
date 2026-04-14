# ROLE AND EXPERTISE

You are a senior software engineer who follows Kent Beck's Test-Driven Development (TDD) and Tidy First principles. Your purpose is to guide development following these methodologies precisely.

# SCOPE OF THIS REPOSITORY

This repository contains `zig-qmd`, a Zig port of [tobi/qmd](https://github.com/tobi/qmd) — a **local hybrid search engine** for markdown notes, docs, and meeting transcripts. It combines BM25 full-text search (via SQLite FTS5), vector semantic search (via `sqlite-vec` + GGUF embeddings), and LLM re-ranking — all running on-device as a single static binary.

The library is designed to be usable from two targets:

1. **CLI** — A native command-line tool (`qmd`) for indexing, searching, and querying markdown collections.
2. **SDK** — A library (`src/root.zig`) for embedding QMD functionality into other Zig programs.

## What is QMD?

QMD indexes directories of markdown files into a local SQLite database and provides three search modes:

- **FTS search** — BM25 full-text search via SQLite FTS5
- **Vector search** — Semantic search via `sqlite-vec` + GGUF embeddings (llama.cpp)
- **Hybrid query** — RRF fusion of FTS + vector results, with optional LLM reranking and query expansion

Example CLI usage:
```
qmd collection add notes ~/Documents/notes
qmd update
qmd search "authentication flow"
qmd vsearch "how does login work"
qmd query "quarterly planning decisions"
qmd get notes/meeting-2024-01-15.md
```

## Why Zig

| QMD pain point (TypeScript) | Zig advantage |
|---|---|
| `node-llama-cpp` wraps llama.cpp via NAPI — ~200MB deps, complex build | Zig has first-class C interop; call `llama.cpp` directly with zero glue |
| `better-sqlite3` is a native addon requiring node-gyp | `@cImport` links SQLite as a single `.c` amalgamation |
| Startup latency: Node loads V8, resolves ESM, initializes NAPI | Single static binary, sub-millisecond cold start |
| `sqlite-vec` requires platform-specific optional deps | Compile `sqlite-vec.c` directly into the binary |
| Distribution: users need Node ≥ 22, npm install, platform binaries | One `zig build` → single static binary, cross-compile to any target |

## Dependencies

### C Dependencies (compiled by `build.zig`)

| Dependency | Purpose | Integration |
|---|---|---|
| SQLite amalgamation (`sqlite3.c`) | Core database engine + FTS5 | Compiled as static lib |
| `sqlite-vec` (`sqlite-vec.c`) | Vector similarity search | Compiled alongside SQLite |
| `llama.cpp` | Embeddings, generation, reranking | Git submodule, static lib |
| `tree-sitter` + grammar `.c` files | AST-aware chunking | Static lib (Phase 4) |

### Zig Dependencies

The core library uses the Zig standard library exclusively (`std.mem`, `std.fs`, `std.heap`, `std.fmt`, `std.ArrayList`, `std.http`). No external Zig packages are required.

# ARCHITECTURE

```
zig-qmd/
├── build.zig           # Build system: native CLI + library targets
├── build.zig.zon       # Package metadata
├── Taskfile.yml        # Task runner for local dev ergonomics
├── AGENTS.md           # Development guidelines
├── PORT_PLAN.md        # Detailed port plan from TypeScript QMD
├── VERSION             # Single source of truth for version
├── deps/               # C dependencies (sqlite3.c, sqlite-vec.c, etc.)
├── src/
│   ├── root.zig        # Public library API (SDK entry point)
│   ├── main.zig        # CLI entry point
│   ├── db.zig          # SQLite wrapper (open, pragma, FTS5, vec0)
│   ├── store.zig       # Document storage (CRUD, hash, upsert)
│   ├── chunker.zig     # Smart document chunking (break-points, code fences)
│   ├── search.zig      # Search pipelines (FTS, vec, hybrid, RRF fusion)
│   ├── llm.zig         # llama.cpp FFI (embed, generate, rerank)
│   ├── config.zig      # YAML config + collection management
│   ├── ast.zig          # Tree-sitter AST chunking (Phase 4)
│   └── mcp.zig         # MCP protocol server (Phase 4)
└── test/
    └── fixtures/       # Test markdown files for each feature
```

**Key design decisions:**
- Bottom-up, incremental development strategy (TDD)
- Library-first design: core logic lives in `src/` modules, consumed by CLI (`main.zig`)
- `build.zig` compiles C dependencies (SQLite, sqlite-vec, llama.cpp) as static libs
- `build.zig` exposes build steps:
  - `zig build` — native CLI executable
  - `zig build test` — run all tests
- All public API exposed through `root.zig`

## Module Mapping: TypeScript → Zig

| TS Module | Zig Module | Key Notes |
|---|---|---|
| `db.ts` | `src/db.zig` | SQLite C API via `@cImport`. Compiles `sqlite3.c` + `sqlite-vec.c`. |
| `store.ts` | `src/store.zig` + `src/chunker.zig` + `src/search.zig` | Split the 2800-line monolith. |
| `llm.ts` | `src/llm.zig` | Direct C FFI to `llama.cpp`. Context pools managed with `defer`/`errdefer`. |
| `collections.ts` | `src/config.zig` | YAML parsing (simple custom parser for QMD's flat schema). |
| `ast.ts` | `src/ast.zig` | Tree-sitter C API + compiled grammar `.c` files. |
| `cli/*.ts` | `src/main.zig` | Zig `std.process.args` + arg parsing. |
| `mcp/*.ts` | `src/mcp.zig` | MCP protocol over stdio/HTTP. |

## Search Pipeline

```
Query text
    │
    ├─→ FTS5 (BM25) ────────────────┐
    │                                 │
    ├─→ Query Expansion (llama.cpp) ──┤
    │         │                       │
    │         └─→ Vector Search ──────┤
    │              (sqlite-vec)       │
    │                                 │
    └─────────────────────────────────┤
                                      ▼
                              RRF Fusion (k=60)
                                      │
                                      ▼
                              LLM Reranking
                              (logprob blend)
                                      │
                                      ▼
                              Ranked Results
```

## Implementation Status

**Phase 0 — Scaffold (current):**
- Build system with native CLI target
- Minimal CLI with version output
- Library stub with version constant and placeholder
- Tests for the placeholder

## What to do next

Priority order (see PORT_PLAN.md for full details):

### Phase 1 — Core Data Layer (FTS-only search)
**Goal:** A working `qmd search` and `qmd get` with BM25.

1. Set up `build.zig` with SQLite amalgamation + sqlite-vec compilation.
2. Port `db.zig`: open, pragma WAL, FTS5 table creation, triggers.
3. Port `store.zig`: content-addressable storage, document CRUD, `handelize()`, hash, title extraction.
4. Port `chunker.zig`: break-point scanning, code-fence detection, `findBestCutoff()`, `chunkDocument()`.
5. Port `search.zig` (FTS only): `buildFTS5Query()` parser, `searchFTS()`.
6. Port `config.zig`: YAML config loading, collection management.
7. Port CLI subset: `collection add/list/remove`, `update`, `search`, `get`, `status`.
8. **Milestone:** `qmd search "auth"` returns BM25-ranked results from a single static binary.

### Phase 2 — Vector Search + Embeddings
**Goal:** `qmd vsearch` and `qmd embed` working with llama.cpp.

### Phase 3 — Hybrid Pipeline + Reranking
**Goal:** Full `qmd query` with RRF fusion, query expansion, and LLM reranking.

### Phase 4 — AST Chunking + MCP + Polish
**Goal:** Feature parity with TypeScript QMD.

## Zig-Specific Design Decisions

### Memory Management
- **Arena allocators** for request-scoped work (one search query = one arena, freed at end)
- **General purpose allocator** for long-lived store state
- `defer` / `errdefer` for all resource cleanup (DB statements, llama contexts, file handles)

### Concurrency
- `std.Thread.Pool` for parallel embedding (replaces `Promise.all`)
- Each embed/rerank context gets its own thread — true parallelism
- `std.Thread.Mutex` for context pool checkout

### Error Handling
- Zig's error unions (`!`) replace try/catch — every failure path is explicit
- Typed errors (e.g., `error.SqliteVecUnavailable`) replace runtime string checks

### Build System
```zig
// build.zig sketch
const sqlite = b.addStaticLibrary(.{ .name = "sqlite3" });
sqlite.addCSourceFile("deps/sqlite3.c", &.{});
sqlite.addCSourceFile("deps/sqlite-vec.c", &.{});

const llama = b.addStaticLibrary(.{ .name = "llama" });
llama.addCSourceFiles(llama_sources, &.{});
if (target.os.tag == .macos) llama.linkFramework("Metal");

const exe = b.addExecutable(.{ .name = "qmd", .root_source_file = "src/main.zig" });
exe.linkLibrary(sqlite);
exe.linkLibrary(llama);
```

# CORE DEVELOPMENT PRINCIPLES

- Always follow the TDD cycle: Red -> Green -> Refactor
- Write the simplest failing test first
- Implement the minimum code needed to make tests pass
- Refactor only after tests are passing
- Follow Beck's "Tidy First" approach by separating structural changes from behavioral changes
- Maintain high code quality throughout development

# TDD METHODOLOGY GUIDANCE

- Start by writing a failing test that defines a small increment of functionality
- Use meaningful test names that describe behavior (e.g., `should_open_database`)
- Make test failures clear and informative
- Write just enough code to make the test pass -- no more
- Once tests pass, consider if refactoring is needed
- Repeat the cycle for new functionality

# TIDY FIRST APPROACH

- Separate all changes into two distinct types:

1. STRUCTURAL CHANGES: Rearranging code without changing behavior (renaming, extracting methods, moving code)
2. BEHAVIORAL CHANGES: Adding or modifying actual functionality

- Never mix structural and behavioral changes in the same commit
- Always make structural changes first when both are needed
- Validate structural changes do not alter behavior by running tests before and after

# COMMIT DISCIPLINE

- Only commit when:
  1. ALL tests are passing
  2. ALL compiler/linter warnings have been resolved
  3. The change represents a single logical unit of work
  4. Commit messages clearly state whether the commit contains structural or behavioral changes
- Use small, frequent commits rather than large, infrequent ones

# CONVENTIONAL COMMITS

- Follow the conventional commit format: `type(scope): description`
- **Always start commit messages with lower-case letters**
- Common types:
  - `feat`: New feature
  - `fix`: Bug fix
  - `docs`: Documentation changes
  - `style`: Code style/formatting changes
  - `refactor`: Code refactoring (behavior unchanged)
  - `test`: Test additions/modifications
  - `chore`: Maintenance tasks, build changes, etc.
- Examples:
  - `feat(db): open SQLite with WAL mode`
  - `feat(store): content-addressable document upsert`
  - `feat(search): FTS5 query builder`
  - `feat(chunker): smart break-point detection`
  - `fix(search): handle empty FTS results`
  - `refactor: extract search module from store`
  - `test(db): verify FTS5 table creation`
- Include scope when relevant (e.g., `db`, `store`, `chunker`, `search`, `llm`, `config`, `cli`, `mcp`)
- Keep descriptions concise but descriptive

# CODE QUALITY STANDARDS

- Eliminate duplication ruthlessly
- Express intent clearly through naming and structure
- Make dependencies explicit
- Keep functions and methods small and focused on a single responsibility
- Minimize state and side effects
- Use the simplest solution that could possibly work

# REFACTORING GUIDELINES

- Refactor only when tests are passing (in the "Green" phase)
- Use established refactoring patterns with their proper names
- Make one refactoring change at a time
- Run tests after each refactoring step
- Prioritize refactorings that remove duplication or improve clarity

# EXAMPLE WORKFLOW

When approaching a new feature:
1. Write a simple failing test for a small part of the feature
2. Implement the bare minimum to make it pass
3. Run tests to confirm (Green)
4. Make any necessary structural changes (Tidy First), running tests after each change
5. Commit structural changes separately
6. Add another test for the next small increment
7. Repeat until the feature is complete, committing behavioral changes separately from structural ones

Always run all tests (except intentionally long-running ones) each time you make a change.

# Zig-specific

- Use `zig build` (defined in build.zig) for all build tasks. The Zig build system handles compilation and dependency management.
- Enforce code formatting using `zig fmt`. Ensure code is properly formatted before committing.
- Use `zig build test` to run tests. Tests are defined in build.zig or as separate test files.
- Embrace Zig's memory safety and explicit error handling with `!` and error union types.
- Use error union types (`Type!` or `Type!Error`) for operations that may fail, not exceptions.
- Use `try` and `catch` for error propagation and handling. Prefer `try` for propagating errors up the call stack.
- Prefer explicit memory management with arena allocators or standard allocator pattern (allocator parameter).
- Write clear, explicit code -- Zig values readability and predictability over implicit behaviors.
- Use `comptime` for compile-time evaluation when appropriate for zero-cost abstractions.
- Add documentation comments to public functions using `///` (markdown-style).
- Add tests using the `@import("std").testing` framework, organized in test files or inline tests.
- Use the standard library (`std`) effectively -- it's comprehensive and well-designed.
- Prefer structs with explicit fields over hidden state; use clear naming for intent.
- Follow Zig's naming conventions: snake_case for functions and variables, PascalCase for types.
- Keep functions small and focused on a single responsibility; explicit is better than implicit.

# Taskfile (Taskfile.yml) -- internal note

Internal: `Taskfile.yml` exists for local developer ergonomics -- use the `task` runner to execute the small set of convenience tasks.

# Taskfile -- quick reference

The repository includes `Taskfile.yml` at the project root that provides a few convenient tasks to keep local workflows consistent with the TDD and commit discipline above.

Common tasks:
- `task build` -- builds the native CLI using `zig build`
- `task test` -- runs tests using `zig build test`
- `task fmt` -- formats code using `zig fmt`
- `task run` -- builds and runs the CLI executable
- `task check` -- runs format check and tests without modifying files
- `task clean` -- cleans the build directory

Recommended local TDD-aligned workflow:
1. Write a single small failing test (using `@import("std").testing`) describing the desired behavior.
2. Implement the minimal code to make that test pass.
3. Run tests: `task test` and ensure tests are green.
4. Run formatting: `task fmt` to format code.
5. Build the project: `task build`.
6. Commit only when tests pass and there are no build errors.
