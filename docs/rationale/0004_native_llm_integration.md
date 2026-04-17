# 0004 Native LLM Integration

## Context

ZMD's vector search pipeline originally used two approaches for embedding:

1. **Subprocess** (`llm.zig`): Spawns `llama-embedding` as a child process,
   parses JSON stdout. Requires the llama.cpp binary to be pre-built and
   discoverable via `QMD_LLAMA_EMBED_BIN`.
2. **FNV fallback** (`LlamaCpp`): When no embedding binary is available,
   produces deterministic FNV-hash-based pseudo-embeddings. Useful for testing
   but semantically meaningless.

Neither approach achieves the original goal of a single static binary with
zero external dependencies.

## Decision

Link llama.cpp as a static library directly into the zmd binary, gated behind
a `-Dllama` build flag so the default build remains small and dependency-free.

### Architecture

```
build.zig  ─── -Dllama flag ──→  link libllama.a, libggml*.a
                                  │
src/c_llama.h  ──────────────→  translate-C module (c_llama)
                                  │
src/llm_native.zig  ─────────→  NativeLlama struct
                                  ├── embed()     → L2-normalized float vectors
                                  ├── generate()  → autoregressive text generation
                                  └── chat()      → Gemma 4 E2B formatted conversation
                                  │
src/search.zig  ──────────────→  EmbedFn / ExpandQueryFn function pointers
                                  (pluggable, null = subprocess fallback)
                                  │
src/main.zig  ────────────────→  g_native_llama module-level pointer
                                  + wrapper functions for EmbedFn/ExpandQueryFn
```

### Key design choices

1. **Compile-time gating via `build_options.enable_llama`.**
   `llm_native.zig` uses `@compileError` when llama is disabled, preventing
   any llama.cpp symbols from leaking into the default build. `root.zig`
   conditionally exports the module: `if (enable_llama) @import("llm_native.zig") else struct {}`.

2. **Function pointer injection, not trait/interface.**
   Zig has no closures or vtables. The search pipeline accepts optional
   `EmbedFn` and `ExpandQueryFn` function pointers that default to `null`
   (falling back to existing subprocess/FNV paths). This avoids refactoring
   the entire search API while making the embedding strategy pluggable.

3. **Module-level mutable state for function-pointer context.**
   Since Zig function pointers cannot capture state, `main.zig` uses a
   module-level `var g_native_llama` that the wrapper functions read. This is
   set before each command's search call and cleared after. Single-threaded
   CLI execution makes this safe.

4. **Dual-model architecture for embedding and generation.**
   Gemma 4 E2B is a generative model, not an embedding model. The
   architecture loads separate GGUF files via two env vars:
   - `QMD_EMBED_MODEL` — dedicated embedding model (e.g., nomic-embed-text)
   - `QMD_MODEL` — generation model for query expansion/chat (e.g., Gemma 4 E2B)
   
   `QMD_EMBED_MODEL` falls back to `QMD_MODEL` if not set. The CLI uses
   `g_embed_llama` for all embedding operations and `g_native_llama` for
   generation. Zig function pointers can't capture state, so module-level
   globals bridge the gap.

5. **Static library linking, not dynamic.**
   All llama.cpp libraries (`libllama.a`, `libggml.a`, `libggml-base.a`,
   `libggml-cpu.a`, `libggml-blas.a`, `libggml-metal.a`, `libcommon.a`) are
   linked statically. On macOS, Metal/Accelerate frameworks are linked for
   GPU acceleration. The result is still a single binary.

## Consequences

**Positive:**
- `zmd` can embed documents and expand queries without any external process.
- Sub-millisecond embedding call overhead (no process spawn, no JSON parse).
- Single binary distribution — no need to ship llama.cpp alongside zmd.
- Backward compatible — builds without `-Dllama` are unchanged.

**Negative:**
- Binary size increases significantly when `-Dllama` is enabled (~50MB+).
- llama.cpp static libs must be pre-built via CMake before `zig build -Dllama`.
- Module-level mutable state (`g_native_llama`) is not thread-safe; would need
  refactoring for concurrent server use (MCP phase).

## Alternatives considered

1. **Dynamic linking (`dlopen`).** Would keep binary small but defeats the
   single-binary goal and complicates distribution.
2. **Zig-native transformer implementation.** Too much effort for marginal
   benefit when llama.cpp already supports the target models.
3. **HTTP API to local llama.cpp server.** Adds deployment complexity and
   latency; contradicts the "zero infrastructure" design goal.
