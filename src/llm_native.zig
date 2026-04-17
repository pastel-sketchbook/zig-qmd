const std = @import("std");
const build_options = @import("build_options");
const c = if (build_options.enable_llama) @import("c_llama") else @compileError("llm_native requires -Dllama build flag");

/// Error set for native llama.cpp operations.
pub const NativeLlamaError = error{
    BackendInitFailed,
    ModelLoadFailed,
    ContextCreateFailed,
    TokenizeFailed,
    EncodeFailed,
    DecodeFailed,
    EmbeddingFailed,
    SamplerFailed,
    OutOfMemory,
};

/// Native llama.cpp model wrapper using direct C FFI.
/// Manages model and context lifecycle for embedding and generation.
pub const NativeLlama = struct {
    model: *c.llama_model,
    allocator: std.mem.Allocator,

    /// Loads a GGUF model file and initializes the llama backend.
    pub fn init(allocator: std.mem.Allocator, model_path: [*:0]const u8) NativeLlamaError!NativeLlama {
        c.llama_backend_init();

        var mparams = c.llama_model_default_params();
        mparams.n_gpu_layers = 99; // offload all layers to GPU when available

        const model = c.llama_model_load_from_file(model_path, mparams) orelse {
            c.llama_backend_free();
            return NativeLlamaError.ModelLoadFailed;
        };

        return .{
            .model = model,
            .allocator = allocator,
        };
    }

    /// Frees the model and cleans up the llama backend.
    pub fn deinit(self: *NativeLlama) void {
        c.llama_model_free(self.model);
        c.llama_backend_free();
    }

    /// Returns the embedding dimension of the loaded model.
    pub fn embeddingDim(self: *const NativeLlama) i32 {
        return c.llama_model_n_embd(self.model);
    }

    /// Returns true if the loaded model natively supports embedding (pooling_type != -1).
    /// Generative-only models (e.g., Gemma 4 E2B) return false.
    pub fn supportsEmbedding(self: *const NativeLlama) bool {
        // Create a temporary context to check default pooling type
        var cparams = c.llama_context_default_params();
        cparams.embeddings = true;
        cparams.pooling_type = c.LLAMA_POOLING_TYPE_UNSPECIFIED;
        const ctx = c.llama_init_from_model(self.model, cparams) orelse return false;
        defer c.llama_free(ctx);
        const pool = c.llama_pooling_type(ctx);
        return pool != c.LLAMA_POOLING_TYPE_UNSPECIFIED and pool != c.LLAMA_POOLING_TYPE_NONE;
    }

    /// Generates an embedding vector for the given text.
    /// The caller owns the returned slice.
    /// Returns EmbeddingFailed if the model does not support embedding.
    pub fn embed(self: *NativeLlama, text: []const u8) NativeLlamaError![]f32 {
        // Create context with embedding mode
        var cparams = c.llama_context_default_params();
        cparams.n_batch = 2048;
        cparams.n_ubatch = 2048;
        cparams.embeddings = true;
        cparams.pooling_type = c.LLAMA_POOLING_TYPE_MEAN;

        const ctx = c.llama_init_from_model(self.model, cparams) orelse
            return NativeLlamaError.ContextCreateFailed;
        defer c.llama_free(ctx);

        // Tokenize
        const vocab = c.llama_model_get_vocab(self.model);
        const max_tokens: i32 = 2048;
        const tokens = self.allocator.alloc(c.llama_token, @intCast(max_tokens)) catch
            return NativeLlamaError.OutOfMemory;
        defer self.allocator.free(tokens);

        const n_tokens = c.llama_tokenize(
            vocab,
            text.ptr,
            @intCast(text.len),
            tokens.ptr,
            max_tokens,
            true, // add_special (BOS)
            true, // parse_special
        );
        if (n_tokens < 0) return NativeLlamaError.TokenizeFailed;

        // Create batch and encode
        const batch = c.llama_batch_get_one(tokens.ptr, n_tokens);
        const encode_result = c.llama_encode(ctx, batch);
        if (encode_result != 0) return NativeLlamaError.EncodeFailed;

        // Extract pooled embedding (use seq-level for pooled models)
        const n_embd = c.llama_model_n_embd(self.model);
        const emb_ptr = c.llama_get_embeddings_seq(ctx, 0) orelse
            c.llama_get_embeddings(ctx);
        if (emb_ptr == null) return NativeLlamaError.EmbeddingFailed;

        // Copy to owned slice
        const result = self.allocator.alloc(f32, @intCast(n_embd)) catch
            return NativeLlamaError.OutOfMemory;
        @memcpy(result, emb_ptr[0..@intCast(n_embd)]);

        // L2 normalize
        var norm: f32 = 0;
        for (result) |x| norm += x * x;
        norm = @sqrt(norm);
        if (norm > 0) {
            for (result) |*x| x.* /= norm;
        }

        return result;
    }

    /// Generates text given a prompt string.
    /// The caller owns the returned slice.
    pub fn generate(self: *NativeLlama, prompt: []const u8, max_tokens: u32) NativeLlamaError![]u8 {
        var cparams = c.llama_context_default_params();
        cparams.n_batch = 2048;
        cparams.n_ubatch = 2048;
        cparams.n_ctx = 4096;

        const ctx = c.llama_init_from_model(self.model, cparams) orelse
            return NativeLlamaError.ContextCreateFailed;
        defer c.llama_free(ctx);

        const vocab = c.llama_model_get_vocab(self.model);

        // Tokenize prompt
        const max_prompt_tokens: i32 = 4096;
        const tokens = self.allocator.alloc(c.llama_token, @intCast(max_prompt_tokens)) catch
            return NativeLlamaError.OutOfMemory;
        defer self.allocator.free(tokens);

        const n_prompt = c.llama_tokenize(
            vocab,
            prompt.ptr,
            @intCast(prompt.len),
            tokens.ptr,
            max_prompt_tokens,
            true,
            true,
        );
        if (n_prompt < 0) return NativeLlamaError.TokenizeFailed;

        // Decode prompt
        const prompt_batch = c.llama_batch_get_one(tokens.ptr, n_prompt);
        if (c.llama_decode(ctx, prompt_batch) != 0)
            return NativeLlamaError.DecodeFailed;

        // Set up sampler chain (greedy for deterministic output)
        const sparams = c.llama_sampler_chain_default_params();
        const sampler = c.llama_sampler_chain_init(sparams) orelse
            return NativeLlamaError.SamplerFailed;
        defer c.llama_sampler_free(sampler);
        c.llama_sampler_chain_add(sampler, c.llama_sampler_init_greedy() orelse
            return NativeLlamaError.SamplerFailed);

        // Generate tokens
        var output: std.ArrayList(u8) = .empty;
        defer output.deinit(self.allocator);

        const eos = c.llama_vocab_eos(vocab);
        var n_decoded: i32 = n_prompt;
        var buf: [256]u8 = undefined;

        for (0..max_tokens) |_| {
            var new_token = c.llama_sampler_sample(sampler, ctx, -1);
            if (new_token == eos) break;

            // Detokenize
            const n = c.llama_token_to_piece(vocab, new_token, &buf, @intCast(buf.len), 0, true);
            if (n > 0) {
                output.appendSlice(self.allocator, buf[0..@intCast(n)]) catch
                    return NativeLlamaError.OutOfMemory;
            }

            // Prepare next batch
            const next_batch = c.llama_batch_get_one(&new_token, 1);
            if (c.llama_decode(ctx, next_batch) != 0)
                return NativeLlamaError.DecodeFailed;
            n_decoded += 1;
        }

        return output.toOwnedSlice(self.allocator) catch return NativeLlamaError.OutOfMemory;
    }

    /// Generates text using Gemma 4 chat template format.
    /// Wraps the prompt in system/user roles with optional thinking mode.
    pub fn chat(
        self: *NativeLlama,
        system_prompt: ?[]const u8,
        user_message: []const u8,
        max_tokens: u32,
        enable_thinking: bool,
    ) NativeLlamaError![]u8 {
        // Build Gemma 4 formatted prompt
        var prompt_buf: std.ArrayList(u8) = .empty;
        defer prompt_buf.deinit(self.allocator);

        // System turn
        if (system_prompt) |sys| {
            prompt_buf.appendSlice(self.allocator, "<start_of_turn>system\n") catch return NativeLlamaError.OutOfMemory;
            if (enable_thinking) {
                prompt_buf.appendSlice(self.allocator, "<|think|>\n") catch return NativeLlamaError.OutOfMemory;
            }
            prompt_buf.appendSlice(self.allocator, sys) catch return NativeLlamaError.OutOfMemory;
            prompt_buf.appendSlice(self.allocator, "<end_of_turn>\n") catch return NativeLlamaError.OutOfMemory;
        } else if (enable_thinking) {
            prompt_buf.appendSlice(self.allocator, "<start_of_turn>system\n<|think|>\n<end_of_turn>\n") catch return NativeLlamaError.OutOfMemory;
        }

        // User turn
        prompt_buf.appendSlice(self.allocator, "<start_of_turn>user\n") catch return NativeLlamaError.OutOfMemory;
        prompt_buf.appendSlice(self.allocator, user_message) catch return NativeLlamaError.OutOfMemory;
        prompt_buf.appendSlice(self.allocator, "<end_of_turn>\n") catch return NativeLlamaError.OutOfMemory;

        // Model turn prompt
        prompt_buf.appendSlice(self.allocator, "<start_of_turn>model\n") catch return NativeLlamaError.OutOfMemory;

        const full_prompt = prompt_buf.items;
        return self.generate(full_prompt, max_tokens);
    }
};

/// Formats a Gemma 4 chat prompt from system + user messages.
/// Useful for building prompts outside of NativeLlama.
pub fn formatGemma4Prompt(
    allocator: std.mem.Allocator,
    system_prompt: ?[]const u8,
    user_message: []const u8,
    enable_thinking: bool,
) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    if (system_prompt) |sys| {
        try buf.appendSlice(allocator, "<start_of_turn>system\n");
        if (enable_thinking) try buf.appendSlice(allocator, "<|think|>\n");
        try buf.appendSlice(allocator, sys);
        try buf.appendSlice(allocator, "<end_of_turn>\n");
    } else if (enable_thinking) {
        try buf.appendSlice(allocator, "<start_of_turn>system\n<|think|>\n<end_of_turn>\n");
    }

    try buf.appendSlice(allocator, "<start_of_turn>user\n");
    try buf.appendSlice(allocator, user_message);
    try buf.appendSlice(allocator, "<end_of_turn>\n");
    try buf.appendSlice(allocator, "<start_of_turn>model\n");

    return buf.toOwnedSlice(allocator);
}

// =============================================================================
// Tests
// =============================================================================

test "formatGemma4Prompt basic" {
    const prompt = try formatGemma4Prompt(
        std.testing.allocator,
        "You are a helpful assistant.",
        "Hello!",
        false,
    );
    defer std.testing.allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "<start_of_turn>system\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "You are a helpful assistant.") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<start_of_turn>user\nHello!") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<start_of_turn>model\n") != null);
    // No thinking token
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|think|>") == null);
}

test "formatGemma4Prompt with thinking" {
    const prompt = try formatGemma4Prompt(
        std.testing.allocator,
        "You are a helpful assistant.",
        "Solve this math problem.",
        true,
    );
    defer std.testing.allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|think|>") != null);
}

test "formatGemma4Prompt no system prompt" {
    const prompt = try formatGemma4Prompt(
        std.testing.allocator,
        null,
        "Hello!",
        false,
    );
    defer std.testing.allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "system") == null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<start_of_turn>user\nHello!") != null);
}

test "formatGemma4Prompt no system with thinking" {
    const prompt = try formatGemma4Prompt(
        std.testing.allocator,
        null,
        "Hello!",
        true,
    );
    defer std.testing.allocator.free(prompt);

    // Should still include system turn with just the think token
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<start_of_turn>system\n<|think|>") != null);
}
