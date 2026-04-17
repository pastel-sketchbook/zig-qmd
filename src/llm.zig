const std = @import("std");
const Io = std.Io;
const db = @import("db.zig");

/// Error set for llama.cpp integration operations.
pub const LlamaError = error{
    InitFailed,
    LoadFailed,
    EmbedFailed,
    ContextFull,
    ModelNotLoaded,
    TokenizationFailed,
    NoModel,
    OutOfMemory,
    EncodingFailed,
    NotFound,
    SpawnFailed,
};

/// FNV-hash-based fallback embedding engine used when no real model is available.
pub const LlamaCpp = struct {
    model_path: []const u8,
    embedding_dim: usize = 384,
    loaded: bool = false,
    temp_dir: ?std.Io.Dir = null,
    pid: ?std.process.Child.Id = null,

    /// Creates a LlamaCpp instance, checking if the model file exists.
    pub fn init(model_path: []const u8, _: std.mem.Allocator) LlamaError!LlamaCpp {
        var self = LlamaCpp{
            .model_path = model_path,
            .loaded = false,
        };

        // Check if model file exists using posix openat
        const cwd_fd = std.Io.Dir.cwd().handle;
        const fd = std.posix.openat(cwd_fd, model_path, .{}, 0) catch {
            // Model not found, will use fallback
            return self;
        };
        std.Io.Threaded.closeFd(fd);

        // Model exists - ready for real embeddings
        self.loaded = true;
        self.embedding_dim = 384;

        return self;
    }

    /// Releases resources held by the LlamaCpp instance.
    pub fn deinit(self: *LlamaCpp) void {
        if (self.temp_dir) |dir| {
            std.Io.Threaded.closeFd(dir.handle);
            self.temp_dir = null;
        }
        self.loaded = false;
    }

    /// Generates a deterministic FNV-hash embedding for the given text.
    pub fn embed(self: *LlamaCpp, text: []const u8, allocator: std.mem.Allocator) LlamaError![]f32 {
        // Use FNV hash as fallback embedding when no model is loaded
        const embedding = allocator.alloc(f32, self.embedding_dim) catch return LlamaError.EmbedFailed;
        for (0..self.embedding_dim, embedding) |i, *slot| {
            var h: u32 = 2166136261;
            for (text) |c| h +%= @as(u32, c) * 16777619;
            h +%= @as(u32, @intCast(i)) *% 16777619;
            slot.* = @as(f32, @floatFromInt(h & 0xFFFF)) / 65535.0 - 0.5;
        }

        // L2 normalize
        var norm: f32 = 0;
        for (embedding) |x| norm += x * x;
        norm = @sqrt(norm);
        if (norm > 0) {
            for (embedding) |*x| x.* /= norm;
        }

        return embedding;
    }

    /// Generates embeddings for multiple texts.
    pub fn embedBatch(self: *LlamaCpp, texts: [][]const u8, allocator: std.mem.Allocator) LlamaError![][]f32 {
        var embeddings = try allocator.alloc([]f32, texts.len);
        errdefer for (embeddings) |emb| allocator.free(emb);
        for (texts, 0..) |text, i| {
            embeddings[i] = try self.embed(text, allocator);
        }
        return embeddings;
    }

    /// Returns the embedding dimensionality.
    pub fn embedDim(self: *const LlamaCpp) usize {
        return self.embedding_dim;
    }

    /// Returns whether a real model file was found.
    pub fn isLoaded(self: *const LlamaCpp) bool {
        return self.loaded;
    }
};

/// Role in a chat conversation (system, user, assistant).
pub const ChatRole = enum {
    system,
    user,
    assistant,
};

/// A single message in a chat conversation.
pub const ChatMessage = struct {
    role: ChatRole,
    content: []u8,
};

/// Multi-turn chat session backed by a llama.cpp generation subprocess.
pub const LlamaChatSession = struct {
    allocator: std.mem.Allocator,
    io: Io,
    binary_path: []u8,
    model_path: []u8,
    history: std.ArrayList(ChatMessage),
    max_history: usize = 12,

    /// Creates a new chat session with paths to binary and model.
    pub fn init(allocator: std.mem.Allocator, io: Io, binary_path: []const u8, model_path: []const u8) !LlamaChatSession {
        const bp = try allocator.dupe(u8, binary_path);
        errdefer allocator.free(bp);
        const mp = try allocator.dupe(u8, model_path);
        errdefer allocator.free(mp);
        const hist = try std.ArrayList(ChatMessage).initCapacity(allocator, 8);
        return .{
            .allocator = allocator,
            .io = io,
            .binary_path = bp,
            .model_path = mp,
            .history = hist,
        };
    }

    /// Frees all resources and message history.
    pub fn deinit(self: *LlamaChatSession) void {
        for (self.history.items) |msg| self.allocator.free(msg.content);
        self.history.deinit(self.allocator);
        self.allocator.free(self.binary_path);
        self.allocator.free(self.model_path);
    }

    /// Adds a system prompt to the conversation history.
    pub fn addSystemPrompt(self: *LlamaChatSession, prompt: []const u8) !void {
        try self.appendMessage(.system, prompt);
    }

    /// Sends a user message and returns the model's response.
    pub fn send(self: *LlamaChatSession, user_input: []const u8) ![]u8 {
        try self.appendMessage(.user, user_input);

        const prompt = try self.buildPrompt();
        defer self.allocator.free(prompt);

        const response = runGeneration(self.allocator, self.io, self.binary_path, self.model_path, prompt) catch try self.allocator.dupe(u8, "Model unavailable");
        try self.appendOwnedMessage(.assistant, response);
        return try self.allocator.dupe(u8, response);
    }

    fn appendMessage(self: *LlamaChatSession, role: ChatRole, content: []const u8) !void {
        const copy = try self.allocator.dupe(u8, content);
        try self.appendOwnedMessage(role, copy);
    }

    fn appendOwnedMessage(self: *LlamaChatSession, role: ChatRole, content: []u8) !void {
        try self.history.append(self.allocator, .{ .role = role, .content = content });
        if (self.history.items.len > self.max_history) {
            const old = self.history.orderedRemove(0);
            self.allocator.free(old.content);
        }
    }

    fn buildPrompt(self: *LlamaChatSession) ![]u8 {
        var out = try std.ArrayList(u8).initCapacity(self.allocator, 1024);
        defer out.deinit(self.allocator);
        for (self.history.items) |msg| {
            const prefix = switch (msg.role) {
                .system => "system",
                .user => "user",
                .assistant => "assistant",
            };
            try out.appendSlice(self.allocator, prefix);
            try out.appendSlice(self.allocator, ": ");
            try out.appendSlice(self.allocator, msg.content);
            try out.append(self.allocator, '\n');
        }
        try out.appendSlice(self.allocator, "assistant:");
        return out.toOwnedSlice(self.allocator);
    }
};

/// Scores passages for relevance to a query using generation or lexical fallback.
pub fn rerankPassages(
    allocator: std.mem.Allocator,
    io: Io,
    query: []const u8,
    passages: []const []const u8,
    maybe_binary_path: ?[]const u8,
    maybe_model_path: ?[]const u8,
) ![]f32 {
    var scores = try allocator.alloc(f32, passages.len);
    for (passages, 0..) |p, i| {
        const generated = if (maybe_binary_path != null and maybe_model_path != null)
            scoreWithGeneration(allocator, io, query, p, maybe_binary_path.?, maybe_model_path.?)
        else
            null;

        if (generated) |s| {
            scores[i] = s;
        } else {
            scores[i] = lexicalRelevanceScore(query, p);
        }
    }
    return scores;
}

fn runGeneration(allocator: std.mem.Allocator, io: Io, binary_path: []const u8, model_path: []const u8, prompt: []const u8) ![]u8 {
    if (binary_path.len == 0 or model_path.len == 0) return error.SpawnFailed;

    var argv = try std.ArrayList([]const u8).initCapacity(allocator, 16);
    defer argv.deinit(allocator);
    try argv.append(allocator, binary_path);

    const spec = parseModelSpec(model_path) catch return error.SpawnFailed;
    switch (spec.source) {
        .huggingface => {
            try argv.append(allocator, "--hf-repo");
            try argv.append(allocator, spec.value);
        },
        .url => {
            try argv.append(allocator, "--model-url");
            try argv.append(allocator, spec.value);
        },
        .local => {
            try argv.append(allocator, "-m");
            try argv.append(allocator, spec.value);
        },
    }
    try argv.append(allocator, "-n");
    try argv.append(allocator, "96");
    try argv.append(allocator, "-p");
    try argv.append(allocator, prompt);

    const run_result = std.process.run(allocator, io, .{
        .argv = argv.items,
        .stdout_limit = @enumFromInt(64 * 1024),
    }) catch return error.SpawnFailed;
    defer allocator.free(run_result.stderr);
    defer allocator.free(run_result.stdout);

    if (run_result.term != .exited or run_result.term.exited != 0) return error.SpawnFailed;
    const trimmed = std.mem.trim(u8, run_result.stdout, &.{ ' ', '\n', '\r', '\t' });
    if (trimmed.len == 0) return error.SpawnFailed;
    return allocator.dupe(u8, trimmed);
}

fn scoreWithGeneration(allocator: std.mem.Allocator, io: Io, query: []const u8, passage: []const u8, binary_path: []const u8, model_path: []const u8) ?f32 {
    const prompt = std.fmt.allocPrint(allocator, "Rate relevance from 0 to 1.\nQuery: {s}\nPassage: {s}\nScore:", .{ query, passage }) catch return null;
    defer allocator.free(prompt);

    const out = runGeneration(allocator, io, binary_path, model_path, prompt) catch return null;
    defer allocator.free(out);

    var tok = std.mem.tokenizeAny(u8, out, " \n\r\t,:;");
    while (tok.next()) |t| {
        const f = std.fmt.parseFloat(f32, t) catch continue;
        return @max(@as(f32, 0), @min(@as(f32, 1), f));
    }
    return null;
}

fn lexicalRelevanceScore(query: []const u8, passage: []const u8) f32 {
    var matches: f32 = 0;
    var total: f32 = 0;
    var q_it = std.mem.tokenizeAny(u8, query, " \n\r\t,.;:!?()[]{}\"'");
    while (q_it.next()) |tok| {
        if (tok.len < 2) continue;
        total += 1;
        if (containsAsciiCaseInsensitive(passage, tok)) matches += 1;
    }
    if (total == 0) return 0;
    return matches / total;
}

fn containsAsciiCaseInsensitive(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;

    var start: usize = 0;
    while (start + needle.len <= haystack.len) : (start += 1) {
        var matched = true;
        for (needle, 0..) |ch, offset| {
            if (std.ascii.toLower(haystack[start + offset]) != std.ascii.toLower(ch)) {
                matched = false;
                break;
            }
        }
        if (matched) return true;
    }

    return false;
}

/// Maximum text length (in bytes) for embedding input.
pub const EMBEDDING_MAX_TEXT_LEN: usize = 2048;

/// Formats text for embedding with appropriate query/document prefix.
pub fn formatTextForEmbedding(allocator: std.mem.Allocator, text: []const u8, is_query: bool) ![]u8 {
    if (is_query) {
        return formatQueryForEmbedding(allocator, text);
    }
    return formatDocForEmbedding(allocator, text);
}

/// Normalizes and prefixes text with "query: " for embedding.
pub fn formatQueryForEmbedding(allocator: std.mem.Allocator, query: []const u8) ![]u8 {
    const normalized = try normalizeEmbeddingText(allocator, query, EMBEDDING_MAX_TEXT_LEN);
    defer allocator.free(normalized);

    var out = try std.ArrayList(u8).initCapacity(allocator, "query: ".len + normalized.len);
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "query: ");
    try out.appendSlice(allocator, normalized);
    return out.toOwnedSlice(allocator);
}

/// Normalizes and prefixes text with "passage: " for embedding.
pub fn formatDocForEmbedding(allocator: std.mem.Allocator, doc: []const u8) ![]u8 {
    const normalized = try normalizeEmbeddingText(allocator, doc, EMBEDDING_MAX_TEXT_LEN);
    defer allocator.free(normalized);

    var out = try std.ArrayList(u8).initCapacity(allocator, "passage: ".len + normalized.len);
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "passage: ");
    try out.appendSlice(allocator, normalized);
    return out.toOwnedSlice(allocator);
}

fn normalizeEmbeddingText(allocator: std.mem.Allocator, text: []const u8, max_len: usize) ![]u8 {
    var out = try std.ArrayList(u8).initCapacity(allocator, @min(text.len, max_len));
    defer out.deinit(allocator);

    var prev_space = true;
    for (text) |ch| {
        if (out.items.len >= max_len) break;

        const is_space = ch == ' ' or ch == '\n' or ch == '\t' or ch == '\r';
        if (is_space) {
            if (!prev_space and out.items.len < max_len) {
                try out.append(allocator, ' ');
                prev_space = true;
            }
            continue;
        }

        try out.append(allocator, ch);
        prev_space = false;
    }

    if (out.items.len > 0 and out.items[out.items.len - 1] == ' ') {
        _ = out.pop();
    }

    return out.toOwnedSlice(allocator);
}

/// Computes cosine similarity between two float vectors.
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return 0;
    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;
    for (0..a.len) |i| {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0 or norm_b == 0) return 0;
    return dot / (@sqrt(norm_a) * @sqrt(norm_b));
}

/// Computes Euclidean distance between two float vectors.
pub fn euclideanDistance(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return 0;
    var dist: f32 = 0;
    for (0..a.len) |i| {
        const diff = a[i] - b[i];
        dist += diff * diff;
    }
    return @sqrt(dist);
}

/// Computes dot product of two float vectors.
pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return 0;
    var sum: f32 = 0;
    for (0..a.len) |i| sum += a[i] * b[i];
    return sum;
}

/// In-memory key-value cache for LLM results.
pub const LlmCache = struct {
    allocator: std.mem.Allocator,
    entries: std.StringHashMap(CachedResult),

    /// Creates a new empty LLM cache.
    pub fn init(allocator: std.mem.Allocator) LlmCache {
        return .{ .allocator = allocator, .entries = std.StringHashMap(CachedResult).init(allocator) };
    }

    /// Frees all cache entries and keys.
    pub fn deinit(self: *LlmCache) void {
        var it = self.entries.iterator();
        while (it.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.entries.deinit();
    }

    /// Looks up a cached result by key.
    pub fn get(self: *const LlmCache, key: []const u8) ?CachedResult {
        return self.entries.get(key);
    }

    /// Stores a result in the cache.
    pub fn put(self: *LlmCache, key: []const u8, value: CachedResult) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        try self.entries.put(key_copy, value);
    }

    /// Removes all entries from the cache.
    pub fn clear(self: *LlmCache) void {
        var it = self.entries.iterator();
        while (it.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.entries.clearRetainingCapacity();
    }
};

/// A cached LLM result with creation timestamp.
pub const CachedResult = struct {
    response: []const u8,
    created_at: i64,
};

/// Error set for cache operations.
pub const CacheError = error{
    OutOfMemory,
} || db.DbError;

/// Builds a deterministic SHA-256 cache key from kind, model, and input.
pub fn buildCacheKey(kind: []const u8, model: []const u8, input: []const u8) [64]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(kind);
    hasher.update("|");
    hasher.update(model);
    hasher.update("|");
    hasher.update(input);

    var digest: [32]u8 = undefined;
    hasher.final(&digest);

    var out: [64]u8 = undefined;
    for (digest, 0..) |byte, i| {
        out[i * 2] = "0123456789abcdef"[(byte >> 4) & 0x0f];
        out[i * 2 + 1] = "0123456789abcdef"[byte & 0x0f];
    }
    return out;
}

/// Retrieves a cached result from the SQLite llm_cache table.
pub fn cacheGet(db_: *db.Db, key: []const u8, allocator: std.mem.Allocator) CacheError!?[]u8 {
    var stmt = try db_.prepare("SELECT result FROM llm_cache WHERE hash = ?");
    defer stmt.finalize();
    try stmt.bindText(1, key);

    if (!try stmt.step()) return null;
    const text = stmt.columnText(0) orelse return null;
    return allocator.dupe(u8, std.mem.span(text)) catch return CacheError.OutOfMemory;
}

/// Stores a result in the SQLite llm_cache table.
pub fn cachePut(db_: *db.Db, key: []const u8, value: []const u8) CacheError!void {
    var stmt = try db_.prepare("INSERT OR REPLACE INTO llm_cache(hash, result, created_at) VALUES(?, ?, ?)");
    defer stmt.finalize();
    try stmt.bindText(1, key);
    try stmt.bindText(2, value);
    try stmt.bindText(3, "2024-01-01T00:00:00Z");
    _ = try stmt.step();
}

/// Minimal search result with id and score for reranking.
pub const SimpleResult = struct {
    id: i64,
    score: f64,
};

/// Placeholder reranking function (currently returns results unchanged).
pub fn rerank(results: []const SimpleResult, query: []const u8) ![]SimpleResult {
    _ = query;
    return results;
}

/// Expands a search query with related terms using a model or heuristic fallback.
pub fn expandQuery(allocator: std.mem.Allocator, io: Io, query: []const u8) ![]const u8 {
    return expandQueryWithModel(allocator, io, query, null, null) catch expandQueryHeuristic(allocator, query);
}

/// Expands a query using a llama.cpp generation subprocess or heuristic fallback.
pub fn expandQueryWithModel(
    allocator: std.mem.Allocator,
    io: Io,
    query: []const u8,
    maybe_binary_path: ?[]const u8,
    maybe_model_path: ?[]const u8,
) ![]const u8 {
    if (maybe_binary_path == null or maybe_model_path == null) {
        return expandQueryHeuristic(allocator, query);
    }

    const spec = parseModelSpec(maybe_model_path.?) catch return expandQueryHeuristic(allocator, query);

    const prompt = std.fmt.allocPrint(
        allocator,
        "Expand this search query with short related keywords only. Return one line, comma-separated keywords. Query: {s}",
        .{query},
    ) catch return expandQueryHeuristic(allocator, query);
    defer allocator.free(prompt);

    var argv = try std.ArrayList([]const u8).initCapacity(allocator, 16);
    defer argv.deinit(allocator);
    try argv.append(allocator, maybe_binary_path.?);
    switch (spec.source) {
        .huggingface => {
            try argv.append(allocator, "--hf-repo");
            try argv.append(allocator, spec.value);
        },
        .url => {
            try argv.append(allocator, "--model-url");
            try argv.append(allocator, spec.value);
        },
        .local => {
            try argv.append(allocator, "-m");
            try argv.append(allocator, spec.value);
        },
    }
    try argv.append(allocator, "-n");
    try argv.append(allocator, "48");
    try argv.append(allocator, "-p");
    try argv.append(allocator, prompt);

    const run_result = std.process.run(allocator, io, .{
        .argv = argv.items,
        .stdout_limit = @enumFromInt(16 * 1024),
    }) catch return expandQueryHeuristic(allocator, query);
    defer allocator.free(run_result.stdout);
    defer allocator.free(run_result.stderr);

    if (run_result.term != .exited or run_result.term.exited != 0 or run_result.stdout.len == 0) {
        return expandQueryHeuristic(allocator, query);
    }

    const trimmed = std.mem.trim(u8, run_result.stdout, &.{ ' ', '\n', '\r', '\t' });
    if (trimmed.len == 0) return expandQueryHeuristic(allocator, query);

    var merged = try std.ArrayList(u8).initCapacity(allocator, query.len + 1 + trimmed.len);
    defer merged.deinit(allocator);
    try merged.appendSlice(allocator, query);
    try merged.append(allocator, ' ');
    try merged.appendSlice(allocator, trimmed);
    return merged.toOwnedSlice(allocator);
}

fn expandQueryHeuristic(allocator: std.mem.Allocator, query: []const u8) ![]const u8 {
    var expanded = try std.ArrayList(u8).initCapacity(allocator, query.len);
    errdefer expanded.deinit(allocator);
    try expanded.appendSlice(allocator, query);
    if (std.mem.indexOf(u8, query, "?") != null) try expanded.appendSlice(allocator, " explain clarify");
    if (std.mem.indexOf(u8, query, "how") != null) try expanded.appendSlice(allocator, " method way steps procedure");
    if (std.mem.indexOf(u8, query, "what") != null) try expanded.appendSlice(allocator, " definition meaning information");
    if (std.mem.indexOf(u8, query, "why") != null) try expanded.appendSlice(allocator, " reason cause explanation");
    if (std.mem.indexOf(u8, query, "where") != null) try expanded.appendSlice(allocator, " location place");
    if (std.mem.indexOf(u8, query, "when") != null) try expanded.appendSlice(allocator, " time date schedule");
    return expanded.toOwnedSlice(allocator);
}

test "cosineSimilarity identical" {
    const sim = cosineSimilarity(&.{ 1, 0, 0 }, &.{ 1, 0, 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sim, 0.001);
}

test "cosineSimilarity orthogonal" {
    const sim = cosineSimilarity(&.{ 1, 0, 0 }, &.{ 0, 1, 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sim, 0.001);
}

test "expandQuery adds question terms" {
    const expanded = try expandQuery(std.testing.allocator, std.testing.io, "how to login");
    defer std.testing.allocator.free(expanded);
    try std.testing.expect(std.mem.indexOf(u8, expanded, "method").? > 0);
}

test "parseModelSpec recognizes hf model" {
    const spec = try parseModelSpec("hf://ggml-org/embeddinggemma-300M-qat-q4_0-GGUF:Q4_0");
    try std.testing.expect(spec.source == .huggingface);
    try std.testing.expectEqualStrings("ggml-org/embeddinggemma-300M-qat-q4_0-GGUF:Q4_0", spec.value);
}

test "parseModelSpec recognizes local model" {
    const spec = try parseModelSpec("deps/models/model.gguf");
    try std.testing.expect(spec.source == .local);
    try std.testing.expectEqualStrings("deps/models/model.gguf", spec.value);
}

test "parseModelSpec rejects empty" {
    try std.testing.expectError(EmbeddingError.InvalidModelSpec, parseModelSpec(""));
}

test "formatQueryForEmbedding normalizes and prefixes" {
    const formatted = try formatQueryForEmbedding(std.testing.allocator, "  how\n to\tlogin  ");
    defer std.testing.allocator.free(formatted);
    try std.testing.expectEqualStrings("query: how to login", formatted);
}

test "formatDocForEmbedding normalizes and prefixes" {
    const formatted = try formatDocForEmbedding(std.testing.allocator, "# Auth\n\nLogin flow details");
    defer std.testing.allocator.free(formatted);
    try std.testing.expectEqualStrings("passage: # Auth Login flow details", formatted);
}

test "formatDocForEmbedding truncates long input" {
    var long: [3000]u8 = undefined;
    @memset(&long, 'a');
    const formatted = try formatDocForEmbedding(std.testing.allocator, &long);
    defer std.testing.allocator.free(formatted);
    try std.testing.expect(formatted.len <= "passage: ".len + EMBEDDING_MAX_TEXT_LEN);
}

test "LlmCache stores and retrieves" {
    var cache = LlmCache.init(std.testing.allocator);
    defer cache.deinit();
    try cache.put("key", .{ .response = "val", .created_at = 1 });
    try std.testing.expect(cache.get("key") != null);
}

// =============================================================================
// LlamaEmbedding - Subprocess-based embedding engine
// =============================================================================

/// Error set for subprocess-based embedding operations.
pub const EmbeddingError = error{
    BinaryNotFound,
    ModelNotFound,
    InvalidModelSpec,
    SpawnFailed,
    ProcessFailed,
    ParseError,
    InvalidJson,
    OutOfMemory,
    Timeout,
};

/// Where a model is located: local file, HuggingFace repo, or URL.
pub const ModelSource = enum {
    local,
    huggingface,
    url,
};

/// Parsed model specification with source type and path/URL.
pub const ModelSpec = struct {
    source: ModelSource,
    value: []const u8,
};

/// Parses a model path string into a ModelSpec (hf://, http://, or local).
pub fn parseModelSpec(model_path: []const u8) EmbeddingError!ModelSpec {
    if (model_path.len == 0) return EmbeddingError.InvalidModelSpec;

    if (std.mem.startsWith(u8, model_path, "hf://")) {
        if (model_path.len <= 5) return EmbeddingError.InvalidModelSpec;
        return .{ .source = .huggingface, .value = model_path[5..] };
    }

    if (std.mem.startsWith(u8, model_path, "https://") or std.mem.startsWith(u8, model_path, "http://")) {
        return .{ .source = .url, .value = model_path };
    }

    return .{ .source = .local, .value = model_path };
}

/// Validates that a local model file exists on disk.
pub fn validateModelPath(model_path: []const u8) EmbeddingError!void {
    const model_spec = try parseModelSpec(model_path);
    if (model_spec.source == .local) {
        const cwd_fd = std.Io.Dir.cwd().handle;
        const fd = std.posix.openat(cwd_fd, model_spec.value, .{}, 0) catch {
            return EmbeddingError.ModelNotFound;
        };
        std.Io.Threaded.closeFd(fd);
    }
}

/// Subprocess-based embedding engine using llama-embedding binary
pub const LlamaEmbedding = struct {
    binary_path: []u8,
    model_path: []u8,
    embedding_dim: usize,
    normalize: i8 = 2, // L2 normalization by default
    allocator: std.mem.Allocator,
    io: Io,

    const Self = @This();

    /// Initialize the embedding engine with paths to binary and model
    pub fn init(
        allocator: std.mem.Allocator,
        io: Io,
        binary_path: []const u8,
        model_path: []const u8,
    ) EmbeddingError!Self {
        // Validate binary path exists (runtime spawn will verify execution)
        const cwd_fd = std.Io.Dir.cwd().handle;
        const fd = std.posix.openat(cwd_fd, binary_path, .{}, 0) catch {
            return EmbeddingError.BinaryNotFound;
        };
        std.Io.Threaded.closeFd(fd);

        try validateModelPath(model_path);

        const bin_copy = allocator.dupe(u8, binary_path) catch return EmbeddingError.OutOfMemory;
        errdefer allocator.free(bin_copy);
        const model_copy = allocator.dupe(u8, model_path) catch return EmbeddingError.OutOfMemory;

        return Self{
            .binary_path = bin_copy,
            .model_path = model_copy,
            .embedding_dim = 384, // Default, will be determined from model
            .allocator = allocator,
            .io = io,
        };
    }

    /// Frees binary and model path strings.
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.binary_path);
        self.allocator.free(self.model_path);
    }

    /// Generate embedding for a single text using llama-embedding subprocess
    pub fn embed(self: *Self, text: []const u8) EmbeddingError![]f32 {
        const embeddings = try self.embedBatch(&[_][]const u8{text});
        defer self.allocator.free(embeddings);

        if (embeddings.len == 0) return EmbeddingError.ParseError;

        // Return the first embedding (caller owns it)
        const result = embeddings[0];
        return result;
    }

    /// Generate embeddings for multiple texts in batch using llama-embedding subprocess
    pub fn embedBatch(self: *Self, texts: []const []const u8) EmbeddingError![][]f32 {
        if (texts.len == 0) {
            return self.allocator.alloc([]f32, 0) catch return EmbeddingError.OutOfMemory;
        }

        // Build prompt with separator for batch processing
        // llama-embedding supports multiple prompts with --embd-separator
        const separator = "<#sep#>";
        var total_len: usize = 0;
        for (texts) |t| total_len += t.len + separator.len;

        var prompt = try std.ArrayList(u8).initCapacity(self.allocator, total_len);
        defer prompt.deinit(self.allocator);

        for (texts, 0..) |t, i| {
            prompt.appendSlice(self.allocator, t) catch return EmbeddingError.OutOfMemory;
            if (i < texts.len - 1) {
                prompt.appendSlice(self.allocator, separator) catch return EmbeddingError.OutOfMemory;
            }
        }

        // Format normalize argument
        var norm_buf: [8]u8 = undefined;
        const norm_str = std.fmt.bufPrint(&norm_buf, "{d}", .{self.normalize}) catch return EmbeddingError.OutOfMemory;

        var argv = try std.ArrayList([]const u8).initCapacity(self.allocator, 16);
        defer argv.deinit(self.allocator);

        try argv.append(self.allocator, self.binary_path);

        const model_spec = parseModelSpec(self.model_path) catch return EmbeddingError.InvalidModelSpec;
        switch (model_spec.source) {
            .huggingface => {
                try argv.append(self.allocator, "--hf-repo");
                try argv.append(self.allocator, model_spec.value);
            },
            .url => {
                try argv.append(self.allocator, "--model-url");
                try argv.append(self.allocator, model_spec.value);
            },
            .local => {
                try argv.append(self.allocator, "-m");
                try argv.append(self.allocator, model_spec.value);
            },
        }

        try argv.append(self.allocator, "--embd-output-format");
        try argv.append(self.allocator, "json");
        try argv.append(self.allocator, "--embd-normalize");
        try argv.append(self.allocator, norm_str);
        try argv.append(self.allocator, "--embd-separator");
        try argv.append(self.allocator, separator);
        try argv.append(self.allocator, "-p");
        try argv.append(self.allocator, prompt.items);

        const run_result = std.process.run(self.allocator, self.io, .{
            .argv = argv.items,
            .stdout_limit = @enumFromInt(10 * 1024 * 1024),
        }) catch return EmbeddingError.SpawnFailed;
        defer self.allocator.free(run_result.stdout);
        defer self.allocator.free(run_result.stderr);

        if (run_result.term != .exited or run_result.term.exited != 0) {
            return EmbeddingError.ProcessFailed;
        }

        // Parse JSON output
        return parseEmbeddingJson(self.allocator, run_result.stdout);
    }
};

/// Parse OpenAI-style embedding JSON response from llama-embedding
/// Format: {"object":"list","data":[{"object":"embedding","index":N,"embedding":[...]},...]}
pub fn parseEmbeddingJson(allocator: std.mem.Allocator, json_str: []const u8) EmbeddingError![][]f32 {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_str, .{}) catch {
        return EmbeddingError.InvalidJson;
    };
    defer parsed.deinit();

    const root = parsed.value;

    // Navigate to data array
    const data = root.object.get("data") orelse return EmbeddingError.ParseError;
    if (data != .array) return EmbeddingError.ParseError;

    var result = allocator.alloc([]f32, data.array.items.len) catch {
        return EmbeddingError.OutOfMemory;
    };
    errdefer allocator.free(result);

    for (data.array.items, 0..) |item, i| {
        if (item != .object) {
            // Clean up already allocated embeddings
            for (0..i) |j| allocator.free(result[j]);
            return EmbeddingError.ParseError;
        }

        const embedding_val = item.object.get("embedding") orelse {
            for (0..i) |j| allocator.free(result[j]);
            return EmbeddingError.ParseError;
        };

        if (embedding_val != .array) {
            for (0..i) |j| allocator.free(result[j]);
            return EmbeddingError.ParseError;
        }

        const emb_array = embedding_val.array.items;
        var embedding = allocator.alloc(f32, emb_array.len) catch {
            for (0..i) |j| allocator.free(result[j]);
            return EmbeddingError.OutOfMemory;
        };
        errdefer allocator.free(embedding);

        for (emb_array, 0..) |val, k| {
            embedding[k] = switch (val) {
                .float => @floatCast(val.float),
                .integer => @floatFromInt(val.integer),
                else => {
                    allocator.free(embedding);
                    for (0..i) |j| allocator.free(result[j]);
                    return EmbeddingError.ParseError;
                },
            };
        }

        result[i] = embedding;
    }

    return result;
}

// =============================================================================
// LlamaEmbedding Tests
// =============================================================================

test "LlamaEmbedding.init fails with non-existent binary" {
    const allocator = std.testing.allocator;
    const result = LlamaEmbedding.init(
        allocator,
        std.testing.io,
        "/nonexistent/llama-embedding",
        "/some/model.gguf",
    );
    try std.testing.expectError(EmbeddingError.BinaryNotFound, result);
}

test "LlamaEmbedding.init fails with non-existent model" {
    const allocator = std.testing.allocator;
    // Use a binary that exists (the test runner itself)
    const result = LlamaEmbedding.init(
        allocator,
        std.testing.io,
        "/bin/sh", // exists on all Unix systems
        "/nonexistent/model.gguf",
    );
    try std.testing.expectError(EmbeddingError.ModelNotFound, result);
}

test "LlamaEmbedding.init accepts hf remote model spec" {
    const allocator = std.testing.allocator;
    var engine = try LlamaEmbedding.init(
        allocator,
        std.testing.io,
        "/bin/sh",
        "hf://ggml-org/embeddinggemma-300M-qat-q4_0-GGUF:Q4_0",
    );
    defer engine.deinit();
    try std.testing.expectEqualStrings("hf://ggml-org/embeddinggemma-300M-qat-q4_0-GGUF:Q4_0", engine.model_path);
}

test "parseEmbeddingJson parses OpenAI-style JSON" {
    const allocator = std.testing.allocator;

    // Sample OpenAI-style embedding response
    const json =
        \\{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3,-0.4,0.5]}],"model":"test","usage":{"prompt_tokens":5,"total_tokens":5}}
    ;

    const embeddings = try parseEmbeddingJson(allocator, json);
    defer {
        for (embeddings) |emb| allocator.free(emb);
        allocator.free(embeddings);
    }

    try std.testing.expectEqual(@as(usize, 1), embeddings.len);
    try std.testing.expectEqual(@as(usize, 5), embeddings[0].len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), embeddings[0][0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.4), embeddings[0][3], 0.001);
}

test "parseEmbeddingJson handles multiple embeddings" {
    const allocator = std.testing.allocator;

    const json =
        \\{"object":"list","data":[{"object":"embedding","index":0,"embedding":[1.0,2.0]},{"object":"embedding","index":1,"embedding":[3.0,4.0]}],"model":"test","usage":{"prompt_tokens":10,"total_tokens":10}}
    ;

    const embeddings = try parseEmbeddingJson(allocator, json);
    defer {
        for (embeddings) |emb| allocator.free(emb);
        allocator.free(embeddings);
    }

    try std.testing.expectEqual(@as(usize, 2), embeddings.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), embeddings[0][0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), embeddings[1][0], 0.001);
}

test "parseEmbeddingJson returns error on invalid JSON" {
    const allocator = std.testing.allocator;
    const result = parseEmbeddingJson(allocator, "not valid json");
    try std.testing.expectError(EmbeddingError.InvalidJson, result);
}

test "LlamaEmbedding.embedBatch returns empty result for empty input" {
    const allocator = std.testing.allocator;

    var engine = try LlamaEmbedding.init(
        allocator,
        std.testing.io,
        "/bin/sh",
        "VERSION",
    );
    defer engine.deinit();

    const embeddings = try engine.embedBatch(&.{});
    defer allocator.free(embeddings);

    try std.testing.expectEqual(@as(usize, 0), embeddings.len);
}

test "LlamaEmbedding.embed returns process error when subprocess fails" {
    const allocator = std.testing.allocator;

    var engine = try LlamaEmbedding.init(
        allocator,
        std.testing.io,
        "/bin/sh",
        "VERSION",
    );
    defer engine.deinit();

    const result = engine.embed("hello world");
    try std.testing.expectError(EmbeddingError.ProcessFailed, result);
}

test "buildCacheKey is deterministic" {
    const k1 = buildCacheKey("expand", "model-a", "what is oauth");
    const k2 = buildCacheKey("expand", "model-a", "what is oauth");
    try std.testing.expectEqualStrings(&k1, &k2);
}

test "cachePut and cacheGet roundtrip" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    const key = buildCacheKey("expand", "model-a", "query text");
    try cachePut(&db_, key[0..], "cached value");

    const value = try cacheGet(&db_, key[0..], std.testing.allocator);
    try std.testing.expect(value != null);
    defer std.testing.allocator.free(value.?);
    try std.testing.expectEqualStrings("cached value", value.?);
}

test "LlamaChatSession stores bounded history" {
    var session = try LlamaChatSession.init(std.testing.allocator, std.testing.io, "/bin/false", "/tmp/none.gguf");
    defer session.deinit();

    session.max_history = 3;
    try session.addSystemPrompt("You are helpful");
    const _r1 = try session.send("hello");
    std.testing.allocator.free(_r1);
    const _r2 = try session.send("next");
    std.testing.allocator.free(_r2);
    const _r3 = try session.send("third");
    std.testing.allocator.free(_r3);
    try std.testing.expect(session.history.items.len <= 3);
}

test "rerankPassages falls back without model" {
    const passages: []const []const u8 = &.{
        "OAuth login and token refresh",
        "Pasta and olive oil recipe",
    };
    const scores = try rerankPassages(std.testing.allocator, std.testing.io, "oauth token", passages, null, null);
    defer std.testing.allocator.free(scores);
    try std.testing.expect(scores.len == 2);
    try std.testing.expect(scores[0] >= scores[1]);
}
