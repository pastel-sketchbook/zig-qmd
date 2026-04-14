const std = @import("std");

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
    ContextParamsFailed,
};

pub const LlamaCpp = struct {
    ctx: ?*anyopaque = null,
    model: ?*anyopaque = null,
    embedding_dim: usize = 384,
    loaded: bool = false,
    model_path: []const u8 = "",

    pub fn init(model_path: []const u8, _: std.mem.Allocator) LlamaError!LlamaCpp {
        var self = LlamaCpp{};
        self.model_path = model_path;
        self.loaded = true;
        self.embedding_dim = 384;
        return self;
    }

    pub fn deinit(self: *LlamaCpp) void {
        self.loaded = false;
        self.ctx = null;
        self.model = null;
    }

    pub fn embed(self: *LlamaCpp, text: []const u8, alloc: std.mem.Allocator) LlamaError![]f32 {
        if (!self.loaded) return LlamaError.ModelNotLoaded;
        const embedding = alloc.alloc(f32, self.embedding_dim) catch return LlamaError.EmbedFailed;
        for (0..self.embedding_dim, embedding) |i, *slot| {
            var h: u32 = 2166136261;
            for (text) |c| h +%= @as(u32, c) * 16777619;
            h +%= @as(u32, @intCast(i)) *% 16777619;
            slot.* = @as(f32, @floatFromInt(h & 0xFFFF)) / 65535.0 - 0.5;
        }
        var norm: f32 = 0;
        for (embedding) |x| norm += x * x;
        norm = @sqrt(norm);
        if (norm > 0) {
            for (embedding) |*x| x.* /= norm;
        }
        return embedding;
    }

    pub fn embedBatch(self: *LlamaCpp, texts: [][]const u8, alloc: std.mem.Allocator) LlamaError![][]f32 {
        var embeddings = try alloc.alloc([]f32, texts.len);
        errdefer for (embeddings) |emb| alloc.free(emb);
        for (texts, 0..) |text, i| {
            embeddings[i] = try self.embed(text, alloc);
        }
        return embeddings;
    }

    pub fn embedDim(self: *const LlamaCpp) usize {
        return self.embedding_dim;
    }

    pub fn isLoaded(self: *const LlamaCpp) bool {
        return self.loaded;
    }
};

pub fn formatTextForEmbedding(text: []const u8, is_query: bool) []u8 {
    _ = is_query;
    if (text.len > 1000) return text[0..1000];
    return text;
}

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

pub fn euclideanDistance(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return 0;
    var dist: f32 = 0;
    for (0..a.len) |i| {
        const diff = a[i] - b[i];
        dist += diff * diff;
    }
    return @sqrt(dist);
}

pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return 0;
    var sum: f32 = 0;
    for (0..a.len) |i| sum += a[i] * b[i];
    return sum;
}

pub const LlmCache = struct {
    entries: std.AutoHashMap([]const u8, CachedResult),

    pub fn init() LlmCache {
        return .{ .entries = std.AutoHashMap([]const u8, CachedResult).init(std.heap.page_allocator) };
    }

    pub fn deinit(self: *LlmCache) void {
        var it = self.entries.iterator();
        while (it.next()) |entry| std.heap.page_allocator.free(entry.key_ptr.*);
        self.entries.deinit();
    }

    pub fn get(self: *const LlmCache, key: []const u8) ?CachedResult {
        return self.entries.get(key);
    }

    pub fn put(self: *LlmCache, key: []const u8, value: CachedResult) !void {
        const key_copy = try std.heap.page_allocator.dupe(u8, key);
        try self.entries.put(key_copy, value);
    }

    pub fn clear(self: *LlmCache) void {
        var it = self.entries.iterator();
        while (it.next()) |entry| std.heap.page_allocator.free(entry.key_ptr.*);
        self.entries.clearRetainingCapacity();
    }
};

pub const CachedResult = struct {
    response: []const u8,
    created_at: i64,
};

pub const SimpleResult = struct {
    id: i64,
    score: f64,
};

pub fn rerank(results: []const SimpleResult, query: []const u8) ![]SimpleResult {
    _ = query;
    return results;
}

pub fn expandQuery(query: []const u8) ![]const u8 {
    var expanded = try std.ArrayList(u8).initCapacity(std.heap.page_allocator, query.len);
    errdefer expanded.deinit(std.heap.page_allocator);
    try expanded.appendSlice(std.heap.page_allocator, query);
    if (std.mem.indexOf(u8, query, "?") != null) try expanded.appendSlice(std.heap.page_allocator, " explain clarify");
    if (std.mem.indexOf(u8, query, "how") != null) try expanded.appendSlice(std.heap.page_allocator, " method way steps procedure");
    if (std.mem.indexOf(u8, query, "what") != null) try expanded.appendSlice(std.heap.page_allocator, " definition meaning information");
    if (std.mem.indexOf(u8, query, "why") != null) try expanded.appendSlice(std.heap.page_allocator, " reason cause explanation");
    if (std.mem.indexOf(u8, query, "where") != null) try expanded.appendSlice(std.heap.page_allocator, " location place");
    if (std.mem.indexOf(u8, query, "when") != null) try expanded.appendSlice(std.heap.page_allocator, " time date schedule");
    return expanded.toOwnedSlice(std.heap.page_allocator);
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
    const expanded = try expandQuery("how to login");
    try std.testing.expect(std.mem.indexOf(u8, expanded, "method").? > 0);
}

test "LlmCache stores and retrieves" {
    var cache = LlmCache.init();
    defer cache.deinit();
    try cache.put("key", .{ .response = "val", .created_at = 1 });
    try std.testing.expect(cache.get("key") != null);
}
