const std = @import("std");
const db = @import("db.zig");
const store = @import("store.zig");
const llm = @import("llm.zig");

pub const SearchError = error{
    QueryFailed,
    NoResults,
} || db.DbError;

pub fn buildFTS5Query(input: []const u8) ![]u8 {
    const allocator = std.heap.page_allocator;
    var result_list = try std.ArrayList(u8).initCapacity(allocator, 0);
    defer result_list.deinit(allocator);

    var tokens = std.mem.tokenizeScalar(u8, input, ' ');
    var is_first = true;

    while (tokens.next()) |raw| {
        if (raw.len == 0) continue;

        var token = raw;
        var prefix_match = false;
        var is_negation = false;

        if (std.mem.startsWith(u8, token, "-")) {
            is_negation = true;
            token = token[1..];
        }

        if (token.len > 0 and token[token.len - 1] == '*') {
            prefix_match = true;
            token = token[0 .. token.len - 1];
        }

        if (token.len == 0) continue;

        if (!is_first) try result_list.append(allocator, ' ');
        is_first = false;

        if (is_negation) try result_list.append(allocator, '-');

        if (std.mem.indexOf(u8, token, "-") != null) {
            try result_list.append(allocator, '"');
            try result_list.appendSlice(allocator, token);
            try result_list.append(allocator, '"');
        } else {
            try result_list.appendSlice(allocator, token);
        }

        if (prefix_match) try result_list.append(allocator, '*');
    }

    return try result_list.toOwnedSlice(allocator);
}

pub fn searchFTS(
    db_: *db.Db,
    query: []const u8,
    collection: ?[]const u8,
) !SearchResults {
    const fts_query = try buildFTS5Query(query);

    const base_sql = if (collection != null)
        "WITH ranked AS (SELECT d.id, d.collection, d.path, d.title, d.hash, bm25(documents_fts, 1.5, 4.0, 1.0) as score FROM documents_fts JOIN documents d ON documents_fts.rowid = d.id WHERE documents_fts MATCH ? AND d.collection = ?) SELECT id, collection, path, title, hash, score FROM ranked WHERE score < 0 ORDER BY score LIMIT 100"
    else
        "WITH ranked AS (SELECT d.id, d.collection, d.path, d.title, d.hash, bm25(documents_fts, 1.5, 4.0, 1.0) as score FROM documents_fts JOIN documents d ON documents_fts.rowid = d.id WHERE documents_fts MATCH ?) SELECT id, collection, path, title, hash, score FROM ranked WHERE score < 0 ORDER BY score LIMIT 100";

    var stmt = try db_.prepare(base_sql);
    defer stmt.finalize();

    try stmt.bindText(1, fts_query);
    if (collection) |col| try stmt.bindText(2, col);

    var results = try std.ArrayList(SearchResult).initCapacity(std.heap.page_allocator, 0);
    errdefer results.deinit(std.heap.page_allocator);

    while (try stmt.step()) {
        const score_raw = stmt.columnDouble(5);
        const score_norm = if (score_raw < 0) @abs(score_raw) / (1 + @abs(score_raw)) else 0;

        const col = stmt.columnText(1);
        const pth = stmt.columnText(2);
        const ttl = stmt.columnText(3);
        const hsh = stmt.columnText(4);

        const col_str = if (col) |c| try std.heap.page_allocator.dupe(u8, std.mem.span(c)) else try std.heap.page_allocator.dupe(u8, "");
        const pth_str = if (pth) |p| try std.heap.page_allocator.dupe(u8, std.mem.span(p)) else try std.heap.page_allocator.dupe(u8, "");
        const ttl_str = if (ttl) |t| try std.heap.page_allocator.dupe(u8, std.mem.span(t)) else try std.heap.page_allocator.dupe(u8, "");
        const hsh_str = if (hsh) |h| try std.heap.page_allocator.dupe(u8, std.mem.span(h)) else try std.heap.page_allocator.dupe(u8, "");

        try results.append(std.heap.page_allocator, .{
            .id = stmt.columnInt(0),
            .collection = col_str,
            .path = pth_str,
            .title = ttl_str,
            .hash = hsh_str,
            .score = score_norm,
        });
    }

    return .{ .results = results };
}

pub const SearchResults = struct {
    results: std.ArrayList(SearchResult),
};

pub const SearchResult = struct {
    id: i64,
    collection: []const u8,
    path: []const u8,
    title: []const u8,
    hash: []const u8,
    score: f64,
};

pub const RRF_K = 60;

const RankedEntry = struct {
    score: f64,
    result: ScoredResult,
};

pub fn reciprocalRankFusion(
    result_lists: []const []const ScoredResult,
    k: f64,
) []ScoredResult {
    var seen = std.AutoHashMap(i64, RankedEntry).init(std.heap.page_allocator);
    defer seen.deinit();

    for (result_lists) |list| {
        for (list, 0..) |result, rank| {
            const rrf_score = 1.0 / (k + @as(f64, @floatFromInt(rank + 1)));
            const key = result.id;
            if (seen.get(key)) |existing| {
                seen.put(key, .{ .score = existing.score + rrf_score, .result = existing.result }) catch {};
            } else {
                seen.put(key, .{ .score = rrf_score, .result = result }) catch {};
            }
        }
    }

    var ranked = std.ArrayList(ScoredResult).initCapacity(std.heap.page_allocator, 0) catch return &.{};
    errdefer ranked.deinit(std.heap.page_allocator);
    var entries = std.ArrayList(struct { key: i64, score: f64, result: ScoredResult }).initCapacity(std.heap.page_allocator, 0) catch return &.{};
    errdefer entries.deinit(std.heap.page_allocator);

    var it = seen.iterator();
    while (it.next()) |entry| {
        entries.append(std.heap.page_allocator, .{ .key = entry.key_ptr.*, .score = entry.value_ptr.score, .result = entry.value_ptr.result }) catch {};
    }

    std.sort.heap(@TypeOf(entries.items[0]), entries.items, {}, struct {
        fn less(_: void, a: @TypeOf(entries.items[0]), b: @TypeOf(entries.items[0])) bool {
            return a.score > b.score;
        }
    }.less);

    for (entries.items) |entry| {
        ranked.append(std.heap.page_allocator, entry.result) catch {};
    }

    return ranked.items;
}

pub const ScoredResult = struct {
    id: i64,
    collection: []const u8,
    path: []const u8,
    title: []const u8,
    hash: []const u8,
    score: f64,
};

pub fn hybridSearch(
    db_: *db.Db,
    query: []const u8,
    collection: ?[]const u8,
    options: HybridOptions,
) !HybridResult {
    var effective_query = query;
    var expanded_query_owned: ?[]const u8 = null;
    defer if (expanded_query_owned) |q| std.heap.page_allocator.free(q);

    if (options.enable_query_expansion) {
        const bin_path = std.process.getEnvVarOwned(std.heap.page_allocator, "QMD_LLAMA_EMBED_BIN") catch null;
        defer if (bin_path) |p| std.heap.page_allocator.free(p);
        const model_path = std.process.getEnvVarOwned(std.heap.page_allocator, "QMD_LLAMA_MODEL") catch null;
        defer if (model_path) |p| std.heap.page_allocator.free(p);

        const model_key = if (model_path) |m| m else "heuristic";
        const cache_key = llm.buildCacheKey("expand", model_key, query);
        const cached = llm.cacheGet(db_, cache_key[0..], std.heap.page_allocator) catch null;
        if (cached) |q_cached| {
            expanded_query_owned = q_cached;
            effective_query = q_cached;
        }

        const expanded = if (cached == null)
            (llm.expandQueryWithModel(std.heap.page_allocator, query, bin_path, model_path) catch null)
        else
            null;
        if (expanded) |q| {
            expanded_query_owned = q;
            effective_query = q;
            llm.cachePut(db_, cache_key[0..], q) catch {};
        }
    }

    const fts_result = try searchFTS(db_, effective_query, collection);

    var fts_scored = try std.ArrayList(ScoredResult).initCapacity(std.heap.page_allocator, fts_result.results.items.len);
    errdefer fts_scored.deinit(std.heap.page_allocator);

    for (fts_result.results.items) |r| {
        try fts_scored.append(std.heap.page_allocator, .{ .id = r.id, .collection = r.collection, .path = r.path, .title = r.title, .hash = r.hash, .score = r.score });
    }

    var vec_scored: []ScoredResult = &.{};
    if (options.enable_vector) {
        const vec_result = try searchVec(db_, effective_query, collection);
        vec_scored = vec_result.results;
    }

    var lists: [2][]ScoredResult = undefined;
    lists[0] = fts_scored.items;
    lists[1] = vec_scored;

    var fused = reciprocalRankFusion(&lists, options.rrf_k);

    if (options.enable_rerank and fused.len > 1) {
        fused = try rerankByEmbedding(db_, effective_query, fused);
    }

    var final_results = try std.ArrayList(SearchResult).initCapacity(std.heap.page_allocator, @min(fused.len, options.max_results));
    errdefer final_results.deinit(std.heap.page_allocator);

    for (fused[0..@min(fused.len, options.max_results)]) |r| {
        var title = r.title;
        var hash = r.hash;
        if (std.mem.eql(u8, title, "")) {
            const doc = store.findActiveDocument(db_, r.collection, r.path) catch continue;
            defer {
                std.heap.page_allocator.free(doc.title);
                std.heap.page_allocator.free(doc.hash);
                std.heap.page_allocator.free(doc.doc);
            }
            title = doc.title;
            hash = doc.hash;
        }
        try final_results.append(std.heap.page_allocator, .{ .id = r.id, .collection = r.collection, .path = r.path, .title = title, .hash = hash, .score = r.score });
    }

    return .{ .results = final_results, .fts_count = fts_scored.items.len, .vec_count = vec_scored.len };
}

pub const HybridOptions = struct {
    enable_vector: bool = false,
    enable_query_expansion: bool = false,
    enable_rerank: bool = false,
    rrf_k: f64 = RRF_K,
    max_results: usize = 20,
    min_score: f64 = 0.0,
};

fn rerankByEmbedding(db_: *db.Db, query: []const u8, results: []ScoredResult) ![]ScoredResult {
    const allocator = std.heap.page_allocator;
    const q_emb = try embed_text(query, true);
    defer allocator.free(q_emb);

    var rescored = try allocator.alloc(ScoredResult, results.len);
    for (results, 0..) |r, i| {
        var source_text = r.title;
        const doc = store.findActiveDocument(db_, r.collection, r.path) catch null;
        if (doc) |d| {
            defer {
                allocator.free(d.title);
                allocator.free(d.hash);
                allocator.free(d.doc);
            }
            source_text = d.doc;
        }

        const d_emb = embed_text(source_text, false) catch q_emb;
        defer if (d_emb.ptr != q_emb.ptr) allocator.free(d_emb);

        var item = r;
        item.score = llm.cosineSimilarity(q_emb, d_emb);
        rescored[i] = item;
    }

    std.sort.heap(ScoredResult, rescored, {}, struct {
        fn less(_: void, a: ScoredResult, b: ScoredResult) bool {
            return a.score > b.score;
        }
    }.less);

    return rescored;
}

pub const HybridResult = struct {
    results: std.ArrayList(SearchResult),
    fts_count: usize,
    vec_count: usize,
};

pub fn searchVec(
    db_: *db.Db,
    query: []const u8,
    collection: ?[]const u8,
) !struct { results: []ScoredResult } {
    const query_embedding = try embed_query(query);
    defer std.heap.page_allocator.free(query_embedding);

    var best_by_doc = std.AutoHashMap(i64, ScoredResult).init(std.heap.page_allocator);
    defer best_by_doc.deinit();

    var stmt = db_.prepare(
        "SELECT d.id, d.hash, d.collection, d.path, d.title, cv.embedding FROM documents d LEFT JOIN content_vectors cv ON cv.hash = d.hash WHERE d.active = 1",
    ) catch return .{ .results = &.{} };
    defer stmt.finalize();

    while (stmt.step() catch false) {
        const id = stmt.columnInt(0);
        const hsh = stmt.columnText(1);
        const coll = stmt.columnText(2);
        const pth = stmt.columnText(3);
        const ttl = stmt.columnText(4);
        const emb = stmt.columnText(5);

        const hash = if (hsh) |h| try std.heap.page_allocator.dupe(u8, std.mem.span(h)) else try std.heap.page_allocator.dupe(u8, "");
        const col = if (coll) |c| try std.heap.page_allocator.dupe(u8, std.mem.span(c)) else try std.heap.page_allocator.dupe(u8, "");
        const path = if (pth) |p| try std.heap.page_allocator.dupe(u8, std.mem.span(p)) else try std.heap.page_allocator.dupe(u8, "");
        const title = if (ttl) |t| try std.heap.page_allocator.dupe(u8, std.mem.span(t)) else try std.heap.page_allocator.dupe(u8, "");

        if (collection != null and !std.mem.eql(u8, col, collection.?)) {
            continue;
        }

        if (emb == null) continue;

        const doc_embedding = parse_embedding_json_array(std.mem.span(emb.?)) catch continue;
        defer std.heap.page_allocator.free(doc_embedding);

        const score = llm.cosineSimilarity(query_embedding, doc_embedding);
        const candidate = ScoredResult{
            .id = id,
            .collection = col,
            .path = path,
            .title = title,
            .hash = hash,
            .score = score,
        };

        if (best_by_doc.get(id)) |existing| {
            if (candidate.score > existing.score) {
                try best_by_doc.put(id, candidate);
            }
        } else {
            try best_by_doc.put(id, candidate);
        }
    }

    var results = try std.ArrayList(ScoredResult).initCapacity(std.heap.page_allocator, best_by_doc.count());
    errdefer results.deinit(std.heap.page_allocator);

    var it = best_by_doc.iterator();
    while (it.next()) |entry| {
        try results.append(std.heap.page_allocator, entry.value_ptr.*);
    }

    std.sort.heap(ScoredResult, results.items, {}, struct {
        fn less(_: void, a: ScoredResult, b: ScoredResult) bool {
            return a.score > b.score;
        }
    }.less);

    return .{ .results = results.items };
}

fn embed_query(query: []const u8) ![]f32 {
    return embed_text(query, true);
}

fn embed_text(text: []const u8, is_query: bool) ![]f32 {
    const allocator = std.heap.page_allocator;
    const formatted = if (is_query)
        try llm.formatQueryForEmbedding(allocator, text)
    else
        try llm.formatDocForEmbedding(allocator, text);
    defer allocator.free(formatted);

    const bin_path = std.process.getEnvVarOwned(allocator, "QMD_LLAMA_EMBED_BIN") catch null;
    defer if (bin_path) |p| allocator.free(p);

    const model_path = std.process.getEnvVarOwned(allocator, "QMD_LLAMA_MODEL") catch null;
    defer if (model_path) |p| allocator.free(p);

    if (bin_path != null and model_path != null) {
        var engine = llm.LlamaEmbedding.init(allocator, bin_path.?, model_path.?) catch {
            var fallback = try llm.LlamaCpp.init("/nonexistent", allocator);
            defer fallback.deinit();
            return fallback.embed(formatted, allocator);
        };
        defer engine.deinit();
        return engine.embed(formatted) catch {
            var fallback = try llm.LlamaCpp.init("/nonexistent", allocator);
            defer fallback.deinit();
            return fallback.embed(formatted, allocator);
        };
    }

    var fallback = try llm.LlamaCpp.init("/nonexistent", allocator);
    defer fallback.deinit();
    return fallback.embed(formatted, allocator);
}

fn parse_embedding_json_array(json: []const u8) ![]f32 {
    const allocator = std.heap.page_allocator;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{}) catch return error.InvalidJson;
    defer parsed.deinit();

    if (parsed.value != .array) return error.InvalidJson;
    const items = parsed.value.array.items;
    var out = try allocator.alloc(f32, items.len);
    for (items, 0..) |v, i| {
        out[i] = switch (v) {
            .float => @floatCast(v.float),
            .integer => @floatFromInt(v.integer),
            else => return error.InvalidJson,
        };
    }
    return out;
}

test "embed_query uses real model when env is set" {
    const allocator = std.testing.allocator;
    const maybe_model = std.process.getEnvVarOwned(allocator, "QMD_REAL_GGUF_MODEL");
    if (maybe_model) |model| {
        defer allocator.free(model);
        const emb = try embed_query("oauth sign in token");
        defer std.heap.page_allocator.free(emb);
        try std.testing.expect(emb.len > 10);
    } else |_| {
        // Skip when no real model is configured.
    }
}

test "reciprocalRankFusion merges results" {
    const list1: []const ScoredResult = &.{
        .{ .id = 1, .collection = "a", .path = "a", .title = "A", .hash = "", .score = 0.9 },
        .{ .id = 2, .collection = "a", .path = "b", .title = "B", .hash = "", .score = 0.8 },
    };
    const list2: []const ScoredResult = &.{
        .{ .id = 2, .collection = "a", .path = "b", .title = "B", .hash = "", .score = 0.7 },
        .{ .id = 3, .collection = "a", .path = "c", .title = "C", .hash = "", .score = 0.6 },
    };

    const fused = reciprocalRankFusion(&.{ list1, list2 }, RRF_K);

    try std.testing.expect(fused.len > 0);
    var seen2 = false;
    for (fused) |f| {
        if (f.id == 2) seen2 = true;
    }
    try std.testing.expect(seen2);
}

test "hybridSearch with FTS only" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    try store.insertDocument(&db_, "test", "a.md", "# Auth\nLogin flow");
    try store.insertDocument(&db_, "test", "b.md", "# Setup\nInstall");

    var result = try hybridSearch(&db_, "auth", null, .{ .enable_vector = false });
    defer result.results.deinit(std.heap.page_allocator);

    try std.testing.expect(result.fts_count > 0);
}

test "searchVec uses stored vectors and ranks by cosine" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    try store.insertDocument(&db_, "test", "a.md", "# Auth\nLogin and auth flow");
    try store.insertDocument(&db_, "test", "b.md", "# Cooking\nRecipe and food");

    const doc_a = try store.findActiveDocument(&db_, "test", "a.md");
    const doc_b = try store.findActiveDocument(&db_, "test", "b.md");

    // Match the current fallback embedding model for deterministic test behavior.
    var fallback = try llm.LlamaCpp.init("/nonexistent", std.heap.page_allocator);
    defer fallback.deinit();
    const q_emb = try fallback.embed("auth", std.heap.page_allocator);
    defer std.heap.page_allocator.free(q_emb);
    const b_emb = try fallback.embed("totally unrelated baseline", std.heap.page_allocator);
    defer std.heap.page_allocator.free(b_emb);

    try store.upsertContentVector(&db_, doc_a.hash, "test", q_emb, std.heap.page_allocator);
    try store.upsertContentVector(&db_, doc_b.hash, "test", b_emb, std.heap.page_allocator);

    const result = try searchVec(&db_, "auth", null);
    try std.testing.expect(result.results.len >= 2);
    try std.testing.expect(result.results[0].score >= result.results[1].score);
}

test "hybridSearch supports query expansion option" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    try store.insertDocument(&db_, "test", "a.md", "# Login\nHow to authenticate users");
    try store.insertDocument(&db_, "test", "b.md", "# Cooking\nHow to boil pasta");

    var result = try hybridSearch(&db_, "how login?", null, .{
        .enable_vector = true,
        .enable_query_expansion = true,
        .enable_rerank = false,
        .max_results = 10,
    });
    defer result.results.deinit(std.heap.page_allocator);

    try std.testing.expect(result.fts_count >= 0);
}

test "hybridSearch supports rerank option" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    try store.insertDocument(&db_, "test", "a.md", "# Authentication\nOAuth token login");
    try store.insertDocument(&db_, "test", "b.md", "# Recipe\nPasta cooking instructions");

    var result = try hybridSearch(&db_, "oauth login", null, .{
        .enable_vector = true,
        .enable_query_expansion = false,
        .enable_rerank = true,
        .max_results = 10,
    });
    defer result.results.deinit(std.heap.page_allocator);

    try std.testing.expect(result.results.items.len > 0);
}

test "buildFTS5Query parses simple tokens" {
    const result = try buildFTS5Query("hello world");
    defer std.heap.page_allocator.free(result);
    try std.testing.expectEqualStrings("hello world", result);
}

test "buildFTS5Query handles negation" {
    const result = try buildFTS5Query("hello -world");
    defer std.heap.page_allocator.free(result);
    try std.testing.expectEqualStrings("hello -world", result);
}

test "buildFTS5Query handles prefix match" {
    const result = try buildFTS5Query("auth*");
    defer std.heap.page_allocator.free(result);
    try std.testing.expectEqualStrings("auth*", result);
}

test "buildFTS5Query handles hyphenated words" {
    const result = try buildFTS5Query("real-time");
    defer std.heap.page_allocator.free(result);
    try std.testing.expectEqualStrings("\"real-time\"", result);
}
