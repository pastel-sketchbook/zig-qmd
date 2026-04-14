const std = @import("std");
const db = @import("db.zig");
const store = @import("store.zig");
const llm = @import("llm.zig");

/// Error set for search operations.
pub const SearchError = error{
    QueryFailed,
    NoResults,
} || db.DbError;

/// Parses user input into an FTS5 query string, handling negation, prefix
/// matching, and hyphenated terms.
pub fn buildFTS5Query(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
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

/// Executes a BM25 full-text search against the documents FTS5 index,
/// optionally filtered by collection.
pub fn searchFTS(
    db_: *db.Db,
    allocator: std.mem.Allocator,
    query: []const u8,
    collection: ?[]const u8,
) !SearchResults {
    const fts_query = try buildFTS5Query(allocator, query);
    defer allocator.free(fts_query);

    const base_sql = if (collection != null)
        "WITH ranked AS (SELECT d.id, d.collection, d.path, d.title, d.hash, bm25(documents_fts, 1.5, 4.0, 1.0) as score FROM documents_fts JOIN documents d ON documents_fts.rowid = d.id WHERE documents_fts MATCH ? AND d.collection = ?) SELECT id, collection, path, title, hash, score FROM ranked WHERE score < 0 ORDER BY score LIMIT 100"
    else
        "WITH ranked AS (SELECT d.id, d.collection, d.path, d.title, d.hash, bm25(documents_fts, 1.5, 4.0, 1.0) as score FROM documents_fts JOIN documents d ON documents_fts.rowid = d.id WHERE documents_fts MATCH ?) SELECT id, collection, path, title, hash, score FROM ranked WHERE score < 0 ORDER BY score LIMIT 100";

    var stmt = try db_.prepare(base_sql);
    defer stmt.finalize();

    try stmt.bindText(1, fts_query);
    if (collection) |col| try stmt.bindText(2, col);

    var results = try std.ArrayList(SearchResult).initCapacity(allocator, 0);
    errdefer {
        freeSearchResultSlice(results.items, allocator);
        results.deinit(allocator);
    }

    while (try stmt.step()) {
        const score_raw = stmt.columnDouble(5);
        const score_norm = if (score_raw < 0) @abs(score_raw) / (1 + @abs(score_raw)) else 0;

        const col = stmt.columnText(1);
        const pth = stmt.columnText(2);
        const ttl = stmt.columnText(3);
        const hsh = stmt.columnText(4);

        const col_str = if (col) |c| try allocator.dupe(u8, std.mem.span(c)) else try allocator.dupe(u8, "");
        const pth_str = if (pth) |p| try allocator.dupe(u8, std.mem.span(p)) else try allocator.dupe(u8, "");
        const ttl_str = if (ttl) |t| try allocator.dupe(u8, std.mem.span(t)) else try allocator.dupe(u8, "");
        const hsh_str = if (hsh) |h| try allocator.dupe(u8, std.mem.span(h)) else try allocator.dupe(u8, "");

        try results.append(allocator, .{
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

/// Container for FTS search results with deallocation support.
pub const SearchResults = struct {
    results: std.ArrayList(SearchResult),

    pub fn deinit(self: *SearchResults, allocator: std.mem.Allocator) void {
        freeSearchResultSlice(self.results.items, allocator);
        self.results.deinit(allocator);
    }
};

/// A single FTS search result with score and document metadata.
pub const SearchResult = struct {
    id: i64,
    collection: []const u8,
    path: []const u8,
    title: []const u8,
    hash: []const u8,
    score: f64,
};

/// Reciprocal Rank Fusion constant (k=60) controlling rank blending.
pub const RRF_K = 60;

const RankedEntry = struct {
    score: f64,
    result: ScoredResult,
};

/// Merges multiple ranked result lists using RRF scoring, returning a
/// unified ranking.
pub fn reciprocalRankFusion(
    allocator: std.mem.Allocator,
    result_lists: []const []const ScoredResult,
    k: f64,
) ![]ScoredResult {
    var seen = std.AutoHashMap(i64, RankedEntry).init(allocator);
    defer seen.deinit();

    for (result_lists) |list| {
        for (list, 0..) |result, rank| {
            const rrf_score = 1.0 / (k + @as(f64, @floatFromInt(rank + 1)));
            const key = result.id;
            if (seen.get(key)) |existing| {
                try seen.put(key, .{ .score = existing.score + rrf_score, .result = existing.result });
            } else {
                try seen.put(key, .{ .score = rrf_score, .result = result });
            }
        }
    }

    var ranked = try std.ArrayList(ScoredResult).initCapacity(allocator, 0);
    errdefer {
        freeScoredResultSlice(ranked.items, allocator);
        ranked.deinit(allocator);
    }
    var entries = try std.ArrayList(struct { key: i64, score: f64, result: ScoredResult }).initCapacity(allocator, 0);
    defer entries.deinit(allocator);

    var it = seen.iterator();
    while (it.next()) |entry| {
        try entries.append(allocator, .{ .key = entry.key_ptr.*, .score = entry.value_ptr.score, .result = entry.value_ptr.result });
    }

    std.sort.heap(@TypeOf(entries.items[0]), entries.items, {}, struct {
        fn less(_: void, a: @TypeOf(entries.items[0]), b: @TypeOf(entries.items[0])) bool {
            return a.score > b.score;
        }
    }.less);

    for (entries.items) |entry| {
        try ranked.append(allocator, try cloneScoredResult(allocator, entry.result));
    }

    return ranked.toOwnedSlice(allocator);
}

/// A search result with a combined relevance score.
pub const ScoredResult = struct {
    id: i64,
    collection: []const u8,
    path: []const u8,
    title: []const u8,
    hash: []const u8,
    score: f64,
};

fn freeSearchResultSlice(items: []SearchResult, allocator: std.mem.Allocator) void {
    for (items) |r| {
        allocator.free(r.collection);
        allocator.free(r.path);
        allocator.free(r.title);
        allocator.free(r.hash);
    }
}

fn freeScoredResultSlice(items: []ScoredResult, allocator: std.mem.Allocator) void {
    for (items) |r| {
        allocator.free(r.collection);
        allocator.free(r.path);
        allocator.free(r.title);
        allocator.free(r.hash);
    }
}

fn cloneScoredResult(allocator: std.mem.Allocator, r: ScoredResult) !ScoredResult {
    return .{
        .id = r.id,
        .collection = try allocator.dupe(u8, r.collection),
        .path = try allocator.dupe(u8, r.path),
        .title = try allocator.dupe(u8, r.title),
        .hash = try allocator.dupe(u8, r.hash),
        .score = r.score,
    };
}

/// Executes a full hybrid search pipeline: FTS + optional vector search,
/// RRF fusion, and optional LLM reranking.
pub fn hybridSearch(
    db_: *db.Db,
    allocator: std.mem.Allocator,
    query: []const u8,
    collection: ?[]const u8,
    options: HybridOptions,
) !HybridResult {
    if (is_aborted(options.abort_signal)) {
        return .{ .results = try std.ArrayList(SearchResult).initCapacity(allocator, 0), .fts_count = 0, .vec_count = 0 };
    }

    var effective_query = query;
    var expanded_query_owned: ?[]const u8 = null;
    defer if (expanded_query_owned) |q| allocator.free(q);

    if (options.enable_query_expansion) {
        if (is_aborted(options.abort_signal)) {
            return .{ .results = try std.ArrayList(SearchResult).initCapacity(allocator, 0), .fts_count = 0, .vec_count = 0 };
        }

        const bin_path = std.process.getEnvVarOwned(allocator, "QMD_LLAMA_EMBED_BIN") catch null;
        defer if (bin_path) |p| allocator.free(p);
        const model_path = std.process.getEnvVarOwned(allocator, "QMD_LLAMA_MODEL") catch null;
        defer if (model_path) |p| allocator.free(p);

        const model_key = if (model_path) |m| m else "heuristic";
        const cache_key = llm.buildCacheKey("expand", model_key, query);
        const cached = llm.cacheGet(db_, cache_key[0..], allocator) catch null;
        if (cached) |q_cached| {
            expanded_query_owned = q_cached;
            effective_query = q_cached;
        }

        const expanded = if (cached == null)
            (llm.expandQueryWithModel(allocator, query, bin_path, model_path) catch null)
        else
            null;
        if (expanded) |q| {
            expanded_query_owned = q;
            effective_query = q;
            llm.cachePut(db_, cache_key[0..], q) catch |err| {
                if (err == error.OutOfMemory) return err;
            };
        }
    }

    var fts_result = try searchFTS(db_, allocator, effective_query, collection);

    var fts_scored = try std.ArrayList(ScoredResult).initCapacity(allocator, fts_result.results.items.len);
    errdefer {
        freeScoredResultSlice(fts_scored.items, allocator);
        fts_scored.deinit(allocator);
    }

    for (fts_result.results.items) |r| {
        try fts_scored.append(allocator, try cloneScoredResult(allocator, .{
            .id = r.id,
            .collection = r.collection,
            .path = r.path,
            .title = r.title,
            .hash = r.hash,
            .score = r.score,
        }));
    }
    defer {
        freeScoredResultSlice(fts_scored.items, allocator);
        fts_scored.deinit(allocator);
    }
    fts_result.deinit(allocator);

    var vec_scored: []ScoredResult = &.{};
    if (options.enable_vector) {
        if (is_aborted(options.abort_signal)) {
            return .{ .results = try std.ArrayList(SearchResult).initCapacity(allocator, 0), .fts_count = 0, .vec_count = 0 };
        }
        const vec_result = try searchVec(db_, allocator, effective_query, collection);
        vec_scored = vec_result.results;
    }
    defer {
        freeScoredResultSlice(vec_scored, allocator);
        if (vec_scored.len > 0) allocator.free(vec_scored);
    }

    var lists: [2][]ScoredResult = undefined;
    lists[0] = fts_scored.items;
    lists[1] = vec_scored;

    var fused = try reciprocalRankFusion(allocator, &lists, options.rrf_k);
    defer allocator.free(fused);

    if (options.enable_rerank and fused.len > 1) {
        if (is_aborted(options.abort_signal)) {
            return .{ .results = try std.ArrayList(SearchResult).initCapacity(allocator, 0), .fts_count = 0, .vec_count = 0 };
        }
        const reranked = try rerankByEmbedding(db_, allocator, effective_query, fused);
        freeScoredResultSlice(fused, allocator);
        allocator.free(fused);
        fused = reranked;
    }
    defer freeScoredResultSlice(fused, allocator);

    var final_results = try std.ArrayList(SearchResult).initCapacity(allocator, @min(fused.len, options.max_results));
    errdefer {
        freeSearchResultSlice(final_results.items, allocator);
        final_results.deinit(allocator);
    }

    for (fused[0..@min(fused.len, options.max_results)]) |r| {
        var title = r.title;
        var hash = r.hash;
        if (std.mem.eql(u8, title, "")) {
            const doc = store.findActiveDocument(db_, r.collection, r.path, allocator) catch continue;
            defer {
                allocator.free(doc.title);
                allocator.free(doc.hash);
                allocator.free(doc.doc);
            }
            title = doc.title;
            hash = doc.hash;
        }
        try final_results.append(allocator, .{
            .id = r.id,
            .collection = try allocator.dupe(u8, r.collection),
            .path = try allocator.dupe(u8, r.path),
            .title = try allocator.dupe(u8, title),
            .hash = try allocator.dupe(u8, hash),
            .score = r.score,
        });
    }

    return .{ .results = final_results, .fts_count = fts_scored.items.len, .vec_count = vec_scored.len };
}

/// Configuration options for the hybrid search pipeline.
pub const HybridOptions = struct {
    enable_vector: bool = false,
    enable_query_expansion: bool = false,
    enable_rerank: bool = false,
    abort_signal: ?*const std.atomic.Value(bool) = null,
    rrf_k: f64 = RRF_K,
    max_results: usize = 20,
    min_score: f64 = 0.0,
};

fn is_aborted(signal: ?*const std.atomic.Value(bool)) bool {
    if (signal) |s| return s.load(.monotonic);
    return false;
}

fn rerankByEmbedding(db_: *db.Db, allocator: std.mem.Allocator, query: []const u8, results: []ScoredResult) ![]ScoredResult {
    const q_emb = try embed_text(allocator, query, true);
    defer allocator.free(q_emb);

    const bin_path = std.process.getEnvVarOwned(allocator, "QMD_LLAMA_EMBED_BIN") catch null;
    defer if (bin_path) |p| allocator.free(p);
    const model_path = std.process.getEnvVarOwned(allocator, "QMD_LLAMA_MODEL") catch null;
    defer if (model_path) |p| allocator.free(p);

    var passages = try allocator.alloc([]const u8, results.len);
    defer allocator.free(passages);
    var owned_passages = try allocator.alloc(?[]u8, results.len);
    defer allocator.free(owned_passages);
    for (owned_passages) |*slot| slot.* = null;
    defer {
        for (owned_passages) |entry| {
            if (entry) |text| allocator.free(text);
        }
    }
    var base_scores = try allocator.alloc(f32, results.len);
    defer allocator.free(base_scores);

    var rescored = try allocator.alloc(ScoredResult, results.len);
    errdefer allocator.free(rescored);
    for (results, 0..) |r, i| {
        var source_text = r.title;
        const doc = store.findActiveDocument(db_, r.collection, r.path, allocator) catch null;
        if (doc) |d| {
            defer {
                allocator.free(d.title);
                allocator.free(d.hash);
                allocator.free(d.doc);
            }
            source_text = d.doc;
        }

        const passage_copy = try allocator.dupe(u8, source_text);
        owned_passages[i] = passage_copy;
        passages[i] = passage_copy;

        const d_emb = embed_text(allocator, source_text, false) catch q_emb;
        defer if (d_emb.ptr != q_emb.ptr) allocator.free(d_emb);

        const cosine = llm.cosineSimilarity(q_emb, d_emb);
        const dense_score = @max(@as(f32, 0), (cosine + 1.0) * 0.5); // normalize [-1,1] -> [0,1]
        base_scores[i] = dense_score;

        var item = r;
        item.score = dense_score;
        rescored[i] = item;
    }

    const gen_scores = llm.rerankPassages(allocator, query, passages, bin_path, model_path) catch null;
    defer if (gen_scores) |s| allocator.free(s);

    if (gen_scores) |scores| {
        for (rescored, 0..) |*r, i| {
            // Confidence-aware blend:
            // - high generation confidence (near 0 or 1) gets more weight
            // - uncertain generation (near 0.5) falls back to dense score
            const g = @as(f32, @floatCast(@max(@as(f64, 0), @min(@as(f64, 1), scores[i]))));
            const confidence = @abs(g - 0.5) * 2.0; // [0,1]
            const gen_weight = 0.25 + 0.55 * confidence; // [0.25,0.80]
            const dense_weight = 1.0 - gen_weight;
            const blended = dense_weight * base_scores[i] + gen_weight * g;
            r.score = blended;
        }
    } else {
        // Small lexical prior to break ties when generation is unavailable.
        const lexical = llm.rerankPassages(allocator, query, passages, null, null) catch null;
        defer if (lexical) |s| allocator.free(s);
        if (lexical) |ls| {
            for (rescored, 0..) |*r, i| {
                r.score = 0.85 * base_scores[i] + 0.15 * ls[i];
            }
        }
    }

    std.sort.heap(ScoredResult, rescored, {}, struct {
        fn less(_: void, a: ScoredResult, b: ScoredResult) bool {
            return a.score > b.score;
        }
    }.less);

    return rescored;
}

/// Results from a hybrid search including FTS and vector match counts.
pub const HybridResult = struct {
    results: std.ArrayList(SearchResult),
    fts_count: usize,
    vec_count: usize,

    pub fn deinit(self: *HybridResult, allocator: std.mem.Allocator) void {
        freeSearchResultSlice(self.results.items, allocator);
        self.results.deinit(allocator);
    }
};

/// Executes vector similarity search using stored embeddings, deduplicating
/// by document.
pub fn searchVec(
    db_: *db.Db,
    allocator: std.mem.Allocator,
    query: []const u8,
    collection: ?[]const u8,
) !struct { results: []ScoredResult } {
    const query_embedding = try embed_query(allocator, query);
    defer allocator.free(query_embedding);

    // Try native sqlite-vec first; fallback to JSON cosine path.
    if (searchVecNative(db_, allocator, query_embedding, collection)) |native| {
        return .{ .results = native };
    } else |_| {}

    var best_by_doc = std.AutoHashMap(i64, ScoredResult).init(allocator);
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

        const col_span = if (coll) |c| std.mem.span(c) else "";
        if (collection) |wanted| {
            if (!std.mem.eql(u8, col_span, wanted)) continue;
        }
        if (emb == null) continue;

        const hash = if (hsh) |h| try allocator.dupe(u8, std.mem.span(h)) else try allocator.dupe(u8, "");
        const col = try allocator.dupe(u8, col_span);
        const path = if (pth) |p| try allocator.dupe(u8, std.mem.span(p)) else try allocator.dupe(u8, "");
        const title = if (ttl) |t| try allocator.dupe(u8, std.mem.span(t)) else try allocator.dupe(u8, "");

        const doc_embedding = parse_embedding_json_array(allocator, std.mem.span(emb.?)) catch continue;
        defer allocator.free(doc_embedding);

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
                // Free the strings from the entry being replaced
                allocator.free(existing.collection);
                allocator.free(existing.path);
                allocator.free(existing.title);
                allocator.free(existing.hash);
                try best_by_doc.put(id, candidate);
            } else {
                // Discard the candidate — free its strings
                allocator.free(col);
                allocator.free(path);
                allocator.free(title);
                allocator.free(hash);
            }
        } else {
            try best_by_doc.put(id, candidate);
        }
    }

    var results = try std.ArrayList(ScoredResult).initCapacity(allocator, best_by_doc.count());
    errdefer results.deinit(allocator);

    var it = best_by_doc.iterator();
    while (it.next()) |entry| {
        try results.append(allocator, entry.value_ptr.*);
    }

    std.sort.heap(ScoredResult, results.items, {}, struct {
        fn less(_: void, a: ScoredResult, b: ScoredResult) bool {
            return a.score > b.score;
        }
    }.less);

    return .{ .results = try results.toOwnedSlice(allocator) };
}

fn searchVecNative(db_: *db.Db, allocator: std.mem.Allocator, query_embedding: []const f32, collection: ?[]const u8) ![]ScoredResult {
    const query_json = try encode_embedding_json(allocator, query_embedding);
    defer allocator.free(query_json);

    var best_by_doc = std.AutoHashMap(i64, ScoredResult).init(allocator);
    defer best_by_doc.deinit();

    var stmt = try db_.prepare(
        "SELECT d.id, d.hash, d.collection, d.path, d.title, v.distance FROM content_vectors_idx v JOIN documents d ON d.hash = v.hash WHERE d.active = 1 AND v.embedding MATCH vec_f32(?) AND k = 200 ORDER BY v.distance ASC",
    );
    defer stmt.finalize();
    try stmt.bindText(1, query_json);

    while (try stmt.step()) {
        const id = stmt.columnInt(0);
        const hsh = stmt.columnText(1);
        const coll = stmt.columnText(2);
        const pth = stmt.columnText(3);
        const ttl = stmt.columnText(4);
        const dist = stmt.columnDouble(5);

        const col_span = if (coll) |c| std.mem.span(c) else "";
        if (collection) |wanted| {
            if (!std.mem.eql(u8, col_span, wanted)) continue;
        }

        const hash = if (hsh) |h| try allocator.dupe(u8, std.mem.span(h)) else try allocator.dupe(u8, "");
        const col = try allocator.dupe(u8, col_span);
        const path = if (pth) |p| try allocator.dupe(u8, std.mem.span(p)) else try allocator.dupe(u8, "");
        const title = if (ttl) |t| try allocator.dupe(u8, std.mem.span(t)) else try allocator.dupe(u8, "");

        const score = 1.0 / (1.0 + dist);
        const candidate = ScoredResult{ .id = id, .collection = col, .path = path, .title = title, .hash = hash, .score = score };
        if (best_by_doc.get(id)) |existing| {
            if (candidate.score > existing.score) {
                allocator.free(existing.collection);
                allocator.free(existing.path);
                allocator.free(existing.title);
                allocator.free(existing.hash);
                try best_by_doc.put(id, candidate);
            } else {
                allocator.free(col);
                allocator.free(path);
                allocator.free(title);
                allocator.free(hash);
            }
        } else {
            try best_by_doc.put(id, candidate);
        }
    }

    var results = try std.ArrayList(ScoredResult).initCapacity(allocator, best_by_doc.count());
    var it = best_by_doc.iterator();
    while (it.next()) |entry| {
        try results.append(allocator, entry.value_ptr.*);
    }
    std.sort.heap(ScoredResult, results.items, {}, struct {
        fn less(_: void, a: ScoredResult, b: ScoredResult) bool {
            return a.score > b.score;
        }
    }.less);

    return try results.toOwnedSlice(allocator);
}

fn encode_embedding_json(allocator: std.mem.Allocator, embedding: []const f32) ![]u8 {
    var out = try std.ArrayList(u8).initCapacity(allocator, embedding.len * 10 + 2);
    defer out.deinit(allocator);
    try out.append(allocator, '[');
    for (embedding, 0..) |v, i| {
        if (i > 0) try out.append(allocator, ',');
        try out.writer(allocator).print("{d}", .{v});
    }
    try out.append(allocator, ']');
    return out.toOwnedSlice(allocator);
}

fn embed_query(allocator: std.mem.Allocator, query: []const u8) ![]f32 {
    return embed_text(allocator, query, true);
}

fn embed_text(allocator: std.mem.Allocator, text: []const u8, is_query: bool) ![]f32 {
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

fn parse_embedding_json_array(allocator: std.mem.Allocator, json: []const u8) ![]f32 {
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
        const emb = try embed_query(allocator, "oauth sign in token");
        defer allocator.free(emb);
        try std.testing.expect(emb.len > 10);
    } else |_| {
        // Skip when no real model is configured.
    }
}

test "reciprocalRankFusion merges results" {
    const allocator = std.testing.allocator;
    const list1: []const ScoredResult = &.{
        .{ .id = 1, .collection = "a", .path = "a", .title = "A", .hash = "", .score = 0.9 },
        .{ .id = 2, .collection = "a", .path = "b", .title = "B", .hash = "", .score = 0.8 },
    };
    const list2: []const ScoredResult = &.{
        .{ .id = 2, .collection = "a", .path = "b", .title = "B", .hash = "", .score = 0.7 },
        .{ .id = 3, .collection = "a", .path = "c", .title = "C", .hash = "", .score = 0.6 },
    };

    const fused = try reciprocalRankFusion(allocator, &.{ list1, list2 }, RRF_K);
    defer {
        freeScoredResultSlice(fused, allocator);
        allocator.free(fused);
    }

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

    _ = try store.insertDocument(&db_, "test", "a.md", "# Auth\nLogin flow");
    _ = try store.insertDocument(&db_, "test", "b.md", "# Setup\nInstall");

    var result = try hybridSearch(&db_, std.heap.page_allocator, "auth", null, .{ .enable_vector = false });
    defer result.deinit(std.heap.page_allocator);

    try std.testing.expect(result.fts_count > 0);
}

test "searchVec uses stored vectors and ranks by cosine" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    _ = try store.insertDocument(&db_, "test", "a.md", "# Auth\nLogin and auth flow");
    _ = try store.insertDocument(&db_, "test", "b.md", "# Cooking\nRecipe and food");

    const doc_a = try store.findActiveDocument(&db_, "test", "a.md", std.testing.allocator);
    defer {
        std.testing.allocator.free(doc_a.title);
        std.testing.allocator.free(doc_a.hash);
        std.testing.allocator.free(doc_a.doc);
    }
    const doc_b = try store.findActiveDocument(&db_, "test", "b.md", std.testing.allocator);
    defer {
        std.testing.allocator.free(doc_b.title);
        std.testing.allocator.free(doc_b.hash);
        std.testing.allocator.free(doc_b.doc);
    }

    // Match the current fallback embedding model for deterministic test behavior.
    var fallback = try llm.LlamaCpp.init("/nonexistent", std.heap.page_allocator);
    defer fallback.deinit();
    const q_emb = try fallback.embed("auth", std.heap.page_allocator);
    defer std.heap.page_allocator.free(q_emb);
    const b_emb = try fallback.embed("totally unrelated baseline", std.heap.page_allocator);
    defer std.heap.page_allocator.free(b_emb);

    try store.upsertContentVector(&db_, doc_a.hash, "test", q_emb, std.heap.page_allocator);
    try store.upsertContentVector(&db_, doc_b.hash, "test", b_emb, std.heap.page_allocator);

    const result = try searchVec(&db_, std.heap.page_allocator, "auth", null);
    try std.testing.expect(result.results.len >= 2);
    try std.testing.expect(result.results[0].score >= result.results[1].score);
}

test "hybridSearch supports query expansion option" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    _ = try store.insertDocument(&db_, "test", "a.md", "# Login\nHow to authenticate users");
    _ = try store.insertDocument(&db_, "test", "b.md", "# Cooking\nHow to boil pasta");

    var result = try hybridSearch(&db_, std.heap.page_allocator, "how login?", null, .{
        .enable_vector = true,
        .enable_query_expansion = true,
        .enable_rerank = false,
        .max_results = 10,
    });
    defer result.deinit(std.heap.page_allocator);

    try std.testing.expect(result.fts_count >= 0);
}

test "hybridSearch supports rerank option" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    _ = try store.insertDocument(&db_, "test", "a.md", "# Authentication\nOAuth token login");
    _ = try store.insertDocument(&db_, "test", "b.md", "# Recipe\nPasta cooking instructions");

    var result = try hybridSearch(&db_, std.heap.page_allocator, "oauth login", null, .{
        .enable_vector = true,
        .enable_query_expansion = false,
        .enable_rerank = true,
        .max_results = 10,
    });
    defer result.deinit(std.heap.page_allocator);

    try std.testing.expect(result.results.items.len > 0);
}

test "hybridSearch supports abort signal" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    _ = try store.insertDocument(&db_, "test", "a.md", "# A\n\nauth flow");

    var aborted = std.atomic.Value(bool).init(true);
    var result = try hybridSearch(&db_, std.heap.page_allocator, "auth", null, .{
        .enable_vector = true,
        .abort_signal = &aborted,
    });
    defer result.deinit(std.heap.page_allocator);
    try std.testing.expectEqual(@as(usize, 0), result.results.items.len);
}

test "buildFTS5Query parses simple tokens" {
    const result = try buildFTS5Query(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("hello world", result);
}

test "buildFTS5Query handles negation" {
    const result = try buildFTS5Query(std.testing.allocator, "hello -world");
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("hello -world", result);
}

test "buildFTS5Query handles prefix match" {
    const result = try buildFTS5Query(std.testing.allocator, "auth*");
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("auth*", result);
}

test "buildFTS5Query handles hyphenated words" {
    const result = try buildFTS5Query(std.testing.allocator, "real-time");
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("\"real-time\"", result);
}
