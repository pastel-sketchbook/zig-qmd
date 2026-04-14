const std = @import("std");
const db = @import("db.zig");
const store = @import("store.zig");

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
) !struct { results: std.ArrayList(SearchResult) } {
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
    const fts_result = try searchFTS(db_, query, collection);

    var fts_scored = try std.ArrayList(ScoredResult).initCapacity(std.heap.page_allocator, fts_result.results.items.len);
    errdefer fts_scored.deinit(std.heap.page_allocator);

    for (fts_result.results.items) |r| {
        try fts_scored.append(std.heap.page_allocator, .{ .id = r.id, .collection = r.collection, .path = r.path, .title = r.title, .hash = r.hash, .score = r.score });
    }

    var vec_scored: []ScoredResult = &.{};
    if (options.enable_vector) {
        const vec_result = try searchVec(db_, query, collection);
        vec_scored = vec_result.results;
    }

    var lists: [2][]ScoredResult = undefined;
    lists[0] = fts_scored.items;
    lists[1] = vec_scored;

    const fused = reciprocalRankFusion(&lists, options.rrf_k);

    var final_results = try std.ArrayList(SearchResult).initCapacity(std.heap.page_allocator, @min(fused.len, options.max_results));
    errdefer final_results.deinit(std.heap.page_allocator);

    for (fused[0..@min(fused.len, options.max_results)]) |r| {
        var title = r.title;
        var hash = r.hash;
        if (std.mem.eql(u8, title, "")) {
            const doc = store.findActiveDocument(db_, r.collection, r.path) catch continue;
            title = doc.title;
            hash = doc.hash;
        }
        try final_results.append(std.heap.page_allocator, .{ .id = r.id, .collection = r.collection, .path = r.path, .title = title, .hash = hash, .score = r.score });
    }

    return .{ .results = final_results, .fts_count = fts_scored.items.len, .vec_count = vec_scored.len };
}

pub const HybridOptions = struct {
    enable_vector: bool = false,
    enable_rerank: bool = false,
    rrf_k: f64 = RRF_K,
    max_results: usize = 20,
    min_score: f64 = 0.0,
};

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
    _ = query;

    var results = try std.ArrayList(ScoredResult).initCapacity(std.heap.page_allocator, 0);
    errdefer results.deinit(std.heap.page_allocator);

    var stmt = db_.prepare("SELECT hash, collection, path, title FROM documents WHERE active = 1") catch return .{ .results = &.{} };
    defer stmt.finalize();

    while (stmt.step() catch false) {
        const hsh = stmt.columnText(0);
        const coll = stmt.columnText(1);
        const pth = stmt.columnText(2);
        const ttl = stmt.columnText(3);

        const hash = if (hsh) |h| try std.heap.page_allocator.dupe(u8, std.mem.span(h)) else try std.heap.page_allocator.dupe(u8, "");
        const col = if (coll) |c| try std.heap.page_allocator.dupe(u8, std.mem.span(c)) else try std.heap.page_allocator.dupe(u8, "");
        const path = if (pth) |p| try std.heap.page_allocator.dupe(u8, std.mem.span(p)) else try std.heap.page_allocator.dupe(u8, "");
        const title = if (ttl) |t| try std.heap.page_allocator.dupe(u8, std.mem.span(t)) else try std.heap.page_allocator.dupe(u8, "");

        if (collection == null or std.mem.eql(u8, col, collection.?[0..collection.?.len])) {
            try results.append(std.heap.page_allocator, .{ .id = 0, .collection = col, .path = path, .title = title, .hash = hash, .score = 0.5 });
        }
    }

    return .{ .results = results.items };
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
    try std.testing.expect(fused[0].score >= fused[1].score);
}

test "hybridSearch with FTS only" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    try store.insertDocument(&db_, "test", "a.md", "# Auth\nLogin flow");
    try store.insertDocument(&db_, "test", "b.md", "# Setup\nInstall");

    const result = try hybridSearch(&db_, "auth", null, .{ .enable_vector = false });
    defer result.results.deinit(std.heap.page_allocator);

    try std.testing.expect(result.fts_count > 0);
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
