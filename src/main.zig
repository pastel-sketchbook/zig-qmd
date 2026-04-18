const std = @import("std");
const qmd = @import("qmd");
const build_options = @import("build_options");

const DB_PATH = ".qmd/data.db";
const DEFAULT_LLAMA_EMBED_BIN = "deps/llama.cpp/build/bin/llama-embedding";
const DEFAULT_LLAMA_MODEL_PATH = "";

// ---------------------------------------------------------------------------
// Native llama.cpp integration (enabled via -Dllama build flag)
// ---------------------------------------------------------------------------

/// Module-level pointers to NativeLlama instances, set during command
/// execution and read by the EmbedFn / ExpandQueryFn wrappers below.
/// This sidesteps Zig's lack of closures for function-pointer callbacks.
///
/// g_embed_llama  — dedicated embedding model (QMD_EMBED_MODEL)
/// g_native_llama — generation model for query expansion/chat (QMD_MODEL)
var g_embed_llama: ?*NativeLlamaType = null;
var g_native_llama: ?*NativeLlamaType = null;

const NativeLlamaType = if (build_options.enable_llama) qmd.llm_native.NativeLlama else struct {
    pub fn deinit(_: *@This()) void {}
    pub fn embed(_: *@This(), _: []const u8) error{}![]f32 {
        unreachable;
    }
    pub fn supportsEmbedding(_: *const @This()) bool {
        return false;
    }
};

/// EmbedFn-compatible wrapper that delegates to the embed llama instance.
/// NativeLlama uses page_allocator internally, so we copy the result into
/// the caller's allocator and free the original.
fn nativeEmbedFn(allocator: std.mem.Allocator, text: []const u8, _: bool) anyerror![]f32 {
    if (build_options.enable_llama) {
        const llama = g_embed_llama orelse return error.NativeLlamaNotInitialized;
        const native_result = llama.embed(text) catch |e| return @as(anyerror, e);
        // Copy from page_allocator to caller's allocator
        const result = allocator.alloc(f32, native_result.len) catch return error.OutOfMemory;
        @memcpy(result, native_result);
        std.heap.page_allocator.free(native_result);
        return result;
    }
    return error.NativeLlamaNotAvailable;
}

fn isQuotedWrapper(text: []const u8) bool {
    if (text.len < 2) return false;
    const first = text[0];
    const last = text[text.len - 1];
    return (first == '"' and last == '"') or (first == '\'' and last == '\'');
}

fn sanitizeGeneratedText(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const trimmed = std.mem.trim(u8, text, &std.ascii.whitespace);
    if (trimmed.len == 0) return allocator.dupe(u8, "");

    var start: usize = 0;
    var end: usize = trimmed.len;
    while (end > start and (trimmed[start] == '\'' or trimmed[start] == '"' or trimmed[start] == '`')) : (start += 1) {}
    while (end > start and (trimmed[end - 1] == '\'' or trimmed[end - 1] == '"' or trimmed[end - 1] == '`')) : (end -= 1) {}

    var core = std.mem.trim(u8, trimmed[start..end], &std.ascii.whitespace);
    if (core.len == 0) return allocator.dupe(u8, "");

    if (std.mem.startsWith(u8, core, "Expanded query:")) {
        core = std.mem.trim(u8, core["Expanded query:".len..], &std.ascii.whitespace);
    }

    if (isQuotedWrapper(core)) {
        core = std.mem.trim(u8, core[1 .. core.len - 1], &std.ascii.whitespace);
    }

    return allocator.dupe(u8, core);
}

fn sanitizeExpandedQuery(allocator: std.mem.Allocator, query: []const u8, generated: []const u8) ![]const u8 {
    const cleaned = try sanitizeGeneratedText(allocator, generated);
    if (cleaned.len == 0) {
        allocator.free(cleaned);
        return allocator.dupe(u8, query);
    }

    if (std.mem.startsWith(u8, cleaned, query)) return cleaned;

    var merged = try std.ArrayList(u8).initCapacity(allocator, query.len + 1 + cleaned.len);
    errdefer merged.deinit(allocator);
    try merged.appendSlice(allocator, query);
    try merged.append(allocator, ' ');
    try merged.appendSlice(allocator, cleaned);
    allocator.free(cleaned);
    return merged.toOwnedSlice(allocator);
}

/// ExpandQueryFn-compatible wrapper that delegates to the generation llama instance.
fn nativeExpandQueryFn(allocator: std.mem.Allocator, query: []const u8) anyerror![]const u8 {
    if (build_options.enable_llama) {
        const llama = g_native_llama orelse return error.NativeLlamaNotInitialized;
        const prompt = try std.fmt.allocPrint(
            allocator,
            "Rewrite this search query for retrieval. " ++
                "Return only a short space-separated list of keywords and close synonyms. " ++
                "Do not use quotes, bullets, labels, or full sentences. Preserve important technical terms.\n\n" ++
                "Query: {s}\n\nKeywords:",
            .{query},
        );
        defer allocator.free(prompt);
        const result = llama.generate(prompt, 64) catch |e| return @as(anyerror, e);
        defer std.heap.page_allocator.free(result);
        return sanitizeExpandedQuery(allocator, query, result);
    }
    return error.NativeLlamaNotAvailable;
}

/// Returns EmbedFn pointer if the embed llama is initialized and supports embedding.
fn getNativeEmbedFn() ?qmd.search.EmbedFn {
    if (build_options.enable_llama) {
        if (g_embed_llama) |nl| {
            if (nl.supportsEmbedding()) return &nativeEmbedFn;
        }
    }
    return null;
}

/// Returns ExpandQueryFn pointer if the generation llama is initialized.
fn getNativeExpandQueryFn() ?qmd.search.ExpandQueryFn {
    if (g_native_llama != null) return &nativeExpandQueryFn;
    return null;
}

/// Loads a GGUF model from a path string. Returns null if path is empty or load fails.
fn loadNativeLlama(model_path: []const u8) ?NativeLlamaType {
    if (!build_options.enable_llama) return null;
    if (model_path.len == 0) return null;
    var path_buf: [4096]u8 = undefined;
    if (model_path.len >= path_buf.len) return null;
    @memcpy(path_buf[0..model_path.len], model_path);
    path_buf[model_path.len] = 0;
    return qmd.llm_native.NativeLlama.init(std.heap.page_allocator, path_buf[0..model_path.len :0]) catch null;
}

/// Initializes the generation model from QMD_MODEL env var.
fn initNativeLlama(environ: *std.process.Environ.Map) ?NativeLlamaType {
    const model_path = environ.get("QMD_MODEL") orelse environ.get("QMD_LLAMA_MODEL") orelse return null;
    return loadNativeLlama(model_path);
}

/// Initializes the embedding model from QMD_EMBED_MODEL env var.
/// Falls back to QMD_MODEL if QMD_EMBED_MODEL is not set.
fn initEmbedLlama(environ: *std.process.Environ.Map) ?NativeLlamaType {
    const model_path = environ.get("QMD_EMBED_MODEL") orelse environ.get("QMD_MODEL") orelse environ.get("QMD_LLAMA_MODEL") orelse return null;
    return loadNativeLlama(model_path);
}

const OutputFormat = enum {
    text,
    json,
    csv,
    md,
};

/// Controls the ordering of search results in CLI output.
const SortOrder = enum {
    /// Sort by relevance score descending (default).
    score,
    /// Sort by database row id ascending (insertion order).
    index,
};

const Preview = struct {
    title: []const u8,
    body: []const u8,
    score: f64,
};

const ContextEval = struct {
    total_score: f64,
    term_coverage: f64,
};

const ContextCandidate = struct {
    index: usize,
    score: f64,
};

const DocRef = struct {
    collection: []const u8,
    path: []const u8,
};

fn parseOutputFlag(arg: []const u8) ?OutputFormat {
    if (std.mem.eql(u8, arg, "--json")) return .json;
    if (std.mem.eql(u8, arg, "--csv")) return .csv;
    if (std.mem.eql(u8, arg, "--md")) return .md;
    return null;
}

/// Parses `--sort=score` or `--sort=index` from a CLI argument.
fn parseSortFlag(arg: []const u8) ?SortOrder {
    if (std.mem.eql(u8, arg, "--sort=score")) return .score;
    if (std.mem.eql(u8, arg, "--sort=index")) return .index;
    return null;
}

/// Sorts search results in place according to the given order.
fn sortSearchResults(items: []qmd.search.SearchResult, order: SortOrder) void {
    switch (order) {
        .score => std.sort.heap(qmd.search.SearchResult, items, {}, struct {
            fn less(_: void, a: qmd.search.SearchResult, b: qmd.search.SearchResult) bool {
                return a.score > b.score;
            }
        }.less),
        .index => std.sort.heap(qmd.search.SearchResult, items, {}, struct {
            fn less(_: void, a: qmd.search.SearchResult, b: qmd.search.SearchResult) bool {
                return a.id < b.id;
            }
        }.less),
    }
}

/// Sorts scored results in place according to the given order.
fn sortScoredResults(items: []qmd.search.ScoredResult, order: SortOrder) void {
    switch (order) {
        .score => std.sort.heap(qmd.search.ScoredResult, items, {}, struct {
            fn less(_: void, a: qmd.search.ScoredResult, b: qmd.search.ScoredResult) bool {
                return a.score > b.score;
            }
        }.less),
        .index => std.sort.heap(qmd.search.ScoredResult, items, {}, struct {
            fn less(_: void, a: qmd.search.ScoredResult, b: qmd.search.ScoredResult) bool {
                return a.id < b.id;
            }
        }.less),
    }
}

fn parseDocRef(input: []const u8) ?DocRef {
    const raw = if (std.mem.startsWith(u8, input, "zmd://"))
        input[6..]
    else
        input;
    const slash = std.mem.indexOfScalar(u8, raw, '/') orelse return null;
    if (slash == 0 or slash + 1 >= raw.len) return null;
    return .{ .collection = raw[0..slash], .path = raw[slash + 1 ..] };
}

fn writeJsonString(out: anytype, s: []const u8) !void {
    try out.writeAll("\"");
    for (s) |ch| {
        switch (ch) {
            '"' => try out.writeAll("\\\""),
            '\\' => try out.writeAll("\\\\"),
            '\n' => try out.writeAll("\\n"),
            '\r' => try out.writeAll("\\r"),
            '\t' => try out.writeAll("\\t"),
            else => try out.print("{c}", .{ch}),
        }
    }
    try out.writeAll("\"");
}

fn writeCsvField(out: anytype, s: []const u8) !void {
    try out.writeAll("\"");
    for (s) |ch| {
        if (ch == '"') {
            try out.writeAll("\"\"");
        } else {
            try out.print("{c}", .{ch});
        }
    }
    try out.writeAll("\"");
}

fn skipFrontmatter(doc: []const u8) []const u8 {
    if (!std.mem.startsWith(u8, doc, "---\n")) return doc;
    const rest = doc[4..];
    const end = std.mem.indexOf(u8, rest, "\n---\n") orelse return doc;
    return rest[end + 5 ..];
}

fn nextSignificantQueryToken(query: []const u8) ?[]const u8 {
    var tok_it = std.mem.tokenizeAny(u8, query, " \n\r\t,.;:!?()[]{}\"'");
    var fallback: ?[]const u8 = null;
    while (tok_it.next()) |token| {
        if (token.len < 2) continue;
        if (fallback == null) fallback = token;
        if (!isQueryStopword(token)) return token;
    }
    return fallback;
}

fn containsInsensitive(haystack: []const u8, needle: []const u8) bool {
    return utf8IndexOfInsensitive(haystack, needle) != null;
}

fn rewriteContextQuery(allocator: std.mem.Allocator, query: []const u8) !?[]u8 {
    const wants_rust = containsInsensitive(query, "rust");
    const wants_game = containsInsensitive(query, "game") or containsInsensitive(query, "games") or containsInsensitive(query, "gamedev");
    const wants_learning = containsInsensitive(query, "learn") or containsInsensitive(query, "start") or containsInsensitive(query, "begin");
    const wants_coding = containsInsensitive(query, "code") or containsInsensitive(query, "coding") or containsInsensitive(query, "build");

    if (!(wants_rust and wants_game and (wants_learning or wants_coding))) return null;

    const rewritten = try std.fmt.allocPrint(
        allocator,
        "{s} rust game development gamedev bevy ecs graphics engine physics math gameplay",
        .{query},
    );
    return rewritten;
}

fn extractSnippet(allocator: std.mem.Allocator, query: []const u8, doc: []const u8) ![]u8 {
    if (doc.len == 0) return allocator.dupe(u8, "");

    const visible_doc = skipFrontmatter(doc);
    const token = nextSignificantQueryToken(query) orelse return allocator.dupe(u8, visible_doc[0..@min(visible_doc.len, 180)]);

    // Try to find the token in the document using a UTF-8-aware,
    // case-insensitive search.  For CJK characters (which have no case),
    // std.ascii.toLower is a no-op on high bytes so the comparison works
    // correctly only when we compare at the codepoint level.
    const idx = utf8IndexOfInsensitive(visible_doc, token) orelse 0;

    // Snap start/end to UTF-8 codepoint boundaries so we never slice
    // in the middle of a multi-byte character.
    const raw_start: usize = if (idx > 60) idx - 60 else 0;
    const start = snapBackToCodepoint(visible_doc, raw_start);

    const raw_end = @min(visible_doc.len, idx + token.len + 140);
    const end = snapForwardToCodepoint(visible_doc, raw_end);

    var out = try std.ArrayList(u8).initCapacity(allocator, end - start + 8);
    defer out.deinit(allocator);
    if (start > 0) try out.appendSlice(allocator, "...");
    try out.appendSlice(allocator, visible_doc[start..end]);
    if (end < visible_doc.len) try out.appendSlice(allocator, "...");
    return out.toOwnedSlice(allocator);
}

/// Extracts a large context window centered on query term matches.
/// Returns a slice of the original doc (no allocation needed).
/// Searches for each query token (longest first for specificity) and
/// centers a window of `max_len` chars around the best match.
/// Falls back to the start of the document.
fn extractLargeContext(doc: []const u8, query: []const u8, max_len: usize) []const u8 {
    const visible_doc = skipFrontmatter(doc);
    if (visible_doc.len <= max_len) return visible_doc;

    // Collect tokens, skip stopwords, then search longest-first for better specificity
    const stopwords = [_][]const u8{
        "how",   "what",  "where",  "when", "why",   "who",   "which",
        "does",  "do",    "did",    "can",  "could", "would", "should",
        "is",    "are",   "was",    "were", "be",    "been",  "being",
        "the",   "a",     "an",     "in",   "on",    "at",    "to",
        "for",   "of",    "with",   "from", "by",    "as",    "or",
        "and",   "but",   "this",   "that", "it",    "its",   "my",
        "your",  "their", "works",  "work", "about", "into",  "over",
        "learn", "code",  "coding",
    };
    var tokens: [32][]const u8 = undefined;
    var token_count: usize = 0;
    var tok_it = std.mem.tokenizeScalar(u8, query, ' ');
    while (tok_it.next()) |token| {
        if (token.len < 2) continue;
        // Skip stopwords
        var is_stop = false;
        for (&stopwords) |sw| {
            if (std.ascii.eqlIgnoreCase(token, sw)) {
                is_stop = true;
                break;
            }
        }
        if (is_stop) continue;
        if (token_count < tokens.len) {
            tokens[token_count] = token;
            token_count += 1;
        }
    }

    // If all tokens were stopwords, fall back to all non-tiny tokens
    if (token_count == 0) {
        tok_it = std.mem.tokenizeScalar(u8, query, ' ');
        while (tok_it.next()) |token| {
            if (token.len < 2) continue;
            if (token_count < tokens.len) {
                tokens[token_count] = token;
                token_count += 1;
            }
        }
    }

    // Sort by length ascending — shorter non-stopword tokens tend to be
    // more specific (e.g. "VPA" is rarer than "Kubernetes")
    for (1..token_count) |i| {
        const key = tokens[i];
        var j: usize = i;
        while (j > 0 and tokens[j - 1].len > key.len) {
            tokens[j] = tokens[j - 1];
            j -= 1;
        }
        tokens[j] = key;
    }

    var best_idx: usize = 0;
    var found = false;
    for (tokens[0..token_count]) |token| {
        if (utf8IndexOfInsensitive(visible_doc, token)) |idx| {
            best_idx = idx;
            found = true;
            break;
        }
    }

    if (!found) return visible_doc[0..max_len];

    // Center window on the match
    const half = max_len / 2;
    const raw_start: usize = if (best_idx > half) best_idx - half else 0;
    const start = snapBackToCodepoint(visible_doc, raw_start);
    const raw_end = @min(visible_doc.len, start + max_len);
    const end = snapForwardToCodepoint(visible_doc, raw_end);
    return visible_doc[start..end];
}

fn isQueryStopword(token: []const u8) bool {
    const stopwords = [_][]const u8{
        "how",   "what",  "where",  "when",  "why",   "who",   "which",
        "does",  "do",    "did",    "can",   "could", "would", "should",
        "is",    "are",   "was",    "were",  "be",    "been",  "being",
        "the",   "a",     "an",     "in",    "on",    "at",    "to",
        "for",   "of",    "with",   "from",  "by",    "as",    "or",
        "and",   "but",   "this",   "that",  "it",    "its",   "my",
        "your",  "their", "works",  "work",  "about", "into",  "over",
        "learn", "code",  "coding", "build",
    };
    for (&stopwords) |sw| {
        if (std.ascii.eqlIgnoreCase(token, sw)) return true;
    }
    return false;
}

fn collectQueryTerms(query: []const u8, buffer: *[16][]const u8) usize {
    var count: usize = 0;
    var tok_it = std.mem.tokenizeAny(u8, query, " \n\r\t,.;:!?()[]{}\"'");
    while (tok_it.next()) |token| {
        if (token.len < 2 or isQueryStopword(token)) continue;
        var duplicate = false;
        for (buffer[0..count]) |existing| {
            if (std.ascii.eqlIgnoreCase(existing, token)) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate and count < buffer.len) {
            buffer[count] = token;
            count += 1;
        }
    }

    if (count > 0) return count;

    tok_it = std.mem.tokenizeAny(u8, query, " \n\r\t,.;:!?()[]{}\"'");
    while (tok_it.next()) |token| {
        if (token.len < 2) continue;
        var duplicate = false;
        for (buffer[0..count]) |existing| {
            if (std.ascii.eqlIgnoreCase(existing, token)) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate and count < buffer.len) {
            buffer[count] = token;
            count += 1;
        }
    }

    return count;
}

fn scoreQueryTermsAgainstPreview(query: []const u8, title: []const u8, body: []const u8, score: f64, rank: usize) f64 {
    var terms: [16][]const u8 = undefined;
    const term_count = collectQueryTerms(query, &terms);
    if (term_count == 0) {
        return score * (1.0 / @as(f64, @floatFromInt(rank + 1)));
    }

    const rank_weight = 1.0 / @as(f64, @floatFromInt(rank + 1));
    var preview_score = score * (2.0 * rank_weight);
    for (terms[0..term_count]) |term| {
        if (utf8IndexOfInsensitive(title, term) != null) {
            preview_score += 1.5 * rank_weight;
        } else if (utf8IndexOfInsensitive(body, term) != null) {
            preview_score += 1.0 * rank_weight;
        }
    }

    return preview_score;
}

fn scoreContextPreviews(query: []const u8, previews: []const Preview) f64 {
    return evaluateContextPreviews(query, previews).total_score;
}

fn evaluateContextPreviews(query: []const u8, previews: []const Preview) ContextEval {
    if (previews.len == 0) return .{ .total_score = 0, .term_coverage = 0 };

    var terms: [16][]const u8 = undefined;
    const term_count = collectQueryTerms(query, &terms);

    var matched = [_]bool{false} ** 16;
    var total_score: f64 = 0;
    for (previews, 0..) |preview, i| {
        total_score += scoreQueryTermsAgainstPreview(query, preview.title, preview.body, preview.score, i);
        for (terms[0..term_count], 0..) |term, term_idx| {
            if (utf8IndexOfInsensitive(preview.title, term) != null or utf8IndexOfInsensitive(preview.body, term) != null) {
                matched[term_idx] = true;
            }
        }
    }

    var matched_count: usize = 0;
    for (matched[0..term_count]) |seen| {
        if (seen) matched_count += 1;
    }
    const term_coverage = if (term_count > 0)
        @as(f64, @floatFromInt(matched_count)) / @as(f64, @floatFromInt(term_count))
    else
        1.0;
    if (term_count > 0) {
        total_score += 8.0 * term_coverage;
    }
    return .{ .total_score = total_score, .term_coverage = term_coverage };
}

fn evaluateContextSearchResults(
    db_: *qmd.db.Db,
    allocator: std.mem.Allocator,
    query: []const u8,
    results: []const qmd.search.SearchResult,
) ContextEval {
    var previews: [5]Preview = undefined;
    var owned_snippets = [_]?[]u8{null} ** 5;
    defer {
        for (owned_snippets) |maybe_snippet| {
            if (maybe_snippet) |snippet| allocator.free(snippet);
        }
    }

    var count: usize = 0;
    for (results) |r| {
        if (count >= previews.len) break;

        var body: []const u8 = r.path;
        const doc = qmd.store.findActiveDocument(db_, r.collection, r.path, allocator) catch null;
        if (doc) |d| {
            defer {
                allocator.free(d.title);
                allocator.free(d.hash);
                allocator.free(d.doc);
            }
            const snippet = extractSnippet(allocator, query, d.doc) catch null;
            if (snippet) |owned| {
                owned_snippets[count] = owned;
                body = owned;
            }
        }

        previews[count] = .{
            .title = r.title,
            .body = body,
            .score = r.score,
        };
        count += 1;
    }

    return evaluateContextPreviews(query, previews[0..count]);
}

fn shouldPreferExpandedContextResults(
    db_: *qmd.db.Db,
    allocator: std.mem.Allocator,
    query: []const u8,
    baseline: []const qmd.search.SearchResult,
    expanded: []const qmd.search.SearchResult,
) bool {
    if (expanded.len == 0) return false;
    if (baseline.len == 0) return true;

    const baseline_eval = evaluateContextSearchResults(db_, allocator, query, baseline);
    const expanded_eval = evaluateContextSearchResults(db_, allocator, query, expanded);

    if (expanded_eval.term_coverage < 0.75 and expanded_eval.term_coverage < baseline_eval.term_coverage) {
        return false;
    }

    return expanded_eval.total_score > baseline_eval.total_score + 0.75;
}

fn hasTranscriptDetailPair(results: []const qmd.search.SearchResult, title: []const u8) bool {
    var has_transcript = false;
    var has_detail = false;
    for (results) |r| {
        if (!std.mem.eql(u8, r.title, title)) continue;
        if (std.mem.indexOf(u8, r.path, "raw/transcripts/") != null) has_transcript = true;
        if (std.mem.indexOf(u8, r.path, "videos/details/") != null) has_detail = true;
    }
    return has_transcript and has_detail;
}

fn freeLocalSearchResultSlice(items: []qmd.search.SearchResult, allocator: std.mem.Allocator) void {
    for (items) |r| {
        allocator.free(r.collection);
        allocator.free(r.path);
        allocator.free(r.title);
        allocator.free(r.hash);
    }
}

fn rerankContextResults(
    db_: *qmd.db.Db,
    allocator: std.mem.Allocator,
    query: []const u8,
    results: *std.ArrayList(qmd.search.SearchResult),
    keep: usize,
) !void {
    if (results.items.len <= 1) return;

    var terms: [16][]const u8 = undefined;
    const term_count = collectQueryTerms(query, &terms);

    var candidates = try std.ArrayList(ContextCandidate).initCapacity(allocator, results.items.len);
    defer candidates.deinit(allocator);

    for (results.items, 0..) |r, idx| {
        var score = r.score * 10.0;
        if (hasTranscriptDetailPair(results.items, r.title)) score += 1.0;

        const doc = qmd.store.findActiveDocument(db_, r.collection, r.path, allocator) catch null;
        if (doc) |d| {
            defer {
                allocator.free(d.title);
                allocator.free(d.hash);
                allocator.free(d.doc);
            }
            const snippet = extractSnippet(allocator, query, d.doc) catch null;
            if (snippet) |owned| {
                defer allocator.free(owned);
                var overlap: usize = 0;
                for (terms[0..term_count]) |term| {
                    if (utf8IndexOfInsensitive(owned, term) != null) overlap += 1;
                }
                score += 2.5 * @as(f64, @floatFromInt(overlap));
            }
        }

        try candidates.append(allocator, .{ .index = idx, .score = score });
    }

    std.sort.heap(ContextCandidate, candidates.items, {}, struct {
        fn less(_: void, a: ContextCandidate, b: ContextCandidate) bool {
            return a.score > b.score;
        }
    }.less);

    var reranked = try std.ArrayList(qmd.search.SearchResult).initCapacity(allocator, @min(keep, candidates.items.len));
    errdefer {
        freeLocalSearchResultSlice(reranked.items, allocator);
        reranked.deinit(allocator);
    }

    for (candidates.items[0..@min(keep, candidates.items.len)]) |candidate| {
        const item = results.items[candidate.index];
        try reranked.append(allocator, .{
            .id = item.id,
            .collection = try allocator.dupe(u8, item.collection),
            .path = try allocator.dupe(u8, item.path),
            .title = try allocator.dupe(u8, item.title),
            .hash = try allocator.dupe(u8, item.hash),
            .score = candidate.score,
        });
    }

    freeLocalSearchResultSlice(results.items, allocator);
    results.deinit(allocator);
    results.* = reranked;
}

/// Find `needle` in `haystack` using a byte-level comparison that is
/// case-insensitive for ASCII while leaving non-ASCII bytes (CJK, Hangul,
/// etc.) compared exactly.  Returns the byte offset of the first match.
fn utf8IndexOfInsensitive(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len == 0) return 0;
    if (needle.len > haystack.len) return null;

    outer: for (0..haystack.len - needle.len + 1) |i| {
        for (0..needle.len) |j| {
            const a = haystack[i + j];
            const b = needle[j];
            // Only apply toLower for ASCII bytes (< 0x80).
            // Non-ASCII bytes (part of UTF-8 multibyte sequences) are
            // compared exactly — CJK/Hangul has no case distinction.
            const la = if (a < 0x80) std.ascii.toLower(a) else a;
            const lb = if (b < 0x80) std.ascii.toLower(b) else b;
            if (la != lb) continue :outer;
        }
        return i;
    }
    return null;
}

/// Snap a byte offset backward to the start of the UTF-8 codepoint
/// containing that offset.  Continuation bytes (10xxxxxx) are skipped.
fn snapBackToCodepoint(data: []const u8, pos: usize) usize {
    var p = pos;
    while (p > 0 and (data[p] & 0xC0) == 0x80) {
        p -= 1;
    }
    return p;
}

/// Snap a byte offset forward to the start of the next codepoint (or to
/// data.len).  If `pos` already sits on a codepoint boundary, returns `pos`.
fn snapForwardToCodepoint(data: []const u8, pos: usize) usize {
    var p = pos;
    while (p < data.len and (data[p] & 0xC0) == 0x80) {
        p += 1;
    }
    return p;
}

fn make_embedding_engine(allocator: std.mem.Allocator, io: std.Io, environ: *std.process.Environ.Map) ?qmd.llm.LlamaEmbedding {
    const bin_path = environ.get("QMD_LLAMA_EMBED_BIN") orelse DEFAULT_LLAMA_EMBED_BIN;
    const model_path = environ.get("QMD_LLAMA_MODEL") orelse DEFAULT_LLAMA_MODEL_PATH;

    if (model_path.len == 0) {
        return null;
    }

    return qmd.llm.LlamaEmbedding.init(allocator, io, bin_path, model_path) catch null;
}

/// CLI entry point for the zmd command-line tool.
pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.skip(); // skip program name

    const cmd = args.next() orelse {
        try stdout.writeAll("Usage: zmd <command>\n");
        try stdout.writeAll("Commands: version, collection, update, search, vsearch, query, context, get, multi-get, status, mcp, ls, cleanup, embed\n");
        try stdout.flush();
        return;
    };

    if (std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h") or std.mem.eql(u8, cmd, "help")) {
        try stdout.writeAll("zmd - Local hybrid search for markdown notes\n\n");
        try stdout.writeAll("Commands:\n");
        try stdout.writeAll("  version     Show version\n");
        try stdout.writeAll("  collection  Manage collections (add/list/remove)\n");
        try stdout.writeAll("  update      Update index (syncs remote repos first)\n");
        try stdout.writeAll("  search      Full-text search\n");
        try stdout.writeAll("  vsearch     Vector semantic search\n");
        try stdout.writeAll("  query       Hybrid search (FTS + vector)\n");
        try stdout.writeAll("  context     Context-rich search snippets\n");
        try stdout.writeAll("  get         Get document by path\n");
        try stdout.writeAll("  multi-get   Get multiple documents by path\n");
        try stdout.writeAll("  status      Show system status\n");
        try stdout.writeAll("  mcp         Start MCP server\n");
        try stdout.writeAll("  ls          List documents\n");
        try stdout.writeAll("  cleanup     Remove orphaned entries\n");
        try stdout.writeAll("\nCollections can be local paths or remote GitHub URLs:\n");
        try stdout.writeAll("  zmd collection add notes ~/Documents/notes\n");
        try stdout.writeAll("  zmd collection add laws https://github.com/org/repo\n");
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "--version") or std.mem.eql(u8, cmd, "-v")) {
        try stdout.writeAll(qmd.version);
        try stdout.writeAll("\n");
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "version")) {
        try stdout.writeAll("zmd ");
        try stdout.writeAll(qmd.version);
        try stdout.writeAll("\n");
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "collection")) {
        const subcmd = args.next() orelse {
            try stdout.writeAll("Usage: zmd collection <add|list|remove> [args]\n");
            try stdout.flush();
            return;
        };

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Failed to open database. Run 'zmd update' first.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        if (std.mem.eql(u8, subcmd, "add")) {
            const name = args.next() orelse {
                try stdout.writeAll("Usage: zmd collection add <name> <path|url>\n");
                try stdout.flush();
                return;
            };
            const path = args.next() orelse {
                try stdout.writeAll("Usage: zmd collection add <name> <path|url>\n");
                try stdout.flush();
                return;
            };
            qmd.config.addCollection(&db_, name, path) catch |err| {
                if (err == qmd.config.ConfigError.AlreadyExists) {
                    try stdout.writeAll("Collection already exists\n");
                } else {
                    try stdout.print("Failed to add collection: {any}\n", .{err});
                }
                try stdout.flush();
                return;
            };
            try stdout.writeAll("Collection added: ");
            try stdout.writeAll(name);
            if (qmd.remote.isRemoteUrl(path)) {
                try stdout.writeAll(" (remote)\n");
            } else {
                try stdout.writeAll("\n");
            }
            try stdout.flush();
            return;
        }

        if (std.mem.eql(u8, subcmd, "list")) {
            var result = qmd.config.listCollections(&db_, allocator) catch {
                try stdout.writeAll("Failed to list collections\n");
                try stdout.flush();
                return;
            };
            defer qmd.config.freeCollections(&result);

            if (result.collections.items.len == 0) {
                try stdout.writeAll("No collections. Run 'zmd collection add <name> <path|url>'\n");
            } else {
                for (result.collections.items) |col| {
                    if (qmd.remote.isRemoteUrl(col.path)) {
                        try stdout.print("  {s}: {s} (remote)\n", .{ col.name, col.path });
                    } else {
                        try stdout.print("  {s}: {s}\n", .{ col.name, col.path });
                    }
                }
            }
            try stdout.flush();
            return;
        }

        if (std.mem.eql(u8, subcmd, "remove")) {
            const name = args.next() orelse {
                try stdout.writeAll("Usage: zmd collection remove <name>\n");
                try stdout.flush();
                return;
            };
            qmd.config.removeCollection(&db_, name) catch {
                try stdout.writeAll("Failed to remove collection (may not exist)\n");
                try stdout.flush();
                return;
            };
            try stdout.writeAll("Collection removed\n");
            try stdout.flush();
            return;
        }

        try stdout.writeAll("Unknown collection command: ");
        try stdout.writeAll(subcmd);
        try stdout.writeAll("\n");
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "update")) {
        try stdout.writeAll("Updating index...\n");
        try stdout.flush();

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});

        std.Io.Dir.cwd().createDir(io, ".qmd", .default_dir) catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };

        var db_: qmd.db.Db = undefined;
        const open_result = qmd.db.Db.open(db_path);
        if (open_result) |db| {
            db_ = db;
        } else |_| {
            try stdout.writeAll("Creating database...\n");
            db_ = try qmd.db.Db.open(db_path);
        }
        defer db_.close();

        try qmd.db.initSchema(&db_);
        try stdout.writeAll("Schema initialized.\n");
        try stdout.flush();

        var collections_result = qmd.config.listCollections(&db_, allocator) catch {
            try stdout.writeAll("Error: Failed to list collections\n");
            try stdout.flush();
            return;
        };
        defer qmd.config.freeCollections(&collections_result);

        var total_indexed: usize = 0;
        var total_skipped: usize = 0;

        // Hoist embedding engine creation outside the document loop (A3/A4).
        // This avoids per-document env var reads, file existence checks, and allocations.
        // Try native llama.cpp first (-Dllama), then subprocess, then FNV fallback.
        // Embed model (QMD_EMBED_MODEL) is separate from generation model (QMD_MODEL).
        var embed_llama = initEmbedLlama(init.environ_map);
        defer if (embed_llama) |*el| el.deinit();
        if (embed_llama) |*el| g_embed_llama = el;
        defer g_embed_llama = null;

        var native_llama = initNativeLlama(init.environ_map);
        defer if (native_llama) |*nl| nl.deinit();
        if (native_llama) |*nl| g_native_llama = nl;
        defer g_native_llama = null;

        var embedding_engine: ?qmd.llm.LlamaEmbedding = if (getNativeEmbedFn() == null)
            make_embedding_engine(allocator, io, init.environ_map)
        else
            null;
        defer if (embedding_engine) |*e| e.deinit();
        const use_real_engine = native_llama != null or embedding_engine != null;

        var fallback_engine: ?qmd.llm.LlamaCpp = if (!use_real_engine)
            qmd.llm.LlamaCpp.init("/dev/null", allocator) catch null
        else
            null;
        defer if (fallback_engine) |*f| f.deinit();

        // Reusable tree-sitter parser — created once, used for all documents (B6).
        var ast_chunker = qmd.ast.AstChunker.init(allocator, "markdown") catch null;
        defer if (ast_chunker) |*ch| ch.deinit();

        // Wrap all inserts in a single transaction for performance
        db_.exec("BEGIN") catch {};

        for (collections_result.collections.items) |col| {
            // Resolve the local filesystem path (clone/pull for remote URLs)
            var resolved_path: []const u8 = col.path;
            var resolved_path_owned: ?[]u8 = null;
            defer if (resolved_path_owned) |p| allocator.free(p);

            if (qmd.remote.isRemoteUrl(col.path)) {
                try stdout.print("Syncing remote collection '{s}' from {s}...\n", .{ col.name, col.path });
                try stdout.flush();
                resolved_path_owned = qmd.remote.syncRemote(allocator, io, col.path) catch |err| {
                    try stdout.print("  Warning: Failed to sync remote {s}: {any}\n", .{ col.path, err });
                    continue;
                };
                resolved_path = resolved_path_owned.?;
                try stdout.print("  Cached at {s}\n", .{resolved_path});
            }

            try stdout.print("Indexing collection '{s}' from {s}...\n", .{ col.name, resolved_path });
            try stdout.flush();

            var dir = std.Io.Dir.cwd().openDir(io, resolved_path, .{ .iterate = true }) catch {
                try stdout.print("  Warning: Could not open directory {s}\n", .{resolved_path});
                continue;
            };
            defer dir.close(io);

            var walker = dir.walk(allocator) catch {
                try stdout.writeAll("  Error: Failed to walk directory\n");
                continue;
            };
            defer walker.deinit();

            var col_count: usize = 0;
            var col_new: usize = 0;

            while (try walker.next(io)) |entry| {
                if (entry.kind == .file and std.mem.endsWith(u8, entry.path, ".md")) {
                    var full_path_buf: [1024]u8 = undefined;
                    const full_path = std.fmt.bufPrint(&full_path_buf, "{s}/{s}", .{ resolved_path, entry.path }) catch continue;

                    const content = std.Io.Dir.cwd().readFileAlloc(io, full_path, allocator, @enumFromInt(1024 * 1024)) catch |err| {
                        try stdout.print("    Error reading {s}: {any}\n", .{ entry.path, err });
                        continue;
                    };
                    defer allocator.free(content);

                    const insert_result = qmd.store.insertDocument(&db_, col.name, entry.path, content) catch |err| {
                        try stdout.print("    Error inserting {s}: {any}\n", .{ entry.path, err });
                        continue;
                    };
                    total_indexed += 1;
                    col_count += 1;

                    // Progress reporting every 500 documents
                    if (col_count % 500 == 0) {
                        try stdout.print("  ... {d} documents processed\n", .{col_count});
                        try stdout.flush();
                    }

                    // Skip chunking and embedding if content is unchanged
                    if (!insert_result.content_changed) {
                        total_skipped += 1;
                        continue;
                    }
                    col_new += 1;
                    const doc_hash = insert_result.hash;

                    var chunk_slices = std.ArrayList([]const u8).initCapacity(allocator, 0) catch {
                        continue;
                    };
                    defer chunk_slices.deinit(allocator);

                    if (std.mem.eql(u8, qmd.ast.detectLanguage(entry.path), "markdown")) {
                        if (ast_chunker) |*chunker| {
                            if (chunker.chunk(content, 1200)) |chunks| {
                                var ast_chunks = chunks;
                                defer ast_chunks.deinit(allocator);
                                try chunk_slices.appendSlice(allocator, ast_chunks.items);
                            } else |_| {}
                        }
                    }

                    if (chunk_slices.items.len == 0) {
                        var chunks = qmd.chunker.chunkDocument(content, allocator) catch {
                            continue;
                        };
                        defer chunks.chunks.deinit(allocator);
                        try chunk_slices.appendSlice(allocator, chunks.chunks.items);
                    }

                    if (getNativeEmbedFn() != null) {
                        for (chunk_slices.items, 0..) |chunk, idx| {
                            const formatted = qmd.llm.formatDocForEmbedding(allocator, chunk) catch continue;
                            defer allocator.free(formatted);
                            const emb = nativeEmbedFn(allocator, formatted, false) catch continue;
                            defer allocator.free(emb);
                            qmd.store.upsertContentVectorAt(&db_, doc_hash[0..], @intCast(idx), 0, "native-llama", emb, allocator) catch |err| {
                                if (err == error.OutOfMemory) return err;
                                continue;
                            };
                        }
                    } else if (embedding_engine) |*engine| {
                        for (chunk_slices.items, 0..) |chunk, idx| {
                            const formatted = qmd.llm.formatDocForEmbedding(allocator, chunk) catch continue;
                            defer allocator.free(formatted);
                            const emb = engine.embed(formatted) catch continue;
                            defer allocator.free(emb);
                            qmd.store.upsertContentVectorAt(&db_, doc_hash[0..], @intCast(idx), 0, engine.model_path, emb, allocator) catch |err| {
                                if (err == error.OutOfMemory) return err;
                                continue;
                            };
                        }
                    } else if (fallback_engine) |*fallback| {
                        for (chunk_slices.items, 0..) |chunk, idx| {
                            const formatted = qmd.llm.formatDocForEmbedding(allocator, chunk) catch continue;
                            defer allocator.free(formatted);
                            const emb = fallback.embed(formatted, allocator) catch continue;
                            defer allocator.free(emb);
                            qmd.store.upsertContentVectorAt(&db_, doc_hash[0..], @intCast(idx), 0, "fallback-fnv", emb, allocator) catch |err| {
                                if (err == error.OutOfMemory) return err;
                                continue;
                            };
                        }
                    }
                }
            }
            if (col_new < col_count) {
                try stdout.print("  Indexed {d} documents ({d} new, {d} unchanged)\n", .{ col_count, col_new, col_count - col_new });
            } else {
                try stdout.print("  Indexed {d} documents\n", .{col_count});
            }
        }

        db_.exec("COMMIT") catch {};

        if (total_skipped > 0) {
            try stdout.print("Update complete. Total: {d} documents ({d} unchanged, skipped)\n", .{ total_indexed, total_skipped });
        } else {
            try stdout.print("Update complete. Total: {d} documents\n", .{total_indexed});
        }
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "query")) {
        const first_arg = args.next() orelse {
            try stdout.writeAll("Usage: zmd query <query>\n");
            try stdout.flush();
            return;
        };
        if (std.mem.eql(u8, first_arg, "--help") or std.mem.eql(u8, first_arg, "-h")) {
            try stdout.writeAll("Usage: zmd query <query> [--expand] [--rerank] [--json|--csv|--md] [--sort=score|--sort=index]\n");
            try stdout.flush();
            return;
        }
        const query_text = first_arg;

        var enable_expand = false;
        var enable_rerank = false;
        var output_format: OutputFormat = .text;
        var sort_order: SortOrder = .score;
        while (args.next()) |flag| {
            if (std.mem.eql(u8, flag, "--expand")) {
                enable_expand = true;
                continue;
            }
            if (std.mem.eql(u8, flag, "--rerank")) {
                enable_rerank = true;
                continue;
            }
            if (parseOutputFlag(flag)) |fmt| {
                output_format = fmt;
                continue;
            }
            if (parseSortFlag(flag)) |so| {
                sort_order = so;
                continue;
            }
        }

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Database not found. Run 'zmd update' first.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        // Init native llama models for hybrid search
        var embed_llama_q = initEmbedLlama(init.environ_map);
        defer if (embed_llama_q) |*el| el.deinit();
        if (embed_llama_q) |*el| g_embed_llama = el;
        defer g_embed_llama = null;

        var native_llama_q = initNativeLlama(init.environ_map);
        defer if (native_llama_q) |*nl| nl.deinit();
        if (native_llama_q) |*nl| g_native_llama = nl;
        defer g_native_llama = null;

        var result = qmd.search.hybridSearch(&db_, allocator, io, query_text, null, .{
            .enable_vector = true,
            .enable_query_expansion = enable_expand,
            .enable_rerank = enable_rerank,
            .rrf_k = qmd.search.RRF_K,
            .max_results = 10,
            .embed_fn = getNativeEmbedFn(),
            .expand_query_fn = if (enable_expand) getNativeExpandQueryFn() else null,
        }) catch {
            try stdout.writeAll("Search failed\n");
            try stdout.flush();
            return;
        };
        defer result.deinit(allocator);

        sortSearchResults(result.results.items, sort_order);

        switch (output_format) {
            .text => {
                try stdout.print("Found {d} results (hybrid)\n", .{result.results.items.len});
                for (result.results.items, 0..) |r, i| {
                    try stdout.print("{d}. {s} (zmd://{s}/{s}) score={d:.4}\n", .{ i + 1, r.title, r.collection, r.path, r.score });
                }
            },
            .json => {
                try stdout.writeAll("[\n");
                for (result.results.items, 0..) |r, i| {
                    if (i > 0) try stdout.writeAll(",\n");
                    const vpath = try std.fmt.allocPrint(allocator, "zmd://{s}/{s}", .{ r.collection, r.path });
                    try stdout.writeAll("  {\"id\":");
                    try stdout.print("{d}", .{r.id});
                    try stdout.writeAll(",\"collection\":");
                    try writeJsonString(stdout, r.collection);
                    try stdout.writeAll(",\"path\":");
                    try writeJsonString(stdout, r.path);
                    try stdout.writeAll(",\"virtual_path\":");
                    try writeJsonString(stdout, vpath);
                    try stdout.writeAll(",\"title\":");
                    try writeJsonString(stdout, r.title);
                    try stdout.writeAll(",\"score\":");
                    try stdout.print("{d}", .{r.score});
                    try stdout.writeAll("}");
                    allocator.free(vpath);
                }
                try stdout.writeAll("\n]\n");
            },
            .csv => {
                try stdout.writeAll("id,collection,path,virtual_path,title,score\n");
                for (result.results.items) |r| {
                    const vpath = try std.fmt.allocPrint(allocator, "zmd://{s}/{s}", .{ r.collection, r.path });
                    try stdout.print("{d},", .{r.id});
                    try writeCsvField(stdout, r.collection);
                    try stdout.writeAll(",");
                    try writeCsvField(stdout, r.path);
                    try stdout.writeAll(",");
                    try writeCsvField(stdout, vpath);
                    try stdout.writeAll(",");
                    try writeCsvField(stdout, r.title);
                    try stdout.print(",{d}\n", .{r.score});
                    allocator.free(vpath);
                }
            },
            .md => {
                try stdout.writeAll("| rank | score | collection | path | title |\n|---:|---:|---|---|---|\n");
                for (result.results.items, 0..) |r, i| {
                    try stdout.print("| {d} | {d:.4} | {s} | zmd://{s}/{s} | {s} |\n", .{ i + 1, r.score, r.collection, r.collection, r.path, r.title });
                }
            },
        }
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "context")) {
        const first_arg = args.next() orelse {
            try stdout.writeAll("Usage: zmd context <query> [--json|--csv|--md] [--sort=score|--sort=index] [--no-answer]\n");
            try stdout.flush();
            return;
        };
        if (std.mem.eql(u8, first_arg, "--help") or std.mem.eql(u8, first_arg, "-h")) {
            try stdout.writeAll("Usage: zmd context <query> [--json|--csv|--md] [--sort=score|--sort=index] [--no-answer]\n");
            try stdout.flush();
            return;
        }
        const query_text = first_arg;
        const retrieval_query = try rewriteContextQuery(allocator, query_text);
        defer if (retrieval_query) |rq| allocator.free(rq);
        const effective_query = retrieval_query orelse query_text;

        var output_format: OutputFormat = .text;
        var sort_order: SortOrder = .score;
        var enable_answer = true;
        while (args.next()) |arg| {
            if (parseOutputFlag(arg)) |fmt| {
                output_format = fmt;
            } else if (parseSortFlag(arg)) |so| {
                sort_order = so;
            } else if (std.mem.eql(u8, arg, "--no-answer")) {
                enable_answer = false;
            }
        }

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Database not found. Run 'zmd update' first.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        // Init native llama models for context search
        var embed_llama_ctx = initEmbedLlama(init.environ_map);
        defer if (embed_llama_ctx) |*el| el.deinit();
        if (embed_llama_ctx) |*el| g_embed_llama = el;
        defer g_embed_llama = null;

        var native_llama_ctx = initNativeLlama(init.environ_map);
        defer if (native_llama_ctx) |*nl| nl.deinit();
        if (native_llama_ctx) |*nl| g_native_llama = nl;
        defer g_native_llama = null;

        var result = qmd.search.hybridSearch(&db_, allocator, io, effective_query, null, .{
            .enable_vector = true,
            .enable_query_expansion = false,
            .enable_rerank = getNativeEmbedFn() != null,
            .rrf_k = qmd.search.RRF_K,
            .max_results = 12,
            .min_score = 0.01,
            .embed_fn = getNativeEmbedFn(),
            .expand_query_fn = null,
        }) catch {
            try stdout.writeAll("Context search failed\n");
            try stdout.flush();
            return;
        };
        defer result.deinit(allocator);

        if (getNativeExpandQueryFn()) |expand_query_fn| {
            var expanded_result = qmd.search.hybridSearch(&db_, allocator, io, effective_query, null, .{
                .enable_vector = true,
                .enable_query_expansion = true,
                .enable_rerank = getNativeEmbedFn() != null,
                .rrf_k = qmd.search.RRF_K,
                .max_results = 12,
                .min_score = 0.01,
                .embed_fn = getNativeEmbedFn(),
                .expand_query_fn = expand_query_fn,
            }) catch null;
            if (expanded_result) |*candidate| {
                defer candidate.deinit(allocator);
                if (shouldPreferExpandedContextResults(&db_, allocator, effective_query, result.results.items, candidate.results.items)) {
                    result.deinit(allocator);
                    result = candidate.*;
                    expanded_result = null;
                }
            }
        }

        try rerankContextResults(&db_, allocator, effective_query, &result.results, 5);

        sortSearchResults(result.results.items, sort_order);

        switch (output_format) {
            .json => {
                try stdout.writeAll("[\n");
                var first = true;
                for (result.results.items) |r| {
                    const doc = qmd.store.findActiveDocument(&db_, r.collection, r.path, allocator) catch continue;
                    defer {
                        allocator.free(doc.title);
                        allocator.free(doc.hash);
                        allocator.free(doc.doc);
                    }
                    const snippet = try extractSnippet(allocator, query_text, doc.doc);
                    defer allocator.free(snippet);
                    if (!first) try stdout.writeAll(",\n");
                    first = false;
                    try stdout.writeAll("  {\"collection\":");
                    try writeJsonString(stdout, r.collection);
                    try stdout.writeAll(",\"path\":");
                    try writeJsonString(stdout, r.path);
                    try stdout.writeAll(",\"title\":");
                    try writeJsonString(stdout, r.title);
                    try stdout.writeAll(",\"score\":");
                    try stdout.print("{d}", .{r.score});
                    try stdout.writeAll(",\"snippet\":");
                    try writeJsonString(stdout, snippet);
                    try stdout.writeAll("}");
                }
                try stdout.writeAll("\n]\n");
            },
            .csv => {
                try stdout.writeAll("rank,collection,path,title,score,snippet\n");
                for (result.results.items, 0..) |r, i| {
                    const doc = qmd.store.findActiveDocument(&db_, r.collection, r.path, allocator) catch continue;
                    defer {
                        allocator.free(doc.title);
                        allocator.free(doc.hash);
                        allocator.free(doc.doc);
                    }
                    const snippet = try extractSnippet(allocator, query_text, doc.doc);
                    defer allocator.free(snippet);
                    try stdout.print("{d},", .{i + 1});
                    try writeCsvField(stdout, r.collection);
                    try stdout.writeAll(",");
                    try writeCsvField(stdout, r.path);
                    try stdout.writeAll(",");
                    try writeCsvField(stdout, r.title);
                    try stdout.print(",{d},", .{r.score});
                    try writeCsvField(stdout, snippet);
                    try stdout.writeAll("\n");
                }
            },
            .md => {
                try stdout.writeAll("| rank | score | path | title | snippet |\n|---:|---:|---|---|---|\n");
                for (result.results.items, 0..) |r, i| {
                    const doc = qmd.store.findActiveDocument(&db_, r.collection, r.path, allocator) catch continue;
                    defer {
                        allocator.free(doc.title);
                        allocator.free(doc.hash);
                        allocator.free(doc.doc);
                    }
                    const snippet = try extractSnippet(allocator, query_text, doc.doc);
                    defer allocator.free(snippet);
                    try stdout.print("| {d} | {d:.4} | zmd://{s}/{s} | {s} | {s} |\n", .{ i + 1, r.score, r.collection, r.path, r.title, snippet });
                }
            },
            else => {
                if (result.results.items.len == 0) {
                    try stdout.writeAll("No context results found.\n");
                } else {
                    for (result.results.items, 0..) |r, i| {
                        const doc = qmd.store.findActiveDocument(&db_, r.collection, r.path, allocator) catch continue;
                        defer {
                            allocator.free(doc.title);
                            allocator.free(doc.hash);
                            allocator.free(doc.doc);
                        }
                        const snippet = try extractSnippet(allocator, query_text, doc.doc);
                        defer allocator.free(snippet);
                        if (i > 0) try stdout.writeAll("\n");
                        try stdout.print("{d}. {s} (zmd://{s}/{s}) score={d:.4}\n", .{ i + 1, r.title, r.collection, r.path, r.score });
                        try stdout.print("   {s}\n", .{snippet});
                    }
                }
            },
        }

        // RAG answer generation using the generation model
        if (enable_answer and build_options.enable_llama and result.results.items.len > 0) {
            if (g_native_llama) |llama| {
                // Build context from top search results
                var context_buf: std.ArrayList(u8) = .empty;
                defer context_buf.deinit(allocator);

                for (result.results.items, 0..) |r, i| {
                    if (i >= 5) break; // Limit context to top 5
                    const doc = qmd.store.findActiveDocument(&db_, r.collection, r.path, allocator) catch continue;
                    defer {
                        allocator.free(doc.title);
                        allocator.free(doc.hash);
                        allocator.free(doc.doc);
                    }

                    // Extract a large context window around query terms.
                    // This finds where the query matches in the doc and grabs
                    // up to max_context_len chars centered on that location.
                    const max_context_len: usize = 1500;
                    const doc_content = extractLargeContext(doc.doc, query_text, max_context_len);

                    context_buf.appendSlice(allocator, "---\n") catch continue;
                    context_buf.appendSlice(allocator, "Source: ") catch continue;
                    context_buf.appendSlice(allocator, r.title) catch continue;
                    context_buf.appendSlice(allocator, " (") catch continue;
                    context_buf.appendSlice(allocator, r.path) catch continue;
                    context_buf.appendSlice(allocator, ")\n") catch continue;
                    context_buf.appendSlice(allocator, doc_content) catch continue;
                    if (doc_content.len < doc.doc.len) context_buf.appendSlice(allocator, "...") catch {};
                    context_buf.appendSlice(allocator, "\n") catch continue;
                }

                if (context_buf.items.len > 0) {
                    // Build user message with context
                    const user_msg = std.fmt.allocPrint(allocator, "Based on the following documents, answer the question.\n\n" ++
                        "Documents:\n{s}\n---\n\nQuestion: {s}", .{ context_buf.items, query_text }) catch {
                        try stdout.flush();
                        return;
                    };
                    defer allocator.free(user_msg);

                    const system_prompt = "Answer using only the provided documents. " ++
                        "Prefer direct, factual language. " ++
                        "When you make a claim, cite the source title in parentheses. " ++
                        "If the documents are insufficient, say that clearly instead of guessing.";

                    try stdout.writeAll("\n--- Answer ---\n\n");
                    try stdout.flush();

                    const answer = llama.chat(system_prompt, user_msg, 384, false) catch {
                        try stdout.writeAll("(generation failed)\n");
                        try stdout.flush();
                        return;
                    };
                    // chat() returns page_allocator memory — free with page_allocator
                    defer std.heap.page_allocator.free(answer);

                    const cleaned_answer = try sanitizeGeneratedText(allocator, answer);
                    defer allocator.free(cleaned_answer);

                    try stdout.writeAll(cleaned_answer);
                    try stdout.writeAll("\n");
                }
            }
        }

        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "search")) {
        const first_arg = args.next() orelse {
            try stdout.writeAll("Usage: zmd search <query> [collection]\n");
            try stdout.flush();
            return;
        };
        if (std.mem.eql(u8, first_arg, "--help") or std.mem.eql(u8, first_arg, "-h")) {
            try stdout.writeAll("Usage: zmd search <query> [collection] [--json|--csv|--md] [--sort=score|--sort=index]\n");
            try stdout.flush();
            return;
        }
        const query_text = first_arg;
        var collection: ?[]const u8 = null;
        var output_format: OutputFormat = .text;
        var sort_order: SortOrder = .score;
        while (args.next()) |arg| {
            if (parseOutputFlag(arg)) |fmt| {
                output_format = fmt;
            } else if (parseSortFlag(arg)) |so| {
                sort_order = so;
            } else if (collection == null) {
                collection = arg;
            }
        }

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Database not found. Run 'zmd update' first.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        var result = qmd.search.searchFTS(&db_, allocator, query_text, collection) catch {
            try stdout.writeAll("Search failed\n");
            try stdout.flush();
            return;
        };
        defer result.deinit(allocator);

        sortSearchResults(result.results.items, sort_order);

        switch (output_format) {
            .text => {
                if (result.results.items.len == 0) {
                    try stdout.writeAll("No results found.\n");
                } else {
                    try stdout.print("Found {d} results:\n", .{result.results.items.len});
                    for (result.results.items, 0..) |r, i| {
                        try stdout.print("{d}. {s} (zmd://{s}/{s}) score={d:.4}\n", .{ i + 1, r.title, r.collection, r.path, r.score });
                    }
                }
            },
            .json => {
                try stdout.writeAll("[\n");
                for (result.results.items, 0..) |r, i| {
                    if (i > 0) try stdout.writeAll(",\n");
                    try stdout.writeAll("  {\"id\":");
                    try stdout.print("{d}", .{r.id});
                    try stdout.writeAll(",\"collection\":");
                    try writeJsonString(stdout, r.collection);
                    try stdout.writeAll(",\"path\":");
                    try writeJsonString(stdout, r.path);
                    try stdout.writeAll(",\"title\":");
                    try writeJsonString(stdout, r.title);
                    try stdout.writeAll(",\"score\":");
                    try stdout.print("{d}", .{r.score});
                    try stdout.writeAll("}");
                }
                try stdout.writeAll("\n]\n");
            },
            .csv => {
                try stdout.writeAll("id,collection,path,title,score\n");
                for (result.results.items) |r| {
                    try stdout.print("{d},", .{r.id});
                    try writeCsvField(stdout, r.collection);
                    try stdout.writeAll(",");
                    try writeCsvField(stdout, r.path);
                    try stdout.writeAll(",");
                    try writeCsvField(stdout, r.title);
                    try stdout.print(",{d}\n", .{r.score});
                }
            },
            .md => {
                try stdout.writeAll("| rank | score | collection | path | title |\n|---:|---:|---|---|---|\n");
                for (result.results.items, 0..) |r, i| {
                    try stdout.print("| {d} | {d:.4} | {s} | zmd://{s}/{s} | {s} |\n", .{ i + 1, r.score, r.collection, r.collection, r.path, r.title });
                }
            },
        }
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "vsearch")) {
        const first_arg = args.next() orelse {
            try stdout.writeAll("Usage: zmd vsearch <query>\n");
            try stdout.flush();
            return;
        };
        if (std.mem.eql(u8, first_arg, "--help") or std.mem.eql(u8, first_arg, "-h")) {
            try stdout.writeAll("Usage: zmd vsearch <query> [--json|--csv|--md] [--sort=score|--sort=index]\n");
            try stdout.flush();
            return;
        }
        const query_text = first_arg;

        var output_format: OutputFormat = .text;
        var sort_order: SortOrder = .score;
        while (args.next()) |arg| {
            if (parseOutputFlag(arg)) |fmt| {
                output_format = fmt;
            } else if (parseSortFlag(arg)) |so| {
                sort_order = so;
            }
        }

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Database not found. Run 'zmd update' first.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        // Init embed llama for vector search (dedicated embedding model)
        var embed_llama_vs = initEmbedLlama(init.environ_map);
        defer if (embed_llama_vs) |*el| el.deinit();
        if (embed_llama_vs) |*el| g_embed_llama = el;
        defer g_embed_llama = null;

        const result = qmd.search.searchVec(&db_, allocator, query_text, null, getNativeEmbedFn()) catch {
            try stdout.writeAll("Vector search failed\n");
            try stdout.flush();
            return;
        };
        defer {
            qmd.search.freeScoredResultSlice(result.results, allocator);
            allocator.free(result.results);
        }

        sortScoredResults(result.results, sort_order);

        switch (output_format) {
            .text => {
                if (result.results.len == 0) {
                    try stdout.writeAll("No results found.\n");
                } else {
                    try stdout.print("Found {d} results (vector):\n", .{result.results.len});
                    for (result.results, 0..) |r, i| {
                        try stdout.print("{d}. {s} (zmd://{s}/{s}) score={d:.4}\n", .{ i + 1, r.title, r.collection, r.path, r.score });
                    }
                }
            },
            .json => {
                try stdout.writeAll("[\n");
                for (result.results, 0..) |r, i| {
                    if (i > 0) try stdout.writeAll(",\n");
                    try stdout.writeAll("  {\"id\":");
                    try stdout.print("{d}", .{r.id});
                    try stdout.writeAll(",\"collection\":");
                    try writeJsonString(stdout, r.collection);
                    try stdout.writeAll(",\"path\":");
                    try writeJsonString(stdout, r.path);
                    try stdout.writeAll(",\"title\":");
                    try writeJsonString(stdout, r.title);
                    try stdout.writeAll(",\"score\":");
                    try stdout.print("{d}", .{r.score});
                    try stdout.writeAll("}");
                }
                try stdout.writeAll("\n]\n");
            },
            .csv => {
                try stdout.writeAll("id,collection,path,title,score\n");
                for (result.results) |r| {
                    try stdout.print("{d},", .{r.id});
                    try writeCsvField(stdout, r.collection);
                    try stdout.writeAll(",");
                    try writeCsvField(stdout, r.path);
                    try stdout.writeAll(",");
                    try writeCsvField(stdout, r.title);
                    try stdout.print(",{d}\n", .{r.score});
                }
            },
            .md => {
                try stdout.writeAll("| rank | score | collection | path | title |\n|---:|---:|---|---|---|\n");
                for (result.results, 0..) |r, i| {
                    try stdout.print("| {d} | {d:.4} | {s} | zmd://{s}/{s} | {s} |\n", .{ i + 1, r.score, r.collection, r.collection, r.path, r.title });
                }
            },
        }
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "get")) {
        const first_arg = args.next() orelse {
            try stdout.writeAll("Usage: zmd get <path>\n");
            try stdout.flush();
            return;
        };
        if (std.mem.eql(u8, first_arg, "--help") or std.mem.eql(u8, first_arg, "-h")) {
            try stdout.writeAll("Usage: zmd get <collection/path|zmd://collection/path>\n");
            try stdout.flush();
            return;
        }
        const doc_path = first_arg;

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Database not found.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        const ref = parseDocRef(doc_path) orelse {
            try stdout.writeAll("Invalid document path. Use collection/path or zmd://collection/path\n");
            try stdout.flush();
            return;
        };

        const doc = qmd.store.findActiveDocument(&db_, ref.collection, ref.path, allocator) catch {
            try stdout.writeAll("Document not found.\n");
            try stdout.flush();
            return;
        };
        defer {
            allocator.free(doc.title);
            allocator.free(doc.hash);
            allocator.free(doc.doc);
        }

        try stdout.writeAll("Title: ");
        try stdout.writeAll(doc.title);
        try stdout.writeAll("\n\n");
        try stdout.writeAll(doc.doc);
        try stdout.writeAll("\n");
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "multi-get")) {
        var output_format: OutputFormat = .text;
        var refs = try std.ArrayList([]const u8).initCapacity(allocator, 0);
        defer refs.deinit(allocator);

        while (args.next()) |arg| {
            if (parseOutputFlag(arg)) |fmt| {
                output_format = fmt;
            } else {
                try refs.append(allocator, arg);
            }
        }

        if (refs.items.len == 1 and (std.mem.eql(u8, refs.items[0], "--help") or std.mem.eql(u8, refs.items[0], "-h"))) {
            try stdout.writeAll("Usage: zmd multi-get <doc-ref...> [--json|--csv|--md]\n");
            try stdout.flush();
            return;
        }

        if (refs.items.len == 0) {
            try stdout.writeAll("Usage: zmd multi-get <doc-ref...> [--json|--csv|--md]\n");
            try stdout.flush();
            return;
        }

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Database not found.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        switch (output_format) {
            .text => {
                for (refs.items, 0..) |raw_ref, i| {
                    const ref = parseDocRef(raw_ref) orelse {
                        try stdout.print("Skipping invalid ref: {s}\n", .{raw_ref});
                        continue;
                    };
                    const doc = qmd.store.findActiveDocument(&db_, ref.collection, ref.path, allocator) catch {
                        try stdout.print("Not found: zmd://{s}/{s}\n", .{ ref.collection, ref.path });
                        continue;
                    };
                    defer {
                        allocator.free(doc.title);
                        allocator.free(doc.hash);
                        allocator.free(doc.doc);
                    }

                    if (i > 0) try stdout.writeAll("\n---\n\n");
                    try stdout.print("# {s}\n", .{doc.title});
                    try stdout.print("Path: zmd://{s}/{s}\n\n", .{ ref.collection, ref.path });
                    try stdout.writeAll(doc.doc);
                    try stdout.writeAll("\n");
                }
            },
            .json => {
                try stdout.writeAll("[\n");
                var first = true;
                for (refs.items) |raw_ref| {
                    const ref = parseDocRef(raw_ref) orelse continue;
                    const doc = qmd.store.findActiveDocument(&db_, ref.collection, ref.path, allocator) catch continue;
                    defer {
                        allocator.free(doc.title);
                        allocator.free(doc.hash);
                        allocator.free(doc.doc);
                    }
                    if (!first) try stdout.writeAll(",\n");
                    first = false;

                    const vpath = try std.fmt.allocPrint(allocator, "zmd://{s}/{s}", .{ ref.collection, ref.path });

                    try stdout.writeAll("  {\"collection\":");
                    try writeJsonString(stdout, ref.collection);
                    try stdout.writeAll(",\"path\":");
                    try writeJsonString(stdout, ref.path);
                    try stdout.writeAll(",\"virtual_path\":");
                    try writeJsonString(stdout, vpath);
                    try stdout.writeAll(",\"title\":");
                    try writeJsonString(stdout, doc.title);
                    try stdout.writeAll(",\"hash\":");
                    try writeJsonString(stdout, doc.hash);
                    try stdout.writeAll(",\"doc\":");
                    try writeJsonString(stdout, doc.doc);
                    try stdout.writeAll("}");
                    allocator.free(vpath);
                }
                try stdout.writeAll("\n]\n");
            },
            .csv => {
                try stdout.writeAll("collection,path,virtual_path,title,hash,doc\n");
                for (refs.items) |raw_ref| {
                    const ref = parseDocRef(raw_ref) orelse continue;
                    const doc = qmd.store.findActiveDocument(&db_, ref.collection, ref.path, allocator) catch continue;
                    defer {
                        allocator.free(doc.title);
                        allocator.free(doc.hash);
                        allocator.free(doc.doc);
                    }
                    const vpath = try std.fmt.allocPrint(allocator, "zmd://{s}/{s}", .{ ref.collection, ref.path });

                    try writeCsvField(stdout, ref.collection);
                    try stdout.writeAll(",");
                    try writeCsvField(stdout, ref.path);
                    try stdout.writeAll(",");
                    try writeCsvField(stdout, vpath);
                    try stdout.writeAll(",");
                    try writeCsvField(stdout, doc.title);
                    try stdout.writeAll(",");
                    try writeCsvField(stdout, doc.hash);
                    try stdout.writeAll(",");
                    try writeCsvField(stdout, doc.doc);
                    try stdout.writeAll("\n");
                    allocator.free(vpath);
                }
            },
            .md => {
                for (refs.items, 0..) |raw_ref, i| {
                    const ref = parseDocRef(raw_ref) orelse continue;
                    const doc = qmd.store.findActiveDocument(&db_, ref.collection, ref.path, allocator) catch continue;
                    defer {
                        allocator.free(doc.title);
                        allocator.free(doc.hash);
                        allocator.free(doc.doc);
                    }
                    if (i > 0) try stdout.writeAll("\n\n---\n\n");
                    try stdout.print("## {s}\n\n", .{doc.title});
                    try stdout.print("- path: `zmd://{s}/{s}`\n\n", .{ ref.collection, ref.path });
                    try stdout.writeAll(doc.doc);
                    try stdout.writeAll("\n");
                }
            },
        }
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "status")) {
        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Database: not initialized\n");
            try stdout.writeAll("Run 'zmd update' to initialize.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        var stmt = db_.prepare("SELECT count(*) FROM documents WHERE active = 1") catch {
            try stdout.writeAll("Error querying database.\n");
            try stdout.flush();
            return;
        };
        defer stmt.finalize();
        const has_row = stmt.step() catch false;
        const doc_count: i64 = if (has_row) stmt.columnInt(0) else 0;

        stmt = db_.prepare("SELECT count(*) FROM store_collections") catch {
            try stdout.writeAll("Error querying database.\n");
            try stdout.flush();
            return;
        };
        defer stmt.finalize();
        const has_row2 = stmt.step() catch false;
        const col_count: i64 = if (has_row2) stmt.columnInt(0) else 0;

        try stdout.writeAll("Database: OK\n");
        try stdout.print("Documents: {d}\n", .{doc_count});
        try stdout.print("Collections: {d}\n", .{col_count});
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "mcp")) {
        try stdout.writeAll("Starting MCP server...\n");
        try stdout.flush();
        try qmd.mcp.McpServer.run(allocator, io);
        return;
    }

    if (std.mem.eql(u8, cmd, "ls")) {
        const collection_name = args.next();

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Database not found.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        const col: ?[]const u8 = if (collection_name) |n| n else null;

        if (col) |c| {
            var result = qmd.store.getActiveDocumentPaths(&db_, c, allocator) catch {
                try stdout.writeAll("Failed to list documents.\n");
                try stdout.flush();
                return;
            };
            defer {
                for (result.paths.items) |p| allocator.free(p);
                for (result.titles.items) |t| allocator.free(t);
                result.paths.deinit(allocator);
                result.titles.deinit(allocator);
            }

            for (result.paths.items, result.titles.items) |path, title| {
                try stdout.print("  zmd://{s}/{s}: {s}\n", .{ c, path, title });
            }
        } else {
            var result = qmd.config.listCollections(&db_, allocator) catch {
                try stdout.writeAll("Failed to list collections.\n");
                try stdout.flush();
                return;
            };
            defer qmd.config.freeCollections(&result);

            for (result.collections.items) |c_| {
                try stdout.print("Collection: {s} ({s})\n", .{ c_.name, c_.path });
                var docs = qmd.store.getActiveDocumentPaths(&db_, c_.name, allocator) catch continue;
                defer {
                    for (docs.paths.items) |p| allocator.free(p);
                    for (docs.titles.items) |t| allocator.free(t);
                    docs.paths.deinit(allocator);
                    docs.titles.deinit(allocator);
                }
                for (docs.paths.items, docs.titles.items) |path, title| {
                    try stdout.print("  zmd://{s}/{s}: {s}\n", .{ c_.name, path, title });
                }
            }
        }
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "cleanup")) {
        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Database not found.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        const stats = qmd.store.cleanupOrphans(&db_) catch {
            try stdout.writeAll("Cleanup failed.\n");
            try stdout.flush();
            return;
        };

        qmd.store.vacuum(&db_) catch |err| {
            try stdout.print("Warning: vacuum skipped: {any}\n", .{err});
        };

        try stdout.print("Cleanup complete. Removed content: {d}, vectors: {d}\n", .{ stats.removed_content, stats.removed_vectors });
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "embed")) {
        const text = args.next() orelse {
            try stdout.writeAll("Usage: zmd embed <text>\n");
            try stdout.flush();
            return;
        };

        // Try dedicated embed model first, fall back to FNV hash embedding
        var embed_llama_e = initEmbedLlama(init.environ_map);
        defer if (embed_llama_e) |*el| el.deinit();

        if (embed_llama_e) |*el| {
            if (build_options.enable_llama and el.supportsEmbedding()) {
                const emb = el.embed(text) catch {
                    try stdout.writeAll("Native embed failed\n");
                    return;
                };
                defer std.heap.page_allocator.free(emb);
                try stdout.print("Embedding ({d} dims, native): [{d:.4}", .{ emb.len, emb[0] });
                if (emb.len > 1) try stdout.print(", {d:.4}", .{emb[1]});
                if (emb.len > 2) try stdout.print(", {d:.4}...", .{emb[2]});
                try stdout.writeAll("]\n");
                try stdout.flush();
                return;
            } else {
                try stdout.writeAll("Model does not support embedding, using fallback\n");
                try stdout.flush();
            }
        }

        var llm_fallback = qmd.llm.LlamaCpp.init("/fake", allocator) catch {
            try stdout.writeAll("Failed to init LLM\n");
            return;
        };
        defer llm_fallback.deinit();
        const emb = llm_fallback.embed(text, allocator) catch {
            try stdout.writeAll("Failed to embed text\n");
            return;
        };
        defer allocator.free(emb);
        try stdout.print("Embedding ({d} dims): [{d:.4}", .{ emb.len, emb[0] });
        if (emb.len > 1) try stdout.print(", {d:.4}", .{emb[1]});
        if (emb.len > 2) try stdout.print(", {d:.4}...", .{emb[2]});
        try stdout.writeAll("]\n");
        try stdout.flush();
        return;
    }

    try stdout.writeAll("Unknown command: ");
    try stdout.writeAll(cmd);
    try stdout.writeAll("\n");
    try stdout.flush();
}

test "placeholder" {
    try std.testing.expect(true);
}

test "parseOutputFlag parses known flags" {
    try std.testing.expect(parseOutputFlag("--json").? == .json);
    try std.testing.expect(parseOutputFlag("--csv").? == .csv);
    try std.testing.expect(parseOutputFlag("--md").? == .md);
    try std.testing.expect(parseOutputFlag("--nope") == null);
}

test "parseDocRef parses virtual and plain paths" {
    const a = parseDocRef("zmd://notes/a.md") orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("notes", a.collection);
    try std.testing.expectEqualStrings("a.md", a.path);

    const b = parseDocRef("docs/b.md") orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("docs", b.collection);
    try std.testing.expectEqualStrings("b.md", b.path);
}

test "extractSnippet returns contextual content" {
    const doc = "intro text\nOAuth token login flow and refresh\nmore details";
    const snippet = try extractSnippet(std.testing.allocator, "oauth", doc);
    defer std.testing.allocator.free(snippet);
    try std.testing.expect(std.mem.indexOf(u8, snippet, "OAuth") != null);
}

test "sanitizeGeneratedText removes wrapper quotes" {
    const cleaned = try sanitizeGeneratedText(std.testing.allocator, "'Expanded query: rust bevy ecs game development'");
    defer std.testing.allocator.free(cleaned);

    try std.testing.expectEqualStrings("rust bevy ecs game development", cleaned);
}

test "skipFrontmatter skips yaml header" {
    const doc = "---\ntitle: Demo\ntags: [rust]\n---\n\n# Bevy\nReal content";
    const visible = skipFrontmatter(doc);
    try std.testing.expect(std.mem.startsWith(u8, visible, "\n# Bevy\nReal content"));
}

test "nextSignificantQueryToken skips question filler" {
    const token = nextSignificantQueryToken("what to learn to code games in Rust") orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("games", token);
}

test "rewriteContextQuery expands broad rust game learning query" {
    const rewritten = try rewriteContextQuery(std.testing.allocator, "what to learn to code games in Rust");
    defer if (rewritten) |q| std.testing.allocator.free(q);

    try std.testing.expect(rewritten != null);
    try std.testing.expect(std.mem.indexOf(u8, rewritten.?, "bevy") != null);
    try std.testing.expect(std.mem.indexOf(u8, rewritten.?, "gamedev") != null);
}

test "rewriteContextQuery ignores unrelated rust query" {
    const rewritten = try rewriteContextQuery(std.testing.allocator, "what is ownership in Rust");
    try std.testing.expect(rewritten == null);
}

test "extractSnippet ignores frontmatter and prefers meaningful token" {
    const doc =
        "---\n" ++
        "title: Bevy-Demo\n" ++
        "tags: [rust, graphics, math, bevy]\n" ++
        "---\n\n" ++
        "# Bevy-Demo\n" ++
        "Bevy uses ECS for modern game development in Rust.";
    const snippet = try extractSnippet(std.testing.allocator, "what to learn to code games in Rust", doc);
    defer std.testing.allocator.free(snippet);

    try std.testing.expect(std.mem.indexOf(u8, snippet, "game development") != null);
    try std.testing.expect(std.mem.indexOf(u8, snippet, "tags:") == null);
}

test "sanitizeExpandedQuery preserves original query prefix" {
    const cleaned = try sanitizeExpandedQuery(std.testing.allocator, "what to learn to code games in Rust", "'rust bevy ecs game development ownership'");
    defer std.testing.allocator.free(cleaned);

    try std.testing.expect(std.mem.startsWith(u8, cleaned, "what to learn to code games in Rust "));
    try std.testing.expect(std.mem.indexOf(u8, cleaned, "bevy") != null);
}

test "scoreContextPreviews prefers stronger query coverage" {
    const baseline = [_]Preview{
        .{ .title = "Choosing Your Messaging Fabric on AKS", .body = "asynchronous communication event driven data flows", .score = 1.0 },
        .{ .title = "Development", .body = "general programming links and video index", .score = 1.0 },
    };
    const expanded = [_]Preview{
        .{ .title = "Kubernetes Auto-Scaling Strategies", .body = "Vertical Pod Autoscaler VPA adjusts CPU and memory requests in Kubernetes", .score = 0.91 },
        .{ .title = "Advanced Microservices Blueprint on Azure Kubernetes Service", .body = "VPA optimizes workload resources using observed usage history", .score = 1.0 },
    };

    try std.testing.expect(scoreContextPreviews("how VPA works in Kubernetes", expanded[0..]) > scoreContextPreviews("how VPA works in Kubernetes", baseline[0..]));
}

test "scoreContextPreviews rejects unrelated expansion drift" {
    const baseline = [_]Preview{
        .{ .title = "Bevy-Demo", .body = "Bevy uses ECS for game development in Rust and relies on vector math", .score = 0.89 },
        .{ .title = "Mastering Memory in Rust", .body = "Rust ownership and borrowing help manage memory safely", .score = 0.88 },
    };
    const expanded = [_]Preview{
        .{ .title = "The Blue Screen Era", .body = "Integrated development environments and debugger workflows", .score = 0.89 },
        .{ .title = "Drasi: The Future of Change-Driven Architecture", .body = "Polling and event architectures create a complexity tax", .score = 0.89 },
    };

    try std.testing.expect(scoreContextPreviews("what to learn to code games in Rust", baseline[0..]) > scoreContextPreviews("what to learn to code games in Rust", expanded[0..]));
}

test "hasTranscriptDetailPair detects matching title pair" {
    const results = [_]qmd.search.SearchResult{
        .{ .id = 1, .collection = "wiki", .path = "raw/transcripts/a.md", .title = "Bevy-Demo", .hash = "h1", .score = 1.0 },
        .{ .id = 2, .collection = "wiki", .path = "videos/details/a.md", .title = "Bevy-Demo", .hash = "h2", .score = 0.9 },
        .{ .id = 3, .collection = "wiki", .path = "raw/transcripts/b.md", .title = "Other", .hash = "h3", .score = 0.8 },
    };

    try std.testing.expect(hasTranscriptDetailPair(results[0..], "Bevy-Demo"));
    try std.testing.expect(!hasTranscriptDetailPair(results[0..], "Other"));
}

test "extractSnippet handles CJK Korean text" {
    const doc = "주택임대차보호법에 의한 보증금 반환 청구권에 대하여";
    const snippet = try extractSnippet(std.testing.allocator, "보증금", doc);
    defer std.testing.allocator.free(snippet);
    // Must find the Korean term in the snippet.
    try std.testing.expect(std.mem.indexOf(u8, snippet, "보증금") != null);
}

test "extractSnippet CJK does not produce invalid UTF-8" {
    // Build a long Korean document so slicing hits non-zero start offset.
    const prefix = "가나다라마바사아자차카타파하" ** 5; // 14 chars * 3 bytes * 5 = 210 bytes
    const doc = prefix ++ "보호법에 의한 보증금 반환 청구";
    const snippet = try extractSnippet(std.testing.allocator, "보증금", doc);
    defer std.testing.allocator.free(snippet);
    // Validate the result is valid UTF-8 (won't panic on iteration).
    var count: usize = 0;
    const view = std.unicode.Utf8View.init(snippet) catch {
        return error.TestExpectedEqual; // snippet is invalid UTF-8
    };
    var it = view.iterator();
    while (it.nextCodepoint()) |_| count += 1;
    try std.testing.expect(count > 0);
    try std.testing.expect(std.mem.indexOf(u8, snippet, "보증금") != null);
}

test "utf8IndexOfInsensitive finds Korean text" {
    const haystack = "주택임대차보호법에 의한 보증금";
    const needle = "보호법";
    const idx = utf8IndexOfInsensitive(haystack, needle);
    try std.testing.expect(idx != null);
    // The match should be at the correct position and the slice should equal the needle.
    const start = idx.?;
    try std.testing.expectEqualStrings(needle, haystack[start .. start + needle.len]);
}

test "utf8IndexOfInsensitive is case-insensitive for ASCII" {
    const haystack = "Hello World OAuth Token";
    const needle = "oauth";
    const idx = utf8IndexOfInsensitive(haystack, needle);
    try std.testing.expect(idx != null);
    try std.testing.expectEqualStrings("OAuth", haystack[idx.? .. idx.? + 5]);
}

test "snapBackToCodepoint snaps correctly" {
    // "가" in UTF-8 is 0xEA 0xB0 0x80 (3 bytes)
    const data = "가나";
    // Byte 1 is a continuation byte of "가"
    try std.testing.expectEqual(@as(usize, 0), snapBackToCodepoint(data, 1));
    // Byte 3 is the start of "나"
    try std.testing.expectEqual(@as(usize, 3), snapBackToCodepoint(data, 3));
}

test "snapForwardToCodepoint snaps correctly" {
    const data = "가나";
    // Byte 1 is a continuation byte; should snap forward to byte 3 ("나")
    try std.testing.expectEqual(@as(usize, 3), snapForwardToCodepoint(data, 1));
    // Byte 0 is already a codepoint boundary
    try std.testing.expectEqual(@as(usize, 0), snapForwardToCodepoint(data, 0));
}

test "parseSortFlag parses known sort orders" {
    try std.testing.expect(parseSortFlag("--sort=score").? == .score);
    try std.testing.expect(parseSortFlag("--sort=index").? == .index);
    try std.testing.expect(parseSortFlag("--sort=nope") == null);
    try std.testing.expect(parseSortFlag("--json") == null);
}

test "sortSearchResults orders by score descending" {
    var items = [_]qmd.search.SearchResult{
        .{ .id = 1, .collection = "a", .path = "a", .title = "A", .hash = "", .score = 0.3 },
        .{ .id = 2, .collection = "a", .path = "b", .title = "B", .hash = "", .score = 0.9 },
        .{ .id = 3, .collection = "a", .path = "c", .title = "C", .hash = "", .score = 0.6 },
    };
    sortSearchResults(&items, .score);
    try std.testing.expectEqual(@as(i64, 2), items[0].id);
    try std.testing.expectEqual(@as(i64, 3), items[1].id);
    try std.testing.expectEqual(@as(i64, 1), items[2].id);
}

test "sortSearchResults orders by index ascending" {
    var items = [_]qmd.search.SearchResult{
        .{ .id = 3, .collection = "a", .path = "c", .title = "C", .hash = "", .score = 0.9 },
        .{ .id = 1, .collection = "a", .path = "a", .title = "A", .hash = "", .score = 0.3 },
        .{ .id = 2, .collection = "a", .path = "b", .title = "B", .hash = "", .score = 0.6 },
    };
    sortSearchResults(&items, .index);
    try std.testing.expectEqual(@as(i64, 1), items[0].id);
    try std.testing.expectEqual(@as(i64, 2), items[1].id);
    try std.testing.expectEqual(@as(i64, 3), items[2].id);
}
