const std = @import("std");
const db = @import("db.zig");
const crypto = std.crypto;

/// Error set for document store operations.
pub const StoreError = error{
    InsertFailed,
    NotFound,
    AlreadyExists,
    OutOfMemory,
} || db.DbError;

const SHA256_HEX_LEN = 64;

/// Represents a document fetched from the store with its metadata.
pub const ActiveDocument = struct {
    id: i64,
    title: []const u8,
    hash: []const u8,
    doc: []const u8,
};

/// Computes SHA-256 hex digest of content into a fixed-size buffer.
pub fn hashContent(content: []const u8, out: *[SHA256_HEX_LEN:0]u8) void {
    var hash: [32]u8 = undefined;
    crypto.hash.sha2.Sha256.hash(content, &hash, .{});
    for (hash, 0..) |byte, i| {
        out.*[i * 2] = "0123456789abcdef"[(byte >> 4) & 0xF];
        out.*[i * 2 + 1] = "0123456789abcdef"[byte & 0xF];
    }
}

/// Normalizes a file path into a lowercase alphanumeric handle.
pub fn handleize(path: []const u8, out: *[SHA256_HEX_LEN]u8) void {
    const normalized = std.mem.trim(u8, path, &.{ ' ', '\t', '\n', '\r' });
    var result: [SHA256_HEX_LEN]u8 = undefined;
    var i: usize = 0;
    for (normalized) |ch| {
        if (ch == '_' or ch == '-' or ch == '.' or ch == '/') {
            result[i] = '_';
        } else if (ch >= 'A' and ch <= 'Z') {
            result[i] = ch + 32;
        } else if (ch >= 'a' and ch <= 'z' or ch >= '0' and ch <= '9') {
            result[i] = ch;
        } else if (ch >= 128) {
            result[i] = '_';
        }
        i += 1;
    }
    while (i > 0 and result[i - 1] == '_') i -= 1;
    std.mem.copyForwards(u8, out.*[0..i], result[0..i]);
    if (i < SHA256_HEX_LEN) out.*[i] = 0;
}

/// Extracts a document title from YAML frontmatter or the first markdown heading.
pub fn extractTitle(content: []const u8) []const u8 {
    var lines = std.mem.splitScalar(u8, content, '\n');

    var in_frontmatter = false;
    var frontmatter_started = false;
    var frontmatter_title: ?[]const u8 = null;

    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, &.{ ' ', '\t', '\r' });

        if (!frontmatter_started and std.mem.eql(u8, trimmed, "---")) {
            frontmatter_started = true;
            in_frontmatter = true;
            continue;
        }

        if (in_frontmatter) {
            if (std.mem.startsWith(u8, trimmed, "title:")) {
                var v = std.mem.trim(u8, trimmed[6..], &.{ ' ', '\t' });
                if (v.len >= 2 and v[0] == '"' and v[v.len - 1] == '"') {
                    v = v[1 .. v.len - 1];
                }
                if (v.len > 0) frontmatter_title = v;
            }
            if (std.mem.eql(u8, trimmed, "---")) {
                in_frontmatter = false;
                if (frontmatter_title) |t| return t;
            }
            continue;
        }

        if (trimmed.len == 0) continue;

        // Markdown heading (supports multiple '#')
        var i: usize = 0;
        while (i < trimmed.len and trimmed[i] == '#') : (i += 1) {}
        if (i > 0 and i < trimmed.len and trimmed[i] == ' ') {
            const heading = std.mem.trim(u8, trimmed[i + 1 ..], &.{ ' ', '\t' });
            if (heading.len > 0) return heading;
            continue;
        }

        return trimmed;
    }

    return "Untitled";
}

/// Result of inserting a document: the content hash and whether the content was new.
pub const InsertResult = struct {
    hash: [SHA256_HEX_LEN]u8,
    /// True when the content blob was inserted for the first time (hash not seen before).
    content_changed: bool,
};

/// Inserts content into the content-addressable store, returning its SHA-256 hash
/// and whether the content was actually new (INSERT OR IGNORE affected a row).
pub fn insertContent(db_: *db.Db, content: []const u8) StoreError!InsertResult {
    var hash: [SHA256_HEX_LEN:0]u8 = undefined;
    hashContent(content, &hash);

    const now = "2024-01-01T00:00:00Z";
    const sql = "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)";
    var stmt = try db_.prepareCached(sql);
    try stmt.bindText(1, hash[0..SHA256_HEX_LEN]);
    try stmt.bindText(2, content);
    try stmt.bindText(3, now);
    _ = try stmt.step();

    const content_changed = db_.changes() > 0;

    return .{ .hash = hash, .content_changed = content_changed };
}

/// Inserts or replaces a document in the store, extracting title and computing content hash.
/// Returns the content SHA-256 hash and whether the content blob was new, so callers
/// can skip expensive downstream work (chunking, embedding) for unchanged documents.
pub fn insertDocument(db_: *db.Db, collection: []const u8, path: []const u8, content: []const u8) StoreError!InsertResult {
    const result = try insertContent(db_, content);

    const title = extractTitle(content);
    const now = "2024-01-01T00:00:00Z";

    const sql = "INSERT OR REPLACE INTO documents (collection, path, title, hash, created_at, modified_at, active) VALUES (?, ?, ?, ?, ?, ?, 1)";
    var stmt = try db_.prepareCached(sql);
    try stmt.bindText(1, collection);
    try stmt.bindText(2, path);
    try stmt.bindText(3, title);
    try stmt.bindText(4, result.hash[0..SHA256_HEX_LEN]);
    try stmt.bindText(5, now);
    try stmt.bindText(6, now);
    _ = try stmt.step();

    return result;
}

/// Looks up an active document by collection and path, returning its metadata and content.
pub fn findActiveDocument(db_: *db.Db, collection: []const u8, path: []const u8, allocator: std.mem.Allocator) StoreError!ActiveDocument {
    const sql = "SELECT d.id, d.title, d.hash, c.doc FROM documents d JOIN content c ON d.hash = c.hash WHERE d.collection = ? AND d.path = ? AND d.active = 1";
    var stmt = try db_.prepare(sql);
    defer stmt.finalize();
    try stmt.bindText(1, collection);
    try stmt.bindText(2, path);

    if (!try stmt.step()) return StoreError.NotFound;

    const ttl = stmt.columnText(1);
    const hsh = stmt.columnText(2);
    const d = stmt.columnText(3);

    const title_copy = if (ttl) |t| try allocator.dupe(u8, std.mem.span(t)) else try allocator.dupe(u8, "");
    const hash_copy = if (hsh) |h| try allocator.dupe(u8, std.mem.span(h)) else try allocator.dupe(u8, "");
    const doc_copy = if (d) |doc_txt| try allocator.dupe(u8, std.mem.span(doc_txt)) else try allocator.dupe(u8, "");

    return .{
        .id = stmt.columnInt(0),
        .title = title_copy,
        .hash = hash_copy,
        .doc = doc_copy,
    };
}

/// Returns the content hash of an active document by collection and path.
pub fn findActiveDocumentHash(db_: *db.Db, collection: []const u8, path: []const u8) StoreError![SHA256_HEX_LEN]u8 {
    const sql = "SELECT d.hash FROM documents d WHERE d.collection = ? AND d.path = ? AND d.active = 1";
    var stmt = try db_.prepare(sql);
    defer stmt.finalize();
    try stmt.bindText(1, collection);
    try stmt.bindText(2, path);

    if (!try stmt.step()) return StoreError.NotFound;
    const hsh = stmt.columnText(0) orelse return StoreError.NotFound;
    const span = std.mem.span(hsh);
    if (span.len < SHA256_HEX_LEN) return StoreError.NotFound;

    var out: [SHA256_HEX_LEN]u8 = undefined;
    std.mem.copyForwards(u8, out[0..], span[0..SHA256_HEX_LEN]);
    return out;
}

/// Lists all active document paths and titles within a collection.
pub fn getActiveDocumentPaths(db_: *db.Db, collection: []const u8, allocator: std.mem.Allocator) StoreError!struct { paths: std.ArrayList([]const u8), titles: std.ArrayList([]const u8) } {
    const sql = "SELECT path, title FROM documents WHERE collection = ? AND active = 1 ORDER BY path";
    var stmt = try db_.prepare(sql);
    defer stmt.finalize();
    try stmt.bindText(1, collection);

    var paths = try std.ArrayList([]const u8).initCapacity(allocator, 0);
    errdefer paths.deinit(allocator);
    var titles = try std.ArrayList([]const u8).initCapacity(allocator, 0);
    errdefer titles.deinit(allocator);

    while (try stmt.step()) {
        const pth = stmt.columnText(0);
        const ttl = stmt.columnText(1);
        const path = if (pth) |p| try allocator.dupe(u8, std.mem.span(p)) else try allocator.dupe(u8, "");
        const title = if (ttl) |t| try allocator.dupe(u8, std.mem.span(t)) else try allocator.dupe(u8, "");
        try paths.append(allocator, path);
        try titles.append(allocator, title);
    }

    return .{ .paths = paths, .titles = titles };
}

/// Marks a document as inactive (soft delete) by collection and path.
pub fn deactivateDocument(db_: *db.Db, collection: []const u8, path: []const u8) StoreError!void {
    const sql = "UPDATE documents SET active = 0 WHERE collection = ? AND path = ? AND active = 1";
    var stmt = try db_.prepare(sql);
    defer stmt.finalize();
    try stmt.bindText(1, collection);
    try stmt.bindText(2, path);
    _ = try stmt.step();
}

/// Stores a vector embedding for a document's content hash at default position.
pub fn upsertContentVector(
    db_: *db.Db,
    hash: []const u8,
    model: []const u8,
    embedding: []const f32,
    allocator: std.mem.Allocator,
) StoreError!void {
    return upsertContentVectorAt(db_, hash, 0, 0, model, embedding, allocator);
}

/// Stores a vector embedding for a specific chunk (seq/pos) of a document's content.
pub fn upsertContentVectorAt(
    db_: *db.Db,
    hash: []const u8,
    seq: i64,
    pos: i64,
    model: []const u8,
    embedding: []const f32,
    allocator: std.mem.Allocator,
) StoreError!void {
    var emb_json = try std.ArrayList(u8).initCapacity(allocator, embedding.len * 10 + 2);
    defer emb_json.deinit(allocator);

    try emb_json.append(allocator, '[');
    for (embedding, 0..) |v, i| {
        if (i > 0) try emb_json.append(allocator, ',');
        try emb_json.writer(allocator).print("{d}", .{v});
    }
    try emb_json.append(allocator, ']');

    const now = "2024-01-01T00:00:00Z";
    var stmt = try db_.prepareCached(
        "INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embedding, embedded_at) VALUES (?, ?, ?, ?, ?, ?)",
    );

    try stmt.bindText(1, hash);
    try stmt.bindInt(2, @intCast(seq));
    try stmt.bindInt(3, @intCast(pos));
    try stmt.bindText(4, model);
    try stmt.bindText(5, emb_json.items);
    try stmt.bindText(6, now);
    _ = try stmt.step();

    var del_idx = try db_.prepareCached("DELETE FROM content_vectors_idx WHERE hash = ? AND seq = ? AND pos = ?");
    try del_idx.bindText(1, hash);
    try del_idx.bindInt(2, @intCast(seq));
    try del_idx.bindInt(3, @intCast(pos));
    _ = try del_idx.step();

    var ins_idx = try db_.prepareCached("INSERT INTO content_vectors_idx(embedding, hash, model, seq, pos) VALUES(vec_f32(?), ?, ?, ?, ?)");
    try ins_idx.bindText(1, emb_json.items);
    try ins_idx.bindText(2, hash);
    try ins_idx.bindText(3, model);
    try ins_idx.bindInt(4, @intCast(seq));
    try ins_idx.bindInt(5, @intCast(pos));
    _ = try ins_idx.step();
}

/// Statistics returned by the orphan cleanup operation.
pub const CleanupStats = struct {
    removed_content: i64,
    removed_vectors: i64,
};

/// Removes content and vectors not referenced by any active document.
pub fn cleanupOrphans(db_: *db.Db) StoreError!CleanupStats {
    const orphan_content = try countOrphanContent(db_);
    const orphan_vectors = try countOrphanVectors(db_);

    try db_.exec(
        "DELETE FROM content WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
    );
    try db_.exec(
        "DELETE FROM content_vectors WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
    );
    try db_.exec(
        "DELETE FROM content_vectors_idx WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
    );

    return .{
        .removed_content = orphan_content,
        .removed_vectors = orphan_vectors,
    };
}

/// Runs SQLite VACUUM to reclaim disk space.
pub fn vacuum(db_: *db.Db) StoreError!void {
    try db_.exec("VACUUM");
}

fn countOrphanContent(db_: *db.Db) StoreError!i64 {
    var stmt = try db_.prepare(
        "SELECT count(*) FROM content WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
    );
    defer stmt.finalize();
    if (!try stmt.step()) return 0;
    return stmt.columnInt(0);
}

fn countOrphanVectors(db_: *db.Db) StoreError!i64 {
    var stmt = try db_.prepare(
        "SELECT count(*) FROM content_vectors WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
    );
    defer stmt.finalize();
    if (!try stmt.step()) return 0;
    return stmt.columnInt(0);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "hashContent produces 64-char hex" {
    var hash: [SHA256_HEX_LEN:0]u8 = undefined;
    hashContent("hello world", &hash);
    try std.testing.expectEqualStrings("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9", &hash);
}

test "handleize normalizes path" {
    var out: [SHA256_HEX_LEN]u8 = undefined;
    handleize("My_Document.md", &out);
    try std.testing.expectEqualStrings("my_document_md", std.mem.sliceTo(&out, 0));
}

test "extractTitle from heading" {
    const title = extractTitle("# Hello World\n\nContent here");
    try std.testing.expectEqualStrings("Hello World", title);
}

test "extractTitle from plain first line" {
    const title = extractTitle("Plain title\n\nContent");
    try std.testing.expectEqualStrings("Plain title", title);
}

test "extractTitle skips leading blank lines" {
    const title = extractTitle("\n\n# Login Flow\n\nContent");
    try std.testing.expectEqualStrings("Login Flow", title);
}

test "extractTitle skips yaml frontmatter" {
    const title = extractTitle(
        "---\nlayout: post\ntags: [auth]\n---\n\n# Authentication\nBody",
    );
    try std.testing.expectEqualStrings("Authentication", title);
}

test "extractTitle uses frontmatter title when present" {
    const title = extractTitle(
        "---\ntitle: \"Frontmatter Title\"\ncategory: docs\n---\n\n# Different Heading\nBody",
    );
    try std.testing.expectEqualStrings("Frontmatter Title", title);
}

test "extractTitle falls back to Untitled" {
    const title = extractTitle("\n\n\n");
    try std.testing.expectEqualStrings("Untitled", title);
}

test "insertContent and retrieve" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    const result = try insertContent(&db_, "hello world");

    var stmt = try db_.prepare("SELECT doc FROM content WHERE hash = ?");
    defer stmt.finalize();
    try stmt.bindText(1, result.hash[0..]);
    const has_row = try stmt.step();
    try std.testing.expect(has_row);
    try std.testing.expect(result.content_changed);
    const doc = stmt.columnText(0);
    try std.testing.expectEqualStrings("hello world", std.mem.span(doc.?));
}

test "findActiveDocument returns document" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    _ = try insertDocument(&db_, "notes", "test.md", "# Test Doc\n\nHello world");

    const doc = try findActiveDocument(&db_, "notes", "test.md", std.testing.allocator);
    defer {
        std.testing.allocator.free(doc.title);
        std.testing.allocator.free(doc.hash);
        std.testing.allocator.free(doc.doc);
    }
    try std.testing.expectEqualStrings("Test Doc", doc.title);
    try std.testing.expect(doc.id > 0);
}

test "deactivateDocument marks inactive" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    _ = try insertDocument(&db_, "notes", "test.md", "# Test\n\nContent");
    try deactivateDocument(&db_, "notes", "test.md");

    const result = findActiveDocument(&db_, "notes", "test.md", std.testing.allocator);
    try std.testing.expectError(StoreError.NotFound, result);
}

test "insertDocument returns stable 64-byte hash and content_changed flag" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    const result = try insertDocument(&db_, "notes", "x.md", "# X\n\ncontent");
    try std.testing.expectEqual(@as(usize, 64), result.hash.len);
    try std.testing.expect(result.content_changed);

    // Verify it matches findActiveDocumentHash (the old way)
    const hash2 = try findActiveDocumentHash(&db_, "notes", "x.md");
    try std.testing.expectEqualStrings(&result.hash, &hash2);

    // Re-insert same content — content_changed should be false
    const result2 = try insertDocument(&db_, "notes", "x.md", "# X\n\ncontent");
    try std.testing.expect(!result2.content_changed);
    try std.testing.expectEqualStrings(&result.hash, &result2.hash);

    // Insert different content — content_changed should be true again
    const result3 = try insertDocument(&db_, "notes", "x.md", "# X\n\nupdated content");
    try std.testing.expect(result3.content_changed);
}

test "getActiveDocumentPaths returns all paths" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    _ = try insertDocument(&db_, "notes", "a.md", "# A\n\nContent");
    _ = try insertDocument(&db_, "notes", "b.md", "# B\n\nContent");

    var result = try getActiveDocumentPaths(&db_, "notes", std.testing.allocator);
    defer {
        for (result.paths.items) |p| std.testing.allocator.free(p);
        for (result.titles.items) |t| std.testing.allocator.free(t);
        result.paths.deinit(std.testing.allocator);
        result.titles.deinit(std.testing.allocator);
    }
    try std.testing.expectEqual(@as(usize, 2), result.paths.items.len);
}

test "upsertContentVector stores embedding JSON" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    _ = try insertDocument(&db_, "notes", "a.md", "# A\n\ncontent");
    const doc = try findActiveDocument(&db_, "notes", "a.md", std.testing.allocator);
    defer {
        std.testing.allocator.free(doc.title);
        std.testing.allocator.free(doc.hash);
        std.testing.allocator.free(doc.doc);
    }

    try upsertContentVector(&db_, doc.hash, "test-model", &.{ 0.1, 0.2, -0.3 }, std.testing.allocator);

    var stmt = try db_.prepare("SELECT model, embedding FROM content_vectors WHERE hash = ? AND seq = 0 AND pos = 0");
    defer stmt.finalize();
    try stmt.bindText(1, doc.hash);
    try std.testing.expect(try stmt.step());

    const model = stmt.columnText(0).?;
    const embedding = stmt.columnText(1).?;
    try std.testing.expectEqualStrings("test-model", std.mem.span(model));
    try std.testing.expect(std.mem.indexOf(u8, std.mem.span(embedding), "0.1") != null);
}

test "upsertContentVectorAt stores multiple chunk vectors" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    _ = try insertDocument(&db_, "notes", "a.md", "# A\n\ncontent");
    const doc = try findActiveDocument(&db_, "notes", "a.md", std.testing.allocator);
    defer {
        std.testing.allocator.free(doc.title);
        std.testing.allocator.free(doc.hash);
        std.testing.allocator.free(doc.doc);
    }

    try upsertContentVectorAt(&db_, doc.hash, 0, 0, "test-model", &.{ 0.1, 0.2 }, std.testing.allocator);
    try upsertContentVectorAt(&db_, doc.hash, 1, 0, "test-model", &.{ 0.3, 0.4 }, std.testing.allocator);

    var stmt = try db_.prepare("SELECT count(*) FROM content_vectors WHERE hash = ?");
    defer stmt.finalize();
    try stmt.bindText(1, doc.hash);
    try std.testing.expect(try stmt.step());
    try std.testing.expectEqual(@as(i64, 2), stmt.columnInt(0));
}

test "insertDocument replaces active row for same path" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    _ = try insertDocument(&db_, "notes", "same.md", "---\ntitle: \"First\"\n---\n# First");
    _ = try insertDocument(&db_, "notes", "same.md", "---\ntitle: \"Second\"\n---\n# Second");

    var stmt = try db_.prepare("SELECT count(*) FROM documents WHERE collection = 'notes' AND path = 'same.md' AND active = 1");
    defer stmt.finalize();
    try std.testing.expect(try stmt.step());
    try std.testing.expectEqual(@as(i64, 1), stmt.columnInt(0));

    const doc = try findActiveDocument(&db_, "notes", "same.md", std.testing.allocator);
    defer {
        std.testing.allocator.free(doc.title);
        std.testing.allocator.free(doc.hash);
        std.testing.allocator.free(doc.doc);
    }
    try std.testing.expectEqualStrings("Second", doc.title);
}

test "cleanupOrphans removes inactive content and vectors" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    const keep_result = try insertDocument(&db_, "notes", "keep.md", "# Keep\n\nactive");
    const drop_result = try insertDocument(&db_, "notes", "drop.md", "# Drop\n\ninactive");

    try upsertContentVector(&db_, keep_result.hash[0..], "m", &.{ 0.1, 0.2 }, std.testing.allocator);
    try upsertContentVector(&db_, drop_result.hash[0..], "m", &.{ 0.3, 0.4 }, std.testing.allocator);

    try deactivateDocument(&db_, "notes", "drop.md");

    const stats = try cleanupOrphans(&db_);
    try std.testing.expect(stats.removed_content >= 1);
    try std.testing.expect(stats.removed_vectors >= 1);

    var stmt = try db_.prepare("SELECT count(*) FROM content");
    defer stmt.finalize();
    try std.testing.expect(try stmt.step());
    try std.testing.expectEqual(@as(i64, 1), stmt.columnInt(0));

    stmt = try db_.prepare("SELECT count(*) FROM content_vectors");
    defer stmt.finalize();
    try std.testing.expect(try stmt.step());
    try std.testing.expectEqual(@as(i64, 1), stmt.columnInt(0));
}
