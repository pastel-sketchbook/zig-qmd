const std = @import("std");
const db = @import("db.zig");
const crypto = std.crypto;

pub const StoreError = error{
    InsertFailed,
    NotFound,
    AlreadyExists,
    OutOfMemory,
} || db.DbError;

const SHA256_HEX_LEN = 64;

pub fn hashContent(content: []const u8, out: *[SHA256_HEX_LEN:0]u8) void {
    var hash: [32]u8 = undefined;
    crypto.hash.sha2.Sha256.hash(content, &hash, .{});
    for (hash, 0..) |byte, i| {
        out.*[i * 2] = "0123456789abcdef"[(byte >> 4) & 0xF];
        out.*[i * 2 + 1] = "0123456789abcdef"[byte & 0xF];
    }
}

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

pub fn extractTitle(content: []const u8) []const u8 {
    var lines = std.mem.splitScalar(u8, content, '\n');
    const first = lines.first();
    if (std.mem.startsWith(u8, first, "# ")) {
        return std.mem.trim(u8, first[2..], &.{ ' ', '\t' });
    }
    return std.mem.trim(u8, first, &.{ ' ', '\t' });
}

pub fn insertContent(db_: *db.Db, content: []const u8) StoreError![SHA256_HEX_LEN]u8 {
    var hash: [SHA256_HEX_LEN:0]u8 = undefined;
    hashContent(content, &hash);

    const now = "2024-01-01T00:00:00Z";
    const sql = "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)";
    var stmt = try db_.prepare(sql);
    defer stmt.finalize();
    try stmt.bindText(1, hash[0..SHA256_HEX_LEN]);
    try stmt.bindText(2, content);
    try stmt.bindText(3, now);
    _ = try stmt.step();

    return hash;
}

pub fn findActiveDocument(db_: *db.Db, collection: []const u8, path: []const u8) StoreError!struct { id: i64, title: []const u8, hash: []const u8, doc: []const u8 } {
    const sql = "SELECT d.id, d.title, d.hash, c.doc FROM documents d JOIN content c ON d.hash = c.hash WHERE d.collection = ? AND d.path = ? AND d.active = 1";
    var stmt = try db_.prepare(sql);
    defer stmt.finalize();
    try stmt.bindText(1, collection);
    try stmt.bindText(2, path);

    if (!try stmt.step()) return StoreError.NotFound;

    const ttl = stmt.columnText(1);
    const hsh = stmt.columnText(2);
    const d = stmt.columnText(3);

    return .{
        .id = stmt.columnInt(0),
        .title = if (ttl) |t| std.mem.span(t) else "",
        .hash = if (hsh) |h| std.mem.span(h) else "",
        .doc = if (d) |doc| std.mem.span(doc) else "",
    };
}

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

pub fn getActiveDocumentPaths(db_: *db.Db, collection: []const u8) StoreError!struct { paths: std.ArrayList([]const u8), titles: std.ArrayList([]const u8) } {
    const sql = "SELECT path, title FROM documents WHERE collection = ? AND active = 1 ORDER BY path";
    var stmt = try db_.prepare(sql);
    defer stmt.finalize();
    try stmt.bindText(1, collection);

    var paths = try std.ArrayList([]const u8).initCapacity(std.heap.page_allocator, 0);
    errdefer paths.deinit(std.heap.page_allocator);
    var titles = try std.ArrayList([]const u8).initCapacity(std.heap.page_allocator, 0);
    errdefer titles.deinit(std.heap.page_allocator);

    while (try stmt.step()) {
        const pth = stmt.columnText(0);
        const ttl = stmt.columnText(1);
        const path = if (pth) |p| std.mem.span(p) else "";
        const title = if (ttl) |t| std.mem.span(t) else "";
        try paths.append(std.heap.page_allocator, path);
        try titles.append(std.heap.page_allocator, title);
    }

    return .{ .paths = paths, .titles = titles };
}

pub fn deactivateDocument(db_: *db.Db, collection: []const u8, path: []const u8) StoreError!void {
    const sql = "UPDATE documents SET active = 0 WHERE collection = ? AND path = ? AND active = 1";
    var stmt = try db_.prepare(sql);
    defer stmt.finalize();
    try stmt.bindText(1, collection);
    try stmt.bindText(2, path);
    _ = try stmt.step();
}

pub fn insertDocument(db_: *db.Db, collection: []const u8, path: []const u8, content: []const u8) StoreError!void {
    const hash = try insertContent(db_, content);

    const title = extractTitle(content);
    const now = "2024-01-01T00:00:00Z";

    const sql = "INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active) VALUES (?, ?, ?, ?, ?, ?, 1)";
    var stmt = try db_.prepare(sql);
    defer stmt.finalize();
    try stmt.bindText(1, collection);
    try stmt.bindText(2, path);
    try stmt.bindText(3, title);
    try stmt.bindText(4, hash[0..SHA256_HEX_LEN]);
    try stmt.bindText(5, now);
    try stmt.bindText(6, now);
    _ = try stmt.step();
}

pub fn upsertContentVector(
    db_: *db.Db,
    hash: []const u8,
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
    var stmt = try db_.prepare(
        "INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embedding, embedded_at) VALUES (?, 0, 0, ?, ?, ?)",
    );
    defer stmt.finalize();

    try stmt.bindText(1, hash);
    try stmt.bindText(2, model);
    try stmt.bindText(3, emb_json.items);
    try stmt.bindText(4, now);
    _ = try stmt.step();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "hashContent produces 64-char hex" {
    var hash: [SHA256_HEX_LEN:0]u8 = undefined;
    hashContent("hello world", &hash);
    try std.testing.expectEqualStrings("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380fc9081103", &hash);
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

test "insertContent and retrieve" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    const hash = try insertContent(&db_, "hello world");

    var stmt = try db_.prepare("SELECT doc FROM content WHERE hash = ?");
    defer stmt.finalize();
    try stmt.bindText(1, hash[0..]);
    const has_row = try stmt.step();
    try std.testing.expect(has_row);
    const doc = stmt.columnText(0);
    try std.testing.expectEqualStrings("hello world", std.mem.span(doc.?));
}

test "findActiveDocument returns document" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    try insertDocument(&db_, "notes", "test.md", "# Test Doc\n\nHello world");

    const doc = try findActiveDocument(&db_, "notes", "test.md");
    try std.testing.expectEqualStrings("Test Doc", doc.title);
    try std.testing.expect(doc.id > 0);
}

test "deactivateDocument marks inactive" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    try insertDocument(&db_, "notes", "test.md", "# Test\n\nContent");
    try deactivateDocument(&db_, "notes", "test.md");

    const result = findActiveDocument(&db_, "notes", "test.md");
    try std.testing.expectError(StoreError.NotFound, result);
}

test "findActiveDocumentHash returns stable 64-byte hash" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    try insertDocument(&db_, "notes", "x.md", "# X\n\ncontent");
    const hash = try findActiveDocumentHash(&db_, "notes", "x.md");
    try std.testing.expectEqual(@as(usize, 64), hash.len);
}

test "getActiveDocumentPaths returns all paths" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    try insertDocument(&db_, "notes", "a.md", "# A\n\nContent");
    try insertDocument(&db_, "notes", "b.md", "# B\n\nContent");

    const result = try getActiveDocumentPaths(&db_, "notes");
    defer {
        result.paths.deinit();
        result.titles.deinit();
    }
    try std.testing.expectEqual(@as(usize, 2), result.paths.items.len);
}

test "upsertContentVector stores embedding JSON" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    try insertDocument(&db_, "notes", "a.md", "# A\n\ncontent");
    const doc = try findActiveDocument(&db_, "notes", "a.md");

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
