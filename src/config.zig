const std = @import("std");
const db = @import("db.zig");

pub const ConfigError = error{
    NotFound,
    AlreadyExists,
} || db.DbError;

pub const Collection = struct {
    name: []const u8,
    path: []const u8,
    pattern: []const u8,
    ignore_patterns: ?[]const u8,
    include_by_default: bool,
    update_command: ?[]const u8,
    context: ?[]const u8,
};

pub fn addCollection(
    db_: *db.Db,
    name: []const u8,
    collection_path: []const u8,
) ConfigError!void {
    const sql = "INSERT INTO store_collections (name, path, pattern) VALUES (?, ?, '**/*.md')";
    var stmt = try db_.prepare(sql);
    defer stmt.finalize();
    try stmt.bindText(1, name);
    try stmt.bindText(2, collection_path);
    _ = try stmt.step();
}

pub fn listCollections(db_: *db.Db) !struct { collections: std.ArrayList(Collection) } {
    const sql = "SELECT name, path, pattern, ignore_patterns, include_by_default, update_command, context FROM store_collections ORDER BY name";
    var stmt = try db_.prepare(sql);
    defer stmt.finalize();

    var collections = try std.ArrayList(Collection).initCapacity(std.heap.page_allocator, 0);
    errdefer collections.deinit(std.heap.page_allocator);

    while (try stmt.step()) {
        const name_b = stmt.columnText(0);
        const path_b = stmt.columnText(1);
        const pattern_b = stmt.columnText(2);
        const ignore_b = stmt.columnText(3);
        const inc_def = stmt.columnInt(4);
        const upd_cmd = stmt.columnText(5);
        const context_b = stmt.columnText(6);

        const name = if (name_b) |b| std.mem.sliceTo(b, 0) else "";
        const path = if (path_b) |b| std.mem.sliceTo(b, 0) else "";
        const pattern = if (pattern_b) |b| std.mem.sliceTo(b, 0) else "**/*.md";
        const ignore_patterns = if (ignore_b) |b| std.mem.sliceTo(b, 0) else null;
        const update_command = if (upd_cmd) |b| std.mem.sliceTo(b, 0) else null;
        const context = if (context_b) |b| std.mem.sliceTo(b, 0) else null;

        try collections.append(std.heap.page_allocator, .{
            .name = try std.heap.page_allocator.dupe(u8, name),
            .path = try std.heap.page_allocator.dupe(u8, path),
            .pattern = try std.heap.page_allocator.dupe(u8, pattern),
            .ignore_patterns = if (ignore_patterns) |s| try std.heap.page_allocator.dupe(u8, s) else null,
            .include_by_default = inc_def == 1,
            .update_command = if (update_command) |s| try std.heap.page_allocator.dupe(u8, s) else null,
            .context = if (context) |s| try std.heap.page_allocator.dupe(u8, s) else null,
        });
    }

    return .{ .collections = collections };
}

pub fn getCollectionByName(db_: *db.Db, name: []const u8) !Collection {
    const sql = "SELECT name, path, pattern, ignore_patterns, include_by_default, update_command, context FROM store_collections WHERE name = ?";
    var stmt = try db_.prepare(sql);
    defer stmt.finalize();
    try stmt.bindText(1, name);

    if (!try stmt.step()) return ConfigError.NotFound;

    const name_b = stmt.columnText(0);
    const path_b = stmt.columnText(1);
    const pattern_b = stmt.columnText(2);
    const ignore_b = stmt.columnText(3);
    const inc_def = stmt.columnInt(4);
    const upd_cmd = stmt.columnText(5);
    const context_b = stmt.columnText(6);

    const name_s = if (name_b) |b| std.mem.sliceTo(b, 0) else "";
    const path_s = if (path_b) |b| std.mem.sliceTo(b, 0) else "";
    const pattern_s = if (pattern_b) |b| std.mem.sliceTo(b, 0) else "**/*.md";
    const ignore_patterns = if (ignore_b) |b| std.mem.sliceTo(b, 0) else null;
    const update_command = if (upd_cmd) |b| std.mem.sliceTo(b, 0) else null;
    const context = if (context_b) |b| std.mem.sliceTo(b, 0) else null;

    return .{
        .name = try std.heap.page_allocator.dupe(u8, name_s),
        .path = try std.heap.page_allocator.dupe(u8, path_s),
        .pattern = try std.heap.page_allocator.dupe(u8, pattern_s),
        .ignore_patterns = if (ignore_patterns) |s| try std.heap.page_allocator.dupe(u8, s) else null,
        .include_by_default = inc_def == 1,
        .update_command = if (update_command) |s| try std.heap.page_allocator.dupe(u8, s) else null,
        .context = if (context) |s| try std.heap.page_allocator.dupe(u8, s) else null,
    };
}

pub fn removeCollection(db_: *db.Db, name: []const u8) ConfigError!void {
    const sql = "DELETE FROM store_collections WHERE name = ?";
    var stmt = try db_.prepare(sql);
    defer stmt.finalize();
    try stmt.bindText(1, name);
    _ = try stmt.step();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "addCollection inserts collection" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    try addCollection(&db_, "notes", "/home/user/notes");

    const col = try getCollectionByName(&db_, "notes");
    try std.testing.expectEqualStrings("notes", col.name);
}

test "listCollections returns all collections" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    try addCollection(&db_, "notes", "/home/user/notes");
    try addCollection(&db_, "docs", "/home/user/docs");

    var result = try listCollections(&db_);
    defer result.collections.deinit(std.heap.page_allocator);

    try std.testing.expectEqual(@as(usize, 2), result.collections.items.len);
}

test "getCollectionByName returns NotFound" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    const result = getCollectionByName(&db_, "nonexistent");
    try std.testing.expectError(ConfigError.NotFound, result);
}

test "removeCollection deletes collection" {
    var db_ = try db.Db.open(":memory:");
    defer db_.close();
    try db.initSchema(&db_);

    try addCollection(&db_, "notes", "/home/user/notes");
    try removeCollection(&db_, "notes");

    const result = getCollectionByName(&db_, "notes");
    try std.testing.expectError(ConfigError.NotFound, result);
}
