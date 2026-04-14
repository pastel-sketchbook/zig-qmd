const std = @import("std");

pub const db = @import("db.zig");
pub const store = @import("store.zig");
pub const chunker = @import("chunker.zig");
pub const search = @import("search.zig");
pub const config = @import("config.zig");
pub const llm = @import("llm.zig");
pub const mcp = @import("mcp.zig");
pub const ast = @import("ast.zig");

/// QMD library version, kept in sync with the VERSION file.
pub const version = "0.1.0";

pub const Qmd = struct {
    allocator: std.mem.Allocator,
    db: db.Db,

    pub fn open(allocator: std.mem.Allocator, db_path: [*:0]const u8) !Qmd {
        var conn = try db.Db.open(db_path);
        try db.initSchema(&conn);
        return .{ .allocator = allocator, .db = conn };
    }

    pub fn close(self: *Qmd) void {
        self.db.close();
    }

    pub fn add_collection(self: *Qmd, name: []const u8, path: []const u8) !void {
        try config.addCollection(&self.db, name, path);
    }

    pub fn update(self: *Qmd) !usize {
        var collections_result = try config.listCollections(&self.db);
        defer config.freeCollections(&collections_result);

        var total_indexed: usize = 0;
        for (collections_result.collections.items) |col| {
            var dir = std.fs.cwd().openDir(col.path, .{ .iterate = true }) catch continue;
            defer dir.close();

            var walker = dir.walk(self.allocator) catch continue;
            defer walker.deinit();

            while (try walker.next()) |entry| {
                if (entry.kind != .file or !std.mem.endsWith(u8, entry.path, ".md")) continue;

                var full_path_buf: [1024]u8 = undefined;
                const full_path = std.fmt.bufPrint(&full_path_buf, "{s}/{s}", .{ col.path, entry.path }) catch continue;
                const content = std.fs.cwd().readFileAlloc(self.allocator, full_path, 1024 * 1024) catch continue;
                defer self.allocator.free(content);

                store.insertDocument(&self.db, col.name, entry.path, content) catch continue;
                const doc_hash = store.findActiveDocumentHash(&self.db, col.name, entry.path) catch continue;

                var chunks = chunker.chunkDocument(content);
                defer chunks.chunks.deinit(std.heap.page_allocator);

                var fallback = llm.LlamaCpp.init("/nonexistent", self.allocator) catch continue;
                defer fallback.deinit();
                for (chunks.chunks.items, 0..) |c, idx| {
                    const formatted = llm.formatDocForEmbedding(self.allocator, c) catch continue;
                    defer self.allocator.free(formatted);
                    const emb = fallback.embed(formatted, self.allocator) catch continue;
                    defer self.allocator.free(emb);
                    store.upsertContentVectorAt(&self.db, doc_hash[0..], @intCast(idx), 0, "fallback-fnv", emb, self.allocator) catch {};
                }

                total_indexed += 1;
            }
        }
        return total_indexed;
    }

    pub fn search_fts(self: *Qmd, query_text: []const u8, collection: ?[]const u8) !search.SearchResults {
        return search.searchFTS(&self.db, query_text, collection);
    }

    pub fn query_hybrid(self: *Qmd, query_text: []const u8, options: search.HybridOptions) !search.HybridResult {
        return search.hybridSearch(&self.db, query_text, null, options);
    }

    pub fn get(self: *Qmd, collection: []const u8, path: []const u8) !store.ActiveDocument {
        return store.findActiveDocument(&self.db, collection, path);
    }
};

pub fn parse_virtual_path(input: []const u8) ?struct { collection: []const u8, path: []const u8 } {
    const raw = if (std.mem.startsWith(u8, input, "qmd://")) input[6..] else input;
    const slash = std.mem.indexOfScalar(u8, raw, '/') orelse return null;
    if (slash == 0 or slash + 1 >= raw.len) return null;
    return .{ .collection = raw[0..slash], .path = raw[slash + 1 ..] };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "version is semantic" {
    const v = version;
    var dots: usize = 0;
    for (v) |ch| {
        if (ch == '.') dots += 1;
    }
    try std.testing.expect(dots == 2);
}

test "parse_virtual_path handles qmd prefix" {
    const parsed = parse_virtual_path("qmd://notes/a.md") orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("notes", parsed.collection);
    try std.testing.expectEqualStrings("a.md", parsed.path);
}

test "Qmd open init add update search get" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var engine = try Qmd.open(allocator, ":memory:");
    defer engine.close();

    try config.addCollection(&engine.db, "notes", "/tmp");
    var cols = try config.listCollections(&engine.db);
    defer config.freeCollections(&cols);
    try std.testing.expect(cols.collections.items.len >= 1);

    try store.insertDocument(&engine.db, "notes", "a.md", "# A\n\nhello auth");
    var res = try engine.search_fts("auth", null);
    defer res.results.deinit(std.heap.page_allocator);
    try std.testing.expect(res.results.items.len >= 1);

    const doc = try engine.get("notes", "a.md");
    defer std.heap.page_allocator.free(doc.title);
    defer std.heap.page_allocator.free(doc.hash);
    defer std.heap.page_allocator.free(doc.doc);
    try std.testing.expectEqualStrings("A", doc.title);
}

test {
    // Pull in tests from submodules
    _ = db;
}
