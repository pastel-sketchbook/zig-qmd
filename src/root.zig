const std = @import("std");
const build_options = @import("build_options");

pub const db = @import("db.zig");
pub const store = @import("store.zig");
pub const chunker = @import("chunker.zig");
pub const search = @import("search.zig");
pub const config = @import("config.zig");
pub const llm = @import("llm.zig");
pub const mcp = @import("mcp.zig");
pub const ast = @import("ast.zig");
pub const remote = @import("remote.zig");
pub const llm_native = if (build_options.enable_llama) @import("llm_native.zig") else struct {};

/// ZMD library version, kept in sync with the VERSION file.
pub const version = "0.4.1";

/// High-level QMD engine providing collection management, indexing, and search.
pub const Qmd = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    db_inst: db.Db,

    /// Opens a QMD database at the given path, initializing the schema if needed.
    pub fn open(allocator: std.mem.Allocator, io: std.Io, db_path: [*:0]const u8) !Qmd {
        var conn = try db.Db.open(db_path);
        try db.initSchema(&conn);
        return .{ .allocator = allocator, .io = io, .db_inst = conn };
    }

    /// Closes the underlying database connection.
    pub fn close(self: *Qmd) void {
        self.db_inst.close();
    }

    /// Registers a new collection by name and filesystem path.
    pub fn add_collection(self: *Qmd, name: []const u8, path: []const u8) !void {
        try config.addCollection(&self.db_inst, name, path);
    }

    /// Scans all collection directories, indexes markdown files, and generates embeddings.
    pub fn update(self: *Qmd) !usize {
        var collections_result = try config.listCollections(&self.db_inst, self.allocator);
        defer config.freeCollections(&collections_result);

        var total_indexed: usize = 0;
        for (collections_result.collections.items) |col| {
            var dir = std.Io.Dir.cwd().openDir(self.io, col.path, .{ .iterate = true }) catch continue;
            defer dir.close(self.io);

            var walker = dir.walk(self.allocator) catch continue;
            defer walker.deinit();

            while (try walker.next(self.io)) |entry| {
                if (entry.kind != .file or !std.mem.endsWith(u8, entry.path, ".md")) continue;

                var full_path_buf: [1024]u8 = undefined;
                const full_path = std.fmt.bufPrint(&full_path_buf, "{s}/{s}", .{ col.path, entry.path }) catch continue;
                const content = std.Io.Dir.cwd().readFileAlloc(self.io, full_path, self.allocator, @enumFromInt(1024 * 1024)) catch continue;
                defer self.allocator.free(content);

                const insert_result = store.insertDocument(&self.db_inst, col.name, entry.path, content) catch continue;
                total_indexed += 1;

                // Skip chunking and embedding when content is unchanged
                if (!insert_result.content_changed) continue;
                const doc_hash = insert_result.hash;

                var chunk_slices = try std.ArrayList([]const u8).initCapacity(self.allocator, 0);
                defer chunk_slices.deinit(self.allocator);

                if (std.mem.eql(u8, ast.detectLanguage(entry.path), "markdown")) {
                    if (ast.AstChunker.init(self.allocator, "markdown")) |chunker_value| {
                        var ast_chunker = chunker_value;
                        defer ast_chunker.deinit();
                        if (ast_chunker.chunk(content, 1200)) |chunks| {
                            var ast_chunks = chunks;
                            defer ast_chunks.deinit(self.allocator);
                            try chunk_slices.appendSlice(self.allocator, ast_chunks.items);
                        } else |_| {}
                    } else |_| {}
                }

                if (chunk_slices.items.len == 0) {
                    var chunks = chunker.chunkDocument(content, self.allocator) catch continue;
                    defer chunks.chunks.deinit(self.allocator);
                    try chunk_slices.appendSlice(self.allocator, chunks.chunks.items);
                }

                var fallback = llm.LlamaCpp.init("/nonexistent", self.allocator) catch continue;
                defer fallback.deinit();
                for (chunk_slices.items, 0..) |c, idx| {
                    const formatted = llm.formatDocForEmbedding(self.allocator, c) catch continue;
                    defer self.allocator.free(formatted);
                    const emb = fallback.embed(formatted, self.allocator) catch continue;
                    defer self.allocator.free(emb);
                    store.upsertContentVectorAt(&self.db_inst, doc_hash[0..], @intCast(idx), 0, "fallback-fnv", emb, self.allocator) catch |err| {
                        if (err == error.OutOfMemory) return err;
                        continue;
                    };
                }
            }
        }
        return total_indexed;
    }

    /// Performs a BM25 full-text search, optionally filtered by collection.
    pub fn search_fts(self: *Qmd, query_text: []const u8, collection: ?[]const u8) !search.SearchResults {
        return search.searchFTS(&self.db_inst, self.allocator, query_text, collection);
    }

    /// Performs a hybrid search combining FTS and vector results with RRF fusion.
    pub fn query_hybrid(self: *Qmd, query_text: []const u8, options: search.HybridOptions) !search.HybridResult {
        return search.hybridSearch(&self.db_inst, self.allocator, self.io, query_text, null, options);
    }

    /// Retrieves a document by collection and path.
    pub fn get(self: *Qmd, collection: []const u8, path: []const u8) !store.ActiveDocument {
        return store.findActiveDocument(&self.db_inst, collection, path, self.allocator);
    }
};

/// Parses a "zmd://collection/path" or "collection/path" string into its components.
pub fn parse_virtual_path(input: []const u8) ?struct { collection: []const u8, path: []const u8 } {
    const raw = if (std.mem.startsWith(u8, input, "zmd://"))
        input[6..]
    else
        input;
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

test "parse_virtual_path handles zmd prefix" {
    const parsed = parse_virtual_path("zmd://notes/a.md") orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("notes", parsed.collection);
    try std.testing.expectEqualStrings("a.md", parsed.path);
}

test "ZMD open init add update search get" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var engine = try Qmd.open(allocator, std.testing.io, ":memory:");
    defer engine.close();

    try config.addCollection(&engine.db_inst, "notes", "/tmp");
    var cols = try config.listCollections(&engine.db_inst, allocator);
    defer config.freeCollections(&cols);
    try std.testing.expect(cols.collections.items.len >= 1);

    _ = try store.insertDocument(&engine.db_inst, "notes", "a.md", "# A\n\nhello auth");
    var res = try engine.search_fts("auth", null);
    defer res.deinit(allocator);
    try std.testing.expect(res.results.items.len >= 1);

    const doc = try engine.get("notes", "a.md");
    defer allocator.free(doc.title);
    defer allocator.free(doc.hash);
    defer allocator.free(doc.doc);
    try std.testing.expectEqualStrings("A", doc.title);
}

test {
    // Pull in tests from submodules
    _ = db;
}
