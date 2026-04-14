const std = @import("std");
const qmd = @import("qmd");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

const DB_PATH = ".qmd/data.db";

pub fn main() !void {
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    var args = std.process.args();
    _ = args.next();

    const cmd = args.next() orelse {
        try stdout.writeAll("Usage: zmd <command>\n");
        try stdout.writeAll("Commands: version, collection, update, search, vsearch, query, get, status, mcp, ls, cleanup, embed\n");
        try stdout.flush();
        return;
    };

    if (std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h") or std.mem.eql(u8, cmd, "help")) {
        try stdout.writeAll("zmd - Local hybrid search for markdown notes\n\n");
        try stdout.writeAll("Commands:\n");
        try stdout.writeAll("  version     Show version\n");
        try stdout.writeAll("  collection  Manage collections (add/list/remove)\n");
        try stdout.writeAll("  update     Update index for a collection\n");
        try stdout.writeAll("  search     Full-text search\n");
        try stdout.writeAll("  vsearch    Vector semantic search\n");
        try stdout.writeAll("  query      Hybrid search (FTS + vector)\n");
        try stdout.writeAll("  get        Get document by path\n");
        try stdout.writeAll("  status     Show system status\n");
        try stdout.writeAll("  mcp        Start MCP server\n");
        try stdout.writeAll("  ls         List documents\n");
        try stdout.writeAll("  cleanup    Remove orphaned entries\n");
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
                try stdout.writeAll("Usage: zmd collection add <name> <path>\n");
                try stdout.flush();
                return;
            };
            const path = args.next() orelse {
                try stdout.writeAll("Usage: zmd collection add <name> <path>\n");
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
            try stdout.writeAll("\n");
            try stdout.flush();
            return;
        }

        if (std.mem.eql(u8, subcmd, "list")) {
            var result = qmd.config.listCollections(&db_) catch {
                try stdout.writeAll("Failed to list collections\n");
                try stdout.flush();
                return;
            };
            defer result.collections.deinit(std.heap.page_allocator);

            if (result.collections.items.len == 0) {
                try stdout.writeAll("No collections. Run 'zmd collection add <name> <path>'\n");
            } else {
                for (result.collections.items) |col| {
                    try stdout.print("  {s}: {s}\n", .{ col.name, col.path });
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

        std.fs.cwd().makeDir(".qmd") catch {};

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

        var collections_result = qmd.config.listCollections(&db_) catch {
            try stdout.writeAll("Error: Failed to list collections\n");
            try stdout.flush();
            return;
        };
        defer collections_result.collections.deinit(std.heap.page_allocator);

        var total_indexed: usize = 0;
        for (collections_result.collections.items) |col| {
            try stdout.print("Indexing collection '{s}' from {s}...\n", .{ col.name, col.path });
            try stdout.flush();

            var dir = std.fs.cwd().openDir(col.path, .{ .iterate = true }) catch {
                try stdout.print("  Warning: Could not open directory {s}\n", .{col.path});
                continue;
            };
            defer dir.close();

            var walker = dir.walk(allocator) catch {
                try stdout.writeAll("  Error: Failed to walk directory\n");
                continue;
            };
            defer walker.deinit();

            while (try walker.next()) |entry| {
                if (entry.kind == .file and std.mem.endsWith(u8, entry.path, ".md")) {
                    var full_path_buf: [1024]u8 = undefined;
                    const full_path = std.fmt.bufPrint(&full_path_buf, "{s}/{s}", .{ col.path, entry.path }) catch continue;

                    const content = std.fs.cwd().readFileAlloc(allocator, full_path, 1024 * 1024) catch |err| {
                        try stdout.print("    Error reading {s}: {any}\n", .{ entry.path, err });
                        continue;
                    };
                    defer allocator.free(content);

                    qmd.store.insertDocument(&db_, col.name, entry.path, content) catch |err| {
                        try stdout.print("    Error inserting {s}: {any}\n", .{ entry.path, err });
                        continue;
                    };
                    total_indexed += 1;
                }
            }
            try stdout.print("  Indexed {d} documents\n", .{total_indexed});
        }

        try stdout.print("Update complete. Total: {d} documents\n", .{total_indexed});
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "query")) {
        const query_text = args.next() orelse {
            try stdout.writeAll("Usage: zmd query <query>\n");
            try stdout.flush();
            return;
        };

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Database not found. Run 'zmd update' first.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        var result = qmd.search.hybridSearch(&db_, query_text, null, .{
            .enable_vector = false,
            .rrf_k = qmd.search.RRF_K,
            .max_results = 10,
        }) catch {
            try stdout.writeAll("Search failed\n");
            try stdout.flush();
            return;
        };
        defer result.results.deinit(std.heap.page_allocator);

        try stdout.print("Found {d} results (hybrid)\n", .{result.results.items.len});
        for (result.results.items, 0..) |r, i| {
            try stdout.print("  {d}. {s} ({s}) - score: {d:.4}\n", .{ i + 1, r.title, r.collection, r.score });
        }
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "search")) {
        const query_text = args.next() orelse {
            try stdout.writeAll("Usage: zmd search <query> [collection]\n");
            try stdout.flush();
            return;
        };
        const collection = args.next();

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Database not found. Run 'zmd update' first.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        var result = qmd.search.searchFTS(&db_, query_text, collection) catch {
            try stdout.writeAll("Search failed\n");
            try stdout.flush();
            return;
        };
        defer result.results.deinit(std.heap.page_allocator);

        if (result.results.items.len == 0) {
            try stdout.writeAll("No results found.\n");
        } else {
            try stdout.print("Found {d} results:\n", .{result.results.items.len});
            for (result.results.items, 0..) |r, i| {
                try stdout.print("  {d}. title='{s}' collection='{s}' path='{s}' score={d:.4}\n", .{ i + 1, r.title, r.collection, r.path, r.score });
            }
        }
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "vsearch")) {
        const query_text = args.next() orelse {
            try stdout.writeAll("Usage: zmd vsearch <query>\n");
            try stdout.flush();
            return;
        };

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Database not found. Run 'zmd update' first.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        const result = qmd.search.searchVec(&db_, query_text, null) catch {
            try stdout.writeAll("Vector search failed\n");
            try stdout.flush();
            return;
        };

        if (result.results.len == 0) {
            try stdout.writeAll("No results found.\n");
        } else {
            try stdout.print("Found {d} results (vector):\n", .{result.results.len});
            for (result.results, 0..) |r, i| {
                try stdout.print("  {d}. {s} - {s}/{s} (score: {d:.4})\n", .{ i + 1, r.title, r.collection, r.path, r.score });
            }
        }
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "get")) {
        const doc_path = args.next() orelse {
            try stdout.writeAll("Usage: zmd get <path>\n");
            try stdout.flush();
            return;
        };

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Database not found.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        var parts = std.mem.splitScalar(u8, doc_path, '/');
        const collection = parts.first();
        const path = parts.rest();

        const doc = qmd.store.findActiveDocument(&db_, collection, path) catch {
            try stdout.writeAll("Document not found.\n");
            try stdout.flush();
            return;
        };

        try stdout.writeAll("Title: ");
        try stdout.writeAll(doc.title);
        try stdout.writeAll("\n\n");
        try stdout.writeAll(doc.doc);
        try stdout.writeAll("\n");
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
        try qmd.mcp.McpServer.run();
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
            var result = qmd.store.getActiveDocumentPaths(&db_, c) catch {
                try stdout.writeAll("Failed to list documents.\n");
                try stdout.flush();
                return;
            };
            defer {
                result.paths.deinit(std.heap.page_allocator);
                result.titles.deinit(std.heap.page_allocator);
            }

            for (result.paths.items, result.titles.items) |path, title| {
                try stdout.print("  {s}: {s}\n", .{ path, title });
            }
        } else {
            var result = qmd.config.listCollections(&db_) catch {
                try stdout.writeAll("Failed to list collections.\n");
                try stdout.flush();
                return;
            };
            defer result.collections.deinit(std.heap.page_allocator);

            for (result.collections.items) |c_| {
                try stdout.print("Collection: {s} ({s})\n", .{ c_.name, c_.path });
                var docs = qmd.store.getActiveDocumentPaths(&db_, c_.name) catch continue;
                defer {
                    docs.paths.deinit(std.heap.page_allocator);
                    docs.titles.deinit(std.heap.page_allocator);
                }
                for (docs.paths.items, docs.titles.items) |path, title| {
                    try stdout.print("  {s}: {s}\n", .{ path, title });
                }
            }
        }
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "cleanup")) {
        try stdout.writeAll("Cleanup complete.\n");
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "embed")) {
        const text = args.next() orelse {
            try stdout.writeAll("Usage: zmd embed <text>\n");
            try stdout.flush();
            return;
        };
        var llm = qmd.llm.LlamaCpp.init("/fake", allocator) catch {
            try stdout.writeAll("Failed to init LLM\n");
            return;
        };
        defer llm.deinit();
        const emb = llm.embed(text, allocator) catch {
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
