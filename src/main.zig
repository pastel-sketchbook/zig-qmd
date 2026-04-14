const std = @import("std");
const qmd = @import("qmd");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

const DB_PATH = ".qmd/data.db";
const DEFAULT_LLAMA_EMBED_BIN = "deps/llama.cpp/build/bin/llama-embedding";
const DEFAULT_LLAMA_MODEL_PATH = "";

const OutputFormat = enum {
    text,
    json,
    csv,
    md,
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

fn parseDocRef(input: []const u8) ?DocRef {
    const raw = if (std.mem.startsWith(u8, input, "qmd://")) input[6..] else input;
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

fn extractSnippet(allocator: std.mem.Allocator, query: []const u8, doc: []const u8) ![]u8 {
    if (doc.len == 0) return allocator.dupe(u8, "");

    var tok_it = std.mem.tokenizeScalar(u8, query, ' ');
    const token = tok_it.next() orelse return allocator.dupe(u8, doc[0..@min(doc.len, 180)]);

    const lower_doc = try allocator.alloc(u8, doc.len);
    defer allocator.free(lower_doc);
    for (doc, 0..) |ch, i| lower_doc[i] = std.ascii.toLower(ch);

    const lower_tok = try allocator.alloc(u8, token.len);
    defer allocator.free(lower_tok);
    for (token, 0..) |ch, i| lower_tok[i] = std.ascii.toLower(ch);

    const idx = std.mem.indexOf(u8, lower_doc, lower_tok) orelse 0;
    const start: usize = if (idx > 60) idx - 60 else 0;
    const end = @min(doc.len, idx + token.len + 140);

    var out = try std.ArrayList(u8).initCapacity(allocator, end - start + 8);
    defer out.deinit(allocator);
    if (start > 0) try out.appendSlice(allocator, "...");
    try out.appendSlice(allocator, doc[start..end]);
    if (end < doc.len) try out.appendSlice(allocator, "...");
    return out.toOwnedSlice(allocator);
}

fn make_embedding_engine(allocator: std.mem.Allocator) ?qmd.llm.LlamaEmbedding {
    const bin_path = std.process.getEnvVarOwned(allocator, "QMD_LLAMA_EMBED_BIN") catch allocator.dupe(u8, DEFAULT_LLAMA_EMBED_BIN) catch return null;

    const model_path = std.process.getEnvVarOwned(allocator, "QMD_LLAMA_MODEL") catch allocator.dupe(u8, DEFAULT_LLAMA_MODEL_PATH) catch {
        allocator.free(bin_path);
        return null;
    };

    if (model_path.len == 0) {
        allocator.free(bin_path);
        allocator.free(model_path);
        return null;
    }

    defer {
        allocator.free(bin_path);
        allocator.free(model_path);
    }

    return qmd.llm.LlamaEmbedding.init(allocator, bin_path, model_path) catch null;
}

pub fn main() !void {
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.next();

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
        try stdout.writeAll("  update     Update index for a collection\n");
        try stdout.writeAll("  search     Full-text search\n");
        try stdout.writeAll("  vsearch    Vector semantic search\n");
        try stdout.writeAll("  query      Hybrid search (FTS + vector)\n");
        try stdout.writeAll("  context    Context-rich search snippets\n");
        try stdout.writeAll("  get        Get document by path\n");
        try stdout.writeAll("  multi-get  Get multiple documents by path\n");
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
            defer qmd.config.freeCollections(&result);

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
        defer qmd.config.freeCollections(&collections_result);

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

                    const doc_hash = qmd.store.findActiveDocumentHash(&db_, col.name, entry.path) catch {
                        total_indexed += 1;
                        continue;
                    };

                    var chunks = qmd.chunker.chunkDocument(content);
                    defer chunks.chunks.deinit(std.heap.page_allocator);

                    if (make_embedding_engine(allocator)) |engine_instance| {
                        var engine = engine_instance;
                        defer engine.deinit();
                        for (chunks.chunks.items, 0..) |chunk, idx| {
                            const formatted = qmd.llm.formatDocForEmbedding(allocator, chunk) catch continue;
                            defer allocator.free(formatted);
                            const emb = engine.embed(formatted) catch continue;
                            defer allocator.free(emb);
                            qmd.store.upsertContentVectorAt(&db_, doc_hash[0..], @intCast(idx), 0, engine.model_path, emb, allocator) catch {};
                        }
                    } else {
                        // fallback deterministic embedding for now
                        var fallback = qmd.llm.LlamaCpp.init("/nonexistent", allocator) catch {
                            total_indexed += 1;
                            continue;
                        };
                        defer fallback.deinit();
                        for (chunks.chunks.items, 0..) |chunk, idx| {
                            const formatted = qmd.llm.formatDocForEmbedding(allocator, chunk) catch continue;
                            defer allocator.free(formatted);
                            const emb = fallback.embed(formatted, allocator) catch continue;
                            defer allocator.free(emb);
                            qmd.store.upsertContentVectorAt(&db_, doc_hash[0..], @intCast(idx), 0, "fallback-fnv", emb, allocator) catch {};
                        }
                    }
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
        const first_arg = args.next() orelse {
            try stdout.writeAll("Usage: zmd query <query>\n");
            try stdout.flush();
            return;
        };
        if (std.mem.eql(u8, first_arg, "--help") or std.mem.eql(u8, first_arg, "-h")) {
            try stdout.writeAll("Usage: zmd query <query> [--expand] [--rerank] [--json|--csv|--md]\n");
            try stdout.flush();
            return;
        }
        const query_text = first_arg;

        var enable_expand = false;
        var enable_rerank = false;
        var output_format: OutputFormat = .text;
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
        }

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Database not found. Run 'zmd update' first.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        var result = qmd.search.hybridSearch(&db_, query_text, null, .{
            .enable_vector = true,
            .enable_query_expansion = enable_expand,
            .enable_rerank = enable_rerank,
            .rrf_k = qmd.search.RRF_K,
            .max_results = 10,
        }) catch {
            try stdout.writeAll("Search failed\n");
            try stdout.flush();
            return;
        };
        defer result.results.deinit(std.heap.page_allocator);

        switch (output_format) {
            .text => {
                try stdout.print("Found {d} results (hybrid)\n", .{result.results.items.len});
                for (result.results.items, 0..) |r, i| {
                    try stdout.print("  {d}. {s} ({s}) - score: {d:.4}\n", .{ i + 1, r.title, r.collection, r.score });
                }
            },
            .json => {
                try stdout.writeAll("[\n");
                for (result.results.items, 0..) |r, i| {
                    if (i > 0) try stdout.writeAll(",\n");
                    const vpath = try std.fmt.allocPrint(allocator, "qmd://{s}/{s}", .{ r.collection, r.path });
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
                    const vpath = try std.fmt.allocPrint(allocator, "qmd://{s}/{s}", .{ r.collection, r.path });
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
                    try stdout.print("| {d} | {d:.4} | {s} | qmd://{s}/{s} | {s} |\n", .{ i + 1, r.score, r.collection, r.collection, r.path, r.title });
                }
            },
        }
        try stdout.flush();
        return;
    }

    if (std.mem.eql(u8, cmd, "context")) {
        const first_arg = args.next() orelse {
            try stdout.writeAll("Usage: zmd context <query> [--json|--csv|--md]\n");
            try stdout.flush();
            return;
        };
        if (std.mem.eql(u8, first_arg, "--help") or std.mem.eql(u8, first_arg, "-h")) {
            try stdout.writeAll("Usage: zmd context <query> [--json|--csv|--md]\n");
            try stdout.flush();
            return;
        }
        const query_text = first_arg;

        var output_format: OutputFormat = .text;
        while (args.next()) |arg| {
            if (parseOutputFlag(arg)) |fmt| output_format = fmt;
        }

        var db_path_buf: [256]u8 = undefined;
        const db_path = try std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH});
        var db_ = qmd.db.Db.open(db_path) catch {
            try stdout.writeAll("Error: Database not found. Run 'zmd update' first.\n");
            try stdout.flush();
            return;
        };
        defer db_.close();

        var result = qmd.search.hybridSearch(&db_, query_text, null, .{
            .enable_vector = true,
            .max_results = 5,
        }) catch {
            try stdout.writeAll("Context search failed\n");
            try stdout.flush();
            return;
        };
        defer result.results.deinit(std.heap.page_allocator);

        switch (output_format) {
            .json => {
                try stdout.writeAll("[\n");
                var first = true;
                for (result.results.items) |r| {
                    const doc = qmd.store.findActiveDocument(&db_, r.collection, r.path) catch continue;
                    defer {
                        std.heap.page_allocator.free(doc.title);
                        std.heap.page_allocator.free(doc.hash);
                        std.heap.page_allocator.free(doc.doc);
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
                    const doc = qmd.store.findActiveDocument(&db_, r.collection, r.path) catch continue;
                    defer {
                        std.heap.page_allocator.free(doc.title);
                        std.heap.page_allocator.free(doc.hash);
                        std.heap.page_allocator.free(doc.doc);
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
                    const doc = qmd.store.findActiveDocument(&db_, r.collection, r.path) catch continue;
                    defer {
                        std.heap.page_allocator.free(doc.title);
                        std.heap.page_allocator.free(doc.hash);
                        std.heap.page_allocator.free(doc.doc);
                    }
                    const snippet = try extractSnippet(allocator, query_text, doc.doc);
                    defer allocator.free(snippet);
                    try stdout.print("| {d} | {d:.4} | qmd://{s}/{s} | {s} | {s} |\n", .{ i + 1, r.score, r.collection, r.path, r.title, snippet });
                }
            },
            else => {
                if (result.results.items.len == 0) {
                    try stdout.writeAll("No context results found.\n");
                } else {
                    for (result.results.items, 0..) |r, i| {
                        const doc = qmd.store.findActiveDocument(&db_, r.collection, r.path) catch continue;
                        defer {
                            std.heap.page_allocator.free(doc.title);
                            std.heap.page_allocator.free(doc.hash);
                            std.heap.page_allocator.free(doc.doc);
                        }
                        const snippet = try extractSnippet(allocator, query_text, doc.doc);
                        defer allocator.free(snippet);
                        if (i > 0) try stdout.writeAll("\n");
                        try stdout.print("{d}. {s} (qmd://{s}/{s}) score={d:.4}\n", .{ i + 1, r.title, r.collection, r.path, r.score });
                        try stdout.print("   {s}\n", .{snippet});
                    }
                }
            },
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
            try stdout.writeAll("Usage: zmd search <query> [collection] [--json|--csv|--md]\n");
            try stdout.flush();
            return;
        }
        const query_text = first_arg;
        var collection: ?[]const u8 = null;
        var output_format: OutputFormat = .text;
        while (args.next()) |arg| {
            if (parseOutputFlag(arg)) |fmt| {
                output_format = fmt;
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

        var result = qmd.search.searchFTS(&db_, query_text, collection) catch {
            try stdout.writeAll("Search failed\n");
            try stdout.flush();
            return;
        };
        defer result.results.deinit(std.heap.page_allocator);

        switch (output_format) {
            .text => {
                if (result.results.items.len == 0) {
                    try stdout.writeAll("No results found.\n");
                } else {
                    try stdout.print("Found {d} results:\n", .{result.results.items.len});
                    for (result.results.items, 0..) |r, i| {
                        try stdout.print("  {d}. title='{s}' collection='{s}' path='{s}' score={d:.4}\n", .{ i + 1, r.title, r.collection, r.path, r.score });
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
                    try stdout.print("| {d} | {d:.4} | {s} | qmd://{s}/{s} | {s} |\n", .{ i + 1, r.score, r.collection, r.collection, r.path, r.title });
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
            try stdout.writeAll("Usage: zmd vsearch <query> [--json|--csv|--md]\n");
            try stdout.flush();
            return;
        }
        const query_text = first_arg;

        var output_format: OutputFormat = .text;
        while (args.next()) |arg| {
            if (parseOutputFlag(arg)) |fmt| output_format = fmt;
        }

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

        switch (output_format) {
            .text => {
                if (result.results.len == 0) {
                    try stdout.writeAll("No results found.\n");
                } else {
                    try stdout.print("Found {d} results (vector):\n", .{result.results.len});
                    for (result.results, 0..) |r, i| {
                        try stdout.print("  {d}. {s} - {s}/{s} (score: {d:.4})\n", .{ i + 1, r.title, r.collection, r.path, r.score });
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
                    try stdout.print("| {d} | {d:.4} | {s} | qmd://{s}/{s} | {s} |\n", .{ i + 1, r.score, r.collection, r.collection, r.path, r.title });
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
            try stdout.writeAll("Usage: zmd get <collection/path|qmd://collection/path>\n");
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
            try stdout.writeAll("Invalid document path. Use collection/path or qmd://collection/path\n");
            try stdout.flush();
            return;
        };

        const doc = qmd.store.findActiveDocument(&db_, ref.collection, ref.path) catch {
            try stdout.writeAll("Document not found.\n");
            try stdout.flush();
            return;
        };
        defer {
            std.heap.page_allocator.free(doc.title);
            std.heap.page_allocator.free(doc.hash);
            std.heap.page_allocator.free(doc.doc);
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
                    const doc = qmd.store.findActiveDocument(&db_, ref.collection, ref.path) catch {
                        try stdout.print("Not found: qmd://{s}/{s}\n", .{ ref.collection, ref.path });
                        continue;
                    };
                    defer {
                        std.heap.page_allocator.free(doc.title);
                        std.heap.page_allocator.free(doc.hash);
                        std.heap.page_allocator.free(doc.doc);
                    }

                    if (i > 0) try stdout.writeAll("\n---\n\n");
                    try stdout.print("# {s}\n", .{doc.title});
                    try stdout.print("Path: qmd://{s}/{s}\n\n", .{ ref.collection, ref.path });
                    try stdout.writeAll(doc.doc);
                    try stdout.writeAll("\n");
                }
            },
            .json => {
                try stdout.writeAll("[\n");
                var first = true;
                for (refs.items) |raw_ref| {
                    const ref = parseDocRef(raw_ref) orelse continue;
                    const doc = qmd.store.findActiveDocument(&db_, ref.collection, ref.path) catch continue;
                    defer {
                        std.heap.page_allocator.free(doc.title);
                        std.heap.page_allocator.free(doc.hash);
                        std.heap.page_allocator.free(doc.doc);
                    }
                    if (!first) try stdout.writeAll(",\n");
                    first = false;

                    const vpath = try std.fmt.allocPrint(allocator, "qmd://{s}/{s}", .{ ref.collection, ref.path });

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
                    const doc = qmd.store.findActiveDocument(&db_, ref.collection, ref.path) catch continue;
                    defer {
                        std.heap.page_allocator.free(doc.title);
                        std.heap.page_allocator.free(doc.hash);
                        std.heap.page_allocator.free(doc.doc);
                    }
                    const vpath = try std.fmt.allocPrint(allocator, "qmd://{s}/{s}", .{ ref.collection, ref.path });

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
                    const doc = qmd.store.findActiveDocument(&db_, ref.collection, ref.path) catch continue;
                    defer {
                        std.heap.page_allocator.free(doc.title);
                        std.heap.page_allocator.free(doc.hash);
                        std.heap.page_allocator.free(doc.doc);
                    }
                    if (i > 0) try stdout.writeAll("\n\n---\n\n");
                    try stdout.print("## {s}\n\n", .{doc.title});
                    try stdout.print("- path: `qmd://{s}/{s}`\n\n", .{ ref.collection, ref.path });
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
                for (result.paths.items) |p| std.heap.page_allocator.free(p);
                for (result.titles.items) |t| std.heap.page_allocator.free(t);
                result.paths.deinit(std.heap.page_allocator);
                result.titles.deinit(std.heap.page_allocator);
            }

            for (result.paths.items, result.titles.items) |path, title| {
                try stdout.print("  qmd://{s}/{s}: {s}\n", .{ c, path, title });
            }
        } else {
            var result = qmd.config.listCollections(&db_) catch {
                try stdout.writeAll("Failed to list collections.\n");
                try stdout.flush();
                return;
            };
            defer qmd.config.freeCollections(&result);

            for (result.collections.items) |c_| {
                try stdout.print("Collection: {s} ({s})\n", .{ c_.name, c_.path });
                var docs = qmd.store.getActiveDocumentPaths(&db_, c_.name) catch continue;
                defer {
                    for (docs.paths.items) |p| std.heap.page_allocator.free(p);
                    for (docs.titles.items) |t| std.heap.page_allocator.free(t);
                    docs.paths.deinit(std.heap.page_allocator);
                    docs.titles.deinit(std.heap.page_allocator);
                }
                for (docs.paths.items, docs.titles.items) |path, title| {
                    try stdout.print("  qmd://{s}/{s}: {s}\n", .{ c_.name, path, title });
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

        qmd.store.vacuum(&db_) catch {};

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

test "parseOutputFlag parses known flags" {
    try std.testing.expect(parseOutputFlag("--json").? == .json);
    try std.testing.expect(parseOutputFlag("--csv").? == .csv);
    try std.testing.expect(parseOutputFlag("--md").? == .md);
    try std.testing.expect(parseOutputFlag("--nope") == null);
}

test "parseDocRef parses virtual and plain paths" {
    const a = parseDocRef("qmd://notes/a.md") orelse return error.TestExpectedEqual;
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
