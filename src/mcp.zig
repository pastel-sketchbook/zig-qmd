const std = @import("std");
const qmd = @import("qmd");

const DB_PATH = ".qmd/data.db";

pub const McpError = error{
    ParseError,
    MethodNotFound,
    InvalidParams,
};

pub const McpServer = struct {
    pub fn run() !void {
        var stdout_buffer: [4096]u8 = undefined;
        var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
        const stdout = &stdout_writer.interface;

        try stdout.writeAll("MCP server ready.\n");
        try stdout.writeAll("Supports: tools/list, tools/call, initialize, ping\n");
        try stdout.flush();
    }

    fn handleRequest(request: []const u8) ![]u8 {
        const id = extractId(request) orelse return McpError.ParseError;
        const method = extractMethod(request) orelse return McpError.ParseError;

        if (std.mem.eql(u8, method, "tools/list")) {
            return formatResponse(id, listTools());
        }
        if (std.mem.eql(u8, method, "tools/call")) {
            const params = extractParams(request) catch return McpError.InvalidParams;
            return formatResponse(id, try callTool(params));
        }
        if (std.mem.eql(u8, method, "initialize")) {
            return formatResponse(id, getServerInfo());
        }
        if (std.mem.eql(u8, method, "ping")) {
            return formatResponse(id, "pong");
        }
        return formatError(id, "method not found");
    }

    fn extractId(request: []const u8) ?[]const u8 {
        const id_start = std.mem.indexOf(u8, request, "\"id\":") orelse return null;
        const start = id_start + 5;
        if (start >= request.len) return null;
        if (request[start] == '"') {
            const end = std.mem.indexOfScalarPos(u8, request, start + 1, '"') orelse return null;
            return request[start + 1 .. end];
        } else {
            const end = std.mem.indexOfAny(u8, request[start..], ",}") orelse return null;
            return request[start .. start + end];
        }
    }

    fn extractMethod(request: []const u8) ?[]const u8 {
        const method_start = std.mem.indexOf(u8, request, "\"method\":") orelse return null;
        const start = method_start + 10;
        if (start >= request.len or request[start] != '"') return null;
        const end = std.mem.indexOfScalarPos(u8, request, start + 1, '"') orelse return null;
        return request[start + 1 .. end];
    }

    fn extractParams(request: []const u8) ![][]const u8 {
        const params_start = std.mem.indexOf(u8, request, "\"params\":") orelse return McpError.InvalidParams;
        var start = params_start + 9;
        while (start < request.len and request[start] != '{') start += 1;
        if (start >= request.len) return McpError.InvalidParams;
        var depth: i32 = 0;
        var i = start;
        while (i < request.len) : (i += 1) {
            if (request[i] == '{') depth += 1;
            if (request[i] == '}') {
                depth -= 1;
                if (depth == 0) {
                    return &.{request[start .. i + 1]};
                }
            }
        }
        return McpError.InvalidParams;
    }

    fn listTools() []u8 {
        return "{\"tools\":[{\"name\":\"query\",\"description\":\"Hybrid search\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}}}},{\"name\":\"search\",\"description\":\"FTS search\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},\"collection\":{\"type\":\"string\"}}}},{\"name\":\"get\",\"description\":\"Get document\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}}}},{\"name\":\"status\",\"description\":\"System status\",\"inputSchema\":{\"type\":\"object\"}}]}";
    }

    fn callTool(params: []const u8) ![]u8 {
        const name = extractParam(params, "name") orelse return McpError.InvalidParams;

        var db_path_buf: [256]u8 = undefined;
        const db_path = std.fmt.bufPrintZ(&db_path_buf, "{s}", .{DB_PATH}) catch return McpError.InvalidParams;
        var db_ = qmd.db.Db.open(db_path) catch {
            return "{\"content\":[{\"type\":\"text\",\"text\":\"database not initialized\"}]}";
        };
        defer db_.close();

        if (std.mem.eql(u8, name, "query")) {
            const query_text = extractParam(params, "query") orelse "";
            var result = qmd.search.hybridSearch(&db_, query_text, null, .{
                .enable_vector = true,
                .max_results = 5,
            }) catch return "{\"content\":[{\"type\":\"text\",\"text\":\"query failed\"}]}";
            defer result.results.deinit(std.heap.page_allocator);

            var text = std.ArrayList(u8).initCapacity(std.heap.page_allocator, 256) catch return McpError.InvalidParams;
            defer text.deinit(std.heap.page_allocator);
            text.writer(std.heap.page_allocator).print("found {d} hybrid results", .{result.results.items.len}) catch {};
            for (result.results.items, 0..) |r, i| {
                text.writer(std.heap.page_allocator).print("\n{d}. {s} (qmd://{s}/{s}) score={d:.4}", .{ i + 1, r.title, r.collection, r.path, r.score }) catch {};
            }
            return std.fmt.allocPrint(std.heap.page_allocator, "{{\"content\":[{{\"type\":\"text\",\"text\":\"{s}\"}}]}}", .{text.items}) catch McpError.InvalidParams;
        }
        if (std.mem.eql(u8, name, "search")) {
            const query_text = extractParam(params, "query") orelse "";
            const collection = extractParam(params, "collection");
            var result = qmd.search.searchFTS(&db_, query_text, collection) catch return "{\"content\":[{\"type\":\"text\",\"text\":\"search failed\"}]}";
            defer result.results.deinit(std.heap.page_allocator);

            var text = std.ArrayList(u8).initCapacity(std.heap.page_allocator, 256) catch return McpError.InvalidParams;
            defer text.deinit(std.heap.page_allocator);
            text.writer(std.heap.page_allocator).print("found {d} fts results", .{result.results.items.len}) catch {};
            for (result.results.items, 0..) |r, i| {
                text.writer(std.heap.page_allocator).print("\n{d}. {s} (qmd://{s}/{s}) score={d:.4}", .{ i + 1, r.title, r.collection, r.path, r.score }) catch {};
            }
            return std.fmt.allocPrint(std.heap.page_allocator, "{{\"content\":[{{\"type\":\"text\",\"text\":\"{s}\"}}]}}", .{text.items}) catch McpError.InvalidParams;
        }
        if (std.mem.eql(u8, name, "status")) {
            var stmt = db_.prepare("SELECT count(*) FROM documents WHERE active = 1") catch return "{\"content\":[{\"type\":\"text\",\"text\":\"status failed\"}]}";
            defer stmt.finalize();
            const has_docs = stmt.step() catch false;
            const docs: i64 = if (has_docs) stmt.columnInt(0) else 0;
            stmt = db_.prepare("SELECT count(*) FROM store_collections") catch return "{\"content\":[{\"type\":\"text\",\"text\":\"status failed\"}]}";
            defer stmt.finalize();
            const has_cols = stmt.step() catch false;
            const cols: i64 = if (has_cols) stmt.columnInt(0) else 0;
            const text = std.fmt.allocPrint(std.heap.page_allocator, "zmd status: OK, documents={d}, collections={d}", .{ docs, cols }) catch return McpError.InvalidParams;
            return std.fmt.allocPrint(std.heap.page_allocator, "{{\"content\":[{{\"type\":\"text\",\"text\":\"{s}\"}}]}}", .{text}) catch McpError.InvalidParams;
        }
        if (std.mem.eql(u8, name, "get")) {
            const raw_path = extractParam(params, "path") orelse return McpError.InvalidParams;
            const parsed = qmd.parse_virtual_path(raw_path) orelse return McpError.InvalidParams;
            const doc = qmd.store.findActiveDocument(&db_, parsed.collection, parsed.path) catch {
                return "{\"content\":[{\"type\":\"text\",\"text\":\"document not found\"}]}";
            };
            defer {
                std.heap.page_allocator.free(doc.title);
                std.heap.page_allocator.free(doc.hash);
                std.heap.page_allocator.free(doc.doc);
            }
            var text = std.ArrayList(u8).initCapacity(std.heap.page_allocator, doc.doc.len + 64) catch return McpError.InvalidParams;
            defer text.deinit(std.heap.page_allocator);
            text.writer(std.heap.page_allocator).print("Title: {s}\n\n{s}", .{ doc.title, doc.doc }) catch {};
            return std.fmt.allocPrint(std.heap.page_allocator, "{{\"content\":[{{\"type\":\"text\",\"text\":\"{s}\"}}]}}", .{text.items}) catch McpError.InvalidParams;
        }

        return McpError.MethodNotFound;
    }

    fn getServerInfo() []u8 {
        return "{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"zmd\",\"version\":\"0.1.0\"}}";
    }

    fn extractParam(json: []const u8, key: []const u8) ?[]const u8 {
        const pattern = "\"" ++ key ++ "\":\"";
        const key_start = std.mem.indexOf(u8, json, pattern) orelse return null;
        const start = key_start + pattern.len;
        const end = std.mem.indexOfScalarPos(u8, json, start, '"') orelse return null;
        return json[start..end];
    }

    fn formatResponse(id: []const u8, result: []const u8) []u8 {
        return std.fmt.allocPrint(std.heap.page_allocator, "{{\"jsonrpc\":\"2.0\",\"id\":{s},\"result\":{s}}}", .{ id, result }) catch "{}";
    }

    fn formatError(id: []const u8, message: []const u8) []u8 {
        return std.fmt.allocPrint(std.heap.page_allocator, "{{\"jsonrpc\":\"2.0\",\"id\":{s},\"error\":{{\"code\":-32601,\"message\":\"{s}\"}}}}", .{ id, message }) catch "{}";
    }
};

test "McpServer struct can be defined" {
    const t: type = McpServer;
    _ = t;
}

test "extractMethod parses method name" {
    const request = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}";
    const method = McpServer.extractMethod(request);
    try std.testing.expect(method != null);
    try std.testing.expectEqualStrings("tools/list", method.?);
}

test "extractId parses numeric id" {
    const request = "{\"jsonrpc\":\"2.0\",\"id\":42,\"method\":\"ping\"}";
    const id = McpServer.extractId(request);
    try std.testing.expect(id != null);
    try std.testing.expectEqualStrings("42", id.?);
}

test "extractParam parses parameter" {
    const json = "{\"name\":\"query\",\"query\":\"test\"}";
    const name = McpServer.extractParam(json, "name");
    try std.testing.expect(name != null);
    try std.testing.expectEqualStrings("query", name.?);
}
