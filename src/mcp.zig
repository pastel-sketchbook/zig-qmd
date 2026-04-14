const std = @import("std");
const db = @import("db.zig");
const config = @import("config.zig");
const search = @import("search.zig");
const store = @import("store.zig");
const root = @import("root.zig");

const DB_PATH = ".qmd/data.db";

pub const McpError = error{
    ParseError,
    MethodNotFound,
    InvalidParams,
};

pub const McpServer = struct {
    pub fn run() !void {
        const allocator = std.heap.page_allocator;

        var stdin_buffer: [4096]u8 = undefined;
        var stdin_reader = std.fs.File.stdin().reader(&stdin_buffer);
        const stdin = &stdin_reader.interface;

        var stdout_buffer: [4096]u8 = undefined;
        var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
        const stdout = &stdout_writer.interface;

        while (true) {
            const msg = readMessage(stdin, allocator) catch break;
            defer allocator.free(msg);

            const response = handleRequestWithDbPath(msg, null) catch |err| {
                const err_json = std.fmt.allocPrint(
                    allocator,
                    "{{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{{\"code\":-32600,\"message\":\"{s}\"}}}}",
                    .{@errorName(err)},
                ) catch "{}";
                defer if (!std.mem.eql(u8, err_json, "{}")) allocator.free(err_json);
                try writeMessage(stdout, err_json);
                continue;
            };
            defer allocator.free(response);

            try writeMessage(stdout, response);
        }

        try stdout.flush();
    }

    fn readMessage(stdin: anytype, allocator: std.mem.Allocator) ![]u8 {
        var content_length: usize = 0;

        while (true) {
            const line = try stdin.takeDelimiterExclusive('\n');
            const trimmed = std.mem.trim(u8, line, &.{'\r'});
            if (trimmed.len == 0) break;

            if (std.mem.startsWith(u8, trimmed, "Content-Length:")) {
                const value = std.mem.trim(u8, trimmed[15..], &.{' '});
                content_length = std.fmt.parseInt(usize, value, 10) catch return McpError.ParseError;
            }
        }

        if (content_length == 0) return McpError.ParseError;

        const body = try allocator.alloc(u8, content_length);
        errdefer allocator.free(body);
        _ = try stdin.readSliceAll(body);
        return body;
    }

    fn writeMessage(stdout: anytype, body: []const u8) !void {
        try stdout.print("Content-Length: {d}\r\n\r\n", .{body.len});
        try stdout.writeAll(body);
        try stdout.flush();
    }

    fn handleRequest(request: []const u8) ![]u8 {
        return handleRequestWithDbPath(request, null);
    }

    fn handleRequestWithDbPath(request: []const u8, db_path_override: ?[]const u8) ![]u8 {
        const version = extractJsonRpc(request) orelse return McpError.ParseError;
        if (!std.mem.eql(u8, version, "2.0")) return McpError.ParseError;

        const id = extractId(request) orelse return McpError.ParseError;
        const method = extractMethod(request) orelse return McpError.ParseError;

        if (std.mem.eql(u8, method, "tools/list")) {
            return try formatResponse(id, listTools());
        }
        if (std.mem.eql(u8, method, "tools/call")) {
            const params = extractParams(request) catch return McpError.InvalidParams;
            return try formatResponse(id, try callToolAtPath(params, db_path_override orelse DB_PATH));
        }
        if (std.mem.eql(u8, method, "initialize")) {
            return try formatResponse(id, getServerInfo());
        }
        if (std.mem.eql(u8, method, "ping")) {
            return try formatResponse(id, "pong");
        }
        return try formatError(id, "method not found");
    }

    fn extractJsonRpc(request: []const u8) ?[]const u8 {
        const start_key = std.mem.indexOf(u8, request, "\"jsonrpc\":\"") orelse return null;
        const start = start_key + 11;
        const end = std.mem.indexOfScalarPos(u8, request, start, '"') orelse return null;
        return request[start..end];
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

    fn extractParams(request: []const u8) ![]const u8 {
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
                    return request[start .. i + 1];
                }
            }
        }
        return McpError.InvalidParams;
    }

    fn listTools() []const u8 {
        return "{\"tools\":[{\"name\":\"query\",\"description\":\"Hybrid search\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}}}},{\"name\":\"search\",\"description\":\"FTS search\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},\"collection\":{\"type\":\"string\"}}}},{\"name\":\"get\",\"description\":\"Get document\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}}}},{\"name\":\"status\",\"description\":\"System status\",\"inputSchema\":{\"type\":\"object\"}}]}";
    }

    fn callTool(params: []const u8) ![]u8 {
        return callToolAtPath(params, DB_PATH);
    }

    fn callToolAtPath(params: []const u8, db_path_raw: []const u8) ![]u8 {
        const name = extractParam(params, "name") orelse return McpError.InvalidParams;

        const db_path = std.heap.page_allocator.dupeZ(u8, db_path_raw) catch return McpError.InvalidParams;
        defer std.heap.page_allocator.free(db_path);

        var db_ = db.Db.open(db_path) catch {
            return std.heap.page_allocator.dupe(u8, "{\"content\":[{\"type\":\"text\",\"text\":\"database not initialized\"}]}") catch McpError.InvalidParams;
        };
        defer db_.close();

        if (std.mem.eql(u8, name, "query")) {
            const query_text = extractParam(params, "query") orelse "";
            var result = search.hybridSearch(&db_, query_text, null, .{
                .enable_vector = true,
                .max_results = 5,
            }) catch return std.heap.page_allocator.dupe(u8, "{\"content\":[{\"type\":\"text\",\"text\":\"query failed\"}]}") catch McpError.InvalidParams;
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
            var result = search.searchFTS(&db_, query_text, collection) catch return std.heap.page_allocator.dupe(u8, "{\"content\":[{\"type\":\"text\",\"text\":\"search failed\"}]}") catch McpError.InvalidParams;
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
            var stmt = db_.prepare("SELECT count(*) FROM documents WHERE active = 1") catch return std.heap.page_allocator.dupe(u8, "{\"content\":[{\"type\":\"text\",\"text\":\"status failed\"}]}") catch McpError.InvalidParams;
            defer stmt.finalize();
            const has_docs = stmt.step() catch false;
            const docs: i64 = if (has_docs) stmt.columnInt(0) else 0;
            stmt = db_.prepare("SELECT count(*) FROM store_collections") catch return std.heap.page_allocator.dupe(u8, "{\"content\":[{\"type\":\"text\",\"text\":\"status failed\"}]}") catch McpError.InvalidParams;
            defer stmt.finalize();
            const has_cols = stmt.step() catch false;
            const cols: i64 = if (has_cols) stmt.columnInt(0) else 0;
            const text = std.fmt.allocPrint(std.heap.page_allocator, "zmd status: OK, documents={d}, collections={d}", .{ docs, cols }) catch return McpError.InvalidParams;
            return std.fmt.allocPrint(std.heap.page_allocator, "{{\"content\":[{{\"type\":\"text\",\"text\":\"{s}\"}}]}}", .{text}) catch McpError.InvalidParams;
        }
        if (std.mem.eql(u8, name, "get")) {
            const raw_path = extractParam(params, "path") orelse return McpError.InvalidParams;
            const parsed = root.parse_virtual_path(raw_path) orelse return McpError.InvalidParams;
            const doc = store.findActiveDocument(&db_, parsed.collection, parsed.path) catch {
                return std.heap.page_allocator.dupe(u8, "{\"content\":[{\"type\":\"text\",\"text\":\"document not found\"}]}") catch McpError.InvalidParams;
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

    fn getServerInfo() []const u8 {
        return "{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"zmd\",\"version\":\"0.1.0\"}}";
    }

    fn extractParam(json: []const u8, key: []const u8) ?[]const u8 {
        const pattern = std.fmt.allocPrint(std.heap.page_allocator, "\"{s}\":\"", .{key}) catch return null;
        defer std.heap.page_allocator.free(pattern);
        const key_start = std.mem.indexOf(u8, json, pattern) orelse return null;
        const start = key_start + pattern.len;
        const end = std.mem.indexOfScalarPos(u8, json, start, '"') orelse return null;
        return json[start..end];
    }

    fn formatResponse(id: []const u8, result: []const u8) ![]u8 {
        return std.fmt.allocPrint(std.heap.page_allocator, "{{\"jsonrpc\":\"2.0\",\"id\":{s},\"result\":{s}}}", .{ id, result });
    }

    fn formatError(id: []const u8, message: []const u8) ![]u8 {
        return std.fmt.allocPrint(std.heap.page_allocator, "{{\"jsonrpc\":\"2.0\",\"id\":{s},\"error\":{{\"code\":-32601,\"message\":\"{s}\"}}}}", .{ id, message });
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

test "handleRequest supports ping" {
    const req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}";
    const resp = try McpServer.handleRequest(req);
    defer std.heap.page_allocator.free(resp);
    try std.testing.expect(std.mem.indexOf(u8, resp, "pong") != null);
}

test "handleRequest rejects missing id" {
    const req = "{\"jsonrpc\":\"2.0\",\"method\":\"ping\"}";
    try std.testing.expectError(McpError.ParseError, McpServer.handleRequest(req));
}

test "handleRequest rejects non-2.0 jsonrpc" {
    const req = "{\"jsonrpc\":\"1.0\",\"id\":1,\"method\":\"ping\"}";
    try std.testing.expectError(McpError.ParseError, McpServer.handleRequest(req));
}

test "handleRequest rejects malformed params in tools/call" {
    const req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":\"oops\"}";
    try std.testing.expectError(McpError.InvalidParams, McpServer.handleRequest(req));
}

test "handleRequest returns method error for unknown tool" {
    const allocator = std.testing.allocator;
    const db_path = try setupMcpTestDb(allocator);
    defer cleanupMcpTestDb(db_path);
    defer allocator.free(db_path);

    const req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"unknown\"}}";
    try std.testing.expectError(McpError.MethodNotFound, McpServer.handleRequestWithDbPath(req, db_path));
}

const FakeReader = struct {
    data: []const u8,
    pos: usize = 0,

    fn takeDelimiterExclusive(self: *FakeReader, delimiter: u8) ![]const u8 {
        const start = self.pos;
        while (self.pos < self.data.len and self.data[self.pos] != delimiter) : (self.pos += 1) {}
        if (self.pos >= self.data.len) return error.EndOfStream;
        const end = self.pos;
        self.pos += 1;
        return self.data[start..end];
    }

    fn readSliceAll(self: *FakeReader, out: []u8) !usize {
        if (self.pos + out.len > self.data.len) return error.EndOfStream;
        std.mem.copyForwards(u8, out, self.data[self.pos .. self.pos + out.len]);
        self.pos += out.len;
        return out.len;
    }
};

const FakeWriter = struct {
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,

    fn print(self: *FakeWriter, comptime fmt: []const u8, args: anytype) !void {
        try self.buf.writer(self.allocator).print(fmt, args);
    }

    fn writeAll(self: *FakeWriter, data: []const u8) !void {
        try self.buf.appendSlice(self.allocator, data);
    }

    fn flush(_: *FakeWriter) !void {}
};

test "readMessage parses framed body" {
    const framed = "Content-Length: 17\r\n\r\n{\"method\":\"ping\"}";
    var reader = FakeReader{ .data = framed };

    const body = try McpServer.readMessage(&reader, std.testing.allocator);
    defer std.testing.allocator.free(body);
    try std.testing.expectEqualStrings("{\"method\":\"ping\"}", body);
}

test "readMessage rejects missing content length" {
    const framed = "X-Test: 1\r\n\r\n{}";
    var reader = FakeReader{ .data = framed };
    try std.testing.expectError(McpError.ParseError, McpServer.readMessage(&reader, std.testing.allocator));
}

test "writeMessage emits content-length framing" {
    var buf = std.ArrayList(u8).initCapacity(std.testing.allocator, 0) catch unreachable;
    defer buf.deinit(std.testing.allocator);

    var writer = FakeWriter{ .buf = &buf, .allocator = std.testing.allocator };
    try McpServer.writeMessage(&writer, "{}");

    try std.testing.expect(std.mem.startsWith(u8, buf.items, "Content-Length: 2\r\n\r\n{}"));
}

test "writeMessage and readMessage roundtrip" {
    var out = std.ArrayList(u8).initCapacity(std.testing.allocator, 0) catch unreachable;
    defer out.deinit(std.testing.allocator);

    var writer = FakeWriter{ .buf = &out, .allocator = std.testing.allocator };
    const payload = "{\"jsonrpc\":\"2.0\",\"method\":\"ping\"}";
    try McpServer.writeMessage(&writer, payload);

    var reader = FakeReader{ .data = out.items };
    const body = try McpServer.readMessage(&reader, std.testing.allocator);
    defer std.testing.allocator.free(body);
    try std.testing.expectEqualStrings(payload, body);
}

fn processFramedRequestForTest(request_body: []const u8, db_path_override: ?[]const u8, allocator: std.mem.Allocator) ![]u8 {
    var inbound = std.ArrayList(u8).initCapacity(allocator, 0) catch unreachable;
    defer inbound.deinit(allocator);
    var in_writer = FakeWriter{ .buf = &inbound, .allocator = allocator };
    try McpServer.writeMessage(&in_writer, request_body);

    var reader = FakeReader{ .data = inbound.items };
    const parsed = try McpServer.readMessage(&reader, allocator);
    defer allocator.free(parsed);

    const response = try McpServer.handleRequestWithDbPath(parsed, db_path_override);
    defer allocator.free(response);

    var outbound = std.ArrayList(u8).initCapacity(allocator, 0) catch unreachable;
    errdefer outbound.deinit(allocator);
    var out_writer = FakeWriter{ .buf = &outbound, .allocator = allocator };
    try McpServer.writeMessage(&out_writer, response);

    var out_reader = FakeReader{ .data = outbound.items };
    const roundtrip = try McpServer.readMessage(&out_reader, allocator);
    outbound.deinit(allocator);
    return roundtrip;
}

fn setupMcpTestDb(allocator: std.mem.Allocator) ![]u8 {
    var rnd: u64 = undefined;
    std.crypto.random.bytes(std.mem.asBytes(&rnd));
    const dir_path = try std.fmt.allocPrint(allocator, "/tmp/zmd-mcp-test-{x}", .{rnd});
    errdefer allocator.free(dir_path);
    std.fs.cwd().makeDir(dir_path) catch {};
    const db_path = try std.fmt.allocPrint(allocator, "{s}/data.db", .{dir_path});
    allocator.free(dir_path);

    const db_path_z = try allocator.dupeZ(u8, db_path);
    var conn = try db.Db.open(db_path_z);
    defer conn.close();
    defer allocator.free(db_path_z);

    try db.initSchema(&conn);
    try config.addCollection(&conn, "wiki", "/tmp");
    try store.insertDocument(&conn, "wiki", "a.md", "# Alpha\n\nhello world");

    return db_path;
}

fn cleanupMcpTestDb(db_path: []const u8) void {
    std.fs.cwd().deleteFile(db_path) catch {};
    const slash = std.mem.lastIndexOfScalar(u8, db_path, '/') orelse return;
    const dir_path = db_path[0..slash];
    std.fs.cwd().deleteDir(dir_path) catch {};
}

test "framed tools/call status returns result envelope" {
    const allocator = std.testing.allocator;
    const db_path = try setupMcpTestDb(allocator);
    defer cleanupMcpTestDb(db_path);
    defer allocator.free(db_path);

    const req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"status\"}}";
    const resp = try processFramedRequestForTest(req, db_path, allocator);
    defer allocator.free(resp);
    try std.testing.expect(std.mem.indexOf(u8, resp, "\"result\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "zmd status: OK") != null);
}

test "framed tools/call query returns ranked text" {
    const allocator = std.testing.allocator;
    const db_path = try setupMcpTestDb(allocator);
    defer cleanupMcpTestDb(db_path);
    defer allocator.free(db_path);

    const req = "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"query\",\"query\":\"hello\"}}";
    const resp = try processFramedRequestForTest(req, db_path, allocator);
    defer allocator.free(resp);
    try std.testing.expect(std.mem.indexOf(u8, resp, "found") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "qmd://") != null);
}

test "framed tools/call search returns fts text" {
    const allocator = std.testing.allocator;
    const db_path = try setupMcpTestDb(allocator);
    defer cleanupMcpTestDb(db_path);
    defer allocator.free(db_path);

    const req = "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"search\",\"query\":\"hello\",\"collection\":\"wiki\"}}";
    const resp = try processFramedRequestForTest(req, db_path, allocator);
    defer allocator.free(resp);
    try std.testing.expect(std.mem.indexOf(u8, resp, "found") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "fts results") != null);
}

test "framed tools/call get returns document content" {
    const allocator = std.testing.allocator;
    const db_path = try setupMcpTestDb(allocator);
    defer cleanupMcpTestDb(db_path);
    defer allocator.free(db_path);

    const req = "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"get\",\"path\":\"wiki/a.md\"}}";
    const resp = try processFramedRequestForTest(req, db_path, allocator);
    defer allocator.free(resp);
    try std.testing.expect(std.mem.indexOf(u8, resp, "Title: Alpha") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "hello world") != null);
}
