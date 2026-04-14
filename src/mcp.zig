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

const ParsedRequest = struct {
    id_json: []u8,
    method: []u8,
    params_json: ?[]u8,

    fn deinit(self: *ParsedRequest, allocator: std.mem.Allocator) void {
        allocator.free(self.id_json);
        allocator.free(self.method);
        if (self.params_json) |params| allocator.free(params);
    }
};

const ParsedToolCall = struct {
    name: []u8,
    query: ?[]u8,
    collection: ?[]u8,
    path: ?[]u8,

    fn deinit(self: *ParsedToolCall, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        if (self.query) |value| allocator.free(value);
        if (self.collection) |value| allocator.free(value);
        if (self.path) |value| allocator.free(value);
    }
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

            const response = handleRequestWithDbPath(msg, null, allocator) catch |err| {
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

    fn handleRequest(request: []const u8, allocator: std.mem.Allocator) ![]u8 {
        return handleRequestWithDbPath(request, null, allocator);
    }

    fn handleRequestWithDbPath(request: []const u8, db_path_override: ?[]const u8, allocator: std.mem.Allocator) ![]u8 {
        var parsed = try parseRequest(request, allocator);
        defer parsed.deinit(allocator);

        if (std.mem.eql(u8, parsed.method, "tools/list")) {
            return try formatResponse(parsed.id_json, listTools(), allocator);
        }
        if (std.mem.eql(u8, parsed.method, "tools/call")) {
            const params = parsed.params_json orelse return McpError.InvalidParams;
            var tool_call = try parseToolCall(params, allocator);
            defer tool_call.deinit(allocator);
            return try formatResponse(parsed.id_json, try callToolNamedAtPath(&tool_call, db_path_override orelse DB_PATH, allocator), allocator);
        }
        if (std.mem.eql(u8, parsed.method, "initialize")) {
            return try formatResponse(parsed.id_json, getServerInfo(), allocator);
        }
        if (std.mem.eql(u8, parsed.method, "ping")) {
            return try formatResponse(parsed.id_json, "pong", allocator);
        }
        return try formatError(parsed.id_json, "method not found", allocator);
    }

    fn parseRequest(request: []const u8, allocator: std.mem.Allocator) !ParsedRequest {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, request, .{}) catch return McpError.ParseError;
        defer parsed.deinit();

        if (parsed.value != .object) return McpError.ParseError;
        const obj = parsed.value.object;

        const version_value = obj.get("jsonrpc") orelse return McpError.ParseError;
        if (version_value != .string or !std.mem.eql(u8, version_value.string, "2.0")) {
            return McpError.ParseError;
        }

        const id_value = obj.get("id") orelse return McpError.ParseError;
        const method_value = obj.get("method") orelse return McpError.ParseError;
        if (method_value != .string) return McpError.ParseError;

        const id_json = try stringifyJsonValue(allocator, id_value);
        errdefer allocator.free(id_json);
        const method = try allocator.dupe(u8, method_value.string);
        errdefer allocator.free(method);

        const params_json = if (obj.get("params")) |params_value|
            try stringifyJsonValue(allocator, params_value)
        else
            null;
        errdefer if (params_json) |params| allocator.free(params);

        return .{ .id_json = id_json, .method = method, .params_json = params_json };
    }

    fn stringifyJsonValue(allocator: std.mem.Allocator, value: std.json.Value) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(allocator);
        defer out.deinit();
        try std.json.Stringify.value(value, .{}, &out.writer);
        return try allocator.dupe(u8, out.written());
    }

    fn listTools() []const u8 {
        return "{\"tools\":[{\"name\":\"query\",\"description\":\"Hybrid search\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"arguments\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}}}}}},{\"name\":\"search\",\"description\":\"FTS search\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"arguments\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},\"collection\":{\"type\":\"string\"}}}}}},{\"name\":\"get\",\"description\":\"Get document\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"arguments\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}}}}}},{\"name\":\"status\",\"description\":\"System status\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"arguments\":{\"type\":\"object\"}}}}]}";
    }

    fn callTool(params: []const u8, allocator: std.mem.Allocator) ![]u8 {
        var tool_call = try parseToolCall(params, allocator);
        defer tool_call.deinit(allocator);
        return callToolNamedAtPath(&tool_call, DB_PATH, allocator);
    }

    fn callToolNamedAtPath(tool_call: *const ParsedToolCall, db_path_raw: []const u8, allocator: std.mem.Allocator) ![]u8 {
        const db_path = allocator.dupeZ(u8, db_path_raw) catch return McpError.InvalidParams;
        defer allocator.free(db_path);

        var db_ = db.Db.open(db_path) catch {
            return allocator.dupe(u8, "{\"content\":[{\"type\":\"text\",\"text\":\"database not initialized\"}]}") catch McpError.InvalidParams;
        };
        defer db_.close();

        if (std.mem.eql(u8, tool_call.name, "query")) {
            const query_text = tool_call.query orelse "";
            var result = search.hybridSearch(&db_, allocator, query_text, null, .{
                .enable_vector = true,
                .max_results = 5,
            }) catch return allocator.dupe(u8, "{\"content\":[{\"type\":\"text\",\"text\":\"query failed\"}]}") catch McpError.InvalidParams;
            defer result.deinit(allocator);

            var text = std.ArrayList(u8).initCapacity(allocator, 256) catch return McpError.InvalidParams;
            defer text.deinit(allocator);
            try text.writer(allocator).print("found {d} hybrid results", .{result.results.items.len});
            for (result.results.items, 0..) |r, i| {
                try text.writer(allocator).print("\n{d}. {s} (zmd://{s}/{s}) score={d:.4}", .{ i + 1, r.title, r.collection, r.path, r.score });
            }
            return std.fmt.allocPrint(allocator, "{{\"content\":[{{\"type\":\"text\",\"text\":\"{s}\"}}]}}", .{text.items}) catch McpError.InvalidParams;
        }
        if (std.mem.eql(u8, tool_call.name, "search")) {
            const query_text = tool_call.query orelse "";
            const collection = tool_call.collection;
            var result = search.searchFTS(&db_, allocator, query_text, collection) catch return allocator.dupe(u8, "{\"content\":[{\"type\":\"text\",\"text\":\"search failed\"}]}") catch McpError.InvalidParams;
            defer result.deinit(allocator);

            var text = std.ArrayList(u8).initCapacity(allocator, 256) catch return McpError.InvalidParams;
            defer text.deinit(allocator);
            try text.writer(allocator).print("found {d} fts results", .{result.results.items.len});
            for (result.results.items, 0..) |r, i| {
                try text.writer(allocator).print("\n{d}. {s} (zmd://{s}/{s}) score={d:.4}", .{ i + 1, r.title, r.collection, r.path, r.score });
            }
            return std.fmt.allocPrint(allocator, "{{\"content\":[{{\"type\":\"text\",\"text\":\"{s}\"}}]}}", .{text.items}) catch McpError.InvalidParams;
        }
        if (std.mem.eql(u8, tool_call.name, "status")) {
            var stmt = db_.prepare("SELECT count(*) FROM documents WHERE active = 1") catch return allocator.dupe(u8, "{\"content\":[{\"type\":\"text\",\"text\":\"status failed\"}]}") catch McpError.InvalidParams;
            defer stmt.finalize();
            const has_docs = stmt.step() catch false;
            const docs: i64 = if (has_docs) stmt.columnInt(0) else 0;
            stmt = db_.prepare("SELECT count(*) FROM store_collections") catch return allocator.dupe(u8, "{\"content\":[{\"type\":\"text\",\"text\":\"status failed\"}]}") catch McpError.InvalidParams;
            defer stmt.finalize();
            const has_cols = stmt.step() catch false;
            const cols: i64 = if (has_cols) stmt.columnInt(0) else 0;
            const text = std.fmt.allocPrint(allocator, "zmd status: OK, documents={d}, collections={d}", .{ docs, cols }) catch return McpError.InvalidParams;
            defer allocator.free(text);
            return std.fmt.allocPrint(allocator, "{{\"content\":[{{\"type\":\"text\",\"text\":\"{s}\"}}]}}", .{text}) catch McpError.InvalidParams;
        }
        if (std.mem.eql(u8, tool_call.name, "get")) {
            const raw_path = tool_call.path orelse return McpError.InvalidParams;
            const parsed = root.parse_virtual_path(raw_path) orelse return McpError.InvalidParams;
            const doc = store.findActiveDocument(&db_, parsed.collection, parsed.path) catch {
                return allocator.dupe(u8, "{\"content\":[{\"type\":\"text\",\"text\":\"document not found\"}]}") catch McpError.InvalidParams;
            };
            defer {
                allocator.free(doc.title);
                allocator.free(doc.hash);
                allocator.free(doc.doc);
            }
            var text = std.ArrayList(u8).initCapacity(allocator, doc.doc.len + 64) catch return McpError.InvalidParams;
            defer text.deinit(allocator);
            try text.writer(allocator).print("Title: {s}\n\n{s}", .{ doc.title, doc.doc });
            return std.fmt.allocPrint(allocator, "{{\"content\":[{{\"type\":\"text\",\"text\":\"{s}\"}}]}}", .{text.items}) catch McpError.InvalidParams;
        }

        return McpError.MethodNotFound;
    }

    fn parseToolCall(json: []const u8, allocator: std.mem.Allocator) !ParsedToolCall {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{}) catch return McpError.InvalidParams;
        defer parsed.deinit();

        if (parsed.value != .object) return McpError.InvalidParams;
        const params_obj = parsed.value.object;

        const name_value = params_obj.get("name") orelse return McpError.InvalidParams;
        if (name_value != .string) return McpError.InvalidParams;

        const args_value = if (params_obj.get("arguments")) |arguments_value|
            arguments_value
        else
            parsed.value;
        if (args_value != .object) return McpError.InvalidParams;
        const args_obj = args_value.object;

        const name = try allocator.dupe(u8, name_value.string);
        errdefer allocator.free(name);
        const query = try dupOptionalString(args_obj.get("query"), allocator);
        errdefer if (query) |value| allocator.free(value);
        const collection = try dupOptionalString(args_obj.get("collection"), allocator);
        errdefer if (collection) |value| allocator.free(value);
        const path = try dupOptionalString(args_obj.get("path"), allocator);
        errdefer if (path) |value| allocator.free(value);

        return .{ .name = name, .query = query, .collection = collection, .path = path };
    }

    fn dupOptionalString(value: ?std.json.Value, allocator: std.mem.Allocator) !?[]u8 {
        const actual = value orelse return null;
        if (actual != .string) return McpError.InvalidParams;
        return try allocator.dupe(u8, actual.string);
    }

    fn getServerInfo() []const u8 {
        return "{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{\"tools\":{\"listChanged\":false}},\"serverInfo\":{\"name\":\"zmd\",\"version\":\"0.1.0\"}}";
    }

    fn formatResponse(id: []const u8, result: []const u8, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator, "{{\"jsonrpc\":\"2.0\",\"id\":{s},\"result\":{s}}}", .{ id, result });
    }

    fn formatError(id: []const u8, message: []const u8, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator, "{{\"jsonrpc\":\"2.0\",\"id\":{s},\"error\":{{\"code\":-32601,\"message\":\"{s}\"}}}}", .{ id, message });
    }
};

test "McpServer struct can be defined" {
    const t: type = McpServer;
    _ = t;
}

test "parseRequest parses method name" {
    const request = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}";
    var parsed = try McpServer.parseRequest(request, std.testing.allocator);
    defer parsed.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("tools/list", parsed.method);
}

test "parseRequest serializes numeric id" {
    const request = "{\"jsonrpc\":\"2.0\",\"id\":42,\"method\":\"ping\"}";
    var parsed = try McpServer.parseRequest(request, std.testing.allocator);
    defer parsed.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("42", parsed.id_json);
}

test "parseToolCall parses direct params" {
    const json = "{\"name\":\"query\",\"query\":\"test\"}";
    var parsed = try McpServer.parseToolCall(json, std.testing.allocator);
    defer parsed.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("query", parsed.name);
    try std.testing.expectEqualStrings("test", parsed.query.?);
}

test "parseToolCall parses MCP arguments object" {
    const json = "{\"name\":\"search\",\"arguments\":{\"query\":\"hello\",\"collection\":\"wiki\"}}";
    var parsed = try McpServer.parseToolCall(json, std.testing.allocator);
    defer parsed.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("search", parsed.name);
    try std.testing.expectEqualStrings("hello", parsed.query.?);
    try std.testing.expectEqualStrings("wiki", parsed.collection.?);
}

test "handleRequest supports ping" {
    const req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}";
    const allocator = std.testing.allocator;
    const resp = try McpServer.handleRequest(req, allocator);
    defer allocator.free(resp);
    try std.testing.expect(std.mem.indexOf(u8, resp, "pong") != null);
}

test "handleRequest rejects missing id" {
    const req = "{\"jsonrpc\":\"2.0\",\"method\":\"ping\"}";
    try std.testing.expectError(McpError.ParseError, McpServer.handleRequest(req, std.testing.allocator));
}

test "handleRequest rejects non-2.0 jsonrpc" {
    const req = "{\"jsonrpc\":\"1.0\",\"id\":1,\"method\":\"ping\"}";
    try std.testing.expectError(McpError.ParseError, McpServer.handleRequest(req, std.testing.allocator));
}

test "handleRequest rejects malformed params in tools/call" {
    const req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":\"oops\"}";
    try std.testing.expectError(McpError.InvalidParams, McpServer.handleRequest(req, std.testing.allocator));
}

test "handleRequest returns method error for unknown tool" {
    const allocator = std.testing.allocator;
    const db_path = try setupMcpTestDb(allocator);
    defer cleanupMcpTestDb(db_path);
    defer allocator.free(db_path);

    const req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"unknown\"}}";
    try std.testing.expectError(McpError.MethodNotFound, McpServer.handleRequestWithDbPath(req, db_path, allocator));
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
    var buf = try std.ArrayList(u8).initCapacity(std.testing.allocator, 0);
    defer buf.deinit(std.testing.allocator);

    var writer = FakeWriter{ .buf = &buf, .allocator = std.testing.allocator };
    try McpServer.writeMessage(&writer, "{}");

    try std.testing.expect(std.mem.startsWith(u8, buf.items, "Content-Length: 2\r\n\r\n{}"));
}

test "writeMessage and readMessage roundtrip" {
    var out = try std.ArrayList(u8).initCapacity(std.testing.allocator, 0);
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
    var inbound = try std.ArrayList(u8).initCapacity(allocator, 0);
    defer inbound.deinit(allocator);
    var in_writer = FakeWriter{ .buf = &inbound, .allocator = allocator };
    try McpServer.writeMessage(&in_writer, request_body);

    var reader = FakeReader{ .data = inbound.items };
    const parsed = try McpServer.readMessage(&reader, allocator);
    defer allocator.free(parsed);

    const response = try McpServer.handleRequestWithDbPath(parsed, db_path_override, allocator);
    defer allocator.free(response);

    var outbound = try std.ArrayList(u8).initCapacity(allocator, 0);
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
    std.fs.cwd().makeDir(dir_path) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };
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
    std.fs.cwd().deleteFile(db_path) catch |err| {
        if (err != error.FileNotFound) return;
    };
    const slash = std.mem.lastIndexOfScalar(u8, db_path, '/') orelse return;
    const dir_path = db_path[0..slash];
    std.fs.cwd().deleteDir(dir_path) catch |err| {
        if (err != error.FileNotFound) return;
    };
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
    try std.testing.expect(std.mem.indexOf(u8, resp, "zmd://") != null);
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

test "framed tools/call supports MCP arguments object" {
    const allocator = std.testing.allocator;
    const db_path = try setupMcpTestDb(allocator);
    defer cleanupMcpTestDb(db_path);
    defer allocator.free(db_path);

    const req = "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/call\",\"params\":{\"name\":\"search\",\"arguments\":{\"query\":\"hello\",\"collection\":\"wiki\"}}}";
    const resp = try processFramedRequestForTest(req, db_path, allocator);
    defer allocator.free(resp);
    try std.testing.expect(std.mem.indexOf(u8, resp, "fts results") != null);
}
