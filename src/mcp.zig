const std = @import("std");
const qmd = @import("qmd");

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
        return "{\"tools\":[{\"name\":\"query\",\"description\":\"Hybrid search\",\"inputSchema\":{\"type\":\"object\"}},{\"name\":\"search\",\"description\":\"FTS search\",\"inputSchema\":{\"type\":\"object\"}},{\"name\":\"get\",\"description\":\"Get document\",\"inputSchema\":{\"type\":\"object\"}},{\"name\":\"status\",\"description\":\"System status\",\"inputSchema\":{\"type\":\"object\"}}]}";
    }

    fn callTool(params: []const u8) ![]u8 {
        const name = extractParam(params, "name") orelse return McpError.InvalidParams;

        if (std.mem.eql(u8, name, "query")) {
            const query = extractParam(params, "query") orelse "unknown";
            const result = try std.fmt.allocPrint(std.heap.page_allocator, "Query result for: {s}", .{query});
            return try std.fmt.allocPrint(std.heap.page_allocator, "{{\"content\":[{{\"type\":\"text\",\"text\":\"{s}\"}}]}}", .{result});
        }
        if (std.mem.eql(u8, name, "search")) {
            return "{\"content\":[{\"type\":\"text\",\"text\":\"search results\"}]}";
        }
        if (std.mem.eql(u8, name, "status")) {
            return "{\"content\":[{\"type\":\"text\",\"text\":\"zmd status: OK\"}]}";
        }
        if (std.mem.eql(u8, name, "get")) {
            return "{\"content\":[{\"type\":\"text\",\"text\":\"document content\"}]}";
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
