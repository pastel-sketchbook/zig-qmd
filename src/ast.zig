const std = @import("std");
const c = @cImport({
    @cInclude("tree_sitter/api.h");
});

extern fn tree_sitter_markdown() ?*const c.TSLanguage;

pub const AstError = error{
    ParseError,
    UnsupportedLanguage,
    OutOfMemory,
};

pub const BreakpointKind = enum {
    heading,
    fence,
    list_item,
    paragraph,
};

pub const Breakpoint = struct {
    offset: usize,
    kind: BreakpointKind,
};

pub const AstChunker = struct {
    language: []const u8,
    allocator: std.mem.Allocator,
    breakpoints: std.ArrayList(Breakpoint),

    pub fn init(allocator: std.mem.Allocator, language: []const u8) AstChunker {
        return .{
            .language = language,
            .allocator = allocator,
            .breakpoints = std.ArrayList(Breakpoint).initCapacity(allocator, 0) catch unreachable,
        };
    }

    pub fn deinit(self: *AstChunker) void {
        self.breakpoints.deinit(self.allocator);
    }

    pub fn extractBreakpoints(self: *AstChunker, content: []const u8) ![]Breakpoint {
        self.breakpoints.clearRetainingCapacity();

        if (std.mem.eql(u8, self.language, "markdown")) {
            try extract_with_tree_sitter(self, content);
            if (self.breakpoints.items.len > 0) {
                std.sort.heap(Breakpoint, self.breakpoints.items, {}, struct {
                    fn less(_: void, a: Breakpoint, b: Breakpoint) bool {
                        return a.offset < b.offset;
                    }
                }.less);
                return self.breakpoints.items;
            }
        }

        var line_start: usize = 0;
        var in_fence = false;
        while (line_start < content.len) {
            const next_nl = std.mem.indexOfScalarPos(u8, content, line_start, '\n') orelse content.len;
            const line = content[line_start..next_nl];
            const trimmed = std.mem.trim(u8, line, &.{ ' ', '\t', '\r' });

            if (trimmed.len >= 3 and std.mem.startsWith(u8, trimmed, "```")) {
                in_fence = !in_fence;
                try self.breakpoints.append(self.allocator, .{ .offset = line_start, .kind = .fence });
            } else if (!in_fence and is_heading(trimmed)) {
                try self.breakpoints.append(self.allocator, .{ .offset = line_start, .kind = .heading });
            } else if (!in_fence and is_list_item(trimmed)) {
                try self.breakpoints.append(self.allocator, .{ .offset = line_start, .kind = .list_item });
            } else if (!in_fence and trimmed.len == 0) {
                const para_start = next_nl + 1;
                if (para_start < content.len) {
                    try self.breakpoints.append(self.allocator, .{ .offset = para_start, .kind = .paragraph });
                }
            }

            if (next_nl >= content.len) break;
            line_start = next_nl + 1;
        }

        std.sort.heap(Breakpoint, self.breakpoints.items, {}, struct {
            fn less(_: void, a: Breakpoint, b: Breakpoint) bool {
                return a.offset < b.offset;
            }
        }.less);

        return self.breakpoints.items;
    }

    pub fn chunk(self: *AstChunker, content: []const u8, max_size: usize) !std.ArrayList([]const u8) {
        _ = try self.extractBreakpoints(content);

        var chunks = try std.ArrayList([]const u8).initCapacity(self.allocator, 8);
        errdefer chunks.deinit(self.allocator);

        if (content.len == 0) return chunks;

        var pos: usize = 0;
        while (pos < content.len) {
            const target_end = @min(content.len, pos + max_size);
            if (target_end == content.len) {
                try chunks.append(self.allocator, content[pos..content.len]);
                break;
            }

            var cut = find_last_breakpoint_before(self.breakpoints.items, pos, target_end) orelse target_end;
            if (cut <= pos) cut = target_end;

            try chunks.append(self.allocator, content[pos..cut]);
            pos = cut;
        }

        return chunks;
    }
};

fn is_heading(line: []const u8) bool {
    if (line.len < 2) return false;
    var i: usize = 0;
    while (i < line.len and line[i] == '#') : (i += 1) {}
    return i > 0 and i < line.len and line[i] == ' ';
}

fn extract_with_tree_sitter(self: *AstChunker, content: []const u8) !void {
    const parser = c.ts_parser_new() orelse return;
    defer c.ts_parser_delete(parser);

    const lang = tree_sitter_markdown() orelse return;
    if (!c.ts_parser_set_language(parser, lang)) return;

    const tree = c.ts_parser_parse_string(parser, null, content.ptr, @intCast(content.len)) orelse return;
    defer c.ts_tree_delete(tree);

    const root_node = c.ts_tree_root_node(tree);
    try walk_node(self, root_node);
}

fn walk_node(self: *AstChunker, node: c.TSNode) !void {
    if (c.ts_node_is_null(node)) return;

    const kind = std.mem.span(c.ts_node_type(node));
    if (std.mem.eql(u8, kind, "atx_heading") or
        std.mem.eql(u8, kind, "setext_heading") or
        std.mem.eql(u8, kind, "fenced_code_block") or
        std.mem.eql(u8, kind, "list_item"))
    {
        try self.breakpoints.append(self.allocator, .{
            .offset = @intCast(c.ts_node_start_byte(node)),
            .kind = if (std.mem.eql(u8, kind, "fenced_code_block")) .fence else if (std.mem.eql(u8, kind, "list_item")) .list_item else .heading,
        });
    }

    const child_count = c.ts_node_child_count(node);
    var i: u32 = 0;
    while (i < child_count) : (i += 1) {
        try walk_node(self, c.ts_node_child(node, i));
    }
}

fn is_list_item(line: []const u8) bool {
    if (line.len >= 2 and (line[0] == '-' or line[0] == '*' or line[0] == '+') and line[1] == ' ') return true;
    if (line.len >= 3 and std.ascii.isDigit(line[0]) and line[1] == '.' and line[2] == ' ') return true;
    return false;
}

fn find_last_breakpoint_before(bps: []const Breakpoint, min_offset: usize, max_offset: usize) ?usize {
    var best: ?usize = null;
    for (bps) |bp| {
        if (bp.offset > min_offset and bp.offset <= max_offset) {
            best = bp.offset;
        }
    }
    return best;
}

pub fn detectLanguage(filename: []const u8) []const u8 {
    if (std.mem.endsWith(u8, filename, ".ts")) return "typescript";
    if (std.mem.endsWith(u8, filename, ".js")) return "javascript";
    if (std.mem.endsWith(u8, filename, ".py")) return "python";
    if (std.mem.endsWith(u8, filename, ".go")) return "go";
    if (std.mem.endsWith(u8, filename, ".rs")) return "rust";
    if (std.mem.endsWith(u8, filename, ".zig")) return "zig";
    if (std.mem.endsWith(u8, filename, ".md")) return "markdown";
    return "text";
}

test "AstChunker extracts heading and fence breakpoints" {
    var chunker = AstChunker.init(std.testing.allocator, "markdown");
    defer chunker.deinit();

    const content = "# Title\n\ntext\n```zig\nconst a = 1;\n```\n## Next\n";
    const bps = try chunker.extractBreakpoints(content);
    try std.testing.expect(bps.len >= 3);
}

test "AstChunker chunk splits by semantic breakpoints" {
    var chunker = AstChunker.init(std.testing.allocator, "markdown");
    defer chunker.deinit();

    const content =
        "# H1\n\nparagraph 1\n\n## H2\n\nparagraph 2\n\n## H3\n\nparagraph 3\n";
    var chunks = try chunker.chunk(content, 24);
    defer chunks.deinit(std.testing.allocator);

    try std.testing.expect(chunks.items.len >= 2);
    try std.testing.expect(std.mem.indexOf(u8, chunks.items[0], "# H1") != null);
}

test "detectLanguage recognizes file types" {
    try std.testing.expectEqualStrings("typescript", detectLanguage("file.ts"));
    try std.testing.expectEqualStrings("python", detectLanguage("file.py"));
    try std.testing.expectEqualStrings("markdown", detectLanguage("file.md"));
    try std.testing.expectEqualStrings("text", detectLanguage("file.unknown"));
}
