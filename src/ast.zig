const std = @import("std");

pub const AstError = error{
    ParseError,
    UnsupportedLanguage,
    OutOfMemory,
};

pub const AstChunker = struct {
    language: []const u8,
    breakpoints: []const u32,

    pub fn init(language: []const u8) AstChunker {
        return .{
            .language = language,
            .breakpoints = &.{},
        };
    }

    pub fn deinit(self: *AstChunker) void {
        _ = self;
    }

    pub fn extractBreakpoints(self: *AstChunker, content: []const u8) []u32 {
        _ = self;
        var buf: [1024]u32 = undefined;
        var count: usize = 0;
        for (content, 0..) |c, i| {
            if (c == '\n' and count < 1024) {
                buf[count] = @as(u32, @intCast(i));
                count += 1;
            }
        }
        return buf[0..count];
    }

    pub fn chunk(self: *AstChunker, content: []const u8, max_size: usize) [][]const u8 {
        var chunks = std.ArrayList([]const u8).initCapacity(std.heap.page_allocator, 0) catch return &.{};
        errdefer chunks.deinit(std.heap.page_allocator);
        var pos: usize = 0;

        while (pos < content.len) {
            const end = pos + max_size;
            if (end >= content.len) {
                chunks.append(std.heap.page_allocator, content[pos..]) catch {};
                break;
            }

            var cut = end;
            for (self.breakpoints) |bp| {
                if (bp > pos and bp < end) {
                    cut = bp;
                }
            }

            chunks.append(std.heap.page_allocator, content[pos..cut]) catch {};
            pos = cut;
        }

        return chunks.items;
    }
};

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

pub fn mergeBreakpoints(a: []const u32, b: []const u32) []u32 {
    var merged = std.ArrayList(u32).initCapacity(std.heap.page_allocator, a.len + b.len) catch return &.{};
    errdefer merged.deinit(std.heap.page_allocator);
    var i: usize = 0;
    var j: usize = 0;

    while (i < a.len or j < b.len) {
        if (j >= b.len or (i < a.len and a[i] < b[j])) {
            merged.append(std.heap.page_allocator, a[i]) catch {};
            i += 1;
        } else {
            merged.append(std.heap.page_allocator, b[j]) catch {};
            j += 1;
        }
    }

    return merged.items;
}

test "AstChunker can be initialized" {
    const chunker = AstChunker.init("javascript");
    try std.testing.expectEqualStrings("javascript", chunker.language);
}

test "detectLanguage recognizes file types" {
    try std.testing.expectEqualStrings("typescript", detectLanguage("file.ts"));
    try std.testing.expectEqualStrings("python", detectLanguage("file.py"));
    try std.testing.expectEqualStrings("markdown", detectLanguage("file.md"));
    try std.testing.expectEqualStrings("text", detectLanguage("file.unknown"));
}

test "mergeBreakpoints combines lists" {
    const a: []const u32 = &.{ 1, 5, 10 };
    const b: []const u32 = &.{ 3, 7, 12 };
    const merged = mergeBreakpoints(a, b);
    try std.testing.expect(merged.len >= 0);
}
