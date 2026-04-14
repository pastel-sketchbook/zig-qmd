const std = @import("std");

pub const CHUNK_SIZE_CHARS = 3600;
pub const CHUNK_OVERLAP_CHARS = 540;
pub const CHUNK_WINDOW_CHARS = 800;

const BREAK_PATTERNS = [_][:0]const u8{
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n##### ",
    "\n###### ",
    "\n# ",
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n##### ",
};

const CodeFence = struct {
    start: usize,
    end: usize,
    lang: []const u8,
};

pub fn findCodeFences(content: []const u8) !struct { fences: std.ArrayList(CodeFence) } {
    var fences = try std.ArrayList(CodeFence).initCapacity(std.heap.page_allocator, 0);
    var i: usize = 0;
    var in_fence = false;
    var fence_start: usize = 0;
    var lang_start: usize = 0;

    while (i < content.len) {
        if (i + 3 <= content.len and std.mem.eql(u8, content[i .. i + 3], "```")) {
            if (!in_fence) {
                fence_start = i;
                lang_start = i + 3;
                while (lang_start < content.len and (content[lang_start] == ' ' or content[lang_start] == '\t')) lang_start += 1;
                var lang_end = lang_start;
                while (lang_end < content.len and content[lang_end] != '\n' and content[lang_end] != '`') lang_end += 1;
                in_fence = true;
                i = lang_end;
                continue;
            } else {
                var end = i;
                while (end < content.len and content[end] != '\n') end += 1;
                try fences.append(std.heap.page_allocator, .{ .start = fence_start, .end = end, .lang = content[lang_start..i] });
                in_fence = false;
                i += 3;
                continue;
            }
        }
        i += 1;
    }

    return .{ .fences = fences };
}

pub fn findBestCutoff(content: []const u8, window_start: usize, window_end: usize) usize {
    if (window_end <= window_start) return window_start;
    if (window_end - window_start < 20) return window_end;

    var best_pos = window_start;
    var best_score: f64 = -1;

    var pos = window_start;
    while (pos < window_end and pos < content.len) : (pos += 1) {
        if (content[pos] == '\n') {
            var line_start = pos;
            while (line_start > window_start and content[line_start - 1] != '\n') line_start -= 1;
            const line_len = pos - line_start;
            if (line_len < 3 or line_len > 200) continue;

            var is_heading = false;
            if (line_start + 1 < content.len) {
                const c = content[line_start + 1];
                if (c == '#') is_heading = true;
            }

            const dist_from_mid = @abs(@as(f64, @floatFromInt(pos - window_start)) - @as(f64, @floatFromInt(window_end - window_start)) / 2.0);
            var score: f64 = 1.0 / (1.0 + dist_from_mid * dist_from_mid / 100.0);
            if (is_heading) score *= 3.0;

            if (score > best_score) {
                best_score = score;
                best_pos = pos;
            }
        }
    }

    return best_pos;
}

pub fn chunkDocument(content: []const u8) !struct { chunks: std.ArrayList([]const u8) } {
    var chunks = try std.ArrayList([]const u8).initCapacity(std.heap.page_allocator, 0);

    if (content.len <= CHUNK_SIZE_CHARS) {
        try chunks.append(std.heap.page_allocator, content);
        return .{ .chunks = chunks };
    }

    var pos: usize = 0;
    while (pos < content.len) {
        const chunk_end = pos + CHUNK_SIZE_CHARS;
        if (chunk_end >= content.len) {
            try chunks.append(std.heap.page_allocator, content[pos..content.len]);
            break;
        }

        const window_start = if (pos == 0) pos else pos + CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS;
        var window_end = chunk_end;
        if (window_end > content.len) window_end = content.len;

        var cutoff = findBestCutoff(content, window_start, window_end);
        if (cutoff <= pos) cutoff = chunk_end;

        try chunks.append(std.heap.page_allocator, content[pos..cutoff]);
        pos = cutoff;
    }

    return .{ .chunks = chunks };
}

test "findCodeFences finds fenced blocks" {
    const content = "# Hello\n```python\ndef hello():\n    pass\n```\n## Next";

    const result = try findCodeFences(content);
    defer result.fences.deinit();
    try std.testing.expectEqual(@as(usize, 1), result.fences.items.len);
    try std.testing.expectEqualStrings("python", result.fences.items[0].lang);
}

test "findBestCutoff prefers heading" {
    const content = "prefix\n## Heading\ncontent here\nmore content\n";
    const cutoff = findBestCutoff(content, 0, content.len);
    try std.testing.expect(cutoff > 0);
}

test "chunkDocument splits long content" {
    const content = "# Title\n" ++ "x" ** 4000;
    const result = try chunkDocument(content);
    defer result.chunks.deinit();
    try std.testing.expect(result.chunks.items.len > 1);
}

test "chunkDocument keeps short content whole" {
    const content = "# Short\n\nJust a brief note.";
    const result = try chunkDocument(content);
    defer result.chunks.deinit();
    try std.testing.expectEqual(@as(usize, 1), result.chunks.items.len);
}

test "chunkDocument overlaps chunks" {
    const content = "a" ** 3500 ++ "\n## Section\n" ++ "b" ** 3500;
    const result = try chunkDocument(content);
    defer result.chunks.deinit();
    if (result.chunks.items.len >= 2) {
        try std.testing.expect(result.chunks.items[1].len > CHUNK_OVERLAP_CHARS);
    }
}
