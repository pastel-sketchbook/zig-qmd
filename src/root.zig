const std = @import("std");

pub const db = @import("db.zig");
pub const store = @import("store.zig");
pub const chunker = @import("chunker.zig");
pub const search = @import("search.zig");
pub const config = @import("config.zig");
pub const llm = @import("llm.zig");
pub const mcp = @import("mcp.zig");
pub const ast = @import("ast.zig");

/// QMD library version, kept in sync with the VERSION file.
pub const version = "0.1.0";

/// Placeholder: will become the public SDK entry point.
/// Returns a greeting to verify the library links and runs.
pub fn hello() []const u8 {
    return "qmd " ++ version;
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

test "hello returns version string" {
    const result = hello();
    try std.testing.expectEqualStrings("qmd 0.1.0", result);
}

test {
    // Pull in tests from submodules
    _ = db;
}
