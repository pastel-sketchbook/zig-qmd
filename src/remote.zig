const std = @import("std");
const Io = std.Io;

/// Returns true if the path looks like a remote git URL.
/// Recognizes https:// URLs (GitHub, GitLab, etc.) and git@ SSH URLs.
pub fn isRemoteUrl(path: []const u8) bool {
    if (std.mem.startsWith(u8, path, "https://")) return true;
    if (std.mem.startsWith(u8, path, "http://")) return true;
    if (std.mem.startsWith(u8, path, "git@")) return true;
    if (std.mem.startsWith(u8, path, "ssh://")) return true;
    return false;
}

/// Derives a deterministic cache directory name from a remote URL.
/// Returns a hex-encoded SHA-256 hash of the URL, suitable for use as
/// a directory name under `.qmd/repos/`.
pub fn cacheKey(url: []const u8) [64]u8 {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(url, &hash, .{});
    return std.fmt.bytesToHex(hash, .lower);
}

/// Builds the full local cache path for a remote collection.
/// Returns `.qmd/repos/<sha256_hex>` as a heap-allocated string.
pub fn cachePath(allocator: std.mem.Allocator, url: []const u8) ![]u8 {
    const key = cacheKey(url);
    return std.fmt.allocPrint(allocator, ".qmd/repos/{s}", .{key[0..]});
}

/// Resolves the local filesystem path for a collection.
/// For remote URLs, returns the cache path under `.qmd/repos/`.
/// For local paths, returns a duplicate of the path as-is.
pub fn resolveCollectionPath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    if (isRemoteUrl(path)) {
        return cachePath(allocator, path);
    }
    return allocator.dupe(u8, path);
}

/// Error set for remote git operations.
pub const RemoteError = error{
    GitNotFound,
    CloneFailed,
    PullFailed,
} || std.mem.Allocator.Error;

/// Clones or updates a remote repository to the local cache directory.
/// Uses `git clone --depth=1` for initial clone, `git pull` for updates.
/// Returns the local cache path.
pub fn syncRemote(allocator: std.mem.Allocator, io: Io, url: []const u8) RemoteError![]u8 {
    const local_path = try cachePath(allocator, url);
    errdefer allocator.free(local_path);

    // Ensure parent directory exists
    std.Io.Dir.cwd().createDirPath(io, ".qmd/repos") catch {};

    // Check if already cloned
    var dir_exists = true;
    std.Io.Dir.cwd().access(io, local_path, .{}) catch {
        dir_exists = false;
    };

    if (dir_exists) {
        // Pull updates
        const result = std.process.run(allocator, io, .{
            .argv = &.{ "git", "-C", local_path, "pull", "--ff-only" },
        }) catch return RemoteError.PullFailed;
        allocator.free(result.stdout);
        allocator.free(result.stderr);
        if (result.term != .exited or result.term.exited != 0) return RemoteError.PullFailed;
    } else {
        // Fresh shallow clone
        const result = std.process.run(allocator, io, .{
            .argv = &.{ "git", "clone", "--depth=1", url, local_path },
        }) catch return RemoteError.CloneFailed;
        allocator.free(result.stdout);
        allocator.free(result.stderr);
        if (result.term != .exited or result.term.exited != 0) return RemoteError.CloneFailed;
    }

    return local_path;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "isRemoteUrl detects https URLs" {
    try std.testing.expect(isRemoteUrl("https://github.com/legalize-kr/legalize-kr"));
    try std.testing.expect(isRemoteUrl("https://github.com/user/repo.git"));
}

test "isRemoteUrl detects git SSH URLs" {
    try std.testing.expect(isRemoteUrl("git@github.com:user/repo.git"));
}

test "isRemoteUrl detects ssh:// URLs" {
    try std.testing.expect(isRemoteUrl("ssh://git@github.com/user/repo.git"));
}

test "isRemoteUrl rejects local paths" {
    try std.testing.expect(!isRemoteUrl("/home/user/notes"));
    try std.testing.expect(!isRemoteUrl("./relative/path"));
    try std.testing.expect(!isRemoteUrl("~/Documents/notes"));
    try std.testing.expect(!isRemoteUrl("notes"));
}

test "cacheKey is deterministic" {
    const key1 = cacheKey("https://github.com/legalize-kr/legalize-kr");
    const key2 = cacheKey("https://github.com/legalize-kr/legalize-kr");
    try std.testing.expectEqualStrings(&key1, &key2);
}

test "cacheKey differs for different URLs" {
    const key1 = cacheKey("https://github.com/legalize-kr/legalize-kr");
    const key2 = cacheKey("https://github.com/legalize-kr/precedent-kr");
    try std.testing.expect(!std.mem.eql(u8, &key1, &key2));
}

test "cachePath includes .qmd/repos/ prefix" {
    const path = try cachePath(std.testing.allocator, "https://github.com/user/repo");
    defer std.testing.allocator.free(path);
    try std.testing.expect(std.mem.startsWith(u8, path, ".qmd/repos/"));
    try std.testing.expect(path.len == ".qmd/repos/".len + 64); // 64 hex chars
}

test "resolveCollectionPath returns cache path for remote URLs" {
    const path = try resolveCollectionPath(std.testing.allocator, "https://github.com/user/repo");
    defer std.testing.allocator.free(path);
    try std.testing.expect(std.mem.startsWith(u8, path, ".qmd/repos/"));
}

test "resolveCollectionPath returns local path as-is" {
    const path = try resolveCollectionPath(std.testing.allocator, "/home/user/notes");
    defer std.testing.allocator.free(path);
    try std.testing.expectEqualStrings("/home/user/notes", path);
}
