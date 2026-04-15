const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- SQLite C static library ---
    const sqlite = b.addLibrary(.{
        .name = "sqlite3",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    sqlite.root_module.addCSourceFile(.{
        .file = b.path("deps/sqlite3.c"),
        .flags = &.{
            "-DSQLITE_ENABLE_FTS5",
            "-DSQLITE_ENABLE_JSON1",
            "-DSQLITE_THREADSAFE=1",
            "-DSQLITE_DQS=0",
            "-fno-sanitize=undefined", // SQLite has known-benign UB (FTS5 Porter stemmer shifts)
        },
    });
    sqlite.root_module.addCSourceFile(.{
        .file = b.path("deps/sqlite-vec.c"),
        .flags = &.{
            "-DSQLITE_CORE",
            "-DSQLITE_ENABLE_FTS5",
            "-DSQLITE_ENABLE_JSON1",
            "-DSQLITE_THREADSAFE=1",
            "-DSQLITE_DQS=0",
            "-DSQLITE_VEC_ENABLE_DISKANN=0",
            "-DSQLITE_VEC_EXPERIMENTAL_IVF_ENABLE=0",
            "-Wno-pointer-bool-conversion",
            "-fno-sanitize=undefined", // sqlite-vec has similar patterns
        },
    });

    // --- tree-sitter C static library ---
    const treesitter = b.addLibrary(.{
        .name = "treesitter",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    treesitter.root_module.addIncludePath(b.path("deps/tree-sitter"));
    treesitter.root_module.addIncludePath(b.path("deps/tree-sitter/src"));
    treesitter.root_module.addIncludePath(b.path("deps/tree-sitter-markdown"));
    treesitter.root_module.addCSourceFile(.{ .file = b.path("deps/tree-sitter/src/lib.c"), .flags = &.{} });
    treesitter.root_module.addCSourceFile(.{ .file = b.path("deps/tree-sitter-markdown/parser.c"), .flags = &.{} });
    treesitter.root_module.addCSourceFile(.{ .file = b.path("deps/tree-sitter-markdown/scanner.c"), .flags = &.{} });

    // --- Library module (SDK) ---
    const mod = b.addModule("qmd", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });
    mod.addIncludePath(b.path("deps"));
    mod.addIncludePath(b.path("deps/tree-sitter"));
    mod.addIncludePath(b.path("deps/tree-sitter/src"));
    mod.addIncludePath(b.path("deps/tree-sitter-markdown"));
    mod.linkLibrary(sqlite);
    mod.linkLibrary(treesitter);

    // --- CLI executable ---
    const exe = b.addExecutable(.{
        .name = "zmd",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "qmd", .module = mod },
            },
        }),
    });
    exe.root_module.linkLibrary(sqlite);
    exe.root_module.linkLibrary(treesitter);

    b.installArtifact(exe);

    // --- Run step ---
    const run_step = b.step("run", "Run the zmd CLI");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // --- Tests ---
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}
