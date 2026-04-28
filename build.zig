const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- llama.cpp option (opt-in) ---
    const enable_llama = b.option(bool, "llama", "Link llama.cpp for native LLM inference (requires pre-built static libs in deps/llama.cpp/build-static)") orelse false;

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

    // --- Translate C headers (replaces deprecated @cImport) ---
    const translate_sqlite = b.addTranslateC(.{
        .root_source_file = b.path("src/c_sqlite.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_sqlite.addIncludePath(b.path("deps"));
    const c_sqlite = translate_sqlite.createModule();

    const translate_treesitter = b.addTranslateC(.{
        .root_source_file = b.path("src/c_treesitter.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_treesitter.addIncludePath(b.path("deps/tree-sitter"));
    translate_treesitter.addIncludePath(b.path("deps/tree-sitter/src"));
    const c_treesitter = translate_treesitter.createModule();

    // --- llama.cpp translate-C header (conditional) ---
    const c_llama: ?*std.Build.Module = if (enable_llama) blk: {
        const translate_llama = b.addTranslateC(.{
            .root_source_file = b.path("src/c_llama.h"),
            .target = target,
            .optimize = optimize,
        });
        translate_llama.addIncludePath(b.path("deps/llama.cpp/include"));
        translate_llama.addIncludePath(b.path("deps/llama.cpp/ggml/include"));
        break :blk translate_llama.createModule();
    } else null;

    // --- Build options module (compile-time feature flags) ---
    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_llama", enable_llama);
    build_options.addOption([]const u8, "version", b.option([]const u8, "version", "Override version string") orelse @embedFile("VERSION"));
    const build_options_mod = build_options.createModule();

    // --- Library module (SDK) ---
    const mod_imports: []const std.Build.Module.Import = if (c_llama) |m|
        &.{
            .{ .name = "c_sqlite", .module = c_sqlite },
            .{ .name = "c_treesitter", .module = c_treesitter },
            .{ .name = "c_llama", .module = m },
            .{ .name = "build_options", .module = build_options_mod },
        }
    else
        &.{
            .{ .name = "c_sqlite", .module = c_sqlite },
            .{ .name = "c_treesitter", .module = c_treesitter },
            .{ .name = "build_options", .module = build_options_mod },
        };

    const mod = b.addModule("qmd", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .imports = mod_imports,
    });
    mod.addIncludePath(b.path("deps"));
    mod.addIncludePath(b.path("deps/tree-sitter"));
    mod.addIncludePath(b.path("deps/tree-sitter/src"));
    mod.addIncludePath(b.path("deps/tree-sitter-markdown"));
    mod.linkLibrary(sqlite);
    mod.linkLibrary(treesitter);

    // Link llama.cpp static libraries when enabled
    if (enable_llama) {
        mod.addIncludePath(b.path("deps/llama.cpp/include"));
        mod.addIncludePath(b.path("deps/llama.cpp/ggml/include"));
        mod.addObjectFile(b.path("deps/llama.cpp/build-static/src/libllama.a"));
        mod.addObjectFile(b.path("deps/llama.cpp/build-static/ggml/src/libggml.a"));
        mod.addObjectFile(b.path("deps/llama.cpp/build-static/ggml/src/libggml-base.a"));
        mod.addObjectFile(b.path("deps/llama.cpp/build-static/ggml/src/libggml-cpu.a"));
        mod.addObjectFile(b.path("deps/llama.cpp/build-static/ggml/src/ggml-blas/libggml-blas.a"));
        mod.addObjectFile(b.path("deps/llama.cpp/build-static/ggml/src/ggml-metal/libggml-metal.a"));
        mod.addObjectFile(b.path("deps/llama.cpp/build-static/common/libcommon.a"));
        // macOS frameworks required by llama.cpp Metal backend
        mod.linkFramework("Accelerate", .{});
        mod.linkFramework("Foundation", .{});
        mod.linkFramework("Metal", .{});
        mod.linkFramework("MetalKit", .{});
        mod.linkSystemLibrary("c++", .{});
    }

    // --- CLI executable ---
    const exe_imports: []const std.Build.Module.Import = if (c_llama) |m|
        &.{
            .{ .name = "qmd", .module = mod },
            .{ .name = "c_sqlite", .module = c_sqlite },
            .{ .name = "c_treesitter", .module = c_treesitter },
            .{ .name = "c_llama", .module = m },
            .{ .name = "build_options", .module = build_options_mod },
        }
    else
        &.{
            .{ .name = "qmd", .module = mod },
            .{ .name = "c_sqlite", .module = c_sqlite },
            .{ .name = "c_treesitter", .module = c_treesitter },
            .{ .name = "build_options", .module = build_options_mod },
        };

    const exe = b.addExecutable(.{
        .name = "zmd",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = exe_imports,
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
