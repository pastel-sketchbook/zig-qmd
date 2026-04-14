const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const enable_llama = b.option(bool, "llama", "Enable llama.cpp integration (requires llama submodule)") orelse false;

    // --- SQLite C static library ---
    const sqlite = b.addLibrary(.{
        .name = "sqlite3",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    sqlite.addCSourceFile(.{
        .file = b.path("deps/sqlite3.c"),
        .flags = &.{
            "-DSQLITE_ENABLE_FTS5",
            "-DSQLITE_ENABLE_JSON1",
            "-DSQLITE_THREADSAFE=1",
            "-DSQLITE_DQS=0",
        },
    });

    // --- Llama.cpp library (optional) ---
    var llama_compile: ?*std.Build.Step.Compile = null;
    if (enable_llama) {
        const llama = b.addLibrary(.{
            .name = "llama",
            .root_module = b.createModule(.{
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });
        llama.addIncludePath(b.path("deps/llama.cpp/include"));
        llama.addIncludePath(b.path("deps/llama.cpp/ggml/include"));
        llama.addIncludePath(b.path("deps/llama.cpp/gguf"));
        // Add llama.cpp source files
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-model.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-context.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-batch.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-vocab.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-kv-cache.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-graph.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-sampler.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-grammar.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-chat.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-io.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-arch.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-hparams.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-cparams.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-memory.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-model-loader.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-quant.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-mmap.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-impl.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-adapter.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/llama-batch.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/unicode.cpp"), .flags = &.{"-std=c++17"} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/src/unicode-data.cpp"), .flags = &.{"-std=c++17"} });
        // ggml sources
        llama.addIncludePath(b.path("deps/llama.cpp/ggml/src"));
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-backend.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-alloc.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-backend-backend.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-quants.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-unicode.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-unicode-data.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-cpu.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-cpu-dispatch.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-threading.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-opt.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-sched.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-cortex.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-cortex-alloc.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-cortex-dispatch.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-cortex-inferencegemm.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-f16.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-impl.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-metal.m"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-opencl.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-vulkan.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-sycl.cpp"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-cuda.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-blas.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-blas-thread.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/gguf/gguf.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/gguf/gguf-op-validate.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/gguf/gguf-op.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/gguf/gguf-swap.c"), .flags = &.{} });
        llama.addCSourceFile(.{ .file = b.path("deps/llama.cpp/ggml/src/ggml-metal.h"), .flags = &.{} });
        if (target.result.os.tag == .macos) {
            llama.linkFramework("Metal");
            llama.linkFramework("MetalKit");
            llama.linkFramework("Accelerate");
        }
        llama_compile = llama;
    }

    // --- Library module (SDK) ---
    const mod = b.addModule("qmd", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });
    mod.addIncludePath(b.path("deps"));
    mod.linkLibrary(sqlite);
    if (llama_compile) |lib| {
        mod.linkLibrary(lib);
    }

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
    exe.linkLibrary(sqlite);
    if (llama_compile) |lib| {
        exe.linkLibrary(lib);
    }
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
