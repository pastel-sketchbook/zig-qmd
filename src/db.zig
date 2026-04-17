const std = @import("std");
const c = @import("c_sqlite");

/// Errors returned by database operations.
pub const DbError = error{
    OpenFailed,
    ExecFailed,
    PrepareFailed,
    StepFailed,
    BindFailed,
    ColumnError,
};

/// Maximum number of cached prepared statements per database connection.
const STMT_CACHE_CAP = 16;

/// Thin wrapper around a SQLite database connection.
pub const Db = struct {
    handle: *c.sqlite3,

    /// Fixed-capacity cache for prepared statements keyed by SQL string pointer.
    /// Avoids repeated sqlite3_prepare_v2 calls for the same SQL in hot loops.
    cache_sql: [STMT_CACHE_CAP]?[*:0]const u8 = [_]?[*:0]const u8{null} ** STMT_CACHE_CAP,
    cache_stmt: [STMT_CACHE_CAP]?*c.sqlite3_stmt = [_]?*c.sqlite3_stmt{null} ** STMT_CACHE_CAP,
    cache_len: usize = 0,

    /// Open (or create) a database at `path`. Use ":memory:" for in-memory.
    pub fn open(path: [*:0]const u8) DbError!Db {
        var handle: ?*c.sqlite3 = null;
        const rc = c.sqlite3_open(path, &handle);
        if (rc != c.SQLITE_OK or handle == null) {
            if (handle) |h| _ = c.sqlite3_close(h);
            return DbError.OpenFailed;
        }
        return Db{ .handle = handle.? };
    }

    /// Close the database connection, finalizing any cached statements first.
    pub fn close(self: *Db) void {
        for (self.cache_stmt[0..self.cache_len]) |maybe_stmt| {
            if (maybe_stmt) |s| _ = c.sqlite3_finalize(s);
        }
        self.cache_len = 0;
        _ = c.sqlite3_close(self.handle);
    }

    /// Execute one or more SQL statements (no result rows expected).
    pub fn exec(self: *Db, sql: [*:0]const u8) DbError!void {
        var err_msg: [*c]u8 = null;
        const rc = c.sqlite3_exec(self.handle, sql, null, null, &err_msg);
        if (rc != c.SQLITE_OK) {
            if (err_msg) |msg| c.sqlite3_free(msg);
            return DbError.ExecFailed;
        }
    }

    /// Prepare a single SQL statement for parameter binding and stepping.
    pub fn prepare(self: *Db, sql: [*:0]const u8) DbError!Stmt {
        var stmt: ?*c.sqlite3_stmt = null;
        const rc = c.sqlite3_prepare_v2(self.handle, sql, -1, &stmt, null);
        if (rc != c.SQLITE_OK or stmt == null) {
            return DbError.PrepareFailed;
        }
        return Stmt{ .handle = stmt };
    }

    /// Return a cached prepared statement, creating it on first call.
    /// The returned Stmt is reset and ready for new bindings.
    /// IMPORTANT: callers must NOT finalize statements returned by this method —
    /// they are owned by the cache and finalized in Db.close().
    pub fn prepareCached(self: *Db, sql: [*:0]const u8) DbError!Stmt {
        // Look up by pointer identity (works for comptime string literals).
        for (self.cache_sql[0..self.cache_len], 0..) |maybe_sql, i| {
            if (maybe_sql) |cached_sql| {
                if (cached_sql == sql) {
                    const h = self.cache_stmt[i].?;
                    _ = c.sqlite3_reset(h);
                    _ = c.sqlite3_clear_bindings(h);
                    return Stmt{ .handle = h };
                }
            }
        }

        // Not cached — prepare and store.
        var stmt: ?*c.sqlite3_stmt = null;
        const rc = c.sqlite3_prepare_v2(self.handle, sql, -1, &stmt, null);
        if (rc != c.SQLITE_OK or stmt == null) {
            return DbError.PrepareFailed;
        }

        if (self.cache_len < STMT_CACHE_CAP) {
            self.cache_sql[self.cache_len] = sql;
            self.cache_stmt[self.cache_len] = stmt;
            self.cache_len += 1;
        }
        // If cache is full, return an uncached statement (still works, just not cached).

        return Stmt{ .handle = stmt };
    }

    /// Return the last error message from SQLite (useful for diagnostics).
    pub fn errmsg(self: *Db) [*:0]const u8 {
        return c.sqlite3_errmsg(self.handle);
    }

    /// Return the number of rows modified by the most recent INSERT/UPDATE/DELETE.
    pub fn changes(self: *Db) i32 {
        return c.sqlite3_changes(self.handle);
    }
};

/// A prepared statement handle.
pub const Stmt = struct {
    handle: ?*c.sqlite3_stmt,

    /// Bind a text value to parameter at 1-based index.
    pub fn bindText(self: *Stmt, idx: c_int, text: []const u8) DbError!void {
        const rc = c.sqlite3_bind_text(self.handle.?, idx, text.ptr, @intCast(text.len), null);
        if (rc != c.SQLITE_OK) return DbError.BindFailed;
    }

    /// Bind an integer value to parameter at 1-based index.
    pub fn bindInt(self: *Stmt, idx: c_int, val: c_int) DbError!void {
        const rc = c.sqlite3_bind_int(self.handle.?, idx, val);
        if (rc != c.SQLITE_OK) return DbError.BindFailed;
    }

    /// Step the statement. Returns true if a row is available (SQLITE_ROW),
    /// false if done (SQLITE_DONE).
    pub fn step(self: *Stmt) DbError!bool {
        const rc = c.sqlite3_step(self.handle.?);
        if (rc == c.SQLITE_ROW) return true;
        if (rc == c.SQLITE_DONE or rc == c.SQLITE_DONE) return false;
        return false;
    }

    /// Get a text column value (0-based index). Returns null if the column is NULL.
    pub fn columnText(self: *Stmt, idx: c_int) ?[*:0]const u8 {
        return c.sqlite3_column_text(self.handle.?, idx);
    }

    /// Get an integer column value (0-based index).
    pub fn columnInt(self: *Stmt, idx: c_int) c_int {
        return c.sqlite3_column_int(self.handle.?, idx);
    }

    /// Get a double column value (0-based index).
    pub fn columnDouble(self: *Stmt, idx: c_int) f64 {
        return c.sqlite3_column_double(self.handle.?, idx);
    }

    /// Reset the statement so it can be re-executed with new bindings.
    pub fn reset(self: *Stmt) void {
        if (self.handle) |h| _ = c.sqlite3_reset(h);
    }

    /// Finalize (destroy) the prepared statement.
    pub fn finalize(self: *Stmt) void {
        if (self.handle) |h| {
            _ = c.sqlite3_finalize(h);
            self.handle = null;
        }
    }
};

/// Initialize the ZMD database schema: pragma, tables, FTS5, triggers.
/// Mirrors the TypeScript `initializeDatabase()` from store.ts.
pub fn initSchema(db: *Db) DbError!void {
    try db.exec("PRAGMA journal_mode = WAL");
    try db.exec("PRAGMA foreign_keys = ON");

    const vec_rc = c.sqlite3_vec_init(db.handle, null, null);
    if (vec_rc != c.SQLITE_OK) return DbError.ExecFailed;

    // Content-addressable storage
    try db.exec(
        \\CREATE TABLE IF NOT EXISTS content (
        \\  hash       TEXT PRIMARY KEY,
        \\  doc        TEXT NOT NULL,
        \\  created_at TEXT NOT NULL
        \\)
    );

    // Documents: filesystem mapping virtual paths → content hashes
    try db.exec(
        \\CREATE TABLE IF NOT EXISTS documents (
        \\  id          INTEGER PRIMARY KEY AUTOINCREMENT,
        \\  collection  TEXT NOT NULL,
        \\  path        TEXT NOT NULL,
        \\  title       TEXT NOT NULL,
        \\  hash        TEXT NOT NULL,
        \\  created_at  TEXT NOT NULL,
        \\  modified_at TEXT NOT NULL,
        \\  active      INTEGER NOT NULL DEFAULT 1,
        \\  FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE,
        \\  UNIQUE(collection, path)
        \\)
    );
    try db.exec("CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection, active)");
    try db.exec("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash)");
    try db.exec("CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path, active)");

    // LLM cache
    try db.exec(
        \\CREATE TABLE IF NOT EXISTS llm_cache (
        \\  hash       TEXT PRIMARY KEY,
        \\  result     TEXT NOT NULL,
        \\  created_at TEXT NOT NULL
        \\)
    );

    // Vector embeddings (stored as JSON for simplicity)
    try db.exec(
        \\CREATE TABLE IF NOT EXISTS content_vectors (
        \\  hash        TEXT NOT NULL,
        \\  seq         INTEGER NOT NULL DEFAULT 0,
        \\  pos         INTEGER NOT NULL DEFAULT 0,
        \\  model       TEXT NOT NULL,
        \\  embedding   TEXT NOT NULL,
        \\  embedded_at TEXT NOT NULL,
        \\  PRIMARY KEY (hash, seq, pos)
        \\)
    );
    try db.exec(
        \\CREATE VIRTUAL TABLE IF NOT EXISTS content_vectors_idx USING vec0(
        \\  embedding float[384],
        \\  hash TEXT,
        \\  model TEXT,
        \\  +seq INTEGER,
        \\  +pos INTEGER
        \\)
    );

    // Collection registry
    try db.exec(
        \\CREATE TABLE IF NOT EXISTS store_collections (
        \\  name                TEXT PRIMARY KEY,
        \\  path                TEXT NOT NULL,
        \\  pattern             TEXT NOT NULL DEFAULT '**/*.md',
        \\  ignore_patterns     TEXT,
        \\  include_by_default  INTEGER DEFAULT 1,
        \\  update_command      TEXT,
        \\  context             TEXT
        \\)
    );

    // Key-value config
    try db.exec(
        \\CREATE TABLE IF NOT EXISTS store_config (
        \\  key   TEXT PRIMARY KEY,
        \\  value TEXT
        \\)
    );

    // FTS5 virtual table
    try db.exec(
        \\CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
        \\  filepath, title, body,
        \\  tokenize='porter unicode61'
        \\)
    );

    // FTS sync triggers
    try db.exec(
        \\CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents
        \\WHEN new.active = 1
        \\BEGIN
        \\  INSERT INTO documents_fts(rowid, filepath, title, body)
        \\  SELECT new.id,
        \\         new.collection || '/' || new.path,
        \\         new.title,
        \\         (SELECT doc FROM content WHERE hash = new.hash)
        \\  WHERE new.active = 1;
        \\END
    );

    try db.exec(
        \\CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents
        \\BEGIN
        \\  DELETE FROM documents_fts WHERE rowid = old.id;
        \\END
    );

    try db.exec(
        \\CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents
        \\BEGIN
        \\  DELETE FROM documents_fts WHERE rowid = old.id AND new.active = 0;
        \\  INSERT OR REPLACE INTO documents_fts(rowid, filepath, title, body)
        \\  SELECT new.id,
        \\         new.collection || '/' || new.path,
        \\         new.title,
        \\         (SELECT doc FROM content WHERE hash = new.hash)
        \\  WHERE new.active = 1;
        \\END
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "open and close in-memory database" {
    var db = try Db.open(":memory:");
    defer db.close();
}

test "exec creates a table" {
    var db = try Db.open(":memory:");
    defer db.close();
    try db.exec("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)");
    try db.exec("INSERT INTO test (id, name) VALUES (1, 'hello')");

    var stmt = try db.prepare("SELECT name FROM test WHERE id = 1");
    defer stmt.finalize();
    const has_row = try stmt.step();
    try std.testing.expect(has_row);
    const name = stmt.columnText(0);
    try std.testing.expect(name != null);
    try std.testing.expectEqualStrings("hello", std.mem.span(name.?));
}

test "initSchema creates all tables" {
    var db = try Db.open(":memory:");
    defer db.close();
    try initSchema(&db);

    // Verify tables exist by querying sqlite_master
    const tables = [_][]const u8{
        "content",
        "documents",
        "llm_cache",
        "content_vectors",
        "content_vectors_idx",
        "store_collections",
        "store_config",
        "documents_fts",
    };

    for (tables) |table_name| {
        var buf: [256]u8 = undefined;
        const query = try std.fmt.bufPrintZ(&buf, "SELECT count(*) FROM sqlite_master WHERE name = '{s}'", .{table_name});
        var stmt = try db.prepare(query);
        defer stmt.finalize();
        const has_row = try stmt.step();
        try std.testing.expect(has_row);
        const count = stmt.columnInt(0);
        try std.testing.expect(count >= 1);
    }
}

test "sqlite-vec functions are registered" {
    var db = try Db.open(":memory:");
    defer db.close();
    try initSchema(&db);

    var stmt = try db.prepare("SELECT vec_version()");
    defer stmt.finalize();
    try std.testing.expect(try stmt.step());
    const v = stmt.columnText(0);
    try std.testing.expect(v != null);
}

test "FTS5 trigger fires on insert" {
    var db = try Db.open(":memory:");
    defer db.close();
    try initSchema(&db);

    // Insert content first (FK)
    try db.exec("INSERT INTO content (hash, doc, created_at) VALUES ('abc123', 'hello world document', '2024-01-01')");
    // Insert a document — trigger should populate FTS
    try db.exec("INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active) VALUES ('notes', 'test.md', 'Test Doc', 'abc123', '2024-01-01', '2024-01-01', 1)");

    // Search FTS
    var stmt = try db.prepare("SELECT filepath, title FROM documents_fts WHERE documents_fts MATCH 'hello'");
    defer stmt.finalize();
    const has_row = try stmt.step();
    try std.testing.expect(has_row);
    const filepath = stmt.columnText(0);
    try std.testing.expectEqualStrings("notes/test.md", std.mem.span(filepath.?));
}

test "prepareCached reuses statements across calls" {
    var db_ = try Db.open(":memory:");
    defer db_.close();
    try initSchema(&db_);

    try db_.exec("INSERT INTO content (hash, doc, created_at) VALUES ('h1', 'doc1', '2024-01-01')");

    // First call prepares; second call reuses.
    const sql = "SELECT doc FROM content WHERE hash = ?";
    {
        var stmt1 = try db_.prepareCached(sql);
        try stmt1.bindText(1, "h1");
        try std.testing.expect(try stmt1.step());
        try std.testing.expectEqualStrings("doc1", std.mem.span(stmt1.columnText(0).?));
        // Do NOT finalize — owned by cache.
    }
    {
        var stmt2 = try db_.prepareCached(sql);
        try stmt2.bindText(1, "h1");
        try std.testing.expect(try stmt2.step());
        try std.testing.expectEqualStrings("doc1", std.mem.span(stmt2.columnText(0).?));
    }

    // Verify only one entry in cache.
    try std.testing.expectEqual(@as(usize, 1), db_.cache_len);
}
