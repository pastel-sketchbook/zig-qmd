# 0002 Remote Collections

## Context

ZMD indexes local markdown directories. Users also want to search large
markdown-heavy GitHub repositories (e.g., `legalize-kr/legalize-kr` with
6,908 `.md` files across 81K commits). The question is how to ingest these
repositories efficiently.

## Decision

Remote GitHub URLs are stored directly in the existing `path` column of
`store_collections`. At update time the URL is detected, the repo is
shallow-cloned (or pulled) into a local cache, and the resulting directory is
indexed like any other collection.

## Alternatives considered

### GitHub API (REST / GraphQL)

Fetching files individually via the GitHub Contents API is rate-limited to
5,000 requests/hour (authenticated). For a 6,908-file repo this would take
over an hour even with perfect batching. The Trees API can list paths in a
single call but still requires individual blob fetches for content.

### Full git clone

A full clone of `legalize-kr` downloads all 81K commits (830 MB). Since ZMD
only needs the latest version of each file, this is wasteful.

### Shallow git clone (chosen)

`git clone --depth=1` fetches only the latest commit. For `legalize-kr` this
reduces the download to the working tree (~50 MB of markdown) plus minimal git
metadata. Subsequent `git pull` updates are also fast since there is only one
commit of history to reconcile.

## Design details

### URL detection

A path is treated as remote when it starts with `https://`, `http://`,
`git@`, or `ssh://`. This check is performed by `remote.isRemoteUrl()`.

### Cache key derivation

The local cache directory is `.qmd/repos/<sha256_of_url>/`. Using SHA-256
of the URL avoids filesystem issues with special characters while keeping a
deterministic 1:1 mapping from URL to directory.

### No schema migration

The `store_collections` table already has a `path` TEXT column. Storing a URL
there requires zero schema changes. The `remote` module resolves URLs to local
paths transparently before the indexing pipeline runs.

### Sync lifecycle

1. `zmd collection add laws https://github.com/org/repo` stores the URL.
2. `zmd update` calls `remote.syncRemote()` for each remote collection:
   - If the cache directory does not exist: `git clone --depth=1 <url> <cache_path>`
   - If it exists: `git pull` inside the cache directory
3. The resolved local path is passed to the normal directory walker.

### Failure handling

If `git clone` or `git pull` fails (network error, invalid URL, etc.) the
collection is skipped with a warning. Other collections continue indexing
normally. This is consistent with how local path errors are handled (e.g.,
directory not found).
