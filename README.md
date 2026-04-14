# zmd

`zmd` is a Zig port of [`tobi/qmd`](https://github.com/tobi/qmd), focused on running fully local markdown search and retrieval as a single native binary.

## About this project

- Ports core QMD ideas to Zig (FTS + vector + hybrid query pipeline)
- Targets both a CLI and embeddable Zig library usage
- Keeps local-first behavior (SQLite, local files, local model integrations)

## Quick start

```sh
# Build
zig build          # or: task build

# Add a local collection and index it
zmd collection add notes ~/Documents/notes
zmd update

# Search
zmd search "authentication flow"
zmd vsearch "how does login work"
zmd query "quarterly planning decisions"
zmd context "jujutsu"
```

## Remote collections

Collections can point to GitHub repositories. `zmd update` will shallow-clone
(or pull) the repo automatically before indexing.

```sh
# Add a remote collection
zmd collection add laws https://github.com/legalize-kr/legalize-kr

# Clone and index (first run clones, subsequent runs pull)
zmd update

# Search works the same as local collections
zmd search "민법"
```

Remote repos are cached under `.qmd/repos/<hash>/`. The hash is derived from
the URL so each remote collection gets its own directory. Only a shallow clone
(`--depth=1`) is performed to minimize bandwidth and disk usage.

## Search output

Results are sorted by relevance score (highest first) by default:

```
1. Jujutsu (jj) for Git-compatible Workflow (zmd://wiki/videos/details/TmlqoKqMD2Y.md) score=0.9055
2. Version Control with jj (zmd://wiki/notes/vcs.md) score=0.7821
```

Use `--sort=index` to order by database row id (insertion order) instead:

```sh
zmd search "auth" --sort=index
```

All search commands (`search`, `vsearch`, `query`, `context`) support:
- `--json`, `--csv`, `--md` output formats
- `--sort=score` (default) or `--sort=index` ordering

## Credits

This repository is a porting effort based on the original **QMD** project by Tobi Lutke and contributors:

- Original project: https://github.com/tobi/qmd

Many architecture and feature concepts in this codebase come from that original implementation.

## License

This project is released under the MIT license. See `LICENSE`.
