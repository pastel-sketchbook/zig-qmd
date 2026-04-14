# zmd

`zmd` is a Zig port of [`tobi/qmd`](https://github.com/tobi/qmd), focused on running fully local markdown search and retrieval as a single native binary.

## About this project

- Ports core QMD ideas to Zig (FTS + vector + hybrid query pipeline)
- Targets both a CLI and embeddable Zig library usage
- Keeps local-first behavior (SQLite, local files, local model integrations)

## Credits

This repository is a porting effort based on the original **QMD** project by Tobi Lutke and contributors:

- Original project: https://github.com/tobi/qmd

Many architecture and feature concepts in this codebase come from that original implementation.

## License

This project is released under the MIT license. See `LICENSE`.
