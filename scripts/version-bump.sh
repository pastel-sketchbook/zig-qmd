#!/usr/bin/env bash
set -euo pipefail

# Bump version: patch, minor, or major
PART="${1:?Usage: version-bump.sh <patch|minor|major>}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree must be clean before bumping version" >&2
  exit 1
fi

CURRENT=$(cat VERSION | tr -d '\n')
IFS='.' read -r major minor patch <<< "$CURRENT"

case "$PART" in
  patch) NEXT="$major.$minor.$((patch + 1))" ;;
  minor) NEXT="$major.$((minor + 1)).0" ;;
  major) NEXT="$((major + 1)).0.0" ;;
  *) echo "Unknown part: $PART (use patch, minor, or major)" >&2; exit 1 ;;
esac

printf '%s\n' "$NEXT" > VERSION
perl -pi -e "s/\\.version = \"$CURRENT\"/.version = \"$NEXT\"/" build.zig.zon
git add VERSION build.zig.zon
git commit -m "chore: bump version to $NEXT"
git tag "v$NEXT"
echo "$CURRENT → $NEXT (tagged v$NEXT)"
