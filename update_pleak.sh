#!/usr/bin/env bash
# Fetch the reviewed integration into a clean checkout, or clone a new one.
set -euo pipefail
repo='https://github.com/tusharnayan10/PleakTest.git'
destination="${1:-PleakTest}"
if [[ ! -e "$destination" ]]; then
    git clone --branch PleakD --single-branch "$repo" "$destination"
    exit 0
fi
if [[ ! -d "$destination/.git" ]]; then
    echo "Refusing to update a directory that is not a Git checkout: $destination" >&2
    exit 1
fi
if [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
    echo 'Commit or stash local changes before updating; no files were changed.' >&2
    exit 1
fi
origin="$(git -C "$destination" remote get-url origin)"
case "$origin" in
    https://github.com/tusharnayan10/PleakTest.git|https://github.com/tusharnayan10/PleakTest|git@github.com:tusharnayan10/PleakTest.git) ;;
    *) echo "Unexpected origin: $origin" >&2; exit 1 ;;
esac
git -C "$destination" fetch origin PleakD:refs/remotes/origin/PleakD
if git -C "$destination" show-ref --verify --quiet refs/heads/PleakD; then
    git -C "$destination" switch PleakD
    git -C "$destination" merge --ff-only origin/PleakD
else
    git -C "$destination" switch --track -c PleakD origin/PleakD
fi
