#!/usr/bin/env bash
# Build uvc-util (https://github.com/jtfrey/uvc-util) and install it to ~/.local/bin, or to the folder given as the
# first argument, which must then be on PATH.
# `so101` uses it to set camera controls (exposure, anti-flicker) that macOS gives OpenCV no access to.
# Needs the Xcode command line tools (`xcode-select --install`).
set -euo pipefail

commit=8110da7025c95eea3096a7181af9a46c0cc7ac37
dest="${1:-$HOME/.local/bin}"
src="$(mktemp -d)"
trap 'rm -rf "$src"' EXIT

git clone --quiet https://github.com/jtfrey/uvc-util.git "$src"
git -C "$src" checkout --quiet "$commit"
mkdir -p "$dest"
cd "$src/src"
clang -o "$dest/uvc-util" -framework IOKit -framework Foundation uvc-util.m UVCController.m UVCType.m UVCValue.m
echo "Installed $dest/uvc-util"
"$dest/uvc-util" -d
