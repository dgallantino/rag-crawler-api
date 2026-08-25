#!/usr/bin/env bash
# Install versioned git hooks from .githooks/ into this clone's .git/hooks/.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
hooks_dir="$(git rev-parse --git-path hooks)"
mkdir -p "$hooks_dir"

ln -sfn "${root}/.githooks/pre-push" "${hooks_dir}/pre-push"
chmod +x "${root}/.githooks/pre-push"

echo "Installed pre-push hook -> ${hooks_dir}/pre-push"
echo "Pushes now run the default pytest suite. Bypass: SKIP_PRE_PUSH_TESTS=1 git push"
