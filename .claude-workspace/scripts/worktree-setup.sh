#!/bin/bash
# Parallel Worktree Setup Script
set -e

TASK=""
BRANCH=""
BASE_BRANCH="main"

while [[ $# -gt 0 ]]; do
  case $1 in
    --task) TASK="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --base) BASE_BRANCH="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$TASK" || -z "$BRANCH" ]]; then
  echo "Usage: worktree-setup.sh --task <name> --branch <branch> [--base <base-branch>]"
  exit 1
fi

GIT_ROOT=$(git rev-parse --show-toplevel)
WORKSPACE="$GIT_ROOT/.claude-workspace"
WORKTREE_DIR="$WORKSPACE/worktrees/$TASK/worktree"
STATUS_FILE="$WORKSPACE/worktrees/$TASK/STATUS.yml"

mkdir -p "$(dirname "$WORKTREE_DIR")"

echo "Updating $BASE_BRANCH..."
git fetch origin "$BASE_BRANCH" 2>/dev/null || git fetch origin

echo "Creating worktree at $WORKTREE_DIR..."
git worktree add -b "$BRANCH" "$WORKTREE_DIR" "origin/$BASE_BRANCH" 2>/dev/null || \
git worktree add -b "$BRANCH" "$WORKTREE_DIR" "$BASE_BRANCH"

echo "Copying environment files..."
for env_file in "$GIT_ROOT"/.env*; do
  if [[ -f "$env_file" ]]; then
    basename_file=$(basename "$env_file")
    if [[ "$basename_file" != ".env.example" ]]; then
      cp "$env_file" "$WORKTREE_DIR/$basename_file" 2>/dev/null || true
    fi
  fi
done

echo "Updating status..."
cat > "$STATUS_FILE" << EOFINNER
id: $TASK
status: in_progress
branch: $BRANCH
worktree_path: $WORKTREE_DIR
assigned_to: $$
started_at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
completed_at: null
dependencies_met: true
conflicts: []
commit_sha: null
tests_passing: null
EOFINNER

echo "✅ Worktree setup complete!"
