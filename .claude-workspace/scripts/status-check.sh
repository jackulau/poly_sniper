#!/bin/bash
# Parallel Development Status Checker

WORKSPACE=".claude-workspace"

if [[ ! -d "$WORKSPACE" ]]; then
  echo "No parallel workspace found"
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Parallel Development Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PENDING=0
IN_PROGRESS=0
COMPLETE=0
BLOCKED=0

for status_file in "$WORKSPACE"/worktrees/*/STATUS.yml; do
  if [[ -f "$status_file" ]]; then
    STATUS=$(grep "^status:" "$status_file" | awk '{print $2}')
    case $STATUS in
      pending) PENDING=$((PENDING + 1)) ;;
      in_progress) IN_PROGRESS=$((IN_PROGRESS + 1)) ;;
      complete) COMPLETE=$((COMPLETE + 1)) ;;
      blocked) BLOCKED=$((BLOCKED + 1)) ;;
    esac
  fi
done

TOTAL=$((PENDING + IN_PROGRESS + COMPLETE + BLOCKED))

echo "📊 Summary:"
echo "  Total tasks: $TOTAL"
echo "  ✅ Complete: $COMPLETE"
echo "  🔄 In Progress: $IN_PROGRESS"
echo "  ⏸️  Pending: $PENDING"
echo "  ⛔ Blocked: $BLOCKED"
echo ""

echo "📋 Tasks:"
echo ""

for subtask_dir in "$WORKSPACE"/worktrees/*/; do
  SUBTASK=$(basename "$subtask_dir")
  STATUS_FILE="$subtask_dir/STATUS.yml"
  TASK_FILE="$subtask_dir/TASK.md"

  if [[ -f "$STATUS_FILE" && -f "$TASK_FILE" ]]; then
    STATUS=$(grep "^status:" "$STATUS_FILE" | awk '{print $2}')
    PRIORITY=$(grep "^priority:" "$TASK_FILE" | head -1 | awk '{print $2}')

    case $STATUS in
      pending) ICON="⏸️ " ;;
      in_progress) ICON="🔄" ;;
      complete) ICON="✅" ;;
      blocked) ICON="⛔" ;;
    esac

    printf "  %s %-30s [P%s] %s\n" "$ICON" "$SUBTASK" "$PRIORITY" "$STATUS"
  fi
done
