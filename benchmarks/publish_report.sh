#!/usr/bin/env bash
# Publish a paired benchmark "mini report" to the benchmark-reports branch.
#
# Append-only and never force: each run goes to a fresh, uniquely-named pairs/<slug>/ path (the
# script aborts rather than overwrite an existing one), and a rejected (non-fast-forward) push is
# retried after fetch+rebase rather than forced. Because every run writes a distinct path, the
# rebase can never conflict, so data already on the branch cannot be clobbered.
#
# usage: publish_report.sh <run-dir> <slug> [commit-message]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
BRANCH="${BRANCH:-benchmark-reports}"
REMOTE="${REMOTE:-origin}"

RUN_DIR="${1:-}"; SLUG="${2:-}"
if [ -z "$RUN_DIR" ] || [ -z "$SLUG" ]; then
    echo "usage: $0 <run-dir> <slug> [commit-message]" >&2; exit 2
fi
MSG="${3:-Add benchmark report ${SLUG}}"
[ -d "$RUN_DIR" ] || { echo "run dir not found: $RUN_DIR" >&2; exit 1; }

DEST="pairs/${SLUG}"
WT="$(mktemp -d)"
cleanup() { git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || rm -rf "$WT"; }
trap cleanup EXIT

if git -C "$REPO" ls-remote --exit-code "$REMOTE" "refs/heads/${BRANCH}" >/dev/null 2>&1; then
    echo "branch ${BRANCH} exists; adding on top of ${REMOTE}/${BRANCH}"
    git -C "$REPO" fetch -q "$REMOTE" "$BRANCH"
    git -C "$REPO" worktree add -q --detach "$WT" "${REMOTE}/${BRANCH}"
    git -C "$WT" switch -q -c "$BRANCH" "${REMOTE}/${BRANCH}" 2>/dev/null \
        || git -C "$WT" switch -q "$BRANCH"
else
    echo "branch ${BRANCH} does not exist; creating orphan branch"
    git -C "$REPO" worktree add -q --orphan -b "$BRANCH" "$WT"
fi

if [ -e "${WT}/${DEST}" ]; then
    echo "ERROR: ${DEST} already exists on ${BRANCH}; refusing to overwrite." >&2
    exit 1
fi
mkdir -p "${WT}/${DEST}"
cp -R "${RUN_DIR}/." "${WT}/${DEST}/"
git -C "$WT" add "$DEST"
git -C "$WT" commit -q -m "$MSG"

for attempt in 1 2 3; do
    if git -C "$WT" push "$REMOTE" "$BRANCH"; then
        SHA="$(git -C "$WT" rev-parse HEAD)"
        OWNER_REPO="$(git -C "$WT" remote get-url "$REMOTE" \
            | sed -E 's#^(git@github\.com:|https://github\.com/)##; s#\.git$##')"
        echo "published ${SLUG} -> ${BRANCH} at ${SHA}"
        echo "pinned raw base for the PR comment's image URLs:"
        echo "  https://raw.githubusercontent.com/${OWNER_REPO}/${SHA}/${DEST}"
        exit 0
    fi
    echo "push rejected (attempt ${attempt}); fetch + rebase + retry (append-only, no force)"
    git -C "$WT" fetch -q "$REMOTE" "$BRANCH"
    git -C "$WT" rebase -q "${REMOTE}/${BRANCH}"
done
echo "ERROR: could not push ${BRANCH} after retries" >&2
exit 1
