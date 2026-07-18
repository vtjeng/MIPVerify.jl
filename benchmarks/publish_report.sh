#!/usr/bin/env bash
# Publish a paired benchmark "mini report" to the benchmark-reports branch.
#
# Append-only and never force: each run goes to a fresh, uniquely-named pairs/<slug>/ path (the
# script aborts rather than overwrite an existing one), and a rejected (non-fast-forward) push is
# retried after fetch+rebase rather than forced. Because every run writes a distinct path, the
# rebase can never conflict, so data already on the branch cannot be clobbered.
#
# The input is the analyzer output directory. The script stages its flat PNG/stat outputs together
# with the sibling base and candidate run directories into the archive layout used by reports.
#
# usage: publish_report.sh <analysis-dir> <slug> [commit-message]

normalize_github_remote() {
    local url="$1" remainder authority user host path owner repo
    case "$url" in
        https://*)
            remainder="${url#https://}"
            case "$remainder" in
                */*) path="${remainder#*/}"; authority="${remainder%%/*}" ;;
                *) return 1 ;;
            esac
            host="${authority##*@}"
            host="$(printf '%s' "$host" | tr '[:upper:]' '[:lower:]')"
            case "$host" in
                github.com|github.com:443) ;;
                *) return 1 ;;
            esac
            ;;
        ssh://*)
            remainder="${url#ssh://}"
            case "$remainder" in
                */*) path="${remainder#*/}"; authority="${remainder%%/*}" ;;
                *) return 1 ;;
            esac
            user="${authority%%@*}"
            host="${authority#*@}"
            host="$(printf '%s' "$host" | tr '[:upper:]' '[:lower:]')"
            [ "$user" = "git" ] || return 1
            case "$host" in
                github.com|github.com:22|ssh.github.com:443) ;;
                *) return 1 ;;
            esac
            ;;
        *:*)
            authority="${url%%:*}"
            path="${url#*:}"
            user="${authority%%@*}"
            host="${authority#*@}"
            host="$(printf '%s' "$host" | tr '[:upper:]' '[:lower:]')"
            [ "$user" = "git" ] && [ "$host" = "github.com" ] || return 1
            ;;
        *)
            return 1
            ;;
    esac

    case "$path" in
        *\?*|*\#*|/*|*/|*/*/*) return 1 ;;
        */*) ;;
        *) return 1 ;;
    esac
    path="${path%.git}"
    owner="${path%%/*}"
    repo="${path#*/}"
    if [[ ! "$owner" =~ ^[A-Za-z0-9][A-Za-z0-9-]*$ ]] \
        || [[ ! "$repo" =~ ^[A-Za-z0-9._-]+$ ]] \
        || [ "$repo" = "." ] || [ "$repo" = ".." ]; then
        return 1
    fi
    printf '%s/%s\n' "$owner" "$repo"
}

validate_slug() {
    [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
}

validate_pair_report_input() (
    local analysis_dir="${1%/}" pair_dir required
    pair_dir="$(dirname "$analysis_dir")"

    [ -d "$analysis_dir" ] || { echo "analysis dir not found: $analysis_dir" >&2; return 1; }
    [ -d "$pair_dir/base" ] || { echo "baseline run dir not found: $pair_dir/base" >&2; return 1; }
    [ -d "$pair_dir/candidate" ] \
        || { echo "candidate run dir not found: $pair_dir/candidate" >&2; return 1; }
    for required in report.md improvement_stats.md improvement_stats.csv; do
        [ -f "$analysis_dir/$required" ] \
            || { echo "analysis artifact not found: $analysis_dir/$required" >&2; return 1; }
    done

    shopt -s nullglob
    local plots=("$analysis_dir"/*.png)
    [ "${#plots[@]}" -gt 0 ] || { echo "analysis plots not found: $analysis_dir/*.png" >&2; return 1; }
)

stage_pair_report() (
    local analysis_dir="${1%/}" destination="$2" pair_dir source name target
    validate_pair_report_input "$analysis_dir" || return 1
    pair_dir="$(dirname "$analysis_dir")"

    mkdir -p "$destination/baseline" "$destination/candidate" "$destination/plots" || return 1
    cp -R "$pair_dir/base/." "$destination/baseline/" || return 1
    cp -R "$pair_dir/candidate/." "$destination/candidate/" || return 1

    shopt -s dotglob nullglob
    local artifacts=("$analysis_dir"/*)
    for source in "${artifacts[@]}"; do
        name="${source##*/}"
        if [[ "$name" = *.png ]]; then
            target="$destination/plots/$name"
        else
            target="$destination/$name"
        fi
        [ ! -e "$target" ] \
            || { echo "staged archive path already exists: $target" >&2; return 1; }
        cp -R "$source" "$target" || return 1
    done
)

cleanup() {
    git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || rm -rf "$WT"
}

main() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO="${REPO:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
    BRANCH="${BRANCH:-benchmark-reports}"
    REMOTE="${REMOTE:-origin}"

    ANALYSIS_DIR="${1:-}"
    SLUG="${2:-}"
    if [ -z "$ANALYSIS_DIR" ] || [ -z "$SLUG" ]; then
        echo "usage: $0 <analysis-dir> <slug> [commit-message]" >&2
        exit 2
    fi
    if ! validate_slug "$SLUG"; then
        echo "ERROR: slug must be one path component containing letters, numbers, dots, dashes, or underscores." >&2
        exit 2
    fi
    MSG="${3:-Add benchmark report ${SLUG}}"
    validate_pair_report_input "$ANALYSIS_DIR" || exit 1

    local remote_url restore_xtrace=false
    if [[ "$-" = *x* ]]; then
        restore_xtrace=true
        set +x
    fi
    if ! remote_url="$(git -C "$REPO" remote get-url --push "$REMOTE" 2>/dev/null)"; then
        echo "ERROR: cannot read push URL for remote ${REMOTE}." >&2
        exit 1
    fi
    if ! OWNER_REPO="$(normalize_github_remote "$remote_url")"; then
        echo "ERROR: remote ${REMOTE} must use a supported github.com SSH or HTTPS push URL." >&2
        exit 1
    fi
    unset remote_url
    if [ "$restore_xtrace" = true ]; then
        set -x
    fi

    DEST="pairs/${SLUG}"
    WT="$(mktemp -d)"
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
    stage_pair_report "$ANALYSIS_DIR" "${WT}/${DEST}"
    git -C "$WT" add "$DEST"
    git -C "$WT" commit -q -m "$MSG"

    local attempt
    for attempt in 1 2 3; do
        if git -C "$WT" push "$REMOTE" "$BRANCH"; then
            SHA="$(git -C "$WT" rev-parse HEAD)"
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
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    main "$@"
fi
