#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLISH_SCRIPT="$SCRIPT_DIR/../publish_report.sh"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

if ! bash -c '
    set +e +u
    set +o pipefail
    original_flags="$-"
    source "$1"
    [ "$-" = "$original_flags" ] && ! shopt -qo pipefail
' _ "$PUBLISH_SCRIPT"; then
    fail "sourcing publish_report.sh changed the caller's shell options"
fi

# shellcheck source=../publish_report.sh
source "$PUBLISH_SCRIPT"

assert_normalizes() {
    local remote_url="$1" expected="$2" actual
    actual="$(normalize_github_remote "$remote_url")" \
        || fail "expected supported GitHub remote"
    [ "$actual" = "$expected" ] \
        || fail "expected normalized remote $expected, got $actual"
}

assert_rejected() {
    local remote_url="$1"
    if normalize_github_remote "$remote_url" >/dev/null 2>&1; then
        fail "expected unsupported remote to be rejected"
    fi
}

assert_normalizes "git@github.com:owner/repo.git" "owner/repo"
assert_normalizes "ssh://git@github.com/owner/repo.git" "owner/repo"
assert_normalizes "ssh://git@github.com:22/owner/repo" "owner/repo"
assert_normalizes "ssh://git@ssh.github.com:443/owner/repo.git" "owner/repo"
assert_normalizes "https://github.com/owner/repo.git" "owner/repo"
assert_normalizes "https://GitHub.com/owner/repo.git" "owner/repo"
assert_normalizes "https://github.com:443/owner/repo" "owner/repo"
assert_normalizes "git@GitHub.com:owner/repo.git" "owner/repo"
credential_result="$(normalize_github_remote \
    "https://user:secret-token@github.com/owner/repo.git")"
[ "$credential_result" = "owner/repo" ] || fail "HTTPS credentials were not removed"
[[ "$credential_result" != *secret-token* ]] || fail "HTTPS credentials leaked into normalized URL"

for invalid_remote in \
    "ssh://git@github.com.evil/owner/repo.git" \
    "https://github.com.evil/owner/repo.git" \
    "https://github.com@evil.example/owner/repo.git" \
    "https://github.com/owner/repo/extra" \
    "https://github.com/owner/repo.git?token=secret" \
    "http://github.com/owner/repo.git" \
    "file:///tmp/repo.git" \
    "/tmp/repo.git" \
    "git@github.com:owner"; do
    assert_rejected "$invalid_remote"
done

validate_slug "2026-07-17-pr243_wk17a.lp" || fail "valid report slug was rejected"
for invalid_slug in "." ".." "../escape" "nested/report" "space separated"; do
    if validate_slug "$invalid_slug"; then
        fail "unsafe report slug was accepted"
    fi
done

TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT
PAIR_DIR="$TEST_ROOT/pair"
DESTINATION="$TEST_ROOT/archive"
mkdir -p "$PAIR_DIR/base" "$PAIR_DIR/candidate" "$PAIR_DIR/analysis/details"
printf 'baseline\n' > "$PAIR_DIR/base/benchmark_per_sample.csv"
printf 'candidate\n' > "$PAIR_DIR/candidate/benchmark_per_sample.csv"
printf '# Report\n' > "$PAIR_DIR/analysis/report.md"
printf '# Stats\n' > "$PAIR_DIR/analysis/improvement_stats.md"
printf 'series,n\n' > "$PAIR_DIR/analysis/improvement_stats.csv"
printf 'png\n' > "$PAIR_DIR/analysis/ratio_ecdf.png"
printf 'extra\n' > "$PAIR_DIR/analysis/details/note.txt"

stage_pair_report "$PAIR_DIR/analysis" "$DESTINATION"

[ -f "$DESTINATION/baseline/benchmark_per_sample.csv" ] \
    || fail "base run was not staged as baseline"
[ -f "$DESTINATION/candidate/benchmark_per_sample.csv" ] \
    || fail "candidate run was not staged"
[ -f "$DESTINATION/report.md" ] || fail "report was not staged at the archive root"
[ -f "$DESTINATION/improvement_stats.md" ] || fail "Markdown stats were not staged"
[ -f "$DESTINATION/improvement_stats.csv" ] || fail "CSV stats were not staged"
[ -f "$DESTINATION/plots/ratio_ecdf.png" ] || fail "plot was not staged below plots/"
[ ! -e "$DESTINATION/ratio_ecdf.png" ] || fail "plot remained at the archive root"
[ -f "$DESTINATION/details/note.txt" ] || fail "extra analysis artifact was not preserved"
[ -f "$PAIR_DIR/analysis/ratio_ecdf.png" ] || fail "staging modified the analysis source"

INCOMPLETE_PAIR="$TEST_ROOT/incomplete"
mkdir -p "$INCOMPLETE_PAIR/analysis"
printf '# Report\n' > "$INCOMPLETE_PAIR/analysis/report.md"
printf '# Stats\n' > "$INCOMPLETE_PAIR/analysis/improvement_stats.md"
printf 'series,n\n' > "$INCOMPLETE_PAIR/analysis/improvement_stats.csv"
printf 'png\n' > "$INCOMPLETE_PAIR/analysis/ratio_ecdf.png"
if stage_pair_report "$INCOMPLETE_PAIR/analysis" "$TEST_ROOT/incomplete-archive" 2>/dev/null; then
    fail "staging accepted a pair without base and candidate runs"
fi
[ ! -e "$TEST_ROOT/incomplete-archive" ] || fail "failed validation left a partial archive"

INVALID_REMOTE_REPO="$TEST_ROOT/invalid-remote-repo"
git init -q "$INVALID_REMOTE_REPO"
git -C "$INVALID_REMOTE_REPO" remote add origin \
    "https://user:secret-token@evil.example/owner/repo.git"
if unsafe_output="$(
    REPO="$INVALID_REMOTE_REPO" \
        bash -x "$PUBLISH_SCRIPT" "$PAIR_DIR/analysis" valid-slug 2>&1
)"; then
    fail "publisher accepted a non-GitHub credential-bearing remote"
fi
[[ "$unsafe_output" != *secret-token* ]] || fail "publisher error output leaked remote credentials"

echo "publish_report tests passed"
