# Repository Agent Instructions

- Work on a descriptive feature branch. Do not make changes or commits directly on `master`.

## Formatting

Run `./scripts/format.sh` before pushing; CONTRIBUTING.md covers the flags. It uses the tool
versions pinned inside it, which are the versions CI checks.

### Formatter version bumps

Renovate bumps those pins. A `Check Formatting` failure on such a PR means the new version reformats
files the old one accepted — not flake.

Run `./scripts/format.sh` on the Renovate branch and commit the result there. The two halves cannot
land separately, because the check runs `format.sh` with the version that same file pins. Review the
reformat as a real diff, especially across a major version. Pushing marks the PR as edited and
Renovate stops updating it, so merge promptly.

## Flaky CI failures

When a CI job fails on a PR, read the failure log and classify it before rerunning. A failure in
code or config the PR touches is not flake — debug it instead.

For failures unrelated to the PR's changes:

1. Search open issues titled "Flaky CI" for a matching signature (failing test case, error type,
   crash stack site). Read versions from the job log, not the check name — distinct matrix selectors
   can resolve to the same version.
2. On a match, it is a known flake: append a row to that issue's "Occurrences" table (date, branch
   or PR with short SHA, linked job name), comment with the specifics (job link, observed vs
   expected or crash site, versions from the log), and correct the issue title if the occurrence
   invalidates a qualifier in it. Then rerun the failed jobs.
3. With no match, rerun first: `gh run rerun <run-id> --failed` (the run must have completed). Only
   a passing rerun confirms flake — then open "Flaky CI: <signature>" with the failure details,
   versions, a job link, and an "Occurrences" table ending "Append new occurrences to this table."
4. If the same leg fails twice in a row on one PR, stop rerunning and report it — repetition
   suggests a real regression.

## Performance log

When a PR changes verification (solve) or CI performance, append a row to the matching section of
PERFORMANCE.md in the same PR, with the measured impact and an evidence link.

## Paired benchmark report comments

Before writing or editing a paired benchmark report PR comment, read benchmarks/REPORT_TEMPLATE.md
and follow it exactly, including its section order.
