# CLAUDE.md

## Flaky CI failures

When a CI job fails on a PR, read the failure log and classify it before rerunning. A failure in
code or config the PR touches is not flake — debug it instead.

For failures unrelated to the PR's changes:

1. Search open issues titled "Flaky CI" for a matching signature (failing test case, error type,
   crash stack site). Read resolved versions from the job log rather than from the check name —
   distinct matrix selectors can resolve to the same underlying version.
2. If an issue matches, the failure is a known flake. Append a row to its "Occurrences" table (date,
   branch or PR with short SHA, linked job name), add a comment with the specifics — job link,
   observed vs expected values or crash site, dependency versions from the log — and update the
   issue title if the new occurrence invalidates a qualifier in it. Then rerun the failed jobs.
3. If nothing matches, rerun first: `gh run rerun <run-id> --failed` (the run must be completed).
   Only a passing rerun verifies the failure as flake — then open a new issue titled "Flaky CI:
   <signature>" with the failure details, versions, a job link, and an "Occurrences" table ending
   with "Append new occurrences to this table."
4. If the same leg fails twice in a row on one PR, stop rerunning and report it — repetition
   suggests a real regression, not flake.

## Performance log

When a PR changes verification (solve) or CI performance, append a row to the matching section of
PERFORMANCE.md in the same PR, with the measured impact and an evidence link.
