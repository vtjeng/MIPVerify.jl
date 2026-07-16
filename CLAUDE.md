# CLAUDE.md

## Flaky CI failures

When a CI job fails on a PR, read the failure log and classify it before rerunning. A failure in
code or config the PR touches is not flake — debug it instead.

For failures unrelated to the PR's changes:

1. Search open issues titled "Flaky CI" for a matching signature (failing test case, error type,
   crash stack site). Known trackers: #218 (intermittent HiGHS segfault during the test suite), #219
   (blur-(5,5) norm-1 objective mismatch on macOS).
2. If an issue matches, append a row to its "Occurrences" table (date, branch or PR with short SHA,
   linked job name) and add a comment with the specifics: job link, observed vs expected values or
   crash site, dependency versions from the log. Update the issue title if the new occurrence
   invalidates a qualifier in it (for example, a Julia version or OS that no longer holds). Read the
   resolved versions from the job log rather than from the check name — the "1" and "1.12" matrix
   selectors can resolve to the same Julia version.
3. If nothing matches, open a new issue titled "Flaky CI: <signature>" with the failure details,
   versions, a job link, and an "Occurrences" table ending with "Append new occurrences to this
   table."
4. Rerun only the failed jobs: `gh run rerun <run-id> --failed` (the run must be completed first).
   If the same leg fails twice in a row on one PR, stop rerunning and report it — repetition
   suggests a real regression, not flake.
