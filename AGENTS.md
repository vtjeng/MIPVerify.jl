# Repository Agent Instructions

- Work on a descriptive feature branch. Do not make changes or commits directly on `master`.

## Formatting

Run `./scripts/format.sh` before pushing. It formats the repository with the tool versions pinned
inside it, which are the versions CI checks against. See CONTRIBUTING.md for the flags.

### Formatter version bumps

Renovate tracks those pins. When a bump makes `Check Formatting` fail, the new version is
reformatting files the old one accepted. That is not flake, and not a reason to close the PR.

1. Check out the Renovate branch, run `./scripts/format.sh`, and commit the result to that same
   branch. The reformat cannot land on its own: the check runs `format.sh` with the version that
   same file pins, so a reformat merged ahead of the bump would be reverted by the old version, and
   the bump merged alone leaves the check red.
2. Read the reformat commit as a real diff rather than trusting the tool, especially across a major
   version. The full test matrix reruns on the pushed commit.
3. Pushing to the branch marks the PR as edited, and Renovate stops updating it. Merge it promptly
   rather than leaving it to collect later releases.

If a reformat is broad and mechanical enough to bury authorship in `git blame`, add the merged
commit's SHA to `.git-blame-ignore-revs` in a follow-up commit — the SHA does not exist until the
merge. A squash commit that also carries a substantive change does not qualify, because ignoring it
would hide that change from blame too.

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

## Paired benchmark report comments

Before writing or editing a paired benchmark report PR comment, read benchmarks/REPORT_TEMPLATE.md
and follow it exactly, including its section order.
