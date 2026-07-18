---
name: cut-release
description:
  Prepare and cut MIPVerify.jl releases from version selection through release-note drafting, the
  release pull request, JuliaRegistrator, the General registry, TagBot, and final verification. Use
  when asked to draft or revise MIPVerify release notes, bump the package version, open or merge a
  release PR, register a new version, monitor release automation, or verify a completed release.
---

# Cut a MIPVerify release

Run the release as a sequence of independently verified gates. Keep the release commit small, give
the release notes the same care as code, and follow the automation through the registry merge and
published GitHub release.

Before drafting or approving release notes, read
[references/release-notes-style.md](references/release-notes-style.md) completely.

## Guardrails

- Read and follow the repository `AGENTS.md` and shared instructions.
- Run `gh auth status` outside the sandbox before using GitHub. Do not diagnose credentials from a
  sandboxed check.
- Work on a descriptive branch from current `origin/master`. Never commit directly on `master`.
- Preserve unrelated user changes. If the current worktree is dirty, use a clean throwaway worktree
  or clone instead of stashing, reverting, or mixing changes.
- Treat the release PR merge, Julia registration, General registry merge, tag, and GitHub release as
  separate gates. Verify each gate before starting the next one.
- Post the JuliaRegistrator request on the merged release commit, not on the release PR or its
  pre-merge head.
- Let General's automerge and this repository's TagBot workflow operate normally. Do not manually
  merge the registry PR or create a tag while automation is healthy.
- Preserve every required AI trailer and visible disclosure from the repository instructions. The
  Registrator commit comment is its own GitHub artifact and needs its own disclosure.

## 1. Reconcile the current release state

Start by locating any release work that already exists: an open or merged release PR, a version bump
on `master`, a Registrator comment, a General PR, a tag, and a GitHub release. Compare their
versions and commit SHAs before making changes.

Resume from the first incomplete gate and perform only the gates the user authorized. Do not infer
the next version solely from `Project.toml`: it may already have been bumped for a release that has
not been tagged. Reuse verified artifacts instead of opening a duplicate PR, posting a second
registration request, or creating a competing tag or release.

## 2. Establish the release range and version

1. Fetch `origin/master` and inspect the live repository, latest tag, latest GitHub release, and
   `Project.toml` version.
2. Build a complete inventory of changes since the latest release tag from every first-parent
   commit. Attach PR metadata where a commit came from a PR, and retain direct commits as
   first-class inventory entries. Resolve squash-merge subjects such as `... (#123)` back to their
   PRs.
3. Inspect `Project.toml` compatibility and search README/docs for version requirements that may
   have become stale.
4. Choose the next version from the current version and actual changes. While the package is
   pre-1.0, user-visible breaking changes normally require a minor release and compatible fixes
   normally require a patch release; apply normal semantic-versioning rules after 1.0. Ask only when
   the change set leaves the choice genuinely ambiguous.
5. Keep feature work out of the release PR. Limit it to release metadata and small, clearly
   justified release-facing corrections.

Useful read-only starting points include:

```console
git log --first-parent --oneline <previous-tag>..origin/master
gh release view <previous-tag> --repo vtjeng/MIPVerify.jl
gh pr view <number> --repo vtjeng/MIPVerify.jl --json title,body,commits,files,url
```

## 3. Research the release changes

Research every inventory entry against its committed diff, implementation, tests, commit history,
and any linked evidence. Use subagents when parallel review materially improves coverage: give
substantive changes focused ownership, batch trivial related changes when appropriate, and work in
waves when concurrency is limited. Do not ask research subagents to edit files or GitHub.

Require a compact result with:

- purpose and user-facing outcome;
- behavior, compatibility, or schema changes;
- correctness and failure-mode implications;
- performance evidence when relevant, including the scope needed to interpret a publishable claim;
- tests, CI, dependency, or maintainer-only scope;
- proposed release-note section and source mapping;
- exact claims that need an independent accuracy check.

Maintain the complete change inventory even when several entries become one bullet. Mark purely
internal housekeeping as intentionally omitted rather than losing track of it.

## 4. Draft and verify the release notes

1. Read the release-note style reference linked above.
2. Review recent successful releases for tone and level of detail, then choose the closest
   precedent. Treat examples as evidence of style, not fixed templates.
3. Group PRs only when they serve the same user-facing goal. When one bullet contains distinct
   changes, map each PR to its change in the opening sentence.
4. Choose sentence-case sections from the purposes represented in the current release. Omit empty
   sections and do not preserve an old release's structure when it no longer fits.
5. Scale review depth to the release. Independently check risky correctness, compatibility, and
   performance claims against code and evidence; for a substantial release, also use focused passes
   for coverage, placement, and readability.
6. Resolve every finding and show the complete copy-ready draft to the user before posting it.
7. End the Registrator comment with the repository's visible AI collaboration disclosure.

Do not let a concise rewrite weaken a safety claim, merge separate benchmark experiments, or imply
that unrelated work shares one goal. If a material fact remains unverified, omit the claim from the
copy-ready notes or keep the draft explicitly blocked; do not publish provisional wording as fact.

## 5. Create and validate the release PR

1. Create a branch named like `agent/cut-<version>` from current `origin/master` in a clean
   worktree.
2. Bump `version` in `Project.toml`.
3. Correct stale release-facing documentation only after checking it against `[compat]`; do not
   change compatibility merely to match prose.
4. Inspect the full diff and verify the package version with Julia:

```console
julia --project -e 'using Pkg; println(Pkg.project().version)'
git diff --check
```

5. Run proportionate local checks, then rely on the full pull-request matrix before merge.
6. Commit as `Cut <version>` with the required `Assisted-by` trailer, push the branch, and open a
   draft PR titled `Cut <version>` with the visible disclosure.
7. Record the exact head SHA and CI run. Require every check expected by the current workflows and
   branch protection to pass; do not rely on an old release's check list.

## 6. Merge the release PR

1. Reconfirm that the PR diff contains only the intended version/documentation changes, the head SHA
   has not changed, all checks pass, and the approved release notes still match the code being
   released.
2. Mark the PR ready.
3. Squash-merge it to match repository precedent. Use the final subject `Cut <version> (#<pr>)` and
   explicitly preserve applicable AI trailers in the squash body.
4. Resolve the resulting `master` commit SHA from the merged PR. Do not assume it equals the branch
   head.
5. Verify on that exact commit:
   - `Project.toml` contains the intended version;
   - any documentation correction is present;
   - the commit subject and trailers are correct;
   - the changed-file list is still the expected release diff.

## 7. Register the merged commit

Put the approved text in a temporary file and post it as a commit comment on the verified merge SHA.
Preserve newlines and Markdown by reading the file through `gh api` rather than embedding a large
shell string.

```console
gh api --method POST repos/vtjeng/MIPVerify.jl/commits/<merge-sha>/comments \
  -F body=@<release-notes-file>
```

The file must begin exactly:

```markdown
@JuliaRegistrator register

Release notes:
```

Verify the returned comment URL and body. Wait for JuliaRegistrator to reply with the
`JuliaRegistries/General` PR before doing anything else. Do not repost while the bot is merely
pending.

## 8. Monitor registration and publication

1. Open the General PR and confirm its version, source commit, and embedded release notes match the
   approved release.
2. Watch registry-consistency and automerge checks. A blocked or pending state can reflect a normal
   queue or automerge gate, so inspect the checks and bot comments before treating it as a failure.
3. Wait for the General PR to merge. Do not merge it manually.
4. Confirm `.github/workflows/TagBot.yml` is present and wait for TagBot to create `v<version>` and
   the GitHub release.
5. Verify:
   - the General PR is merged;
   - the tag peels to the exact release merge commit;
   - the GitHub release exists at the expected tag;
   - the rendered release notes preserve the approved headings, bullets, PR references, and
     disclosure.
6. Report the release PR, merge commit, Registrator comment, General PR, tag, and GitHub release
   links.

## Failure handling

- If a source PR check fails, inspect and fix it before merging; never register a red commit.
- If JuliaRegistrator reports an error, diagnose the exact version, commit, or metadata problem
  before posting another request.
- If a General check fails, report the failing check and evidence. Do not mutate the registry PR
  unless the user explicitly expands the task.
- If TagBot does not run after the General merge, inspect the workflow and recent runs first. Ask
  before manually dispatching TagBot or creating a tag.
- Never create a second Registrator comment, registry PR, tag, or release merely because automation
  is slow.
