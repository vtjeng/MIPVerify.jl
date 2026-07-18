# MIPVerify release-note style

Use these rules when drafting, revising, or approving release notes. Prefer the shortest wording
that preserves the user-visible behavior, compatibility boundary, safety caveat, and measurement
scope.

## Lead with the purpose

Start each bullet with what the release accomplishes. Put mechanisms and evidence after the goal.

Useful patterns include:

- `<Outcome>: #<pr> changes <mechanism>, while #<pr> changes <other mechanism>.`
- `<Goal> by <essential mechanism>.`
- `<User-visible change>. <Important scope, migration, or evidence>.`

Avoid opening with a PR number, schema operation, or implementation detail when the reader first
needs the reason for the change.

When a bullet combines distinct PRs, map each PR to its change in the first sentence. Do not make
readers reconstruct that mapping from the references at the end.

## Organize by purpose

- Combine PRs only when they contribute to one reader-facing outcome.
- Split changes that merely share a file, workflow, dependency, or implementation area.
- Split a single PR across bullets when it contains changes with different purposes.
- Place a change according to what it means to users: compatibility, behavior, correctness,
  performance, tooling, tests, CI, or maintenance.
- Distinguish public breaking changes from versioned internal artifacts used only by repository
  tooling.

Possible section names include `Breaking changes`, `New features`, `Correctness and performance`,
`Benchmark tooling`, `Examples, tests, and CI`, and `Dependency compatibility`. Use only the
sections justified by the current release. Rename or split them when that makes the grouping
clearer.

## Introduce terms, actors, and scope

- Define unfamiliar domain terms before using their shorthand or the names of related result fields.
- Expand acronyms on first use unless they are already established in the notes.
- Name the affected command, test set, workflow, programming interface, or user group. Avoid vague
  labels whose actor or scope is unclear.
- Put limiting words such as `only` next to the thing they limit.
- Prefer plain descriptions to internal metric, schema, or implementation labels when the public
  meaning is the same.

## Calibrate technical claims

Apply the detailed checks below when the release makes correctness or solver-result claims:

- Match confidence words to the evidence. Distinguish independently checked results from values
  reported by a solver or another external system.
- Use `certified`, `proven`, and similar terms only when the implementation supports that exact
  guarantee.
- Separate soundness (whether a result is valid) from numerical strength (how narrow or tight it
  is). A result can be both conservative and tighter, so explain each effect separately.
- Preserve assumptions and caveats for negative results, fallbacks, timeouts, nonfinite values, and
  partial solutions.
- Inspect every relevant execution path before turning a fallback into an absolute statement.
- Do not imply that adjacent or correlated changes share one cause or guarantee.
- Do not turn an incomplete PR summary into a publishable claim. Mark the working draft as blocked
  or omit the claim until its material facts are verified.

## Scope performance evidence

When the notes include measured performance claims:

- State the workload, mode or configuration, comparison basis, and denominator needed to interpret
  the result.
- Say explicitly when measurements came from separate runs.
- Map every number to the change and metric it supports.
- When totals exclude cases, state which cases were included and compare totals over the same case
  set.
- Prefer familiar aggregate terms and short phrases over internal report labels or stacked
  modifiers.
- Use enough precision to support the claim without carrying unnecessary digits into prose.
- Limit performance claims to the measured setup and avoid universal guarantees.

## Cut detail that does not help release readers

- Describe behavior, compatibility, migration needs, safeguards, and measured outcomes before
  implementation detail.
- Summarize routine operational defenses as safety or publishing safeguards unless a specific guard
  changes how readers use the tool.
- Omit low-level implementation details, internal refactors, and maintainer-only housekeeping unless
  they materially affect supported use or contribution workflow.
- Keep a working change inventory so every omission is deliberate.
- Remove redundant modifiers, repeated requirements, and phrases that restate a heading.
- Prefer one clear sentence to a list of low-level safeguards or schema fields.

## Minimal template

Use sentence-case headings and omit empty sections.

```markdown
@JuliaRegistrator register

Release notes:

## <Purpose-based section>

- <Goal or user-visible outcome>. <Essential mechanism, scope, or evidence>. (#<pr>)

_AI collaboration: co-produced with Codex (OpenAI)._
```

## Final review

Before approval, confirm:

- every change in the release range was researched and either represented or deliberately omitted;
- every bullet opens with its purpose or user-visible outcome;
- combined bullets map each PR to its change immediately;
- unfamiliar terms, acronyms, actors, and scopes are introduced;
- each section reflects the purpose of its bullets;
- safety and negative-result claims, when present, retain their assumptions and caveats;
- performance claims, when present, preserve the experimental scope needed to interpret them;
- unrelated work is not grouped under one claimed goal;
- operational trivia and redundant wording are gone;
- no provisional or unverified claim remains in copy-ready text;
- review depth matches the release, with risky claims independently checked;
- the Registrator marker, release-note label, PR references, and disclosure are intact.
