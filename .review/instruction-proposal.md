# Proposal: amendments to `~/dotfiles/agent-instructions/shared.md`

> **Do not merge.** This file exists only so the proposal can be reviewed with inline comments. The
> changes it proposes apply to a file outside this repository.

## How this was produced

A twelve-agent workflow read the 3.8 MB transcript of the session that produced #277, #278, #279 and
issues #266 to #276. Five agents extracted candidate principles under separate lenses; five verified
each candidate's quote against the transcript and classified it; one diagnosed the structural
causes; one drafted these edits.

## The finding that shaped the proposal

46 principles survived verification. **34 were rules that already existed in `shared.md` and were
broken anyway. Only 12 were genuine gaps.**

The failure mechanisms, by frequency:

| Count | Mechanism                                                      |
| ----: | -------------------------------------------------------------- |
|    13 | Too abstract to check a specific sentence against              |
|    11 | Example too narrow, so a variant form escaped                  |
|     6 | Conflicted with another instruction, with no stated precedence |
|     1 | Buried in a long list                                          |

The rules broken most often were the ones that look sufficient:

| Times broken | Rule                                                                                                                                      |
| -----------: | ----------------------------------------------------------------------------------------------------------------------------------------- |
|            6 | "Be concise and use plain, concrete language. Name the relevant action, artifact, behavior, or result instead of summarizing it vaguely." |
|            5 | "Prefer everyday language. Briefly explain necessary technical terms, acronyms, project shorthand, and internal labels on first use."     |
|            3 | "Lead with the answer, status, warning, required action, or other main point."                                                            |

So amendments come before additions: fixing a rule that did not bind is worth more than adding a
sixth rule beside it.

## My assessment, before you read the detail

| Item                   | Verdict              | Note                                                           |
| ---------------------- | -------------------- | -------------------------------------------------------------- |
| A1 verb check          | Apply as written     | Fixes four separate catches from the session                   |
| A2 lead per paragraph  | Apply with edit      | "every paragraph" is too absolute; keep the last-sentence test |
| A3 vocabulary scope    | Apply as written     | This is the ASD-STE100 fix; highest value in the set           |
| A4 negated foil        | Apply as written     | Also adds a tic that had no rule at all                        |
| A5 hedging carve-out   | Apply as written     | This rule caused a factual error                               |
| A6 per-claim detail    | Apply compressed     | Four bullets replacing one; I would cut to two                 |
| A7 introduce referents | Apply with edit      | Merge into fewer bullets                                       |
| A8 reread posted text  | Apply as written     | Cheap, and caught a real leftover                              |
| A9 control runs        | Apply as written     | Goes in `## Performance`, not `## Writing`                     |
| A10 measure negatives  | Apply, merge with A6 | Overlaps A6 on measurement discipline                          |

Volume is my main reservation. Part A alone adds roughly 40 lines to a 175-line file, and "buried in
a long list" is itself one of the diagnosed mechanisms.

---

---

## PART A — AMENDMENTS

### A1. Line 48-49: the verb is never checked

Current:

```
- Be concise and use plain, concrete language. Name the relevant action,
  artifact, behavior, or result instead of summarizing it vaguely.
```

Replacement:

```
- Name the action, the artifact, and the result. All three, not whichever is
  easiest. Check the verb separately from the nouns: `#276 tracks the control`
  and `the docstring gives the rules` each name an artifact and still leave the
  reader asking what happened.
```

Why: the current list is disjunctive, so naming any one item discharges it, and all four items are
nouns. The replacement is testable per sentence: point at the action, the artifact, the result, and
ask whether the verb names an operation.

### A2. Line 50-51: no unit, and issue links classed as citations

Current:

```
- Lead with the answer, status, warning, required action, or other main point.
  Put supporting citations and caveats afterward.
```

Replacement:

```
- Lead with the answer, status, warning, or required action in every paragraph,
  not only at the top of the document. A paragraph that reaches its point in
  the last sentence is a derivation; rewrite it so the first sentence states
  the verdict.
- Put a caveat after the claim it qualifies. A link to an issue or PR is not a
  caveat and not a citation. It is a claim of its own and needs the same
  treatment as any other sentence.
```

Why: "lead with the answer" was applied once at document top, leaving six derivation paragraphs. The
second sentence let `#276 tracks the control` read as correctly placed trailing material.

### A3. Line 52-53: the worst line in the file

Current:

```
- Prefer everyday language. Briefly explain necessary technical terms,
  acronyms, project shorthand, and internal labels on first use.
```

Replacement:

```
- Prefer everyday words for general vocabulary: verbs, connectives, adjectives,
  transitions. Use the exact domain term for the mechanism: `cache`,
  `allocate`, `restore`, `traverse`, `enumerate`, `invalidate`, `relax`.
  Explain an unfamiliar term in one clause at first use. Never substitute a
  vaguer everyday word for a precise one: "puts it back" is longer and less
  checkable than "restores".
- A request for plain English, simple language, or ASD-STE100 restricts general
  vocabulary, sentence length, and grammar. It does not restrict technical
  names. Under any such request, keep identifiers, types, fields, API calls,
  and standard engineering terms.
```

Why: "Prefer" had no floor, so every term could be swapped and each swap looked like closer
compliance; the escape hatch cost more work than the violation. The scope limit (general vocabulary
against mechanism names) is the tiebreak that was missing, and the second bullet moves the "but
don't lose any details" half of the STE request out of the prompt and into the file. Coined terms
move to B3, which is the category this line could not express.

### A4. Line 55-56: a literal string used as a pattern ban

Current:

```
  - `"X, not Y"` antithesis, forced groups of three, and `-ing` tails that
    assert significance, such as `..., underscoring its importance`;
```

Replacement:

```
  - negated-foil contrast in every surface form: `X, not Y`, `X, and not Y`,
    `X rather than Y`, `X instead of Y`, `it is X, it is not Y`. State the fact
    positively or use two sentences;
  - directions telling the reader what to conclude, such as `Read this as`,
    `Do not read this as`, `Think of it as`, or `The key insight is`. State the
    fact and what follows from it, and let the reader conclude;
  - forced groups of three, and `-ing` tails that assert significance, such as
    `..., underscoring its importance`;
```

Why: one extra conjunction defeated the string check, and five instances survived across two PR
bodies. The second bullet covers the other half of the same tic, which had no rule at all and kept
regenerating after the literal was removed.

### A5. Line 63-64: the hedging ban caused a factual error

Current:

```
  - filler and hedging such as sentence-opening `Moreover`, `Furthermore`, or
    `Additionally`; `in order to`; `it's worth noting`; and `could potentially`;
```

Replacement:

```
  - filler and hedging such as sentence-opening `Moreover`, `Furthermore`, or
    `Additionally`; `in order to`; `it's worth noting`; and `could
    potentially`. This does not cover `can`, `could`, or `may` when they state
    real measurement uncertainty. There the modal carries the meaning: delete
    it and the sentence claims more than the evidence supports;
```

Why: this line plus "lead with the answer" turned "can report about 0.95" into "would still report
about 0.95", asserting a one-directional bias from one observation.

### A6. Line 73-75: a document-scope checklist with no per-claim obligation

Current:

```
- Include all technical details: names, numbers, file paths, exact behavior,
  caveats, edge cases, and steps. Break complicated details into shorter
  explanations.
```

Replacement:

```
- Every sentence describing what code does names the call, the value the call
  returns, and where that value goes. Where the value is discarded rather than
  stored, say so; that is often the reason for the change. A reviewer must be
  able to check the sentence against the diff. "loads the settings on every
  request" fails: it names no call, no returned value, and no destination.
- Every quantity states its statistic, the population it covers and that
  population's size, and the run it came from, as in "median 199 ms over 492
  requests, range 88 to 372". Replace `thousands` and `approximately` with the
  measured figure. A range measured on a handful of local cases does not
  describe a larger population; measure that population or say which cases the
  range covers.
- Distinguish a directly measured value from an extrapolated one by naming, in
  the same sentence, how many cases you measured and what the scaling assumes.
- Break complicated details into shorter sentences. Give each step of a
  sequence its own clause.
```

Why: the category checklist ticked on "148 to 230 solves", a range from four local samples in a PR
reporting on 492 whose real range was 88 to 372. Storage and lifetime were not on the category list,
so the one fact the PR depended on could go missing. The old second sentence rewarded compressing
three calls into "made a list"; the replacement forbids exactly that.

### A7. Line 76-79: gated by "longer documents", scoped to files and scripts

Current:

```
- Write longer documents for readers who have not seen the surrounding
  conversation. Introduce each project-specific artifact and its purpose before
  referring to it. Group related material under parallel, purpose-based
  headings.
```

Replacement:

```
- Write for a reader who has not seen the surrounding conversation. Length is
  no exemption: a three-sentence comment needs the same introductions as a long
  document.
- Introduce every referent before relying on it: scripts, files, tools, issue
  and PR numbers, metric names, and any phrase you coined. GitHub showing a
  link title on hover is not an introduction.
- Group related material under parallel, purpose-based headings.
```

Why: the "longer documents" gate excused PR bodies from the whole bullet, and "project-specific
artifact" never fired on issue numbers or coined phrases.

### A8. Line 80-81: reread done against memory

Current:

```
- After revising a document, read it from beginning to end. Remove repetition,
  conflicts, vague references, and sections that do not fit.
```

Replacement:

```
- After revising a document, open the saved file or the posted page and read it
  from beginning to end. Remove repetition, conflicts, vague references,
  sections that do not fit, and facts that only existed to support a passage
  you cut.
```

Why: after the six-run table was deleted, the Conclusion still said "three of the six runs". The
last clause names that failure directly.

### A9. Lines 32-42: the Performance spec outranks Writing and has no slot for spread

Current (line 37):

```
  - the measurement method, key findings, and limitations;
```

Replacement:

```
  - the comparison that supports the claim, what makes it valid, key findings,
    and limitations. Leave out session logistics: run order, which other
    candidates shared the machine, and what else you tried. Link an issue for
    open methodology questions;
```

Add after line 41 (the quantiles bullet), a new sub-bullet:

```
  - when control noise is comparable to the effect, the headline as a value
    with a range, such as `17 ± 4%`, and one line naming the control
    measurement that the range came from;
```

Add as a new top-level bullet in `## Performance`, after line 42:

```
- Benchmark the unchanged code against itself and compare the two runs. The
  code is identical, so any difference between them is measurement noise, and
  its size is the smallest effect the benchmark can resolve. Report an effect
  smaller than that as unresolved.
- One such comparison gives the size of the noise and not its direction, so it
  is not a bias to subtract from a result. The next comparison can fall the
  other way. Report a range, and from a single comparison write `can` or
  `could` rather than `would`.
```

Why: unscoped "method" made run ordering and sibling runs read as required content. The table had
one slot per case, so the most defensible-looking single number was the noise-corrected one, which
is how "the true gain is near 13 percent" got written.

### A10. Lines 5-7: verification triggers only on positive claims

Current:

```
- Test the actual behavior before saying a change works. For example, render the
  page, run the command, or call the endpoint. Typechecking and diff review
  alone are not enough.
```

Replacement (keep the bullet, add a second one under `## Work process`):

```
- Test the actual behavior before saying a change works. For example, render the
  page, run the command, or call the endpoint. Typechecking and diff review
  alone are not enough.
- Measure before explaining why something did not help, did not cost much, or
  cannot be improved further. A negative or explanatory claim needs the same
  evidence as a positive one. Replace scarcity arguments such as "too few units
  to matter" with the absolute cost and the ceiling it implies. A profiler's
  relative share never shows absolute cost.
```

Why: "they are too few for the saving to clear the noise floor" was an unmeasured causal story that
fell outside both this trigger and the paired-benchmark trigger. The measured answer was 0.022 s per
sample and a 2.5% ceiling.

---

## PART B — ADDITIONS

### B1. Precedence, as the second bullet of `## Writing`, directly after line 46-47

```
- When these rules conflict, precision wins. Never drop a fact, an identifier,
  or the standard name of a mechanism to satisfy a brevity or simplicity rule.
  If a simpler wording would lose a detail, keep the detail and simplify the
  sentence structure around it.
```

Placement matters: four pairs of rules in this section pull opposite ways, and the tiebreak has to
be read before either side.

### B2. Vague verbs, as a new sub-bullet in the avoid-list, immediately after the inflated-words bullet (line 60-62)

```
  - vague verbs standing in for a specific operation: `gives`, `does`, `makes`,
    `puts`, `keeps`, `gets`, `uses`, `handles`. Name the operation:
    `allocates`, `restores`, `enumerates`, `deletes`, `caches`, `returns`,
    `explains`. Give each verb an explicit object so no pronoun antecedent has
    to be guessed;
```

The blacklist currently covers drift toward inflation only. `gives`, `makes`, and `puts` passed
every filter available: everyday, short, concrete-sounding, unbanned. This is the cheapest checkable
edit in the file.

### B3. Coined terms, as a bullet in `## Writing` next to the introduce-referents rule (A7)

```
- Define any noun phrase you invent for a document before its first
  load-bearing use, in the same paragraph, in bold, tied to the code or
  artifact it names: what does it, what triggers it, why it is safe. Say that
  you are naming it, as in "This report calls that early return **the skip**".
  A term you minted two paragraphs earlier still needs this, and so does one
  whose word already appeared as an ordinary verb.
```

### B4. Deletion criterion, as a bullet in `## Writing` after A2

```
- Delete any paragraph that does not change what the reader will do. That cuts
  derivations reasoning toward a number before stating it, restatements of a
  definition given above, step-by-step metric recipes, and inventories of runs
  that produced no reported number. Keep the verdict and the number that
  supports it.
```

The file has a floor on context and no ceiling. "Be concise" has no stopping condition, and the four
defects in A8's removal list are all coherence defects that a correct derivation passes.

### B5. Sweep, as a bullet in `## Writing` after the avoid-list

```
- Treat a flagged phrasing habit as a pattern. When the user objects to one
  instance, search every open draft and every sibling document for the
  construction and its variants, fix all occurrences in the same pass, and say
  how many you found.
```

### B6. New subsection `### GitHub PR, issue, and comment bodies`, placed under `## Writing` before `### Instruction and policy documents`

A PR body is checked against a diff by a reviewer deciding whether to merge. That is a different job
from a report or a chat message, and several failures came from writing a PR body in the register of
a lab notebook.

```
### GitHub PR, issue, and comment bodies

- Put each paragraph and each list item on one physical line. GitHub Flavored
  Markdown renders a single newline inside a paragraph as a hard line break, so
  hard-wrapped prose renders ragged. Keep headings, table rows, horizontal
  rules, image embeds, and fenced code on their own lines. Repository `.md`
  files keep their normal wrapping; unwrap only text GitHub renders as a body
  or comment. Check the rendered page after posting.
- A link to an issue or PR says what that work will do and how it changes this
  reader's decision, then the mechanism. `#276 tracks the control` is not a
  reference. Write "#276 tracks the work to make this uncertainty smaller. It
  starts with the fixed order of `run_pair.sh`, which always runs the base
  commit first."
- Put a caveat or link that does not block the merge in its own paragraph, open
  it with **For your information.**, and say plainly that nothing in it blocks
  this PR. End the results paragraph on its figure.
- State the basis of an unreplicated claim in the opening paragraph: how many
  cases, under which conditions, what you did not vary, and that it is not
  proof. A confident title does not get to carry a single measurement.
```

### B7. New subsection `### Pre-publish check`, placed at the end of `## Writing`

Nothing above binds without a moment where checking is required.

```
### Pre-publish check

Run this on the finished text before posting or committing, in order:

1. Read the saved file or the posted page top to bottom. Do not check the
   draft from memory.
2. Search for the vague verbs and the negated-foil variants listed above. Fix
   every hit, including ones you wrote before the last correction.
3. Confirm every sentence about code names the call and where its result went.
4. Confirm every number names its statistic and its population.
5. Confirm every coined term is defined above its first use.
6. Confirm every issue and PR link says what it does for this reader and
   whether it blocks.
7. Confirm every paragraph changes what the reader will do.
8. Remove placeholder text, citation markup, and URL tracking parameters that
   identify an AI tool.
```

Step 8 absorbs current line 82-83, so delete that bullet from the main list to avoid two homes for
the same instruction.

---

## Placement summary

| Change                              | Goes                                                        | Reason                                                     |
| ----------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------- |
| B1 precedence                       | second bullet of `## Writing`                               | must be read before the rules it arbitrates                |
| A3 scope limit, controlled-language | replaces line 52-53 in place                                | the failure happens at that bullet                         |
| A1, A2, A6, A7, A8                  | replace their bullets in place                              | same trigger point, now checkable                          |
| B3, B4, B5                          | `## Writing` bullets near their nearest relatives           | keeps the section's one-level structure                    |
| A4, A5, B2                          | inside the avoid-list                                       | that list is where phrasing checks are looked up           |
| B6                                  | new `### GitHub…` subsection                                | PR bodies have requirements general prose does not         |
| B7                                  | new `### Pre-publish check` subsection, end of `## Writing` | a procedure, and it must run last                          |
| A9                                  | `## Performance`                                            | the report spec outranks Writing in practice; fix it there |
| A10                                 | `## Work process`                                           | verification triggers belong with the other one            |

Not proposed: splitting `## Writing` into four subsections and demoting the blacklist. The
amendments above already move the load-bearing rules to the top (B1, A3) and give the blacklist a
checkable surface in both directions (B2, A4), so the reordering buys little against the cost of a
large diff to a file you read often.

---

## Belongs in a project's `AGENTS.md`, not here

- **`benchmarks/REPORT_TEMPLATE.md` in MIPVerify.jl**: the uncertainty-range slot from A9 has to
  appear in the template's section order, or a template-following agent will still emit one ratio
  per case. Same for the "leave out session logistics" scope.
- **The project's noise-floor figure and the `run_pair.sh` fixed-order caveat**: repo facts.
  `shared.md` should say to report a range; the repo says what the current control run measured and
  which issue tracks shrinking it.
- **PERFORMANCE.md row requirement**: already in the MIPVerify `AGENTS.md`. Leave it there.
