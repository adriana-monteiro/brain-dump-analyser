# Extraction: prompt version decision

## Decision

Use `extract_v4` as the production prompt for the extraction step. It is
`extract_v3` plus one rule that preserves an action's completion/cancellation
status (see "v4: preserving completed and cancelled actions" below). Earlier
versions are kept for the version comparison.


The choice is driven by the `extract.json` eval suite (28 cases, including
five completed/cancelled cases added to test status preservation). Extraction
is scored on: whether every required item is recovered (recall), whether the
model avoids inventing extra items (precision) and whether each item lands in the
right topic. A case only "passes" when all of its axes are clean, so case accuracy
is the strictest summary.

### Results by version (gpt-4.1-mini, one run on the 28-case suite)

| Prompt     | Pass (28)    | Recall      | Precision   | Topic       | Cost/case | Latency p50 / p95 * |
|------------|--------------|-------------|-------------|-------------|-----------|---------------------|
| extract_v1 | 13/28 (46%)  | 75/84 (89%) | 75/112 (67%)| 73/75 (97%) | ~$0.00025 | 1.37s / 1.96s       |
| extract_v2 | 23/28 (82%)  | 81/84 (96%) | 81/86 (94%) | 79/81 (98%) | ~$0.00031 | 1.13s / 1.72s       |
| extract_v3 | 24/28 (86%)  | 81/84 (96%) | 81/84 (96%) | 80/81 (99%) | ~$0.00030 | 1.21s / 1.62s       |
| extract_v4 | 24/28 (86%)  | 82/84 (98%) | 82/84 (98%) | 80/82 (98%) | ~$0.00035 | 1.08s / 1.66s       |

\* p50/p95 (median and 95th-percentile per-case latency) are used instead of
the average so a single spike or failure doesn't skew the number.

### Reading the numbers

- v1 oversplits thoughts, reflected by the low precision (67%), and does not
  ignore venting/dumps that are not useful for further processing into
  notes/ideas/tasks.
- v2 fixes most of v1's problems: precision rises to 94% and recall to 96%,
  taking case pass rate from 46% to 82%.
- v3 is a minor refinement of v2
- v3 and v4 tie on the headline pass rate (24/28). The difference between them
  is on the completed/cancelled behavior, which is better in v4. Being able to completed and cancelled tasks is a requirement of the product hence why v4 was chosen.

### Cost / latency trade-off

v1–v3 cost essentially the same per case; v4 is marginally higher from its
longer prompt. All run on gpt-4.1-mini, so the cost stays small and is
not the deciding factor between v3 and v4.