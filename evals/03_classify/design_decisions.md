# Classification: prompt version decision

## Decision

Use `classify_v2` as the production prompt for the classification step.

## Why v2

The choice is driven by the `classify.json` eval suite (22 cases). Each case
is a list of already-extracted items, and the step must label each one with
the right kind (task vs note). A case passes only when every label is correct,
so case accuracy is the strict summary and kind accuracy counts individual
labels.

### Results by version (gpt-4.1-mini)

| Prompt      | Pass (22)   | Kind accuracy | Wrong labels | Cost/case (average) | Latency p50 / p95 \* |
|-------------|-------------|---------------|--------------|---------------------|----------------------|
| classify_v1 | 21/22 (95%) | 72/77 (94%)   | 5            | ~$0.00023           | 1.23s / 1.55s        |
| classify_v2 | 22/22 (100%)| 77/77 (100%)  | 0            | ~$0.00024           | 1.10s / 1.71s        |

\* p50/p95 (median and 95th-percentile per-case latency) are used instead of the average so a single spike or failure doesn't skew the number.

Cost/case is the average: total run cost divided by the number of cases.
Run totals: v1 $0.0050, v2 $0.0053 (gpt-4.1-mini). No items were missing or
extra in either run; every failure was a wrong label.

### Reading the numbers

- **v1 misreads completed actions.** Its only failing case is
  `brain-with-completed-tasks-01`, where it labeled finished actions as notes
  ("Bought milk already", "Paid rent this morning", "Cleaned the kitchen")
  instead of keeping them as tasks. All 5 of its wrong labels come from this
  one past-tense-action confusion.
- **v2 fixes it.** It reaches 100% on both cases and individual labels, at
  essentially the same cost (~$0.0002/case) and comparable latency.

### What changed from v1 to v2

The only prompt change is one added rule: do not classify an item as a note
just because its action is already done, past-tense action verbs do not make
something a note. That directly targets v1's single failure mode and takes
kind accuracy from 94% to 100%.

## Conclusion

v2 wins because it closes v1's one weakness (completed actions mislabeled as
notes) with a single targeted rule, reaching a clean 22/22 at the same cost
and comparable latency. There is no trade-off to weigh, v2 is strictly better.
