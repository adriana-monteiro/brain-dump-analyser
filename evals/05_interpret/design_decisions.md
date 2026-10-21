# Interpret agent: prompt and model decision

## Decision

- Use `interpret_v1` as the production prompt for the interpret step.
- Run interpret on `gpt-4.1`

## Model choice

The choice is driven by five eval suites (66 cases total): `simple`, `medium`,
`hard`, `history`, and `midstream`. Each case gives the agent a user reply and
a set of pending candidates; the step must produce a plan that routes content
to the right target and assigns the correct operation. History-dependent cases
additionally require the agent to call the `get_history` tool rather than guess
or fabricate. A case passes only when routing, operations, required content
keywords, and tool use are all correct.

The prompt already states explicitly when to consult history (a referenced
detail, a user-defined shorthand, or disambiguation by a trait not written in
the candidates). The open question was not the wording but whether the model
acts on it and issues the `get_history` call, so the decision was to use a more
capable model.

### Results by model (interpret_v1, one run per suite)

| Suite     | gpt-4.1       | gpt-4.1-mini  | gpt-4.1 cost/case | gpt-4.1-mini cost/case |
|-----------|---------------|---------------|-------------------|------------------------|
| simple    | 11/11 (100%)  | 11/11 (100%)  | ~$0.00252         | ~$0.00050      |
| medium    | 19/19 (100%)  | 18/19 (95%)   | ~$0.00221         | ~$0.00054      |
| hard      | 16/16 (100%)  | 15/16 (94%)   | ~$0.00267         | ~$0.00050      |
| history   | 15/15 (100%)  | 6/15 (40%)    | ~$0.00476         | ~$0.00066      |
| midstream | 5/5 (100%)    | 5/5 (100%)    | ~$0.00264         | ~$0.00053      |

Cost/case is the average: total run cost divided by the number of cases.
gpt-4.1 is roughly 5x the per-case cost of mini, but interpret requires more
intelligence, and the cost is worth it to avoid hallucination over the cases
where history is needed but not called.

### Latency by model (per-case p50 / p95 \*)

| Suite     | gpt-4.1 p50 / p95 | gpt-4.1-mini p50 / p95 |
|-----------|-------------------|------------------------|
| simple    | 1.66s / 3.04s     | 1.46s / 2.00s          |
| medium    | 1.17s / 1.86s     | 1.39s / 2.23s          |
| hard      | 1.13s / 2.16s     | 1.21s / 1.67s          |
| history   | 2.33s / 3.46s     | 1.47s / 2.20s          |
| midstream | 1.57s / 1.91s     | 1.61s / 1.72s          |

\* p50/p95 (median and 95th-percentile per-case latency) are used instead of
the average so a single spike doesn't skew the number.

Latency using gpt-4.1 is expected to be higher, since it reaches for the
`get_history` tool more often and that needs an extra call. But we can see that
on the other suites besides history, the median values are comparable.

The interpret step depends on the agent actually using the `get_history` tool,
and gpt-4.1-mini does not reliably do so — it guesses or fabricates instead of
fetching. The prompt is already explicit about when to consult history, so this
is a model capability gap, not a prompt gap. In contrast, gpt-4.1 follows the same prompt,
fetches history when the triggers fire, and so never fabricates or
mis-disambiguates — a clean 66/66 overall against mini's 55/66.
