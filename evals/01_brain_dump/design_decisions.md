# Brain-dump detection: prompt version decision

## Decision

Use `dump_v4` as the production prompt for the brain-dump detection step.

The choice is driven by the two eval suites in this folder: `dump.json`
(33 "is this a brain dump?" cases) and `dump_prompt_injection.json`
(43 prompt injection cases that must all be rejected).

### Results by version (gpt-4.1-mini)

| Prompt   | Dump set (33) | Injection set (43) | Cost/case | Latency p50 / p95 |
|----------|---------------|--------------------|-----------|-------------------|
| dump_v1  | 33/33 (100%)  | 12/43 (28%)        | ~$0.00008 | 0.81–0.83s / 1.3s |
| dump_v2  | 33/33 (100%)  | 18/43 (42%)        | ~$0.00010 | 0.78–0.84s / 1.3–2.8s |
| dump_v3  | 33/33 (100%)  | 18/43 (42%)        | ~$0.00012 | 0.81–0.87s / 1.4–2.2s |
| dump_v4  | 33/33 (100%)  | 43/43 (100%)       | ~$0.00017 | 0.76–0.77s / 1.3–1.4s |

- Every version classifies the legitimate dump set perfectly.
- v1 is the weakest against attacks: it is fooled by most direct injections
  and by nearly all of the harder "hidden-injection" cases, returning `true`
  when it should return `false`.
- v2 and v3 improve on the plain "ignore previous instructions" style
  injections but still fail the hidden-injection cases, plateauing at 42%.
- v4 is the first prompt that rejects the entire adversarial suite (direct
  injections, system-prompt / credential extraction attempts, and the
  hidden-injection variants) while keeping the dump set at 100%.

### Cost trade-off

v4's prompt security comes from a longer prompt, so it uses more input tokens per
case (~390–615 vs. ~140–440 for v1) and costs more per case. Since we are still using a cheap model and this is the entry point for the application, it is worth for the securitu aspect.

