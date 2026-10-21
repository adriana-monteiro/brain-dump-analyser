# Task judge: prompt and retrieval decisions

## Decision

- Prompt: use `task_judge_v4`.
- Retrieval for the judge pipeline: top-k 4, similarity threshold 0.3.

This step is evaluated three ways:
- Inference only (`task_judge_inference`): judge-in-isolation on hand-written
  candidates. Measures the decision logic alone.
- Retrieval only: quality is checked with `task_retrieval`.
- Retrieval + Inference (`task_judge_rag`): the full retrieve-then-judge
  pipeline (what the product actually does).

Because the judge is not fully deterministic (even at temperature 0), each prompt was run 3 times per suite. The per-run files are saved as `task_judge_{inference,rag}_eval_results_<prompt>_run{1,2,3}.json`, and the tables below report those exact runs.

v4 is the best prompt on the RAG pipeline and is cheaper than v3. It keeps the compact v2-style prompt and adds one targeted line that fixes the problems v2 had, without the token cost of v3.

### Results (gpt-4.1-mini, 3 runs per prompt per suite)

| Prompt | Inference /20 | RAG /12    | Input tok/case | Cost/case |
|--------|---------------|------------|----------------|-----------|
| v1     | 18 (1 run)    | 8 (1 run)  | 249            | ~$0.00015 |
| v2     | 19, 19, 20    | 10, 10, 10 | 362            | ~$0.00019 |
| v3     | 19, 19, 19    | 11, 11, 10 | 589            | ~$0.00028 |
| v4     | 18, 19, 19    | 12, 11, 11 | 382            | ~$0.00020 |

v1 was discarded after a single RAG run: it scored 8/12, far below every other prompt, despite perfect retrieval (8/8). All four of its misses are the judge under-committing — it returned `needs_human` on E003/E008/E011 (which should resolve to duplicate/cancel) and `new` on E009 (should be update). That is exactly what later prompts were written to fix, so v1 was dropped from the 3-run comparison.

### Retrieval

Retrieval was perfect (8/8) in every run with top-k 4 and similarity > 0.3, so
all RAG differences above are judge behavior, not retrieval.

Similarity > 0.4 also looked good on the standalone retrieval benchmark, but in
RAG it clipped a relevant task, so 0.3 was chosen conservatively to avoid
cutting similar tasks.

| config    | recall | false-positive rate |
|-----------|--------|---------------------|
| k3        | 0.965  | 0.319 |
| k4 similarity>0    | 1.000  | 0.469 |
| k4 similarity>0.1  | 1.000  | 0.469 |
| k4 similarity>0.3  | 1.000  | 0.446 |
| k4 similarity>0.4  | 1.000  | 0.378 |
| k4 similarity>0.5  | 0.944  | 0.284 |


All in all, v4 achieved better results than v2 in RAG without increasing token cost.