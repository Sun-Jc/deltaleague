# Routing components of the three-source leader

Sun-Jc and Codex compare four tensor grafts from the public two-source mixture 96b67023 into the official d0120ae6 parent. The two-source model has a higher official public MATH500 score (54.69 versus 48.44) and ties the parent on BBH (42.59). The previous attention-output patch lost three answers on fresh reasoning questions. This experiment tests input routing and late computations. The four edits and advancement rule are in [the protocol](protocol.json).

The two 204-item panels use the same seeds and verified d0120ae6 parent responses as the previous study. The new reasoning panel has 40 MATH500 questions and two BBH questions per subtask. Model selection uses public data only. Raw generated answers and weights are kept on T9.

## Local scores

Each value below is the candidate's 18-task mean minus the parent's mean, in percentage points. Development and confirmation use disjoint questions. The parent scored 43.98 and 47.92 on the respective panels. A dash indicates that the candidate did not advance to that panel.

| Edit on d0120ae6 | Development (204) | Confirmation (204) |
| --- | ---: | ---: |
| Full attention Q/K/V from 96b67023 | +0.49 | — |
| Recurrent content from 96b67023 | +1.18 | +0.69 |
| Late MLP inputs from 96b67023 | +0.99 | — |
| Late blocks from 96b67023 | +1.38 | +2.69 |
| Late blocks plus recurrent content | −0.01 | +0.89 |
| Fill missing seed-3 changes | +2.65 | — |
| Add half the seed-3 change | +3.54 | −2.69 |
| Revert changes opposed to seed 3 | +3.76 | −0.79 |

On two additional reasoning panels totaling 80 MATH500 and 108 BBH items, the late-block edit tied its parent on MATH500 and gained nine correct BBH answers. The recurrent-content edit gained one on each task. Neither edit met the prespecified two-cohort reasoning criterion. On a third 18-task panel of 332 items, late-block scored 50.55 against 50.77 for its parent. The paired 90% interval for the difference was −2.26 to +1.90 points. These local results use sampled generation and have substantial uncertainty. [Routing](results/decision.json) · [Combined edit](union/results/decision.json) · [Two-cohort reasoning results](replication/results/decision.json) · [Late-block generalization](generalization/results/decision.json) · [Seed-3 edits](residual/results/decision.json)

## Reproduction

With the project CLI, Torch and safetensors installed, run the following from this research directory. The builder checks every official file in each model directory, reconstructs the edit from those inputs and verifies the exact manifest and payload hashes against the registered build record.

```sh
models=/path/to/models
delta model pull d0120ae6-20e2-4cd8-a73e-136470311df5 --output "$models/d0120ae6"
delta model pull 96b67023-ce48-4cfa-be8c-8db86694f154 --output "$models/96b67023"
python reproduce.py "$models" protocol.json late-block /path/to/new-submission
delta validate /path/to/new-submission
```

On the Mac evaluation bed, the paired local comparison uses the pinned helpers from the previous experiment at local commit `e0b30ba` (same helper files in [PR 108](https://github.com/ilblackdragon/deltaleague/pull/108)). The baseline cache contains reports from the actual d0120ae6 checkpoint. The runner checks the model and suite hashes, sample IDs, generation settings and raw-report receipts before reuse. Both models generate answers independently for the fresh panel. The official configuration is `private1000-v6-thinking8k-20261111`. The 18-task panels use seeds 20261111 and 20261112, the fresh seed is recorded per protocol, and all use thinking enabled with 8192 tokens, temperature 1, top-p 0.95, top-k 20 and presence penalty 1.5.

```sh
source /Volumes/T9/ai-models/deltaleague/public-eval/env.sh
export DELTA_DATASET_SOURCES="$(cat "$DELTA_EVAL_ROOT/dataset-sources.json")"
run_dir=/Volumes/T9/ai-models/deltaleague/delta-experiments/new-routing-run
delta_bin=/Users/jc/delta/deltaleague/.venv/bin
"$DELTA_EVAL_ROOT/.venv/bin/python" study.py fresh "$run_dir"
"$delta_bin/python" study.py build "$run_dir"
for submission in "$run_dir"/*/submission; do
    "$delta_bin/delta" validate "$submission" > "$(dirname "$submission")/validation.log" || exit 1
done
"$DELTA_EVAL_ROOT/.venv/bin/python" -u study.py evaluate "$run_dir"
```
