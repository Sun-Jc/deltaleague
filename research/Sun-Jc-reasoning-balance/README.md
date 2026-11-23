# Joint reasoning and general transfer

Compare middle-to-late attention concentration and a third GRPO seed blend against official version 917d6b4f-fea3-432c-9264-e4f89e45890a. The two attention edits scale the effective query RMSNorm gain on full-attention layers 11, 15 and 19. The seed blend uses weights 1/2, 1/4 and 1/4 for eff1e1b7, a6eb9f13 and fdfd46a4 on layers 0–22. All edits retain the leading model's last layer, token embedding and output normalization. The inputs are public versioned checkpoints. There is no task-specific fitting.

The selection and replication rules are in `protocol.json`. Every candidate receives the same 18 native local benchmark groups. The scorer measures MATH500 with math verification and BBH with its native filtered exact match. A development group has eight mathematics questions or one question per each of 27 BBH tasks. A complete score averages the 18 benchmark scores equally. The previous leader run is reused only after reconstructing its scores from native reports and checking all model, suite and runtime identities.

## Reproduction

Use the T9 Python, MLX and evaluation environments from the Mac evaluation guide. The laboratory batch server, baseline verifier and analysis come from the GRPO calibration branch at commit 450bf5a. Retain the previous verified leader run, the disjoint suite and the pinned native data on T9. Run from this research folder with a new directory under delta-experiments.

```sh
root=/Volumes/T9/ai-models/deltaleague
exp="$root/delta-experiments"
out="$exp/reasoning-balance-reproduction"
bin=/Users/jc/delta/deltaleague/.venv/bin
mkdir "$out"
"$bin/delta" model pull fdfd46a4-3a55-4277-91c1-70c739d6139b --output "$exp/models/fdfd46a4"
for arm in query-sharp query-soft seed-three; do
    "$bin/python" build.py "$out" "$arm" || exit 1
    "$bin/delta" validate "$out/$arm/submission" > "$out/$arm/validation.log" || exit 1
done
source "$root/public-eval/env.sh"
export DELTA_DATASET_SOURCES="$(cat "$DELTA_EVAL_ROOT/dataset-sources.json")"
"$DELTA_EVAL_ROOT/.venv/bin/python" -u evaluate.py "$out"
```
