# HG-MLFM late-training numerical stability

The observed run first logged a NaN loss at epoch 94, still validated normally
at epoch 120, and had non-finite model parameters by epoch 149. This establishes
numerical failure, not the first offending operator. No historical failing
batch/optimizer state was available locally to prove that operator's identity.

## Changes

- `mlfm_hg_options.force_fp32: true` disables autocast for the entire HG-MLFM:
  reduce, HGConv, restore, residual addition, channel alignment, Add/Cat branches
  and scalar mixing. Its output stays FP32. The rest of the model still uses AMP.
  Parameters must remain FP32, as in the training script; do not call model.half().
  No graph formula, residual identity, gamma constraint, initialization or loss
  weighting changes. Default false preserves old configs and weight keys.
- Training checks input, logits and loss before backward, with rank-wide failure
  agreement. Errors identify epoch, batch step, sample IDs and invalid tensor.
- After unscale, any rank's invalid gradients skip the update on all ranks. Scale
  backs off together; finite-gradient norm overflow and invalid clipped gradients
  abort before stepping. Actual optimizer post-step execution drives scheduling,
  not `scale_after >= scale_before`. Parameters and optimizer moments are checked
  after every successful update, before scheduler advancement.
- The stable config starts scale at 1024; stops at scale below 2^-16 or 8
  consecutive gradient-overflow skips. Scientific-notation logs do not round
  small scales to zero. Forward NaNs are fatal, not repeatedly skipped.
- Validation rejects non-finite inputs/logits/loss. Because distributed eval
  shards can have unequal lengths, failure agreement happens after local loops,
  before metric synchronization. A failed validation publishes no scores and
  cannot select best weights. The metrics helper also rejects non-finite logits.
- Inference validates weights, inputs, logits and losses before predictions are
  saved. Resume and checkpoint writes reject non-finite model/optimizer state;
  invalid best scores are not saved. Existing valid checkpoints are retained.

There is no nan_to_num, hidden clipping of model features, automatic precision
switch for the entire network, or automatic resumption from damaged last.pt.
Guards add synchronization/scanning cost. FP32 fusion may increase memory/time;
use the full-size four-GPU smoke test. These safeguards prevent silent bad
training, but short tests cannot guarantee a 150-epoch run will never diverge.

## Isolated rerun

The original config and best weight remain intact. New config:
`configs/train_kust4k_mlfm_hg_stable.yaml`, with the same 150-epoch schedule,
data/loss, four three-block ResNet decoder stages and HG residual design.
Only FP32 fusion, scaler safety settings and output paths differ.

```bash
cd /data/BUAS/HJK/ACMMamba
python tools/check_numerical_safety.py --device cuda --ddp
python tools/check_mlfm_hg.py --device cuda --amp --force-fp32 --full-model
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 train.py --config configs/train_kust4k_mlfm_hg_stable.yaml --smoke-test
# After checks pass, start a fresh run. Do not resume the damaged last.pt.
bash tools/start_ablation.sh configs/train_kust4k_mlfm_hg_stable.yaml
tail -f logs/experiments/kust4k/mlfm_hg_resnet3_fp32/train.log
```

New checkpoint directory: `checkpoints/experiments/kust4k/mlfm_hg_resnet3_fp32`.
Old best weights may be evaluated with their original config as before. A model-
only best weight is NOT a full optimizer/scheduler/scaler resume checkpoint.

## Regression coverage

Failure injection covers: a constructed large-finite-weight FP16 HG-MLFM overflow
that stays finite in FP32; output/formula and gradient checks; actual optimizer
and LR progress; backward overflow skip and recovery; zero scale; repeated skips;
overflow of gradient norm despite individually finite gradients; poisoned Adam
moments; metrics rejecting NaN; checkpoint refusal without overwriting best;
partial gradient accumulation. Two CPU/Gloo ranks exercise a one-rank bad forward,
uneven-length bad validation and one-rank gradient overflow followed by recovery.
The constructed overflow is a targeted regression, not a reproduction of the
unknown original epoch-94 sample.
