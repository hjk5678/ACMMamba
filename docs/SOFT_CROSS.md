# Learnable Soft Cross

New experiments use **CCCC + Soft Cross + MLFM + the original ResNet U-Net decoder**.
Each decoder stage retains three residual blocks. The encoder uses depths
**[2, 2, 4, 2]**; each stage serially stacks complete dual-branch scan blocks.
Each block re-scans its inputs and performs soft exchange before its independent
AS6 processors. AS6 internals and MLFM are unchanged; MLFM runs once at each
stage's end, not once per block.

For each encoder block, let `alpha3, alpha4` be the fractions imported from B
into A, and `beta3, beta4` the fractions imported from A into B:

```text
A3' = (1 - alpha3) * A3 + alpha3 * B3
A4' = (1 - alpha4) * A4 + alpha4 * B4
B3' = (1 - beta3)  * B3 + beta3  * A3
B4' = (1 - beta4)  * B4 + beta4  * A4
```

All right-hand sides use the original paths, before either branch is updated.
Paths 1 and 2 are unchanged. Routing happens before the branch-specific AS6
processors and their inverse-direction mapping and elementwise sum.

Each fraction is `sigmoid(logit)` with its own trainable scalar. These are
global learned parameters, not input-dependent gates and not channelwise maps.
Zero approaches self-routing; one approaches the original hard exchange.
The initialization is **0.5**, configurable via `soft_cross_init` in `(0,1)`.
At initialization, A/B paths 3 and 4 contain the same average, but the branch
AS6 weights and residual paths remain independent. Coefficients can subsequently
diverge. No cross-branch, cross-direction, cross-block or cross-stage tying is used.

With `depths: [2,2,4,2]`, there are 10 dual-branch blocks and **40 exchange scalars**.
Every block contains four AS6 processors per modality: 80 independent AS6
instances in total (40 per modality). The earlier [1,1,1,1] setting had 16
exchange scalars.
Deeper stages allocate another 4 scalars per additional cross block. The logits
are 1-D parameters and the existing AdamW grouping applies no weight decay.

## Configuration and backward compatibility

```yaml
model:
  depths: [2, 2, 4, 2]
  stage_modes: [cross, cross, cross, cross]
  cross_mode: soft
  soft_cross_init: 0.5
  fusion_mode: mlfm
  decoder_type: unet
  decoder_blocks_per_stage: 3
```

Existing configurations still default to `cross_mode: hard`; no logits are
created for hard routing or self stages, so old checkpoint keys are unchanged.
The new standalone configurations preserve each dataset's split, preprocessing,
loss and training settings, and start from scratch (`resume: null`):

- `configs/ablations/train_kust4k_encoder_cccc_soft.yaml` (RS-RGB-T)
- `configs/ablations/train_korea_no_cloud_encoder_cccc_soft.yaml` (Dongying alias)
- `configs/ablations/train_bjroad_encoder_cccc_soft.yaml`

Logs and weights use `ablations/<dataset>/encoder/cccc_soft_2242/`; old experiments
are not overwritten. Checkpoints include the learned logits and model config.
Use the corresponding soft config when resuming or running `infer.py`.
Do not resume a hard-exchange checkpoint as a soft model for this ablation.
The earlier soft [1,1,1,1] checkpoint also cannot be resumed directly as [2,2,4,2].
The Python model defaults and all non-soft experimental configs stay unchanged
for compatibility with existing weights.

Every epoch records coefficients in `train.log`, `history.jsonl` and TensorBoard.
The order is **A3, A4, B3, B4**, always interpreted as the fraction imported
from the other modality. `model.encoder.soft_cross_weights()` exposes the values.

## Server checks and launch (RS-RGB-T example)

```bash
cd /data/BUAS/HJK/ACMMamba
python tools/check_soft_cross.py --device cuda --amp
python train.py --config configs/ablations/train_kust4k_encoder_cccc_soft.yaml --smoke-test
bash tools/start_ablation.sh configs/ablations/train_kust4k_encoder_cccc_soft.yaml
tail -f logs/ablations/kust4k/encoder/cccc_soft_2242/train.log
```

The isolated CPU test covers formula correctness, untouched paths, gradients,
soft limits, independent blocks, old routing, full small-model forward/backward,
optimizer updates and checkpoint round trips. GPU/AMP should also be checked on
the server. A missing-Triton import guard in `vmamba.py` allows actual PyTorch
reference-backend CPU tests without changing the Triton kernel computation.

This is a learnable-exchange experiment, not a guaranteed accuracy improvement.
To isolate the exchange effect, compare against **hard CCCC + MLFM** with the
same [2,2,4,2] depths and training settings. Comparing to older [1,1,1,1] results
changes both model capacity and routing. Extra depth increases training cost
and memory requirements; run a full-size server smoke test before launch.
