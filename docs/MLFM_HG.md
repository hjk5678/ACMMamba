# MLFM with independent pre-fusion hypergraph refinement

`fusion_mode: mlfm_hg` adds two independent, single-HGConv refinements to each
encoder-stage MLFM. Original `fusion_mode: mlfm` remains unchanged, including
parameter names and strict checkpoint loading. Decoder hypergraphs are separate.

## Actual forward

For each stage's A and B features, before MLFM spatial/channel alignment:

```
A: [B,Ca,H,W] -> HG_A -> delta_A: [B,Ca,H,W] -> A' = A + delta_A
B: [B,Cb,h,w] -> HG_B -> delta_B: [B,Cb,h,w] -> B' = B + delta_B
A', B' -> original MLFM spatial alignment and independent 1x1 projections
       -> Add -> original FusionConv -> F_add
       -> Cat -> original FusionConv -> F_cat
S = a * F_add + b * F_cat
```

The encoder normally supplies equal channels and resolutions. For general MLFM
inputs, B is still resized to A's size after its own HG refinement. HG_A and HG_B
have independent weights and independently constructed per-image graphs. All
four stages use the new module: eight HGConv layers total, one per modality per
stage, never stacked within one branch. This does not add another cross-modal
graph. Prior CrossMamba interactions may already be present in the input features.

Only fusion/skip outputs S1-S4 change. The A/B features propagated to the next
encoder stage are unchanged; S4 still enters both bottleneck and Decoder1 skip.

## Reused hypergraph implementation

Reuse `decoder/hypergraph.py` without changing its mathematics or RHDB behavior:

1. Independent 1x1 channel reduction (experiment hidden dimension 64).
2. Adaptive region pooling to at most 16x16 nodes; each axis is clamped to the
   input size. This is a region-node graph, not a full-resolution pixel graph.
3. Per-image distance `0.7*(1-cosine) + 0.3*spatial`, normalized square-grid
   coordinates as in the existing branch; original aspect ratio is not modeled.
4. Each edge contains its center plus 8 OTHER nearest nodes, or all available
   nodes if there are fewer than 9. Self distances are excluded before top-k.
5. One propagation `Dv^-1/2 H De^-1 H^T Dv^-1/2 X`, then Linear and GELU;
   hyperedge weights are identity. Graph building and HG propagation run in FP32.
6. Multiply by independent learnable gamma, restore spatial resolution and
   channels, obtaining an increment. Add the original modality feature once.

Gamma defaults to 0, preserving the original fusion mapping at initialization
when the common fusion weights are the same. HG computations still execute.
The first update gives gamma a gradient, but HG internal weights have zero
gradients until gamma moves away from zero. They remain part of the autograd
graph (not unused DDP parameters). `gamma_init` is configurable. Gammas are
unconstrained scalars, distinct from MLFM's Add/Cat weights a and b.

## Configuration

```yaml
model:
  fusion_mode: mlfm_hg
  mlfm_hg_options:
    hypergraph_hidden_dim: 64
    node_grid: 16
    k: 8
    alpha: 0.7
    gamma_init: 0.0
    eps: 1.0e-6
```

Options on other fusion modes are rejected rather than silently ignored.
Training logs and history record `mlfm_hg` gamma_A/gamma_B for each stage;
`encoder.hypergraph_fusion_weights()` exposes these values for inspection.

`configs/train_kust4k_mlfm_hg.yaml` is a separate RS-RGB-T experiment based on
CCCC soft-once + 2242 + HG-MLFM + ResNet in all four decoder stages. Each stage
has three residual blocks (12 total); upsampling remains bilinear. No RHDB,
SAPA or MSCAN is enabled. HG remains only in the eight pre-fusion A/B branches.
It preserves the previous configuration's data split, normalization, loss,
150 epochs, validation every 10 epochs and accumulation=4. It starts from scratch
and saves to `logs/experiments/kust4k/mlfm_hg_resnet3` and
`checkpoints/experiments/kust4k/mlfm_hg_resnet3/kust4k_mlfm_hg_resnet3_best_miou.pt`.
The new output directory avoids mixing weights with the earlier RHDB variant.

Old MLFM checkpoints are not full HG-MLFM checkpoints: do not use them for strict
resume of the new architecture. Existing old configurations still load their
original weights strictly. New checkpoints load with their matching new config.

## Server verification and training

For the late-training NaN safeguards and FP32-fusion rerun, use
[NUMERICAL_STABILITY.md](NUMERICAL_STABILITY.md) and
`configs/train_kust4k_mlfm_hg_stable.yaml`. The original config below remains
available for reproducing the earlier experiment.

```bash
cd /data/BUAS/HJK/ACMMamba
python tools/check_mlfm_hg.py --device cuda --amp --full-model --ddp
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 train.py --config configs/train_kust4k_mlfm_hg.yaml --smoke-test
# After both checks pass:
bash tools/start_ablation.sh configs/train_kust4k_mlfm_hg.yaml
tail -f logs/experiments/kust4k/mlfm_hg_resnet3/train.log
```

The optional `--ddp` test uses two CPU/Gloo ranks and three updates. It does not
replace the full-size CUDA/NCCL smoke test. Full-model checks use reduced channels
to test config/weight round trips, backward and unchanged encoder propagation.
