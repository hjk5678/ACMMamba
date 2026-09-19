# Residual-Hypergraph Dual-Branch Decoder (RHDB)

RHDB is optional (`model.decoder_type: rhdb`). The default `unet` decoder
remains available with unchanged legacy state keys.
No encoder, MLFM, label, loss or segmentation-head changes are required.

## Stage naming and placement

The request names decoder stages by their corresponding encoder scale; this
project numbers decoder stages in execution order (deep to shallow):

| Request | Project | Inputs | Default block |
|---|---|---|---|
| D4 | decoder1 | Bottleneck(S4), S4 | RHDB |
| D3 | decoder2 | decoder1 output, S3 | RHDB |
| D2 | decoder3 | decoder2 output, S2 | Original ResNet UpBlock, 3 blocks |
| D1 | decoder4 | decoder3 output, S1 | Original ResNet UpBlock, 3 blocks |

RHDB's local branch defaults to **one** BasicBlock as specified in this design,
independently of `decoder_blocks_per_stage: 3` for the shallow original blocks.
Set `rhdb_options.num_residual_blocks: 3` to retain a deeper local branch.

## Computation

1. Bilinear upsample deep feature to the exact skip size; concatenate and use
   a 1x1 convolution to obtain F with the desired output channels.
2. In parallel, feed the SAME F to:
   - a local residual branch (3x3/GN/GELU/3x3/GN + identity, then GELU);
   - a region hypergraph branch (never the local branch's output).
3. Hypergraph: 1x1 projection to Ch (default max(1,min(64,C//2))), adaptive
   average pooling, flatten to [B,N,Ch]. The grid is clamped separately to H/W:
   `(min(H,gh),min(W,gw))`, so small maps are not expanded to fake extra regions.
   Cell-center coordinates lie in [0,1], in matching row-major order.
4. Each sample independently computes `alpha*(1-cosine) + (1-alpha)*spatial_L2`.
   A center plus K OTHER nearest nodes forms one hyperedge. K is clamped to N-1
   and every center is included explicitly. Hard top-k topology is detached;
   the differentiable node features are not detached from propagation.
5. H is [B,vertex,hyperedge], built with topk/scatter, not adjacency. With W=I,
   propagation is `Dv^-1/2 H De^-1 H^T Dv^-1/2 X`, implemented as two batched
   matrix products and degree-vector broadcasting, without constructing G or
   dense diagonal matrices. Degree denominators are clamped by eps.
6. Apply one Linear(Ch,Ch) and GELU, then form ONLY the node increment
   `delta_nodes = gamma * HGConv(X)`, with no `X + ...` residual. Gamma is an
   unconstrained learnable scalar, initially 0. Reshape, bilinear upsample and
   project Ch back to C with a **bias-free** 1x1 convolution to yield delta_hg.
7. Keep the existing local ResidualStage unchanged, but use
   `delta_local = local_branch(F) - F`. This is its complete net change,
   including post-add GELU and any stacked blocks, not just the convolution path.
8. Default spatial gate: `A=sigmoid(Conv1x1(cat(delta_local,delta_hg)))`,
   [B,1,H,W]. Gate weights start at zero and bias at -2, so initially A=0.119203.
   `update=delta_local+A*delta_hg`: local is never multiplied by (1-A).
   The output is `Post(F+update)`, where Post is 3x3 Conv / GN / GELU.
   The outer identity is added only once; no full local or pooled node identity
   is added again.

KNN, pooling, degree normalization, propagation, HG linear and node increment
run in FP32 under an autocast-disabled region. Convolutions can use outer AMP.
No external graph libraries, per-pixel NxN graph, extra loss, multi-hypergraph,
boundary module or additional attention module are introduced.

At gamma=0, the entire HG increment is exactly zero for finite inputs, because
the restoration convolution has no bias. HGConv, reduction/restoration and
gate gradients are initially zero (not unused/None), but gamma can receive a
gradient. Once gamma changes these parameters can learn. There is no conditional
early return at gamma=0, so autograd retains all enabled parameter paths.
In gate/add mode the initial result is mathematically Post(local_branch(F)).
This is NOT identical to the original UpBlock architecture.
The fusion gate is learned from content; its spatial interpretation (boundaries
vs regions) is a design intention, not a guaranteed learned behavior.

## Options and ablations

```yaml
model:
  decoder_type: rhdb
  decoder_blocks_per_stage: 3  # shallow original UpBlocks
  rhdb_options:
    hypergraph_stages: [1, 2]  # project deep-to-shallow numbering
    use_hypergraph: true
    fusion_mode: gate         # gate / add / concat
    k: 8                     # excludes center: normal edge degree is 9
    node_grid: 16             # or [height, width]
    alpha: 0.7                # fixed semantic-distance weight
    hypergraph_hidden_dim: 64 # omit for min(64,C//2), clamped to >=1
    gamma_init: 0.0
    eps: 1.0e-6
    num_residual_blocks: 1
    dropout: 0.0
```

`add` means delta_local+delta_hg. `concat` uses a learned 1x1 projection of
cat(delta_local,delta_hg) from 2C to C. `gate` means delta_local+A*delta_hg.
All three use Post(F+update). Concat is an ablation: its learned projection
(including bias) need not preserve delta_local when gamma=0, unlike gate/add.
These are separate from encoder
`model.fusion_mode: mlfm`.

`use_hypergraph: false` removes HG and fusion-gate parameters entirely, so no
unused parameters are introduced under DDP. It retains RHDB's input projection,
local branch and post convolution: Post(F+(local_full-F)) = Post(local_full),
up to floating-point cancellation rounding. This is an intra-RHDB ablation,
NOT the original UpBlock. Use `decoder_type: unet` for the original baseline;
`hypergraph_stages: []` also leaves all four stages as original UpBlocks.

Gamma values are logged each epoch in train.log, history.jsonl and TensorBoard.
`model.decoder.hypergraph_gammas()` returns detached values for enabled stages.
Weights include HG projections, gamma, gate and the model configuration; use
the same RHDB config for inference. Do not directly resume a ResNet checkpoint
as RHDB: the deep decoder architecture and state keys differ.

The old full-feature RHDB checkpoints are also not interchangeable with this
increment version: restoration bias was removed and fusion semantics changed.
Do not suppress strict loading errors to resume them. Train a new experiment;
the original `unet` decoder and its strict checkpoint compatibility are unchanged.

## RS-RGB-T experiment template

`configs/ablations/train_kust4k_rhdb_delta_soft_once_2242_150e.yaml` preserves the
RS-RGB-T official split, RGB/TIR normalization, nine classes, loss, optimizer,
soft-once CCCC encoder with depths2242, and 150-epoch training schedule.
Only the decoder and output locations change. This template is not auto-launched.

```bash
cd /data/BUAS/HJK/ACMMamba
python tools/check_rhdb.py --device cuda --amp --full-model
python tools/check_rhdb_ddp.py
python train.py --config configs/ablations/train_kust4k_rhdb_delta_soft_once_2242_150e.yaml --smoke-test
```

After successful full-resolution checks, launch if desired:

```bash
bash tools/start_ablation.sh configs/ablations/train_kust4k_rhdb_delta_soft_once_2242_150e.yaml
tail -f logs/ablations/kust4k/decoder/rhdb_delta_soft_once_2242_150e/train.log
```

Weights: `checkpoints/ablations/kust4k/decoder/rhdb_delta_soft_once_2242_150e/`.
Prefix: `kust4k_rhdb_delta_soft_once_2242_150e`. No existing experiment is overwritten.

Local reduced-channel CPU checks cover explicit HGNN numerical equivalence,
incidence orientation, batch isolation, K clipping, singleton/zero nodes,
gamma learning, parallel branches, all fusion modes, HG disabling, full decoder
2/5/9-class gradients, legacy decoder equality, and full encoder+decoder weight
round trips. The dedicated DDP check uses two CPU/Gloo ranks and three updates
per mode, with find_unused_parameters=False, gradient and synchronization checks.
It requires a working Gloo transport (the local Windows runtime failed Gloo
initialization); all six mode/HG combinations passed on the user's Linux server.
CPU BF16 checks additionally confirm
FP32 graph/HG operations under autocast; they do not validate CUDA FP16 kernels.
Server CUDA/AMP, NCCL multi-GPU and actual dataset-memory checks remain necessary.

The AMP regression test uses GradScaler, unscales parameter gradients before
inspection, and runs backward outside autocast, as in training. Its two-step
HG gradient probe uses a fixed random spatial target and prints gamma, unscaled
HGConv gradient L1 and the loss scale. It keeps gamma initialized to zero and
retains the nonzero-gradient assertion after the first optimizer update.
Autocast without loss scaling can underflow tiny gradients just after gamma
leaves zero; that incomplete AMP test is not representative of train.py.

## Four-stage RHDB experiment

`configs/ablations/train_kust4k_rhdb4_delta_soft_once_2242_150e.yaml` changes
`hypergraph_stages` to `[1, 2, 3, 4]` and uses separate `rhdb4_delta_soft_once_2242_150e`
log/checkpoint directories. All other experiment settings remain unchanged,
including the 150 epochs, once-per-stage soft CCCC encoder, 2242 depths, and
one local ResidualBlock in each RHDB. Each stage has independent branch/gate/gamma
parameters. `decoder_blocks_per_stage: 3` only applies to legacy UpBlocks;
with RHDB at all four stages it does not override `num_residual_blocks: 1`.

The bottleneck and segmentation head are unchanged. Each HG branch still pools
to at most 16x16 nodes; shallow branches restore updates to their skip resolution.
This changes the shallow local architecture as well as adding HG there, so it is
an RHDB-placement comparison, not an isolated HG-only ablation. Start from scratch,
not from the two-stage RHDB checkpoint. Actual GPU memory/runtime require a smoke test.

```bash
python tools/check_rhdb.py --device cuda --amp --full-model --all-rhdb-stages
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 train.py --config configs/ablations/train_kust4k_rhdb4_delta_soft_once_2242_150e.yaml --smoke-test
# After the smoke test passes:
bash tools/start_ablation.sh configs/ablations/train_kust4k_rhdb4_delta_soft_once_2242_150e.yaml
```

## Baseline on Korea/no_cloud

The two-stage increment RHDB is the selected baseline; four-stage RHDB remains
a decoder-placement ablation. `configs/train_korea_no_cloud_baseline_rhdb.yaml`
uses the same baseline architecture as the RS-RGB-T two-stage experiment, with
RGB3/SAR1 inputs and five output classes. Existing Korea/no_cloud split files,
normalization, augmentation defaults, class frequencies and training settings
are preserved from its 150-epoch soft-once/2242 experiment. No split is regenerated.

Train from scratch (`resume: null`); RS-RGB-T weights are not used. Output paths:

- Logs: `/data/BUAS/HJK/ACMMamba/logs/baseline/korea_no_cloud`
- Checkpoints: `/data/BUAS/HJK/ACMMamba/checkpoints/baseline/korea_no_cloud`
- Best weight: `korea_no_cloud_baseline_rhdb_best_miou.pt`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 train.py --config configs/train_korea_no_cloud_baseline_rhdb.yaml --smoke-test
# After the smoke test passes, train 150 epochs, validating every 10:
bash tools/start_ablation.sh configs/train_korea_no_cloud_baseline_rhdb.yaml
# Only after training, evaluate the best validation checkpoint on the test set:
CUDA_VISIBLE_DEVICES=0 python infer.py --config configs/train_korea_no_cloud_baseline_rhdb.yaml --split test --device cuda:0 --batch-size 1 --max-comparisons 50 --output-dir results/baseline/korea_no_cloud/test
```

## Baseline on BJRoad

`configs/train_bjroad_baseline_rhdb.yaml` uses the same two-stage RHDB baseline
and 150-epoch schedule, with RGB3/GPS1 inputs and two output classes. It retains
all data/loss/visualization settings from `configs/train_bjroad.yaml`, including
the existing train/val split, separate test directories, filename templates,
normalization, disabled geometric augmentation and label threshold 128. No input
resizing/cropping is added. Training starts from scratch; old checkpoints remain
untouched. Logs and weights are under `logs/baseline/bjroad` and
`checkpoints/baseline/bjroad`; the prefix is `bjroad_baseline_rhdb`.

BJRoad's previous shallow-encoder experiment already used about 43 GiB on the
server. The new 2242 encoder changes activation memory, so run the real-input
four-GPU smoke test before launching. Gradient accumulation does not reduce
the memory needed by an individual sample. If it runs out of memory, diagnose
that first rather than silently changing input resolution or baseline depths.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 train.py --config configs/train_bjroad_baseline_rhdb.yaml --smoke-test
# After it passes:
bash tools/start_ablation.sh configs/train_bjroad_baseline_rhdb.yaml
# After training finishes:
CUDA_VISIBLE_DEVICES=0 python infer.py --config configs/train_bjroad_baseline_rhdb.yaml --split test --device cuda:0 --batch-size 1 --max-comparisons 70 --output-dir results/baseline/bjroad/test
```
