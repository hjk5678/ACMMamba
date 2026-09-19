# Shallow MSCAN decoder experiment

This is a configurable replacement of **decoder3.refine and decoder4.refine**,
not a change to the established baseline. Project stage numbers run from deep
to shallow. Decoder1/2 remain the increment-only RHDB; encoder, MLFM, bottleneck,
skip connections, upsampling, UpBlock.reduce and segmentation head are unchanged.

| Stage | Deep input | Skip | After original reduce + concat | Refinement output |
|---|---|---|---|---|
| Decoder3 | D2, 384 channels | S2, 192 channels | 384 channels | MSCAN depth2, 192 channels, H/8 x W/8 |
| Decoder4 | D3, 192 channels | S1, 96 channels | 192 channels | MSCAN depth2, 96 channels, H/4 x W/4 |

The standalone MSCANStage only changes channels, never spatial size:

```
deep -> existing bilinear resize -> existing Conv1x1 reduce
                                     + skip -> concatenate
                                                |
                         MSCANStage: Conv1x1(no bias) + GN
                                                |
                                     MSCANBlock x 2
                                                |
                                  same-resolution output
```

## Exact sublayer equations

For spatial attention, all three strip-convolution branches take the SAME 5x5
depthwise output, rather than being applied successively:

```
base = DW5x5(x)
attn = Conv1x1(base + DW7x1(DW1x7(base))
                   + DW11x1(DW1x11(base)) + DW21x1(DW1x21(base)))
SpatialAttention(x) = x * attn               # no sigmoid / softmax
Attention(x) = x + Proj2(SpatialAttention(GELU(Proj1(x))))

MLP(x) = Dropout(Conv1x1(Dropout(GELU(DW3x3(Conv1x1(x))))))
x1 = x  + DropPath(gamma1 * Attention(GN1(x)))
y  = x1 + DropPath(gamma2 * MLP(GN2(x1)))
```

Both gamma parameters have shape [C], broadcast as [1,C,1,1], initialized to
0.01. The Attention internal shortcut AND the two Block residuals are retained
as requested; they are different from the RHDB increment fusion design.
Everything stays NCHW, and all convolutions have stride1 with shape-preserving
padding. No Linear, BatchNorm, channel-last LayerNorm or extra attention is added.

`make_group_norm` selects the largest divisor of C no larger than 32. Its
GroupNorm subclass handles the degenerate B=H=W=1, one-channel-per-group case
with the mathematical zero-normalized result plus affine bias, maintaining
autograd and avoiding PyTorch's singleton-input rejection. Ordinary inputs use
the normal GroupNorm implementation. The existing encoder DropPath was moved
unchanged into `common_layers.py` and re-exported from `encoder.layers`, so both
modules share the same pure-PyTorch implementation without importing VMamba just
to use standalone MSCAN. Encoder computation and parameter keys are unchanged.

## Configuration

```
model:
  decoder_type: rhdb
  rhdb_options:
    hypergraph_stages: [1, 2]
    # Keep the other baseline RHDB options.
  mscan_options:
    depth: 2
    mlp_ratio: 4.0
    drop: 0.0
    drop_path: 0.0
    layer_scale_init_value: 1.0e-2
```

Omitting `mscan_options` (or setting it to null) preserves the old decoder and
strict state-dict compatibility. An empty dict enables MSCAN with defaults.
The option only applies to shallow stages3/4 and requires RHDB stages1/2;
conflicting all-four-RHDB or unknown options are rejected. `depth` applies
equally to both shallow stages. `decoder_blocks_per_stage` still controls old
ResNet UpBlocks, not MSCAN depth or RHDB local depth.

The new experiment uses drop=0 and drop_path=0 per the requested initial design.
They are independent of encoder drop_path_rate and existing decoder_dropout
(which still governs unchanged parts, e.g. bottleneck). Compared with the old
three-block shallow ResNet, module family, depth and local regularization differ;
describe it as a decoder-refinement replacement, not an isolated attention test.

## BJRoad experiment (not launched automatically)

`configs/train_bjroad_rhdb_mscan.yaml` keeps all BJRoad baseline data, label,
optimizer and 150-epoch settings. Only shallow MSCAN options and output paths
differ. Start from scratch; baseline shallow ResNet checkpoint weights cannot
be strictly loaded into the new MSCAN refinement. Do not bypass this with
strict=False and present the result as a valid resumed baseline run.

```
logs/experiments/bjroad/rhdb_mscan/
checkpoints/experiments/bjroad/rhdb_mscan/bjroad_rhdb_mscan_best_miou.pt
```

Sync all changed files, including `common_layers.py`, then run on the server:

```bash
cd /data/BUAS/HJK/ACMMamba
python tools/check_mscan.py --device cuda --amp --full-model
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 train.py --config configs/train_bjroad_rhdb_mscan.yaml --smoke-test
# Only after successful memory/DDP checks:
bash tools/start_ablation.sh configs/train_bjroad_rhdb_mscan.yaml
```

The tool checks exact attention/residual formulas, per-sample stochastic depth,
the requested [2,384,64,64] -> [2,256,64,64] example (1,624,576 parameters),
odd/prime-channel/singleton inputs, finite and present gradients, 2/5/9-class
decoders, unchanged non-refinement state shapes and old baseline outputs.
`--full-model` also exercises a reduced-channel full soft-cross2242 encoder,
decoder and strict config/weight round trip. CUDA AMP uses GradScaler and
unscales parameter gradients before inspection. Full-resolution server memory
and NCCL behavior must still be checked; fewer parameters do not guarantee
lower activation memory with large MLP feature maps.

## RS-RGB-T experiment (current dataset choice)

Use `configs/train_kust4k_rhdb_mscan.yaml` instead of the BJRoad configuration.
It preserves all data/label/normalization, augmentation, loss and training
settings of the two-stage RHDB RS-RGB-T baseline: RGB3/TIR3, nine classes,
original split files, 150 epochs and validation every 10 epochs. Only MSCAN
options for stages3/4 and output paths differ. Existing baseline/BJRoad configs
are retained, and the run starts from scratch.

```bash
cd /data/BUAS/HJK/ACMMamba
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 train.py --config configs/train_kust4k_rhdb_mscan.yaml --smoke-test
# After the smoke test passes:
bash tools/start_ablation.sh configs/train_kust4k_rhdb_mscan.yaml
tail -f logs/experiments/kust4k/rhdb_mscan/train.log
```

Best weight: `checkpoints/experiments/kust4k/rhdb_mscan/kust4k_rhdb_mscan_best_miou.pt`.
