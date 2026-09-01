# Dual-Modal Mamba U-Net

## Inputs

- Modality A: RGB, shape `[B, 3, H, W]`
- Modality B: SAR, shape `[B, 1, H, W]`
- Output: five-class logits, shape `[B, 5, H, W]`

## Encoder

The encoder has two fully independent branches. Patch embedding, stage
downsampling, AS6 blocks, output projections and FFNs do not share weights.

| Stage | Policy | Channels | Resolution | A routes | B routes |
|---|---|---:|---:|---|---|
| 1 | Self | 96 | H/4 | a1,a2,a3,a4 | b1,b2,b3,b4 |
| 2 | Cross | 192 | H/8 | a1,a2,b3,b4 | b1,b2,a3,a4 |
| 3 | Self | 384 | H/16 | a1,a2,a3,a4 | b1,b2,b3,b4 |
| 4 | Cross | 768 | H/32 | a1,a2,b3,b4 | b1,b2,a3,a4 |

Each path passes through its branch-specific Adaptor-S6. The four restored 2-D
features are summed element by element. Stage 3 consumes the branch outputs of
Stage 2, so it preserves previously learned cross-modal information while adding
no new cross-modal scan routes.

Adaptor-S6 contains:

1. A selective S6 recurrence implemented by the selective-scan kernel from
   `vmamba.py`.
2. A content-dependent memory adaptor over configurable causal history offsets.
3. A depthwise-separable spatial adaptor after inverse directional mapping.

## Fusion and decoder

An independent Multimodal Learnable Fusion Module (MLFM) follows every stage
and creates `S1`, `S2`, `S3`, `S4`. Each MLFM first uses independent 1x1
convolutions to align the two modalities.
It then computes an Add branch and a Cat branch; each branch uses convolution to
refine the feature and restore the requested output channels. Two independent,
unconstrained learnable scalars combine the results:

```text
F_add = Conv(A' + B')
F_cat = Conv(Cat(A', B'))
F = a * F_add + b * F_cat
```

Both `a` and `b` are initialized to 0.5.

The U-Net decoder is symmetric with the four encoder stages. `S4` first passes
through a stride-2 bottleneck to produce `S4'`; the original `S4` remains a skip
connection. The decoder dataflow is:

```text
S4' = Bottleneck(S4)       # H/64
D1  = Decoder1(S4', S4)   # H/32
D2  = Decoder2(D1, S3)    # H/16
D3  = Decoder3(D2, S2)    # H/8
D4  = Decoder4(D3, S1)    # H/4
Y   = SegmentationHead(D4) -> upsample to H x W
```

Each of `Decoder1` through `Decoder4` contains three serial ResNet basic blocks.
The first residual block projects the concatenated skip feature to the target
stage channels; the next two blocks preserve the channel count. GroupNorm is
used instead of BatchNorm to remain stable with small remote-sensing batches.

## Shape check

Run the complete default model on the GPU:

```bash
python tools/check_model.py --device cuda
```

For a lightweight CPU-compatible structural check:

```bash
python tools/check_model.py --device cpu --image-size 64 --light
```
