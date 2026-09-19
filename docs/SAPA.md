# SAPA reference upsampling experiment

This implements the user's SAPA-B local affiliation specification in pure
PyTorch, not a binding to the official/custom CUDA operator. `decoder/sapa.py`
depends only on torch, torch.nn and torch.nn.functional. The default remains
bilinear, keeping all existing baseline and MSCAN experiment configs valid.

## Baseline integration

The new RS-RGB-T experiment starts from the selected **two-stage RHDB + shallow
three-block ResNet baseline**, not the MSCAN experiment. Only the four skip-guided
decoder upsamplers change. Each stage has independently learned projections.

| Project stage | Low-resolution decoder input | Query source (MLFM skip) | SAPA output channels |
|---|---|---|---|
| Decoder1 | Bottleneck(S4), 768 channels | S4, 768 channels | 768 |
| Decoder2 | D1, 768 channels | S3, 384 channels | 768 |
| Decoder3 | D2, 384 channels | S2, 192 channels | 384 |
| Decoder4 | D3, 192 channels | S1, 96 channels | 192 |

RHDB: `SAPA(skip, deep) -> concat(skip) -> original fuse / parallel branches / Post`.
Shallow UpBlock: `SAPA(skip, deep) -> original reduce1x1 -> concat(skip) -> original ResidualStage`.
SAPA replaces only the former bilinear operation, not the fusion stage. It never
adds skip values into decoder values and never performs a second interpolation.
Encoder, MLFM, bottleneck and segmentation head are unchanged. The final
segmentation-head bilinear resize stays in place because it has no paired skip.
Hypergraph's internal node-to-feature interpolation also stays unchanged.

## Exact computation

```
x: [B,Cx,H,W]      y: [B,Cy,sH,sW]
q = Conv1x1(ChannelLayerNorm(y))     # [B,D,sH,sW]
k = Conv1x1(ChannelLayerNorm(x))     # [B,D,H,W]
v = x                              # ORIGINAL values, no norm/projection
q = reshape(q, [B,D,1,H,s,W,s])
k_patch = reshape(unfold(k), [B,D,K*K,H,1,W,1])
v_patch = reshape(unfold(x), [B,Cx,K*K,H,1,W,1])
score = sum_D(q * k_patch)          # broadcast subpixel axes, no repeat
attention = softmax(score, dim=1)   # [B,K*K,H,s,W,s], neighbors sum to one
out = sum_KK(v_patch * attention[:, None])  # [B,Cx,H,s,W,s]
out = reshape(out, [B,Cx,sH,sW])
```

ChannelLayerNorm permutes NCHW -> NHWC -> LayerNorm(C) -> NCHW. There is no
BatchNorm, sigmoid, cosine similarity, sqrt(D) scaling, value projection or
output channel projection. The high-res pixel (r,c) uses the neighborhood around
low-res (r//s,c//s). `unfold` zero padding includes zero-key/zero-value neighbors
in the softmax, per the supplied reference. No validity mask or renormalization
is added: constant input values can attenuate at image edges. This is an explicit
reference boundary convention, not an assertion of exact official kernel parity.

Under AMP, projection convolutions may use FP16; similarity, softmax and value
aggregation use FP32 to limit overflow/underflow, and output is cast back to the
decoder input dtype. Double precision is retained for gradcheck. There is no
detach: gradients pass to both inputs and all projection/normalization weights.

Q/K 1x1 projections use `nn.init.trunc_normal_(weight, std=0.02)` and zero
biases, matching the Linear Q/K initialization in the
[official SAPA implementation](https://github.com/poppinace/sapa/blob/main/SAPA.py).
The PyTorch default truncation bounds (-2, 2) are retained; these are absolute
bounds, not +/- two standard deviations. LayerNorm retains weight=1, bias=0.
This changes fresh-model initialization only: loading an existing SAPA checkpoint
overwrites these values, with unchanged parameter names and shapes. Mathematical
equivalence to the previous repeat version is tested using identical weights.

## Shape and memory constraints

Input channels, batch size, device, floating dtype and exact scale are checked.
Odd kernel_size is mandatory; defaults are D=64, s=2, K=5. If the skip is not
exactly twice the decoder H/W, an informative ValueError is raised, with no
silent resize/crop fallback. With the extra bottleneck, input dimensions divisible
by 64 are sufficient for all four strict x2 transitions. RS-RGB-T 512x640 meets
this condition; arbitrary odd-size images may not.

Neighborhoods remain on the low-resolution grid: value patch storage is
B*Cx*K*K*H_low*W_low elements. With s=2 this is one quarter of the old repeated
patch storage. At RS-RGB-T Decoder4 (B=1,Cx=192,H_high=128,W_high=160,K=5),
this is 24,576,000 elements, about 93.75 MiB in FP32 instead of 375 MiB.
Broadcast multiplication still materializes high-resolution product tensors;
this is NOT a claim of fourfold lower overall peak memory. Keys, casts,
reductions and autograd also consume memory. Do a real-input four-GPU smoke
test before a long training run.
Do not silently reduce kernel size, crop inputs or use an official CUDA extension
for this first reference experiment.

## Configuration and execution

```
model:
  decoder_type: rhdb
  # Keep baseline RHDB stages [1,2]; omit mscan_options.
  upsample_mode: sapa
  sapa_options:
    embedding_dim: 64
    up_factor: 2
    kernel_size: 5
    qkv_bias: true
```

`upsample_mode: bilinear` (default) with no sapa_options preserves legacy state
keys and computations. Supplying SAPA options while selecting bilinear raises
an error. The same upsampler can coexist with existing unet or MSCAN configs,
but the first experiment deliberately excludes MSCAN to isolate SAPA.

`configs/train_kust4k_rhdb_sapa.yaml` preserves the RS-RGB-T official split,
RGB3/TIR3 normalization, nine classes, loss, optimizer, 150 epochs and validation
every 10 epochs. Start from scratch (`resume: null`), not an old baseline checkpoint
missing SAPA parameters. Config-aware training/inference recreates the module;
no changes to infer.py are necessary. Existing experiments are not overwritten.

```bash
cd /data/BUAS/HJK/ACMMamba
python tools/check_sapa.py --device cuda --amp --full-model
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 train.py --config configs/train_kust4k_rhdb_sapa.yaml --smoke-test
# Only after the smoke test passes:
bash tools/start_ablation.sh configs/train_kust4k_rhdb_sapa.yaml
tail -f logs/experiments/kust4k/rhdb_sapa/train.log
# After training finishes:
CUDA_VISIBLE_DEVICES=0 python infer.py --config configs/train_kust4k_rhdb_sapa.yaml --split test --device cuda:0 --batch-size 1 --max-comparisons 50 --output-dir results/experiments/kust4k/rhdb_sapa/test
```

Best checkpoint: `checkpoints/experiments/kust4k/rhdb_sapa/kust4k_rhdb_sapa_best_miou.pt`.

The check tool covers the specified [2,256,32,32]+[2,128,64,64] example (25,472
parameters), same-weight output and all input/parameter gradient comparisons
against the old repeat version (FP32/FP64, optional CUDA AMP), initialization,
noncontiguous inputs, low-resolution patch storage, an independent pixel-loop
numerical/gradient reference, softmax
axis, channel normalization, raw values, boundary behavior, strict errors,
double-precision gradcheck, 2/5/9-class decoders, and baseline compatibility.
`--full-model` adds reduced-channel encoder/decoder backward and strict
config/checkpoint round-trip tests. These do not replace full-size CUDA/NCCL tests.
