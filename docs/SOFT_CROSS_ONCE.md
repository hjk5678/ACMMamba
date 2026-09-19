# RS-RGB-T: one soft cross per stage, AS6 depths 2242, 150 epochs

Config: `configs/ablations/train_kust4k_encoder_cccc_soft_once_2242_150e.yaml`.

The first block of each stage performs soft cross before AS6. The remaining
blocks re-scan updated spatial features and run Self-AS6, without introducing
new cross-modal paths. Previous cross-modal information remains in the features.

| Stage | Block routes |
|---|---|
| 1 | Cross, Self |
| 2 | Cross, Self |
| 3 | Cross, Self, Self, Self |
| 4 | Cross, Self |

2242 counts complete AS6 processing blocks (including their existing
normalization, projections, residuals and FFN), not bare S6 recurrences.
MLFM fuses final A/B features once at the end of each stage. The original
ResNet U-Net decoder retains three residual blocks per decoder stage.

There are 10 dual-branch blocks, 80 AS6 instances and 16 independent exchange
scalars (4 per stage), initialized to sigmoid(0)=0.5. No unused exchange
parameters are created in Self blocks.

`model.cross_frequency: once_per_stage` selects this behavior. The default
remains `every_block` to preserve existing experiments/checkpoint structures.
Use the new config for inference too. Do not directly resume an every-block
soft checkpoint as this model; the exchange parameter keys differ.

Training starts from scratch: 150 epochs, validation every 10 epochs,
batch size 1 per GPU, accumulation 4, AMP, learning rate 6e-5, warmup 5 epochs.
The scheduler uses the new 150-epoch duration. Splits, preprocessing and loss
are unchanged. Best weights are selected by validation mIoU. Class metrics and
learned exchange coefficients are logged as before.

Logs: `logs/ablations/kust4k/encoder/cccc_soft_once_2242_150e/`.
Weights: `checkpoints/ablations/kust4k/encoder/cccc_soft_once_2242_150e/`.
Best filename: `kust4k_encoder_cccc_soft_once_2242_150e_best_miou.pt`.

## Server commands

```bash
cd /data/BUAS/HJK/ACMMamba
python tools/check_soft_cross.py --device cuda --amp --cross-frequency once_per_stage
python train.py --config configs/ablations/train_kust4k_encoder_cccc_soft_once_2242_150e.yaml --smoke-test
```

After successful checks, start the existing four-GPU setsid launcher:

```bash
bash tools/start_ablation.sh configs/ablations/train_kust4k_encoder_cccc_soft_once_2242_150e.yaml
tail -f logs/ablations/kust4k/encoder/cccc_soft_once_2242_150e/train.log
```

Full-resolution GPU memory must be checked on the server; local regression tests
use reduced channel counts. Compare exchange frequencies at the same 2242 depth
and 150 epochs: older 100-epoch results differ in training duration as well.
