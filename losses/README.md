# Segmentation loss

The default training objective is:

```text
L = 1.0 * L_weighted_CE + 1.0 * L_soft_Dice
```

- `ignore_index=255`; ignored pixels contribute to neither term.
- Both terms support the five class weights computed from the complete Shandong
  label distribution.
- Inverse square-root frequency balancing is used and normalized to a mean of
  one. This is less aggressive than full inverse-frequency balancing.
- Dice is accumulated in float32 for stable mixed-precision training.
- Classes absent from a minibatch are excluded from the Dice average.

Typical use:

```python
from losses import build_segmentation_loss

criterion = build_segmentation_loss().cuda()
logits = model(batch["rgb"].cuda(), batch["sar"].cuda())
loss = criterion(logits, batch["label"].cuda())
```

For logging individual components:

```python
losses = criterion(logits, target, return_components=True)
losses["loss"].backward()
print(losses["cross_entropy"], losses["dice"])
```
