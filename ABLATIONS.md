# ACMMamba消融实验

原基线以及编码器、融合消融实验使用 ResNet 残差解码器：
`decoder_type: unet`、`decoder_blocks_per_stage: 3`。
RHDB 双分支解码器使用独立配置及 `decoder_type: rhdb`，详见 `docs/RHDB.md`。
新的编码器和融合消融应从相应对照配置生成，保持解码器设置一致。

## MLFM消融模式

模型参数`model.fusion_mode`支持以下取值：

| 模式 | 融合公式 | 目的 |
|---|---|---|
| `mlfm` | `a*F_add + b*F_cat` | 完整基线，`a,b`可学习 |
| `mean` | `(F_A + F_B)/2` | 去除MLFM的无参数主消融 |
| `add_only` | `FusionConv(F_A + F_B)` | 检验Add分支 |
| `cat_only` | `FusionConv(cat(F_A,F_B))` | 检验Cat分支 |
| `dual_fixed` | `0.5*F_add + 0.5*F_cat` | 检验可学习权重的贡献 |

旧配置没有`fusion_mode`时默认使用`mlfm`，因此已有基线和权重不受影响。

## 推荐实验协议

1. 所有数据集运行`mean`，和完整基线比较，验证MLFM是否普遍有效。
2. 在论文主数据集运行`add_only`、`cat_only`、`dual_fixed`，解释双分支与可学习权重的贡献。
3. 除`fusion_mode`外，数据划分、随机种子、训练轮数、学习率、Loss和数据增强全部保持不变。
4. 使用验证集mIoU选择最佳权重，最终只在测试集评估一次。
5. 至少报告参数量、mIoU、mF1、mAcc、OA和逐类IoU；若算力允许，使用3个随机种子报告均值和标准差。

## 生成配置

当前主消融数据集为`DDHRNet/korea/no_cloud`。从该基线生成4份配置：

```bash
python tools/make_fusion_ablation_configs.py \
  --base configs/train_korea_no_cloud.yaml \
  --output-dir configs/ablations
```

只生成主消融`mean`：

```bash
python tools/make_fusion_ablation_configs.py \
  --base configs/train_korea_no_cloud.yaml \
  --output-dir configs/ablations \
  --modes mean
```

每份配置使用独立日志、checkpoint目录和权重前缀，不会覆盖基线结果。

## 启动实验

先运行单卡冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/ablations/train_korea_no_cloud_fusion_mean.yaml \
  --smoke-test
```

然后使用通用后台启动脚本运行四卡实验：

```bash
bash tools/start_ablation.sh \
  configs/ablations/train_korea_no_cloud_fusion_mean.yaml
```
