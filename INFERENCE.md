# ACMMamba通用推理与评估

`infer.py`由训练配置驱动，同一个入口支持：

- RGB+单通道SAR以及RGB+三通道TIR；
- 不同类别数量、类别名称和可视化配色；
- `train`、`val`、`test`任意划分；
- 不同数据集配置与任意兼容checkpoint组合；
- 原始预测、彩色预测、四联图、逐类指标和混淆矩阵导出。

通用调用格式：

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --config configs/<训练配置>.yaml \
  --checkpoint /path/to/<权重>.pt \
  --split val \
  --output-dir results/<实验名>/val
```

如果省略`--checkpoint`，程序会根据配置中的`checkpoint_dir`和
`checkpoint_prefix`自动选择`*_best_miou.pt`。如果省略`--output-dir`，
默认输出到`results/<checkpoint_prefix>/<split>`。

## Kust4K验证集

使用第100轮最佳权重评估官方403张验证图：

```bash
cd /data/BUAS/HJK/ACMMamba
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --config configs/train_kust4k.yaml \
  --checkpoint checkpoints/kust4k/kust4k_best_miou.pt \
  --split val \
  --output-dir results/kust4k/val \
  --max-comparisons 50
```

`--max-comparisons 50`只限制四联图数量，不会限制评估样本数；所有403张图
仍会参与Loss、mIoU、mF1、mAcc、OA、FWIoU和逐类指标计算。若需保存全部
四联图，改为`--max-comparisons -1`。

## 山东单张样本四图推理

使用山东最佳权重，对`data/test`中的`RGB.jpg`和`SAR.jpg`推理。GT会按`GT.png`、`GT.tif`、`GT.tiff`、`GT.jpg`、`GT.jpeg`的顺序自动查找：

```bash
CUDA_VISIBLE_DEVICES=0 python infer_single.py \
  --config configs/train_shandong.yaml \
  --checkpoint checkpoints/best_miou.pt \
  --input-dir /data/BUAS/HJK/ACMMamba/data/test
```

四图结果保存为`data/test/shandong_4panel_result.png`，同时保存原始预测、彩色预测和该图片的指标JSON。重复运行时需添加`--overwrite`。

推理另一组图片时使用独立输出前缀，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python infer_single.py \
  --config configs/train_shandong.yaml \
  --checkpoint checkpoints/best_miou.pt \
  --input-dir /data/BUAS/HJK/ACMMamba/data/test \
  --rgb-name RGB225.jpg \
  --sar-name SAR225.jpg \
  --gt-name GT225.png \
  --output-prefix shandong_225
```

## 西安云区测试集示例

在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --config configs/train_xian_cloud.yaml \
  --checkpoint checkpoints/xian_cloud/xian_cloud_best_miou.pt \
  --split test \
  --output-dir results/xian_cloud/test
```

默认开启AMP并保存全部424张对比图。若只保存前50张对比图：

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --config configs/train_xian_cloud.yaml \
  --split test \
  --max-comparisons 50
```

配置中已经记录最佳权重的默认位置，因此可以省略`--checkpoint`。输出目录非空时程序会停止，防止混合不同实验的结果；确认需要覆盖时传入`--overwrite`。

## 输出目录

```text
results/xian_cloud/test/
├── raw_masks/             # 原始类别索引PNG，像素值为0至4
├── color_masks/           # 彩色预测PNG
├── comparisons/           # RGB、SAR、GT、Prediction四联图
├── metrics.json           # 完整指标与混淆矩阵
├── confusion_matrix.csv   # 混淆矩阵表格
├── summary.txt            # 便于阅读的测试摘要
└── inference.log          # 推理日志
```

类别名称和颜色由对应YAML配置决定。旧的5类配置在未指定颜色时继续使用原有
调色板；Kust4K在`configs/train_kust4k.yaml`中定义了9类颜色。忽略值255
默认为白色，且不参与Loss和指标。

## 其他划分与参数

- 验证集：`--split val`
- 训练集：`--split train`
- 指定GPU：`--device cuda:1`
- 禁用AMP：`--no-amp`
- 调整批量：`--batch-size 4`
- 不保存四联图：`--max-comparisons 0`
- 允许写入已有输出目录：`--overwrite`

注意：配置必须与权重的模型结构匹配，包括第二模态通道数和类别数；不匹配时
严格权重加载会立即报错，避免静默得到无效结果。
