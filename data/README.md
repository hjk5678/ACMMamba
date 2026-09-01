# 数据模块

## 生成数据划分

在项目根目录运行：

```bash
python data/split_dataset.py
```

默认使用随机种子 `42`，按照训练集、测试集、验证集 `7:2:1` 的比例生成：

```text
data/splits/
├── train.txt
├── test.txt
└── val.txt
```

每一行是一个样本 ID。脚本只生成索引文件，不会复制、移动或修改原图。

重新执行相同命令会用相同内容覆盖划分文件。若要使用另一个随机种子：

```bash
python data/split_dataset.py --seed 123
```

## Dataset 输出

`PairedRemoteSensingDataset` 每次返回一个字典：

- `rgb`：归一化后的 `float32` 张量，形状 `[3, 256, 256]`
- `sar`：归一化后的 `float32` 张量，形状 `[1, 256, 256]`
- `label`：`int64` 张量，形状 `[256, 256]`，保留忽略值 `255`
- `id`：样本 ID

训练集启用同步随机翻转和 90 度旋转；测试集与验证集不进行随机增强。
