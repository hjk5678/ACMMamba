# Soft Cross 模型架构图

依据当前源码及用户确认的 `cross_frequency: once_per_stage` 绘制。
参考配置为 `configs/ablations/train_kust4k_encoder_cccc_soft_once_2242_150e.yaml`。
图中输入模态和类别数采用通用标注，RGB-T 实验具体为 3+3 输入通道、9 类输出。

## 图件

- 总体架构：`figures/acmmamba_soft_cross_once_architecture.png`
- 总体架构可编辑矢量版：`figures/acmmamba_soft_cross_once_architecture.svg`
- Soft Cross、AS6、MLFM 内部结构：`figures/acmmamba_soft_cross_modules.png`
- 内部结构可编辑矢量版：`figures/acmmamba_soft_cross_modules.svg`

重绘：`python tools/draw_soft_cross_architecture.py`，需要 matplotlib、PyYAML。
脚本读取配置核对深度、通道、交互频率及解码器设置，不加载模型权重、不修改训练配置。

## 与代码对应的关键结构

| Stage | 输出通道 | 空间尺寸 | block 顺序 |
|---|---:|---|---|
| 1 | 96 | H/4 × W/4 | Soft Cross → Self |
| 2 | 192 | H/8 × W/8 | Soft Cross → Self |
| 3 | 384 | H/16 × W/16 | Soft Cross → Self → Self → Self |
| 4 | 768 | H/32 × W/32 | Soft Cross → Self |

空间比例以 H、W 可被 64 整除为前提。实际代码根据张量尺寸插值。
每个 block 包含两路独立的归一化、预处理、四方向 AS6、输出投影、残差和 FFN。
Self block 使用已经交互过的空间特征重新扫描，保留已有跨模态信息，但不引入新的路径交换。
共 10 个双分支 block、80 个独立 AS6 实例。每阶段只有首个 block 创建 4 个交换参数，共 16 个。

四方向为行、列、反向行、反向列，记作 1、2、3、4。
Soft Cross 发生在 AS6 序列处理之前，只混合反向路径：

```text
A_d_new = (1 - alpha_d) * A_d + alpha_d * B_d
B_d_new = (1 - beta_d)  * B_d + beta_d  * A_d    d = 3, 4
alpha_d = sigmoid(theta_A_d)
beta_d  = sigmoid(theta_B_d)
```

两个等式均使用更新前的路径。正向路径 1、2 保持本模态输入。
每个目标模态、反向方向、阶段各自拥有独立标量。初始化为 sigmoid(0)=0.5。
权重在样本、通道和空间位置间共享，是训练得到的参数，不是输入相关的动态门控。

AS6 的序列流程：输入投影与门控拆分 → DWConv1d / SiLU → Selective S6 → Memory Adaptor → LN / 门控乘法 → 输出投影。
然后逐方向逆映射至二维空间，执行 Spatial Adaptor，再对四方向结果求和。
Memory Adaptor 使用 S6 输出序列；Spatial Adaptor 在逆映射之后执行。

MLFM 每阶段末执行一次。A/B 特征继续进入下一编码阶段，融合结果 S1–S4 用于解码器。
MLFM 对齐通道后分别进行 Add、Concat 及卷积细化，再使用两个无约束可学习标量组合。
MLFM 标量与 Soft Cross 的 sigmoid 交换参数是不同的参数集合。

解码器：S4 → 额外 stride-2 bottleneck → D1(S4′,S4) → D2(D1,S3) → D3(D2,S2) → D4(D3,S1)。
每级 UpBlock 为双线性插值 → 1×1 调整通道 → 拼接 skip → 3 个 ResNet 残差块。
分割头为 ConvBlock → 1×1 分类卷积 → 双线性插值至输入尺寸，返回 logits。

## 图注草稿

**中文：** ACMMamba 总体架构。网络采用四阶段双分支编码器，各阶段深度分别为 2、2、4、2。
每个阶段的首个 block 在 AS6 建模之前，通过独立可学习系数对两个模态的反向扫描序列进行软交互，
其余 block 在更新后的特征上进行模态内建模。每阶段末的 MLFM 生成多尺度融合特征，
经跳跃连接传入残差 U-Net 解码器，实现像素级分割。

**English:** Overview of ACMMamba. The dual-branch encoder contains four stages with depths of
2, 2, 4, and 2. At the first block of each stage, Soft Cross mixes the reverse-direction
sequences of both modalities using independent learned coefficients before AS6 processing.
The remaining blocks perform self-routed modeling on the updated features. An MLFM module
at each stage end produces fused features for the residual U-Net decoder through skip connections.

验证范围：静态源码与配置核对、公式核对、图件渲染检查。未执行训练或 GPU 推理验证。
