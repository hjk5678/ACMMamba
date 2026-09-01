# ACMMamba 训练说明

训练入口为 `train.py`，默认配置为 `configs/train_shandong.yaml`。训练程序已经包含：

- 单 GPU 与 `torchrun` 多 GPU DDP 训练；
- FP16 自动混合精度和梯度缩放；
- 梯度累积、梯度裁剪、AdamW；
- warmup + polynomial learning-rate decay；
- 加权交叉熵 + Soft Dice Loss，标签 `255` 不参与损失和指标；
- 验证集总体 mIoU、mF1、mAcc、OA、FWIoU，以及逐类别 IoU/F1/Acc；
- TensorBoard、JSONL 日志、最佳权重和完整断点保存；
- tqdm 训练和验证进度条。

## 1. 运行前检查

在服务器项目根目录中执行：

```bash
conda activate vmpsam1
export PYTHONNOUSERSITE=1
which python
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

当前配置中的数据路径对应：

```text
/data/BUAS/HJK/ACMMamba/data/DDHRNet/shandong/GF2
/data/BUAS/HJK/ACMMamba/data/DDHRNet/shandong/GF3
/data/BUAS/HJK/ACMMamba/data/DDHRNet/shandong/label
/data/BUAS/HJK/ACMMamba/data/splits
```

## 2. 先做一次冒烟测试

冒烟测试只运行一个训练 batch 和一个验证 batch，不写 checkpoint：

```bash
cd /data/BUAS/HJK/ACMMamba
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/train_shandong.yaml \
  --smoke-test
```

它会执行完整的前向、Loss、反向传播、一次优化器更新和验证指标计算，因此比只检查模型前向更接近真实训练。

## 3. 使用 setsid 启动四卡正式训练

确认已经激活 `vmpsam1` 环境，然后执行项目提供的启动脚本：

```bash
cd /data/BUAS/HJK/ACMMamba
conda activate vmpsam1
bash tools/start_train.sh
```

脚本使用 `setsid` 启动四卡 DDP，退出 SSH 后训练仍会继续。等价的完整命令为：

```bash
mkdir -p /data/BUAS/HJK/ACMMamba/logs \
         /data/BUAS/HJK/ACMMamba/checkpoints

cd /data/BUAS/HJK/ACMMamba
export PYTHONNOUSERSITE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 setsid python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train.py --config configs/train_shandong.yaml \
  > /data/BUAS/HJK/ACMMamba/logs/console_manual.log 2>&1 < /dev/null &

echo $! > /data/BUAS/HJK/ACMMamba/logs/train.pid
```

配置中的 `batch_size` 是“每张 GPU 的 batch size”。默认每卡为 1、梯度累积为 4；四卡时每次优化更新的有效 batch size 为：

```text
1 × 4 GPU × 4 accumulation = 16
```

实时查看启动日志和结构化日志：

```bash
tail -f /data/BUAS/HJK/ACMMamba/logs/console_*.log
tail -f /data/BUAS/HJK/ACMMamba/logs/train.log
```

默认训练 100 个 epoch，每 10 个 epoch 运行一次完整验证。逐类指标按照以下顺序记录：

```text
0 farmland  农田
1 greenery  绿地/植被
2 road      道路
3 building  建筑
4 water     水体
```

逐类 `OA` 按该类别相对其余类别的 one-vs-rest 准确率计算，逐类 `Acc` 是该类别的像素召回率；总体 `OA` 是全部有效像素的整体准确率。

## 4. 断点续训

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --config configs/train_shandong.yaml \
  --resume /data/BUAS/HJK/ACMMamba/checkpoints/last.pt
```

`last.pt` 包含模型、优化器、学习率调度器、GradScaler、epoch 和历史最佳 mIoU，可进行完整续训。

## 5. 输出文件

日志目录为 `/data/BUAS/HJK/ACMMamba/logs`：

```text
resolved_config.yaml       本次运行的完整配置快照
history.jsonl              每个 epoch 的训练损失和验证指标
train.log                  不含进度条控制字符的结构化文本日志
console_*.log              setsid 标准输出和训练进度条
tensorboard/               TensorBoard event 文件
```

权重目录为 `/data/BUAS/HJK/ACMMamba/checkpoints`：

```text
last.pt                    最近 epoch 的完整续训断点
best_miou.pt               验证集总体 mIoU 历史最高的模型权重
```

查看曲线：

```bash
tensorboard --logdir /data/BUAS/HJK/ACMMamba/logs/tensorboard --port 6006
```

若显存不足，优先保持 `batch_size: 1`，并通过增大 `accumulation_steps` 调整有效 batch size；若 DataLoader 造成内存或共享内存压力，再降低 `num_workers`。
