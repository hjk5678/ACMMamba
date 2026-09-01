# ACMMamba

## Getting Started

### Installation

#### Step 1: Clone the ACMMamba repository

```bash
git clone https://github.com/hjk5678/ACMMamba.git
cd ACMMamba
```

#### Step 2: Set up the environment

A Conda environment with Python 3.10 is recommended. The dependency versions
in this repository use PyTorch 2.6.0 and CUDA 12.6.

```bash
conda create -n acmmamba python=3.10 -y
conda activate acmmamba

python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

#### Step 3: Install the VMamba selective-scan kernel

The selective-scan CUDA extension is compiled separately so that it matches
the PyTorch, CUDA toolkit, and GPU architecture of the target machine. Clone
the [official VMamba repository](https://github.com/MzeroMiko/VMamba) and build
the extension inside the ACMMamba environment:

```bash
cd ..
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba/kernels/selective_scan
python -m pip install .
cd ../../../ACMMamba
```

A working CUDA toolkit and a compatible C++ compiler are required. Verify the
installation with:

```bash
python -c "import torch, selective_scan_cuda_oflex; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### Dataset Preparation

Set `rgb_dir`, `sar_dir`, `label_dir`, and `split_dir` under `data` in the
selected YAML file in `configs/`. The split directory must contain
`train.txt`, `val.txt`, and `test.txt`, with one sample ID per line.

To generate the default `7:1:2` train/validation/test split with seed 42:

```bash
python data/split_dataset.py
```

See [data/README.md](data/README.md) for the expected dataset output and split
format.

### Model Training and Inference

#### Training

Run a one-batch training and validation smoke test before a full experiment:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/train_shandong.yaml \
  --smoke-test
```

Train on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/train_shandong.yaml
```

Train on four GPUs with DistributedDataParallel:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py --config configs/train_shandong.yaml
```

Replace the YAML path with the configuration for the target dataset. See
[TRAINING.md](TRAINING.md) for background execution, checkpoint resumption,
logging, and output details.

#### Inference and evaluation

Evaluate a dataset split with a trained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --config configs/train_shandong.yaml \
  --checkpoint /path/to/best_miou.pt \
  --split val \
  --output-dir results/shandong/val
```

Run inference on a single paired sample:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_single.py \
  --config configs/train_shandong.yaml \
  --checkpoint /path/to/best_miou.pt \
  --input-dir data/test
```

See [INFERENCE.md](INFERENCE.md) for dataset-specific examples, output files,
metrics, visualization, and additional arguments.

## Documentation

- [Model architecture](MODEL_ARCHITECTURE.md)
- [Training guide](TRAINING.md)
- [Inference and evaluation guide](INFERENCE.md)
- [Dataset guide](data/README.md)
