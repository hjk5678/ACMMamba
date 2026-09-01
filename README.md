# ACMMamba

## Installation

Create and activate a Python environment. This project currently uses PyTorch
2.6.0 with CUDA 12.6; install its official wheels first:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

Then install the remaining Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Install the VMamba selective-scan kernel

The selective-scan CUDA extension is not included in `requirements.txt` because
it must be compiled against the PyTorch and CUDA versions in the target
environment. Clone the [official VMamba repository](https://github.com/MzeroMiko/VMamba)
and install the extension from its source directory:

```bash
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba/kernels/selective_scan
python -m pip install .
```

Run the command above inside the same Python environment used for ACMMamba.
A working CUDA toolkit and a compiler compatible with the installed PyTorch
version are required to build the extension.
