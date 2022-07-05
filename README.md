# Distributed Deep Learning with PyTorch

This repo contains the source code for the course project of Parallel Computing. The main purpose of this repo is to train UNet using different distributed strategies from PyTorch (`DataParallel`, `DistributedDataParallel`, and `Pipeline`).

## Set up environment and install packages

```
conda create -n unetdist
conda activate unetdist
```

```
git clone https://github.com/notnitsuj/DistributedPyTorch.git
cd DistributedPyTorch
bash install.sh
```

[Download](https://drive.google.com/drive/folders/1ZR3wnSVMZTkcV0D4yqNLM-2ek1j_zCTG?usp=sharing) the data and unzip into the `data` folder.

## Train

To train normally with one GPU, run

```
python3 train.py
```

To train using 2 GPUs with DataParallel, run

```
python3 train.py -t DP
```

To train using 2 GPUs with DistributedDataParallel, run

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py -t DDP -b 2
```

To train using 2 GPUs with Pipeline model parallelism, run

```
python3 train.py -t MP
```
