# WeedSwin Quick Start Guide

This guide provides a quick overview of using the WeedSwin model for weed detection.

## Installation

```bash
# 1. Create environment
conda create -n weedswin python=3.8
conda activate weedswin

# 2. Install PyTorch
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install MMDetection
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0"

# 4. Install additional dependencies
pip install -r requirements.txt
```

## Dataset Setup

Organize your dataset in COCO format:

```
data/
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── train/
│   └── *.jpg
├── val/
│   └── *.jpg
└── test/
    └── *.jpg
```

Update the `data_root` path in [configs/weedswin.py](configs/weedswin.py) to point to your dataset.

## Training

### Quick Training

```bash
python train_weedswin.py
```

### Advanced Training (with custom config)

```bash
python tools/train.py configs/weedswin.py --work-dir work_dirs/weedswin
```

### Distributed Training

```bash
bash tools/dist_train.sh configs/weedswin.py 8  # 8 GPUs
```

## Testing

### Quick Testing

```bash
python test_weedswin.py --checkpoint work_dirs/weedswin/epoch_12.pth
```

### Advanced Testing

```bash
python tools/test.py configs/weedswin.py work_dirs/weedswin/epoch_12.pth --show-dir results/
```

## Model Configuration

The WeedSwin model is configured in [configs/weedswin.py](configs/weedswin.py):

- **Backbone**: Swin Transformer (192 embed dims, [2,2,18,4] depths)
- **Head**: DINO detection head
- **Classes**: 174 weed classes (16 species × 11 growth stages)
- **Training**: 12 epochs with AdamW optimizer
- **Pretrained**: DINO-Swin weights from COCO

## Key Parameters

Edit [configs/weedswin.py](configs/weedswin.py) to customize:

- `data_root`: Path to your dataset
- `max_epochs`: Number of training epochs (default: 12)
- `batch_size`: Training batch size (default: 2)
- `lr`: Learning rate (default: 0.0001)
- `num_classes`: Number of weed classes (default: 174)

## Model Performance

- **mAP**: 0.993 ± 0.004
- **mAR**: 0.985
- **FPS**: 218.27
- **Parameters**: Optimized for weed detection across multiple growth stages

## Directory Structure

```
weedswin/
├── configs/
│   └── weedswin.py          # Main config file
├── mmdet/                    # MMDetection framework
├── tools/                    # Training and testing scripts
├── train_weedswin.py         # Simple training script
├── test_weedswin.py          # Simple testing script
├── requirements.txt          # Dependencies
└── README.md                 # Full documentation
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in config:
```python
train_dataloader = dict(batch_size=1, ...)
```

### Dataset Not Found
Verify paths in config:
```python
data_root = 'path/to/your/data/'
ann_file = 'annotations/train.json'
```

### Import Errors
Verify installation:
```bash
python -c "import torch, mmcv, mmdet; print(torch.__version__, mmcv.__version__, mmdet.__version__)"
```

## Citation

```bibtex
@article{islam2025weedswin,
  title={WeedSwin hierarchical vision transformer with SAM-2 for multi-stage weed detection and classification},
  author={Islam, Taminul and Sarker, Toqi Tahamid and Ahmed, Khaled R and Rankrape, Cristiana Bernardi and Gage, Karla},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={23274},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## Dataset

Download the Weed Growth Stage Dataset from Zenodo:
https://doi.org/10.5281/zenodo.15808623
