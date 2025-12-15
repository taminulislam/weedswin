# WeedSwin Repository Structure

This repository contains the essential files for running the WeedSwin model for weed detection and classification.

## Directory Structure

```
weedswin/
├── configs/                    # Configuration files
│   ├── weedswin.py            # Main WeedSwin config (self-contained)
│   ├── weedswin/              # Additional config directory
│   └── _base_/                # Base configs (if needed)
│
├── mmdet/                      # MMDetection framework (33MB)
│   ├── models/                # Model implementations
│   │   ├── backbones/        # Including Swin Transformer
│   │   ├── detectors/        # Including DINO
│   │   └── ...
│   ├── datasets/              # Dataset loading
│   ├── apis/                  # Training/testing APIs
│   └── ...
│
├── tools/                      # Training and testing utilities (518KB)
│   ├── train.py               # Main training script
│   ├── test.py                # Main testing script
│   ├── dist_train.sh          # Distributed training
│   └── dist_test.sh           # Distributed testing
│
├── resources/                  # Documentation resources (2MB)
│   ├── WeedSwin.png           # Architecture diagram
│   └── WeedSwin.pdf           # Paper PDF
│
├── train_weedswin.py          # Simplified training script
├── test_weedswin.py           # Simplified testing script
├── requirements.txt           # Python dependencies
├── README.md                  # Full documentation
├── QUICKSTART.md              # Quick start guide
└── STRUCTURE.md               # This file
```

## Essential Files for Running WeedSwin

### 1. Configuration
- **[configs/weedswin.py](configs/weedswin.py)** - Complete model configuration including:
  - Swin Transformer backbone settings
  - DINO detection head configuration
  - 174 weed class definitions
  - Training hyperparameters
  - Data pipeline settings

### 2. Model Implementation
- **mmdet/** - MMDetection framework containing:
  - Swin Transformer backbone (`mmdet/models/backbones/swin.py`)
  - DINO detector (`mmdet/models/detectors/dino.py`)
  - All necessary model components

### 3. Training Scripts
- **[train_weedswin.py](train_weedswin.py)** - Simple training interface
- **[tools/train.py](tools/train.py)** - Advanced training with full options

### 4. Testing Scripts
- **[test_weedswin.py](test_weedswin.py)** - Simple testing interface
- **[tools/test.py](tools/test.py)** - Advanced testing with visualization

### 5. Dependencies
- **[requirements.txt](requirements.txt)** - All required packages

## Model Components

### WeedSwin Architecture

The WeedSwin model combines:

1. **Swin Transformer Backbone**
   - Embed dimensions: 192
   - Depths: [2, 2, 18, 4]
   - Num heads: [6, 12, 24, 48]
   - Window size: 12
   - Location: `mmdet/models/backbones/swin.py`

2. **DINO Detection Head**
   - 4-scale features
   - 300 queries
   - Deformable attention
   - Location: `mmdet/models/detectors/dino.py`, `mmdet/models/dense_heads/dino_head.py`

3. **Dataset Handler**
   - COCO format support
   - 174 weed classes
   - Location: `mmdet/datasets/coco.py`

## What Was Removed

To keep only essential WeedSwin files, the following were removed:

- ❌ `notebooks/` - Jupyter notebooks (converted to Python scripts)
- ❌ `local_configs/` - Old configuration files (consolidated)
- ❌ `tests/` - Unit tests
- ❌ `docs/` - Extensive documentation (kept README)
- ❌ Unused model configs (100+ other detection models)
- ❌ Optional dependencies

## Minimal Setup

To run WeedSwin, you only need:

1. **Python Environment** (Python 3.8+)
2. **Core packages**: PyTorch, MMEngine, MMCV, MMDetection
3. **This repository** with:
   - `configs/weedswin.py`
   - `mmdet/` directory
   - `train_weedswin.py` or `tools/train.py`
   - Your dataset in COCO format

## Total Repository Size

- **mmdet/**: ~33 MB (core framework)
- **tools/**: ~518 KB (training/testing scripts)
- **configs/**: ~261 KB (configuration files)
- **resources/**: ~2 MB (documentation)
- **Total**: ~36 MB (minimal, focused on WeedSwin)

## Quick Commands

```bash
# Train
python train_weedswin.py

# Test
python test_weedswin.py --checkpoint work_dirs/weedswin/epoch_12.pth

# Verify setup
python -c "import mmdet; print(mmdet.__version__)"
```

## Dataset Format

WeedSwin expects COCO format:

```json
{
  "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [x, y, w, h]}],
  "categories": [{"id": 0, "name": "ABUTH_week_1"}, ...]
}
```

Update `data_root` in [configs/weedswin.py](configs/weedswin.py) to point to your dataset.

## Performance

- **mAP**: 0.993 ± 0.004
- **mAR**: 0.985
- **FPS**: 218.27
- **Classes**: 174 (16 weed species × 11 growth stages)

## Citation

```bibtex
@article{islam2025weedswin,
  title={WeedSwin hierarchical vision transformer with SAM-2 for multi-stage weed detection and classification},
  author={Islam, Taminul and Sarker, Toqi Tahamid and Ahmed, Khaled R and Rankrape, Cristiana Bernardi and Gage, Karla},
  journal={Scientific Reports},
  year={2025}
}
```
