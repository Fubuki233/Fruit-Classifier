# Experiment Results - Fruit Classifier

This document records the setup and results of all experiments conducted within this project, serving as a basis for subsequent CA report writing and for comparing the performance of different models.
---

## 0. Common Settings 
- Dataset: Fruits (Apple / Banana / Orange / Mixed)
- Train folder: 'data/train'
- Test folder: 'data/test'
- Framework: TensorFlow/Keras
- Base Image size: 224x224 (except E6: 128x128)
- Batch size: 32
- Epochs: 50 (with EarlyStopping)

### Data Preprocessing Module Usage

All experiments use the unified `data_preprocessing.py` module:

```python
from data_preprocessing import get_preprocessor, load_from_directory

# Check dataset information
train_info = load_from_directory('data/train')
print(f"Total images: {train_info['total']}")
print(f"Classes: {train_info['classes']}")

# Get preprocessed data generators
train_gen, val_gen, test_gen = get_preprocessor(
    method='moderate',    # baseline, light, moderate, heavy, minimal
    img_size=224,
    batch_size=32
)

# Use in training
X_batch, y_batch = next(train_gen)
```

**Available Methods:**
- `baseline` - Only rescaling (1./255)
- `light` - Rotation (15°) + Horizontal flip
- `moderate` - Rotation (20°) + Shift + Zoom + Flip
- `heavy` - Full augmentation (rotation, shift, zoom, shear, brightness, flips)
- `minimal` - Minimal augmentation (rotation 10°, zoom 0.1)

**Generate Sample Images:**
```bash
python data_preprocessing.py
```
Saves preprocessed samples to `data/preprocess/` for visual comparison.

## 1. Experiment Summary

| Exp ID | Script File                  | Model Description                  | Augmentation                      | Regularization      | Train Acc | Val Acc | Test Acc | Notes |
|--------|------------------------------|------------------------------------|-----------------------------------|---------------------|-----------|---------|----------|-------|
| E1     | [E1_baseline.py](E1_baseline.py)               | Basic CNN (3 Conv layers)          | None                              | None                |           |         |          | Baseline without augmentation |
| E2     | [E2_augmentation_light.py](E2_augmentation_light.py)     | Basic CNN                          | Rotation + Flip                   | None                |           |         |          | Light augmentation |
| E3     | [E3_augmentation_heavy.py](E3_augmentation_heavy.py)     | Basic CNN                          | Rotation + Shift + Zoom + Brightness | None           |           |         |          | Heavy augmentation |
| E4     | [E4_batchnorm_dropout.py](E4_batchnorm_dropout.py)      | CNN + BatchNorm                    | Rotation + Shift + Zoom + Flip    | Dropout (0.25-0.5)  |           |         |          | BatchNorm + Dropout |
| E5     | [E5_transfer_mobilenet.py](E5_transfer_mobilenet.py)     | MobileNetV2 Transfer Learning      | Rotation + Shift + Zoom + Flip    | Dropout (0.5)       |           |         |          | Pretrained model |
| E6     | [E6_smaller_image.py](E6_smaller_image.py)          | Lightweight CNN (128x128)          | Rotation + Shift + Zoom + Flip    | Dropout (0.2-0.4)   |           |         |          | Smaller image size |

