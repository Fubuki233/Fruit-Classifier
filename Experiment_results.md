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

**Preprocessing in CNN.ipynb:**

```python
**Available Methods:**
- `baseline` - Only rescaling (1./255)
- `light` - Rotation (15°) + Horizontal flip
- `moderate` - Rotation (20°) + Shift + Zoom + Flip
- `heavy` - Full augmentation (rotation, shift, zoom, shear, brightness, flips)
- `minimal` - Minimal augmentation (rotation 10°, zoom 0.1)
- `color_boost` - moderate geometry + brightness [0.7-1.3] + channel shift

**Generate Sample Images:**
```bash
python data_preprocessing.py
```
Saves preprocessed samples to `data/preprocess/` for visual comparison.

## 1. Experiment Summary

| Exp ID | Script File          | Model Description         | Geometry(Shift/Zoom/Shear)                                       | Color/Brightness   | Train Acc | Val Acc | Test Acc | Notes |
|--------|----------------------|---------------------------|------------------------------------------------------------------|--------------------|-----------|---------|----------|-------|
| E1     | Baseline             | Rotation:None Flips:None  | None                                                             | None               |          | Baseline without augmentation |
| E2     | Minimal              | Rotation:10° Flips:None   | Rotation + Flip                                                  | None               |           |         |     Test Accuracy: 90.00%     | Light augmentation |
| E3     | Light                | Rotation:15° Flips:H      | Rotation + Shift + Zoom + Brightness                             | None               |           |         |          | Heavy augmentation | Test Accuracy: 91.67%
| E4     | Moderate             | Rotation:20° Flips:H      | Rotation + Shift + Zoom + Flip                                   | Dropout (0.25-0.5) |           |         |          | BatchNorm + Dropout |
| E5     | Heavy(Original)      | Rotation:30° Flips:H&V    | Rotation + Shift + Zoom + Flip                                   | Dropout (0.5)      |           |         |          | Pretrained model |
| E6     | Heavy1               | Rotation:45° Flips:H      | Rotation + Shift + Zoom + Flip                                   | Dropout (0.2-0.4)  |           |         |          | Smaller image size |
| E7     | Heavy2(Special)      | Rotation:45° Flips:H      | Rotation + Shift + Zoom + Flip                                   | None               |           |         |          | Increased Depth (Testing model capacity)|
| E8     | Heavy3(Heavy1+Heavy) | Rotation:20               | Rotation + Shift + Zoom + Flip                                   | Dropout (0.5)      |           |         |          | Heavy augmentation + Dropout |
| E9     | Color_Boost          | Basic CNN (3 Conv layers) | Moderate geo + brightness [0.7-1.3] + channel shift              | Dropout (0.2/0.2)  |           |  Test Accuracy: %    |   
| E10    | Mixed                | Basic CNN (3 Conv layers) | Intensive mix: strong geo + brightness [0.5-1.5] + channel shift | Dropout (0.2/0.2)  |        |         |          |  Test Accuracy: % |
