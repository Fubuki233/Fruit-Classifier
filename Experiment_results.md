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

| Exp ID | Script File     | Model Description        | Geometry(Shift/Zoom/Shear)      | Color/Brightness                         | Val Acc | Training Acc | Notes                                    |
|--------|-----------------|--------------------------|---------------------------------|------------------------------------------|---------|--------------|------------------------------------------|
| E1     | Baseline        | Rotation:None Flips:None | None                            | None                                     | 0.8281  | 0.7617(6)    | Only rescaling                           |
| E2     | Minimal         | Rotation:10° Flips:None  | Zoom:0.1                        | None                                     | 0.9062  | 0.9258(41)   | Slight rotation and zoom                 |
| E3     | Light           | Rotation:15° Flips:H     | None                            | None                                     | 0.9375  | 0.8984(37)   | Basic rotation and horizontal flipping   |
| E4     | Moderate        | Rotation:20° Flips:H     | Shift:0.2 Zoom:0.2              | None                                     | 0.8281  | 0.7656(7)    | Balanced augmentation                    |
| E5     | Heavy(Original) | Rotation:30° Flips:H&V   | Shift:0.2 Shear:0.2 Zoom:0.2    | Brightness:[0.8, 1.2]                    | 0.8750  | 0.8008(42)   | Original "heavy" setting.                |
| E6     | Heavy1          | Rotation:45° Flips:H     | Shift:0.2 Shear:0.2 Zoom:0.2    | None                                     | 0.8906  | 0.8086(32)   | Increased rotation to 45°, but removes brightness and vertical flip, <br/>added fill_mode = “nearest”                                      |
| E7     | Heavy(Special)  | Rotation:45° Flips:H     | Shift:0.2 Shear:0.2 Zoom:0.2    | None                                     | 0.9062  | 0.8086(35)   | Identical augmentation parameters to Heavy1, <br/>but explicitly specifies validation_split=0.2  |
| E8     | Heavy1+Heavy    | Rotation:45° Flips:H&V   | Shift:0.2 Shear:0.2 Zoom:0.2    | Brightness:[0.8, 1.2]                    | 0.8750  | 0.7773(41)   | Combines max rotation with full geometric and brightness augmentation.             |
| E9     | Color_Boost     | Rotation:20° Flips:H     | Shift:0.15 Zoom:0.15            | Brightness:[0.7, 1.3] Channel Shift:15.0 | 0.8594  | 0.6914(32)   | Focuses on color with stronger brightness and channel shifting.                                          |   
| E10    | Mixed           | Rotation:40° Flips:H&V   | Shift:0.25 Shear:0.25 Zoom:0.35 | Brightness:[0.5, 1.5] Channel Shift:35.0 | 0.8594  | 0.8203(35)   | Most aggressive overall: high rotation, maximum geometric, <br/>and intense color/brightness/channel shifts.                          |
