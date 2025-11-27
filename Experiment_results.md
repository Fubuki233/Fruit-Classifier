# Experiment Results - Fruit Classifier

This document records the setup and results of all experiments conducted within this project, serving as a basis for subsequent CA report writing and for comparing the performance of different models.
---

## 0. Common Settings 
- Dataset: Fruits (Apple / Banana / Orange / Mixed)
- Train folder: 'data/train'
- Test folder: 'data/test'
- Framework: 
- Image size:

## 1. Experiment Summary

| Exp ID | Notebook / Script     | Model Settings Description         | Data augmentation | Dropout | Train Acc | Val Acc | Test Acc | Notes |
|-------|------------------------|------------------------------------|-------------------|---------|-----------|---------|----------|-------|
| E1    | 01_baseline.ipynb      | Baseline CNN                       | None              | 0.0     |           |         |          | E1: No data augmentation |
| E2    | 02_aug_rotation.ipynb  | Baseline + more filters            | Rotation          | 0.0     |           |         |          | E2: Rotation only        |
| E3    | 03_aug_dropout.ipynb   | Baseline + Aug + Dropout(0.5)      | Flip + Rotation   | 0.5     |           |         |          | E3: Aug + Dropout        |
| ...   | ...                    | ...                                | ...               | ...     |           |         |          | ...   |


## 2. Experiment Details

### E1 - Baseline CNN
...
...
...
### E2 – ...

