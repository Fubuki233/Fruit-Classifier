from data_preprocessing import get_preprocessor

IMG_SIZE = 128
BATCH_SIZE = 32

train_gen, val_gen, test_gen = get_preprocessor(method='moderate', img_size=IMG_SIZE, batch_size=BATCH_SIZE)

print(f"E6 - Smaller image (128x128) preprocessing completed")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Test samples: {test_gen.samples}")
print(f"Data generators ready for model training")
