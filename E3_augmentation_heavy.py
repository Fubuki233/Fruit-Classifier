from data_preprocessing import get_preprocessor

IMG_SIZE = 224
BATCH_SIZE = 32

train_gen, val_gen, test_gen = get_preprocessor(method='heavy', img_size=IMG_SIZE, batch_size=BATCH_SIZE)

print(f"E3 - Heavy augmentation preprocessing completed")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Test samples: {test_gen.samples}")
print(f"Data generators ready for model training")
