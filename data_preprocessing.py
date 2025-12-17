import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from pathlib import Path

def load_from_directory(directory):
    """
    
    Args:
        directory: path to data directory
        
    Returns:
        dict with class names and image counts
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory not found: {directory}")
    
    data_info = {
        'path': directory,
        'classes': [],
        'counts': {}
    }
    
    for item in os.listdir(directory):
        class_path = os.path.join(directory, item)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            if images:
                data_info['classes'].append(item)
                data_info['counts'][item] = len(images)
    
    data_info['total'] = sum(data_info['counts'].values())
    data_info['num_classes'] = len(data_info['classes'])
    
    return data_info


def get_preprocessor(method='baseline', img_size=224, batch_size=32, train_dir='data/train', test_dir='data/test', val_split=0.2):
    """
    Get data generators with different preprocessing methods
    
    Args:
        method: 'baseline', 'light', 'moderate', 'minimal', 'heavy', 'heavy_01', 'heavy_02', 'heavy_03', 'color_boost', 'mixed'
        img_size: image dimension (default 224)
        batch_size: batch size for training (default 32)
        train_dir: training data directory
        test_dir: test data directory
        val_split: validation split ratio (default 0.2)
    
    Returns:
        train_gen, val_gen, test_gen
    """
    if method == 'baseline':
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split)
        
    elif method == 'light':
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            horizontal_flip=True,
            validation_split=val_split
        )
       
    elif method == 'moderate':
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=val_split
        )
        
    elif method == 'minimal':
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            zoom_range=0.1,
            validation_split=val_split
        )

    elif method == 'heavy':
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            validation_split=val_split
        )

    elif method == 'heavy_01':
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True, # no vertical flip
            fill_mode='nearest',
            validation_split=val_split
        )

    elif method == 'heavy_02':
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True, # no vertical flip
            fill_mode='nearest',
            validation_split=0.2 # explicit specification of 0.2
        )

    elif method == 'heavy_03':
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=45, # max rotation
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2], 
            validation_split=val_split
        )

    elif method == 'color_boost':
        train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],   # vary lighting
        channel_shift_range=15.0,      # vary colors
        validation_split=val_split
    )
        
    elif method == 'mixed':
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=0.25,
            zoom_range=0.35,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.5, 1.5],
            channel_shift_range=35.0,
            fill_mode='nearest',
            validation_split=val_split
        )

    else:
        raise ValueError(f"Unknown method: {method}. Choose from: baseline, light, moderate, minimal, heavy, heavy_01, heavy_02, heavy_03, color_boost, mixed")
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen

def list_methods():
    methods = {
        'baseline': 'No augmentation, only rescaling',
        'light': 'Rotation (15°) + Horizontal flip',
        'moderate': 'Balanced augmentation (rotation, shift, zoom, flip)',
        'minimal': 'Minimal augmentation (slight rotation, zoom)',
        'heavy': 'Full augmentation (rotation, shift, zoom, shear, brightness, flips)',
        'heavy_01': 'Modified augmentation from heavy (rotation increased to 45 deg, shift, zoom, shear, no brightness, no vertical flip)',
        'heavy_02': 'Modified augmentation from heavy_01, but explicitly specified validation_split = 0.2' ,
        'heavy_03': 'Modified augmentation from heavy_01 and heavy (rotation increased to 45 deg, shift, zoom, shear, brightness, flips)',
        'color_boost': 'Moderate geometry + brightness [0.7–1.3] + channel shift',
        'mixed': 'Intensive mix: strong geo + brightness [0.5–1.5] + channel shift'
    }
    print("Available preprocessing methods:")
    for key, desc in methods.items():
        print(f"  - {key:12s}: {desc}")
    return methods

if __name__ == "__main__":
    import sys
    import numpy as np
    from PIL import Image
    
    output_dir = 'data/preprocess'
    os.makedirs(output_dir, exist_ok=True)
    
    methods = ['baseline', 'light', 'moderate', 'minimal', 'heavy', 'heavy_01', 'heavy_02', 'heavy_03', 'color_boost', 'mixed']
    samples_per_method = 5
    
    for method in methods:
        print(f"\nProcessing: {method}")
        method_dir = os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        
        train_gen, _, _ = get_preprocessor(method=method, img_size=224, batch_size=samples_per_method)
        
        X_batch, y_batch = next(train_gen)
        
        for i in range(min(samples_per_method, len(X_batch))):
            img_array = (X_batch[i] * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            
            class_idx = np.argmax(y_batch[i])
            class_name = list(train_gen.class_indices.keys())[class_idx]
            
            filename = f"{method}_{class_name}_{i+1}.jpg"
            filepath = os.path.join(method_dir, filename)
            img.save(filepath)
        
        print(f"  Saved {samples_per_method} samples to {method_dir}")
    
    print("\n" + "="*60)
