import numpy as np
from PIL import Image
import os
from pathlib import Path
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

import random

# Source train data paths
train_dir = '/home/zyh/Fruit-Classifier/data/train'
# Augmented data paths
augmented_train_dir = '/home/zyh/Fruit-Classifier/data/train_augmented'

def augment_dataset(source_dir, target_dir, target_count=100, img_size=(224, 224)):
    """
    Expand the dataset to have the same number of images for each class
    
    Args:
        source_dir: Source data directory
        target_dir: Target data directory (augmented data)
        target_count: Target number of images per class
        img_size: Uniform image size (width, height)
    """
    
    # If target directory exists, delete it first
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    # Create target directory
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # Define data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 1.5],
        fill_mode='nearest'
    )
    
    # Get all classes
    class_names = os.listdir(source_dir)
    
    print("Starting dataset expansion...")
    print(f"Target: {target_count} images per class")
    print(f"Uniform image size: {img_size[0]}x{img_size[1]}")
    
    for class_name in class_names:
        class_source_dir = os.path.join(source_dir, class_name)
        class_target_dir = os.path.join(target_dir, class_name)
        
        if not os.path.isdir(class_source_dir):
            continue
            
        # Create target class directory
        Path(class_target_dir).mkdir(parents=True, exist_ok=True)
        
        # Get original image list
        image_files = [f for f in os.listdir(class_source_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"\nProcessing class: {class_name}")
        print(f"  Original image count: {len(image_files)}")
        
        # Copy original images to target directory (and unify size)
        copied_count = 0
        for img_file in image_files:
            src_path = os.path.join(class_source_dir, img_file)
            dst_path = os.path.join(class_target_dir, img_file)
            
            # Open image, unify size and save
            img = Image.open(src_path)
            img = img.convert('RGB')  # Ensure RGB format
            img = img.resize(img_size)  # Unify size
            img.save(dst_path)
            copied_count += 1
        
        # If augmentation is needed
        if len(image_files) < target_count:
            images_to_generate = target_count - len(image_files)
            print(f"  Need to generate {images_to_generate} augmented images")
            
            # Load and preprocess original images
            original_images = []
            for img_file in image_files:
                img_path = os.path.join(class_source_dir, img_file)
                img = Image.open(img_path)
                img = img.convert('RGB')  # Ensure RGB format
                img = img.resize(img_size)  # Unify size
                img_array = np.array(img)
                original_images.append(img_array)
            
            # Convert image list to numpy array
            original_images_array = np.array(original_images)
            print(f"  Original image array shape: {original_images_array.shape}")
            
            # Generate augmented images
            generated_count = 0
            # If original images are too few, use all of them
            if len(original_images) < 8:
                batch_size = len(original_images)
                # Special handling for only 1 image
                if batch_size == 1:
                    # Duplicate image to create batch
                    original_images_array = np.repeat(original_images_array, 8, axis=0)
                    batch_size = 8
            else:
                batch_size = min(32, len(original_images))
            
            # Set random seed for reproducibility
            np.random.seed(42)
            
            # Create data flow
            flow_generator = datagen.flow(
                original_images_array, 
                batch_size=batch_size,
                shuffle=True,
                seed=42
            )
            
            # Generate augmented images
            while generated_count < images_to_generate:
                # Get a batch of augmented images from data flow
                batch_images = next(flow_generator)
                
                for i in range(len(batch_images)):
                    if generated_count >= images_to_generate:
                        break
                        
                    # Convert back to image and save
                    aug_img = Image.fromarray(batch_images[i].astype('uint8'))
                    aug_filename = f"aug_{class_name}_{generated_count:04d}.jpg"
                    aug_path = os.path.join(class_target_dir, aug_filename)
                    aug_img.save(aug_path)
                    
                    generated_count += 1
                    
                    if generated_count % 10 == 0:
                        print(f"    Generated {generated_count}/{images_to_generate} augmented images")
            
            print(f"  Done! Total {copied_count + generated_count} images")
        else:
            print(f"  Class already has sufficient images, no augmentation needed")
    
    print(f"\nDataset expansion completed! Saved to: {target_dir}")
    
    # Verify results
    print("\nVerification results:")
    for class_name in class_names:
        class_target_dir = os.path.join(target_dir, class_name)
        if os.path.isdir(class_target_dir):
            count = len([f for f in os.listdir(class_target_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            print(f"  {class_name}: {count} images")

augment_dataset(train_dir, augmented_train_dir, target_count=76)