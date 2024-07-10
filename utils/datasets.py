import os
import shutil
import random

def splitter(image_dir, label_dir, output_dir, split_ratio=0.8, seed=42):
    train_image_dir = os.path.join(output_dir, 'images/train')
    val_image_dir = os.path.join(output_dir, 'images/val')
    train_label_dir = os.path.join(output_dir, 'labels/train')
    val_label_dir = os.path.join(output_dir, 'labels/val')
    
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    random.seed(seed)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    val_images = images[split_index:]
    
    for img in train_images:
        shutil.move(os.path.join(image_dir, img), os.path.join(train_image_dir, img))

        label = img.replace('.png', '.txt')
        shutil.move(os.path.join(label_dir, label), os.path.join(train_label_dir, label))
    
    for img in val_images:
        shutil.move(os.path.join(image_dir, img), os.path.join(val_image_dir, img))

        label = img.replace('.png', '.txt')
        shutil.move(os.path.join(label_dir, label), os.path.join(val_label_dir, label))