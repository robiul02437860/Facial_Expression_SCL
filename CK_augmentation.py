import os
import cv2
import albumentations as A
from tqdm import tqdm

# Define augmentation pipeline
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussNoise(p=0.2),
])

# Paths
input_dir = "Cropped/"  # Path to original dataset
output_dir = 'Cropped/'  # Path to save augmented dataset
os.makedirs(output_dir, exist_ok=True)

# Loop through each class folder
for class_folder in os.listdir(input_dir):
    class_input_path = os.path.join(input_dir, class_folder)
    class_output_path = os.path.join(output_dir, class_folder)
    os.makedirs(class_output_path, exist_ok=True)
    
    # Loop through each image in the folder
    for image_name in tqdm(os.listdir(class_input_path), desc=f"Processing {class_folder}"):
        image_path = os.path.join(class_input_path, image_name)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Skipping {image_path}, not a valid image.")
            continue
        
        # Apply augmentations
        augmented = augmentations(image=image)
        augmented_image = augmented['image']
        
        # Save augmented image
        output_image_path = os.path.join(class_output_path, f"aug_{image_name}")
        cv2.imwrite(output_image_path, augmented_image)
