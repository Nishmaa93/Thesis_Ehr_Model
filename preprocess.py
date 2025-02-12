import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Directories
train_dir = r'images/train'
val_dir = r'images/val'
test_dir = r'images/test'

output_dirs = {
    'train': 'images/preprocessed/train',
    'val': 'images/preprocessed/val',
    'test': 'images/preprocessed/test'
}

# Ensure output directories exist
for dir_type, output_dir in output_dirs.items():
    Path(output_dir).mkdir(parents=True, exist_ok=True)

# Preprocessing and Enhancement Function
def preprocess_and_enhance(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to standard size (224x224)
    image = cv2.resize(image, (224, 224))
    
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = (clahe.apply((image * 255).astype(np.uint8))).astype(np.float32) / 255.0
    
    # Apply Gaussian Blur for noise reduction
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Save the processed image
    cv2.imwrite(output_path, (image * 255).astype(np.uint8))

# Process all images in a directory
def process_directory(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                
                # Construct output path (maintain directory structure)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Preprocess and enhance the image
                preprocess_and_enhance(input_path, output_path)

# Process train, val, and test directories
process_directory(train_dir, output_dirs['train'])
process_directory(val_dir, output_dirs['val'])
process_directory(test_dir, output_dirs['test'])


## EVALUATION
test_dir = 'images/test'

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Ensure the order of predictions matches the filenames
)

from tensorflow.keras.models import load_model

# Load the best model
model = load_model('best_model.keras') 

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test Loss: {test_loss:.2f}")