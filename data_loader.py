# src/data_loader.py

import os
from glob import glob
from sklearn.model_selection import train_test_split

# Load the images and masks
def load_dataset(base_dir):
    # Fetch all case directories
    case_dirs = glob(os.path.join(base_dir, "*"))

    image_paths = []
    mask_paths = []

    for case_dir in case_dirs:
        # Get images within each case directory
        images = glob(os.path.join(case_dir, "*.tif"))

        for img in images:
            # Assuming the naming convention for masks has the "_mask" suffix
            mask_path = img.replace(".tif", "_mask.tif")
            # Check if the mask file exists
            if os.path.exists(mask_path):
                image_paths.append(img)
                mask_paths.append(mask_path)
            else:
                print(f"Warning: Mask not found for {img}. Expected at {mask_path}")

    return image_paths, mask_paths
