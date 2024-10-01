# src/train.py

class BrainMRIDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image and mask
        image = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Debugging: Check if image and mask are loaded correctly
        if image is None:
            print(f"Error loading image: {self.image_paths[idx]}")
        if mask is None:
            print(f"Error loading mask: {self.mask_paths[idx]}")

        # Apply preprocessing if images are successfully loaded
        if image is not None and mask is not None:
            image = apply_clahe(image)
            image = normalize_image(image)
            return image, mask
        else:
            raise RuntimeError(f"Failed to load image/mask pair: {self.image_paths[idx]}, {self.mask_paths[idx]}")
