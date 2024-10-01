import cv2
import numpy as np

# Apply CLAHE to enhance the FLAIR channel (second channel)
def apply_clahe(image):
    flair_channel = image[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_flair = clahe.apply(flair_channel)
    image[:, :, 1] = enhanced_flair  # Replace the original FLAIR channel
    return image

# Normalize the image
def normalize_image(image):
    return image.astype(np.float32) / 255.0
