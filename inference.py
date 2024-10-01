# src/inference.py

import torch
import cv2
from src.model import AttentionUNet

# Load the best-performing model
model = AttentionUNet()
model.load_state_dict(torch.load("models/attention_unet_weights.pth"))
model.eval()  # Set to evaluation mode

# Inference function
def segment_image(image_path):
    image = cv2.imread(image_path)  # Load the input MRI image
    # Preprocess the image
    image = apply_clahe(image)
    image = normalize_image(image)
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
    
    # Post-process output (e.g., thresholding)
    mask = torch.sigmoid(output).cpu().numpy()[0, 0]  # Get the mask
    mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask

    return mask
