# src/utils.py

import torch

# Calculate the DICE score for the predicted segmentation mask
def dice_score(pred, target):
    smooth = 1.0
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
