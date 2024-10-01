# src/__init__.py

# Importing functions and classes for easier access
from .data_loader import load_dataset, split_dataset
from .preprocessing import apply_clahe, normalize_image
from .model import NestedUNet, AttentionUNet
from .train import train_model
from .utils import dice_score

# Optional: Package version
__version__ = "0.1.0"

# Optional: Logging setup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Package 'src' has been initialized.")
