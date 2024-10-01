# src/config.py

class Config:
    DATA_DIR = "data/"
    TRAIN_IMAGES_DIR = DATA_DIR + "images/"
    TRAIN_MASKS_DIR = DATA_DIR + "masks/"
    
    # File paths for training/testing sets
    TRAIN_IMAGE_PATHS = "train_images.txt"
    TRAIN_MASKS_PATHS = "train_masks.txt"
    TEST_IMAGE_PATHS = "test_images.txt"
    TEST_MASKS_PATHS = "test_masks.txt"
    
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # Model save paths
    MODEL_SAVE_DIR = "models/"
