from pathlib import Path

# Base directories
PROJECT_ROOT = Path("pneumonia_detection_app")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"          # Original immutable data
CLEANED_DATA_DIR = DATA_DIR / "cleaned"  # Processed data
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = MODELS_DIR / "reports"     # For all report outputs
CACHE_DIR = Path.home() / "pneumonia_image_cache"

# Data subdirectories (raw)
TRAIN_POS = RAW_DATA_DIR / "train" / "PNEUMONIA"
TRAIN_NEG = RAW_DATA_DIR / "train" / "NORMAL"
TEST_POS = RAW_DATA_DIR / "test" / "PNEUMONIA"
TEST_NEG = RAW_DATA_DIR / "test" / "NORMAL"

# Cleaned data paths (mirroring raw structure)
CLEANED_TRAIN_POS = CLEANED_DATA_DIR / "train" / "PNEUMONIA"
CLEANED_TRAIN_NEG = CLEANED_DATA_DIR / "train" / "NORMAL"
CLEANED_TEST_POS = CLEANED_DATA_DIR / "test" / "PNEUMONIA"
CLEANED_TEST_NEG = CLEANED_DATA_DIR / "test" / "NORMAL"

# Create essential directories
for path in [
    MODELS_DIR,
    REPORTS_DIR,
    CACHE_DIR,
    RAW_DATA_DIR,
    CLEANED_DATA_DIR,
    CLEANED_TRAIN_POS,
    CLEANED_TRAIN_NEG,
    CLEANED_TEST_POS,
    CLEANED_TEST_NEG
]:
    path.mkdir(parents=True, exist_ok=True)