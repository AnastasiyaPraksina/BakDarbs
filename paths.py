from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PLOTS_EDA_DIR = BASE_DIR / "plots_eda"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "saved_models"
CLEANED_DATA = DATA_DIR / "cleaned_data.csv"
TRANSFORMED_DATA = DATA_DIR / "transformed_data.csv"
TRAIN_DATA = DATA_DIR / "train_data.csv"
VALIDATION_DATA = DATA_DIR / "validation_data.csv"
TEST_DATA = DATA_DIR / "test_data.csv"
LABELS = DATA_DIR / "labels.csv"
TRAIN_LABELS = DATA_DIR / "train_labels.csv"
VALIDATION_LABELS = DATA_DIR / "validation_labels.csv"
TEST_LABELS = DATA_DIR / "test_labels.csv"