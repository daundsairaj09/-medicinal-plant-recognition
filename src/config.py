import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "val")
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
KNOWLEDGE_BASE = os.path.join(BASE_DIR, "knowledge_base", "plant_uses.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

print("✅ Config loaded successfully")
print("BASE_DIR:", BASE_DIR)
print("MODEL_DIR:", MODEL_DIR)
