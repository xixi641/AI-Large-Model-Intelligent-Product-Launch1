from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'precessed'
LOG_DIR = ROOT_DIR / 'log'
MODELS_DIR = ROOT_DIR / 'models'
PRE_TRAINED_DIR = ROOT_DIR / 'pretrained'
SEQ_LEN = 50
NUMBER_CLASSES = 30
BATCH_SIZE = 42
LEARNING_RATE = 1e-5
EPOCHS = 10