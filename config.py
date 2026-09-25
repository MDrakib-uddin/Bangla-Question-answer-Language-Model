from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_NAME = "google/mt5-small"
DATASET_NAME = "rasheduzzaman/Bangla_question_answer_pair_70K_dataset"

OUTPUT_DIR = PROJECT_ROOT / "qa-mt5"
FINAL_MODEL_DIR = PROJECT_ROOT / "qa-mt5-final"

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 4

HF_REPO_ID = "rakib730/bangla-qa-mt5"
