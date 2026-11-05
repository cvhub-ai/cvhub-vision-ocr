import os

from dotenv import load_dotenv

load_dotenv()


SERVICE_NAME = os.getenv("SERVICE_NAME", "ocr")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "50051"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

DEVICE = os.getenv("DEVICE", "cuda")

CONFIG_PATH = os.getenv(
    "CONFIG_PATH",
    "config/ocr_model_config.yml",
)

MAX_IMAGE_SIZE_MB = int(
    os.getenv(
        "MAX_IMAGE_SIZE_MB",
        "20",
    )
)

MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

MAX_IMAGE_WIDTH = int(
    os.getenv(
        "MAX_IMAGE_WIDTH",
        "10000",
    )
)

MAX_IMAGE_HEIGHT = int(
    os.getenv(
        "MAX_IMAGE_HEIGHT",
        "10000",
    )
)

MAX_BATCH_SIZE = int(
    os.getenv(
        "MAX_BATCH_SIZE",
        "10",
    )
)

if not SERVICE_NAME.strip():
    raise ValueError("SERVICE_NAME must not be empty.")

if not 1 <= PORT <= 65535:
    raise ValueError("PORT must be between 1 and 65535.")

if DEVICE not in {"cpu", "cuda"}:
    raise ValueError("DEVICE must be either 'cpu' or 'cuda'.")

if MAX_IMAGE_SIZE_MB <= 0:
    raise ValueError("MAX_IMAGE_SIZE_MB must be greater than 0.")

if MAX_IMAGE_WIDTH <= 0:
    raise ValueError("MAX_IMAGE_WIDTH must be greater than 0.")

if MAX_IMAGE_HEIGHT <= 0:
    raise ValueError("MAX_IMAGE_HEIGHT must be greater than 0.")

if MAX_BATCH_SIZE <= 0:
    raise ValueError("MAX_BATCH_SIZE must be greater than 0.")
