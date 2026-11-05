import cv2
import pytest

from src.config.config_loader import get_paddleocr_config
from src.models.ocr_input import DetectorType, RecognizerType
from src.ocr.engine.engine_factory import EngineFactory
from src.ocr.engine.ocr_engine import OcrEngine


@pytest.fixture
def det_test_image():
    return cv2.imread("tests/assets/images/det_test.png")


@pytest.fixture
def rec_test_image():
    return cv2.imread("tests/assets/images/rec_test.png")


@pytest.fixture
def engine() -> OcrEngine:
    config_data = get_paddleocr_config()
    engine = EngineFactory.create_engine(
        DetectorType.PADDLE, RecognizerType.PADDLE, config_data
    )
    return engine
