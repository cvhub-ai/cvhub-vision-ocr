import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.config.config_loader import OcrConfigLoader
from src.models.ocr_input import DetectorType, RecognizerType
from src.ocr.detector.paddle_detector import PaddleDetector
from src.ocr.engine.engine_factory import EngineFactory
from src.ocr.engine.ocr_engine import OcrEngine
from src.ocr.recognizer.paddle_recognizer import PaddleRecognizer

load_dotenv()


def test_factory_returns_paddle_engine():
    # Arrange
    config_path_value = os.getenv("CONFIG_PATH")
    if config_path_value is None:
        raise ValueError("Config path is missing")
    config_path_value = Path(config_path_value).resolve()
    ocr_config_loader = OcrConfigLoader(config_path_value)
    pipeline_config, detector_config, recognizer_config = ocr_config_loader.load()

    # Act
    engine = EngineFactory.create_engine(
        pipeline_config, detector_config, recognizer_config
    )

    # Assert
    assert isinstance(engine, OcrEngine)
    assert isinstance(engine.detector, PaddleDetector)
    assert isinstance(engine.recognizer, PaddleRecognizer)


def test_detector_invalid_type() -> None:
    with pytest.raises(ValueError):
        DetectorType("INVALID")


def test_recognizer_invalid_type() -> None:
    with pytest.raises(ValueError):
        RecognizerType("INVALID")
