from pathlib import Path

import pytest
import yaml

from src.config.config_loader import OcrConfigLoader
from src.models.ocr_input import DetectorType, RecognizerType


@pytest.fixture
def valid_config() -> dict:
    return {
        "pipeline": {
            "detector": {
                "type": "PADDLE",
            },
            "recognizer": {
                "type": "PADDLE",
            },
        },
        "detector": {
            "paddle": {
                "model_name": "test_detector",
                "model_dir": "models/detector",
                "device": "cpu",
            }
        },
        "recognizer": {
            "paddle": {
                "model_name": "test_recognizer",
                "model_dir": "models/recognizer",
                "device": "cpu",
            }
        },
    }


@pytest.fixture
def config_path(
    tmp_path: Path,
    valid_config: dict,
) -> Path:
    path = tmp_path / "ocr_config.yml"

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            valid_config,
            file,
        )

    return path


def test_load_returns_configs(
    config_path: Path,
) -> None:
    loader = OcrConfigLoader(config_path)

    pipeline_config, detector_config, recognizer_config = loader.load()

    assert isinstance(pipeline_config, dict)
    assert isinstance(detector_config, dict)
    assert isinstance(recognizer_config, dict)


def test_load_converts_detector_type_to_enum(
    config_path: Path,
) -> None:
    loader = OcrConfigLoader(config_path)

    pipeline_config, _, _ = loader.load()

    assert pipeline_config["detector"]["type"] == DetectorType.PADDLE


def test_load_converts_recognizer_type_to_enum(
    config_path: Path,
) -> None:
    loader = OcrConfigLoader(config_path)

    pipeline_config, _, _ = loader.load()

    assert pipeline_config["recognizer"]["type"] == RecognizerType.PADDLE


def test_load_returns_selected_detector_config(
    config_path: Path,
) -> None:
    loader = OcrConfigLoader(config_path)

    _, detector_config, _ = loader.load()

    assert detector_config["model_name"] == "test_detector"
    assert detector_config["model_dir"] == "models/detector"
    assert detector_config["device"] == "cpu"


def test_load_returns_selected_recognizer_config(
    config_path: Path,
) -> None:
    loader = OcrConfigLoader(config_path)

    _, _, recognizer_config = loader.load()

    assert recognizer_config["model_name"] == "test_recognizer"
    assert recognizer_config["model_dir"] == "models/recognizer"
    assert recognizer_config["device"] == "cpu"


def test_load_raises_error_when_config_file_missing(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "missing.yml"

    loader = OcrConfigLoader(config_path)

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_load_raises_error_when_pipeline_missing(
    tmp_path: Path,
) -> None:
    config = {
        "detector": {},
        "recognizer": {},
    }

    config_path = tmp_path / "ocr_config.yml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    loader = OcrConfigLoader(config_path)

    with pytest.raises(ValueError):
        loader.load()


def test_load_raises_error_for_unsupported_detector_type(
    tmp_path: Path,
    valid_config: dict,
) -> None:
    valid_config["pipeline"]["detector"]["type"] = "UNKNOWN"

    config_path = tmp_path / "ocr_config.yml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(valid_config, file)

    loader = OcrConfigLoader(config_path)

    with pytest.raises(
        ValueError,
        match="Unsupported detector type",
    ):
        loader.load()


def test_load_raises_error_for_unsupported_recognizer_type(
    tmp_path: Path,
    valid_config: dict,
) -> None:
    valid_config["pipeline"]["recognizer"]["type"] = "UNKNOWN"

    config_path = tmp_path / "ocr_config.yml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(valid_config, file)

    loader = OcrConfigLoader(config_path)

    with pytest.raises(
        ValueError,
        match="Unsupported recognizer type",
    ):
        loader.load()


def test_load_raises_error_when_detector_config_missing(
    tmp_path: Path,
    valid_config: dict,
) -> None:
    valid_config["detector"].pop("paddle")

    config_path = tmp_path / "ocr_config.yml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(valid_config, file)

    loader = OcrConfigLoader(config_path)

    with pytest.raises(
        ValueError,
        match="Detector config for type",
    ):
        loader.load()


def test_load_raises_error_when_recognizer_config_missing(
    tmp_path: Path,
    valid_config: dict,
) -> None:
    valid_config["recognizer"].pop("paddle")

    config_path = tmp_path / "ocr_config.yml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(valid_config, file)

    loader = OcrConfigLoader(config_path)

    with pytest.raises(
        ValueError,
        match="Recognizer config for type",
    ):
        loader.load()
