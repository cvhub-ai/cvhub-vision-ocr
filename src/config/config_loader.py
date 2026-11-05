from pathlib import Path
from typing import Any

import yaml

from src.models.ocr_input import DetectorType, RecognizerType


class OcrConfigLoader:
    """Load, validate and parse OCR configuration."""

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path

    def load(
        self,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        """
        Load OCR configuration.

        Returns:
            A tuple containing:
            - pipeline configuration
            - selected detector configuration
            - selected recognizer configuration
        """

        config = self._load_yaml()

        pipeline_config = self._parse_pipeline_config(
            config=config,
        )

        detector_type = pipeline_config["detector"]["type"]
        recognizer_type = pipeline_config["recognizer"]["type"]

        detector_config = self._parse_detector_config(
            config=config,
            detector_type=detector_type,
        )

        recognizer_config = self._parse_recognizer_config(
            config=config,
            recognizer_type=recognizer_type,
        )

        return (
            pipeline_config,
            detector_config,
            recognizer_config,
        )

    def _load_yaml(self) -> dict[str, Any]:
        """Read the YAML configuration file."""

        if not self._config_path.exists():
            raise FileNotFoundError(
                f"OCR config file does not exist: {self._config_path}"
            )

        if not self._config_path.is_file():
            raise ValueError(f"OCR config path is not a file: {self._config_path}")

        with self._config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise TypeError("OCR config must be a mapping.")

        return config

    @staticmethod
    def _parse_pipeline_config(
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Parse and validate pipeline configuration."""

        pipeline_config = config.get("pipeline")

        if pipeline_config is None:
            raise ValueError("Pipeline config is missing.")

        if not isinstance(pipeline_config, dict):
            raise TypeError("Pipeline config must be a mapping.")

        detector_config = pipeline_config.get("detector")

        if detector_config is None:
            raise ValueError("Detector pipeline config is missing.")

        if not isinstance(detector_config, dict):
            raise TypeError("Detector pipeline config must be a mapping.")

        recognizer_config = pipeline_config.get("recognizer")

        if recognizer_config is None:
            raise ValueError("Recognizer pipeline config is missing.")

        if not isinstance(recognizer_config, dict):
            raise TypeError("Recognizer pipeline config must be a mapping.")

        detector_type_value = detector_config.get("type")

        if not detector_type_value:
            raise ValueError("Detector type is missing.")

        recognizer_type_value = recognizer_config.get("type")

        if not recognizer_type_value:
            raise ValueError("Recognizer type is missing.")

        detector_type = OcrConfigLoader._parse_detector_type(detector_type_value)

        recognizer_type = OcrConfigLoader._parse_recognizer_type(recognizer_type_value)

        detector_config["type"] = detector_type
        recognizer_config["type"] = recognizer_type

        return pipeline_config

    @staticmethod
    def _parse_detector_type(
        detector_type_value: Any,
    ) -> DetectorType:
        """Convert detector type string into DetectorType."""

        normalized_value = str(detector_type_value).strip().upper()

        try:
            return DetectorType(normalized_value)

        except ValueError as exception:
            raise ValueError(
                f"Unsupported detector type: {detector_type_value}"
            ) from exception

    @staticmethod
    def _parse_recognizer_type(
        recognizer_type_value: Any,
    ) -> RecognizerType:
        """Convert recognizer type string into RecognizerType."""

        normalized_value = str(recognizer_type_value).strip().upper()

        try:
            return RecognizerType(normalized_value)

        except ValueError as exception:
            raise ValueError(
                f"Unsupported recognizer type: {recognizer_type_value}"
            ) from exception

    @staticmethod
    def _parse_detector_config(
        config: dict[str, Any],
        detector_type: DetectorType,
    ) -> dict[str, Any]:
        """
        Extract configuration for the selected detector type.
        """

        detector_configs = config.get("detector")

        if detector_configs is None:
            raise ValueError("Detector config is missing.")

        if not isinstance(detector_configs, dict):
            raise TypeError("Detector config must be a mapping.")

        detector_config = detector_configs.get(detector_type.value.strip().lower())

        if detector_config is None:
            raise ValueError(
                f"Detector config for type '{detector_type.value}' is missing."
            )

        if not isinstance(detector_config, dict):
            raise TypeError(
                f"Detector config for type '{detector_type.value}' must be a mapping."
            )

        return detector_config

    @staticmethod
    def _parse_recognizer_config(
        config: dict[str, Any],
        recognizer_type: RecognizerType,
    ) -> dict[str, Any]:
        """
        Extract configuration for the selected recognizer type.
        """

        recognizer_configs = config.get("recognizer")

        if recognizer_configs is None:
            raise ValueError("Recognizer config is missing.")

        if not isinstance(recognizer_configs, dict):
            raise TypeError("Recognizer config must be a mapping.")

        recognizer_config = recognizer_configs.get(
            recognizer_type.value.strip().lower()
        )

        if recognizer_config is None:
            raise ValueError(
                f"Recognizer config for type '{recognizer_type.value}' is missing."
            )

        if not isinstance(recognizer_config, dict):
            raise TypeError(
                f"Recognizer config for type "
                f"'{recognizer_type.value}' "
                f"must be a mapping."
            )

        return recognizer_config
