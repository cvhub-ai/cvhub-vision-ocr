from abc import ABC, abstractmethod

import numpy as np

from src.models.ocr_result import DetectionResult


class BaseDetector(ABC):
    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model into memory."""
        ...

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        """
        Detect text regions in the image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format.

        Returns:
            List of DetectionResult containing bounding boxes and confidence scores.
        """
        ...
