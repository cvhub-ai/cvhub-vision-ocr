from abc import ABC, abstractmethod

import numpy as np

from src.models.ocr_result import RecognitionResult


class BaseRecognizer(ABC):
    @abstractmethod
    def load_model(self) -> None:
        """Load the recognition model into memory."""
        ...

    @abstractmethod
    def recognize(
        self,
        image: np.ndarray,
    ) -> RecognitionResult | None:
        """
        Recognize text within detected regions.

        Args:
            image: Input Text ROI as numpy array (H, W, C) in BGR format.

        Returns:
            RecognitionResult with text and confidence or None
        """
        ...
