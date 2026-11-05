import numpy as np

from src.models.ocr_result import DetectionResult, OcrResult, RecognitionResult
from src.ocr.detector.base_detector import BaseDetector
from src.ocr.recognizer.base_recognizer import BaseRecognizer
from src.shared.utils.utils import Utils


class OcrEngine:
    def __init__(self) -> None:
        self.detector: BaseDetector
        self.recognizer: BaseRecognizer

    def run(self, image: np.ndarray) -> list[OcrResult]:
        """
        Run the full OCR pipeline on a single image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format.

        Returns:
            List of OcrResult (DetectionResult + RecognitionResult) containing text, confidence, and bounding box.
        """
        ocr_results: list[OcrResult] = []

        detection_results: list[DetectionResult] = self.detector.detect(image)

        for detection_result in detection_results:
            if detection_result.confidence < 0.5:
                continue

            text_roi = Utils.extract_text_roi(image, detection_result.bbox)
            recognition_result: RecognitionResult | None = self.recognizer.recognize(
                text_roi
            )
            if recognition_result is None:
                continue

            ocr_results.append(
                OcrResult(det_result=detection_result, rec_result=recognition_result)
            )
        return ocr_results
