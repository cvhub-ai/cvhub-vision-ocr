import numpy as np
from paddleocr import TextDetection

from src.models.ocr_result import DetectionResult
from src.ocr.detector.base_detector import BaseDetector


class PaddleDetector(BaseDetector):
    def __init__(self, config_data: dict) -> None:
        super().__init__()
        self.model_name = config_data.get("model_name", None)
        self.model_dir = config_data.get("model_dir", None)
        self.device = config_data.get("device", "cpu")

    def load_model(self) -> None:
        if not self._check_detector_exist():
            return

        self.detector = TextDetection(
            model_name=self.model_name, model_dir=self.model_dir, device=self.device
        )

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        if not self._check_detector_exist():
            return []

        paddle_det_output = self.detector.predict(image)[0]
        det_polys = paddle_det_output.get("dt_polys", None)
        det_scores = paddle_det_output.get("dt_scores", None)

        if det_polys is None or det_scores is None:
            return []

        detection_results = []
        for det_poly, det_score in zip(det_polys, det_scores):
            detection_results.append(DetectionResult(det_poly.tolist(), det_score))
        return detection_results

    def _check_detector_exist(self):
        return self.model_name != None and self.model_dir != None
