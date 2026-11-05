import numpy as np
from paddleocr import TextRecognition

from src.models.ocr_result import RecognitionResult
from src.ocr.recognizer.base_recognizer import BaseRecognizer


class PaddleRecognizer(BaseRecognizer):
    def __init__(self, config_data: dict) -> None:
        super().__init__()
        self.model_name = config_data.get("model_name", None)
        self.model_dir = config_data.get("model_dir", None)
        self.device = config_data.get("device", "cpu")

    def load_model(self) -> None:
        if not self._check_detector_exist():
            return

        self.recognizer = TextRecognition(
            model_name=self.model_name, model_dir=self.model_dir, device=self.device
        )

    def recognize(
        self,
        image: np.ndarray,
    ) -> RecognitionResult | None:
        if not self._check_detector_exist():
            return

        paddle_rec_output = self.recognizer.predict(image)[0]
        rec_text = paddle_rec_output.get("rec_text", "")
        rec_score = paddle_rec_output.get("rec_score", 0)

        if rec_text == "" or rec_score == 0:
            return

        return RecognitionResult(rec_text, rec_score)

    def _check_detector_exist(self):
        return self.model_name != None and self.model_dir != None
