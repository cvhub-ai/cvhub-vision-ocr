from src.config.config_loader import get_paddleocr_config
from src.models.ocr_result import RecognitionResult
from src.ocr.recognizer.paddle_recognizer import PaddleRecognizer


def test_recognizer_loads_model():
    recognizer_config_data = get_paddleocr_config()["recognizer"]
    recognizer = PaddleRecognizer(recognizer_config_data)
    recognizer.load_model()
    assert recognizer.recognizer is not None


def test_recognizer_detects(rec_test_image):
    recognizer_config_data = get_paddleocr_config()["recognizer"]
    recognizer = PaddleRecognizer(recognizer_config_data)
    recognizer.load_model()
    result = recognizer.recognize(rec_test_image)
    print(result)
    assert result is not None
    assert isinstance(result, RecognitionResult)
    assert isinstance(result.text, str)
    assert isinstance(result.confidence, float)
