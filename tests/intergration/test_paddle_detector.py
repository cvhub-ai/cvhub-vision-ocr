from src.config.config_loader import get_paddleocr_config
from src.models.ocr_result import DetectionResult
from src.ocr.detector.paddle_detector import PaddleDetector


def test_detector_loads_model():
    detector_config_data = get_paddleocr_config()["detector"]
    detector = PaddleDetector(detector_config_data)
    detector.load_model()
    assert detector.detector is not None


def test_detector_detects(det_test_image):
    detector_config_data = get_paddleocr_config()["detector"]
    detector = PaddleDetector(detector_config_data)
    detector.load_model()
    results = detector.detect(det_test_image)
    print(results)
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(res, DetectionResult) for res in results)
    assert all(isinstance(res.bbox, list) for res in results)
    assert all(len(res.bbox) == 4 for res in results)
    assert all(isinstance(num, int) for res in results for pt in res.bbox for num in pt)
    assert all(isinstance(res.confidence, float) for res in results)
