import numpy as np

from src.models.ocr_result import DetectionResult, RecognitionResult
from src.ocr.engine.ocr_engine import OcrEngine
from src.shared.utils.utils import Utils


def test_run_returns_ocr_results(mocker):
    # Arrange
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    text_roi = np.zeros((20, 50, 3), dtype=np.uint8)

    detection_result = DetectionResult(
        bbox=[[10, 10], [60, 10], [60, 30], [10, 30]],
        confidence=0.9,
    )

    recognition_result = RecognitionResult(
        text="Hello",
        confidence=0.95,
    )

    detector = mocker.Mock()
    detector.detect.return_value = [detection_result]

    recognizer = mocker.Mock()
    recognizer.recognize.return_value = recognition_result

    mock_extract_roi = mocker.patch.object(
        Utils,
        "extract_text_roi",
        return_value=text_roi,
    )

    engine = OcrEngine()
    engine.detector = detector
    engine.recognizer = recognizer

    # Act
    results = engine.run(image)

    # Assert
    assert len(results) == 1
    assert results[0].det_result is detection_result
    assert results[0].rec_result is recognition_result

    detector.detect.assert_called_once_with(image)
    mock_extract_roi.assert_called_once_with(
        image,
        detection_result.bbox,
    )
    recognizer.recognize.assert_called_once_with(text_roi)
