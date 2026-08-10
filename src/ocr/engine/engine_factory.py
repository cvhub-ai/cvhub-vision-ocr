from src.models.ocr_input import DetectorType, RecognizerType
from src.ocr.engine.ocr_engine import OcrEngine


class EngineFactory:
    @staticmethod
    def create_engine(
        pipeline_config: dict, detector_config: dict, recognizer_config: dict
    ) -> OcrEngine:

        engine = OcrEngine()

        # Detector type (has converted to DetectorType) and detector config have already been validated
        # during configuration loading, so no additional validation is required here.
        detector_type = pipeline_config["detector"]["type"]
        EngineFactory._create_engine_detector(engine, detector_type, detector_config)

        # Recognizer type (has converted to RecognizerType) and recognizer config have already been validated
        # during configuration loading, so no additional validation is required here.
        recognizer_type = pipeline_config["recognizer"]["type"]
        EngineFactory._create_engine_recognizer(
            engine, recognizer_type, recognizer_config
        )

        return engine

    @staticmethod
    def _create_engine_detector(
        engine: OcrEngine, detector_type: DetectorType, detector_config: dict
    ):
        if detector_type == DetectorType.PADDLE:
            from src.ocr.detector.paddle_detector import PaddleDetector

            detector = PaddleDetector(detector_config)
            detector.load_model()
            engine.detector = detector
            return

        raise NotImplementedError(
            f"Detector type '{detector_type.value}' is not implemented."
        )

    @staticmethod
    def _create_engine_recognizer(
        engine: OcrEngine, recognizer_type: RecognizerType, recognizer_config: dict
    ):
        if recognizer_type == RecognizerType.PADDLE:
            from src.ocr.recognizer.paddle_recognizer import PaddleRecognizer

            recognizer = PaddleRecognizer(recognizer_config)
            recognizer.load_model()
            engine.recognizer = recognizer
            return

        raise NotImplementedError(
            f"Recognizer type '{recognizer_type.value}' is not implemented."
        )
