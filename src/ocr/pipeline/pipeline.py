import numpy as np

from src.models.ocr_result import OcrResult
from src.ocr.engine.ocr_engine import OcrEngine
from src.ocr.preprocess.image_processor import Preprocessor


class OcrPipeline:
    def __init__(self, ocr_engine: OcrEngine) -> None:
        self._pre_processor = Preprocessor()
        self._ocr_engine = ocr_engine

    def process(self, image: np.ndarray) -> list[OcrResult]:
        # step 1: pre processing
        self._pre_processor.set_image(image)
        image_processed = (
            self._pre_processor.gaussian_denoise().laplacian_sharpen().result()
        )

        # setp 2: ocr engine run
        ocrResults = self._ocr_engine.run(image_processed)

        return ocrResults
