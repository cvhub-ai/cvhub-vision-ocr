import logging
from collections.abc import Sequence
from time import perf_counter

import cv2
import grpc
import numpy as np

from src.__version__ import PIPELINE_VERSION, SERVICE_VERSION
from src.config.settings import (
    MAX_BATCH_SIZE,
    MAX_IMAGE_HEIGHT,
    MAX_IMAGE_SIZE_BYTES,
    MAX_IMAGE_WIDTH,
    SERVICE_NAME,
)
from src.grpc_service.generated import ocr_pb2, ocr_pb2_grpc
from src.models.ocr_result import OcrResult
from src.ocr.pipeline.pipeline import OcrPipeline
from src.shared.constants.constants import SUPPORTED_IMAGE_MIME_TYPES

logger = logging.getLogger(__name__)


class OcrServicer(ocr_pb2_grpc.OcrServiceServicer):
    def __init__(self, ocr_pipeline: OcrPipeline) -> None:
        self._ocr_pipeline = ocr_pipeline

    def ProcessSingleImage(
        self, request: ocr_pb2.OcrSingleProcessRequest, context: grpc.ServicerContext
    ) -> ocr_pb2.OcrSingleProcessResponse:
        try:
            self._validate_image_input(request.image)

            image = self._decode_image(request.image)

            start_time = perf_counter()

            ocr_results = self._ocr_pipeline.process(image)

            processing_time_ms = int((perf_counter() - start_time) * 1000)

            return ocr_pb2.OcrSingleProcessResponse(
                context=self._build_response_context(request_context=request.context),
                result=self._build_image_result(
                    image=image,
                    ocr_results=ocr_results,
                    processing_time_ms=processing_time_ms,
                    pipeline_version=PIPELINE_VERSION,
                ),
            )
        except ValueError as exception:
            logger.warning("Invalid single-image OCR request: %s", exception)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exception))
        except Exception:
            logger.exception("Failed to process single OCR request")
            context.abort(grpc.StatusCode.INTERNAL, "Internal OCR processing error.")

    def ProcessBatchImages(
        self, request: ocr_pb2.OcrBatchProcessRequest, context: grpc.ServicerContext
    ) -> ocr_pb2.OcrBatchProcessResponse:
        try:
            self._validate_batch_images(request.images)

            item_results: list[ocr_pb2.BatchItemResult] = []

            for image_input in request.images:
                item_results.append(self._process_batch_item(image_input))

            return ocr_pb2.OcrBatchProcessResponse(
                context=self._build_response_context(request_context=request.context),
                results=item_results,
            )
        except ValueError as exception:
            logger.warning("Invalid images OCR request: %s", exception)
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                str(exception),
            )

        except Exception:
            logger.exception("Unexpected error while processing batch images.")
            context.abort(
                grpc.StatusCode.INTERNAL,
                "Internal OCR batch processing error.",
            )

    def _process_batch_item(
        self,
        image_input: ocr_pb2.ImageInput,
    ) -> ocr_pb2.BatchItemResult:
        try:
            self._validate_image_input(image_input)

            image = self._decode_image(image_input)

            start_time = perf_counter()
            ocr_results = self._ocr_pipeline.process(image)
            processing_time_ms = int((perf_counter() - start_time) * 1000)

            return self._build_batch_item_result(
                image_input=image_input,
                image=image,
                ocr_results=ocr_results,
                processing_time_ms=processing_time_ms,
                pipeline_version=PIPELINE_VERSION,
            )

        except ValueError as exception:
            return ocr_pb2.BatchItemResult(
                image_id=image_input.image_id,
                error=ocr_pb2.ItemError(
                    code=ocr_pb2.INVALID_IMAGE,
                    message=str(exception),
                ),
            )

        except Exception:
            logger.exception(
                "Failed to process batch item. image_id=%s",
                image_input.image_id,
            )

            return ocr_pb2.BatchItemResult(
                image_id=image_input.image_id,
                error=ocr_pb2.ItemError(
                    code=ocr_pb2.PROCESSING_FAILED,
                    message="OCR processing failed.",
                ),
            )

    def GetCapabilities(
        self,
        request: ocr_pb2.GetCapabilitiesRequest,
        context: grpc.ServicerContext,
    ) -> ocr_pb2.GetCapabilitiesResponse:
        return ocr_pb2.GetCapabilitiesResponse(
            context=self._build_response_context(request.context),
            service_name=SERVICE_NAME,
            service_version=SERVICE_VERSION,
            pipeline_version=PIPELINE_VERSION,
            supported_mime_types=sorted(SUPPORTED_IMAGE_MIME_TYPES),
            max_image_width=MAX_IMAGE_WIDTH,
            max_image_height=MAX_IMAGE_HEIGHT,
            max_image_size_bytes=MAX_IMAGE_SIZE_BYTES,
            max_batch_size=MAX_BATCH_SIZE,
            supports_batch_processing=True,
        )

    @staticmethod
    def _validate_image_input(image_input: ocr_pb2.ImageInput) -> None:
        if not image_input.image_id.strip():
            raise ValueError("Image ID must not be empty")

        if not image_input.data:
            raise ValueError("Image data must not be empty")

        if len(image_input.data) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(
                f"Image data exceeds the maximum allowed size of "
                f"{MAX_IMAGE_SIZE_BYTES} bytes. "
                f"image_id={image_input.image_id}"
            )

        mime_type = image_input.mime_type.strip().lower()
        if not mime_type:
            raise ValueError(
                f"MIME type must not be empty. image_id={image_input.image_id}"
            )
        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError(
                f"Unsupported MIME type '{image_input.mime_type}'. "
                f"image_id={image_input.image_id}"
            )

    @staticmethod
    def _validate_batch_images(input_images: Sequence[ocr_pb2.ImageInput]) -> None:
        if not input_images:
            raise ValueError("Image batch must contain at least one image.")

        if len(input_images) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size exceeds the maximum allowed size of {MAX_BATCH_SIZE}."
            )

        image_ids = [image.image_id.strip() for image in input_images]

        non_empty_image_ids = [image_id for image_id in image_ids if image_id]

        if len(non_empty_image_ids) != len(set(non_empty_image_ids)):
            raise ValueError("Image IDs must be unique within a batch request.")

    @staticmethod
    def _decode_image(image_input: ocr_pb2.ImageInput) -> np.ndarray:
        encoded_image = np.frombuffer(image_input.data, dtype=np.uint8)
        image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(
                f"Image data could not be decoded. image_id={image_input.image_id}"
            )

        height, width = image.shape[2]
        if width > MAX_IMAGE_WIDTH:
            raise ValueError(
                f"Image width exceeds the maximum allowed width of "
                f"{MAX_IMAGE_WIDTH} pixels. "
                f"actual_width={width}, "
                f"image_id={image_input.image_id}"
            )
        if height > MAX_IMAGE_HEIGHT:
            raise ValueError(
                f"Image height exceeds the maximum allowed height of "
                f"{MAX_IMAGE_HEIGHT} pixels. "
                f"actual_height={height}, "
                f"image_id={image_input.image_id}"
            )
        return image

    @staticmethod
    def _build_response_context(
        request_context: ocr_pb2.RequestContext,
    ) -> ocr_pb2.ResponseContext:
        return ocr_pb2.ResponseContext(
            request_id=request_context.request_id, client_id=request_context.client_id
        )

    @staticmethod
    def _build_image_result(
        image: np.ndarray,
        ocr_results: list[OcrResult],
        processing_time_ms: int,
        pipeline_version: str,
    ) -> ocr_pb2.ImageResult:
        proto_regions = []
        for ocr_result in ocr_results:
            vertices = ocr_result.det_result.vertices
            det_confidence = ocr_result.det_result.confidence

            text = ocr_result.rec_result.text
            rec_confidence = ocr_result.rec_result.confidence

            proto_regions.append(
                ocr_pb2.TextRegion(
                    bounding_box=ocr_pb2.DetectionRegion(
                        top_left=ocr_pb2.Point(
                            x=int(vertices[0][0]), y=int(vertices[0][1])
                        ),
                        top_right=ocr_pb2.Point(
                            x=int(vertices[1][0]), y=int(vertices[1][1])
                        ),
                        bottom_right=ocr_pb2.Point(
                            x=int(vertices[2][0]), y=int(vertices[2][1])
                        ),
                        bottom_left=ocr_pb2.Point(
                            x=int(vertices[3][0]), y=int(vertices[3][1])
                        ),
                    ),
                    detection_confidence=float(det_confidence),
                    text=text,
                    recognition_confidence=float(rec_confidence),
                )
            )
        return ocr_pb2.ImageResult(
            image_width=image.shape[1],
            image_height=image.shape[0],
            regions=proto_regions,
            metadata=ocr_pb2.ProcessingMetadata(
                processing_time_ms=processing_time_ms, pipeline_version=pipeline_version
            ),
        )

    @staticmethod
    def _build_batch_item_result(
        image_input: ocr_pb2.ImageInput,
        image: np.ndarray,
        ocr_results: list[OcrResult],
        processing_time_ms: int,
        pipeline_version: str,
    ) -> ocr_pb2.BatchItemResult:
        image_result = OcrServicer._build_image_result(
            image, ocr_results, processing_time_ms, pipeline_version
        )
        return ocr_pb2.BatchItemResult(
            image_id=image_input.image_id, result=image_result
        )
