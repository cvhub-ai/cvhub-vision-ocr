import logging
import os
from concurrent import futures
from pathlib import Path

import grpc
from dotenv import load_dotenv

from src.config.config_loader import OcrConfigLoader
from src.config.settings import HOST, LOG_LEVEL, PORT
from src.grpc_service.generated import ocr_pb2_grpc
from src.grpc_service.servicer import OcrServicer
from src.ocr.engine.engine_factory import EngineFactory
from src.ocr.pipeline.pipeline import OcrPipeline
from src.shared.logging.logging_config import LoggingConfig

logger = logging.getLogger(__name__)

load_dotenv()


def create_ocr_pipeline() -> OcrPipeline:
    config_path_value = os.getenv("CONFIG_PATH")
    if config_path_value is None:
        raise ValueError("Config path is missing")
    config_path_value = Path(config_path_value).resolve()
    ocr_config_loader = OcrConfigLoader(config_path_value)
    pipeline_config, detector_config, recognizer_config = ocr_config_loader.load()

    ocr_engine = EngineFactory.create_engine(
        pipeline_config, detector_config, recognizer_config
    )

    ocr_pipeline = OcrPipeline(ocr_engine)

    return ocr_pipeline


def create_server(ocr_pipeline: OcrPipeline) -> grpc.Server:

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))

    servicer = OcrServicer(ocr_pipeline=ocr_pipeline)

    ocr_pb2_grpc.add_OcrServiceServicer_to_server(servicer, server)

    address = f"{HOST}:{PORT}"

    server.add_insecure_port(address)
    return server


def serve() -> None:
    LoggingConfig.configure(
        log_level=LOG_LEVEL,
    )

    logger.info("Initializing OCR pipeline...")
    ocr_pipeline = create_ocr_pipeline()
    logger.info("OCR pipeline initialized.")

    server = create_server(ocr_pipeline=ocr_pipeline)

    server.start()

    logger.info(
        "OCR gRPC server started on %s:%s",
        HOST,
        PORT,
    )

    try:
        server.wait_for_termination()

    except KeyboardInterrupt:
        logger.info("Stopping OCR gRPC server...")

        server.stop(grace=5)

        logger.info("OCR gRPC server stopped.")
