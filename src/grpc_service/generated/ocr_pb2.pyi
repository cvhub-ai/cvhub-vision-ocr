from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class ItemErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ITEM_ERROR_CODE_UNSPECIFIED: _ClassVar[ItemErrorCode]
    INVALID_IMAGE: _ClassVar[ItemErrorCode]
    UNSUPPORTED_FORMAT: _ClassVar[ItemErrorCode]
    IMAGE_TOO_LARGE: _ClassVar[ItemErrorCode]
    PROCESSING_FAILED: _ClassVar[ItemErrorCode]

ITEM_ERROR_CODE_UNSPECIFIED: ItemErrorCode
INVALID_IMAGE: ItemErrorCode
UNSUPPORTED_FORMAT: ItemErrorCode
IMAGE_TOO_LARGE: ItemErrorCode
PROCESSING_FAILED: ItemErrorCode

class RequestContext(_message.Message):
    __slots__ = ("client_id", "request_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    client_id: str
    def __init__(
        self, request_id: str | None = ..., client_id: str | None = ...
    ) -> None: ...

class ResponseContext(_message.Message):
    __slots__ = ("client_id", "request_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    client_id: str
    def __init__(
        self, request_id: str | None = ..., client_id: str | None = ...
    ) -> None: ...

class OcrSingleProcessRequest(_message.Message):
    __slots__ = ("context", "image", "options")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    image: ImageInput
    options: OcrOptions
    def __init__(
        self,
        context: RequestContext | _Mapping | None = ...,
        image: ImageInput | _Mapping | None = ...,
        options: OcrOptions | _Mapping | None = ...,
    ) -> None: ...

class OcrSingleProcessResponse(_message.Message):
    __slots__ = ("context", "result")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    context: ResponseContext
    result: ImageResult
    def __init__(
        self,
        context: ResponseContext | _Mapping | None = ...,
        result: ImageResult | _Mapping | None = ...,
    ) -> None: ...

class OcrBatchProcessRequest(_message.Message):
    __slots__ = ("context", "images", "options")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    IMAGES_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    images: _containers.RepeatedCompositeFieldContainer[ImageInput]
    options: OcrOptions
    def __init__(
        self,
        context: RequestContext | _Mapping | None = ...,
        images: _Iterable[ImageInput | _Mapping] | None = ...,
        options: OcrOptions | _Mapping | None = ...,
    ) -> None: ...

class OcrBatchProcessResponse(_message.Message):
    __slots__ = ("context", "results")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    context: ResponseContext
    results: _containers.RepeatedCompositeFieldContainer[BatchItemResult]
    def __init__(
        self,
        context: ResponseContext | _Mapping | None = ...,
        results: _Iterable[BatchItemResult | _Mapping] | None = ...,
    ) -> None: ...

class BatchItemResult(_message.Message):
    __slots__ = ("error", "image_id", "result")
    IMAGE_ID_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    image_id: str
    result: ImageResult
    error: ItemError
    def __init__(
        self,
        image_id: str | None = ...,
        result: ImageResult | _Mapping | None = ...,
        error: ItemError | _Mapping | None = ...,
    ) -> None: ...

class ImageInput(_message.Message):
    __slots__ = ("data", "filename", "image_id", "mime_type")
    IMAGE_ID_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    image_id: str
    data: bytes
    mime_type: str
    filename: str
    def __init__(
        self,
        image_id: str | None = ...,
        data: bytes | None = ...,
        mime_type: str | None = ...,
        filename: str | None = ...,
    ) -> None: ...

class OcrOptions(_message.Message):
    __slots__ = ("language",)
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    language: str
    def __init__(self, language: str | None = ...) -> None: ...

class ImageResult(_message.Message):
    __slots__ = ("image_height", "image_width", "metadata", "regions")
    IMAGE_WIDTH_FIELD_NUMBER: _ClassVar[int]
    IMAGE_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    REGIONS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    image_width: int
    image_height: int
    regions: _containers.RepeatedCompositeFieldContainer[TextRegion]
    metadata: ProcessingMetadata
    def __init__(
        self,
        image_width: int | None = ...,
        image_height: int | None = ...,
        regions: _Iterable[TextRegion | _Mapping] | None = ...,
        metadata: ProcessingMetadata | _Mapping | None = ...,
    ) -> None: ...

class TextRegion(_message.Message):
    __slots__ = (
        "bounding_box",
        "detection_confidence",
        "recognition_confidence",
        "text",
    )
    BOUNDING_BOX_FIELD_NUMBER: _ClassVar[int]
    DETECTION_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    RECOGNITION_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    bounding_box: DetectionRegion
    detection_confidence: float
    text: str
    recognition_confidence: float
    def __init__(
        self,
        bounding_box: DetectionRegion | _Mapping | None = ...,
        detection_confidence: float | None = ...,
        text: str | None = ...,
        recognition_confidence: float | None = ...,
    ) -> None: ...

class DetectionRegion(_message.Message):
    __slots__ = ("bottom_left", "bottom_right", "top_left", "top_right")
    TOP_LEFT_FIELD_NUMBER: _ClassVar[int]
    TOP_RIGHT_FIELD_NUMBER: _ClassVar[int]
    BOTTOM_RIGHT_FIELD_NUMBER: _ClassVar[int]
    BOTTOM_LEFT_FIELD_NUMBER: _ClassVar[int]
    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point
    def __init__(
        self,
        top_left: Point | _Mapping | None = ...,
        top_right: Point | _Mapping | None = ...,
        bottom_right: Point | _Mapping | None = ...,
        bottom_left: Point | _Mapping | None = ...,
    ) -> None: ...

class Point(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    def __init__(self, x: int | None = ..., y: int | None = ...) -> None: ...

class ProcessingMetadata(_message.Message):
    __slots__ = ("pipeline_version", "processing_time_ms")
    PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_VERSION_FIELD_NUMBER: _ClassVar[int]
    processing_time_ms: int
    pipeline_version: str
    def __init__(
        self, processing_time_ms: int | None = ..., pipeline_version: str | None = ...
    ) -> None: ...

class ItemError(_message.Message):
    __slots__ = ("code", "message")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    code: ItemErrorCode
    message: str
    def __init__(
        self, code: ItemErrorCode | str | None = ..., message: str | None = ...
    ) -> None: ...

class GetCapabilitiesRequest(_message.Message):
    __slots__ = ("context",)
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    def __init__(self, context: RequestContext | _Mapping | None = ...) -> None: ...

class GetCapabilitiesResponse(_message.Message):
    __slots__ = (
        "context",
        "max_batch_size",
        "max_image_height",
        "max_image_size_bytes",
        "max_image_width",
        "pipeline_version",
        "service_name",
        "service_version",
        "supported_mime_types",
        "supports_batch_processing",
    )
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    SERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    SERVICE_VERSION_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_VERSION_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_MIME_TYPES_FIELD_NUMBER: _ClassVar[int]
    MAX_IMAGE_WIDTH_FIELD_NUMBER: _ClassVar[int]
    MAX_IMAGE_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    MAX_IMAGE_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_BATCH_PROCESSING_FIELD_NUMBER: _ClassVar[int]
    context: ResponseContext
    service_name: str
    service_version: str
    pipeline_version: str
    supported_mime_types: _containers.RepeatedScalarFieldContainer[str]
    max_image_width: int
    max_image_height: int
    max_image_size_bytes: int
    max_batch_size: int
    supports_batch_processing: bool
    def __init__(
        self,
        context: ResponseContext | _Mapping | None = ...,
        service_name: str | None = ...,
        service_version: str | None = ...,
        pipeline_version: str | None = ...,
        supported_mime_types: _Iterable[str] | None = ...,
        max_image_width: int | None = ...,
        max_image_height: int | None = ...,
        max_image_size_bytes: int | None = ...,
        max_batch_size: int | None = ...,
        supports_batch_processing: bool | None = ...,
    ) -> None: ...
