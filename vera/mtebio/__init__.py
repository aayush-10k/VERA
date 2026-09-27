"""
VERA MTEB I/O Package
=====================
- VERASearchProtocol: MTEB v2 SearchProtocol implementation (dense -> verify -> boost)
- build_encoder_only_model / VERAAbsEncoder: encoder-only fallback path
- save_mteb_task_result / validate_mteb_result_schema: TaskResult persistence + validation
"""

from vera.mtebio.encoder_fallback import VERAAbsEncoder, build_encoder_only_model
from vera.mtebio.search_protocol import VERASearchProtocol, build_model_meta
from vera.mtebio.serializer import (
    VERAJSONEncoder,
    load_result,
    main_score,
    save_mteb_task_result,
    validate_mteb_result_schema,
)

__all__ = [
    "VERASearchProtocol",
    "build_model_meta",
    "VERAAbsEncoder",
    "build_encoder_only_model",
    "VERAJSONEncoder",
    "save_mteb_task_result",
    "validate_mteb_result_schema",
    "load_result",
    "main_score",
]
