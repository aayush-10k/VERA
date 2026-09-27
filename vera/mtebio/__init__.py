"""
VERA MTEB I/O Package
=====================
Submission harness interfaces:
- VERASearchProtocol: Primary MTEB v2 SearchProtocol implementation
- VERAAbsEncoder: Fallback AbsEncoder embedding interface
- save_mteb_task_result: Robust JSON serializer
- validate_mteb_result_schema: Schema validator
"""

from vera.mtebio.encoder_fallback import VERAAbsEncoder
from vera.mtebio.search_protocol import VERASearchProtocol
from vera.mtebio.serializer import (
    VERAJSONEncoder,
    save_mteb_task_result,
    validate_mteb_result_schema,
)

__all__ = [
    "VERASearchProtocol",
    "VERAAbsEncoder",
    "VERAJSONEncoder",
    "save_mteb_task_result",
    "validate_mteb_result_schema",
]
