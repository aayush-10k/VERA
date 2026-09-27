"""Static I/O signature extraction, example-shape compatibility gate, uncertainty router (R3)."""

from vera.gate.router import ExampleShape, SignatureGate, UncertaintyRouter, example_shape, signature_compatible
from vera.gate.signature import ProgramSignature, extract_program_signature

__all__ = ["ExampleShape", "SignatureGate", "UncertaintyRouter", "example_shape", "signature_compatible",
           "ProgramSignature", "extract_program_signature"]
