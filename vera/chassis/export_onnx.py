"""
VERA ONNX Export & int8 Dynamic Quantization
=============================================
Exports fine-tuned chassis models to ONNX and quantizes to int8
to ensure high-throughput, low-latency CPU reproduction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def export_transformer_to_onnx(
    model: Any,
    tokenizer: Any,
    output_onnx_path: Path,
    max_length: int = 8192,
) -> Path:
    """
    Exports a PyTorch transformer model to ONNX with dynamic batch and sequence axes.
    """
    import torch

    output_onnx_path = Path(output_onnx_path)
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = tokenizer(
        "def example(): pass",
        return_tensors="pt",
        max_length=64,
        padding="max_length",
        truncation=True,
    )

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            str(output_onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "embeddings": {0: "batch_size"},
            },
            opset_version=14,
        )

    print(f"[ONNX Export] Saved ONNX model to {output_onnx_path} ({output_onnx_path.stat().st_size / 1024 / 1024:.1f} MB).")
    return output_onnx_path


def quantize_onnx_int8(
    input_onnx_path: Path,
    output_int8_path: Path,
) -> Path:
    """
    Performs dynamic int8 quantization on the exported ONNX model for CPU speedup.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        input_onnx_path = Path(input_onnx_path)
        output_int8_path = Path(output_int8_path)
        output_int8_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[ONNX Quantize] Quantizing {input_onnx_path} to int8...")
        quantize_dynamic(
            model_input=str(input_onnx_path),
            model_output=str(output_int8_path),
            weight_type=QuantType.QInt8,
        )
        print(f"[ONNX Quantize] int8 model saved to {output_int8_path} ({output_int8_path.stat().st_size / 1024 / 1024:.1f} MB).")
        return output_int8_path
    except ImportError:
        print("[ONNX Quantize] onnxruntime.quantization not available; keeping unquantized model.")
        return input_onnx_path


class ONNXChassisInference:
    """Fast CPU inference runner using ONNXRuntime session."""

    def __init__(self, onnx_model_path: Path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            str(onnx_model_path),
            providers=["CPUExecutionProvider"],
        )

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        inputs = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
        }
        outputs = self.session.run(None, inputs)
        return outputs[0]
