"""
Model optimization utilities for reducing latency.
Supports ONNX export, quantization, and distillation helpers.
"""
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_to_onnx(model_dir: str, output_path: str, max_length: int = 256):
    """
    Export a PyTorch model to ONNX format for faster inference.
    """
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    
    # Create dummy input
    dummy_text = "this is a sample text for export"
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size", 1: "sequence"}
        }
    )
    print(f"ONNX model saved to {output_path}")
    
    # Verify the model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")


def quantize_onnx_model(onnx_path: str, output_path: str):
    """
    Apply dynamic quantization to ONNX model for faster CPU inference.
    """
    print(f"Quantizing ONNX model: {onnx_path}")
    quantize_dynamic(
        onnx_path,
        output_path,
        weight_type=QuantType.QUInt8
    )
    print(f"Quantized model saved to {output_path}")


def quantize_pytorch_model(model_dir: str, output_dir: str):
    """
    Apply dynamic quantization to PyTorch model.
    """
    print(f"Loading model from {model_dir}...")
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    print("Applying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving quantized model to {output_dir}...")
    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Quantization complete")


def main():
    parser = argparse.ArgumentParser(description="Model optimization for latency reduction")
    parser.add_argument("--model_dir", required=True, help="Directory containing trained model")
    parser.add_argument("--output_dir", required=True, help="Output directory for optimized model")
    parser.add_argument("--method", choices=["onnx", "quantize", "onnx-quantize"], default="quantize",
                       help="Optimization method: onnx (export), quantize (PyTorch), onnx-quantize (both)")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.method in ["onnx", "onnx-quantize"]:
        onnx_path = os.path.join(args.output_dir, "model.onnx")
        export_to_onnx(args.model_dir, onnx_path, args.max_length)
        
        if args.method == "onnx-quantize":
            quantized_path = os.path.join(args.output_dir, "model_quantized.onnx")
            quantize_onnx_model(onnx_path, quantized_path)
    
    elif args.method == "quantize":
        quantize_pytorch_model(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
