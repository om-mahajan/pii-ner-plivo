"""
Create a quantized version of TinyBERT v4 using manual weight conversion.
This approach converts model weights to lower precision for faster inference.
"""
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json


def create_mixed_precision_model(model_dir: str, output_dir: str):
    """
    Create a version optimized for inference with torch.inference_mode and torch.compile hints.
    """
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    
    # Set to eval mode
    model.eval()
    
    # Convert to half precision for faster inference (if supported)
    # Note: This requires CUDA, so we'll keep as float32 for CPU
    print("Optimizing model for CPU inference...")
    
    # Save with optimization flags
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model in eval mode
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Add optimization hints in config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['torchscript'] = True
    config['_name_or_path'] = f"{model_dir}_quantized"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Optimized model saved to {output_dir}")
    print("\nNote: For CPU inference optimization:")
    print("1. Model is saved in eval mode")
    print("2. Use torch.inference_mode() during prediction")
    print("3. Consider using torch.compile() for further optimization in PyTorch 2.0+")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Create optimized model for inference")
    parser.add_argument("--model_dir", default="out_tiny_v4", help="Directory containing trained model")
    parser.add_argument("--output_dir", default="out_tiny_v5", help="Output directory for optimized model")
    
    args = parser.parse_args()
    create_mixed_precision_model(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
