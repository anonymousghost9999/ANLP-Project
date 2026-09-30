#!/usr/bin/env python3
"""
Model Downloader & Verifier for JarvisLabs GPU Execution.

Pre-downloads HuggingFace weights and tokenizer into cache to verify authentication
and prevent download delays during experimental execution.
"""

import argparse
import os
import sys

def download_model(model_name: str, hf_token: str = None):
    print("=" * 80)
    print(f"PRE-DOWNLOADING & VERIFYING MODEL: {model_name}")
    print("=" * 80)

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        from huggingface_hub import login
        if token:
            print("[1/3] Authenticating with Hugging Face Hub...")
            login(token=token)
        else:
            print("[1/3] No explicit token provided. Using cached HF authentication credentials.")
    except Exception as e:
        print(f"Authentication note: {e}")

    print(f"[2/3] Downloading tokenizer for {model_name}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, use_fast=True)
    print(f"✔ Tokenizer loaded successfully (vocab size: {len(tokenizer):,})")

    print(f"[3/3] Pre-caching model architecture & weights for {model_name}...")
    # Pre-download configuration and weights without creating full GPU allocation yet
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, token=token)
    print(f"✔ Model Config: {config.architectures} | Hidden Size: {getattr(config, 'hidden_size', 'N/A')}")

    try:
        # Download weight shards into HF cache
        import torch
        print("  Downloading model weights to HF cache...")
        AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            device_map="cpu",  # Load on CPU first to cache weights without taking VRAM
            low_cpu_mem_usage=True,
        )
        print(f"\nSUCCESS: {model_name} is fully downloaded and ready for GPU execution!")
    except Exception as e:
        print(f"\nERROR pre-downloading model weights: {e}")
        print("Hint: If using gated models like Llama 3, ensure you have accepted Meta terms on HuggingFace")
        print("and provided your read token via --hf_token or HF_TOKEN env var.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-download LLM models for JarvisLabs GPU run")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace model ID (e.g. meta-llama/Meta-Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional HuggingFace User Access Token (Read)",
    )
    args = parser.parse_args()
    download_model(args.model_name, args.hf_token)
