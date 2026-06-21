#!/usr/bin/env python3
"""
Quick test script to verify all 4 target VLMs can load and generate responses.
Tests each model with a single image from the dataset.

Usage:
    python stealth_visual_prompting/scripts/test_models.py
    python stealth_visual_prompting/scripts/test_models.py --manifest /path/to/experiment_config.json
"""

import sys
import os
import json
import argparse
from pathlib import Path
from PIL import Image

# Add project root to Python path
proj_root = Path(__file__).parent.parent.absolute()
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from src.models.unified_handler import VLMHandler

def test_single_model(model_id: str, image_path: str, prompt: str):
    """Test a single VLM model with one image."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"{'='*60}")
    
    try:
        # Load model
        print("Loading model...", flush=True)
        handler = VLMHandler(model_id)
        print("✓ Model loaded successfully")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        print(f"✓ Image loaded: {image.size}")
        
        # Generate response
        print("Generating response...", flush=True)
        response = handler.generate(
            image=image, 
            prompt=prompt, 
            max_new_tokens=100,
            temperature=0.0
        )
        
        print("✓ Response generated:")
        print(f"📝 {response}")
        
        # Cleanup
        del handler
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test all VLM models with a single image")
    parser.add_argument(
        "--manifest", 
        default="data/processed/color/sentiment/custom_short/DroidSans/aa_on/experiment_config.json",
        help="Path to experiment config JSON"
    )
    parser.add_argument(
        "--image-index", 
        type=int, 
        default=0,
        help="Index of image to test (default: 0 = first image)"
    )
    args = parser.parse_args()
    
    # Target models
    models = [
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        "Qwen/Qwen2-VL-7B-Instruct", 
        "HuggingFaceM4/idefics2-8b",
    ]
    
    # Load manifest
    manifest_path = Path(proj_root) / args.manifest
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return 1
        
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    # Get test data
    experiments = manifest.get('experiments', [])
    if not experiments:
        print("ERROR: No experiments found in manifest")
        return 1
        
    if args.image_index >= len(experiments):
        print(f"ERROR: Image index {args.image_index} out of range (max: {len(experiments)-1})")
        return 1
    
    exp = experiments[args.image_index]
    task_name = manifest.get('task_name', 'sentiment')
    
    # Get baseline image path
    baseline_image = None
    for condition, image_path in exp['images'].items():
        if 'baseline' in condition:
            baseline_image = image_path
            break
    
    if not baseline_image:
        # Fallback to first available image
        baseline_image = next(iter(exp['images'].values()))
    
    baseline_path = Path(baseline_image)
    if not baseline_path.exists():
        print(f"ERROR: Test image not found: {baseline_path}")
        return 1
    
    # Determine prompt based on task
    if task_name == 'qa':
        question = exp.get('qa', {}).get('question', 'What do you see in this image?')
        prompt = f"USER: Based on the image, answer the following question: {question}\nASSISTANT:"
    else:
        # Sentiment task
        prompt = ("USER: <image>\nPlease analyze the sentiment of the text shown in this image. "
                 "Provide your analysis and conclude with either 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'.\nASSISTANT:")
    
    print(f"Test Configuration:")
    print(f"  Manifest: {manifest_path}")
    print(f"  Task: {task_name}")
    print(f"  Test image: {baseline_path}")
    print(f"  Models to test: {len(models)}")
    
    # Test each model
    results = {}
    for model_id in models:
        success = test_single_model(model_id, str(baseline_path), prompt)
        results[model_id] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    successful = 0
    for model_id, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status:8} {model_id}")
        if success:
            successful += 1
    
    print(f"\nResult: {successful}/{len(models)} models working")
    
    if successful == len(models):
        print("🎉 All models are ready for experiments!")
        return 0
    else:
        print("⚠️  Some models failed. Check error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
