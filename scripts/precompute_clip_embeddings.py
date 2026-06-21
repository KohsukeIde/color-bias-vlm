#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import re
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Set
from tqdm import tqdm

# Allow direct execution: add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from datasets import load_dataset
except ImportError:
    raise RuntimeError("datasets not installed; pip install datasets")

try:
    from transformers import CLIPProcessor, CLIPTextModel
except ImportError:
    raise RuntimeError("transformers not installed; pip install transformers")


def extract_vocabulary_from_squad(n_samples: int = 10000) -> Set[str]:
    """SQuADデータセットから語彙を抽出する"""
    print(f"Loading SQuAD dataset (first {n_samples} samples)...")
    ds = load_dataset("squad", split="train").shuffle(seed=42).select(range(min(n_samples, 87599)))
    
    vocabulary = set()
    print("Extracting vocabulary from contexts and questions...")
    
    for entry in tqdm(ds, desc="Processing entries"):
        # Context text from vocabulary
        context = entry.get('context', '')
        words = re.findall(r'\b\w+\b', context.lower())
        vocabulary.update(words)
        
        # Question text from vocabulary
        question = entry.get('question', '')
        words = re.findall(r'\b\w+\b', question.lower())
        vocabulary.update(words)
        
        # Answer text from vocabulary
        answers = (entry.get('answers', {}) or {}).get('text', []) or []
        for answer in answers:
            words = re.findall(r'\b\w+\b', (answer or '').lower())
            vocabulary.update(words)
    
    # Filter out very short words and common stop words to reduce noise
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
        'to', 'was', 'will', 'with', 'or', 'but', 'if', 'this', 'they', 
        'we', 'you', 'have', 'had', 'his', 'her', 'him', 'them', 'their'
    }
    
    filtered_vocab = {word for word in vocabulary if len(word) >= 2 and word not in stop_words}
    
    print(f"Total vocabulary size: {len(vocabulary)}")
    print(f"Filtered vocabulary size: {len(filtered_vocab)}")
    
    return filtered_vocab


def precompute_squad_embeddings(n_samples: int = 10000) -> str:
    """SQuADの語彙のCLIP embeddingを事前計算して保存する"""
    
    # Extract vocabulary
    vocabulary = extract_vocabulary_from_squad(n_samples)
    vocab_list = sorted(list(vocabulary))
    
    print(f"Computing embeddings for {len(vocab_list)} words...")
    
    # Load CLIP model
    model_id = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP model: {model_id}")
    
    try:
        model = CLIPTextModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        raise
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    embeddings = {}
    batch_size = 256
    
    print("Computing embeddings in batches...")
    for i in tqdm(range(0, len(vocab_list), batch_size), desc="Processing batches"):
        batch = vocab_list[i:i + batch_size]
        
        try:
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get pooled output and normalize
            batch_embeddings = outputs.pooler_output.cpu().numpy()
            # Normalize embeddings for better cosine similarity computation
            batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            
            for word, emb in zip(batch, batch_embeddings):
                embeddings[word] = emb
                
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            # Skip this batch and continue
            continue
    
    # Save embeddings
    save_dir = Path(_PROJECT_ROOT) / "data" / "processed"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "squad_clip_embeddings.pt"
    
    print(f"Saving {len(embeddings)} embeddings to {save_path}...")
    torch.save(embeddings, save_path)
    
    # Also save metadata
    metadata = {
        "model_id": model_id,
        "vocab_size": len(embeddings),
        "n_samples_used": n_samples,
        "embedding_dim": batch_embeddings.shape[1] if 'batch_embeddings' in locals() else 512,
        "device_used": device,
    }
    
    metadata_path = save_dir / "squad_clip_embeddings_meta.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Metadata saved to {metadata_path}")
    print("✅ Embedding computation completed successfully!")
    
    return str(save_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Precompute CLIP embeddings for SQuAD vocabulary")
    parser.add_argument("--n-samples", type=int, default=10000, 
                       help="Number of SQuAD samples to process for vocabulary extraction")
    args = parser.parse_args()
    
    try:
        embedding_path = precompute_squad_embeddings(args.n_samples)
        print(f"🎉 Embeddings saved successfully at: {embedding_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
