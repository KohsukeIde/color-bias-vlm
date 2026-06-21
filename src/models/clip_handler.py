#!/usr/bin/env python3
"""
CLIP/SigLIP統合ハンドラー for ステルス・ビジュアルプロンプト研究

このモジュールは以下の機能を提供します：
- CLIP/SigLIPモデルの統一インターフェース
- 画像・テキスト埋め込み取得
- 意味軸投影量計算
- バッチ処理対応
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
from pathlib import Path
import json

try:
    from transformers import CLIPModel, CLIPProcessor
    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import SiglipModel, SiglipProcessor
    SIGLIP_AVAILABLE = True
except ImportError:
    SIGLIP_AVAILABLE = False

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

HF_AVAILABLE = HF_TRANSFORMERS_AVAILABLE or SIGLIP_AVAILABLE or OPEN_CLIP_AVAILABLE

if not HF_AVAILABLE:
    print("WARNING: transformers or open_clip not available. Some functionality will be limited.")


class CLIPHandler:
    """統一されたCLIP/SigLIPハンドラー"""
    
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-large-patch14",
        model_type: str = "hf",  # "hf", "openclip", "siglip"
        device: str = "auto"
    ):
        self.model_name = model_name
        self.model_type = model_type
        # デバイス設定（Apple Silicon MPS対応）
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        self._load_model()
        
    def _load_model(self):
        """モデルとプロセッサーをロード"""
        try:
            if self.model_type == "hf" and "clip" in self.model_name.lower():
                # Hugging Face CLIP
                if not HF_TRANSFORMERS_AVAILABLE:
                    raise RuntimeError("transformers library with CLIP support is required")
                self.model = CLIPModel.from_pretrained(self.model_name)
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
                
            elif self.model_type == "siglip" or "siglip" in self.model_name.lower():
                # SigLIP
                if not SIGLIP_AVAILABLE:
                    raise RuntimeError("transformers library with SigLIP support is required")
                self.model = SiglipModel.from_pretrained(self.model_name)
                self.processor = SiglipProcessor.from_pretrained(self.model_name)
                
            elif self.model_type == "openclip":
                # OpenCLIP
                if not OPEN_CLIP_AVAILABLE:
                    raise RuntimeError("open_clip library is required")
                model_name_parts = self.model_name.split("/")
                if len(model_name_parts) == 2:
                    arch, pretrained = model_name_parts
                else:
                    arch = self.model_name
                    pretrained = "openai"
                
                self.model, _, self.processor = open_clip.create_model_and_transforms(
                    arch, pretrained=pretrained, device=self.device
                )
                self.tokenizer = open_clip.get_tokenizer(arch)
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type} for {self.model_name}")
                
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    @torch.inference_mode()
    def encode_images(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """画像を埋め込みベクトルにエンコード"""
        if isinstance(images, Image.Image):
            images = [images]
        
        if self.model_type == "openclip":
            # OpenCLIP
            image_tensors = []
            for img in images:
                img_tensor = self.processor(img).unsqueeze(0).to(self.device)
                image_tensors.append(img_tensor)
            batch_tensor = torch.cat(image_tensors, dim=0)
            
            with torch.no_grad():
                embeddings = self.model.encode_image(batch_tensor)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # normalize
                
        else:
            # Hugging Face (CLIP/SigLIP)
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if hasattr(self.model, 'get_image_features'):
                    embeddings = self.model.get_image_features(**inputs)
                else:
                    outputs = self.model.vision_model(**inputs)
                    embeddings = outputs.pooler_output
                
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # normalize
        
        return embeddings.cpu().numpy()
    
    @torch.inference_mode()
    def encode_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """テキストを埋め込みベクトルにエンコード"""
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model_type == "openclip":
            # OpenCLIP
            text_tokens = self.tokenizer(texts).to(self.device)
            with torch.no_grad():
                embeddings = self.model.encode_text(text_tokens)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # normalize
                
        else:
            # Hugging Face (CLIP/SigLIP)
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if hasattr(self.model, 'get_text_features'):
                    embeddings = self.model.get_text_features(**inputs)
                else:
                    outputs = self.model.text_model(**inputs)
                    embeddings = outputs.pooler_output
                
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # normalize
        
        return embeddings.cpu().numpy()
    
    def compute_semantic_projection(
        self, 
        image_embeddings: np.ndarray, 
        semantic_axis: np.ndarray
    ) -> np.ndarray:
        """意味軸への投影量を計算"""
        # semantic_axis は正規化済みと仮定
        projections = np.dot(image_embeddings, semantic_axis)
        return projections
    
    def compute_cosine_similarity(
        self, 
        embeddings1: np.ndarray, 
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """コサイン類似度を計算"""
        # 両方とも正規化済みと仮定
        similarities = np.sum(embeddings1 * embeddings2, axis=1)
        return similarities


class SemanticAxes:
    """意味軸の管理クラス"""
    
    def __init__(self, clip_handler: CLIPHandler):
        self.clip_handler = clip_handler
        self.axes: Dict[str, np.ndarray] = {}
        self.word_pairs: Dict[str, Tuple[str, str]] = {}
    
    def define_axis(self, axis_name: str, positive_word: str, negative_word: str) -> np.ndarray:
        """意味軸を定義"""
        # テキスト埋め込みを取得
        pos_emb = self.clip_handler.encode_texts([positive_word])[0]
        neg_emb = self.clip_handler.encode_texts([negative_word])[0]
        
        # 意味軸ベクトルを計算 (正規化)
        axis_vector = pos_emb - neg_emb
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        
        self.axes[axis_name] = axis_vector
        self.word_pairs[axis_name] = (positive_word, negative_word)
        
        return axis_vector
    
    def get_axis(self, axis_name: str) -> np.ndarray:
        """意味軸を取得"""
        if axis_name not in self.axes:
            raise KeyError(f"Semantic axis '{axis_name}' not found")
        return self.axes[axis_name]
    
    def save_axes(self, save_path: str):
        """意味軸を保存"""
        save_data = {
            'word_pairs': self.word_pairs,
            'axes': {name: axis.tolist() for name, axis in self.axes.items()},
            'model_name': self.clip_handler.model_name,
            'model_type': self.clip_handler.model_type
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def load_axes(self, load_path: str):
        """意味軸を読み込み"""
        with open(load_path, 'r') as f:
            save_data = json.load(f)
        
        self.word_pairs = save_data['word_pairs']
        self.axes = {name: np.array(axis) for name, axis in save_data['axes'].items()}


# 研究用の標準的な意味軸定義
STANDARD_SEMANTIC_AXES = [
    ("safety", "safe", "dangerous"),
    ("valence", "good", "bad"), 
    ("arousal", "calm", "chaotic"),
    ("temperature", "warm", "cold"),
    ("emotion", "happy", "sad"),
    ("morality", "moral", "immoral"),
    ("activity", "active", "passive"),
    ("strength", "strong", "weak"),
    ("size", "large", "small"),
    ("speed", "fast", "slow")
]


def create_standard_axes(clip_handler: CLIPHandler) -> SemanticAxes:
    """標準的な意味軸セットを作成"""
    semantic_axes = SemanticAxes(clip_handler)
    
    print("Creating standard semantic axes...")
    for axis_name, pos_word, neg_word in STANDARD_SEMANTIC_AXES:
        print(f"  - {axis_name}: {pos_word} <-> {neg_word}")
        semantic_axes.define_axis(axis_name, pos_word, neg_word)
    
    return semantic_axes


if __name__ == "__main__":
    # テスト用のサンプルコード
    print("Testing CLIP Handler...")
    
    # CLIP ハンドラーを初期化（ViT-L/14@336px使用）
    clip_handler = CLIPHandler("openai/clip-vit-large-patch14-336", model_type="hf")
    
    # 意味軸を作成
    semantic_axes = create_standard_axes(clip_handler)
    
    # テスト画像（ダミー）
    test_image = Image.new('RGB', (336, 336), color='red')  # ViT-L/14@336px用
    
    # 画像埋め込みを取得
    img_embeddings = clip_handler.encode_images([test_image])
    
    # 意味軸投影を計算
    safety_axis = semantic_axes.get_axis("safety")
    projection = clip_handler.compute_semantic_projection(img_embeddings, safety_axis)
    
    print(f"Safety projection: {projection[0]:.4f}")
    print("Test completed successfully!")

