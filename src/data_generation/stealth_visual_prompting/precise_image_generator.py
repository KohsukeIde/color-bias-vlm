#!/usr/bin/env python3
"""
精密画像生成器 for ステルス・ビジュアルプロンプト研究

このモジュールは以下の機能を提供します：
- 色相・彩度・明度の精密制御
- コントラスト（ΔE）の正確な計算と制御
- フォントサイズの対数スケール制御
- バッチ生成とパラメータ空間の網羅的探索
"""

from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Any
from dataclasses import dataclass, field
import colorsys
import math
from tqdm import tqdm

# 既存のユーティリティをインポート
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.image_utils import choose_font


@dataclass
class PreciseImageConfig:
    """精密画像生成の設定"""
    # 基本設定（既存色実験と統一）
    image_size: Tuple[int, int] = (800, 600)  # 既存実験と同じサイズ
    background_color: Tuple[int, int, int] = (255, 255, 255)  # 白背景
    
    # テキスト設定（既存実験と統一）
    text: str = "SAMPLE"
    font_family: str = "DroidSans"  # 既存実験と同じフォント
    font_sizes: List[int] = field(default_factory=lambda: [28, 32, 40, 48, 56, 64, 72, 80])  # 動的調整ベースサイズ（大きめに設定）
    
    # 色相制御 (HSV)
    hue_values: List[float] = field(default_factory=lambda: np.linspace(0, 360, 36, endpoint=False).tolist())  # 10度刻み
    saturation_values: List[float] = field(default_factory=lambda: [0.2, 0.5, 0.8, 1.0])
    brightness_values: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9])
    
    # コントラスト制御 (ΔE - CIE Delta E 2000)
    contrast_delta_e_values: List[float] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128])
    
    # 位置制御
    text_positions: List[Tuple[float, float]] = field(default_factory=lambda: [(0.5, 0.5)])  # 中央
    
    # レンダリング設定
    antialiasing: bool = True
    

def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """HSVからRGBに変換 (0-255スケール)"""
    r, g, b = colorsys.hsv_to_rgb(h/360.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """RGBからCIE LABに変換（簡易版）"""
    # 簡易的なsRGB -> XYZ -> LAB変換
    r, g, b = [x/255.0 for x in rgb]
    
    # Gamma correction
    def gamma_correct(c):
        if c > 0.04045:
            return ((c + 0.055) / 1.055) ** 2.4
        else:
            return c / 12.92
    
    r, g, b = [gamma_correct(c) for c in [r, g, b]]
    
    # sRGB -> XYZ (D65 illuminant)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    # XYZ -> LAB
    def xyz_to_lab_component(t):
        if t > 0.008856:
            return t ** (1/3)
        else:
            return (7.787 * t) + (16/116)
    
    # D65 white point
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    
    fx = xyz_to_lab_component(x / xn)
    fy = xyz_to_lab_component(y / yn)
    fz = xyz_to_lab_component(z / zn)
    
    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return (L, a, b)


def calculate_delta_e(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    """CIE Delta E 2000の簡易計算"""
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    
    # 簡易版Delta E (実際のDelta E 2000はより複雑)
    dL = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    
    return math.sqrt(dL**2 + da**2 + db**2)


def find_color_for_contrast(
    background_rgb: Tuple[int, int, int], 
    target_delta_e: float,
    base_hue: float = 0.0,
    saturation: float = 1.0,
    max_iterations: int = 100
) -> Tuple[int, int, int]:
    """指定されたΔEになる色を二分探索で見つける"""
    
    low_v, high_v = 0.0, 1.0
    tolerance = 0.5  # ΔEの許容誤差
    
    for _ in range(max_iterations):
        mid_v = (low_v + high_v) / 2.0
        test_rgb = hsv_to_rgb(base_hue, saturation, mid_v)
        current_delta_e = calculate_delta_e(background_rgb, test_rgb)
        
        if abs(current_delta_e - target_delta_e) < tolerance:
            return test_rgb
        elif current_delta_e < target_delta_e:
            low_v = mid_v
        else:
            high_v = mid_v
    
    # 最適解が見つからない場合は最後の値を返す
    return hsv_to_rgb(base_hue, saturation, mid_v)


class PreciseImageGenerator:
    """精密画像生成器"""
    
    def __init__(self, config: PreciseImageConfig):
        self.config = config
        self._font_cache: Dict[Tuple[str, int], ImageFont.ImageFont] = {}
    
    def _get_font(self, family: str, size: int) -> ImageFont.ImageFont:
        """フォントをキャッシュ付きで取得"""
        cache_key = (family, size)
        if cache_key not in self._font_cache:
            self._font_cache[cache_key] = choose_font([family], size=size)
        return self._font_cache[cache_key]
    
    def generate_single_image(
        self,
        text: str,
        font_size: int,
        text_color: Tuple[int, int, int],
        position: Tuple[float, float] = (0.5, 0.5),
        background_color: Optional[Tuple[int, int, int]] = None
    ) -> Image.Image:
        """単一画像を生成（既存のrender_document_image使用で動的フォントサイズ対応）"""
        bg_color = background_color or self.config.background_color
        
        # 既存のrender_document_image関数を使用（動的フォントサイズ対応）
        from src.utils.image_utils import render_document_image
        
        # 色をhex形式に変換
        hex_color = f"#{text_color[0]:02x}{text_color[1]:02x}{text_color[2]:02x}"
        
        # render_document_imageを使用して動的フォントサイズで生成
        img = render_document_image(
            title="",  # タイトルなし
            text=text,
            condition_name="",  # 条件名なし
            color_settings={"default": hex_color},
            target_words={},  # カテゴリなし（全て同じ色）
            image_size=self.config.image_size,
            title_font=None,
            body_font=self._get_font(self.config.font_family, font_size),
            aa_mode="aa_on",
            baseline_text_color=text_color,
            font_family_name=self.config.font_family,
            auto_fit_body=True,  # 動的フォントサイズ調整を有効化
            vertical_center=True,  # 垂直中央揃え
            center_align=True   # 水平中央揃え
        )
        
        return img
    
    def generate_hue_series(
        self,
        text: str,
        font_size: int,
        saturation: float = 1.0,
        brightness: float = 0.5
    ) -> Iterator[Tuple[Dict[str, Any], Image.Image]]:
        """色相を変化させた画像シリーズを生成"""
        for hue in self.config.hue_values:
            text_color = hsv_to_rgb(hue, saturation, brightness)
            
            img = self.generate_single_image(
                text=text,
                font_size=font_size,
                text_color=text_color
            )
            
            params = {
                'text': text,
                'font_size': font_size,
                'hue': hue,
                'saturation': saturation,
                'brightness': brightness,
                'text_color_rgb': text_color,
                'background_color_rgb': self.config.background_color
            }
            
            yield params, img
    
    def generate_contrast_series(
        self,
        text: str,
        font_size: int,
        base_hue: float = 0.0,
        saturation: float = 1.0
    ) -> Iterator[Tuple[Dict[str, Any], Image.Image]]:
        """コントラスト（ΔE）を変化させた画像シリーズを生成"""
        for delta_e in self.config.contrast_delta_e_values:
            text_color = find_color_for_contrast(
                self.config.background_color, 
                delta_e, 
                base_hue, 
                saturation
            )
            
            img = self.generate_single_image(
                text=text,
                font_size=font_size,
                text_color=text_color
            )
            
            # 実際のΔEを計算
            actual_delta_e = calculate_delta_e(self.config.background_color, text_color)
            
            params = {
                'text': text,
                'font_size': font_size,
                'target_delta_e': delta_e,
                'actual_delta_e': actual_delta_e,
                'base_hue': base_hue,
                'saturation': saturation,
                'text_color_rgb': text_color,
                'background_color_rgb': self.config.background_color
            }
            
            yield params, img
    
    def generate_font_size_series(
        self,
        text: str,
        text_color: Tuple[int, int, int]
    ) -> Iterator[Tuple[Dict[str, Any], Image.Image]]:
        """フォントサイズを変化させた画像シリーズを生成"""
        for font_size in self.config.font_sizes:
            img = self.generate_single_image(
                text=text,
                font_size=font_size,
                text_color=text_color
            )
            
            params = {
                'text': text,
                'font_size': font_size,
                'text_color_rgb': text_color,
                'background_color_rgb': self.config.background_color
            }
            
            yield params, img
    
    def generate_comprehensive_dataset(
        self,
        words: List[str],
        output_dir: Path,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """包括的なデータセットを生成"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_data = []
        total_combinations = len(words) * (
            len(self.config.hue_values) * len(self.config.saturation_values) * 
            len(self.config.brightness_values) * len(self.config.font_sizes) +
            len(self.config.contrast_delta_e_values) * len(self.config.font_sizes)
        )
        
        pbar = tqdm(total=total_combinations, desc="Generating images")
        
        for word_idx, word in enumerate(words):
            word_dir = output_dir / f"word_{word_idx:03d}_{word}"
            word_dir.mkdir(exist_ok=True)
            
            image_idx = 0
            
            # 色相シリーズ
            for sat in self.config.saturation_values:
                for bright in self.config.brightness_values:
                    for font_size in self.config.font_sizes:
                        for params, img in self.generate_hue_series(word, font_size, sat, bright):
                            # 画像を保存
                            img_path = word_dir / f"hue_{image_idx:05d}.png"
                            img.save(img_path)
                            
                            # メタデータを記録
                            params.update({
                                'word_idx': word_idx,
                                'image_idx': image_idx,
                                'image_path': str(img_path),
                                'series_type': 'hue_variation'
                            })
                            all_data.append(params)
                            
                            image_idx += 1
                            pbar.update(1)
                            
                            if progress_callback:
                                progress_callback(params, img_path)
            
            # コントラストシリーズ
            for hue in [0, 120, 240]:  # 赤、緑、青
                for font_size in self.config.font_sizes:
                    for params, img in self.generate_contrast_series(word, font_size, hue):
                        # 画像を保存
                        img_path = word_dir / f"contrast_{image_idx:05d}.png"
                        img.save(img_path)
                        
                        # メタデータを記録
                        params.update({
                            'word_idx': word_idx,
                            'image_idx': image_idx,
                            'image_path': str(img_path),
                            'series_type': 'contrast_variation'
                        })
                        all_data.append(params)
                        
                        image_idx += 1
                        pbar.update(1)
                        
                        if progress_callback:
                            progress_callback(params, img_path)
        
        pbar.close()
        
        # メタデータをJSON形式で保存
        metadata_path = output_dir / "generation_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"Generated {len(all_data)} images in {output_dir}")
        print(f"Metadata saved to {metadata_path}")
        
        return all_data


if __name__ == "__main__":
    # テスト用のサンプルコード
    print("Testing Precise Image Generator...")
    
    # 設定を作成（既存実験と統一）
    config = PreciseImageConfig(
        image_size=(800, 600),
        font_family="DroidSans",
        font_sizes=[28, 40, 64],
        hue_values=[0, 120, 240],  # 赤、緑、青
        saturation_values=[0.5, 1.0],
        brightness_values=[0.5, 0.8],
        contrast_delta_e_values=[2, 8, 32]
    )
    
    # 生成器を作成
    generator = PreciseImageGenerator(config)
    
    # テスト画像を生成
    test_words = ["SAFE", "DANGER"]
    output_dir = Path("test_output")
    
    # データセットを生成
    metadata = generator.generate_comprehensive_dataset(test_words, output_dir)
    
    print(f"Generated {len(metadata)} test images")
    print("Test completed successfully!")

