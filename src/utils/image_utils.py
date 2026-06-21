#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont


def find_existing_path(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def get_default_font_paths() -> Dict[str, List[str]]:
    return {
        "DejaVuSans": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ],
        "DroidSans": [
            "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf",
        ],
        "LiberationSans": [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ],
        "NotoSans": [
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        ],
        "Arial": [
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",  # M1 Mac用
            "/Library/Fonts/Arial.ttf",             # M1 Mac用
        ],
        "TimesNewRoman": [
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            "/System/Library/Fonts/Times.ttc",      # M1 Mac用
        ],
        "CourierNew": [
            "/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf",
            "/System/Library/Fonts/Courier.ttc",    # M1 Mac用
        ],
        "Helvetica": [  # M1 Mac標準フォント
            "/System/Library/Fonts/Helvetica.ttc",
        ],
    }


def choose_font(preferred: List[str] = None, size: int = 18) -> ImageFont.FreeTypeFont:
    preferred = preferred or ["Helvetica", "Arial", "DejaVuSans", "LiberationSans", "NotoSans"]
    candidates_by_name = get_default_font_paths()
    for name in preferred:
        font_path = find_existing_path(candidates_by_name.get(name, []))
        if font_path:
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    start_xy: Tuple[int, int],
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_height: int = 26,
):
    import re

    words = re.findall(r"\S+", text)
    x0, y0 = start_xy
    x, y = x0, y0
    for word in words:
        w = draw.textbbox((0, 0), word, font=font)[2]
        if x + w > x0 + max_width:
            x = x0
            y += line_height
        draw.text((x, y), word, fill=fill, font=font)
        x += w + 8


def render_document_image(
    title: str,
    text: str,
    condition_name: str,
    color_settings: Dict[str, str],
    target_words: Dict[str, List[str]],
    image_size: Tuple[int, int] = (800, 600),
    title_font: Optional[ImageFont.ImageFont] = None,
    body_font: Optional[ImageFont.ImageFont] = None,
    aa_mode: str = "aa_on",  # "aa_on" | "aa_off"
    baseline_text_color: Optional[Tuple[int, int, int]] = None,
    font_family_name: Optional[str] = None,
    auto_fit_body: bool = False,
    vertical_center: bool = False,
    center_align: bool = False,
) -> Image.Image:
    W, H = image_size
    bg = Image.new("RGB", (W, H), "#FFFFFF")

    title_font = title_font or choose_font(size=24)
    body_font = body_font or choose_font(size=18)

    # Header layer (always AA)
    header = Image.new("RGB", (W, H), "#FFFFFF")
    hdraw = ImageDraw.Draw(header)
    y = 30
    hdraw.text((30, y), title, fill="#000000", font=title_font)
    y += max(24, int(getattr(title_font, "size", 24) * 1.2))
    hdraw.text((30, y), f"Condition: {condition_name}", fill="#666666", font=body_font)
    y += max(18, int(getattr(body_font, "size", 18) * 1.0))

    if baseline_text_color:
        hex_c = f"#{baseline_text_color[0]:02x}{baseline_text_color[1]:02x}{baseline_text_color[2]:02x}"
        hdraw.text((30, y), f"Text Color: {hex_c} (RGB: {baseline_text_color})", fill="#888888", font=body_font)
    else:
        pos_c = color_settings.get("positive", "#000000")
        neg_c = color_settings.get("negative", "#000000")
        neu_c = color_settings.get("neutral", "#000000")
        hdraw.text((30, y), f"Positive: {pos_c}, Negative: {neg_c}, Neutral: {neu_c}", fill="#888888", font=body_font)

    # Body layer: allow AA toggle via mask compositing
    body_y = y + max(24, int(getattr(body_font, "size", 18) * 1.6))

    # layout helpers
    def _calc_line_h(f: ImageFont.ImageFont) -> int:
        try:
            ascent, descent = f.getmetrics()
            return int(ascent + descent + max(4, round((ascent + descent) * 0.15)))
        except Exception:
            return max(20, int(getattr(f, "size", 18) * 1.4))

    def _space_w_for(f: ImageFont.ImageFont) -> int:
        _tmp = ImageDraw.Draw(Image.new("RGB", (W, H)))
        try:
            sw = _tmp.textbbox((0, 0), " ", font=f)[2]
            return sw if sw > 0 else max(8, int(getattr(f, "size", 18) * 0.4))
        except Exception:
            return max(8, int(getattr(f, "size", 18) * 0.4))

    x0, max_text_width = 30, (W - 60)
    line_h = _calc_line_h(body_font)
    space_w = _space_w_for(body_font)

    # Precompute lines and optionally auto-fit to height
    import re
    words_all = re.findall(r"\S+", text)

    def layout_with_font(f: ImageFont.ImageFont):
        tmp_d = ImageDraw.Draw(Image.new("RGB", (W, H)))
        lh = _calc_line_h(f)
        spw = _space_w_for(f)
        lines: List[List[Tuple[str, int]]] = []
        cur: List[Tuple[str, int]] = []
        cur_w = 0
        for w in words_all:
            ww = tmp_d.textbbox((0, 0), w, font=f)[2]
            if cur and (cur_w + ww) > max_text_width:
                lines.append(cur)
                cur = [(w, ww)]
                cur_w = ww + spw
            else:
                cur.append((w, ww))
                cur_w += ww + spw
        if cur:
            lines.append(cur)
        total_h = len(lines) * lh
        widths = [sum(w for _, w in line) + max(0, (len(line) - 1)) * spw for line in lines]
        return lines, widths, total_h, lh, spw

    if auto_fit_body and font_family_name:
        # binary search downwards to fit into available height
        available_h = max(40, H - body_y - 30)
        size_low, size_high = 8, int(getattr(body_font, "size", 18))
        best = None
        while size_low <= size_high:
            mid = (size_low + size_high) // 2
            f_mid = choose_font([font_family_name], size=mid)
            _lines, _widths, _total_h, _lh, _spw = layout_with_font(f_mid)
            if _total_h <= available_h:
                best = (f_mid, _lines, _widths, _total_h, _lh, _spw)
                size_low = mid + 1
            else:
                size_high = mid - 1
        if best is not None:
            body_font, lines, line_widths, total_h, line_h, space_w = best[0], best[1], best[2], best[3], best[4], best[5]
        else:
            lines, line_widths, total_h, line_h, space_w = layout_with_font(body_font)
    else:
        lines, line_widths, total_h, line_h, space_w = layout_with_font(body_font)

    # Vertical centering of body block if requested
    if vertical_center:
        body_y = max(body_y, int((H - total_h) / 2))
    # Prepare category sets and order
    category_order: List[str] = list(target_words.keys())
    category_sets: Dict[str, set] = {k: set(v) for k, v in target_words.items()}

    if aa_mode == "aa_off":
        # Build base mask for all words first
        base_mask = Image.new("1", (W, H), 0)
        bmdraw = ImageDraw.Draw(base_mask)
        yline = body_y
        for li, line in enumerate(lines):
            if center_align:
                start_x = int((W - line_widths[li]) / 2)
            else:
                start_x = x0
            x = start_x
            for word, ww in line:
                bmdraw.text((x, yline), word, fill=1, font=body_font)
                x += ww + space_w
            yline += line_h

        # Composite black text for all words
        composed = Image.composite(Image.new("RGB", (W, H), "#000000"), Image.new("RGB", (W, H), "#FFFFFF"), base_mask)

        # Overlay category colors
        for cat in category_order:
            cat_color = color_settings.get(cat, "#000000")
            if cat_color == "#000000":
                continue
            words_set = category_sets.get(cat, set())
            cat_mask = Image.new("1", (W, H), 0)
            cdraw = ImageDraw.Draw(cat_mask)
            yline = body_y
            for li, line in enumerate(lines):
                if center_align:
                    start_x = int((W - line_widths[li]) / 2)
                else:
                    start_x = x0
                x = start_x
                for word, ww in line:
                    clean = re.sub(r"[^\w]", "", word.lower())
                    if clean in words_set:
                        cdraw.text((x, yline), word, fill=1, font=body_font)
                    x += ww + space_w
                yline += line_h
            layer = Image.new("RGB", (W, H), cat_color)
            composed = Image.alpha_composite(composed.convert("RGBA"), Image.composite(layer, Image.new("RGB", (W, H), "#FFFFFF"), cat_mask).convert("RGBA")).convert("RGB")

        body = composed
    else:
        # Standard AA rendering
        body = Image.new("RGB", (W, H), "#FFFFFF")
        bdraw = ImageDraw.Draw(body)
        yline = body_y
        for li, line in enumerate(lines):
            if center_align:
                start_x = int((W - line_widths[li]) / 2)
            else:
                start_x = x0
            x = start_x
            for word, ww in line:
                clean = re.sub(r"[^\w]", "", word.lower())
                
                # === 変更後 (より柔軟なロジック) ===
                # 1. まず全体のデフォルト色を決定する
                if baseline_text_color is not None:
                    default_fill = f"#{baseline_text_color[0]:02x}{baseline_text_color[1]:02x}{baseline_text_color[2]:02x}"
                else:
                    default_fill = color_settings.get("default", "#000000") # "default"キーがなければ黒
                
                fill = default_fill

                # 2. もし単語が特定カテゴリに属し、その色が指定されていれば上書きする
                for cat in category_order:
                    if clean in category_sets.get(cat, set()):
                        if cat in color_settings:
                            fill = color_settings[cat]
                            break # 最初にマッチしたカテゴリで確定
                # === 変更ここまで ===
                bdraw.text((x, yline), word, fill=fill, font=body_font)
                x += ww + space_w
            yline += line_h

    # Merge layers
    merged = Image.alpha_composite(bg.convert("RGBA"), header.convert("RGBA")).convert("RGB")
    merged = Image.alpha_composite(merged.convert("RGBA"), body.convert("RGBA")).convert("RGB")
    return merged



