#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional
from PIL import Image
import torch


class VLMHandler:
    """Unified VLM handler using Transformers with robust fallbacks.

    Tries model-specific classes first, then falls back to pipeline(image-to-text).
    Works with many HF VLMs when trust_remote_code=True.
    """

    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.pipeline = None
        self.is_qwen2vl = False

        # 0) Explicit Qwen2-VL path (AutoModelForCausalLM + AutoProcessor)
        try:
            if any(k in model_name.lower() for k in ["qwen2-vl", "qwen2_vl", "qwen/qwen2-vl"]):
                from transformers import AutoProcessor
                from transformers import Qwen2VLForConditionalGeneration
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
                )
                # Qwen2-VL tokenizer is available via processor
                self.tokenizer = getattr(self.processor, "tokenizer", None)
                self.is_qwen2vl = True
                return
        except Exception:
            self.model = None
            self.processor = None

        # 1) Standard AutoModelForImageTextToText path (LLaVA-1.6, IDEFICS2)
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
            )
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            return
        except Exception:
            self.model = None
            self.processor = None

        # 2) Fallback to pipeline (try image-text-to-text then image-to-text)
        try:
            from transformers import pipeline
            try:
                self.pipeline = pipeline(
                    task="image-text-to-text",
                    model=model_name,
                    torch_dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                )
            except Exception:
                self.pipeline = pipeline(
                    task="image-to-text",
                    model=model_name,
                    torch_dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load VLM model '{model_name}': {e}")

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        num_beams: int = 1,
    ) -> str:
        # Qwen2-VL path: use processor(text, images) + AutoModelForCausalLM.generate
        if self.is_qwen2vl and self.model is not None and self.processor is not None:
            # Preferred path per official docs
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
            # Prepare text prompt using chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Prepare vision inputs
            try:
                from qwen_vl_utils import process_vision_info  # type: ignore
                image_inputs, video_inputs = process_vision_info(messages)
            except Exception:
                image_inputs, video_inputs = [image], None

            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            )
            # Move to device
            try:
                inputs = inputs.to(self.model.device)  # type: ignore[attr-defined]
            except Exception:
                for k, v in list(inputs.items()):
                    if hasattr(v, "to"):
                        inputs[k] = v.to(getattr(self.model, "device", "cuda" if torch.cuda.is_available() else "cpu"))

            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=bool(temperature and temperature > 0))
            if temperature and temperature > 0:
                gen_kwargs['temperature'] = temperature
            if num_beams > 1:
                gen_kwargs['num_beams'] = num_beams

            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = self.processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            return output_text[0] if output_text else ""


        if self.pipeline is not None:
            if temperature and temperature > 0:
                outputs = self.pipeline(image, prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            else:
                outputs = self.pipeline(image, prompt=prompt, max_new_tokens=max_new_tokens)
            if isinstance(outputs, list) and len(outputs) and isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
                return outputs[0]["generated_text"].strip()
            return str(outputs)

        if self.model is None or self.processor is None:
            raise RuntimeError("VLM not initialized")



        # Build processor inputs robustly for various processors
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        # Some processors return dict of tensors; move to model device
        try:
            inputs = inputs.to(self.model.device)  # type: ignore[attr-defined]
        except Exception:
            for k, v in list(inputs.items()):
                if hasattr(v, "to"):
                    inputs[k] = v.to(getattr(self.model, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
        if temperature and temperature > 0:
            gen_kwargs['temperature'] = temperature
        if num_beams > 1:
            gen_kwargs['num_beams'] = num_beams
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        tokenizer = self.tokenizer or getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Fallback decode
        try:
            return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        except Exception:
            return str(output_ids[0].tolist())









