### Models

Unified interface for open-source VLMs.

Files
- `unified_handler.py`: Robust loader for HF VLMs. Tries model-specific classes, then generic, then `pipeline('image-to-text')`.
- (Optional) `llava_handler.py`: Direct LLaVA HF handler (kept for reference).

Supported (Phase-1 set)
- LLaVA-1.6 Mistral 7B: `llava-hf/llava-v1.6-mistral-7b-hf`
- Qwen2-VL 7B Instruct: `Qwen/Qwen2-VL-7B-Instruct`
- InternVL2 8B: `OpenGVLab/InternVL2-8B`
- IDEFICS2 8B: `HuggingFaceM4/idefics2-8b`
- CogVLM2 19B Instruct: `THUDM/cogvlm2-19b-instruct`

Usage
```python
from PIL import Image
from src.models.unified_handler import VLMHandler

handler = VLMHandler("llava-hf/llava-v1.6-mistral-7b-hf")
text = handler.generate(Image.open("/path/to/img.png").convert("RGB"), prompt="USER: <image> ... ASSISTANT:")
```








