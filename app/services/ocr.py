"""Service OCR avec TrOCR ou Qwen2-VL pour manuscrits français."""
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Cache des modèles
_trocr_model = None
_trocr_processor = None
_qwen_model = None
_qwen_processor = None
_device = None
_current_model = "trocr"  # "trocr" ou "qwen"

# Modèles disponibles
AVAILABLE_MODELS = {
    "trocr": {
        "name": "TrOCR French",
        "description": "Spécialisé manuscrit français, rapide (~2 Go)",
        "model_id": "agomberto/trocr-large-handwritten-fr"
    },
    "qwen": {
        "name": "Qwen2-VL 2B",
        "description": "Vision-language généraliste, plus lent (~6 Go)",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct"
    }
}


def get_device() -> str:
    """Retourne le device optimal (MPS sur Mac, CUDA sur GPU, sinon CPU)."""
    global _device
    if _device is None:
        if torch.backends.mps.is_available():
            _device = "mps"
        elif torch.cuda.is_available():
            _device = "cuda"
        else:
            _device = "cpu"
    return _device


def get_current_model() -> str:
    """Retourne le modèle actuellement sélectionné."""
    return _current_model


def set_current_model(model_key: str):
    """Change le modèle à utiliser."""
    global _current_model
    if model_key in AVAILABLE_MODELS:
        _current_model = model_key


def get_trocr_model_and_processor():
    """Charge et met en cache le modèle TrOCR français."""
    global _trocr_model, _trocr_processor

    if _trocr_model is None or _trocr_processor is None:
        model_name = AVAILABLE_MODELS["trocr"]["model_id"]
        _trocr_processor = TrOCRProcessor.from_pretrained(model_name)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)

        device = get_device()
        _trocr_model = _trocr_model.to(device)
        _trocr_model.eval()

    return _trocr_model, _trocr_processor


def get_qwen_model_and_processor():
    """Charge et met en cache le modèle Qwen2-VL."""
    global _qwen_model, _qwen_processor

    if _qwen_model is None or _qwen_processor is None:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        model_name = AVAILABLE_MODELS["qwen"]["model_id"]

        _qwen_processor = AutoProcessor.from_pretrained(model_name)
        _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if get_device() != "cpu" else torch.float32,
            device_map="auto"
        )

    return _qwen_model, _qwen_processor


def transcribe_line_trocr(image: Image.Image) -> dict:
    """Transcrit une ligne avec TrOCR."""
    model, processor = get_trocr_model_and_processor()
    device = get_device()

    if image.mode != "RGB":
        image = image.convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=128)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {"text": text, "confidence": None}


def transcribe_line_qwen(image: Image.Image) -> dict:
    """Transcrit une ligne avec Qwen2-VL."""
    model, processor = get_qwen_model_and_processor()

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Prompt pour l'OCR de manuscrit
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Transcris exactement le texte manuscrit de cette image. Réponds uniquement avec le texte, sans explication."}
            ]
        }
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_input],
        images=[image],
        return_tensors="pt",
        padding=True
    )

    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    return {"text": text.strip(), "confidence": None}


def transcribe_line(image: Image.Image, model_key: str = None) -> dict:
    """
    Transcrit une seule ligne de texte manuscrit.

    Args:
        image: Image PIL d'une ligne de texte
        model_key: "trocr" ou "qwen" (utilise le modèle courant si None)

    Returns:
        dict avec text et confidence
    """
    if model_key is None:
        model_key = _current_model

    if model_key == "qwen":
        return transcribe_line_qwen(image)
    else:
        return transcribe_line_trocr(image)


def transcribe_lines(images: list[Image.Image], model_key: str = None, progress_callback=None) -> list[dict]:
    """
    Transcrit plusieurs lignes de texte.

    Args:
        images: Liste d'images PIL (une par ligne)
        model_key: "trocr" ou "qwen"
        progress_callback: Fonction optionnelle appelée avec (current, total)

    Returns:
        Liste de dicts avec text et confidence
    """
    results = []
    total = len(images)

    for i, image in enumerate(images):
        result = transcribe_line(image, model_key=model_key)
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total)

    return results


def transcribe_from_segmentation(
    original_image: Image.Image,
    segmentation: dict,
    model_key: str = None,
    progress_callback=None
) -> list[dict]:
    """
    Transcrit toutes les lignes à partir d'une segmentation.

    Args:
        original_image: Image originale complète
        segmentation: Résultat de segment_image()
        model_key: "trocr" ou "qwen"
        progress_callback: Fonction optionnelle appelée avec (current, total)

    Returns:
        Liste de dicts avec line_number, text, confidence, bbox
    """
    results = []
    lines = segmentation["lines"]
    total = len(lines)

    # Convertir en RGB si nécessaire
    if original_image.mode != "RGB":
        original_image = original_image.convert("RGB")

    for i, line in enumerate(lines):
        bbox = line["bounding_box"]

        # Extraire la région avec marge
        margin = 5
        left = max(0, bbox["x"] - margin)
        top = max(0, bbox["y"] - margin)
        right = min(original_image.width, bbox["x"] + bbox["width"] + margin)
        bottom = min(original_image.height, bbox["y"] + bbox["height"] + margin)

        line_image = original_image.crop((left, top, right, bottom))

        # Transcrire
        ocr_result = transcribe_line(line_image, model_key=model_key)

        results.append({
            "line_number": line["line_number"],
            "text": ocr_result["text"],
            "confidence": ocr_result["confidence"],
            "bounding_box": bbox
        })

        if progress_callback:
            progress_callback(i + 1, total)

    return results
