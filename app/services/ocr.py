"""Service OCR avec TrOCR pour manuscrits français."""
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Cache du modèle
_model = None
_processor = None
_device = None

# Modèle utilisé
MODEL_NAME = "agomberto/trocr-large-handwritten-fr"


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


def get_model_and_processor():
    """Charge et met en cache le modèle TrOCR français."""
    global _model, _processor

    if _model is None or _processor is None:
        _processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
        _model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

        device = get_device()
        _model = _model.to(device)
        _model.eval()

    return _model, _processor


def transcribe_line(image: Image.Image) -> dict:
    """
    Transcrit une seule ligne de texte manuscrit.

    Args:
        image: Image PIL d'une ligne de texte

    Returns:
        dict avec text et confidence
    """
    model, processor = get_model_and_processor()
    device = get_device()

    if image.mode != "RGB":
        image = image.convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=128)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {"text": text, "confidence": None}


def transcribe_lines(images: list[Image.Image], progress_callback=None) -> list[dict]:
    """
    Transcrit plusieurs lignes de texte.

    Args:
        images: Liste d'images PIL (une par ligne)
        progress_callback: Fonction optionnelle appelée avec (current, total)

    Returns:
        Liste de dicts avec text et confidence
    """
    results = []
    total = len(images)

    for i, image in enumerate(images):
        result = transcribe_line(image)
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total)

    return results


def transcribe_from_segmentation(
    original_image: Image.Image,
    segmentation: dict,
    progress_callback=None
) -> list[dict]:
    """
    Transcrit toutes les lignes à partir d'une segmentation.

    Args:
        original_image: Image originale complète
        segmentation: Résultat de segment_image()
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
        ocr_result = transcribe_line(line_image)

        results.append({
            "line_number": line["line_number"],
            "text": ocr_result["text"],
            "confidence": ocr_result["confidence"],
            "bounding_box": bbox
        })

        if progress_callback:
            progress_callback(i + 1, total)

    return results
