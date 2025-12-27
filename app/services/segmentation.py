"""Service de segmentation avec Kraken."""
from pathlib import Path
from PIL import Image

from kraken import blla


def segment_image(image: Image.Image) -> dict:
    """
    Segmente une image pour détecter les lignes de texte.

    Args:
        image: Image PIL à segmenter

    Returns:
        dict avec:
            - lines: liste de dicts avec bounding_box, baseline, polygon
            - regions: régions détectées (si disponibles)
    """
    # Convertir en RGB si nécessaire
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Lancer la segmentation (utilise le modèle par défaut de Kraken)
    result = blla.segment(image)

    # Extraire les lignes
    lines = []
    for idx, line in enumerate(result.lines):
        # Baseline : liste de points (x, y)
        baseline_points = [(int(p[0]), int(p[1])) for p in line.baseline]

        # Polygon : contour de la ligne
        polygon_points = [(int(p[0]), int(p[1])) for p in line.boundary] if line.boundary else []

        # Bounding box depuis le polygon ou la baseline
        if polygon_points:
            xs = [p[0] for p in polygon_points]
            ys = [p[1] for p in polygon_points]
        else:
            xs = [p[0] for p in baseline_points]
            ys = [p[1] for p in baseline_points]

        bbox = {
            "x": min(xs),
            "y": min(ys),
            "width": max(xs) - min(xs),
            "height": max(ys) - min(ys)
        }

        lines.append({
            "line_number": idx + 1,
            "bounding_box": bbox,
            "baseline": baseline_points,
            "polygon": polygon_points
        })

    return {
        "lines": lines,
        "image_size": {"width": image.width, "height": image.height}
    }


def draw_segmentation_overlay(image: Image.Image, segmentation: dict) -> Image.Image:
    """
    Dessine les lignes de segmentation sur l'image.

    Args:
        image: Image originale
        segmentation: Résultat de segment_image()

    Returns:
        Image avec overlay des segmentations
    """
    from PIL import ImageDraw

    # Copier l'image pour ne pas modifier l'originale
    overlay = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    for line in segmentation["lines"]:
        # Dessiner le polygon (zone de la ligne) en bleu transparent
        if line["polygon"]:
            # Créer un calque semi-transparent
            polygon_overlay = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
            polygon_draw = ImageDraw.Draw(polygon_overlay)
            polygon_draw.polygon(line["polygon"], fill=(0, 100, 255, 50), outline=(0, 100, 255, 200))
            overlay = Image.alpha_composite(overlay, polygon_overlay)
            draw = ImageDraw.Draw(overlay)

        # Dessiner la baseline en rouge
        if len(line["baseline"]) >= 2:
            draw.line(line["baseline"], fill=(255, 50, 50, 255), width=2)

        # Numéro de ligne
        if line["baseline"]:
            x, y = line["baseline"][0]
            draw.text((x - 25, y - 10), str(line["line_number"]), fill=(255, 255, 0, 255))

    return overlay


def extract_line_images(image: Image.Image, segmentation: dict, output_dir: Path) -> list[Path]:
    """
    Extrait les images individuelles de chaque ligne.

    Args:
        image: Image originale
        segmentation: Résultat de segment_image()
        output_dir: Dossier de sortie

    Returns:
        Liste des chemins des images extraites
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for line in segmentation["lines"]:
        bbox = line["bounding_box"]

        # Ajouter une marge
        margin = 5
        left = max(0, bbox["x"] - margin)
        top = max(0, bbox["y"] - margin)
        right = min(image.width, bbox["x"] + bbox["width"] + margin)
        bottom = min(image.height, bbox["y"] + bbox["height"] + margin)

        # Extraire la région
        line_img = image.crop((left, top, right, bottom))

        # Sauvegarder
        path = output_dir / f"line_{line['line_number']:03d}.png"
        line_img.save(path)
        paths.append(path)

    return paths
