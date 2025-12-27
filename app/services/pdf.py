"""Service d'export PDF."""
from io import BytesIO
from datetime import datetime
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib import colors


def generate_searchable_pdf(
    original_image: Image.Image,
    lines_data: list,
    use_corrected: bool = True
) -> bytes:
    """
    Génère un PDF searchable avec l'image en fond et le texte superposé.

    Le texte est invisible mais sélectionnable (comme OCRmyPDF).

    Args:
        original_image: Image originale (PIL)
        lines_data: Liste des lignes avec bounding_box et text/text_corrected
        use_corrected: Utiliser le texte corrigé si disponible

    Returns:
        bytes du PDF généré
    """
    buffer = BytesIO()

    # Dimensions de l'image
    img_width, img_height = original_image.size

    # Créer le canvas avec la taille de l'image
    c = canvas.Canvas(buffer, pagesize=(img_width, img_height))

    # Convertir et ajouter l'image en fond
    img_buffer = BytesIO()
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    original_image.save(img_buffer, format='JPEG', quality=95)
    img_buffer.seek(0)

    # Dessiner l'image (origine en bas à gauche pour ReportLab)
    from reportlab.lib.utils import ImageReader
    img_reader = ImageReader(img_buffer)
    c.drawImage(img_reader, 0, 0, width=img_width, height=img_height)

    # Mode texte invisible (rendering mode 3 = invisible)
    c.setFillColor(colors.Color(0, 0, 0, alpha=0))  # Transparent

    # Superposer le texte sur chaque ligne
    for line in lines_data:
        bbox = line.get("bounding_box", {})

        # Choisir le texte à utiliser
        if use_corrected and line.get("text_corrected"):
            text = line["text_corrected"]
        else:
            text = line.get("text", "")

        if not text or not bbox:
            continue

        # Coordonnées (convertir Y car ReportLab a l'origine en bas)
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        width = bbox.get("width", 100)
        height = bbox.get("height", 20)

        # Y inversé pour ReportLab
        y_pdf = img_height - y - height

        # Calculer la taille de police pour que le texte rentre dans la bbox
        # Approximation : largeur moyenne d'un caractère = 0.6 * font_size
        if len(text) > 0:
            font_size = min(height * 0.8, (width / len(text)) / 0.5)
            font_size = max(6, min(font_size, height))  # Entre 6 et hauteur de la bbox
        else:
            font_size = height * 0.8

        c.setFont("Helvetica", font_size)

        # Dessiner le texte (invisible mais sélectionnable)
        # On utilise le mode de rendu 3 (invisible)
        text_obj = c.beginText(x, y_pdf + height * 0.2)
        text_obj.setTextRenderMode(3)  # Invisible
        text_obj.textLine(text)
        c.drawText(text_obj)

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def generate_report_pdf(
    document_name: str,
    original_image: Image.Image = None,
    raw_text: str = None,
    corrected_text: str = None,
    summary: str = None,
    document_context: str = None
) -> bytes:
    """
    Génère un PDF rapport avec la transcription formatée.

    Args:
        document_name: Nom du document
        original_image: Image originale (PIL)
        raw_text: Texte OCR brut
        corrected_text: Texte corrigé
        summary: Résumé des plaintes
        document_context: Contexte du document

    Returns:
        bytes du PDF généré
    """
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()

    # Styles personnalisés
    styles.add(ParagraphStyle(
        name='DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name='DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.gray,
        alignment=TA_CENTER,
        spaceAfter=30
    ))
    styles.add(ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#2c3e50')
    ))
    styles.add(ParagraphStyle(
        name='TranscriptText',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='SummaryText',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        leftIndent=20,
        spaceAfter=4
    ))

    story = []

    # Titre
    story.append(Paragraph(document_name, styles['DocTitle']))
    subtitle = f"Transcription générée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}"
    story.append(Paragraph(subtitle, styles['DocSubtitle']))

    # Contexte
    if document_context:
        story.append(Paragraph("Contexte", styles['SectionTitle']))
        safe_ctx = document_context.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        story.append(Paragraph(safe_ctx, styles['TranscriptText']))
        story.append(Spacer(1, 10))

    # Image miniature
    if original_image:
        story.append(Paragraph("Document original", styles['SectionTitle']))
        img_buffer = BytesIO()
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        original_image.save(img_buffer, format='JPEG', quality=85)
        img_buffer.seek(0)

        page_width = A4[0] - 4*cm
        img_w, img_h = original_image.size
        ratio = min(page_width / img_w, 400 / img_h)
        rl_image = RLImage(img_buffer, width=img_w*ratio, height=img_h*ratio)
        story.append(rl_image)
        story.append(Spacer(1, 20))

    # Texte corrigé
    if corrected_text:
        story.append(Paragraph("Transcription", styles['SectionTitle']))
        for para in corrected_text.split('\n'):
            if para.strip():
                safe_para = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(safe_para, styles['TranscriptText']))
        story.append(Spacer(1, 20))

    # Résumé
    if summary:
        story.append(PageBreak())
        story.append(Paragraph("Analyse et résumé", styles['SectionTitle']))
        for line in summary.split('\n'):
            if line.strip():
                safe_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                if line.startswith('## '):
                    story.append(Spacer(1, 10))
                    story.append(Paragraph(f"<b>{safe_line[3:]}</b>", styles['TranscriptText']))
                elif line.startswith('- '):
                    story.append(Paragraph(f"• {safe_line[2:]}", styles['SummaryText']))
                else:
                    story.append(Paragraph(safe_line, styles['SummaryText']))

    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph(
        "Document généré par HTR-Local",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.gray, alignment=TA_CENTER)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
