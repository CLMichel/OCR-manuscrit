"""HTR-Local - Application de transcription de manuscrits."""
import sys
from pathlib import Path

# Ajouter le chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image
from datetime import datetime
from sqlmodel import select

from app.database.models import Document, Page, Line, PageStatus
from app.database.db import init_db, get_session

# Import optionnel de la segmentation (peut échouer si lzma manque)
try:
    from app.services.segmentation import segment_image, draw_segmentation_overlay
    SEGMENTATION_AVAILABLE = True
except ImportError as e:
    SEGMENTATION_AVAILABLE = False
    SEGMENTATION_ERROR = str(e)

# Import du service OCR
from app.services.ocr import transcribe_from_segmentation, get_device

# Import du service LLM
from app.services.llm import correct_transcriptions, correct_full_text, summarize_text

# Import du service PDF
from app.services.pdf import generate_searchable_pdf, generate_report_pdf

# Configuration de la page
st.set_page_config(
    page_title="HTR-Local",
    page_icon="📜",
    layout="wide"
)

# Initialiser la base de données
init_db()

# Dossiers
DATA_DIR = Path(__file__).parent.parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
SEGMENTS_DIR = DATA_DIR / "segments"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)


def init_session_state():
    """Initialise l'état de session."""
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "segmentation" not in st.session_state:
        st.session_state.segmentation = None
    if "current_page_id" not in st.session_state:
        st.session_state.current_page_id = None
    if "current_doc_id" not in st.session_state:
        st.session_state.current_doc_id = None
    if "transcriptions" not in st.session_state:
        st.session_state.transcriptions = None
    if "corrections" not in st.session_state:
        st.session_state.corrections = None
    if "full_text_corrected" not in st.session_state:
        st.session_state.full_text_corrected = None
    if "text_summary" not in st.session_state:
        st.session_state.text_summary = None


def load_document(doc_id: int):
    """Charge un document existant et restaure l'état."""
    with get_session() as session:
        doc = session.exec(select(Document).where(Document.id == doc_id)).first()
        if not doc:
            return False

        # Charger la première page
        page = session.exec(
            select(Page).where(Page.document_id == doc_id).order_by(Page.page_number)
        ).first()

        if not page:
            return False

        # Charger l'image
        image_path = Path(page.image_path)
        if image_path.exists():
            st.session_state.current_image = Image.open(image_path)
        else:
            st.session_state.current_image = None

        st.session_state.current_doc_id = doc_id
        st.session_state.current_page_id = page.id

        # Charger les lignes si elles existent
        lines = session.exec(
            select(Line).where(Line.page_id == page.id).order_by(Line.line_number)
        ).all()

        if lines:
            # Reconstruire la segmentation
            st.session_state.segmentation = {
                "lines": [
                    {
                        "line_number": l.line_number,
                        "bounding_box": l.bounding_box,
                        "baseline": l.baseline.get("points", []) if l.baseline else [],
                        "polygon": l.bounding_box.get("polygon", [])
                    }
                    for l in lines
                ],
                "image_size": {"width": st.session_state.current_image.width,
                               "height": st.session_state.current_image.height}
                if st.session_state.current_image else {}
            }

            # Reconstruire les transcriptions
            if any(l.text_raw for l in lines):
                st.session_state.transcriptions = [
                    {
                        "line_number": l.line_number,
                        "text": l.text_raw or "",
                        "bounding_box": l.bounding_box,
                        "confidence": l.confidence_score
                    }
                    for l in lines
                ]

            # Reconstruire les corrections
            if any(l.text_corrected for l in lines):
                st.session_state.corrections = [
                    {
                        "line_number": l.line_number,
                        "text": l.text_raw or "",
                        "text_corrected": l.text_corrected or "",
                        "bounding_box": l.bounding_box
                    }
                    for l in lines
                ]
        else:
            st.session_state.segmentation = None
            st.session_state.transcriptions = None
            st.session_state.corrections = None

        # Reset les corrections globales (pas stockées en DB pour l'instant)
        st.session_state.full_text_corrected = None
        st.session_state.text_summary = None

        return True


def save_lines_to_db(lines_data: list, page_id: int):
    """Sauvegarde les lignes en base de données."""
    with get_session() as session:
        # Supprimer les lignes existantes pour cette page
        existing = session.exec(select(Line).where(Line.page_id == page_id)).all()
        for line in existing:
            session.delete(line)
        session.commit()

        # Ajouter les nouvelles lignes
        for data in lines_data:
            line = Line(
                page_id=page_id,
                line_number=data["line_number"],
                bounding_box=data.get("bounding_box", {}),
                baseline={"points": data.get("baseline", [])},
                text_raw=data.get("text"),
                text_corrected=data.get("text_corrected"),
                confidence_score=data.get("confidence")
            )
            session.add(line)
        session.commit()


def update_page_status(page_id: int, status: PageStatus):
    """Met à jour le statut d'une page."""
    with get_session() as session:
        page = session.exec(select(Page).where(Page.id == page_id)).first()
        if page:
            page.status = status
            session.add(page)
            session.commit()


def main():
    st.title("📜 HTR-Local")
    st.caption("Transcription de manuscrits cursifs français")

    init_session_state()

    # Sidebar
    st.sidebar.header("Documents")

    # Charger les documents existants
    with get_session() as session:
        docs = session.exec(select(Document).order_by(Document.created_at.desc())).all()

    doc_options = ["+ Nouveau document"] + [f"{d.id}: {d.name}" for d in docs]
    selected = st.sidebar.selectbox("Document", doc_options, key="doc_selector")

    if selected != "+ Nouveau document":
        doc_id = int(selected.split(":")[0])
        if st.session_state.current_doc_id != doc_id:
            if load_document(doc_id):
                st.sidebar.success(f"Document chargé")
            else:
                st.sidebar.error("Erreur de chargement")

    st.sidebar.divider()
    st.sidebar.header("Pipeline")
    step = st.sidebar.radio(
        "Étape",
        ["1. Upload", "2. Segmentation", "3. Transcription", "4. Correction", "5. Export"],
        key="pipeline_step"
    )

    # Affichage selon l'étape
    if step == "1. Upload":
        show_upload_step()
    elif step == "2. Segmentation":
        show_segmentation_step()
    elif step == "3. Transcription":
        show_transcription_step()
    elif step == "4. Correction":
        show_correction_step()
    elif step == "5. Export":
        show_export_step()


def show_upload_step():
    """Étape 1 : Upload d'image."""
    st.header("1. Upload d'image")

    uploaded_file = st.file_uploader(
        "Choisir une image de manuscrit",
        type=["png", "jpg", "jpeg", "tiff", "tif"],
        key="image_uploader"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state.current_image = image
        st.image(image, caption=f"{uploaded_file.name} ({image.width}x{image.height})", width="stretch")

        col1, col2 = st.columns([3, 1])
        with col1:
            doc_name = st.text_input("Nom du document", value=Path(uploaded_file.name).stem)
        with col2:
            if st.button("Créer le document", type="primary"):
                image_path = UPLOADS_DIR / f"{doc_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                image.save(image_path)

                with get_session() as session:
                    doc = Document(name=doc_name)
                    session.add(doc)
                    session.commit()
                    session.refresh(doc)

                    page = Page(
                        document_id=doc.id,
                        image_path=str(image_path),
                        page_number=1,
                        status=PageStatus.UPLOADED
                    )
                    session.add(page)
                    session.commit()
                    session.refresh(page)

                    st.session_state.current_doc_id = doc.id
                    st.session_state.current_page_id = page.id

                # Reset autres états
                st.session_state.segmentation = None
                st.session_state.transcriptions = None
                st.session_state.corrections = None

                st.success(f"Document '{doc_name}' créé ! Passez à l'étape Segmentation.")
                st.rerun()


def show_segmentation_step():
    """Étape 2 : Segmentation des lignes."""
    st.header("2. Segmentation")

    if not SEGMENTATION_AVAILABLE:
        st.error(f"Segmentation non disponible: {SEGMENTATION_ERROR}")
        return

    if st.session_state.current_image is None:
        st.warning("Veuillez d'abord uploader une image à l'étape 1.")
        return

    image = st.session_state.current_image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Segmentation")

        if st.button("Lancer la segmentation", type="primary"):
            with st.spinner("Segmentation en cours..."):
                try:
                    segmentation = segment_image(image)
                    st.session_state.segmentation = segmentation

                    overlay = draw_segmentation_overlay(image, segmentation)
                    st.image(overlay, width="stretch")
                    st.success(f"{len(segmentation['lines'])} lignes détectées")

                    # Sauvegarder en base
                    if st.session_state.current_page_id:
                        save_lines_to_db(segmentation["lines"], st.session_state.current_page_id)
                        update_page_status(st.session_state.current_page_id, PageStatus.SEGMENTED)

                except Exception as e:
                    st.error(f"Erreur de segmentation: {e}")

        elif st.session_state.segmentation:
            overlay = draw_segmentation_overlay(image, st.session_state.segmentation)
            st.image(overlay, width="stretch")
            st.info(f"{len(st.session_state.segmentation['lines'])} lignes détectées")

    if st.session_state.segmentation:
        st.subheader("Lignes détectées")
        for line in st.session_state.segmentation["lines"]:
            with st.expander(f"Ligne {line['line_number']}"):
                st.json(line)


def show_transcription_step():
    """Étape 3 : Transcription OCR."""
    st.header("3. Transcription OCR")

    if st.session_state.current_image is None:
        st.warning("Veuillez d'abord uploader une image à l'étape 1.")
        return

    if st.session_state.segmentation is None:
        st.warning("Veuillez d'abord segmenter l'image à l'étape 2.")
        return

    image = st.session_state.current_image
    segmentation = st.session_state.segmentation
    lines = segmentation["lines"]

    st.info(f"Device: **{get_device().upper()}** | Lignes à transcrire: **{len(lines)}**")

    if st.button("Lancer la transcription", type="primary"):
        progress_bar = st.progress(0, text="Chargement du modèle TrOCR...")

        def update_progress(current, total):
            progress_bar.progress(current / total, text=f"Transcription ligne {current}/{total}")

        try:
            transcriptions = transcribe_from_segmentation(image, segmentation, progress_callback=update_progress)
            st.session_state.transcriptions = transcriptions
            progress_bar.progress(1.0, text="Transcription terminée !")
            st.success(f"{len(transcriptions)} lignes transcrites")

            # Sauvegarder en base
            if st.session_state.current_page_id:
                save_lines_to_db(transcriptions, st.session_state.current_page_id)
                update_page_status(st.session_state.current_page_id, PageStatus.TRANSCRIBED)

        except Exception as e:
            st.error(f"Erreur de transcription: {e}")
            return

    if st.session_state.transcriptions:
        st.subheader("Résultats")
        if image.mode != "RGB":
            image = image.convert("RGB")

        for trans in st.session_state.transcriptions:
            line_num = trans["line_number"]
            bbox = trans["bounding_box"]
            margin = 5
            left = max(0, bbox["x"] - margin)
            top = max(0, bbox["y"] - margin)
            right = min(image.width, bbox["x"] + bbox["width"] + margin)
            bottom = min(image.height, bbox["y"] + bbox["height"] + margin)
            line_image = image.crop((left, top, right, bottom))

            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(line_image, caption=f"Ligne {line_num}", width="stretch")
            with col2:
                st.text_area(f"Texte ligne {line_num}", value=trans["text"], key=f"text_line_{line_num}", height=68)
            st.divider()


def show_correction_step():
    """Étape 4 : Correction LLM avec OpenAI."""
    st.header("4. Correction LLM")

    if st.session_state.transcriptions is None:
        st.warning("Veuillez d'abord transcrire l'image à l'étape 3.")
        return

    transcriptions = st.session_state.transcriptions
    image = st.session_state.current_image

    # Init état pour correction globale
    if "full_text_corrected" not in st.session_state:
        st.session_state.full_text_corrected = None

    # Champ de contexte documentaire
    st.subheader("Contexte du document")
    default_context = ""
    if st.session_state.current_doc_id:
        with get_session() as session:
            doc = session.exec(select(Document).where(Document.id == st.session_state.current_doc_id)).first()
            if doc and doc.metadata_:
                default_context = doc.metadata_.get("context", "")

    document_context = st.text_area(
        "Décris le document pour aider le LLM",
        value=default_context,
        placeholder="Ex: Livre de plaintes de la gare de Rambervillers (Vosges), 1905-1910. Contient des réclamations de voyageurs, noms de chefs de gare, références aux lignes ferroviaires de l'Est...",
        height=100,
        key="doc_context"
    )

    # Sauvegarder le contexte si modifié
    if document_context != default_context and st.session_state.current_doc_id:
        with get_session() as session:
            doc = session.exec(select(Document).where(Document.id == st.session_state.current_doc_id)).first()
            if doc:
                if doc.metadata_ is None:
                    doc.metadata_ = {}
                doc.metadata_["context"] = document_context
                session.add(doc)
                session.commit()

    st.divider()

    # Onglets pour les modes de correction et résumé
    tab_global, tab_lines, tab_summary = st.tabs(["Correction globale", "Ligne par ligne", "Résumé"])

    # === ONGLET CORRECTION GLOBALE ===
    with tab_global:
        st.info("Envoie tout le texte OCR au LLM pour une correction avec vue d'ensemble.")

        # Option modèle
        use_gpt4 = st.checkbox(
            "Utiliser GPT-4o (meilleur, plus lent et plus cher)",
            value=False,
            help="GPT-4o donne de meilleurs résultats mais coûte ~30x plus cher que GPT-4o-mini"
        )

        # Texte OCR complet (recalculé à chaque fois)
        full_text_raw = "\n".join([t["text"] for t in transcriptions])

        # Afficher le nombre de lignes pour debug
        st.caption(f"{len(transcriptions)} lignes | {len(full_text_raw)} caractères")

        col1, col2 = st.columns(2)

        # Clé unique par document pour éviter le cache
        doc_key = st.session_state.current_doc_id or "new"

        with col1:
            st.subheader("Texte OCR brut")
            st.text_area(
                "OCR",
                value=full_text_raw,
                height=400,
                key=f"full_raw_{doc_key}",
                disabled=True
            )

        with col2:
            st.subheader("Texte corrigé")
            model_name = "gpt-4o" if use_gpt4 else "gpt-4o-mini"

            if st.button(f"Corriger avec {model_name}", type="primary", key="btn_correct_full"):
                with st.spinner(f"Correction en cours avec {model_name}..."):
                    try:
                        corrected = correct_full_text(full_text_raw, document_context, use_gpt4=use_gpt4)
                        st.session_state.full_text_corrected = corrected
                        st.success("Correction terminée !")
                    except ValueError as e:
                        st.error(f"Erreur: {e}")
                        st.info("Crée un fichier `.env` avec ta clé API:\n```\nOPENAI_API_KEY=sk-xxx\n```")
                    except Exception as e:
                        st.error(f"Erreur: {e}")

            if st.session_state.full_text_corrected:
                st.text_area(
                    "Corrigé",
                    value=st.session_state.full_text_corrected,
                    height=400,
                    key=f"full_corrected_{doc_key}"
                )

                # Bouton pour copier
                st.download_button(
                    "Télécharger le texte corrigé",
                    data=st.session_state.full_text_corrected,
                    file_name="texte_corrige.txt",
                    mime="text/plain"
                )

    # === ONGLET LIGNE PAR LIGNE ===
    with tab_lines:
        st.info(f"Corrige chaque ligne individuellement avec contexte. | **{len(transcriptions)} lignes**")

        if st.button("Lancer la correction ligne par ligne", type="primary", key="btn_correct_lines"):
            progress_bar = st.progress(0, text="Connexion à OpenAI...")

            def update_progress(current, total):
                progress_bar.progress(current / total, text=f"Correction ligne {current}/{total}")

            try:
                corrections = correct_transcriptions(
                    transcriptions,
                    context_window=2,
                    document_context=document_context,
                    progress_callback=update_progress
                )
                st.session_state.corrections = corrections
                progress_bar.progress(1.0, text="Correction terminée !")
                st.success(f"{len(corrections)} lignes corrigées")

                # Sauvegarder en base
                if st.session_state.current_page_id:
                    save_lines_to_db(corrections, st.session_state.current_page_id)
                    update_page_status(st.session_state.current_page_id, PageStatus.CORRECTED)

            except ValueError as e:
                st.error(f"Erreur: {e}")
                st.info("Crée un fichier `.env` avec ta clé API:\n```\nOPENAI_API_KEY=sk-xxx\n```")
            except Exception as e:
                st.error(f"Erreur de correction: {e}")

        if st.session_state.corrections:
            st.subheader("Résultats (OCR vs Corrigé)")
            if image.mode != "RGB":
                image = image.convert("RGB")

            for corr in st.session_state.corrections:
                line_num = corr["line_number"]
                bbox = corr["bounding_box"]
                margin = 5
                left = max(0, bbox["x"] - margin)
                top = max(0, bbox["y"] - margin)
                right = min(image.width, bbox["x"] + bbox["width"] + margin)
                bottom = min(image.height, bbox["y"] + bbox["height"] + margin)
                line_image = image.crop((left, top, right, bottom))

                st.image(line_image, caption=f"Ligne {line_num}", width="stretch")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("OCR brut", value=corr["text"], key=f"raw_{line_num}", height=68, disabled=True)
                with col2:
                    st.text_area("Corrigé LLM", value=corr["text_corrected"], key=f"corrected_{line_num}", height=68)
                st.divider()

    # === ONGLET RÉSUMÉ ===
    with tab_summary:
        st.info("Analyse le texte et résume chaque plainte identifiée.")

        # Init état pour résumé
        if "text_summary" not in st.session_state:
            st.session_state.text_summary = None

        # Option modèle
        use_gpt4_summary = st.checkbox(
            "Utiliser GPT-4o pour le résumé",
            value=False,
            key="gpt4_summary",
            help="GPT-4o donne de meilleurs résultats d'analyse"
        )

        # Choisir la source du texte
        source_options = ["Texte OCR brut"]
        if st.session_state.full_text_corrected:
            source_options.append("Texte corrigé (global)")

        text_source = st.radio(
            "Texte à analyser",
            source_options,
            horizontal=True,
            key="summary_source"
        )

        if text_source == "Texte corrigé (global)" and st.session_state.full_text_corrected:
            text_to_analyze = st.session_state.full_text_corrected
        else:
            text_to_analyze = "\n".join([t["text"] for t in transcriptions])

        st.text_area("Texte analysé", value=text_to_analyze, height=200, disabled=True, key="text_to_summarize")

        model_name = "gpt-4o" if use_gpt4_summary else "gpt-4o-mini"
        if st.button(f"Générer le résumé avec {model_name}", type="primary", key="btn_summary"):
            with st.spinner("Analyse en cours..."):
                try:
                    summary = summarize_text(text_to_analyze, document_context, use_gpt4=use_gpt4_summary)
                    st.session_state.text_summary = summary
                    st.success("Analyse terminée !")
                except ValueError as e:
                    st.error(f"Erreur: {e}")
                except Exception as e:
                    st.error(f"Erreur: {e}")

        if st.session_state.text_summary:
            st.subheader("Résumé des plaintes")
            st.markdown(st.session_state.text_summary)

            st.download_button(
                "Télécharger le résumé",
                data=st.session_state.text_summary,
                file_name="resume_plaintes.md",
                mime="text/markdown"
            )


def show_export_step():
    """Étape 5 : Export PDF."""
    st.header("5. Export PDF")

    if st.session_state.transcriptions is None:
        st.warning("Veuillez d'abord transcrire l'image à l'étape 3.")
        return

    # Récupérer les données
    transcriptions = st.session_state.transcriptions
    corrections = st.session_state.get("corrections")
    image = st.session_state.current_image
    raw_text = "\n".join([t["text"] for t in transcriptions])
    corrected_text = st.session_state.get("full_text_corrected")
    summary = st.session_state.get("text_summary")

    # Récupérer le nom et contexte du document
    doc_name = "Document"
    doc_context = ""
    if st.session_state.current_doc_id:
        with get_session() as session:
            doc = session.exec(select(Document).where(Document.id == st.session_state.current_doc_id)).first()
            if doc:
                doc_name = doc.name
                if doc.metadata_:
                    doc_context = doc.metadata_.get("context", "")

    # Deux onglets pour les deux types d'export
    tab_searchable, tab_report = st.tabs(["PDF Searchable", "PDF Rapport"])

    # === PDF SEARCHABLE ===
    with tab_searchable:
        st.info("Image originale avec texte invisible superposé. Tu peux sélectionner/copier le texte sur l'image !")

        # Choisir quel texte utiliser
        use_corrected = st.checkbox(
            "Utiliser le texte corrigé (ligne par ligne)",
            value=bool(corrections),
            disabled=not corrections,
            help="Utilise les corrections ligne par ligne si disponibles"
        )

        # Données pour le PDF
        lines_for_pdf = corrections if (use_corrected and corrections) else transcriptions

        if st.button("Générer le PDF Searchable", type="primary", key="btn_searchable"):
            with st.spinner("Génération en cours..."):
                try:
                    pdf_bytes = generate_searchable_pdf(
                        original_image=image,
                        lines_data=lines_for_pdf,
                        use_corrected=use_corrected
                    )

                    st.success("PDF généré !")
                    st.download_button(
                        label="Télécharger le PDF Searchable",
                        data=pdf_bytes,
                        file_name=f"{doc_name}_searchable.pdf",
                        mime="application/pdf",
                        key="dl_searchable"
                    )
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # === PDF RAPPORT ===
    with tab_report:
        st.info("Rapport formaté avec transcription, résumé et miniature de l'image.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Texte OCR", f"{len(raw_text)} car.")
        with col2:
            st.metric("Texte corrigé", f"{len(corrected_text)} car." if corrected_text else "—")
        with col3:
            st.metric("Résumé", f"{len(summary)} car." if summary else "—")

        st.divider()

        include_summary = st.checkbox(
            "Inclure le résumé",
            value=bool(summary),
            disabled=not summary
        )

        if st.button("Générer le PDF Rapport", type="primary", key="btn_report"):
            with st.spinner("Génération en cours..."):
                try:
                    pdf_bytes = generate_report_pdf(
                        document_name=doc_name,
                        original_image=image,
                        corrected_text=corrected_text or raw_text,
                        summary=summary if include_summary else None,
                        document_context=doc_context
                    )

                    st.success("PDF généré !")
                    st.download_button(
                        label="Télécharger le PDF Rapport",
                        data=pdf_bytes,
                        file_name=f"{doc_name}_rapport.pdf",
                        mime="application/pdf",
                        key="dl_report"
                    )
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    import traceback
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
