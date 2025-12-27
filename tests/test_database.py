"""Test du CRUD pour la base de données."""
import sys
from pathlib import Path

# Ajouter le chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlmodel import select
from app.database.models import Document, Page, Line, PageStatus
from app.database.db import init_db, get_session


def test_crud():
    """Test complet du CRUD."""
    print("=== Test CRUD Base de Données ===\n")

    # 1. Initialiser la DB
    print("1. Initialisation de la base de données...")
    init_db()
    print("   OK\n")

    with get_session() as session:
        # 2. Créer un document
        print("2. Création d'un document...")
        doc = Document(
            name="Lettre_1905_001",
            metadata_={"source": "Archives familiales", "year": 1905}
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)
        print(f"   Document créé: ID={doc.id}, name={doc.name}")
        print(f"   Métadonnées: {doc.metadata_}\n")

        # 3. Ajouter une page
        print("3. Ajout d'une page...")
        page = Page(
            document_id=doc.id,
            image_path="/data/uploads/lettre_001_p1.png",
            page_number=1,
            status=PageStatus.UPLOADED
        )
        session.add(page)
        session.commit()
        session.refresh(page)
        print(f"   Page créée: ID={page.id}, status={page.status}\n")

        # 4. Ajouter des lignes
        print("4. Ajout de lignes...")
        lines_data = [
            {
                "line_number": 1,
                "bounding_box": {"x": 50, "y": 100, "width": 800, "height": 40},
                "baseline": {"points": [[50, 130], [850, 135]]},
                "text_raw": "Mon cher ami,",
                "confidence_score": 0.92
            },
            {
                "line_number": 2,
                "bounding_box": {"x": 50, "y": 150, "width": 800, "height": 40},
                "baseline": {"points": [[50, 180], [850, 185]]},
                "text_raw": "Je vous ecris pour vous faire part",
                "confidence_score": 0.85
            },
        ]

        for line_data in lines_data:
            line = Line(page_id=page.id, **line_data)
            session.add(line)

        session.commit()
        print(f"   {len(lines_data)} lignes créées\n")

        # 5. Mettre à jour le status de la page
        print("5. Mise à jour du status de la page...")
        page.status = PageStatus.TRANSCRIBED
        session.add(page)
        session.commit()
        print(f"   Nouveau status: {page.status}\n")

        # 6. Corriger une ligne avec LLM
        print("6. Simulation correction LLM...")
        line = session.exec(
            select(Line).where(Line.page_id == page.id).where(Line.line_number == 2)
        ).first()
        line.text_corrected = "Je vous écris pour vous faire part"
        line.is_validated = True
        session.add(line)
        session.commit()
        print(f"   Ligne {line.line_number}:")
        print(f"   - Raw:       {line.text_raw}")
        print(f"   - Corrigé:   {line.text_corrected}")
        print(f"   - Validé:    {line.is_validated}\n")

        # 7. Lecture des données
        print("7. Lecture des données persistées...")
        doc_read = session.exec(select(Document).where(Document.id == doc.id)).first()
        print(f"   Document: {doc_read.name}")
        print(f"   Pages: {len(doc_read.pages)}")
        for p in doc_read.pages:
            print(f"     - Page {p.page_number}: {p.status}, {len(p.lines)} lignes")

    print("\n=== Test CRUD réussi ===")


if __name__ == "__main__":
    test_crud()
