from pathlib import Path
from sqlmodel import SQLModel, Session, create_engine

# Chemin de la base de données
DB_PATH = Path(__file__).parent.parent.parent / "data" / "htr_local.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Engine SQLite
engine = create_engine(DATABASE_URL, echo=False)


def init_db():
    """Initialise la base de données et crée les tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    """Retourne une nouvelle session de base de données."""
    return Session(engine)
