from datetime import datetime
from enum import Enum
from typing import Optional
from sqlmodel import SQLModel, Field, JSON, Column
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship


class PageStatus(str, Enum):
    UPLOADED = "uploaded"
    SEGMENTED = "segmented"
    TRANSCRIBED = "transcribed"
    CORRECTED = "corrected"
    VALIDATED = "validated"


class Document(SQLModel, table=True):
    __tablename__ = "documents"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata_: Optional[dict] = Field(default=None, sa_column=Column(JSON))


class Page(SQLModel, table=True):
    __tablename__ = "pages"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", index=True)
    image_path: str
    page_number: int = Field(default=1)
    status: PageStatus = Field(default=PageStatus.UPLOADED)
    created_at: datetime = Field(default_factory=datetime.now)


class Line(SQLModel, table=True):
    __tablename__ = "lines"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    page_id: int = Field(foreign_key="pages.id", index=True)
    line_number: int
    bounding_box: dict = Field(sa_column=Column(JSON))
    baseline: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    image_path: Optional[str] = None
    text_raw: Optional[str] = None
    text_corrected: Optional[str] = None
    confidence_score: Optional[float] = None
    is_validated: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.now)
