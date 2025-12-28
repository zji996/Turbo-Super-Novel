"""Data models for novel video projects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProjectStatus(str, Enum):
    """Project lifecycle status."""
    CREATED = "CREATED"
    PARSING = "PARSING"
    READY = "READY"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SceneStatus(str, Enum):
    """Scene processing status."""
    PENDING = "PENDING"
    TTS_RUNNING = "TTS_RUNNING"
    TTS_DONE = "TTS_DONE"
    IMAGEGEN_RUNNING = "IMAGEGEN_RUNNING"
    IMAGEGEN_DONE = "IMAGEGEN_DONE"
    VIDEOGEN_RUNNING = "VIDEOGEN_RUNNING"
    VIDEOGEN_DONE = "VIDEOGEN_DONE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class PipelineType(str, Enum):
    """Types of pipelines that can be executed."""
    FULL = "FULL"  # TTS + ImageGen + VideoGen + Compose
    TTS_ONLY = "TTS_ONLY"
    IMAGEGEN_ONLY = "IMAGEGEN_ONLY"
    VIDEOGEN_ONLY = "VIDEOGEN_ONLY"
    COMPOSE_ONLY = "COMPOSE_ONLY"


@dataclass
class NovelProject:
    """In-memory representation of a novel video project.
    
    This is a lightweight data transfer object. For database persistence,
    use the corresponding SQLAlchemy model in libs/dbcore.
    """
    id: str
    name: str
    status: ProjectStatus
    source_text: str
    config: dict[str, Any]
    output_bucket: str | None = None
    output_key: str | None = None
    
    @classmethod
    def from_db_row(cls, row: Any) -> NovelProject:
        """Create from database row."""
        return cls(
            id=str(row.id),
            name=row.name,
            status=ProjectStatus(row.status),
            source_text=row.source_text,
            config=row.config or {},
            output_bucket=row.output_bucket,
            output_key=row.output_key,
        )


@dataclass
class NovelScene:
    """In-memory representation of a scene within a project."""
    id: str
    project_id: str
    sequence: int
    text: str
    image_prompt: str | None
    status: SceneStatus
    audio_bucket: str | None = None
    audio_key: str | None = None
    image_bucket: str | None = None
    image_key: str | None = None
    video_bucket: str | None = None
    video_key: str | None = None
    
    @classmethod
    def from_db_row(cls, row: Any) -> NovelScene:
        """Create from database row."""
        return cls(
            id=str(row.id),
            project_id=str(row.project_id),
            sequence=row.sequence,
            text=row.text,
            image_prompt=row.image_prompt,
            status=SceneStatus(row.status),
            audio_bucket=row.audio_bucket,
            audio_key=row.audio_key,
            image_bucket=row.image_bucket,
            image_key=row.image_key,
            video_bucket=row.video_bucket,
            video_key=row.video_key,
        )


@dataclass
class NovelPipeline:
    """In-memory representation of a pipeline execution."""
    id: str
    project_id: str
    pipeline_type: PipelineType
    status: PipelineStatus
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    
    @classmethod
    def from_db_row(cls, row: Any) -> NovelPipeline:
        """Create from database row."""
        return cls(
            id=str(row.id),
            project_id=str(row.project_id),
            pipeline_type=PipelineType(row.pipeline_type),
            status=PipelineStatus(row.status),
            started_at=row.started_at.isoformat() if row.started_at else None,
            completed_at=row.completed_at.isoformat() if row.completed_at else None,
            error=row.error,
        )
