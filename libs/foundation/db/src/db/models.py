from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TurboDiffusionJob(Base):
    __tablename__ = "turbodiffusion_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    status: Mapped[str] = mapped_column(String(32), index=True)

    prompt: Mapped[str] = mapped_column(Text)
    seed: Mapped[int] = mapped_column()
    num_steps: Mapped[int] = mapped_column()
    quantized: Mapped[bool] = mapped_column()

    input_bucket: Mapped[str] = mapped_column(String(255))
    input_key: Mapped[str] = mapped_column(Text)
    output_bucket: Mapped[str] = mapped_column(String(255))
    output_key: Mapped[str] = mapped_column(Text)

    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class NovelProject(Base):
    """Novel video project - the top-level entity for a novel-to-video conversion."""

    __tablename__ = "novel_projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(
        String(32), index=True
    )  # CREATED, PARSING, READY, PROCESSING, COMPLETED, FAILED
    source_text: Mapped[str] = mapped_column(Text)
    config: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Final output
    output_bucket: Mapped[str | None] = mapped_column(String(255), nullable=True)
    output_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class NovelScene(Base):
    """A scene within a novel project - represents one segment of the story."""

    __tablename__ = "novel_scenes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    sequence: Mapped[int] = mapped_column()  # Order in the project
    text: Mapped[str] = mapped_column(Text)  # The narration text for this scene
    image_prompt: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # Prompt for image generation
    status: Mapped[str] = mapped_column(
        String(32), index=True
    )  # PENDING, TTS_RUNNING, etc.

    # Generated assets
    audio_bucket: Mapped[str | None] = mapped_column(String(255), nullable=True)
    audio_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_bucket: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_bucket: Mapped[str | None] = mapped_column(String(255), nullable=True)
    video_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Metadata
    audio_duration_seconds: Mapped[float | None] = mapped_column(nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class NovelPipeline(Base):
    """Pipeline execution record - tracks a run of the novel video generation pipeline."""

    __tablename__ = "novel_pipelines"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    pipeline_type: Mapped[str] = mapped_column(
        String(32)
    )  # FULL, TTS_ONLY, IMAGEGEN_ONLY, etc.
    status: Mapped[str] = mapped_column(
        String(32), index=True
    )  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED

    # Progress tracking
    total_tasks: Mapped[int] = mapped_column(default=0)
    completed_tasks: Mapped[int] = mapped_column(default=0)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
