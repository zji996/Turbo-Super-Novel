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

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
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

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

