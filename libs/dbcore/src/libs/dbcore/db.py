from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import TurboDiffusionJob


def database_url() -> str:
    return os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/tsn")


@lru_cache
def engine() -> Engine:
    return create_engine(database_url(), pool_pre_ping=True)


@lru_cache
def _sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=engine(), autoflush=False, autocommit=False, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = _sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all() -> None:
    from .models import Base

    Base.metadata.create_all(bind=engine())


def try_insert_job(job: TurboDiffusionJob) -> tuple[bool, str | None]:
    try:
        with session_scope() as session:
            session.add(job)
        return True, None
    except Exception as exc:
        return False, str(exc)


def try_update_job(
    job_id: str | uuid.UUID,
    *,
    status: str | None = None,
    error: str | None = None,
    result: dict | None = None,
) -> tuple[bool, str | None]:
    try:
        job_uuid = job_id if isinstance(job_id, uuid.UUID) else uuid.UUID(str(job_id))
    except Exception as exc:
        return False, f"invalid job_id: {exc}"

    try:
        with session_scope() as session:
            row = session.get(TurboDiffusionJob, job_uuid)
            if row is None:
                return False, "job not found"
            if status is not None:
                row.status = status
            row.error = error
            row.result = result
        return True, None
    except Exception as exc:
        return False, str(exc)
