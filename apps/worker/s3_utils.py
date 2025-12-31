"""Shared S3 utilities for worker tasks."""

from __future__ import annotations

import os
from functools import lru_cache

import boto3
from botocore.config import Config


def _require_env(key: str) -> str:
    """Get required environment variable or raise."""

    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


@lru_cache(maxsize=1)
def get_s3_client():
    """Get cached S3 client instance."""

    endpoint = _require_env("S3_ENDPOINT")
    access_key = _require_env("S3_ACCESS_KEY")
    secret_key = _require_env("S3_SECRET_KEY")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name=os.getenv("S3_REGION") or "us-east-1",
    )


def get_s3_bucket_name() -> str:
    """Get S3 bucket name from environment."""

    return os.getenv("S3_BUCKET_NAME") or os.getenv("S3_BUCKET") or "tsn"

