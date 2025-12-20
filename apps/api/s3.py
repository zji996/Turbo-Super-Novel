from __future__ import annotations

import os
from functools import lru_cache

import boto3
from botocore.config import Config


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


@lru_cache
def s3_client():
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


def s3_bucket_name() -> str:
    return _require_env("S3_BUCKET_NAME")


def ensure_bucket_exists(bucket: str) -> None:
    client = s3_client()
    try:
        client.head_bucket(Bucket=bucket)
        return
    except Exception:
        pass

    try:
        client.create_bucket(Bucket=bucket)
    except Exception:
        client.head_bucket(Bucket=bucket)

