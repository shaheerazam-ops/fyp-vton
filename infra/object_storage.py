"""
Object storage layer for the VTON platform.

Works against any S3-compatible backend (AWS S3, Cloudflare R2,
MinIO, GCS via interop mode). The rest of the codebase never talks
to boto3 directly — everything goes through this module, so the
backend can be swapped without touching model/API code.

Key layout convention:
    products/{brand_id}/{product_id}/garment.jpg
    products/{brand_id}/{product_id}/mask.png
    sessions/{session_id}/person.jpg
    sessions/{session_id}/result.png
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional

import boto3
from botocore.client import Config
from PIL import Image


@dataclass
class StorageConfig:
    bucket: str
    endpoint_url: Optional[str] = None      # None => real AWS S3; set for R2/MinIO
    region: str = "auto"
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    signed_url_ttl_seconds: int = 3600


class ObjectStore:
    def __init__(self, cfg: StorageConfig):
        self.cfg = cfg
        self._client = boto3.client(
            "s3",
            endpoint_url=cfg.endpoint_url,
            region_name=cfg.region,
            aws_access_key_id=cfg.access_key or os.environ.get("STORAGE_ACCESS_KEY"),
            aws_secret_access_key=cfg.secret_key or os.environ.get("STORAGE_SECRET_KEY"),
            config=Config(signature_version="s3v4"),
        )

    # ---- writes -------------------------------------------------------

    def put_image(self, key: str, image: Image.Image, format: str = "JPEG") -> str:
        """Upload a PIL image and return its storage key."""
        buf = io.BytesIO()
        image.save(buf, format=format, quality=92)
        buf.seek(0)
        self._client.put_object(
            Bucket=self.cfg.bucket,
            Key=key,
            Body=buf,
            ContentType=f"image/{format.lower()}",
        )
        return key

    def put_bytes(self, key: str, data: bytes, content_type: str) -> str:
        self._client.put_object(
            Bucket=self.cfg.bucket, Key=key, Body=data, ContentType=content_type
        )
        return key

    # ---- reads ----------------------------------------------------------

    def get_image(self, key: str) -> Image.Image:
        obj = self._client.get_object(Bucket=self.cfg.bucket, Key=key)
        return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.cfg.bucket, Key=key)
            return True
        except self._client.exceptions.ClientError:
            return False

    # ---- delivery -------------------------------------------------------

    def signed_url(self, key: str) -> str:
        """Time-limited URL the frontend can render directly (or point a CDN at)."""
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.cfg.bucket, "Key": key},
            ExpiresIn=self.cfg.signed_url_ttl_seconds,
        )

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.cfg.bucket, Key=key)


# ---- key helpers (keep naming convention in one place) ------------------

def person_key(session_id: str) -> str:
    return f"sessions/{session_id}/person.jpg"

def result_key(session_id: str) -> str:
    return f"sessions/{session_id}/result.png"

def garment_key(brand_id: str, product_id: str) -> str:
    return f"products/{brand_id}/{product_id}/garment.jpg"

def garment_mask_key(brand_id: str, product_id: str) -> str:
    return f"products/{brand_id}/{product_id}/mask.png"
