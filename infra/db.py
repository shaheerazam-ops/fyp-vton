"""SQLAlchemy models + session factory, mirroring infra/schema.sql."""

from __future__ import annotations

import os
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine, func
from sqlalchemy.dialects.postgresql import ENUM, JSONB, UUID
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/vton")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Brand(Base):
    __tablename__ = "brands"
    brand_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    api_key_hash = Column(String, nullable=False, unique=True)
    plan_tier = Column(String, nullable=False, default="trial")
    storage_bucket = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Customer(Base):
    __tablename__ = "customers"
    customer_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.brand_id"))
    external_ref = Column(String)
    email = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Product(Base):
    __tablename__ = "products"
    product_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.brand_id"))
    sku = Column(String, nullable=False)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    garment_image_key = Column(String, nullable=False)
    garment_mask_key = Column(String)
    size_chart = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class TryOnSession(Base):
    __tablename__ = "try_on_sessions"
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.brand_id"))
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.customer_id"), nullable=True)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.product_id"))

    person_image_key = Column(String, nullable=False)
    result_image_key = Column(String, nullable=True)

    model_used = Column(String, nullable=False, default="pretrained")
    status = Column(String, nullable=False, default="pending")
    error_message = Column(Text, nullable=True)

    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
