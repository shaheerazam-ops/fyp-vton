"""
API layer — the only thing brand websites talk to.

Endpoints:
  POST /v1/try-on           kick off a try-on session (returns session_id immediately)
  GET  /v1/sessions/{id}    poll for status + signed result URL once done

Heavy lifting (model inference) happens in a separate worker process
(see worker.py) that pulls jobs off a queue — this keeps the HTTP
layer fast and stateless regardless of model latency.
"""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

from infra.db import get_db, TryOnSession, Product
from infra.object_storage import ObjectStore, StorageConfig, person_key
from infra.queue import enqueue_job

app = FastAPI(title="VTON Platform API")
store = ObjectStore(StorageConfig(bucket="vton-platform-images"))


class TryOnResponse(BaseModel):
    session_id: str
    status: str


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    result_url: str | None = None
    model_used: str | None = None
    error_message: str | None = None


@app.post("/v1/try-on", response_model=TryOnResponse)
async def create_try_on(
    product_id: str,
    customer_id: str | None = None,
    model_variant: str = "pretrained",     # "pretrained" | "scratch" — routing choice
    person_photo: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    product = db.query(Product).filter_by(product_id=product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Unknown product_id")

    session_id = str(uuid.uuid4())

    # 1. Stream the upload straight into object storage
    image = Image.open(person_photo.file).convert("RGB")
    key = person_key(session_id)
    store.put_image(key, image)

    # 2. Create the DB row (status=pending)
    session = TryOnSession(
        session_id=session_id,
        brand_id=product.brand_id,
        customer_id=customer_id,
        product_id=product_id,
        person_image_key=key,
        model_used=model_variant,
        status="pending",
    )
    db.add(session)
    db.commit()

    # 3. Hand off to the GPU worker pool — API returns immediately
    enqueue_job({"session_id": session_id})

    return TryOnResponse(session_id=session_id, status="pending")


@app.get("/v1/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(TryOnSession).filter_by(session_id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    result_url = store.signed_url(session.result_image_key) if session.result_image_key else None

    return SessionStatusResponse(
        session_id=session_id,
        status=session.status,
        result_url=result_url,
        model_used=session.model_used,
        error_message=session.error_message,
    )
