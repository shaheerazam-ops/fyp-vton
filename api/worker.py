"""GPU worker with configurable FiT-Layer model backend."""
from __future__ import annotations
import os
import time
import traceback
from infra.db import SessionLocal, TryOnSession, Product
from infra.object_storage import ObjectStore, StorageConfig, garment_key, result_key
from infra.queue import consume_jobs
from models.base import VTONInput
from models.pretrained_model import PretrainedVTONModel
from models.scratch_model import ScratchVTONModel
from models.demo_model import DemoVTONModel

store = ObjectStore(StorageConfig(bucket=os.environ.get("STORAGE_BUCKET", "vton-platform-images")))

def _device():
    return os.environ.get("VTON_DEVICE", "cuda")

MODELS = {
    "pretrained": PretrainedVTONModel(
        model_id=os.environ.get("IDM_VTON_MODEL", "yisol/IDM-VTON"),
        source_dir=os.environ.get("IDM_VTON_SOURCE", "third_party/IDM-VTON"),
        device=_device(),
        num_inference_steps=int(os.environ.get("VTON_STEPS", "30")),
        guidance_scale=float(os.environ.get("VTON_GUIDANCE", "2.0")),
        cpu_offload=os.environ.get("IDM_VTON_CPU_OFFLOAD", "0") == "1",
    ),
    "idm-vton": PretrainedVTONModel(
        model_id=os.environ.get("IDM_VTON_MODEL", "yisol/IDM-VTON"),
        source_dir=os.environ.get("IDM_VTON_SOURCE", "third_party/IDM-VTON"),
        device=_device(),
    ),
    "demo": DemoVTONModel(),
}

scratch_path=os.environ.get("SCRATCH_WEIGHTS", "checkpoints/scratch_vton.pt")
if os.path.exists(scratch_path):
    MODELS["scratch"] = ScratchVTONModel(weights_path=scratch_path, device=_device())


def load_models():
    selected=os.environ.get("VTON_MODEL", "pretrained")
    if selected not in MODELS:
        raise ValueError(f"Unknown VTON_MODEL={selected}. Available: {', '.join(MODELS)}")
    print(f"Loading model: {selected}")
    MODELS[selected].load()


def process_job(job: dict):
    session_id=job["session_id"]; db=SessionLocal(); session=None
    try:
        session=db.query(TryOnSession).filter_by(session_id=session_id).first()
        if not session: raise ValueError(f"Unknown session {session_id}")
        session.status="processing"; db.commit()
        product=db.query(Product).filter_by(product_id=session.product_id).first()
        if not product: raise ValueError(f"Unknown product {session.product_id}")
        person_image=store.get_image(session.person_image_key)
        garment_image=store.get_image(garment_key(str(product.brand_id),str(product.product_id)))
        model=MODELS[session.model_used]
        t0=time.time()
        output=model.run(VTONInput(
            person_image=person_image,
            garment_image=garment_image,
            garment_category=product.category,
            garment_description=product.name,
            product_id=str(product.product_id),
            seed=int(job.get("seed",42)),
        ))
        out_key=result_key(session_id); store.put_image(out_key,output.result_image,format="PNG")
        session.result_image_key=out_key; session.status="done"
        session.latency_ms=int((time.time()-t0)*1000); db.commit()
    except Exception as exc:
        db.rollback()
        if session:
            session.status="failed"; session.error_message=str(exc); db.commit()
        traceback.print_exc()
    finally: db.close()


def main():
    load_models(); print("Worker ready, waiting for jobs...")
    for job in consume_jobs(): process_job(job)

if __name__ == "__main__": main()
