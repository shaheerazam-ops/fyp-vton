"""Zero-infrastructure local FiT-Layer test API.

Run:
    uvicorn api.local_demo:app --reload --port 8000

Then open http://127.0.0.1:8000/docs.
This endpoint uses the CPU demo backend, so it verifies the request/response
contract without Postgres, Redis, S3, or a GPU.
"""
from __future__ import annotations
from io import BytesIO
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response
from PIL import Image
from models.base import VTONInput
from models.demo_model import DemoVTONModel

app=FastAPI(title="FiT-Layer Local Test API",version="2.0")
model=DemoVTONModel(); model.load()

@app.get('/health')
def health(): return {'status':'ok','backend':model.name}

@app.post('/v1/try-on')
async def try_on(
    person_photo: UploadFile=File(...),
    garment_photo: UploadFile=File(...),
    category: str=Form('top'),
    description: str=Form('clothing'),
    seed: int=Form(42),
):
    person=Image.open(BytesIO(await person_photo.read())).convert('RGB')
    garment=Image.open(BytesIO(await garment_photo.read())).convert('RGB')
    result=model.run(VTONInput(person,garment,category,description,seed=seed))
    buf=BytesIO(); result.result_image.save(buf,format='PNG')
    return Response(content=buf.getvalue(),media_type='image/png',headers={
        'X-VTON-Backend': result.model_name,
        'X-VTON-Latency-Ms': str(round(result.latency_ms,1)),
    })
