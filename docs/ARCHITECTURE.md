# FiT-Layer VTON Architecture

## Model path

```text
                FiT-Layer request
                       |
             +---------+---------+
             |                   |
        Shopper photo       Product garment
             |                   |
       EXIF correction      SKU/category
       3:4 crop/resize      description
             |                   |
      +------+-------+           |
      |      |       |           |
   OpenPose SCHP DensePose       |
      |      |       |           |
      +------+-------+           |
             |                   |
       category mask             |
             |                   |
             +---------+---------+
                       |
                 IDM-VTON
                       |
            custom person UNet
            garment/reference UNet
            CLIP image conditioning
            dual text conditioning
                       |
                 generated crop
                       |
              paste into original
                       |
                  result image
```

The official IDM-VTON implementation uses a custom try-on pipeline rather
than a plain `AutoPipelineForInpainting` call. It loads a person UNet, garment
UNet, image encoder, dual text encoders and tokenizers, then conditions the
pipeline with DensePose, an inpaint/agnostic mask, garment pixels and image
embeddings.

## FiT-Layer product abstraction

`VTONInput` contains:

- `person_image`
- `garment_image`
- `garment_category`
- `garment_description`
- `product_id`
- `seed`

This lets the model layer remain independent from the ecommerce database while
still receiving the information needed for faithful garment conditioning.

## Backends

| Backend | Purpose | GPU required | Quality |
|---|---|---:|---|
| `demo` | API/integration testing | No | Not a real VTON model |
| `pretrained` | Production/R&D IDM-VTON path | CUDA/NVIDIA | High-quality research backbone |
| `idm-vton` | Explicit alias of pretrained | CUDA/NVIDIA | Same |
| `scratch` | Future proprietary research model | Depends | Requires trained weights |

## Infrastructure

The existing production layers remain:

- FastAPI for HTTP
- PostgreSQL for product/session metadata
- S3-compatible storage for image bytes
- Redis for GPU job dispatch
- GPU workers for inference

A separate `api/local_demo.py` exists so model development does not require
running the whole SaaS infrastructure.

## Commercial licensing

The official IDM-VTON repository and checkpoints state CC BY-NC-SA 4.0.
Treat the current backend as an experimentation/FYP backend unless the
necessary commercial rights are obtained. The adapter is intentionally
separated so a commercially licensed VTON model can replace it later without
changing the FiT-Layer API contract.
