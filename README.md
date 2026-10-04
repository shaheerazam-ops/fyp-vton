# FiT-Layer VTON Platform

A B2B virtual try-on backend organized around **FiT-Layer** product data and a
swappable inference layer.

## What changed in this version

The old pretrained path used `AutoPipelineForInpainting` plus a generic
IP-Adapter. That is **not the official IDM-VTON architecture**. The pretrained
path is now an adapter around the official IDM-VTON implementation: custom
try-on UNet + garment/reference UNet + CLIP image encoder + dual text
conditioning + OpenPose + SCHP parsing + DensePose.

The official IDM-VTON inference target is 768x1024, 30 steps and guidance 2.0;
its local demo also uses OpenPose/SCHP masking and DensePose. See the official
implementation for those details.

### FiT-Layer product flow

```text
Shopper photo
    -> EXIF correction + 3:4 crop
    -> OpenPose + SCHP + DensePose
    -> category-specific agnostic mask
    -> IDM-VTON
    -> paste generated crop back into original photo
    -> result

Brand product
    -> category + SKU + description
    -> garment normalization
    -> IDM-VTON garment conditioning
```

## Important licensing note

The official IDM-VTON code **and checkpoints** are released under
CC BY-NC-SA 4.0. That is suitable for experimentation subject to the license,
but you should not assume it is commercially cleared for a paid FiT-Layer SaaS.
For a commercial launch, obtain the necessary rights or replace the backend
with a model whose license permits your intended use.

## 1. Fastest test on your Windows PC

This tests the complete image-processing/model interface without Postgres,
Redis, S3 or a GPU.

```powershell
cd vton_platform
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Install PyTorch using the official command appropriate to your machine.
# PyTorch 2.6+ supports Python 3.13; the old project pin (2.4.1) was removed.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

pytest -q
```

You should see the CPU smoke tests pass.

### Local API test

```powershell
uvicorn api.local_demo:app --reload --port 8000
```

Open:

`http://127.0.0.1:8000/docs`

Upload a person image and garment image to `POST /v1/try-on`.

**The demo backend is intentionally not presented as real VTON quality.** It
only validates FiT-Layer's API contract and image plumbing on CPU.

### Direct CLI test

```powershell
python scripts/try_on.py `
  --backend demo `
  --person data/person.jpg `
  --garment data/shirt.jpg `
  --category top `
  --description "black oversized cotton t-shirt"
```

Result: `outputs/tryon_result.png`.

## 2. High-quality IDM-VTON test

The high-quality backend is a real IDM-VTON integration, not the previous
generic inpainting approximation.

The official repository requires the custom `src` pipeline and preprocessing
assets. Run:

```powershell
cd vton_platform
.\scripts\setup_idm_vton.ps1
```

Then install the dependencies required by the official IDM-VTON preprocessing
stack in a supported CUDA/NVIDIA environment. The setup script downloads the
DensePose, human-parsing and OpenPose assets; the ~29 GB model repository is
loaded from Hugging Face on first inference rather than being copied into
this project archive.

Then:

```powershell
$env:VTON_DEVICE="cuda"
python scripts/try_on.py `
  --backend idm-vton `
  --person data/person.jpg `
  --garment data/shirt.jpg `
  --category top `
  --description "black oversized cotton t-shirt" `
  --seed 42
```

For a Linux/NVIDIA deployment, use the same Python package set plus the
official IDM-VTON preprocessing dependencies and run the shell setup script:

```bash
./scripts/setup_idm_vton.sh
```

### Hardware reality

The high-quality backend is CUDA-oriented. The current project is therefore
split into:

- `demo` — CPU, zero-infrastructure integration testing.
- `pretrained` / `idm-vton` — real high-quality IDM-VTON inference on a
  supported CUDA/NVIDIA environment.
- `scratch` — research path, only enabled when trained weights exist.

Do not expect the 29 GB IDM-VTON checkpoint to run on an 8 GB GPU without a
major memory/quantization/offload strategy, and do not use the CPU demo as a
quality benchmark.

## 3. Production API architecture

```text
Brand storefront
      |
      v
 FastAPI API
      |
      +---- Postgres (brand/product/session metadata)
      +---- S3/R2 (person, garment, result images)
      +---- Redis queue
                    |
                    v
              GPU worker
                    |
          +---------+---------+
          |                   |
      IDM-VTON            scratch model
          |
   FiT-Layer preprocessing
          |
      result + metrics
```

The original API/worker/database/object-storage layers remain in place. The
new direct local API exists solely so model work can be tested without setting
up the whole SaaS infrastructure first.

## Project layout

```text
vton_platform/
├── api/
│   ├── main.py                 # production API
│   ├── local_demo.py           # zero-infrastructure CPU test API
│   └── worker.py               # GPU worker
├── models/
│   ├── base.py                 # FiT-Layer request/output contract
│   ├── preprocessing.py        # framing, category mapping, dev parser
│   ├── idm_vton.py              # real IDM-VTON adapter
│   ├── pretrained_model.py     # compatibility alias -> IDM-VTON
│   ├── demo_model.py           # CPU integration-test backend
│   └── scratch_model.py        # research model
├── scripts/
│   ├── setup_idm_vton.ps1
│   ├── setup_idm_vton.sh
│   └── try_on.py
├── infra/
├── training/
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```
