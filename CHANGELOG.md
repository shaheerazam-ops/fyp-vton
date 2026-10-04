# FiT-Layer VTON update

## 2026-10-03

- Replaced the previous generic SD inpainting + IP-Adapter implementation with a real IDM-VTON adapter path.
- Added official IDM-VTON preprocessing flow: OpenPose + SCHP + DensePose + category mask.
- Added 768x1024 model framing with aspect-ratio-preserving crop and paste-back.
- Added FiT-Layer product metadata: product id, category normalization, garment description, deterministic seed.
- Added a CPU `demo` backend for testing the API/model contract without GPU infrastructure.
- Added a zero-infrastructure FastAPI test server at `api/local_demo.py`.
- Added direct CLI inference at `scripts/try_on.py`.
- Added Windows/Linux setup scripts for the official IDM-VTON source and preprocessing checkpoints.
- Removed the old PyTorch 2.4.1 pin; the project now expects PyTorch 2.6+ for Python 3.13 environments.
- Added 5 CPU smoke tests and verified them locally.
- Added `.gitignore` protection for model checkpoints and third-party source.
