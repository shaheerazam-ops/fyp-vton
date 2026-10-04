#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d third_party/IDM-VTON ]; then
  git clone --depth 1 https://github.com/yisol/IDM-VTON.git third_party/IDM-VTON
fi
python -m pip install -U huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
root=Path('third_party/IDM-VTON/ckpt'); root.mkdir(parents=True,exist_ok=True)
snapshot_download(repo_id='yisol/IDM-VTON',local_dir=str(root),allow_patterns=[
'densepose/model_final_162be9.pkl','humanparsing/parsing_atr.onnx',
'humanparsing/parsing_lip.onnx','openpose/ckpts/body_pose_model.pth'])
print('Preprocessing checkpoints downloaded.')
PY
echo 'Setup complete. Use a CUDA/NVIDIA environment for IDM-VTON.'
