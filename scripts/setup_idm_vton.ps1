$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "== FiT-Layer IDM-VTON setup ==" -ForegroundColor Cyan

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required. Install Git for Windows first."
}

$target = Join-Path (Get-Location) "third_party\IDM-VTON"
if (-not (Test-Path $target)) {
    git clone --depth 1 https://github.com/yisol/IDM-VTON.git $target
} else {
    Write-Host "IDM-VTON source already exists: $target"
}

python -m pip install -U huggingface_hub
@'
from huggingface_hub import snapshot_download
from pathlib import Path
root=Path("third_party/IDM-VTON/ckpt")
root.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id="yisol/IDM-VTON",
    local_dir=str(root),
    allow_patterns=[
        "densepose/model_final_162be9.pkl",
        "humanparsing/parsing_atr.onnx",
        "humanparsing/parsing_lip.onnx",
        "openpose/ckpts/body_pose_model.pth",
    ],
)
print("Preprocessing checkpoints downloaded to", root)
'@ | python -

Write-Host "`nIMPORTANT:" -ForegroundColor Yellow
Write-Host "The high-quality IDM-VTON path needs a CUDA/NVIDIA environment." -ForegroundColor Yellow
Write-Host "Your AMD Windows GPU should use the demo backend locally or run IDM-VTON in a Linux/NVIDIA environment." -ForegroundColor Yellow
Write-Host "`nSource ready. Model weights will be downloaded by Hugging Face on first IDM-VTON load." -ForegroundColor Green
