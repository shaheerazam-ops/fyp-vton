"""Direct local FiT-Layer test runner; no Postgres/Redis/S3 required.

CPU integration test:
  python scripts/try_on.py --backend demo --person data/person.jpg --garment data/shirt.jpg --category top

High-quality backend (CUDA/NVIDIA):
  python scripts/try_on.py --backend idm-vton --person ... --garment ... --category top --description "black oversized cotton t-shirt"
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from PIL import Image
from models.base import VTONInput
from models.demo_model import DemoVTONModel
from models.idm_vton import IDMVTONModel

p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['demo','idm-vton'],default='demo')
p.add_argument('--person',required=True); p.add_argument('--garment',required=True)
p.add_argument('--category',required=True); p.add_argument('--description',default='clothing')
p.add_argument('--output',default='outputs/tryon_result.png'); p.add_argument('--seed',type=int,default=42)
args=p.parse_args()
if args.backend=='demo': model=DemoVTONModel()
else: model=IDMVTONModel(source_dir='third_party/IDM-VTON')
model.load()
out=model.run(VTONInput(Image.open(args.person),Image.open(args.garment),args.category,args.description,seed=args.seed))
Path(args.output).parent.mkdir(parents=True,exist_ok=True); out.result_image.save(args.output)
print(f'backend={out.model_name} latency_ms={out.latency_ms:.0f} seed={out.seed}')
print(f'output={Path(args.output).resolve()}')
if out.quality_flags: print('flags=',','.join(out.quality_flags))
