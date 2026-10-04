"""FiT-Layer image preprocessing and legacy/dev segmentation utilities."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from PIL import Image, ImageOps, ImageFilter

CLOTHES_LABELS = {0:"background",1:"hat",2:"hair",3:"sunglasses",4:"upper_clothes",5:"skirt",6:"pants",7:"dress",8:"belt",9:"left_shoe",10:"right_shoe",11:"face",12:"left_leg",13:"right_leg",14:"left_arm",15:"right_arm",16:"bag",17:"scarf"}
LABEL_IDS = {v:k for k,v in CLOTHES_LABELS.items()}
CATEGORY_TO_LABELS = {
    "top":["upper_clothes"], "outerwear":["upper_clothes","left_arm","right_arm"],
    "bottom":["pants","skirt"], "dress":["dress"],
    "full_body":["upper_clothes","dress","pants","skirt","left_arm","right_arm"],
}

def clothing_mask_for_category(parse_map: np.ndarray, category: str) -> np.ndarray:
    ids=[LABEL_IDS[x] for x in CATEGORY_TO_LABELS.get(category,["upper_clothes"]) if x in LABEL_IDS]
    return np.isin(parse_map,ids).astype(np.uint8)

@dataclass
class PersonFeatures:
    image: Image.Image; body_mask: np.ndarray; parse_map: np.ndarray
    pose_keypoints: np.ndarray; torso_bbox: tuple[int,int,int,int]

@dataclass
class GarmentFeatures:
    image: Image.Image; mask: np.ndarray

def normalize_rgb(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")

def center_crop_3_4(image: Image.Image):
    image=normalize_rgb(image); w,h=image.size
    tw=min(w,int(round(h*3/4))); th=min(h,int(round(w*4/3)))
    left=(w-tw)//2; top=(h-th)//2; box=(left,top,left+tw,top+th)
    return image.crop(box), box

def prepare_person(image: Image.Image, size=(768,1024)):
    cropped,box=center_crop_3_4(image)
    return cropped.resize(size,Image.Resampling.LANCZOS),box

def prepare_garment(image: Image.Image, size=(768,1024)) -> Image.Image:
    image=normalize_rgb(image)
    canvas=Image.new("RGB",size,(255,255,255))
    image.thumbnail((size[0]-32,size[1]-32),Image.Resampling.LANCZOS)
    canvas.paste(image,((size[0]-image.width)//2,(size[1]-image.height)//2))
    return canvas

class HumanParsingModel:
    def __init__(self,checkpoint="mattmdjaga/segformer_b2_clothes",device="cpu"):
        self.checkpoint=checkpoint; self.device=device; self._processor=None; self._model=None
    def load(self):
        from transformers import AutoImageProcessor,SegformerForSemanticSegmentation
        self._processor=AutoImageProcessor.from_pretrained(self.checkpoint)
        self._model=SegformerForSemanticSegmentation.from_pretrained(self.checkpoint).to(self.device); self._model.eval()
    def __call__(self,image):
        import torch
        inputs=self._processor(images=image,return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits=self._model(**inputs).logits
            logits=torch.nn.functional.interpolate(logits,size=image.size[::-1],mode="bilinear",align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

class Preprocessor:
    def __init__(self,human_parsing_model:Optional[HumanParsingModel]=None,pose_model=None,garment_seg_model=None):
        self._parser=human_parsing_model; self._pose=pose_model; self._garment_seg=garment_seg_model
    def load(self):
        if self._parser: self._parser.load()
    def process_person(self,image):
        image=normalize_rgb(image); parse=self._parser(image) if self._parser else _stub_parse_map(image)
        body=(parse>0).astype(np.uint8); pose=self._pose(image) if self._pose else _stub_keypoints()
        return PersonFeatures(image,body,parse,pose,_bbox_from_mask(body))
    def process_garment(self,image):
        image=normalize_rgb(image); mask=self._garment_seg(image) if self._garment_seg else _stub_garment_mask(image)
        return GarmentFeatures(image,mask)

def build_soft_mask(parse_map,category,dilation=8,blur=3.0):
    from scipy.ndimage import binary_dilation
    mask=clothing_mask_for_category(parse_map,category).astype(bool)
    if dilation: mask=binary_dilation(mask,iterations=dilation)
    return Image.fromarray((mask.astype(np.uint8)*255),mode="L").filter(ImageFilter.GaussianBlur(blur))

def _stub_garment_mask(image):
    arr=np.asarray(image.convert("RGB")); fg=np.any(arr<245,axis=2).astype(np.uint8)
    if fg.mean()<0.01:
        h,w=fg.shape; fg[h//8:7*h//8,w//8:7*w//8]=1
    return fg

def _stub_parse_map(image):
    w,h=image.size; return np.ones((h,w),dtype=np.uint8)

def _stub_keypoints(): return np.zeros((18,3),dtype=np.float32)

def _bbox_from_mask(mask):
    ys,xs=np.where(mask>0)
    if len(xs)==0:
        h,w=mask.shape; return (0,0,w,h)
    return (int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()))
