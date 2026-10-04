"""CPU-only integration-test backend.

This is NOT a quality VTON model. It exists so the complete FiT-Layer API,
queue, storage and frontend contract can be tested without a GPU/checkpoint.
It overlays the normalized garment in the category region while preserving
face/background pixels.
"""
from __future__ import annotations
import time
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from .base import BaseVTONModel, VTONInput, VTONOutput, normalize_category
from .preprocessing import normalize_rgb, prepare_person, prepare_garment

class DemoVTONModel(BaseVTONModel):
    name = "demo"
    def load(self): pass
    def run(self, data: VTONInput) -> VTONOutput:
        t0=time.time()
        category=normalize_category(data.garment_category)
        person,crop=prepare_person(normalize_rgb(data.person_image),(768,1024))
        garment=prepare_garment(data.garment_image,(768,1024))
        arr=np.array(person).astype(np.float32)
        g=np.array(garment).astype(np.float32)
        h,w=arr.shape[:2]
        if category=="upper_body":
            y0,y1=int(.25*h),int(.62*h); x0,x1=int(.22*w),int(.78*w)
        elif category=="lower_body":
            y0,y1=int(.50*h),int(.94*h); x0,x1=int(.25*w),int(.75*w)
        else:
            y0,y1=int(.20*h),int(.92*h); x0,x1=int(.18*w),int(.82*w)
        region=np.zeros((h,w),dtype=np.float32); region[y0:y1,x0:x1]=1.0
        region=Image.fromarray((region*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(18))
        alpha=np.array(region).astype(np.float32)/255.0
        # Keep the effect intentionally subtle so this is clearly a plumbing test.
        alpha*=0.72
        g=g*(0.82)+np.array(person).astype(np.float32)*0.18
        out=arr*(1-alpha[...,None])+g*alpha[...,None]
        result=Image.fromarray(np.clip(out,0,255).astype(np.uint8)).resize((crop[2]-crop[0],crop[3]-crop[1]),Image.Resampling.LANCZOS)
        original=normalize_rgb(data.person_image)
        original.paste(result,(crop[0],crop[1]))
        return VTONOutput(original,self.name,(time.time()-t0)*1000,data.seed,["demo_backend_not_real_vton"])
