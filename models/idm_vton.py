"""
High-quality FiT-Layer backend for the official IDM-VTON architecture.

This replaces the old generic AutoPipelineForInpainting wrapper. IDM-VTON's
official implementation uses a modified SDXL inpaint UNet, garment/reference
UNet, CLIP image encoder, dual text encoders, DensePose, OpenPose and SCHP
human parsing. The official inference target is 768x1024.

The official IDM-VTON source repository is intentionally NOT vendored here;
it is licensed CC BY-NC-SA and should be obtained separately. Run
scripts/setup_idm_vton.ps1 (Windows) or scripts/setup_idm_vton.sh (Linux).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from .base import BaseVTONModel, VTONInput, VTONOutput, normalize_category
from .preprocessing import normalize_rgb, prepare_person, prepare_garment


class IDMVTONModel(BaseVTONModel):
    name = "idm-vton"

    def __init__(
        self,
        model_id: str = "yisol/IDM-VTON",
        source_dir: str = "third_party/IDM-VTON",
        device: str | None = None,
        width: int = 768,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 2.0,
        seed: int = 42,
        cpu_offload: bool = False,
    ):
        self.model_id = model_id
        self.source_dir = Path(source_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.default_seed = seed
        self.cpu_offload = cpu_offload
        self.pipe = None
        self.parsing_model = None
        self.openpose_model = None
        self._loaded = False

    def _add_official_source(self):
        source = self.source_dir.resolve()
        if not source.exists():
            raise FileNotFoundError(
                f"IDM-VTON source not found at {source}. "
                "Run scripts/setup_idm_vton.ps1 or scripts/setup_idm_vton.sh first."
            )
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        return source

    def load(self) -> None:
        if self._loaded:
            return
        if self.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "The high-quality IDM-VTON backend requires a CUDA-capable "
                "PyTorch environment. Use the local 'demo' backend for CPU/API "
                "integration tests, or run IDM-VTON on a supported NVIDIA/Linux GPU."
            )

        self._add_official_source()

        from diffusers import AutoencoderKL, DDPMScheduler
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from src.unet_hacked_garmnet import UNet2DConditionModel as GarmentUNet
        from src.unet_hacked_tryon import UNet2DConditionModel as TryOnUNet
        from transformers import (
            AutoTokenizer,
            CLIPImageProcessor,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPVisionModelWithProjection,
        )

        dtype = torch.float16
        print(f"[IDM-VTON] loading {self.model_id} on {self.device}")

        scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae", torch_dtype=dtype)
        unet = TryOnUNet.from_pretrained(self.model_id, subfolder="unet", torch_dtype=dtype)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.model_id, subfolder="image_encoder", torch_dtype=dtype
        )
        unet_encoder = GarmentUNet.from_pretrained(
            self.model_id, subfolder="unet_encoder", torch_dtype=dtype
        )
        text_encoder_one = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder", torch_dtype=dtype
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            self.model_id, subfolder="text_encoder_2", torch_dtype=dtype
        )
        tokenizer_one = AutoTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer", use_fast=False
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer_2", use_fast=False
        )

        for module in (unet, vae, image_encoder, unet_encoder, text_encoder_one, text_encoder_two):
            module.requires_grad_(False)

        self.pipe = TryonPipeline.from_pretrained(
            self.model_id,
            unet=unet,
            vae=vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            scheduler=scheduler,
            image_encoder=image_encoder,
            unet_encoder=unet_encoder,
            torch_dtype=dtype,
        )
        self.pipe.unet_encoder = unet_encoder

        if self.cpu_offload:
            self.pipe.enable_model_cpu_offload()
            self.pipe.unet_encoder.to("cuda")
        else:
            self.pipe = self.pipe.to(self.device)
            self.pipe.unet_encoder.to(self.device)

        # These are the official preprocessing components used by IDM-VTON's
        # local demo. DensePose is invoked through apply_net below.
        from preprocess.humanparsing.run_parsing import Parsing
        from preprocess.openpose.run_openpose import OpenPose

        self.parsing_model = Parsing(0)
        self.openpose_model = OpenPose(0)
        self._loaded = True
        print("[IDM-VTON] model + OpenPose + SCHP loaded")

    @staticmethod
    def _tensor_transform(image: Image.Image) -> torch.Tensor:
        from torchvision import transforms
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])(image)

    def _densepose(self, human_img: Image.Image) -> Image.Image:
        self._add_official_source()
        import apply_net
        from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

        bgr = _apply_exif_orientation(human_img.resize((384, 512)))
        bgr = convert_PIL_to_numpy(bgr, format="BGR")
        cfg = self.source_dir / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml"
        weights = self.source_dir / "ckpt" / "densepose" / "model_final_162be9.pkl"
        if not cfg.exists() or not weights.exists():
            raise FileNotFoundError(
                "DensePose assets are missing. Expected "
                f"{cfg} and {weights}. See scripts/setup_idm_vton.ps1."
            )
        args = apply_net.create_argument_parser().parse_args((
            "show", str(cfg), str(weights), "dp_segm", "-v", "--opts", "MODEL.DEVICE", "cuda"
        ))
        pose = args.func(args, bgr)
        pose = pose[:, :, ::-1]
        return Image.fromarray(pose).resize((self.width, self.height), Image.Resampling.BILINEAR)

    def _build_inputs(self, person: Image.Image, garment: Image.Image, category: str):
        from .base import normalize_category as _normalize
        idm_category = _normalize(category)

        human_img = prepare_person(person, (self.width, self.height))[0]
        garm_img = prepare_garment(garment, (self.width, self.height))

        # Official IDM-VTON auto mask: OpenPose + SCHP + get_mask_location.
        from utils_mask import get_mask_location
        small = human_img.resize((384, 512), Image.Resampling.BILINEAR)
        keypoints = self.openpose_model(small)
        model_parse, _ = self.parsing_model(small)
        mask, _mask_gray = get_mask_location("hd", idm_category, model_parse, keypoints)
        mask = mask.resize((self.width, self.height), Image.Resampling.NEAREST)

        pose_img = self._densepose(human_img)
        return human_img, garm_img, mask, pose_img

    @torch.inference_mode()
    def run(self, data: VTONInput) -> VTONOutput:
        if not self._loaded:
            self.load()
        t0 = time.time()
        category = normalize_category(data.garment_category)
        human_img, garm_img, mask, pose_img = self._build_inputs(
            normalize_rgb(data.person_image), normalize_rgb(data.garment_image), data.garment_category
        )

        prompt = "model is wearing " + (data.garment_description or category)
        garment_prompt = "a photo of " + (data.garment_description or category)
        negative = "monochrome, lowres, bad anatomy, worst quality, low quality, distorted garment"

        pe = self.pipe.encode_prompt(
            prompt, num_images_per_prompt=1, do_classifier_free_guidance=True,
            negative_prompt=negative,
        )
        prompt_embeds, negative_prompt_embeds, pooled, negative_pooled = pe
        cloth_pe = self.pipe.encode_prompt(
            garment_prompt, num_images_per_prompt=1, do_classifier_free_guidance=False,
            negative_prompt=negative,
        )[0]

        pose_tensor = self._tensor_transform(pose_img).unsqueeze(0).to(self.device, torch.float16)
        garment_tensor = self._tensor_transform(garm_img).unsqueeze(0).to(self.device, torch.float16)
        generator = torch.Generator(device=self.device).manual_seed(data.seed or self.default_seed)

        images = self.pipe(
            prompt_embeds=prompt_embeds.to(self.device, torch.float16),
            negative_prompt_embeds=negative_prompt_embeds.to(self.device, torch.float16),
            pooled_prompt_embeds=pooled.to(self.device, torch.float16),
            negative_pooled_prompt_embeds=negative_pooled.to(self.device, torch.float16),
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            strength=1.0,
            pose_img=pose_tensor,
            text_embeds_cloth=cloth_pe.to(self.device, torch.float16),
            cloth=garment_tensor,
            mask_image=mask,
            image=human_img,
            height=self.height,
            width=self.width,
            guidance_scale=self.guidance_scale,
            ip_adapter_image=garm_img,
        )[0]

        # Restore the generated crop into the user's original image so FiT-Layer
        # does not force every shopper photo into a permanently cropped 3:4 image.
        original = normalize_rgb(data.person_image)
        _, crop_box = prepare_person(original, (self.width, self.height))
        generated = images[0].resize((crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]), Image.Resampling.LANCZOS)
        original.paste(generated, (crop_box[0], crop_box[1]))

        return VTONOutput(
            result_image=original,
            model_name=self.name,
            latency_ms=(time.time() - t0) * 1000,
            seed=data.seed,
        )


# Backwards-compatible spelling for early FiT-Layer experiments.
IDMVTonModel = IDMVTONModel
