"""Compatibility entry point for the FiT-Layer high-quality backend.

Historically this class wrapped generic Stable Diffusion inpainting. That has
been removed: the production pretrained path now delegates to the official
IDM-VTON architecture through models.idm_vton.IDMVTonModel.
"""
from .idm_vton import IDMVTonModel

class PretrainedVTONModel(IDMVTonModel):
    name = "pretrained"
