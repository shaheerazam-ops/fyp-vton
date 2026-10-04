"""Shared FiT-Layer VTON interfaces."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from PIL import Image

SUPPORTED_CATEGORIES = {
    "top": "upper_body", "shirt": "upper_body", "tshirt": "upper_body",
    "blouse": "upper_body", "jacket": "upper_body", "coat": "upper_body",
    "outerwear": "upper_body", "bottom": "lower_body", "pants": "lower_body",
    "jeans": "lower_body", "trousers": "lower_body", "skirt": "lower_body",
    "dress": "dresses", "gown": "dresses", "full_body": "dresses",
}

def normalize_category(category: str) -> str:
    key = category.strip().lower()
    if key not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Unsupported garment category '{category}'. Supported values: {', '.join(sorted(SUPPORTED_CATEGORIES))}")
    return SUPPORTED_CATEGORIES[key]

@dataclass
class VTONInput:
    person_image: Image.Image
    garment_image: Image.Image
    garment_category: str
    garment_description: str = "clothing"
    product_id: str | None = None
    seed: int = 42

@dataclass
class VTONOutput:
    result_image: Image.Image
    model_name: str
    latency_ms: float
    seed: int = 42
    quality_flags: list[str] = field(default_factory=list)

class BaseVTONModel(ABC):
    name: str = "base"
    @abstractmethod
    def load(self) -> None: raise NotImplementedError
    @abstractmethod
    def run(self, data: VTONInput) -> VTONOutput: raise NotImplementedError
