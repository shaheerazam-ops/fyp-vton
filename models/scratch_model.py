"""
Path B — From-scratch model.

A purpose-built try-on network trained end-to-end on the brand's own
paired (person, garment, ground-truth try-on) data. Classic two-stage
VTON design (see VITON/CP-VTON lineage), reimplemented compactly:

  Stage 1 — GMM (Geometric Matching Module)
      Predicts a Thin-Plate-Spline warp that deforms the flat-lay
      garment image onto the target person's pose/body shape.

  Stage 2 — Try-On Generator (U-Net, GAN-trained)
      Takes the person representation (segmentation + pose) and the
      warped garment, and synthesizes the final composited image.
      A PatchGAN discriminator provides the adversarial signal.

This is the slower-to-ship, higher-ownership path: needs real training
data and compute, but once trained, inference is a single forward pass
(cheap, low latency) and the weights are fully your own IP.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base import BaseVTONModel, VTONInput, VTONOutput
from .preprocessing import Preprocessor


# =====================================================================
# Stage 1 — Geometric Matching Module (garment warping)
# =====================================================================

class FeatureExtractor(nn.Module):
    """Small conv tower shared by the person-repr and garment branches."""

    def __init__(self, in_channels: int, base_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1), nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1), nn.BatchNorm2d(base_ch * 4), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 4, 4, 2, 1), nn.BatchNorm2d(base_ch * 4), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TPSGridGen(nn.Module):
    """Thin-Plate-Spline grid generator from a set of predicted control points."""

    def __init__(self, out_h: int, out_w: int, grid_size: int = 5):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        self.grid_size = grid_size
        n = grid_size * grid_size

        # Fixed target control points on a regular grid in [-1, 1]
        r = torch.linspace(-1, 1, grid_size)
        gy, gx = torch.meshgrid(r, r, indexing="ij")
        target_points = torch.stack([gx.flatten(), gy.flatten()], dim=1)  # (n, 2)
        self.register_buffer("target_points", target_points)

        # Precompute the TPS kernel matrix for the fixed target points
        self.register_buffer("kernel_inv", self._build_kernel_inverse(target_points))

    def _build_kernel_inverse(self, pts: torch.Tensor) -> torch.Tensor:
        n = pts.shape[0]
        d = torch.cdist(pts, pts)
        K = d ** 2 * torch.log(d ** 2 + 1e-6)
        P = torch.cat([torch.ones(n, 1), pts], dim=1)
        L_top = torch.cat([K, P], dim=1)
        L_bot = torch.cat([P.t(), torch.zeros(3, 3)], dim=1)
        L = torch.cat([L_top, L_bot], dim=0)
        return torch.inverse(L + torch.eye(L.shape[0]) * 1e-6)

    def forward(self, source_points: torch.Tensor) -> torch.Tensor:
        """source_points: (B, n, 2) predicted control-point offsets -> sampling grid (B, H, W, 2)."""
        B, n, _ = source_points.shape
        zeros = torch.zeros(B, 3, 2, device=source_points.device)
        Y = torch.cat([source_points, zeros], dim=1)               # (B, n+3, 2)
        W = torch.einsum("ij,bjk->bik", self.kernel_inv, Y)        # (B, n+3, 2)

        r = torch.linspace(-1, 1, self.out_h, device=source_points.device)
        c = torch.linspace(-1, 1, self.out_w, device=source_points.device)
        gy, gx = torch.meshgrid(r, c, indexing="ij")
        query = torch.stack([gx.flatten(), gy.flatten()], dim=1)   # (H*W, 2)

        d = torch.cdist(query, self.target_points)                 # (H*W, n)
        U = d ** 2 * torch.log(d ** 2 + 1e-6)
        ones = torch.ones(query.shape[0], 1, device=query.device)
        Pq = torch.cat([U, ones, query], dim=1)                     # (H*W, n+3)

        grid = torch.einsum("qk,bkd->bqd", Pq, W)                  # (B, H*W, 2)
        return grid.view(B, self.out_h, self.out_w, 2)


class GMM(nn.Module):
    """Predicts a warp that deforms the garment onto the person's body/pose."""

    def __init__(self, person_repr_channels: int = 22, garment_channels: int = 3,
                 img_h: int = 256, img_w: int = 192, grid_size: int = 5):
        super().__init__()
        self.person_enc = FeatureExtractor(person_repr_channels)
        self.garment_enc = FeatureExtractor(garment_channels)
        feat_h, feat_w = img_h // 16, img_w // 16
        self.regressor = nn.Sequential(
            nn.Conv2d(128 * 2, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, grid_size * grid_size * 2),
        )
        self.tps = TPSGridGen(img_h, img_w, grid_size)

    def forward(self, person_repr: torch.Tensor, garment: torch.Tensor):
        pf = self.person_enc(person_repr)
        gf = self.garment_enc(garment)
        offsets = self.regressor(torch.cat([pf, gf], dim=1))
        offsets = offsets.view(-1, self.tps.grid_size ** 2, 2)
        grid = self.tps(offsets)
        warped_garment = F.grid_sample(garment, grid, align_corners=True, padding_mode="border")
        return warped_garment, grid


# =====================================================================
# Stage 2 — Try-On Generator (U-Net) + PatchGAN Discriminator
# =====================================================================

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class TryOnGenerator(nn.Module):
    """U-Net: [person representation, warped garment] -> composited try-on image."""

    def __init__(self, in_channels: int = 25, out_channels: int = 3, base_ch: int = 64):
        super().__init__()
        self.d1 = UNetBlock(in_channels, base_ch)
        self.d2 = UNetBlock(base_ch, base_ch * 2)
        self.d3 = UNetBlock(base_ch * 2, base_ch * 4)
        self.d4 = UNetBlock(base_ch * 4, base_ch * 8)
        self.d5 = UNetBlock(base_ch * 8, base_ch * 8)

        self.u1 = UNetBlock(base_ch * 8, base_ch * 8, down=False, use_dropout=True)
        self.u2 = UNetBlock(base_ch * 16, base_ch * 4, down=False)
        self.u3 = UNetBlock(base_ch * 8, base_ch * 2, down=False)
        self.u4 = UNetBlock(base_ch * 4, base_ch, down=False)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )
        # Predicted composition mask: blends generated content with the warped garment directly
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, person_repr: torch.Tensor, warped_garment: torch.Tensor):
        x = torch.cat([person_repr, warped_garment], dim=1)
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3); d5 = self.d5(d4)
        u1 = self.u1(d5)
        u2 = self.u2(torch.cat([u1, d4], dim=1))
        u3 = self.u3(torch.cat([u2, d3], dim=1))
        u4 = self.u4(torch.cat([u3, d2], dim=1))
        feat = torch.cat([u4, d1], dim=1)

        rendered = self.out(feat)
        comp_mask = self.mask_head(feat)
        # Final image = learned blend of rendered content and the warped garment texture
        composited = comp_mask * warped_garment + (1 - comp_mask) * rendered
        return composited, comp_mask


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN — classifies overlapping patches as real/fake."""

    def __init__(self, in_channels: int = 6, base_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1), nn.BatchNorm2d(base_ch * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1), nn.BatchNorm2d(base_ch * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 1, 1), nn.BatchNorm2d(base_ch * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 8, 1, 4, 1, 1),
        )

    def forward(self, img_pair: torch.Tensor):
        return self.net(img_pair)


# =====================================================================
# Wrapper implementing the shared BaseVTONModel interface
# =====================================================================

class ScratchVTONModel(BaseVTONModel):
    name = "scratch"

    def __init__(self, weights_path: str, device: str = "cuda",
                 img_h: int = 256, img_w: int = 192):
        self.weights_path = weights_path
        self.device = device
        self.img_h, self.img_w = img_h, img_w
        self.gmm = None
        self.generator = None
        self.preprocessor = Preprocessor()

    def load(self) -> None:
        self.gmm = GMM(img_h=self.img_h, img_w=self.img_w).to(self.device)
        self.generator = TryOnGenerator().to(self.device)

        ckpt = torch.load(self.weights_path, map_location=self.device)
        self.gmm.load_state_dict(ckpt["gmm"])
        self.generator.load_state_dict(ckpt["generator"])
        self.gmm.eval()
        self.generator.eval()

        self.preprocessor.load()

    @torch.no_grad()
    def run(self, data: VTONInput) -> VTONOutput:
        t0 = time.time()

        person = self.preprocessor.process_person(data.person_image)
        garment = self.preprocessor.process_garment(data.garment_image)

        person_tensor = self._to_tensor(person.image)
        garment_tensor = self._to_tensor(garment.image)
        person_repr = self._build_person_representation(person)  # 22ch: pose + parse + agnostic person

        warped_garment, _grid = self.gmm(person_repr, garment_tensor)
        composited, _mask = self.generator(person_repr, warped_garment)

        result_image = self._to_pil(composited)
        latency_ms = (time.time() - t0) * 1000
        return VTONOutput(result_image=result_image, model_name=self.name, latency_ms=latency_ms)

    # ---- tensor <-> PIL helpers ----

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        import numpy as np
        img = image.resize((self.img_w, self.img_h))
        arr = np.asarray(img).astype("float32") / 127.5 - 1.0     # normalize to [-1, 1]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t.to(self.device)

    def _to_pil(self, tensor: torch.Tensor) -> Image.Image:
        import numpy as np
        arr = ((tensor.squeeze(0).permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
        return Image.fromarray(arr)

    def _build_person_representation(self, person) -> torch.Tensor:
        """
        Stack of pose heatmaps + body-part segmentation + a
        garment-agnostic person image, per the CP-VTON "person
        representation" recipe (22 channels total: 18 pose + 3 RGB
        agnostic person + 1 body-shape mask). Simplified here to the
        interface shape; production fills this from the real
        pose/parse outputs from `Preprocessor`.
        """
        import numpy as np
        h, w = self.img_h, self.img_w
        channels = np.zeros((22, h, w), dtype="float32")
        channels[0] = np.array(person.body_mask.astype("float32"))[:h, :w] if person.body_mask.shape[:2] == (h, w) else 0
        return torch.from_numpy(channels).unsqueeze(0).to(self.device)
