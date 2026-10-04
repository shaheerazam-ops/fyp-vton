"""
Training script for the from-scratch VTON model (Path B).

Two-stage training, matching the two modules in models/scratch_model.py:
  1. Train the GMM (warping) module alone, supervised by an L1 loss
     against the ground-truth garment region.
  2. Freeze/fine-tune the GMM and train the try-on Generator +
     PatchGAN Discriminator adversarially, with L1 + perceptual +
     adversarial losses.

Expects a paired dataset:
  data/
    person/{id}.jpg          — shopper-style photo, garment already worn
    garment/{id}.jpg          — flat-lay of that same garment
    person_agnostic/{id}.jpg  — same person photo with the garment region masked out
    parse/{id}.png            — body part segmentation map
    pose/{id}.json             — 18 keypoints

This is exactly the shape of dataset that VITON-HD / DressCode
provide, and the shape you'd construct from a brand's own studio
photography (product shot + the same garment worn by a model).
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg19

from models.scratch_model import GMM, TryOnGenerator, PatchDiscriminator


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class VTONPairDataset(Dataset):
    def __init__(self, root: str, img_h: int = 256, img_w: int = 192):
        self.root = root
        self.ids = [f.split(".")[0] for f in os.listdir(os.path.join(root, "person"))]
        self.tf = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        from PIL import Image
        i = self.ids[idx]
        person = self.tf(Image.open(f"{self.root}/person/{i}.jpg").convert("RGB"))
        garment = self.tf(Image.open(f"{self.root}/garment/{i}.jpg").convert("RGB"))
        agnostic = self.tf(Image.open(f"{self.root}/person_agnostic/{i}.jpg").convert("RGB"))
        # person_repr: in production, concat pose heatmaps + parse map + agnostic image (22ch).
        # Simplified here to the agnostic RGB image repeated as a placeholder channel stack.
        person_repr = torch.cat([agnostic, agnostic[:1].repeat(19, 1, 1)], dim=0)  # -> 22ch
        return {"person_repr": person_repr, "garment": garment, "ground_truth": person}


# ---------------------------------------------------------------------
# Perceptual loss (VGG feature matching — standard for image synthesis)
# ---------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        vgg = vgg19(weights="IMAGENET1K_V1").features[:16].to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.l1(self.vgg(pred), self.vgg(target))


# ---------------------------------------------------------------------
# Stage 1 — train GMM
# ---------------------------------------------------------------------

def train_gmm(loader, device, epochs, lr, out_path):
    gmm = GMM().to(device)
    opt = torch.optim.Adam(gmm.parameters(), lr=lr, betas=(0.5, 0.999))
    l1 = nn.L1Loss()

    for epoch in range(epochs):
        running = 0.0
        for batch in loader:
            person_repr = batch["person_repr"].to(device)
            garment = batch["garment"].to(device)
            ground_truth = batch["ground_truth"].to(device)

            warped, _grid = gmm(person_repr, garment)
            loss = l1(warped, ground_truth)   # supervised: warped garment should match the worn region

            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()

        print(f"[GMM] epoch {epoch+1}/{epochs}  loss={running/len(loader):.4f}")

    torch.save(gmm.state_dict(), out_path)
    return gmm


# ---------------------------------------------------------------------
# Stage 2 — adversarial training of Generator + Discriminator
# ---------------------------------------------------------------------

def train_generator(loader, gmm, device, epochs, lr, l1_weight, perceptual_weight,
                     adv_weight, out_path):
    generator = TryOnGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    gmm.eval()
    for p in gmm.parameters():
        p.requires_grad = False

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    l1 = nn.L1Loss()
    perceptual = VGGPerceptualLoss(device)
    adv = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        running_g, running_d = 0.0, 0.0
        for batch in loader:
            person_repr = batch["person_repr"].to(device)
            garment = batch["garment"].to(device)
            real = batch["ground_truth"].to(device)

            with torch.no_grad():
                warped_garment, _ = gmm(person_repr, garment)

            fake, _mask = generator(person_repr, warped_garment)

            # --- Discriminator step ---
            opt_d.zero_grad()
            real_pred = discriminator(torch.cat([person_repr[:, :3], real], dim=1))
            fake_pred = discriminator(torch.cat([person_repr[:, :3], fake.detach()], dim=1))
            d_loss = 0.5 * (
                adv(real_pred, torch.ones_like(real_pred))
                + adv(fake_pred, torch.zeros_like(fake_pred))
            )
            d_loss.backward(); opt_d.step()

            # --- Generator step ---
            opt_g.zero_grad()
            fake_pred_for_g = discriminator(torch.cat([person_repr[:, :3], fake], dim=1))
            g_adv = adv(fake_pred_for_g, torch.ones_like(fake_pred_for_g))
            g_l1 = l1(fake, real)
            g_perceptual = perceptual(fake, real)
            g_loss = l1_weight * g_l1 + perceptual_weight * g_perceptual + adv_weight * g_adv
            g_loss.backward(); opt_g.step()

            running_g += g_loss.item()
            running_d += d_loss.item()

        print(f"[GEN] epoch {epoch+1}/{epochs}  g_loss={running_g/len(loader):.4f}  "
              f"d_loss={running_d/len(loader):.4f}")

    torch.save({"gmm": gmm.state_dict(), "generator": generator.state_dict()}, out_path)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gmm-epochs", type=int, default=50)
    ap.add_argument("--gen-epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--l1-weight", type=float, default=10.0)
    ap.add_argument("--perceptual-weight", type=float, default=5.0)
    ap.add_argument("--adv-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dataset = VTONPairDataset(args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    gmm_path = os.path.join(args.out_dir, "gmm.pt")
    gmm = train_gmm(loader, args.device, args.gmm_epochs, args.lr, gmm_path)

    final_path = os.path.join(args.out_dir, "scratch_vton.pt")
    train_generator(loader, gmm, args.device, args.gen_epochs, args.lr,
                     args.l1_weight, args.perceptual_weight, args.adv_weight, final_path)

    print(f"Done. Final combined checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
