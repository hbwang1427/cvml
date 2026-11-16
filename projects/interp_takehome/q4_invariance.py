"""Q4 -- what did the training objective teach the model to ignore, and do its patch features know where
objects are?

    python q4_invariance.py invariance --n 200 --out results/invariance
    python q4_invariance.py patches    --n 200 --out results/invariance

invariance: apply a transformation T (flip, crop, grey-scale, hue shift, blur, rotation, patch shuffle)
  to each image and compare embeddings. Two measures:
    - normalised similarity  (cos(f(x), f(Tx)) - b) / (1 - b), where b is the mean cosine similarity
      between *different* images: 1 = fully invariant, 0 = as different as an unrelated image;
    - retrieval: is f(Tx) closer to f(x) than to the embeddings of all other original images?
patches: PCA of the last-layer patch features of each image. The first principal component, thresholded
  at 0, gives an unsupervised foreground mask (the DINO/DINOv2 visualisation). Compared with the union of
  COCO object masks by IoU, against a fixed centre-square baseline.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

from interp_utils import Encoder, load_coco, object_masks, open_image, square_crop

COLORS = {"resnet50": "#4f9e2a", "vit_sup": "#8b5cf6", "clip": "#2a78d6", "dinov2": "#eb6834",
          "clip_random": "#8a8984"}


# ---------------------------------------------------------------- transformations (on 224 x 224 crops)
def hue_shift(img, shift=0.33):
    h, s, v = img.convert("HSV").split()
    h = h.point(lambda p: (p + int(255 * shift)) % 256)
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def crop_half(img, rng):
    side = int(img.size[0] * 0.71)                                 # ~50% of the area, random position
    x, y = rng.integers(0, img.size[0] - side + 1, 2)
    return img.crop((x, y, x + side, y + side)).resize(img.size, Image.BICUBIC)


def patch_shuffle(img, rng, grid=4):
    a = np.asarray(img)
    s = a.shape[0] // grid
    tiles = [a[i * s:(i + 1) * s, j * s:(j + 1) * s] for i in range(grid) for j in range(grid)]
    order = rng.permutation(len(tiles))
    rows = [np.concatenate([tiles[order[i * grid + j]] for j in range(grid)], 1) for i in range(grid)]
    return Image.fromarray(np.concatenate(rows, 0))


TRANSFORMS = {
    "hflip": lambda im, rng: ImageOps.mirror(im),
    "crop50": crop_half,
    "grayscale": lambda im, rng: ImageOps.grayscale(im).convert("RGB"),
    "hue_shift": lambda im, rng: hue_shift(im),
    "blur": lambda im, rng: im.filter(ImageFilter.GaussianBlur(3)),
    "rotate90": lambda im, rng: im.rotate(90),
    "rotate180": lambda im, rng: im.rotate(180),
    "shuffle4x4": patch_shuffle,
}


def embed(enc, images, bs=32):
    z = torch.cat([enc.embed(enc.pixels(images[i:i + bs])).float().cpu() for i in range(0, len(images), bs)])
    return F.normalize(z, dim=-1)


def run_invariance(args, recs, out):
    images = [square_crop(open_image(r)) for r in recs]
    res = {}
    for name in args.models:
        enc = Encoder(name)
        z0 = embed(enc, images)
        sim = z0 @ z0.T
        b = float(sim[~torch.eye(len(z0), dtype=bool)].mean())
        res[name] = {"baseline_cos": b}
        for t, fn in TRANSFORMS.items():
            rng = np.random.default_rng(0)
            zt = embed(enc, [fn(im, rng) for im in images])
            cos = (zt * z0).sum(1)
            score = ((cos - b) / (1 - b)).numpy()
            top1 = ((zt @ z0.T).argmax(1) == torch.arange(len(z0))).float().numpy()
            res[name][t] = {"score": float(score.mean()), "score_se": float(score.std() / np.sqrt(len(score))),
                            "retrieval_top1": float(top1.mean())}
        print(name, {t: round(v["score"], 2) for t, v in res[name].items() if t != "baseline_cos"})
        del enc
    json.dump(res, open(out / "invariance.json", "w"), indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    ts = list(TRANSFORMS)
    w = 0.8 / len(res)
    for j, (name, r) in enumerate(res.items()):
        x = np.arange(len(ts)) + j * w
        axes[0].bar(x, [r[t]["score"] for t in ts], w, yerr=[1.96 * r[t]["score_se"] for t in ts],
                    color=COLORS.get(name), label=name, error_kw={"lw": 0.6})
        axes[1].bar(x, [r[t]["retrieval_top1"] for t in ts], w, color=COLORS.get(name))
    for ax, lab in zip(axes, ["normalised similarity (1 = invariant)", "retrieval top-1 (finds its original)"]):
        ax.set_xticks(np.arange(len(ts)) + 0.4 - w / 2, ts, rotation=30, fontsize=8); ax.set_ylabel(lab)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Invariance of the final embedding to image transformations (n = {len(images)})", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "invariance.png", dpi=140)
    print("wrote", out / "invariance.png")


def run_patches(args, recs, out):
    images = [square_crop(open_image(r)) for r in recs]
    gt_full = []
    for r in recs:
        m = object_masks(r)
        gt_full.append(np.any(list(m.values()), 0) if m else np.zeros((224, 224), bool))
    res, rgb = {}, {}
    for name in args.models:
        enc = Encoder(name)
        P = torch.cat([enc.spatial(enc.pixels(images[i:i + 32])).float().cpu() for i in range(0, len(images), 32)])
        n, g, _, d = P.shape
        # PCA *within each image*: the components separate regions of that image. (A PCA across all
        # images mostly separates images from each other -- try it.)
        flat = P.reshape(n, g * g, d)
        flat = flat - flat.mean(1, keepdim=True)
        _, _, V = torch.linalg.svd(flat, full_matrices=False)          # V: [n, d', d]
        pcs = torch.einsum("npd,nkd->npk", flat, V[:, :3]).reshape(n, g, g, 3)
        pc1 = pcs[..., 0].clone()
        border = torch.cat([pc1[:, 0], pc1[:, -1], pc1[:, :, 0], pc1[:, :, -1]], 1).mean(1)
        pc1 = pc1 * torch.where(border > 0, -1.0, 1.0)[:, None, None]  # sign: the border is "background"
        fg = (pc1 > 0).numpy()
        gt = np.stack([F.interpolate(torch.from_numpy(m.astype(np.float32))[None, None], size=(g, g),
                                     mode="area")[0, 0].numpy() > 0.5 for m in gt_full])
        centre = np.zeros((g, g), bool)
        lo, hi = round(g * 0.146), round(g * 0.854)                   # centred square, ~50% of the area
        centre[lo:hi, lo:hi] = True
        iou = lambda a, b: (a & b).sum((1, 2)) / np.maximum((a | b).sum((1, 2)), 1)   # noqa: E731
        res[name] = {"grid": g, "iou_pc1": float(iou(fg, gt).mean()),
                     "iou_centre_square": float(iou(np.broadcast_to(centre, gt.shape), gt).mean())}
        print(name, res[name])
        q = pcs.reshape(n, -1, 3)
        lo3, hi3 = q.amin(1)[:, None, None], q.amax(1)[:, None, None]
        rgb[name] = ((pcs - lo3) / (hi3 - lo3 + 1e-9)).numpy()
        del enc
    json.dump(res, open(out / "patch_pca.json", "w"), indent=1)

    show = list(range(6))
    fig, axes = plt.subplots(len(show), len(res) + 1, figsize=(1.9 * (len(res) + 1), 1.9 * len(show)))
    for row, i in enumerate(show):
        axes[row, 0].imshow(images[i])
        for col, name in enumerate(res, 1):
            axes[row, col].imshow(rgb[name][i], interpolation="nearest")
            if row == 0:
                axes[row, col].set_title(f"{name}\nmIoU {res[name]['iou_pc1']:.2f}", fontsize=8)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_title("image", fontsize=8)
    fig.suptitle("First 3 principal components of each image's last-layer patch features, as RGB", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "patch_pca.png", dpi=120)
    print("wrote", out / "patch_pca.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["invariance", "patches"])
    ap.add_argument("--data", default="data/coco1k")
    ap.add_argument("--models", nargs="+", default=["resnet50", "vit_sup", "clip", "dinov2", "clip_random"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="results/invariance")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    recs = load_coco(args.data)["images"][: args.n]
    (run_invariance if args.what == "invariance" else run_patches)(args, recs, out)


if __name__ == "__main__":
    main()
