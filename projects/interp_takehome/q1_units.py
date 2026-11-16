"""Q1 -- what do individual units of a CNN detect, and how does that change with depth?

    python q1_units.py --model resnet50          --out results/units
    python q1_units.py --model resnet50_random   --out results/units      # control: same architecture, no training

Three classic "look inside" tools on a supervised ResNet-50:
  1. First-layer filters: the 64 conv1 kernels shown as 7 x 7 RGB images.
  2. Dataset examples (Zeiler & Fergus 2014): for a unit (one channel of a stage's output), the image
     regions where it fires most strongly, cropped around the peak activation.
  3. Network dissection, light version (Bau et al. 2017): binarise each unit's activation map at its
     top-1% level, upsample it, and measure IoU with every COCO object mask over the whole dataset. A unit
     whose best IoU is above --iou is called a detector for that object.

COCO only has *object* masks, so units that detect colours, textures or parts will look "unlabelled"
here. Keep that in mind when you compare stages.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from interp_utils import DEVICE, SIZE, Encoder, load_coco, object_masks, open_image, square_crop

STAGES = {"layer1": 3, "layer2": 7, "layer3": 13, "layer4": 16}     # index into Encoder.layers()
CROP = {"layer1": 48, "layer2": 80, "layer3": 128, "layer4": 224}    # crop size in pixels for examples
M = 56                                                                # mask resolution for dissection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/coco1k")
    ap.add_argument("--model", default="resnet50", choices=["resnet50", "resnet50_random"])
    ap.add_argument("--quantile", type=float, default=0.99, help="per-unit activation threshold")
    ap.add_argument("--iou", type=float, default=0.04, help="IoU above which a unit counts as a detector")
    ap.add_argument("--min-images", type=int, default=10, help="concepts need masks in this many images")
    ap.add_argument("--units-per-stage", type=int, default=4)
    ap.add_argument("--out", default="results/units")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    recs = load_coco(args.data)["images"]
    enc = Encoder(args.model)
    bs = 32

    # ---------------------------------------------------------------- 1. first-layer filters
    w = enc.model.conv1.weight.detach().cpu()                         # [64, 3, 7, 7]
    w = (w - w.amin((1, 2, 3), keepdim=True)) / (w.amax((1, 2, 3), keepdim=True) - w.amin((1, 2, 3), keepdim=True))
    fig, axes = plt.subplots(4, 16, figsize=(12, 3.2))
    for ax, f in zip(axes.ravel(), w):
        ax.imshow(f.permute(1, 2, 0).numpy(), interpolation="nearest"); ax.axis("off")
    fig.suptitle(f"{args.model}: the 64 first-layer filters (7x7x3, each min-max scaled)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"filters_{args.model}.png", dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- 2. thresholds from a subsample
    rng = np.random.default_rng(0)
    samples = {s: [] for s in STAGES}
    for i in range(0, 128, bs):
        acts = enc.layers(enc.pixels([open_image(r) for r in recs[i:i + bs]]))
        for s, li in STAGES.items():
            a = acts[li].permute(1, 0, 2, 3).flatten(1)               # [C, B*h*w]
            idx = torch.from_numpy(rng.choice(a.shape[1], min(4096, a.shape[1]), replace=False)).to(a.device)
            samples[s].append(a[:, idx].float().cpu())
    thr = {s: torch.quantile(torch.cat(v, 1), args.quantile, dim=1).to(DEVICE) for s, v in samples.items()}

    # ---------------------------------------------------------------- 3. dissection pass over all images
    masks = [object_masks(r) for r in recs]
    concepts = sorted({c for m in masks for c, v in m.items() if v.any()})
    n_img = {c: sum(1 for m in masks if c in m and m[c].any()) for c in concepts}
    concepts = [c for c in concepts if n_img[c] >= args.min_images]
    cidx = {c: k for k, c in enumerate(concepts)}
    print(f"{len(concepts)} concepts with masks in >= {args.min_images} images")

    C = {s: enc.layers(enc.pixels([open_image(recs[0])]))[li].shape[1] for s, li in STAGES.items()}
    inter = {s: torch.zeros(C[s], len(concepts), device=DEVICE) for s in STAGES}
    act_area = {s: torch.zeros(C[s], device=DEVICE) for s in STAGES}
    mask_area = torch.zeros(len(concepts), device=DEVICE)
    peak = {s: [] for s in STAGES}                                   # [N, C] max activation per image
    where = {s: [] for s in STAGES}                                  # [N, C] flat argmax position
    for i in range(0, len(recs), bs):
        batch = recs[i:i + bs]
        acts = enc.layers(enc.pixels([open_image(r) for r in batch]))
        mk = torch.zeros(len(batch), len(concepts), M * M, device=DEVICE)
        for b, m in enumerate(masks[i:i + bs]):
            for c, v in m.items():
                if c in cidx:
                    small = torch.from_numpy(v.astype(np.float32))[None, None]
                    mk[b, cidx[c]] = (F.interpolate(small, size=(M, M), mode="area")[0, 0] > 0.5).flatten().float()
        mask_area += mk.sum((0, 2))
        for s, li in STAGES.items():
            a = acts[li]
            pk, pos = a.flatten(2).max(2)
            peak[s].append(pk.cpu()); where[s].append(pos.cpu())
            up = F.interpolate(a, size=(M, M), mode="bilinear", align_corners=False)
            on = (up > thr[s][None, :, None, None]).flatten(2).float()   # [B, C, M*M]
            act_area[s] += on.sum((0, 2))
            inter[s] += torch.einsum("bup,bcp->uc", on, mk)
        print(f"\r  {min(i + bs, len(recs))}/{len(recs)} images", end="", flush=True)
    print()

    summary = {"model": args.model, "quantile": args.quantile, "iou_threshold": args.iou,
               "n_images": len(recs), "concepts": {c: n_img[c] for c in concepts}, "stages": {}}
    best = {}
    for s in STAGES:
        iou = (inter[s] / (act_area[s][:, None] + mask_area[None, :] - inter[s] + 1e-9)).cpu().numpy()
        b, bi = iou.max(1), iou.argmax(1)
        best[s] = (b, bi)
        det = b > args.iou
        counts = {}
        for k in bi[det]:
            counts[concepts[k]] = counts.get(concepts[k], 0) + 1
        summary["stages"][s] = {"units": int(len(b)), "detectors": int(det.sum()),
                                "fraction_detectors": float(det.mean()),
                                "detectors_per_concept": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
                                "top_units": [{"unit": int(u), "concept": concepts[bi[u]], "iou": float(b[u])}
                                              for u in np.argsort(-b)[:10]]}
        print(f"{s}: {det.sum()}/{len(b)} units with IoU > {args.iou}; top concepts "
              f"{list(summary['stages'][s]['detectors_per_concept'].items())[:5]}")
        np.savez(out / f"dissect_{args.model}_{s}.npz", iou=iou, peak=torch.cat(peak[s]).numpy(),
                 where=torch.cat(where[s]).numpy())
    json.dump(summary, open(out / f"dissection_{args.model}.json", "w"), indent=1)

    # ---------------------------------------------------------------- 4. dataset examples per stage
    k_img = 6
    for s, li in STAGES.items():
        b, bi = best[s]
        top = list(np.argsort(-b)[: args.units_per_stage // 2])       # best-matching units ...
        rest = [u for u in rng.permutation(len(b)) if u not in top]   # ... and random ones, to stay honest
        units = top + rest[: args.units_per_stage - len(top)]
        pk = torch.cat(peak[s]).numpy(); pos = torch.cat(where[s]).numpy()
        g = int(round(SIZE / 32 * {"layer1": 8, "layer2": 4, "layer3": 2, "layer4": 1}[s]))   # map side
        fig, axes = plt.subplots(len(units), k_img, figsize=(1.6 * k_img, 1.75 * len(units)))
        for row, u in enumerate(units):
            for col, n in enumerate(np.argsort(-pk[:, u])[:k_img]):
                img = np.asarray(square_crop(open_image(recs[n])))
                yy, xx = divmod(int(pos[n, u]), g)
                cy, cx = (yy + 0.5) * SIZE / g, (xx + 0.5) * SIZE / g
                h = CROP[s] // 2
                y0, x0 = int(np.clip(cy - h, 0, SIZE - 2 * h)), int(np.clip(cx - h, 0, SIZE - 2 * h))
                ax = axes[row, col]
                ax.imshow(img[y0:y0 + 2 * h, x0:x0 + 2 * h]); ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    tag = "best IoU" if u in top else "random"
                    ax.set_ylabel(f"unit {u}\n{tag}: {concepts[bi[u]]}\nIoU {b[u]:.3f}", fontsize=7)
        fig.suptitle(f"{args.model} {s}: top-{k_img} dataset examples ({CROP[s]}px crops around the peak)",
                     fontsize=10)
        fig.tight_layout(); fig.savefig(out / f"examples_{args.model}_{s}.png", dpi=120); plt.close(fig)

    # ---------------------------------------------------------------- 5. summary plot (+ the other model if run)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for j, name in enumerate(["resnet50", "resnet50_random"]):
        f = out / f"dissection_{name}.json"
        if f.exists():
            d = json.load(open(f))["stages"]
            ax.bar(np.arange(len(STAGES)) + 0.38 * j, [d[s]["fraction_detectors"] for s in STAGES], 0.38,
                   label=name, color=["#2a78d6", "#8a8984"][j])
    ax.set_xticks(np.arange(len(STAGES)) + 0.19, list(STAGES))
    ax.set_ylabel(f"fraction of units with IoU > {args.iou}")
    ax.set_title("Network dissection with COCO object masks", loc="left", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "dissection.png", dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
