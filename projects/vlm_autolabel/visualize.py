"""Side-by-side ground truth vs. auto-label sets, for failure analysis and reports.

    python visualize.py --images data/coco_subset/images --gt data/coco_subset/gt.json \
        --preds results/preds_A.json results/preds_E.json --names "OWLv2 only" "Harness" \
        --n 3 --out compare.jpg
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def draw(ax, image, anns, cat_name, cat_color, title):
    ax.imshow(image); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, loc="left", fontsize=10)
    for a in anns:
        x, y, w, h = a["bbox"]
        c = cat_color[a["category_id"]]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=c, linewidth=1.8))
        label = cat_name[a["category_id"]] + (f' {a["score"]:.2f}' if "score" in a else "")
        ax.text(x, y - 2, label, fontsize=6.5, color="white", bbox=dict(facecolor=c, edgecolor="none", pad=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--ids", nargs="*", type=int, help="image ids (default: the first --n)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=0.3, help="hide low-confidence boxes")
    ap.add_argument("--out", default="compare.jpg")
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    cat_name = {c["id"]: c["name"] for c in gt["categories"]}
    cat_color = {c["id"]: COLORS[i % len(COLORS)] for i, c in enumerate(gt["categories"])}
    images = {im["id"]: im for im in gt["images"]}
    ids = args.ids or sorted(images)[: args.n]

    sets = [("Ground truth", gt["annotations"])]
    names = args.names or [Path(p).stem for p in args.preds]
    for name, p in zip(names, args.preds):
        sets.append((name, [a for a in json.load(open(p)) if a["score"] >= args.min_score]))
    by_img = [defaultdict(list) for _ in sets]
    for k, (_, anns) in enumerate(sets):
        for a in anns:
            by_img[k][a["image_id"]].append(a)

    fig, axes = plt.subplots(len(ids), len(sets), figsize=(4.6 * len(sets), 3.6 * len(ids)), squeeze=False)
    for r, img_id in enumerate(ids):
        image = Image.open(Path(args.images) / images[img_id]["file_name"]).convert("RGB")
        for c, (name, _) in enumerate(sets):
            draw(axes[r, c], image, by_img[c][img_id], cat_name, cat_color,
                 f"{name}  ({len(by_img[c][img_id])} boxes)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, pil_kwargs={"quality": 85})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
