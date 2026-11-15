"""Part A1 -- do saliency maps explain CLIP? Randomisation sanity checks + deletion curves.

    python part_a_saliency.py --data data/coco400 --n 30 --out results/saliency

For each image (whose zero-shot top-1 is a class actually in the image), computes gradient, gradient x
input and integrated-gradients maps for log P(class), then
  1. cascading randomisation: rank correlation between the map of the trained model and the map after
     re-initialising the top k vision blocks (k = 1, 3, 6, 12);
  2. deletion curves: P(class) as the most-salient pixels are removed, vs. a random pixel order.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from interp_utils import ClipZeroShot, deletion_curve, load_coco, open_image, randomize_top_blocks, saliency

LABELS = ["person", "car", "bus", "dog", "cat", "bird", "horse", "pizza", "train", "airplane", "boat",
          "giraffe", "elephant", "zebra", "bicycle", "motorcycle", "truck", "sheep", "cow", "bear"]
METHODS = ["grad", "gxi", "ig"]
CASCADE = [1, 3, 6, 12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/coco400")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--fill", default="blur", choices=["blur", "grey"])
    ap.add_argument("--out", default="results/saliency")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    clf = ClipZeroShot(LABELS)
    picked = []
    for r in load_coco(args.data)["images"]:
        img = open_image(r)
        t = clf.log_probs(clf.pixels([img]))[0].argmax().item()
        if LABELS[t] in r["classes"]:                 # explain correct predictions only
            picked.append((r, img, t))
        if len(picked) == args.n:
            break
    print(f"{len(picked)} correctly classified images")

    maps = {m: [saliency(clf, clf.pixels([img]), t, m).numpy() for _, img, t in picked] for m in METHODS}
    rng = np.random.default_rng(0)
    deletion = {m: [deletion_curve(clf, img, s, t, fill=args.fill) for (_, img, t), s in zip(picked, maps[m])]
                for m in METHODS}
    deletion["random"] = [deletion_curve(clf, img, rng.random((224, 224)), t, fill=args.fill)
                          for _, img, t in picked]

    corr = {m: {} for m in METHODS}
    for k in CASCADE:
        clf_r = ClipZeroShot(LABELS)
        randomize_top_blocks(clf_r, k)
        for m in METHODS:
            corr[m][k] = [spearmanr(s.ravel(), saliency(clf_r, clf_r.pixels([img]), t, m).numpy().ravel()).statistic
                          for (_, img, t), s in zip(picked, maps[m])]
        print("randomised top", k, {m: round(float(np.mean(corr[m][k])), 3) for m in METHODS})

    summary = {"n": len(picked), "fill": args.fill,
               "deletion_auc": {m: float(np.mean([np.trapezoid(c, dx=1 / (len(c) - 1)) for c in v]))
                                for m, v in deletion.items()},
               "rank_corr_after_randomisation": {m: {k: float(np.mean(v)) for k, v in d.items()}
                                                 for m, d in corr.items()}}
    json.dump(summary, open(out / "saliency_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for m, c in zip(METHODS + ["random"], ["#2a78d6", "#eb6834", "#1baf7a", "#52514e"]):
        curves = np.array(deletion[m])
        x = np.linspace(0, 1, curves.shape[1])
        mu, se = curves.mean(0), curves.std(0) / np.sqrt(len(curves))
        axes[0].plot(x, mu, color=c, label=f"{m} (AUC {summary['deletion_auc'][m]:.2f})")
        axes[0].fill_between(x, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.15)
    axes[0].set_xlabel("fraction of pixels removed"); axes[0].set_ylabel("P(class)")
    axes[0].set_title("Deletion curves (lower = more faithful)", loc="left"); axes[0].legend(fontsize=8)
    for m, c in zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a"]):
        mu = [np.mean(corr[m][k]) for k in CASCADE]
        se = [np.std(corr[m][k]) / np.sqrt(len(corr[m][k])) for k in CASCADE]
        axes[1].errorbar(CASCADE, mu, yerr=1.96 * np.array(se), color=c, marker="o", label=m, capsize=3)
    axes[1].set_xlabel("top-k vision blocks re-initialised"); axes[1].set_ylabel("rank corr. with trained map")
    axes[1].set_title("Cascading randomisation (lower = depends on weights)", loc="left"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "saliency_checks.png", dpi=140)
    print("wrote", out / "saliency_checks.png")


if __name__ == "__main__":
    main()
