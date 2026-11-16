"""Q2 -- how is the final embedding space organised, and how does that depend on the training signal?

    python q2_embeddings.py --models resnet50 vit_sup clip dinov2 clip_random --out results/embed

For each encoder, embeds every image once (cached), then
  1. k-nearest-neighbour classification (cosine, k=10) of the image's dominant object supercategory:
     "are images of the same kind close together?";
  2. a linear probe (logistic regression) on the same labels: "is the category linearly decodable?";
  3. a 2-D t-SNE map coloured by category;
  4. nearest-neighbour retrieval for a few query images.
Labels: the supercategory of the largest object, keeping images where that object covers >= --min-frac of
the image and supercategories with >= --min-count images. Metric: balanced accuracy (chance = 1/K).
"""
import argparse
import warnings
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from interp_utils import Encoder, bootstrap_ci, load_coco, open_image, run_batched, split_ids, square_crop

warnings.filterwarnings("ignore", message="y_pred contains classes")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e24a94", "#4f9e2a", "#8b5cf6", "#e33d3d",
           "#52514e", "#00a3c4", "#b07a3b", "#9aa2a8"]


def knn_predict(Ztr, ytr, Zte, k=10):
    Ztr = Ztr / np.linalg.norm(Ztr, axis=1, keepdims=True)
    Zte = Zte / np.linalg.norm(Zte, axis=1, keepdims=True)
    nn = np.argsort(-(Zte @ Ztr.T), axis=1)[:, :k]
    return np.array([Counter(ytr[row]).most_common(1)[0][0] for row in nn])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/coco1k")
    ap.add_argument("--models", nargs="+", default=["resnet50", "vit_sup", "clip", "dinov2", "clip_random"])
    ap.add_argument("--min-frac", type=float, default=0.05)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="results/embed")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    recs = load_coco(args.data)["images"]
    keep = [r for r in recs if r["dominant_frac"] >= args.min_frac]
    counts = Counter(r["dominant"] for r in keep)
    classes = sorted(c for c, n in counts.items() if n >= args.min_count)
    keep = [r for r in keep if r["dominant"] in classes]
    print(f"{len(keep)} images, {len(classes)} classes:", {c: counts[c] for c in classes})
    y = np.array([classes.index(r["dominant"]) for r in keep])
    train, _ = split_ids(keep)
    tr = np.array([r["id"] in train for r in keep])

    emb = {}
    for name in args.models:
        f = out / f"emb_{name}.npy"
        if not f.exists():
            enc = Encoder(name)
            np.save(f, run_batched(lambda ims: enc.embed(enc.pixels(ims)), recs))
            del enc
        Z = np.load(f)
        emb[name] = Z[[recs.index(r) for r in keep]] if len(Z) == len(recs) else Z

    results = {}
    for name, Z in emb.items():
        pred_knn = knn_predict(Z[tr], y[tr], Z[~tr], args.k)
        sc = StandardScaler().fit(Z[tr])
        clf = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(Z[tr]), y[tr])
        pred_lin = clf.predict(sc.transform(Z[~tr]))
        yt = y[~tr]
        res = {}
        for tag, p in (("knn", pred_knn), ("linear", pred_lin)):
            acc = balanced_accuracy_score(yt, p)
            lo, hi = bootstrap_ci(lambda i: balanced_accuracy_score(yt[i], p[i]), len(yt))
            res[tag] = {"balanced_acc": float(acc), "ci": [float(lo), float(hi)]}
        results[name] = res
        print(f"{name:12s} kNN {res['knn']['balanced_acc']:.3f} [{res['knn']['ci'][0]:.2f}, "
              f"{res['knn']['ci'][1]:.2f}]   linear {res['linear']['balanced_acc']:.3f} "
              f"[{res['linear']['ci'][0]:.2f}, {res['linear']['ci'][1]:.2f}]")
    json.dump({"classes": classes, "n_train": int(tr.sum()), "n_test": int((~tr).sum()),
               "chance": 1 / len(classes), "results": results}, open(out / "embed_summary.json", "w"), indent=1)

    # ---------------------------------------------------------------- t-SNE maps
    fig, axes = plt.subplots(1, len(emb), figsize=(3.3 * len(emb), 3.6))
    for ax, (name, Z) in zip(np.atleast_1d(axes), emb.items()):
        Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        P = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Zn)
        for c in range(len(classes)):
            ax.scatter(*P[y == c].T, s=5, color=PALETTE[c % len(PALETTE)], label=classes[c])
        ax.set_title(f"{name}\nkNN bal. acc {results[name]['knn']['balanced_acc']:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    np.atleast_1d(axes)[-1].legend(fontsize=7, markerscale=2.5, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout(); fig.savefig(out / "tsne.png", dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- retrieval for 3 test queries
    test_idx = np.where(~tr)[0]
    queries = [next(i for i in test_idx if classes[y[i]] == c) for c in ("animal", "vehicle", "food")
               if c in classes][:3]
    for qi, q in enumerate(queries):
        fig, axes = plt.subplots(len(emb), 6, figsize=(9, 1.6 * len(emb)))
        for row, (name, Z) in enumerate(emb.items()):
            Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
            sims = Zn @ Zn[q]
            sims[q] = -np.inf
            for col, j in enumerate([q] + list(np.argsort(-sims)[:5])):
                ax = axes[row, col]
                ax.imshow(square_crop(open_image(keep[j]), 112)); ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel("query" if col == 0 else classes[y[j]], fontsize=7)
                if col == 0:
                    ax.set_ylabel(name, fontsize=8)
        fig.suptitle("Nearest neighbours (cosine) in each embedding space", fontsize=10)
        fig.tight_layout(); fig.savefig(out / f"retrieval_{qi}.png", dpi=110); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
