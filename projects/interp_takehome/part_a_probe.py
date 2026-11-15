"""Part A2 -- layer-wise linear probes: where in a vision encoder is "this image contains a {class}" decodable?

    python part_a_probe.py --data data/coco400 --models clip dinov2 clip_random --out results/probe

For each encoder and each layer, fits one logistic-regression probe per class on frozen features
(image-level split) and reports test AUROC with a bootstrap 95% CI. A shuffled-label control shows how
much a probe can score with no real signal.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from interp_utils import layer_features, load_coco, load_vision, open_image, split_ids

COLORS = {"clip": "#2a78d6", "dinov2": "#eb6834", "clip_random": "#52514e"}


def bootstrap_auc(y, s, n=500, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            aucs.append(roc_auc_score(y[i], s[i]))
    return np.percentile(aucs, [2.5, 97.5])


def probe(Xtr, ytr, Xte, yte, C=1.0):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/coco400")
    ap.add_argument("--models", nargs="+", default=["clip", "dinov2", "clip_random"])
    ap.add_argument("--classes", nargs="+", default=["person", "chair", "dining table", "car", "cup"])
    ap.add_argument("--pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--C", type=float, default=0.1, help="inverse L2 strength of the probe")
    ap.add_argument("--out", default="results/probe")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    recs = load_coco(args.data)["images"]
    train, test = split_ids(recs)
    tr = np.array([r["id"] in train for r in recs])
    images = [open_image(r) for r in recs]
    Y = {c: np.array([c in r["classes"] for r in recs], int) for c in args.classes}
    print({c: f"{y.sum()}/{len(y)} positive" for c, y in Y.items()})

    rows = []
    for name in args.models:
        feats = out / f"feats_{name}_{args.pool}.npy"
        if feats.exists():
            X = np.load(feats)
        else:
            model, proc = load_vision(name)
            X = layer_features(model, proc, images, pool=args.pool)
            np.save(feats, X)
        for layer in range(X.shape[0]):
            for c, y in Y.items():
                s = probe(X[layer][tr], y[tr], X[layer][~tr], y[~tr], args.C)
                lo, hi = bootstrap_auc(y[~tr], s)
                y_shuf = np.random.default_rng(layer).permutation(y[tr])       # control: no real signal
                s_ctrl = probe(X[layer][tr], y_shuf, X[layer][~tr], y[~tr], args.C)
                rows.append(dict(model=name, layer=layer, cls=c, auc=roc_auc_score(y[~tr], s), lo=lo, hi=hi,
                                 control_auc=roc_auc_score(y[~tr], s_ctrl)))
            print(name, layer, np.mean([r["auc"] for r in rows if r["model"] == name and r["layer"] == layer]))
    json.dump(rows, open(out / "probe_results.json", "w"), indent=1)

    fig, axes = plt.subplots(1, len(args.classes), figsize=(3.2 * len(args.classes), 3.4), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), args.classes):
        for name in args.models:
            rr = [r for r in rows if r["model"] == name and r["cls"] == c]
            L = [r["layer"] for r in rr]
            ax.plot(L, [r["auc"] for r in rr], color=COLORS.get(name), label=name)
            ax.fill_between(L, [r["lo"] for r in rr], [r["hi"] for r in rr], color=COLORS.get(name), alpha=0.15)
        ax.axhline(0.5, color="#999", lw=0.8, ls=":")
        ax.set_title(f"{c} (n_test+={Y[c][~tr].sum()})", loc="left", fontsize=10)
        ax.set_xlabel("layer")
    np.atleast_1d(axes)[0].set_ylabel("test AUROC (95% bootstrap CI)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "probe_by_layer.png", dpi=140)
    print("wrote", out / "probe_by_layer.png")


if __name__ == "__main__":
    main()
