"""Q3 -- how does the representation change with depth, and do different models learn similar ones?

    python q3_layers.py --out results/layers

1. Layer-wise linear probes (Alain & Bengio 2016): at every layer, a logistic-regression probe predicts
   "does the image contain a {class}?" from the pooled activations. Test AUROC with a bootstrap 95% CI,
   plus a shuffled-label control (Hewitt & Liang 2019) and a random-weight network.
2. Linear CKA (Kornblith et al. 2019) between every pair of layers of two models, computed on the same
   images. Shows which layers of model A carry the same information as which layers of model B.

Pooling: ViTs use the [CLS] token (--pool cls) or the mean of the patch tokens (--pool mean) for the
probes. ResNets have no [CLS] token and always use global average pooling. CKA always uses mean pooling.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from interp_utils import Encoder, bootstrap_ci, linear_cka, load_coco, run_batched, split_ids

COLORS = {"resnet50": "#4f9e2a", "vit_sup": "#8b5cf6", "clip": "#2a78d6", "dinov2": "#eb6834",
          "clip_random": "#8a8984", "resnet50_random": "#c4c3be"}
CKA_PAIRS = [("dinov2", "dinov2"), ("clip", "clip_random"), ("clip", "dinov2"), ("vit_sup", "dinov2"),
             ("resnet50", "vit_sup")]


def layer_feats(name, recs, pool, cache):
    f = cache / f"layers_{name}_{pool}.npz"
    if not f.exists():
        enc = Encoder(name)
        feats = run_batched(lambda ims: enc.pooled_layers(enc.pixels(ims), pool), recs)
        np.savez(f, *feats)
        del enc
    z = np.load(f)
    return [z[f"arr_{i}"] for i in range(len(z.files))]


def probe(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr).decision_function(sc.transform(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/coco1k")
    ap.add_argument("--models", nargs="+", default=["resnet50", "vit_sup", "clip", "dinov2", "clip_random"])
    ap.add_argument("--classes", nargs="+", default=["person", "car", "chair", "cup", "dog"])
    ap.add_argument("--pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--C", type=float, default=0.1, help="inverse L2 strength of the probe")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--skip-cka", action="store_true")
    ap.add_argument("--out", default="results/layers")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    recs = load_coco(args.data)["images"]
    train, _ = split_ids(recs)
    tr = np.array([r["id"] in train for r in recs])
    Y = {c: np.array([c in r["classes"] for r in recs], int) for c in args.classes}
    print({c: f"{y[~tr].sum()} test positives" for c, y in Y.items()})

    # ---------------------------------------------------------------- 1. probes
    if not args.skip_probe:
        rows = []
        for name in args.models:
            X = layer_feats(name, recs, "mean" if name.startswith("resnet") else args.pool, out)
            for layer, Xl in enumerate(X):
                for c, y in Y.items():
                    s = probe(Xl[tr], y[tr], Xl[~tr], args.C)
                    yt = y[~tr]
                    lo, hi = bootstrap_ci(lambda i: roc_auc_score(yt[i], s[i]) if 0 < yt[i].sum() < len(i)
                                          else np.nan, len(yt))
                    y_shuf = np.random.default_rng(layer).permutation(y[tr])     # control: no real signal
                    s_ctrl = probe(Xl[tr], y_shuf, Xl[~tr], args.C)
                    rows.append(dict(model=name, layer=layer, depth=layer / (len(X) - 1), cls=c,
                                     auc=roc_auc_score(yt, s), lo=lo, hi=hi, control_auc=roc_auc_score(yt, s_ctrl)))
            last = [r["auc"] for r in rows if r["model"] == name and r["layer"] == len(X) - 1]
            print(f"{name:12s} last-layer mean AUROC {np.mean(last):.3f}")
        json.dump(rows, open(out / f"probe_{args.pool}.json", "w"), indent=1)

        fig, axes = plt.subplots(1, len(args.classes), figsize=(3.2 * len(args.classes), 3.5), sharey=True)
        for ax, c in zip(np.atleast_1d(axes), args.classes):
            for name in args.models:
                rr = [r for r in rows if r["model"] == name and r["cls"] == c]
                d = [r["depth"] for r in rr]
                ax.plot(d, [r["auc"] for r in rr], color=COLORS.get(name), label=name, marker=".", ms=3)
                ax.fill_between(d, [r["lo"] for r in rr], [r["hi"] for r in rr], color=COLORS.get(name), alpha=0.12)
            ctrl = [r["control_auc"] for r in rows if r["cls"] == c]
            ax.axhspan(min(ctrl), max(ctrl), color="#bbb", alpha=0.25, lw=0)
            ax.axhline(0.5, color="#999", lw=0.8, ls=":")
            ax.set_title(f"{c} (n_test+={Y[c][~tr].sum()})", loc="left", fontsize=10)
            ax.set_xlabel("relative depth (0 = input, 1 = last layer)")
        np.atleast_1d(axes)[0].set_ylabel("test AUROC (95% bootstrap CI)")
        np.atleast_1d(axes)[0].legend(fontsize=7)
        fig.suptitle(f"Layer-wise probes ({args.pool} pooling for ViTs; grey band = shuffled-label controls)",
                     fontsize=10)
        fig.tight_layout(); fig.savefig(out / f"probe_by_layer_{args.pool}.png", dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- 2. CKA
    if not args.skip_cka:
        pairs = [(a, b) for a, b in CKA_PAIRS if a in args.models and b in args.models]
        feats = {m: layer_feats(m, recs, "mean", out) for m in {x for p in pairs for x in p}}
        res = {}
        fig, axes = plt.subplots(1, len(pairs), figsize=(3.6 * len(pairs), 3.4))
        fig.subplots_adjust(wspace=0.45)
        for ax, (a, b) in zip(np.atleast_1d(axes), pairs):
            K = np.array([[linear_cka(xa, xb) for xb in feats[b]] for xa in feats[a]])
            res[f"{a}|{b}"] = K.tolist()
            im = ax.imshow(K, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xlabel(f"{b} layer"); ax.set_ylabel(f"{a} layer")
            ax.set_title(f"{a} vs {b}", fontsize=9)
            print(f"CKA {a} vs {b}: last-last {K[-1, -1]:.2f}, max {K.max():.2f}")
        fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=0.8, label="linear CKA")
        fig.savefig(out / "cka.png", dpi=140, bbox_inches="tight"); plt.close(fig)
        json.dump(res, open(out / "cka.json", "w"))
    print("wrote", out)


if __name__ == "__main__":
    main()
