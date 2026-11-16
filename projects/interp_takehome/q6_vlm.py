"""Q6 -- where and how does a small VLM use the image? POPE-style questions + attention knockout.

    # 1) answer POPE-style yes/no questions, save P(yes) and last-token hidden states
    python q6_vlm.py pope --data data/coco1k --n 150 --out results/vlm
    # 2) attention knockout: block text->image attention in one window of LLM layers at a time
    python q6_vlm.py knockout --data data/coco1k --n 40 --window 3 --out results/vlm

Question types follow POPE (Li et al. 2023): for each image one *present* object, and absent objects
sampled at random, from the most frequent COCO classes ("popular"), or from the classes that co-occur
most with what is in the image ("adversarial").
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from interp_utils import VLM, VLMYesNo, load_coco, open_image

QUESTION = "Is there a {} in the image? Answer yes or no."


def build_questions(meta, n, seed=0):
    rng = np.random.default_rng(seed)
    cats, freq, cooc = meta["categories"], meta["frequency"], meta["cooccurrence"]
    popular = sorted(cats, key=lambda c: -freq.get(c, 0))
    qs = []
    for r in meta["images"][:n]:
        present = set(r["classes"])
        absent = [c for c in cats if c not in present]
        # present object: the largest instance, so "present" is not dominated by tiny background objects
        big = max(r["objects"], key=lambda o: o["area"])["category"]
        score = {c: sum(cooc.get(p, {}).get(c, 0) for p in present) for c in absent}
        qs += [
            dict(id=r["id"], object=big, kind="present", label=1),
            dict(id=r["id"], object=str(rng.choice(absent)), kind="random", label=0),
            dict(id=r["id"], object=next(c for c in popular if c not in present), kind="popular", label=0),
            dict(id=r["id"], object=max(absent, key=score.get), kind="adversarial", label=0),
        ]
    return qs


def cmd_pope(args, meta, vlm, by_id, out):
    qs = build_questions(meta, args.n)
    hidden = []
    for i, q in enumerate(qs):
        res = vlm(open_image(by_id[q["id"]]), QUESTION.format(q["object"]), hidden=True)
        q["p_yes"] = res["p_yes"]
        hidden.append(res["hidden"])
        print(f"\r{i + 1}/{len(qs)}", end="")
    print()
    np.save(out / "pope_hidden.npy", np.stack(hidden))        # [n_questions, n_layers + 1, d]
    with open(out / "pope.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(qs[0]))
        w.writeheader(); w.writerows(qs)
    summary = {}
    for kind in ["present", "random", "popular", "adversarial"]:
        p = np.array([q["p_yes"] for q in qs if q["kind"] == kind])
        summary[kind] = {"n": len(p), "mean_p_yes": float(p.mean()), "said_yes": float((p > 0.5).mean())}
    json.dump(summary, open(out / "pope_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


def cmd_knockout(args, meta, vlm, by_id, out):
    qs = [q for q in build_questions(meta, args.n) if q["kind"] in ("present", "adversarial")]
    L = len(vlm.layers)
    starts = list(range(0, L, args.window))
    table = np.zeros((len(qs), len(starts) + 1))
    for i, q in enumerate(qs):
        img, text = open_image(by_id[q["id"]]), QUESTION.format(q["object"])
        table[i, 0] = vlm(img, text)["p_yes"]
        for j, s in enumerate(starts, start=1):
            table[i, j] = vlm(img, text, block_layers=range(s, min(L, s + args.window)))["p_yes"]
        print(f"\r{i + 1}/{len(qs)}", end="")
    print()
    np.save(out / "knockout.npy", table)
    json.dump(qs, open(out / "knockout_questions.json", "w"))

    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(starts) + 1)
    for kind, c in [("present", "#2a78d6"), ("adversarial", "#e34948")]:
        rows = table[[q["kind"] == kind for q in qs]]
        mu, se = rows.mean(0), rows.std(0) / np.sqrt(len(rows))
        ax.errorbar(x, mu, yerr=1.96 * se, color=c, marker="o", capsize=3, label=f"{kind} (n={len(rows)})")
    ax.set_xticks(x, ["none"] + [f"{s}–{min(L, s + args.window) - 1}" for s in starts], rotation=30)
    ax.set_xlabel("LLM layers with text→image attention blocked"); ax.set_ylabel("mean P(yes) ± 95% CI")
    ax.set_title(f"Attention knockout, {args.model.split('/')[-1]}", loc="left"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "knockout.png", dpi=140)
    print("wrote", out / "knockout.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pope", "knockout"])
    ap.add_argument("--data", default="data/coco1k")
    ap.add_argument("--model", default=VLM)
    ap.add_argument("--n", type=int, default=150, help="number of images")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--out", default="results/vlm")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = load_coco(args.data)
    by_id = {r["id"]: r for r in meta["images"]}
    vlm = VLMYesNo(args.model)
    {"pope": cmd_pope, "knockout": cmd_knockout}[args.cmd](args, meta, vlm, by_id, out)


if __name__ == "__main__":
    main()
