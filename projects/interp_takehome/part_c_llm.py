"""Part C -- activation patching on GPT-2 small: indirect-object identification (IOI).

    python part_c_llm.py --corruption swap --out results/llm
    python part_c_llm.py --corruption abc  --out results/llm

Denoising patch: run the *corrupted* prompts and copy one activation from the *clean* run. The metric is
the share of the clean-vs-corrupted logit difference that this single activation restores:
    (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt),   LD = logit(IO) - logit(S).
Outputs a layer x position map for the residual stream and a layer x head map for attention-head
outputs at the final position.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from interp_utils import IOI_TEMPLATES, LLM, ioi_dataset, load_llm, logit_diff, run_patched, run_with_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LLM)
    ap.add_argument("--template", type=int, default=0, help=f"index into {len(IOI_TEMPLATES)} IOI templates")
    ap.add_argument("--corruption", default="swap", choices=["swap", "abc"])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", default="results/llm")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model, tok = load_llm(args.model)
    template = IOI_TEMPLATES[args.template]
    clean, corrupt, io, s = ioi_dataset(tok, template, n=args.n, corruption=args.corruption)
    clean_logits, cache = run_with_cache(model, clean)
    ld_clean = logit_diff(clean_logits, io, s)
    ld_corr = logit_diff(run_with_cache(model, corrupt)[0], io, s)
    print(f"logit diff: clean {ld_clean:.2f}, corrupted {ld_corr:.2f}")
    norm = lambda ld: (ld - ld_corr) / (ld_clean - ld_corr)          # noqa: E731

    L, H, T = model.config.n_layer, model.config.n_head, clean.shape[1]
    d_head = model.config.n_embd // H
    resid = np.zeros((L, T))
    for l in range(L):
        for t in range(T):
            patched = run_patched(model, corrupt, [("resid", l, t, None, cache[("resid", l)][:, t])])
            resid[l, t] = norm(logit_diff(patched, io, s))
    heads = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            z = cache[("z", l)][:, -1, h * d_head:(h + 1) * d_head]
            heads[l, h] = norm(logit_diff(run_patched(model, corrupt, [("z", l, -1, h, z)]), io, s))
        print(f"\rlayer {l}", end="")
    print()

    tag = f"{args.corruption}_t{args.template}"
    np.save(out / f"resid_{tag}.npy", resid); np.save(out / f"heads_{tag}.npy", heads)
    top = np.argsort(-np.abs(heads).ravel())[:8]
    summary = {"template": template, "corruption": args.corruption, "n": args.n,
               "ld_clean": ld_clean, "ld_corrupt": ld_corr,
               "top_heads": [{"head": f"{i // H}.{i % H}", "restored": float(heads.ravel()[i])} for i in top]}
    json.dump(summary, open(out / f"summary_{tag}.json", "w"), indent=1)
    print(json.dumps(summary["top_heads"], indent=1))

    labels = [tok.decode([t]) for t in clean[0].tolist()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    im = axes[0].imshow(resid, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(range(T), [repr(x)[1:-1] for x in labels], rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("layer"); axes[0].set_title("Residual stream (first prompt's tokens shown)", loc="left")
    fig.colorbar(im, ax=axes[0], fraction=0.03)
    im = axes[1].imshow(heads, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    axes[1].set_xlabel("head"); axes[1].set_ylabel("layer")
    axes[1].set_title("Head outputs at the final position", loc="left")
    fig.colorbar(im, ax=axes[1], fraction=0.04)
    fig.suptitle(f"Share of logit difference restored — corruption: {args.corruption}", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out / f"patching_{tag}.png", dpi=140)
    print("wrote", out / f"patching_{tag}.png")


if __name__ == "__main__":
    main()
