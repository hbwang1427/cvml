"""Regenerate the figures for Lecture 01 (Vision Foundation Models).

All numbers are as reported in the cited papers / model cards. They are meant to show
trends, not to serve as a leaderboard -- training recipes and eval protocols differ.

    python make_figures.py   # writes PNGs into ./figures
"""
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# Reference categorical palette (fixed order) + neutral inks.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("wrote", OUT / name)


def label_points(ax, pts, color=INK, dy=None, fontsize=8.5):
    """pts: (x, y, text[, (dx, dy) offset in points])"""
    for p in pts:
        x, y, t = p[:3]
        off = p[3] if len(p) > 3 else (6, 4)
        ax.annotate(t, (x, y), xytext=off, textcoords="offset points",
                    fontsize=fontsize, color=color)


# ---------------------------------------------------------------- 1. model scale
def fig_model_scale():
    # (year + month/12, params in millions, name, label offset)
    models = [
        (2012.9, 61, "AlexNet", (6, -10)),
        (2014.7, 138, "VGG-16", (6, 4)),
        (2015.9, 25.6, "ResNet-50", (6, -10)),
        (2020.8, 632, "ViT-H/14", (-62, 6)),
        (2021.1, 304, "CLIP ViT-L/14", (6, -12)),
        (2021.5, 1843, "ViT-G/14", (-58, 4)),
        (2023.1, 22000, "ViT-22B", (-52, 2)),
        (2023.3, 636, "SAM ViT-H", (6, -12)),
        (2023.3, 1100, "DINOv2 ViT-g", (6, 2)),
        (2023.95, 5900, "InternViT-6B", (-80, 0)),
        (2024.1, 18000, "EVA-CLIP-18B", (6, -4)),
        (2025.6, 6700, "DINOv3 ViT-7B", (-30, 8)),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = [m[0] for m in models]; ys = [m[1] for m in models]
    ax.scatter(xs, ys, s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    label_points(ax, [(m[0], m[1], m[2], m[3]) for m in models])
    ax.set_yscale("log")
    ax.set_xlim(2012, 2026.5)
    ax.set_ylabel("Parameters (millions, log scale)")
    ax.set_title("Vision backbones grew ~300× in a decade")
    ax.text(0.0, -0.16, "Image encoder only where a model has multiple towers. Sources: original papers.",
            transform=ax.transAxes, fontsize=8, color=INK2)
    save(fig, "01_backbone_scale.png")


# ---------------------------------------------------------------- 2. data scale
def fig_data_scale():
    # (name, images in millions, kind) ; kind: labeled / image-text / self-sup / masks
    data = [
        ("ImageNet-1k (2009)", 1.28, "labeled"),
        ("JFT-300M (2017)", 300, "labeled"),
        ("LVD-142M — DINOv2 (2023)", 142, "unlabeled (curated)"),
        ("SA-1B — SAM (2023)", 11, "masks"),
        ("CLIP WIT (2021)", 400, "image–text"),
        ("ALIGN (2021)", 1800, "image–text"),
        ("JFT-3B (2021)", 3000, "labeled"),
        ("LAION-5B (2022)", 5850, "image–text"),
        ("WebLI (2022)", 10000, "image–text"),
        ("DataComp pool (2023)", 12800, "image–text"),
        ("LVD-1689M — DINOv3 (2025)", 1689, "unlabeled (curated)"),
    ]
    data.sort(key=lambda d: d[1])
    color = {"labeled": INK2, "image–text": BLUE, "unlabeled (curated)": AQUA, "masks": ORANGE}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (n, v, k) in enumerate(data):
        ax.barh(i, v, color=color[k], height=0.62, label=k)
        txt = f"{v:,.0f}M" if v >= 10 else f"{v:g}M"
        ax.text(v * 1.15, i, txt, va="center", fontsize=8.5, color=INK2)
    ax.text(11 * 1.15 * 3.2, [d[0] for d in data].index("SA-1B — SAM (2023)"),
            "+ 1.1B masks", va="center", fontsize=8.5, color=ORANGE)
    ax.set_yticks(range(len(data)), [d[0] for d in data])
    ax.set_xscale("log")
    ax.set_xlim(0.5, 1e5)
    ax.set_xlabel("Images (millions, log scale)")
    ax.grid(axis="y", visible=False)
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=8.5, title="Supervision", title_fontsize=8.5)
    ax.set_title("Pre-training data moved from labels to the web")
    save(fig, "02_pretraining_data.png")


# ---------------------------------------------------------------- 3. ImageNet
def fig_imagenet():
    finetuned = [
        (2012.9, 63.3, "AlexNet"), (2015.9, 78.6, "ResNet-152"), (2019.4, 84.3, "EfficientNet-B7"),
        (2020.8, 88.6, "ViT-H/14"), (2021.5, 90.9, "CoAtNet-7"), (2022.4, 91.0, "CoCa"),
    ]
    zeroshot = [
        (2021.1, 76.2, "CLIP L/14@336"), (2022.4, 86.3, "CoCa"),
        (2023.0, 80.1, "OpenCLIP bigG"), (2024.1, 83.8, "EVA-CLIP-18B"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot([s[0] for s in finetuned], [s[1] for s in finetuned], "-o", color=BLUE, lw=2, ms=7,
            mec=SURFACE, mew=1.5, label="Trained / fine-tuned on ImageNet labels")
    # Zero-shot entries are independent models, not a progression -> markers only.
    ax.plot([s[0] for s in zeroshot], [s[1] for s in zeroshot], "o", color=ORANGE, ms=8,
            mec=SURFACE, mew=1.5, label="Zero-shot (no ImageNet labels used)")
    offs = {"CoAtNet-7": (-40, 8), "CoCa": (8, 2), "AlexNet": (8, -3)}
    for x, y, t in finetuned:
        ax.annotate(t, (x, y), xytext=offs.get(t, (8, -12)), textcoords="offset points", fontsize=8.5, color=INK2)
    for x, y, t in zeroshot:
        ax.annotate(t, (x, y), xytext=(8, -4), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_ylabel("ImageNet-1k top-1 (%)")
    ax.set_ylim(60, 95); ax.set_xlim(2012, 2025.6)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Zero-shot CLIP-style models reach supervised-era accuracy")
    save(fig, "03_imagenet_zero_shot.png")


# ---------------------------------------------------------------- 4. generative FID
def fig_fid():
    fid = [
        (2018.8, 6.95, "BigGAN-deep", "GAN"), (2021.4, 4.59, "ADM-G", "Diffusion (pixel)"),
        (2021.95, 3.60, "LDM-4-G", "Diffusion (latent)"), (2022.95, 2.27, "DiT-XL/2", "Diffusion (latent)"),
        (2024.05, 2.06, "SiT-XL", "Diffusion (latent)"), (2024.3, 1.73, "VAR-d30", "Autoregressive"),
        (2024.45, 1.55, "MAR-H", "Autoregressive"), (2024.8, 1.42, "REPA (SiT-XL)", "Diffusion (latent)"),
        (2025.05, 1.35, "LightningDiT", "Diffusion (latent)"),
    ]
    color = {"GAN": INK2, "Diffusion (pixel)": AQUA, "Diffusion (latent)": BLUE, "Autoregressive": ORANGE}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot([f[0] for f in fid], [f[1] for f in fid], color=GRID, lw=2, zorder=1)
    for x, y, n, k in fid:
        ax.scatter(x, y, s=60, color=color[k], edgecolor=SURFACE, linewidth=1.5, zorder=3, label=k)
    offs = {"BigGAN-deep": (6, 4), "ADM-G": (6, 4), "LDM-4-G": (6, 4), "DiT-XL/2": (-30, 10),
            "SiT-XL": (-14, 9), "VAR-d30": (-78, -4), "MAR-H": (-6, 9),
            "REPA (SiT-XL)": (-40, -16), "LightningDiT": (8, -3)}
    for x, y, n, k in fid:
        ax.annotate(f"{n}  {y}", (x, y), xytext=offs[n], textcoords="offset points", fontsize=8.5, color=INK2)
    h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h))
    ax.legend(u.values(), u.keys(), loc="upper right", fontsize=9)
    ax.set_ylabel("FID ↓ (ImageNet 256×256, class-cond.)")
    ax.set_xlim(2018.3, 2026.1); ax.set_ylim(0, 7.8)
    ax.set_title("Generative vision: 5× lower FID since BigGAN")
    ax.text(0.0, -0.16, "REPA / LightningDiT align the generator with a pretrained encoder (DINOv2) — "
            "representation learning now drives generation.", transform=ax.transAxes, fontsize=8, color=INK2)
    save(fig, "04_generative_fid.png")


# ---------------------------------------------------------------- 5. domain timeline
def fig_domain_timeline():
    # (year, month), name, label level: +n above the lane, -n below (avoids collisions)
    lanes = {
        "General vision": (BLUE, [
            ((2021, 1), "CLIP", 1), ((2023, 4), "SAM", -1), ((2023, 4), "DINOv2", 1), ((2023, 4), "LLaVA", 2),
            ((2024, 7), "SAM 2", 1), ((2025, 2), "Qwen2.5-VL", -1), ((2025, 2), "SigLIP 2", 1),
            ((2025, 8), "DINOv3", 2), ((2025, 11), "SAM 3", 1)]),
        "Healthcare": (AQUA, [
            ((2023, 3), "BiomedCLIP", 1), ((2023, 4), "MedSAM", -1), ((2023, 6), "LLaVA-Med", 2),
            ((2023, 7), "Med-PaLM M", -2), ((2023, 8), "UNI / CONCH", 3), ((2023, 9), "RETFound", -3),
            ((2024, 5), "Prov-GigaPath", 1), ((2024, 5), "BiomedParse", -1), ((2025, 5), "MedGemma", 1)]),
        "Robotics": (ORANGE, [
            ((2022, 12), "RT-1", 1), ((2023, 3), "PaLM-E", -1), ((2023, 7), "RT-2", 1),
            ((2023, 10), "Open X-Emb.", -1), ((2024, 5), "Octo", 1), ((2024, 6), "OpenVLA", -1),
            ((2024, 10), "π0", 1), ((2025, 3), "GR00T N1", -1), ((2025, 3), "Gemini Robotics", 2),
            ((2025, 4), "π0.5", 1)]),
        "Autonomous driving": (VIOLET, [
            ((2022, 12), "UniAD", 1), ((2023, 9), "GAIA-1", -1), ((2023, 12), "DriveLM", 1),
            ((2024, 2), "DriveVLM", -1), ((2024, 10), "EMMA", 1), ((2025, 1), "Cosmos WFM", -1),
            ((2025, 3), "GAIA-2", 1)]),
    }
    fig, ax = plt.subplots(figsize=(10, 6.6))
    names = list(lanes)
    for li, name in enumerate(names):
        y = len(names) - 1 - li
        c, items = lanes[name]
        ax.axhline(y, color=GRID, lw=1, zorder=0)
        for (yr, mo), t, lvl in items:
            d = date(yr, mo, 15)
            ax.scatter(d, y, s=50, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            dy = 8 + 12 * (lvl - 1) if lvl > 0 else -(16 + 12 * (-lvl - 1))
            ax.annotate(t, (d, y), xytext=(0, dy), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(names)), names[::-1])
    for lbl, name in zip(ax.get_yticklabels(), names[::-1]):
        lbl.set_color(lanes[name][0]); lbl.set_fontweight("bold")
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlim(date(2020, 9, 1), date(2026, 3, 1))
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title("General-purpose models seed domain foundation models within months")
    ax.text(0.0, -0.1, "Dates = first public release (arXiv / blog). Not exhaustive.",
            transform=ax.transAxes, fontsize=8, color=INK2)
    save(fig, "05_domain_timeline.png")


# ---------------------------------------------------------------- 6. VLA size
def fig_vla():
    vla = [  # name, params (B), year
        ("RT-1 (2022)", 0.035), ("Octo-Base (2024)", 0.093), ("SmolVLA (2025)", 0.45),
        ("GR00T N1 (2025)", 2.2), ("π0 (2024)", 3.3), ("OpenVLA (2024)", 7),
        ("RT-2 PaLI-X (2023)", 55), ("PaLM-E (2023)", 562),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (n, p) in enumerate(vla):
        c = ORANGE if p <= 10 else INK2
        ax.barh(i, p, color=c, height=0.6)
        ax.text(p * 1.15, i, f"{p:g}B", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(vla)), [v[0] for v in vla])
    ax.set_xscale("log"); ax.set_xlim(0.01, 3000)
    ax.set_xlabel("Parameters (billions, log scale)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Robot policies: open VLAs settled at 0.5–7B to run on-robot")
    ax.text(0.0, -0.2, "Orange = open weights, deployable on a single GPU. Grey = closed, cloud-scale.",
            transform=ax.transAxes, fontsize=8, color=INK2)
    save(fig, "06_vla_size.png")


# ---------------------------------------------------------------- 7. labeling cost
def fig_label_cost():
    rows = [  # seconds per instance mask / box (Kirillov et al. 2023, SAM §4)
        ("COCO polygon mask (manual)", 91, INK2),
        ("SAM data engine — stage 1 start", 34, BLUE),
        ("SAM data engine — stage 1 end", 14, BLUE),
        ("Box via extreme clicks", 7, AQUA),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (n, s, c) in enumerate(rows):
        ax.barh(i, s, color=c, height=0.58)
        ax.text(s + 1.5, i, f"{s} s", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("Human seconds per object")
    ax.set_xlim(0, 105)
    ax.grid(axis="y", visible=False)
    ax.set_title("Model-in-the-loop labeling: 6.5× faster masks")
    ax.text(0.0, -0.3, "SA-1B data engine: 4.3M assisted masks → 5.9M semi-auto → 1.1B fully automatic.",
            transform=ax.transAxes, fontsize=8, color=INK2)
    save(fig, "07_labeling_cost.png")


if __name__ == "__main__":
    fig_model_scale(); fig_data_scale(); fig_imagenet(); fig_fid()
    fig_domain_timeline(); fig_vla(); fig_label_cost()
