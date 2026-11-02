"""Generate the illustrative example images for Lecture 01 by running real models.

Every panel is an actual model output (CLIP, OWLv2, SAM, DINOv2, Depth Anything V2, SmolVLM) on
openly licensed COCO val2017 photos and scikit-image sample data -- no figures copied from papers.

    pip install torch torchvision transformers pillow matplotlib scikit-image pooch
    python make_examples.py            # writes JPGs into ./figures/examples (~5 min on a laptop)
    python make_examples.py robotics   # just one figure
"""
import textwrap
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image, ImageEnhance, ImageOps
from transformers import AutoModel, SamModel, SamProcessor, pipeline

from make_figures import AQUA, BLUE, GREEN, INK, INK2, MAGENTA, ORANGE, RED, SURFACE, VIOLET, YELLOW

HERE = Path(__file__).parent
OUT = HERE / "figures" / "examples"
CACHE = HERE / ".cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
PALETTE = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# COCO val2017 images (Flickr, CC BY / CC BY-SA 2.0) -- attribution is listed in README.md.
COCO = {"street": 176037, "table": 541634, "dog": 261161, "bird": 407518}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
})


# ---------------------------------------------------------------- helpers
def coco_image(key):
    f = CACHE / f"{COCO[key]:012d}.jpg"
    if not f.exists():
        urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{f.name}", f)
    return Image.open(f).convert("RGB")


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (alpha,)


def overlay_masks(ax, image, masks, colors, alpha=0.55):
    ax.imshow(image)
    for m, c in zip(masks, colors):
        layer = np.zeros(m.shape + (4,))
        layer[m] = rgba(c, alpha)
        ax.imshow(layer)
        ax.contour(m, levels=[0.5], colors=[c], linewidths=1.2)


def draw_boxes(ax, dets, colors, fontsize=8, scores=True):
    for d, c in zip(dets, colors):
        b = d["box"]
        ax.add_patch(Rectangle((b["xmin"], b["ymin"]), b["xmax"] - b["xmin"], b["ymax"] - b["ymin"],
                               fill=False, edgecolor=c, linewidth=2))
        text = f'{d["label"]} {d["score"]:.2f}' if scores else d["label"]
        ax.text(b["xmin"], b["ymin"] - 3, text, fontsize=fontsize,
                color="white", bbox=dict(facecolor=c, edgecolor="none", pad=1.5))


def clean(ax, title=None):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, loc="left")


def save(fig, name):
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight", pil_kwargs={"quality": 85})
    plt.close(fig)
    print("wrote", OUT / name)


class Models:
    """Lazily loaded so each figure only pays for what it uses."""
    _cache = {}

    @classmethod
    def get(cls, name):
        if name not in cls._cache:
            cls._cache[name] = {
                "clip": lambda: pipeline("zero-shot-image-classification",
                                         model="openai/clip-vit-base-patch32", device=DEVICE),
                "owl": lambda: pipeline("zero-shot-object-detection",
                                        model="google/owlv2-base-patch16-ensemble", device=DEVICE),
                "depth": lambda: pipeline("depth-estimation",
                                          model="depth-anything/Depth-Anything-V2-Small-hf", device=DEVICE),
                "vlm": lambda: pipeline("image-text-to-text",
                                        model="HuggingFaceTB/SmolVLM-500M-Instruct", device=DEVICE),
                "sam": lambda: (SamProcessor.from_pretrained("facebook/sam-vit-base"),
                                SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE).eval()),
                "sam_auto": lambda: pipeline("mask-generation", model="facebook/sam-vit-base",
                                             device=DEVICE, points_per_batch=16),
                "dino": lambda: AutoModel.from_pretrained("facebook/dinov2-small").to(DEVICE).eval(),
            }[name]()
        return cls._cache[name]


@torch.no_grad()
def sam_predict(image, boxes=None, points=None, multimask=False):
    """Box prompts -> one mask per box; a single point prompt -> 3 candidate masks + IoU scores."""
    proc, model = Models.get("sam")
    kw = {"input_boxes": [boxes]} if boxes is not None else {"input_points": [[points]]}
    inputs = proc(image, return_tensors="pt", **kw)
    prompt = {k: inputs[k].float().to(DEVICE) for k in kw}  # MPS has no float64
    out = model(pixel_values=inputs["pixel_values"].to(DEVICE), multimask_output=multimask, **prompt)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"])[0]
    return masks.numpy().astype(bool), out.iou_scores.cpu().numpy()


def ask_vlm(image, question, max_new_tokens=60):
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": question}]}]
    out = Models.get("vlm")(text=msgs, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"][-1]["content"].strip()


def iou(a, b):
    ix = max(0, min(a["xmax"], b["xmax"]) - max(a["xmin"], b["xmin"]))
    iy = max(0, min(a["ymax"], b["ymax"]) - max(a["ymin"], b["ymin"]))
    area = lambda r: (r["xmax"] - r["xmin"]) * (r["ymax"] - r["ymin"])
    return ix * iy / (area(a) + area(b) - ix * iy + 1e-9)


def detect(image, queries, threshold=0.2, top_per_label=None):
    dets = Models.get("owl")(image, candidate_labels=queries, threshold=threshold)
    kept = []  # per-class non-maximum suppression: OWLv2 often returns near-duplicate boxes
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(k["label"] != d["label"] or iou(k["box"], d["box"]) < 0.5 for k in kept):
            kept.append(d)
    dets = kept
    if top_per_label:
        kept, count = [], {}
        for d in sorted(dets, key=lambda d: -d["score"]):
            if count.get(d["label"], 0) < top_per_label:
                kept.append(d); count[d["label"]] = count.get(d["label"], 0) + 1
        dets = kept
    return dets


def box_list(dets):
    return [[d["box"][k] for k in ("xmin", "ymin", "xmax", "ymax")] for d in dets]


def depth_map(image):
    return Models.get("depth")(image)["predicted_depth"].squeeze().cpu().numpy()  # relative inverse depth


# ---------------------------------------------------------------- Section 2: vision functions
def fig_vision_functions():
    img = coco_image("street")
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    a = axes.flat

    a[0].imshow(img); clean(a[0], "Input image")

    scenes = ["a city street at night", "a city street during the day", "a highway",
              "a bus station interior", "a country road", "a parking garage"]
    res = Models.get("clip")(img, candidate_labels=scenes)
    a[1].barh(range(len(res)), [r["score"] for r in res][::-1], color=BLUE, height=0.6)
    a[1].set_yticks(range(len(res)), [r["label"] for r in res][::-1], fontsize=9)
    a[1].set_xlim(0, 1); a[1].set_xlabel("probability", color=INK2)
    a[1].spines[["top", "right"]].set_visible(False)
    a[1].set_title("Classification — CLIP zero-shot", loc="left")

    queries = ["bus", "car", "pedestrian", "traffic light"]
    dets = detect(img, queries, threshold=0.18)
    col = {q: PALETTE[i] for i, q in enumerate(queries)}
    a[2].imshow(img); draw_boxes(a[2], dets, [col[d["label"]] for d in dets], fontsize=7, scores=False)
    clean(a[2], "Open-vocabulary detection — OWLv2")

    big = [d for d in dets if d["label"] != "traffic light"]
    masks, _ = sam_predict(img, boxes=box_list(big))
    overlay_masks(a[3], img, [m[0] for m in masks], [col[d["label"]] for d in big])
    clean(a[3], "Segmentation — SAM prompted by boxes")

    a[4].imshow(depth_map(img), cmap="magma"); clean(a[4], "Monocular depth — Depth Anything V2")

    cap = ask_vlm(img, "Describe this image in one short sentence.", 40)
    a[5].imshow(img, alpha=0.25); clean(a[5], "Captioning — SmolVLM (0.5B VLM)")
    a[5].text(0.5, 0.5, "\n".join(textwrap.wrap(f"“{cap}”", 34)), transform=a[5].transAxes,
              ha="center", va="center", fontsize=11, color=INK, style="italic")
    fig.suptitle("One photo, six vision functions — all from pretrained foundation models, no training",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "ex_vision_functions.jpg")


# ---------------------------------------------------------------- Section 3: representation learning
def dinov2_pca(img, size=448):
    """First 3 PCA components of DINOv2 patch features on the foreground (background found by PCA too)."""
    x = np.asarray(img.resize((size, size)), dtype=np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)[None].to(DEVICE)
    with torch.no_grad():
        feats = Models.get("dino")(pixel_values=x).last_hidden_state[0, 1:].cpu().numpy()
    g = size // 14
    f = feats - feats.mean(0)
    pcs = f @ np.linalg.svd(f, full_matrices=False)[2][:3].T
    border = np.zeros((g, g), bool); border[[0, -1], :] = border[:, [0, -1]] = True
    border = border.ravel()
    # Foreground = the (component, sign) split that touches the image border least.
    cands = [pcs[:, c] * s > 0 for c in range(3) for s in (1, -1)]
    cands = [m for m in cands if 0.05 < m.mean() < 0.5]
    fg = min(cands, key=lambda m: m[border].mean())
    ff = feats[fg] - feats[fg].mean(0)
    comps = ff @ np.linalg.svd(ff, full_matrices=False)[2][:3].T
    comps = (comps - comps.min(0)) / (comps.max(0) - comps.min(0))
    rgb = np.zeros((g * g, 3)); rgb[fg] = comps
    return Image.fromarray((rgb.reshape(g, g, 3) * 255).astype(np.uint8)).resize(img.size, Image.NEAREST)


def fig_representation():
    rng = np.random.default_rng(0)
    rows = [coco_image("dog"), coco_image("bird")]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4))
    for r, img in enumerate(rows):
        w, h = img.size
        v1 = ImageEnhance.Color(img.crop((w * 0.35, h * 0.0, w * 0.95, h * 0.85))).enhance(0.2)
        v2 = ImageOps.mirror(ImageEnhance.Brightness(img.crop((w * 0.2, h * 0.2, w, h))).enhance(1.4))

        sq = img.resize((224, 224)); arr = np.asarray(sq).copy()
        keep = rng.random((14, 14)) > 0.75
        for i in range(14):
            for j in range(14):
                if not keep[i, j]:
                    arr[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16] = 128

        panels = [(img, "Input"), (v1, "View 1 (crop + color)"), (v2, "View 2 (crop + flip + light)"),
                  (Image.fromarray(arr), "MAE input: 75% patches masked"), (dinov2_pca(img), "DINOv2 patch features (PCA→RGB)")]
        for c, (p, t) in enumerate(panels):
            axes[r, c].imshow(p); clean(axes[r, c], t if r == 0 else None)
    fig.text(0.215, 0.005, "Contrastive / self-distillation (SimCLR, DINO): two views of one image → same embedding",
             fontsize=9.5, color=INK2)
    fig.text(0.625, 0.005, "Masked modeling (MAE)", fontsize=9.5, color=INK2)
    fig.text(0.81, 0.005, "Emergent parts — no labels used", fontsize=9.5, color=INK2)
    fig.suptitle("Self-supervised representation learning: the training signals and what the features learn",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "ex_representation.jpg")


# ---------------------------------------------------------------- Section 4: promptable + VLM
def fig_sam_ambiguity():
    img = coco_image("dog")
    pt = [400, 140]  # a click on the dog's head
    masks, scores = sam_predict(img, points=[pt], multimask=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    axes[0].imshow(img); axes[0].scatter(*pt, s=160, c=RED, edgecolors="white", linewidths=2, marker="*")
    clean(axes[0], "Prompt: one click")
    order = np.argsort([m.sum() for m in masks[0]])
    for k, (ax, i) in enumerate(zip(axes[1:], order)):
        overlay_masks(ax, img, [masks[0][i]], [PALETTE[k]])
        ax.scatter(*pt, s=100, c=RED, edgecolors="white", linewidths=1.5, marker="*")
        clean(ax, f"Mask {k + 1}   (predicted IoU {scores[0, 0, i]:.2f})")
    fig.suptitle("SAM resolves an ambiguous prompt by returning several valid masks (sub-part → part → whole)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "ex_sam_ambiguity.jpg")


def fig_open_vocab_vlm():
    img = coco_image("dog")
    queries = ["a dog's tongue", "a red bench", "a leash", "a tree trunk", "a white chest patch"]
    dets = detect(img, queries, threshold=0.1, top_per_label=1)
    col = {q: PALETTE[i] for i, q in enumerate(queries)}
    qa = [(q, ask_vlm(img, q + " Answer briefly.", 30)) for q in
          ["What animal is this, and what breed might it be?", "What color is the bench?",
           "How many dogs are in the picture?", "Is the dog on a leash?"]]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].imshow(img); draw_boxes(axes[0], dets, [col[d["label"]] for d in dets])
    clean(axes[0], "OWLv2: free-text queries that are not COCO classes")
    clean(axes[1], "SmolVLM-500M: visual question answering")
    y = 0.95
    for q, a in qa:
        axes[1].text(0.02, y, "Q: " + q, transform=axes[1].transAxes, fontsize=10.5, color=INK2, va="top")
        a_txt = "\n".join(textwrap.wrap("A: " + a, 60))
        axes[1].text(0.02, y - 0.075, a_txt, transform=axes[1].transAxes, fontsize=10.5,
                     color=INK, va="top", fontweight="bold")
        y -= 0.13 + 0.06 * a_txt.count("\n") + 0.06
    axes[1].set_facecolor(SURFACE)
    fig.suptitle("Talking to vision models: text in, boxes or answers out", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "ex_open_vocab_vlm.jpg")


# ---------------------------------------------------------------- Section 5: diffusion
def fig_diffusion():
    img = coco_image("bird")
    x0 = np.asarray(img.resize((256, 256)), dtype=np.float32) / 127.5 - 1
    eps = np.random.default_rng(0).standard_normal(x0.shape).astype(np.float32)
    ts = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    fig, axes = plt.subplots(1, len(ts), figsize=(16, 3.2))
    for ax, t in zip(axes, ts):
        abar = np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2  # cosine schedule (Nichol & Dhariwal 2021)
        xt = np.sqrt(abar) * x0 + np.sqrt(1 - abar) * eps
        ax.imshow(np.clip((xt + 1) / 2, 0, 1)); clean(ax, f"t = {t:.1f}   ᾱ = {abar:.2f}")
    fig.text(0.02, 0.0, "Forward process (fixed):  x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε   — add noise  →",
             fontsize=10.5, color=INK2)
    fig.text(0.98, 0.0, "←  Reverse process (learned): a network predicts ε (or velocity) and denoises step by step",
             fontsize=10.5, color=INK2, ha="right", va="bottom", transform=fig.transFigure)
    fig.suptitle("Diffusion models generate images by learning to undo noise", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "ex_diffusion.jpg")


# ---------------------------------------------------------------- Section 6: domains
def fig_healthcare():
    from skimage import data
    cells = data.human_mitosis()[256:512, 0:256]
    cells_rgb = Image.fromarray(cells).convert("RGB").resize((512, 512), Image.BILINEAR)
    auto = Models.get("sam_auto")(cells_rgb, points_per_side=32, pred_iou_thresh=0.8)
    nuclei = [np.asarray(m, dtype=bool) for m in auto["masks"]]
    nuclei = [m for m in nuclei if m.sum() < 0.02 * m.size]  # drop background-sized masks

    ct = data.brain()[5].astype(np.float32)
    ct = np.clip((ct - ct.min()) / (np.percentile(ct, 99.5) - ct.min()), 0, 1)
    ct_rgb = Image.fromarray((ct * 255).astype(np.uint8)).convert("RGB").resize((512, 512))
    pt, box = [256, 300], [125, 95, 390, 450]
    pmasks, pscores = sam_predict(ct_rgb, points=[pt], multimask=True)
    bmasks, bscores = sam_predict(ct_rgb, boxes=[box], multimask=True)
    pbest, bbest = int(np.argmax(pscores[0, 0])), int(np.argmax(bscores[0, 0]))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    axes[0].imshow(cells_rgb); clean(axes[0], "Fluorescence microscopy (nuclei)")
    overlay_masks(axes[1], cells_rgb, nuclei, [PALETTE[i % 8] for i in range(len(nuclei))], alpha=0.5)
    clean(axes[1], f"SAM automatic: {len(nuclei)} nuclei, zero-shot")
    overlay_masks(axes[2], ct_rgb, [pmasks[0][pbest]], [ORANGE])
    axes[2].scatter(*pt, s=160, c=RED, edgecolors="white", linewidths=2, marker="*")
    clean(axes[2], "Head CT, click → whole head + table ✗")
    overlay_masks(axes[3], ct_rgb, [bmasks[0][bbest]], [AQUA])
    axes[3].add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, edgecolor=RED, lw=2))
    clean(axes[3], "Head CT, box → intracranial region ✓")
    fig.suptitle("Healthcare: SAM segments nuclei zero-shot, but on CT the prompt decides the answer — "
                 "MedSAM fine-tunes SAM with box prompts", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "ex_healthcare.jpg")


def fig_robotics():
    img = coco_image("table")
    instr = {"the bowl of soup": "Push the bowl of soup", "chopsticks": "Pick up the chopsticks",
             "the iced tea with a straw": "Hand me the iced tea"}
    dets = detect(img, list(instr), threshold=0.05, top_per_label=1)
    masks, _ = sam_predict(img, boxes=box_list(dets))
    fig, axes = plt.subplots(1, len(dets), figsize=(5.2 * len(dets), 5))
    for k, (ax, d, m) in enumerate(zip(np.atleast_1d(axes), dets, masks)):
        m = m[0]
        overlay_masks(ax, img, [m], [PALETTE[k]])
        ys, xs = np.nonzero(m)
        ax.scatter(xs.mean(), ys.mean(), s=220, marker="+", c="white", linewidths=3)
        ax.scatter(xs.mean(), ys.mean(), s=150, marker="+", c=INK, linewidths=1.5)
        clean(ax, f'“{instr[d["label"]]}”')
        ax.set_xlabel(f'target: {d["label"]}  (score {d["score"]:.2f})', color=INK2)
    fig.suptitle("Robotics: language → grounded object mask → target point "
                 "(the perception front-end of language-conditioned manipulation)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "ex_robotics.jpg")


def fig_driving():
    img = coco_image("street")
    dets = detect(img, ["bus", "car", "pedestrian"], threshold=0.15)  # road users only
    depth = depth_map(img)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    for d in dets:
        b = d["box"]
        d["near"] = float(np.median(depth[int(b["ymin"]):int(b["ymax"]) + 1, int(b["xmin"]):int(b["xmax"]) + 1]))
    question = "What should a driver watch out for in this scene? Answer in one sentence."
    hazards = ask_vlm(img, question, 40)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    axes[0].imshow(img)
    dets = sorted(dets, key=lambda d: -d["near"])[:6]  # the six nearest objects
    for rank, d in enumerate(dets, 1):
        d["label"] = f'#{rank} {d["label"]}'
    draw_boxes(axes[0], dets, [PALETTE[0] if i < 3 else INK2 for i in range(len(dets))], fontsize=7, scores=False)
    clean(axes[0], "Detection, ranked by depth (#1 = nearest)")
    axes[1].imshow(depth, cmap="magma"); clean(axes[1], "Relative depth (bright = near)")
    clean(axes[2], "VLM scene reasoning (SmolVLM-500M)")
    axes[2].text(0.03, 0.9, "\n".join(textwrap.wrap("Q: " + question, 44)), transform=axes[2].transAxes,
                 fontsize=10.5, color=INK2, va="top")
    axes[2].text(0.03, 0.7, "\n".join(textwrap.wrap("A: " + hazards, 44)), transform=axes[2].transAxes,
                 fontsize=10.5, color=INK, va="top", fontweight="bold")
    axes[2].text(0.03, 0.4, "COCO ground truth: bus, cars, people,\ntraffic lights, backpacks.\n\n"
                 "Small VLMs often name objects that are not\nthere — check answers against grounded\n"
                 "detections before acting on them.", transform=axes[2].transAxes, fontsize=9.5,
                 color=INK2, va="top")
    fig.suptitle("Self-driving: detection, geometry and language reasoning from one camera frame",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "ex_driving.jpg")


FIGURES = {f.__name__[4:]: f for f in [
    fig_vision_functions, fig_representation, fig_sam_ambiguity, fig_open_vocab_vlm,
    fig_diffusion, fig_healthcare, fig_robotics, fig_driving]}

if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or FIGURES:  # e.g. `python make_examples.py healthcare driving`
        FIGURES[name]()
