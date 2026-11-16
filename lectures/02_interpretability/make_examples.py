"""Generate the example figures for Lecture 02 (Interpretability) by running real models.

Every panel is an actual model output or a tiny model trained here; nothing is copied from papers.
Models (all small, laptop-friendly): ResNet-50 and ViT-Tiny ImageNet classifiers, CLIP ViT-B/32,
DINOv2-S, SmolVLM-256M, GPT-2 small.

The representation figures (units, embedding, cka) use the take-home's 1,000-image COCO subset: run
projects/interp_takehome/make_data.py first.

    pip install torch torchvision transformers pillow matplotlib scikit-learn
    python make_examples.py              # writes JPGs into ./figures (~6 min on a laptop)
    python make_examples.py llm_patching # just one figure
"""
import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
LEC1 = HERE.parent / "01_foundation_models"
sys.path.insert(0, str(LEC1))
from make_figures import AQUA, BLUE, GRID, INK, INK2, ORANGE, RED, VIOLET  # noqa: E402


def _lecture1_helpers():
    """Load Lecture 01's make_examples.py by path (same file name as this script)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("lecture01_examples", LEC1 / "make_examples.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_l1 = _lecture1_helpers()
clean, coco_image = _l1.clean, _l1.coco_image   # shared COCO cache + axis helper

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

plt.rcParams.update({"axes.grid": False, "axes.titlelocation": "left", "axes.titlesize": 11})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)
    print("wrote", OUT / name)


def heat(ax, image, sal, title, cmap="inferno"):
    sal = np.asarray(sal, dtype=float)
    lo, hi = np.percentile(sal, [1, 99])          # clip outliers so faint maps stay visible
    sal = np.clip((sal - lo) / (hi - lo + 1e-8), 0, 1)
    ax.imshow(image)
    ax.imshow(np.array(Image.fromarray((sal * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)),
              cmap=cmap, alpha=0.6)
    clean(ax, title)


# ================================================================ 0. representation basics (uses the take-home data)
TAKEHOME = HERE.parents[1] / "projects" / "interp_takehome"
TH_DATA = TAKEHOME / "data" / "coco1k"


def _takehome():
    """Import the take-home's interp_utils and load its COCO subset (run its make_data.py first)."""
    if not (TH_DATA / "labels.json").exists():
        raise SystemExit(f"{TH_DATA} not found: run projects/interp_takehome/make_data.py first")
    sys.path.insert(0, str(TAKEHOME))
    import interp_utils as iu
    return iu, iu.load_coco(TH_DATA)["images"]


# hand-picked from projects/interp_takehome/q1_units.py (best-IoU units per stage), with a short tag
UNITS = [("layer1", 101, "green"), ("layer1", 4, "orange / yellow"), ("layer2", 51, "text, signs"),
         ("layer3", 434, "street vehicles"), ("layer4", 754, "cats"), ("layer4", 797, "bananas")]
STAGE_IDX = {"layer1": 3, "layer2": 7, "layer3": 13, "layer4": 16}
STAGE_GRID = {"layer1": 56, "layer2": 28, "layer3": 14, "layer4": 7}
STAGE_CROP = {"layer1": 48, "layer2": 80, "layer3": 128, "layer4": 224}
STAGE_RF = {"layer1": 35, "layer2": 91, "layer3": 267, "layer4": 427}


@torch.no_grad()
def fig_units():
    iu, recs = _takehome()
    enc = iu.Encoder("resnet50")
    peak = {u: [] for u in UNITS}
    for i in range(0, len(recs), 32):
        acts = enc.layers(enc.pixels([iu.open_image(r) for r in recs[i:i + 32]]))
        for u in UNITS:
            a = acts[STAGE_IDX[u[0]]][:, u[1]].flatten(1)
            v, pos = a.max(1)
            peak[u].append(torch.stack([v, pos.float()], 1).cpu())
    k = 5
    fig = plt.figure(figsize=(10, 6.6))
    gs = fig.add_gridspec(len(UNITS), 2 + k, width_ratios=[3.2, 0.7] + [1] * k, wspace=0.06, hspace=0.1)
    w = enc.model.conv1.weight.detach().cpu()
    w = (w - w.amin((1, 2, 3), keepdim=True)) / (w.amax((1, 2, 3), keepdim=True) - w.amin((1, 2, 3), keepdim=True))
    grid = torch.ones(3, 8 * 8, 8 * 8)
    for j, f in enumerate(w):
        r, c = divmod(j, 8)
        grid[:, r * 8:r * 8 + 7, c * 8:c * 8 + 7] = f
    ax = fig.add_subplot(gs[1:5, 0])
    ax.imshow(grid.permute(1, 2, 0).numpy(), interpolation="nearest")
    clean(ax, "conv1: all 64 filters (7x7 RGB)")
    for row, u in enumerate(UNITS):
        st, unit, tag = u
        pk = torch.cat(peak[u]).numpy()
        g = STAGE_GRID[st]
        for col, n in enumerate(np.argsort(-pk[:, 0])[:k]):
            img = np.asarray(iu.square_crop(iu.open_image(recs[n])))
            yy, xx = divmod(int(pk[n, 1]), g)
            cy, cx = (yy + 0.5) * 224 / g, (xx + 0.5) * 224 / g
            h = STAGE_CROP[st] // 2
            y0, x0 = int(np.clip(cy - h, 0, 224 - 2 * h)), int(np.clip(cx - h, 0, 224 - 2 * h))
            ax = fig.add_subplot(gs[row, 2 + col])
            ax.imshow(img[y0:y0 + 2 * h, x0:x0 + 2 * h]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if col == 0:
                ax.set_ylabel(f"{st} #{unit}\n{tag}", fontsize=8, color=INK)
    fig.suptitle("ResNet-50 (ImageNet): first-layer filters and top dataset examples of single units",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT / "ex_units.jpg", dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / "ex_units.jpg")


@torch.no_grad()
def fig_embedding():
    from collections import Counter
    from sklearn.manifold import TSNE
    iu, recs = _takehome()
    keep = [r for r in recs if r["dominant_frac"] >= 0.05]
    cnt = Counter(r["dominant"] for r in keep)
    classes = sorted(c for c, n in cnt.items() if n >= 20)
    keep = [r for r in keep if r["dominant"] in classes]
    y = np.array([classes.index(r["dominant"]) for r in keep])
    train, _ = iu.split_ids(keep)
    tr = np.array([r["id"] in train for r in keep])
    pal = [BLUE, ORANGE, AQUA, "#eda100", "#e24a94", "#4f9e2a", VIOLET, RED, "#52514e"]
    models = [("clip", "CLIP (trained)"), ("clip_random", "same architecture, random weights")]
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 13, height_ratios=[3.2, 1], width_ratios=[1] * 6 + [0.3] + [1] * 6,
                          hspace=0.25, wspace=0.08)
    query = next(i for i in np.where(~tr)[0] if classes[y[i]] == "animal")
    for m, (name, title) in enumerate(models):
        enc = iu.Encoder(name)
        Z = iu.run_batched(lambda ims: enc.embed(enc.pixels(ims)), keep, verbose=False)
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        nn = np.argsort(-(Z[~tr] @ Z[tr].T), 1)[:, :10]
        pred = np.array([Counter(y[tr][r]).most_common(1)[0][0] for r in nn])
        from sklearn.metrics import balanced_accuracy_score
        acc = balanced_accuracy_score(y[~tr], pred)
        P = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z)
        ax = fig.add_subplot(gs[0, 7 * m:7 * m + 6])
        for c in range(len(classes)):
            ax.scatter(*P[y == c].T, s=5, color=pal[c], label=classes[c])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\n10-NN balanced accuracy {acc:.2f} (chance {1 / len(classes):.2f})", fontsize=10)
        if m == 1:
            ax.legend(fontsize=7, markerscale=2.5, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
        sims = Z @ Z[query]
        sims[query] = -np.inf
        for col, j in enumerate([query] + list(np.argsort(-sims)[:5])):
            ax = fig.add_subplot(gs[1, 7 * m + col])
            ax.imshow(iu.square_crop(iu.open_image(keep[j]), 112)); ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel("query" if col == 0 else classes[y[j]], fontsize=7)
            if col == 0:
                ax.set_ylabel("CLIP" if m == 0 else "random", fontsize=8)
        del enc
    fig.suptitle(f"Embedding space of {len(keep)} COCO images, coloured by the largest object's supercategory "
                 "(t-SNE), and nearest neighbours of one query", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT / "ex_embedding.jpg", dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / "ex_embedding.jpg")


@torch.no_grad()
def fig_cka(n=500):
    iu, recs = _takehome()
    recs = recs[:n]
    feats = {}
    for name in ("dinov2", "clip"):
        enc = iu.Encoder(name)
        feats[name] = iu.run_batched(lambda ims: enc.pooled_layers(enc.pixels(ims), "mean"), recs, verbose=False)
        del enc
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    fig.subplots_adjust(wspace=0.35)
    for ax, (a, b) in zip(axes, [("dinov2", "dinov2"), ("clip", "dinov2")]):
        K = np.array([[iu.linear_cka(xa, xb) for xb in feats[b]] for xa in feats[a]])
        im = ax.imshow(K, origin="lower", vmin=0, vmax=1, cmap="viridis")
        ax.set_xlabel(f"{b} layer"); ax.set_ylabel(f"{a} layer")
        ax.set_xticks(range(0, 13, 2)); ax.set_yticks(range(0, 13, 2))
        ax.set_title(f"{a} vs {b}", fontsize=10)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.85, label="linear CKA")
    fig.suptitle(f"Linear CKA between layers (mean-pooled tokens, {n} COCO images)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.savefig(OUT / "ex_cka.jpg", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / "ex_cka.jpg")


@torch.no_grad()
def fig_patch_pca(size=448):
    from transformers import AutoImageProcessor, AutoModel
    sys.path.insert(0, str(TAKEHOME))
    from interp_utils import square_crop
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(DEVICE).eval()
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    keys = ["dog", "bird", "street", "table"]
    fig, axes = plt.subplots(3, len(keys), figsize=(3.0 * len(keys), 9.0))
    for col, key in enumerate(keys):
        img = square_crop(coco_image(key), size)
        x = (torch.from_numpy(np.asarray(img, np.float32) / 255) - torch.tensor(proc.image_mean)) / \
            torch.tensor(proc.image_std)
        tok = model(pixel_values=x.permute(2, 0, 1)[None].to(DEVICE)).last_hidden_state[0, 1:].float().cpu()
        g = int(len(tok) ** 0.5)
        tok = tok - tok.mean(0)
        _, _, V = torch.linalg.svd(tok, full_matrices=False)
        pcs = (tok @ V[:3].T).reshape(g, g, 3)
        pc1 = pcs[..., 0]
        if torch.cat([pc1[0], pc1[-1], pc1[:, 0], pc1[:, -1]]).mean() > 0:
            pc1 = -pc1
        rgb = (pcs - pcs.amin((0, 1))) / (pcs.amax((0, 1)) - pcs.amin((0, 1)))
        axes[0, col].imshow(img); clean(axes[0, col], key)
        axes[1, col].imshow(rgb.numpy(), interpolation="nearest"); clean(axes[1, col], "PCs 1-3 as RGB")
        axes[2, col].imshow(img)
        axes[2, col].imshow(np.array(Image.fromarray(((pc1 > 0).numpy() * 255).astype(np.uint8)).resize(
            img.size, Image.NEAREST)), cmap="gray", alpha=0.55)
        clean(axes[2, col], "PC 1 > 0 (bright)")
    fig.suptitle(f"DINOv2-S patch features ({g}x{g} patches): PCA within each image", x=0.01, ha="left",
                 fontsize=11, color=INK)
    save(fig, "ex_patch_pca.jpg")


# ================================================================ 1. saliency + sanity check
VIT = "WinKawaks/vit-tiny-patch16-224"   # ImageNet-1k ViT-Ti/16


def vit_saliency(model, proc, image, target=None, ig_steps=32):
    """Vanilla gradient, gradient x input, integrated gradients and attention rollout for one image."""
    x = proc(images=image, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        out = model(pixel_values=x, output_attentions=True)
    target = out.logits[0].argmax().item() if target is None else target

    def grad_at(inp):
        inp = inp.clone().requires_grad_(True)
        model(pixel_values=inp).logits[:, target].sum().backward()
        return inp.grad

    g = grad_at(x)[0]
    vanilla = g.abs().amax(0)
    gxi = (g * x[0]).sum(0).abs()
    # Integrated gradients from a black (all-zero after normalisation = mean-grey) baseline.
    alphas = torch.linspace(1 / ig_steps, 1, ig_steps).view(-1, 1, 1, 1)
    ig = (grad_at(alphas * x).mean(0) * x[0]).sum(0).abs()
    # Attention rollout (Abnar & Zuidema 2020): average heads, add identity, multiply through layers.
    roll = torch.eye(out.attentions[0].shape[-1])
    for a in out.attentions:
        a = a[0].mean(0)
        a = (a + torch.eye(a.shape[0])) / 2
        roll = (a / a.sum(-1, keepdim=True)) @ roll
    n = int((roll.shape[0] - 1) ** 0.5)
    rollout = roll[0, 1:].reshape(n, n)
    label = model.config.id2label[target].split(",")[0]
    return target, label, {"Gradient": vanilla, "Gradient × input": gxi, "Integrated gradients": ig,
                   "Attention rollout": rollout}


def vit_blocks(model):
    """Transformer blocks of an HF ViT classifier (attribute name changed across transformers versions)."""
    vit = model.vit
    return vit.layers if hasattr(vit, "layers") else vit.encoder.layer


def randomize_top(model, n_blocks):
    """Adebayo et al. (2018) cascading randomisation: re-initialise the classifier + top n blocks."""
    m = copy.deepcopy(model)
    mods = [m.classifier] + list(vit_blocks(m)[::-1][:n_blocks])
    for mod in mods:
        for name, p in mod.named_parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, std=0.02)
            elif "layernorm" in name or "norm" in name.split(".")[-2:][0]:
                torch.nn.init.ones_(p) if name.endswith("weight") else torch.nn.init.zeros_(p)
            else:
                torch.nn.init.zeros_(p)
    return m


def fig_saliency_sanity():
    from transformers import AutoImageProcessor, ViTForImageClassification
    proc = AutoImageProcessor.from_pretrained(VIT)
    model = ViTForImageClassification.from_pretrained(VIT, attn_implementation="eager").eval()
    image = coco_image("dog").resize((224, 224))
    rand = randomize_top(model, n_blocks=12).eval()

    target, label, maps = vit_saliency(model, proc, image)
    _, _, rmaps = vit_saliency(rand, proc, image, target=target)   # same class, random weights

    from scipy.stats import spearmanr
    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    rows = [(f'trained ViT-Ti\npredicts "{label}"', maps),
            ("same ViT, head + all 12\nblocks re-initialised", rmaps)]
    for row, (name, mp) in enumerate(rows):
        axes[row, 0].imshow(image); clean(axes[row, 0])
        axes[row, 0].set_ylabel(name, fontsize=10, color=INK)
        for j, (k, v) in enumerate(mp.items(), start=1):
            v = v.detach().numpy()
            if row == 0:
                heat(axes[row, j], image, v, k)
            else:
                rho = spearmanr(maps[k].detach().numpy().ravel(), v.ravel()).statistic
                heat(axes[row, j], image, v, None)
                axes[row, j].set_title(f"rank corr. with trained: {rho:.2f}", loc="left", fontsize=9,
                                       fontweight="normal", color=INK2)
    axes[0, 0].set_title("input", loc="left")
    fig.suptitle("Sanity check (Adebayo et al. 2018): a map that survives weight randomisation is explaining the "
                 "image, not the model", x=0.01, ha="left", fontsize=11, color=INK2)
    save(fig, "ex_saliency_sanity.jpg")


# ================================================================ 2. CLIP: where it looks & typographic attack
CLIP = "openai/clip-vit-base-patch32"


def text_image(word, size=(480, 360)):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    font = _font(110)
    tw = d.textlength(word, font=font)
    d.text(((size[0] - tw) / 2, size[1] / 2 - 70), word, fill="black", font=font)
    return img


def _font(px):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", px)
    except OSError:
        return ImageFont.load_default(size=px)


def add_sticker(img, word, frac=0.35):
    """Typographic attack (Goh et al. 2021): paste a large hand-label with a word onto the photo."""
    a = img.copy()
    d = ImageDraw.Draw(a)
    w, h = a.size
    f = _font(int(h * frac * 0.6))
    x0, y0 = int(w * 0.05), int(h * 0.55)
    d.rectangle((x0, y0, x0 + d.textlength(word, font=f) + 30, y0 + int(h * frac * 0.8)),
                fill="white", outline="black", width=3)
    d.text((x0 + 15, y0 + 5), word, fill="black", font=f)
    return a


def fig_clip():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP).eval()
    proc = CLIPProcessor.from_pretrained(CLIP)

    def embed_text(texts):  # CLIP joint-space embeddings, L2-normalised
        x = proc(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            t = model.get_text_features(**x)
        t = getattr(t, "pooler_output", t)
        return F.normalize(t, dim=-1)

    def embed_images(imgs):
        x = proc(images=imgs, return_tensors="pt")
        with torch.no_grad():
            v = model.get_image_features(**x)
        v = getattr(v, "pooler_output", v)
        return F.normalize(v, dim=-1)

    def probs(img, labels):
        t = embed_text([f"a photo of a {c}" for c in labels])
        return (100 * embed_images([img]) @ t.T).softmax(-1)[0].numpy()

    bus = coco_image("street")
    labels = ["bus", "car", "boat", "airplane"]
    shown = {"photo": bus, 'photo + "boat" label': add_sticker(bus, "boat"),
             'just the word "boat"': text_image("boat")}
    rows = {k: probs(v, labels) for k, v in shown.items()}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.3, 1, 1.3]})
    axes[0].imshow(shown['photo + "boat" label']); clean(axes[0], 'Typographic attack: a "boat" label')
    axes[1].imshow(shown['just the word "boat"']); clean(axes[1], "No picture, only text")
    ax = axes[2]
    y = np.arange(len(labels))
    for k, ((name, p), c) in enumerate(zip(rows.items(), [BLUE, ORANGE, VIOLET])):
        ax.barh(y + (k - 1) * 0.27, p, 0.27, color=c, label=name)
        for yi, pi in zip(y, p):
            if pi > 0.05:
                ax.text(pi + 0.01, yi + (k - 1) * 0.27, f"{pi:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, 1.12)
    ax.set_xlabel('CLIP zero-shot probability ("a photo of a {label}")'); ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Zero-shot prediction")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.suptitle("CLIP ViT-B/32 reads text: the word alone is a confident 'boat'. Pasted on a clear bus photo, the "
                 "label only moves P(boat) a little — replicate famous demos before you trust them.",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    save(fig, "ex_clip_typographic.jpg")
    return rows


# ================================================================ 3. DINOv2 attention heads
def fig_dino_heads():
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small", attn_implementation="eager").eval()
    imgs = [coco_image("dog"), coco_image("bird"), coco_image("street")]
    fig, axes = plt.subplots(len(imgs), 5, figsize=(12.5, 7.4))
    for r, img in enumerate(imgs):
        img = img.resize((448, 448))
        x = proc(images=img, return_tensors="pt", do_resize=False, do_center_crop=False)
        with torch.no_grad():
            att = model(**x, output_attentions=True).attentions[-1][0, :, 0, 1:]   # heads x patches (CLS row)
        n = int(att.shape[-1] ** 0.5)
        axes[r, 0].imshow(img); clean(axes[r, 0], "input" if r == 0 else None)
        for j, hd in enumerate([0, 1, 2, 4]):
            heat(axes[r, j + 1], img, att[hd].reshape(n, n).numpy(), f"last layer, head {hd}" if r == 0 else None)
    fig.suptitle("DINOv2-S, never trained with labels: different [CLS] attention heads in the last layer pick out "
                 "the main object and different parts of it",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    save(fig, "ex_dino_heads.jpg")


# ================================================================ 4. VLM: logit lens on image tokens + image ablation
VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"


def load_vlm():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(VLM)
    # eager attention so every layer receives an explicit 4-D mask we can edit (attention knockout)
    model = AutoModelForImageTextToText.from_pretrained(VLM, dtype=torch.float32,
                                                        attn_implementation="eager").eval()
    return proc, model


def vlm_inputs(proc, image, question):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    return proc(text=prompt, images=[image], return_tensors="pt", do_image_splitting=False)


def fig_vlm_logit_lens():
    proc, model = load_vlm()
    tok = proc.tokenizer
    lm = model.model.text_model
    image = coco_image("dog")
    x = vlm_inputs(proc, image, "What is in this image?")
    img_pos = (x["input_ids"][0] == tok.convert_tokens_to_ids("<image>")).nonzero()[:, 0]
    with torch.no_grad():
        hs = model(**x, output_hidden_states=True).hidden_states   # embeddings + 30 layers

    def decode(h):
        return model.lm_head(lm.norm(h))

    words = ["dog", "bench", "grass", "wood", "park"]
    word_ids = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in words}
    # Fraction of image tokens whose top-20 decoded tokens contain the word, per layer.
    curves = {w: [] for w in words}
    for h in hs:
        top = decode(h[0, img_pos]).topk(20, dim=-1).indices
        for w, i in word_ids.items():
            curves[w].append((top == i).any(-1).float().mean().item())

    layer = int(np.argmax(curves["dog"]))
    top1 = decode(hs[layer][0, img_pos]).argmax(-1)
    n = int(len(img_pos) ** 0.5)
    grid = [tok.decode([t]).strip()[:7] for t in top1.tolist()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.05, 1]})
    seen = image.resize((512, 512))
    axes[0].imshow(seen, extent=(0, n, n, 0))
    for k, wd in enumerate(grid):
        r, c = divmod(k, n)
        hit = wd.lower().startswith(("dog", "pupp", "mount", "bern", "bulld", "german", "hound"))  # dog / breed words
        axes[0].text(c + 0.5, r + 0.5, wd, ha="center", va="center", fontsize=7.5,
                     color="white", fontweight="bold" if hit else "normal",
                     bbox=dict(facecolor=ORANGE if hit else "black", alpha=0.75 if hit else 0.45,
                               edgecolor="none", pad=1))
    clean(axes[0], f"Top decoded token of each image token at LLM layer {layer}")
    ax = axes[1]
    for w, c in zip(words, [ORANGE, BLUE, AQUA, VIOLET, INK2]):
        ax.plot(range(len(hs)), curves[w], color=c, lw=2, label=f'"{w}"')
    ax.set_xlabel("LLM layer (0 = projector output)"); ax.set_ylabel("share of 64 image tokens with the word in top-20")
    ax.set_title("Object words appear only in the last few layers")
    ax.legend(); ax.grid(color=GRID)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.suptitle("Logit lens on SmolVLM-256M: decode each image token's hidden state with the LLM's final norm + "
                 "unembedding. Orange = dog / breed words.",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    save(fig, "ex_vlm_logit_lens.jpg")


def fig_vlm_knockout(window=3):
    """Where does the LLM read the image? Block text->image attention in one window of layers at a time."""
    proc, model = load_vlm()
    tok = proc.tokenizer
    layers = model.model.text_model.layers
    img_id = tok.convert_tokens_to_ids("<image>")
    yes = [tok.encode(s, add_special_tokens=False)[0] for s in ("Yes", " Yes")]
    no = [tok.encode(s, add_special_tokens=False)[0] for s in ("No", " No")]
    # (image, object) pairs; the first group is present in COCO val2017 GT, the second is absent.
    present = [("street", "bus"), ("street", "person"), ("dog", "dog"), ("dog", "bench"),
               ("table", "broccoli"), ("table", "bowl")]
    absent = [("street", "bicycle"), ("dog", "cat"), ("table", "pizza")]

    def p_yes(image, obj, blocked=()):
        x = vlm_inputs(proc, image, f"Is there a {obj} in the image? Answer yes or no.")
        img = x["input_ids"][0] == img_id
        q, k = (~img).nonzero()[:, 0], img.nonzero()[:, 0]     # text queries, image keys

        def pre(mod, args, kwargs):
            m = kwargs["attention_mask"].clone()
            m[:, :, q[:, None], k[None, :]] = torch.finfo(m.dtype).min
            kwargs["attention_mask"] = m
            return args, kwargs
        hooks = [layers[i].self_attn.register_forward_pre_hook(pre, with_kwargs=True) for i in blocked]
        try:
            with torch.no_grad():
                pr = model(**x).logits[0, -1].softmax(-1)
        finally:
            for h in hooks:
                h.remove()
        return (pr[yes].sum() / (pr[yes].sum() + pr[no].sum())).item()

    L = len(layers)
    starts = list(range(0, L, window))
    pairs = present + absent
    table = np.array([[p_yes(coco_image(i), o)] + [p_yes(coco_image(i), o, range(s, min(L, s + window)))
                                                   for s in starts] for i, o in pairs])

    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(table, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            ax.text(c, r, f"{table[r, c]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if table[r, c] > 0.55 else INK)
    ax.axvline(0.5, color=INK, lw=1.5)
    ax.axhline(len(present) - 0.5, color=INK, lw=1.5)
    ax.set_xticks(range(len(starts) + 1), ["none"] + [f"{s}–{min(L, s + window) - 1}" for s in starts])
    ax.set_xlabel("LLM layers where text→image attention is blocked")
    ax.set_yticks(range(len(pairs)), [f"{o}  ({i})" + ("" if (i, o) in present else "  · absent")
                                      for i, o in pairs])
    ax.set_title('P(yes) for "Is there a {object} in the image?"')
    fig.colorbar(im, ax=ax, fraction=0.025, label="P(yes)")
    fig.suptitle(f"Attention knockout in SmolVLM-256M ({L} LLM layers). Cutting the image off in one narrow band of "
                 "middle layers erases the evidence;\ncutting it off anywhere else barely matters. "
                 "(9 questions on 3 COCO images — a demo, not a study.)",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    save(fig, "ex_vlm_knockout.jpg")
    return table


# ================================================================ 5. LLM: logit lens + activation patching on GPT-2
def load_gpt2():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    return tok, model


@torch.no_grad()
def fig_llm_logit_lens():
    tok, model = load_gpt2()
    prompt = "The Eiffel Tower is located in the city of"
    x = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        hs = model(**x, output_hidden_states=True).hidden_states
    toks = [tok.decode([t]) for t in x["input_ids"][0]]
    L, T = len(hs), len(toks)
    top = np.empty((L, T), dtype=object)
    prob = np.zeros((L, T))
    for l, h in enumerate(hs):
        h = h[0] if l == L - 1 else model.transformer.ln_f(h[0])   # last entry already has ln_f applied
        p = model.lm_head(h).softmax(-1)
        prob[l] = p.max(-1).values.numpy()
        top[l] = [tok.decode([i]) for i in p.argmax(-1).tolist()]
    paris = tok.encode(" Paris")[0]
    p_paris = [(model.lm_head(h[0, -1] if l == L - 1 else model.transformer.ln_f(h[0, -1]))
                .softmax(-1)[paris]).item() for l, h in enumerate(hs)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [2.3, 1]})
    ax = axes[0]
    ax.imshow(prob, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    for l in range(L):
        for t in range(T):
            ax.text(t, l, repr(top[l, t])[1:-1][:8], ha="center", va="center", fontsize=7,
                    color="white" if prob[l, t] > 0.5 else INK)
    ax.set_xticks(range(T), [repr(s)[1:-1] for s in toks], rotation=30, ha="right")
    ax.set_yticks(range(L), ["emb"] + [str(i) for i in range(1, L)])
    ax.set_ylabel("layer"); ax.set_title("GPT-2 logit lens: top next-token guess at every layer and position")
    ax = axes[1]
    ax.plot(range(L), p_paris, color=ORANGE, lw=2, marker="o", ms=3)
    ax.set_xlabel("layer"); ax.set_ylabel('P(" Paris") at the last position'); ax.grid(color=GRID)
    ax.set_title('When does " Paris" appear?')
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "ex_llm_logit_lens.jpg")


IOI_TEMPLATE = "When {A} and {B} went to the store, {B} gave a drink to"
NAMES = [("Mary", "John"), ("Alice", "Bob"), ("Sarah", "Tom"), ("Emma", "James"), ("Anna", "David"),
         ("Laura", "Mark"), ("Lisa", "Paul"), ("Kate", "Mike"), ("Rachel", "Daniel"), ("Jessica", "Kevin"),
         ("Amy", "Ryan"), ("Grace", "Eric")]


def ioi_batch(tok):
    """Indirect-object identification (Wang et al. 2023). Clean: '...A and B..., B gave a drink to' -> A.
    Corrupted: swap the names so the correct answer becomes B. Every name is one GPT-2 token, so all prompts
    align position by position (needed for patching)."""
    clean, corrupt, io_ids, s_ids = [], [], [], []
    for a, b in NAMES + [(b, a) for a, b in NAMES]:
        clean.append(IOI_TEMPLATE.format(A=a, B=b))
        corrupt.append(IOI_TEMPLATE.format(A=b, B=a))
        io_ids.append(tok.encode(" " + a)[0]); s_ids.append(tok.encode(" " + b)[0])
    c, k = tok(clean, return_tensors="pt"), tok(corrupt, return_tensors="pt")
    return c, k, torch.tensor(io_ids), torch.tensor(s_ids)


def logit_diff(logits, io, s):
    last = logits[:, -1]
    return (last.gather(1, io[:, None]) - last.gather(1, s[:, None])).mean()


@torch.no_grad()
def fig_llm_patching():
    tok, model = load_gpt2()
    clean_x, corr_x, io, s = ioi_batch(tok)
    # In the corrupted prompt the correct answer is the other name, so measure clean IO vs clean S.
    clean_out = model(**clean_x, output_hidden_states=True)
    corr_out = model(**corr_x)
    ld_clean, ld_corr = logit_diff(clean_out.logits, io, s).item(), logit_diff(corr_out.logits, io, s).item()
    blocks = model.transformer.h
    T, L = clean_x["input_ids"].shape[1], len(blocks)

    # (a) residual stream patching: copy the clean residual at (layer, position) into the corrupted run.
    resid = np.zeros((L, T))
    for l in range(L):
        for t in range(T):
            def hook(mod, inp, out, l=l, t=t):
                h = out[0] if isinstance(out, tuple) else out
                h[:, t] = clean_out.hidden_states[l + 1][:, t]
                return out
            hk = blocks[l].register_forward_hook(hook)
            resid[l, t] = logit_diff(model(**corr_x).logits, io, s).item()
            hk.remove()
    resid = (resid - ld_corr) / (ld_clean - ld_corr)

    # (b) per-head patching of the attention output (pre-projection) at the final position.
    n_head = model.config.n_head
    d_head = model.config.n_embd // n_head
    cache = {}

    def grab(l):
        def h(mod, args):
            cache[l] = args[0].detach().clone()
        return h
    hks = [blocks[l].attn.c_proj.register_forward_pre_hook(grab(l)) for l in range(L)]
    model(**clean_x)
    for hk in hks:
        hk.remove()
    heads = np.zeros((L, n_head))
    for l in range(L):
        for hd in range(n_head):
            sl = slice(hd * d_head, (hd + 1) * d_head)

            def patch(mod, args, l=l, sl=sl):
                z = args[0].clone()
                z[:, -1, sl] = cache[l][:, -1, sl]
                return (z,)
            hk = blocks[l].attn.c_proj.register_forward_pre_hook(patch)
            heads[l, hd] = logit_diff(model(**corr_x).logits, io, s).item()
            hk.remove()
    heads = (heads - ld_corr) / (ld_clean - ld_corr)

    toks = IOI_TEMPLATE.replace(",", " ,").split()
    toks = [{"{A}": "A", "{B}": "B"}.get(w, w) for w in toks]
    toks[toks.index("B", toks.index("B") + 1)] = "B (S2)"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    im = ax.imshow(resid, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(T), toks, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(L)); ax.set_ylabel("layer (output of block)")
    ax.set_title("Residual-stream patching (clean → corrupted)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="share of logit difference restored")
    ax = axes[1]
    im = ax.imshow(heads, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xlabel("head"); ax.set_ylabel("layer"); ax.set_xticks(range(n_head)); ax.set_yticks(range(L))
    ax.set_title("Patching single heads' output at the last token")
    fig.colorbar(im, ax=ax, fraction=0.04, label="share restored")
    order = np.argsort(heads.ravel())
    pos = ", ".join(f"{i // n_head}.{i % n_head}" for i in order[::-1][:3])
    neg = ", ".join(f"{i // n_head}.{i % n_head}" for i in order[:2])
    fig.suptitle(f"GPT-2 small, indirect-object identification, {len(io)} prompts (logit diff: clean {ld_clean:.2f}, "
                 f"corrupted {ld_corr:.2f}). The answer moves from the S2 token to the last token around layers 7–9.\n"
                 f"Heads {pos} (layer.head) restore it most; {neg} push the other way — the same S-inhibition, name-mover and "
                 "negative name-mover heads Wang et al. (2023) report.", x=0.01, ha="left", fontsize=9.5, color=INK2)
    save(fig, "ex_llm_patching.jpg")
    return resid, heads


# ================================================================ 6. Toy model of superposition
def fig_superposition():
    """Elhage et al. (2022): 5 sparse features squeezed into 2 dimensions, x' = ReLU(W^T W x + b)."""
    n, m = 5, 2
    importance = 0.9 ** torch.arange(n)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, sparsity in zip(axes, [0.0, 0.7, 0.95]):
        torch.manual_seed(0)
        W = torch.nn.Parameter(torch.randn(m, n) * 0.3)
        b = torch.nn.Parameter(torch.zeros(n))
        opt = torch.optim.Adam([W, b], lr=1e-2)
        for _ in range(4000):
            x = torch.rand(1024, n) * (torch.rand(1024, n) > sparsity)
            y = torch.relu(x @ W.T @ W + b)
            loss = (importance * (y - x) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        Wd = W.detach().numpy()
        cols = [BLUE, ORANGE, AQUA, VIOLET, RED]
        for i in range(n):
            ax.annotate("", xy=Wd[:, i], xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=cols[i], lw=2))
            ax.text(*(Wd[:, i] * 1.15), f"f{i + 1}", color=cols[i], ha="center", va="center", fontsize=9)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
        ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        kept = int((np.linalg.norm(Wd, axis=0) > 0.5).sum())
        ax.set_title(f"sparsity {sparsity:.2f}: {kept} of 5 features represented")
    fig.suptitle("Toy model of superposition: when features are rare, the model packs more features than "
                 "dimensions (as an interfering, non-orthogonal pentagon)", x=0.01, ha="left", fontsize=10, color=INK2)
    save(fig, "ex_superposition.jpg")


FIGS = {"units": fig_units, "embedding": fig_embedding, "cka": fig_cka, "patch_pca": fig_patch_pca,
        "saliency": fig_saliency_sanity, "clip": fig_clip, "dino": fig_dino_heads,
        "vlm_logit_lens": fig_vlm_logit_lens, "vlm_knockout": fig_vlm_knockout,
        "llm_logit_lens": fig_llm_logit_lens, "llm_patching": fig_llm_patching,
        "superposition": fig_superposition}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
