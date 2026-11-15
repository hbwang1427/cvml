"""Shared helpers for the interpretability take-home: data, models, hooks.

Everything here runs on a laptop (Apple MPS / CPU) or a free Colab T4. Read it before you trust it:
part of the exercise is checking that the tools measure what they claim to measure.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

CLIP = "openai/clip-vit-base-patch32"
DINO = "facebook/dinov2-small"
VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"       # also try SmolVLM-500M-Instruct
LLM = "gpt2"


# ================================================================ data
def load_coco(data_dir):
    """Records from make_data.py: [{'id', 'file', 'objects': [...], 'classes': [...]}, ...] + metadata."""
    data_dir = Path(data_dir)
    meta = json.load(open(data_dir / "labels.json"))
    for r in meta["images"]:
        r["path"] = data_dir / r["file"]
    return meta


def open_image(rec):
    return Image.open(rec["path"]).convert("RGB")


def split_ids(records, test_frac=0.3, seed=0):
    """Split by *image*, so no image contributes to both train and test."""
    ids = sorted(r["id"] for r in records)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = int(len(ids) * test_frac)
    return set(ids[n_test:]), set(ids[:n_test])


# ================================================================ vision encoders
def load_vision(name):
    """Return (model, processor) for 'clip', 'dinov2' or 'clip_random' (CLIP architecture, random weights)."""
    from transformers import AutoImageProcessor, CLIPVisionModel, CLIPVisionConfig, AutoModel
    if name == "clip":
        return CLIPVisionModel.from_pretrained(CLIP).to(DEVICE).eval(), AutoImageProcessor.from_pretrained(CLIP)
    if name == "clip_random":
        torch.manual_seed(0)
        model = CLIPVisionModel(CLIPVisionConfig.from_pretrained(CLIP))
        return model.to(DEVICE).eval(), AutoImageProcessor.from_pretrained(CLIP)
    if name == "dinov2":
        return AutoModel.from_pretrained(DINO).to(DEVICE).eval(), AutoImageProcessor.from_pretrained(DINO)
    raise ValueError(name)


@torch.no_grad()
def layer_features(model, proc, images, batch_size=16, pool="cls"):
    """Per-layer image features: array [n_layers + 1, N, D] (index 0 = patch embeddings).

    pool='cls' takes the [CLS] token, pool='mean' averages the patch tokens.
    """
    feats = []
    for i in range(0, len(images), batch_size):
        x = proc(images=images[i:i + batch_size], return_tensors="pt")["pixel_values"].to(DEVICE)
        hs = model(pixel_values=x, output_hidden_states=True).hidden_states
        h = torch.stack(hs)                                   # [L+1, B, T, D]
        h = h[:, :, 0] if pool == "cls" else h[:, :, 1:].mean(2)
        feats.append(h.float().cpu())
    return torch.cat(feats, 1).numpy()


# ================================================================ CLIP zero-shot + saliency
class ClipZeroShot:
    """CLIP zero-shot classifier over a fixed label set, differentiable w.r.t. the pixels."""

    def __init__(self, labels, template="a photo of a {}."):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(CLIP).to(DEVICE).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP)
        self.labels = list(labels)
        x = self.proc(text=[template.format(c) for c in labels], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            t = self.model.get_text_features(**x)
        self.text = F.normalize(getattr(t, "pooler_output", t), dim=-1)

    def pixels(self, images):
        return self.proc(images=images, return_tensors="pt")["pixel_values"].to(DEVICE)

    def log_probs(self, pixel_values):
        v = self.model.get_image_features(pixel_values=pixel_values)
        v = F.normalize(getattr(v, "pooler_output", v), dim=-1)
        return (self.model.logit_scale.exp() * v @ self.text.T).log_softmax(-1)

    def vision_blocks(self):
        enc = self.model.vision_model.encoder
        return enc.layers


def saliency(clf, pixel_values, target, method="grad", steps=32):
    """Pixel attributions [H, W] for log P(target) under CLIP zero-shot.

    method: 'grad' (|gradient| max over channels), 'gxi' (gradient x input),
            'ig' (integrated gradients from a zero baseline, i.e. the dataset-mean image).
    """
    x = pixel_values[:1]

    def grad_at(inp):
        inp = inp.detach().clone().requires_grad_(True)
        clf.log_probs(inp)[:, target].sum().backward()
        return inp.grad

    if method == "grad":
        return grad_at(x)[0].abs().amax(0).cpu()
    if method == "gxi":
        return (grad_at(x)[0] * x[0]).sum(0).abs().cpu()
    if method == "ig":
        alphas = torch.linspace(1 / steps, 1, steps, device=x.device).view(-1, 1, 1, 1)
        g = torch.cat([grad_at(a * x) for a in alphas.split(8)]).mean(0)
        return (g * x[0]).sum(0).abs().cpu()
    raise ValueError(method)


def randomize_top_blocks(clf, k, seed=0):
    """Cascading randomisation (Adebayo et al. 2018), in place: re-initialise the top-k vision blocks.
    Reload the classifier to undo."""
    torch.manual_seed(seed)
    for block in list(clf.vision_blocks())[::-1][:k]:
        for name, p in block.named_parameters():
            with torch.no_grad():
                if p.dim() > 1:
                    p.normal_(0, 0.02)
                elif "layer_norm" in name and name.endswith("weight"):
                    p.fill_(1.0)
                else:
                    p.zero_()


@torch.no_grad()
def deletion_curve(clf, image, sal, target, steps=16, fill="blur"):
    """Remove the most-salient pixels first; return P(target) after each step (Petsiuk et al. 2018).
    A faithful map makes P drop fast (low area under the curve).

    fill='blur' replaces removed pixels with a blurred copy, 'grey' with mid-grey. Removing pixels creates
    out-of-distribution images either way -- think about what that does to the metric.
    """
    img = image.resize((224, 224))
    a = np.array(img).astype(np.float32)
    rep = np.array(img.filter(ImageFilter.GaussianBlur(10))).astype(np.float32) if fill == "blur" \
        else np.full_like(a, 128.0)
    s = np.array(Image.fromarray(np.asarray(sal, dtype=np.float32)).resize((224, 224), Image.BILINEAR))
    order = np.argsort(-s.ravel())
    frames = []
    for k in np.linspace(0, len(order), steps + 1).astype(int):
        m = np.zeros(len(order), bool)
        m[order[:k]] = True
        m = m.reshape(224, 224)[..., None]
        frames.append(Image.fromarray(np.where(m, rep, a).astype(np.uint8)))
    lp = torch.cat([clf.log_probs(clf.pixels(frames[i:i + 8])) for i in range(0, len(frames), 8)])
    return lp[:, target].exp().cpu().numpy()


# ================================================================ VLM (SmolVLM / Idefics3)
def load_vlm(name=VLM):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(name)
    # eager attention => every layer receives an explicit 4-D mask that hooks can edit
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.float32,
                                                        attn_implementation="eager").to(DEVICE).eval()
    return model, proc


class VLMYesNo:
    """Ask yes/no questions about an image; read P(yes) and the hidden states at the answer position."""

    def __init__(self, name=VLM):
        self.model, self.proc = load_vlm(name)
        tok = self.proc.tokenizer
        self.image_id = tok.convert_tokens_to_ids("<image>")
        self.yes = [tok.encode(s, add_special_tokens=False)[0] for s in ("Yes", " Yes", "yes", " yes")]
        self.no = [tok.encode(s, add_special_tokens=False)[0] for s in ("No", " No", "no", " no")]
        self.layers = self.model.model.text_model.layers

    def inputs(self, image, question):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        return self.proc(text=prompt, images=[image], return_tensors="pt", do_image_splitting=False).to(DEVICE)

    @torch.no_grad()
    def __call__(self, image, question, block_layers=(), hidden=False):
        """Return dict(p_yes, [hidden: [n_layers + 1, d] at the last prompt token]).

        block_layers: LLM layer indices in which text tokens may not attend to image tokens
        (attention knockout). Image tokens still attend to each other.
        """
        x = self.inputs(image, question)
        img = x["input_ids"][0] == self.image_id
        q, k = (~img).nonzero()[:, 0], img.nonzero()[:, 0]

        def pre(mod, args, kwargs):
            m = kwargs["attention_mask"].clone()
            m[:, :, q[:, None], k[None, :]] = torch.finfo(m.dtype).min
            kwargs["attention_mask"] = m
            return args, kwargs

        hooks = [self.layers[i].self_attn.register_forward_pre_hook(pre, with_kwargs=True) for i in block_layers]
        try:
            out = self.model(**x, output_hidden_states=hidden)
        finally:
            for h in hooks:
                h.remove()
        pr = out.logits[0, -1].float().softmax(-1)
        res = {"p_yes": (pr[self.yes].sum() / (pr[self.yes].sum() + pr[self.no].sum())).item()}
        if hidden:
            res["hidden"] = torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()
        return res


# ================================================================ GPT-2 (indirect-object identification)
IOI_TEMPLATES = [
    "When {A} and {B} went to the store, {B} gave a drink to",
    "When {A} and {B} went to the park, {B} threw a ball to",
    "After {A} and {B} arrived at the office, {B} sent an email to",
    "While {A} and {B} were working at the school, {B} wrote a note to",
]
NAMES = ["Mary", "John", "Alice", "Bob", "Sarah", "Tom", "Emma", "James", "Anna", "David", "Laura", "Mark",
         "Lisa", "Paul", "Kate", "Mike", "Rachel", "Daniel", "Jessica", "Kevin", "Amy", "Ryan", "Grace", "Eric"]


def load_llm(name=LLM):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager").to(DEVICE).eval()
    return model, tok


def ioi_dataset(tok, template, n=48, corruption="swap", seed=0):
    """Clean prompts '... A and B ..., B gave ... to' (answer A) and matched corrupted prompts.

    corruption='swap': swap A and B, so the corrupted answer is B (flips the behaviour).
    corruption='abc':  replace both names with two *new* names, so neither A nor B is a sensible answer
                       (removes the information instead of flipping it).
    Returns (clean_ids, corrupt_ids, io_ids, s_ids); all prompts have equal token length.
    """
    rng = np.random.default_rng(seed)
    for nm in NAMES:
        assert len(tok.encode(" " + nm)) == 1, f"{nm} is not a single token"
    clean, corrupt, io, s = [], [], [], []
    for _ in range(n):
        a, b, c, d = rng.choice(NAMES, 4, replace=False)
        clean.append(template.format(A=a, B=b))
        corrupt.append(template.format(A=b, B=a) if corruption == "swap" else template.format(A=c, B=d))
        io.append(tok.encode(" " + a)[0]); s.append(tok.encode(" " + b)[0])
    enc = lambda xs: torch.tensor([tok.encode(x) for x in xs])   # noqa: E731
    return enc(clean).to(DEVICE), enc(corrupt).to(DEVICE), torch.tensor(io, device=DEVICE), \
        torch.tensor(s, device=DEVICE)


def logit_diff(logits, io, s):
    """Mean over prompts of logit(IO) - logit(S) at the final position."""
    last = logits[:, -1]
    return (last.gather(1, io[:, None]) - last.gather(1, s[:, None])).mean().item()


@torch.no_grad()
def run_with_cache(model, ids):
    """Forward pass that also records residual stream (block outputs) and per-head attention outputs."""
    blocks = model.transformer.h
    cache, hooks = {}, []
    for l, blk in enumerate(blocks):
        hooks.append(blk.register_forward_hook(
            lambda m, i, o, l=l: cache.__setitem__(("resid", l), (o[0] if isinstance(o, tuple) else o).clone())))
        hooks.append(blk.attn.c_proj.register_forward_pre_hook(
            lambda m, a, l=l: cache.__setitem__(("z", l), a[0].clone())))   # concat of head outputs
    try:
        logits = model(ids).logits
    finally:
        for h in hooks:
            h.remove()
    return logits, cache


@torch.no_grad()
def run_patched(model, ids, patches):
    """Run on `ids`, overwriting activations. patches: list of (kind, layer, pos, head, value) where kind is
    'resid' (value [B, d], head ignored) or 'z' (value [B, d_head]); pos may be an int or slice."""
    blocks = model.transformer.h
    d_head = model.config.n_embd // model.config.n_head
    hooks = []
    for kind, l, pos, head, value in patches:
        if kind == "resid":
            def fh(m, i, o, pos=pos, value=value):
                h = o[0] if isinstance(o, tuple) else o
                h[:, pos] = value
                return o
            hooks.append(blocks[l].register_forward_hook(fh))
        else:
            sl = slice(head * d_head, (head + 1) * d_head)

            def ph(m, a, pos=pos, sl=sl, value=value):
                z = a[0].clone()
                z[:, pos, sl] = value
                return (z,)
            hooks.append(blocks[l].attn.c_proj.register_forward_pre_hook(ph))
    try:
        return model(ids).logits
    finally:
        for h in hooks:
            h.remove()
