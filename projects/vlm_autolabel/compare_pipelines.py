"""Compare five auto-labeling strategies built from off-the-shelf models (no training).

  A  detector      OWLv2 open-vocabulary detector alone
  B  vlm-ground    Florence-2 phrase grounding alone (a VLM that outputs boxes)
  C  det+verify    OWLv2 proposals re-scored by a VLM verifier (SmolVLM: "is this a {class}?")
  D  ensemble      OWLv2 + Florence-2 boxes fused (agreement boosts confidence)
  E  harness       VLM tags which classes are present -> D restricted to those classes
                   -> every box verified by the VLM -> final score

    python compare_pipelines.py --images data/coco_subset/images --gt data/coco_subset/gt.json \
        --n 50 --out results/

Model outputs are cached in --out, so re-running to try a new fusion rule takes seconds.
"""
import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          Florence2ForConditionalGeneration, pipeline)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OWL = "google/owlv2-base-patch16-ensemble"
FLORENCE = "florence-community/Florence-2-base"
VLM = "HuggingFaceTB/SmolVLM-500M-Instruct"


# ---------------------------------------------------------------- model wrappers
class Owl:
    def __init__(self):
        self.pipe = pipeline("zero-shot-object-detection", model=OWL, device=DEVICE)

    def __call__(self, image, classes):
        dets = self.pipe(image, candidate_labels=classes, threshold=0.05)
        return [{"label": d["label"], "score": d["score"],
                 "box": [d["box"][k] for k in ("xmin", "ymin", "xmax", "ymax")]} for d in dets]


class Florence:
    """Phrase grounding: returns boxes for the class names it finds; it gives no confidence scores."""

    def __init__(self):
        self.proc = AutoProcessor.from_pretrained(FLORENCE)
        self.model = Florence2ForConditionalGeneration.from_pretrained(FLORENCE).to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, image, classes):
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        inputs = self.proc(text=task + ". ".join(classes), images=image, return_tensors="pt").to(DEVICE)
        out = self.model.generate(**inputs, max_new_tokens=512, num_beams=3)
        text = self.proc.batch_decode(out, skip_special_tokens=False)[0]
        res = self.proc.post_process_generation(text, task=task, image_size=image.size)[task]
        return [{"label": l.strip().lower(), "box": b} for b, l in zip(res["bboxes"], res["labels"])
                if l.strip().lower() in classes]


class YesNoVLM:
    """P(yes) from a single forward pass -- a calibrated-ish score instead of free text."""

    def __init__(self):
        self.proc = AutoProcessor.from_pretrained(VLM)
        self.model = AutoModelForImageTextToText.from_pretrained(VLM, dtype=torch.float32).to(DEVICE).eval()
        tok = self.proc.tokenizer
        self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes", " yes")]
        self.no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no", " no")]

    @torch.no_grad()
    def __call__(self, image, question):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.proc(text=prompt, images=[image], return_tensors="pt", do_image_splitting=False).to(DEVICE)
        logits = self.model(**inputs).logits[0, -1].float()
        y, n = torch.logsumexp(logits[self.yes], 0), torch.logsumexp(logits[self.no], 0)
        return float(torch.sigmoid(y - n))


# ---------------------------------------------------------------- helpers
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def crop(image, box, pad=0.15, min_size=48):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, 1 + 2 * pad
    w, h = max(w * s, min_size), max(h * s, min_size)
    return image.crop((max(0, cx - w / 2), max(0, cy - h / 2),
                       min(image.width, cx + w / 2), min(image.height, cy + h / 2)))


def fuse(owl, flo, iou_thr=0.5):
    """Weighted box fusion-lite: boxes both models agree on get a confidence bonus."""
    out, used = [], set()
    for f in flo:
        match = max(((i, iou(o["box"], f["box"])) for i, o in enumerate(owl) if o["label"] == f["label"]),
                    key=lambda t: t[1], default=(None, 0))
        if match[1] >= iou_thr:
            o = owl[match[0]]; used.add(match[0])
            box = [(a + b) / 2 for a, b in zip(o["box"], f["box"])]
            out.append({"label": f["label"], "box": box, "score": min(1.0, o["score"] + 0.5), "src": "both",
                        "p_yes": max(o.get("p_yes", 0), f.get("p_yes", 0))})
        else:
            out.append(dict(f, score=0.3, src="florence"))
    out += [dict(o, src="owl") for i, o in enumerate(owl) if i not in used]
    return out


def to_coco(dets, img_id, cat_id):
    return [{"image_id": img_id, "category_id": cat_id[d["label"]], "score": float(d["score"]),
             "bbox": [d["box"][0], d["box"][1], d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]]}
            for d in dets]


def evaluate(gt_path, preds, img_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_path)
        if not preds:
            return {"AP": 0.0, "AP50": 0.0, "AR100": 0.0, "per_class_AP50": {}}
        ev = COCOeval(gt, gt.loadRes(preds), "bbox")
        ev.params.imgIds = img_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    per = {}
    for k, cid in enumerate(ev.params.catIds):
        p = ev.eval["precision"][0, :, k, 0, 2]  # IoU=.50, all areas, maxDets=100
        per[gt.cats[cid]["name"]] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return {"AP": ev.stats[0], "AP50": ev.stats[1], "AR100": ev.stats[8], "per_class_AP50": per}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--n", type=int, default=50, help="number of images to use")
    ap.add_argument("--out", default="results")
    ap.add_argument("--min-owl", type=float, default=0.1, help="OWLv2 score floor for proposals")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gt = json.load(open(args.gt))
    classes = [c["name"] for c in gt["categories"]]
    cat_id = {c["name"]: c["id"] for c in gt["categories"]}
    images = sorted(gt["images"], key=lambda im: im["id"])[: args.n]

    cache_f = out / "cache.json"
    cache = json.load(open(cache_f)) if cache_f.exists() else {}
    models, timing = {}, {}

    def get(name, cls):
        if name not in models:
            models[name] = cls()
        return models[name]

    for k, im in enumerate(images):
        key = str(im["id"])
        c = cache.setdefault(key, {})
        image = Image.open(Path(args.images) / im["file_name"]).convert("RGB")
        steps = [
            ("owl", lambda: [d for d in get("owl", Owl)(image, classes) if d["score"] >= args.min_owl]),
            ("florence", lambda: get("florence", Florence)(image, classes)),
            ("tags", lambda: {cl: get("vlm", YesNoVLM)(image, f"Is there a {cl} in this image? Answer yes or no.")
                              for cl in classes}),
        ]
        for name, fn in steps:
            if name not in c:
                t0 = time.time(); c[name] = fn(); timing[name] = timing.get(name, 0) + time.time() - t0
        # Verify every candidate box (OWLv2 proposals and Florence boxes) once.
        if "verify" not in c:
            t0 = time.time()
            vlm = get("vlm", YesNoVLM)
            c["verify"] = {
                src: [vlm(crop(image, d["box"]), f"Is this a {d['label']}? Answer yes or no.") for d in c[src]]
                for src in ("owl", "florence")}
            timing["verify"] = timing.get("verify", 0) + time.time() - t0
        print(f"\r{k + 1}/{len(images)} images", end="")
        if (k + 1) % 10 == 0:
            json.dump(cache, open(cache_f, "w"))
    json.dump(cache, open(cache_f, "w"))
    print()

    # ---- build the five label sets from the cached model outputs
    methods = {m: [] for m in ["A detector (OWLv2)", "B VLM grounding (Florence-2)",
                               "C detector + VLM verifier", "D ensemble (OWLv2 + Florence-2)",
                               "E harness (tag → ensemble → verify)"]}
    for im in images:
        c = cache[str(im["id"])]
        owl = c["owl"]
        flo = [dict(d, score=1.0) for d in c["florence"]]  # no confidence available
        owl_v = [dict(d, p_yes=p) for d, p in zip(owl, c["verify"]["owl"])]
        flo_v = [dict(d, p_yes=p) for d, p in zip(flo, c["verify"]["florence"])]

        methods["A detector (OWLv2)"] += to_coco(owl, im["id"], cat_id)
        methods["B VLM grounding (Florence-2)"] += to_coco(flo, im["id"], cat_id)
        methods["C detector + VLM verifier"] += to_coco(
            [dict(d, score=np.sqrt(d["score"] * d["p_yes"])) for d in owl_v], im["id"], cat_id)
        fused = fuse(owl_v, flo_v)
        methods["D ensemble (OWLv2 + Florence-2)"] += to_coco(fused, im["id"], cat_id)
        present = {cl for cl, p in c["tags"].items() if p >= 0.5}
        harness = [dict(d, score=np.sqrt(d["score"] * d["p_yes"])) for d in fused if d["label"] in present]
        methods["E harness (tag → ensemble → verify)"] += to_coco(harness, im["id"], cat_id)

    img_ids = [im["id"] for im in images]
    results = {}
    for name, preds in methods.items():
        json.dump(preds, open(out / f"preds_{name.split()[0]}.json", "w"))
        results[name] = evaluate(args.gt, preds, img_ids) | {"boxes": len(preds)}

    lines = ["| Method | boxes | AP@[.5:.95] | AP@.5 | AR@100 |", "|---|---:|---:|---:|---:|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['boxes']} | {r['AP']:.3f} | {r['AP50']:.3f} | {r['AR100']:.3f} |")
    lines += ["", "Per-class AP@.5:", "", "| Method | " + " | ".join(classes) + " |",
              "|---|" + "---:|" * len(classes)]
    for name, r in results.items():
        lines.append(f"| {name.split()[0]} | " + " | ".join(f"{r['per_class_AP50'][c]:.2f}" for c in classes) + " |")
    if timing:
        lines += ["", "Model time this run (s): " + ", ".join(f"{k} {v:.0f}" for k, v in timing.items())]
    table = "\n".join(lines)
    (out / "results.md").write_text(table + "\n")
    json.dump(results, open(out / "results.json", "w"), indent=1)
    print(table)


if __name__ == "__main__":
    main()
