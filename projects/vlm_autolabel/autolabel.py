"""Zero-shot auto-labeling: open-vocabulary detector (+ optional SAM masks) -> COCO JSON.

    python autolabel.py --images data/coco_subset/images \
        --gt data/coco_subset/gt.json --out preds_owlv2.json
    python autolabel.py --images data/coco_subset/images --gt data/coco_subset/gt.json \
        --model google/owlv2-large-patch14-ensemble --out preds_owlv2_large.json
    python autolabel.py ... --sam            # also write instance masks (RLE)

Any HF "zero-shot-object-detection" model works with --model. (Grounding DINO needs its own
"cat. dog." prompt format and post-processing -- a good extension exercise.)

With --gt, class names/ids are taken from the ground truth and COCO mAP is printed.
Without --gt, pass --classes "a,b,c" to label your own images.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sam(device):
    from transformers import SamModel, SamProcessor
    proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()
    return proc, model


@torch.no_grad()
def boxes_to_masks(sam, image, boxes_xyxy, device):
    """Prompt SAM with each detected box; returns a list of HxW bool masks."""
    proc, model = sam
    inputs = proc(image, input_boxes=[boxes_xyxy], return_tensors="pt")
    out = model(pixel_values=inputs["pixel_values"].to(device),
                input_boxes=inputs["input_boxes"].float().to(device),  # MPS has no float64
                multimask_output=False)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"])
    return [m[0].numpy() for m in masks[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", help="COCO ground-truth JSON (for class list + evaluation)")
    ap.add_argument("--classes", help="comma-separated class names (if no --gt)")
    ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--prompt", default="a photo of a {}", help="text template per class")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--sam", action="store_true", help="add SAM masks for each box")
    ap.add_argument("--out", default="preds.json")
    args = ap.parse_args()

    if args.gt:
        gt = json.load(open(args.gt))
        cats = gt["categories"]
        files = {im["file_name"]: im["id"] for im in gt["images"]}
    else:
        cats = [{"id": i + 1, "name": n.strip()} for i, n in enumerate(args.classes.split(","))]
        files = {p.name: i for i, p in enumerate(sorted(Path(args.images).iterdir()))
                 if p.suffix.lower() in IMG_EXT}
    prompts = [args.prompt.format(c["name"]) for c in cats]
    prompt_to_id = {p: c["id"] for p, c in zip(prompts, cats)}

    device = pick_device()
    det = pipeline("zero-shot-object-detection", model=args.model, device=device)
    sam = load_sam(device) if args.sam else None

    preds, t0 = [], time.time()
    for k, (fname, img_id) in enumerate(files.items()):
        image = Image.open(Path(args.images) / fname).convert("RGB")
        dets = det(image, candidate_labels=prompts, threshold=args.threshold)
        boxes = [[d["box"]["xmin"], d["box"]["ymin"], d["box"]["xmax"], d["box"]["ymax"]] for d in dets]
        masks = boxes_to_masks(sam, image, boxes, device) if sam and boxes else [None] * len(dets)
        for d, (x0, y0, x1, y1), m in zip(dets, boxes, masks):
            p = {"image_id": img_id, "category_id": prompt_to_id[d["label"]],
                 "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(d["score"])}
            if m is not None:
                from pycocotools import mask as mask_utils
                rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode()
                p["segmentation"] = rle
            preds.append(p)
        print(f"\r{k + 1}/{len(files)} images, {len(preds)} boxes", end="")
    dt = time.time() - t0
    print(f"\n{dt:.1f}s total, {dt / max(len(files), 1):.2f}s/image on {device}")

    json.dump(preds, open(args.out, "w"))
    print("wrote", args.out)

    if args.gt and preds:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco_gt = COCO(args.gt)
        coco_dt = coco_gt.loadRes(args.out)
        for iou_type in ["bbox"] + (["segm"] if args.sam else []):
            print(f"\n=== {iou_type} ===")
            ev = COCOeval(coco_gt, coco_dt, iou_type)
            ev.evaluate(); ev.accumulate(); ev.summarize()


if __name__ == "__main__":
    main()
