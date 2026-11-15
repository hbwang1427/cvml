"""Sample COCO val2017 images with *all* their labels (every class, every box) for the take-home.

Unlike ../vlm_autolabel/make_subset.py, images are not filtered by class, and labels.json keeps every
annotation so that you can tell which objects are truly absent from an image.

    # one-time: get the annotations (~250 MB zip)
    curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip annotations_trainval2017.zip -d data

    python make_data.py --ann data/annotations/instances_val2017.json --n 400 --out data/coco400
"""
import argparse
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="path to instances_val2017.json")
    ap.add_argument("--n", type=int, default=400, help="number of images")
    ap.add_argument("--out", default="data/coco400")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coco = json.load(open(args.ann))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    images = [im for im in coco["images"] if anns[im["id"]]]          # at least one object
    random.Random(args.seed).shuffle(images)
    images = sorted(images[: args.n], key=lambda im: im["id"])

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for i, im in enumerate(images):
        dst = out / "images" / im["file_name"]
        if not dst.exists():
            urllib.request.urlretrieve(im["coco_url"], dst)
        objs = [{"category": cats[a["category_id"]], "bbox": a["bbox"], "area": a["area"],
                 "iscrowd": a["iscrowd"]} for a in anns[im["id"]]]
        records.append({"id": im["id"], "file": f"images/{im['file_name']}", "width": im["width"],
                        "height": im["height"], "objects": objs,
                        "classes": sorted({o["category"] for o in objs})})
        print(f"\rdownloaded {i + 1}/{len(images)}", end="")
    print()

    # Co-occurrence counts over the whole val set: used for "popular"/"adversarial" absent objects (POPE).
    cooc = defaultdict(lambda: defaultdict(int))
    freq = defaultdict(int)
    for objs in anns.values():
        present = {cats[a["category_id"]] for a in objs}
        for c in present:
            freq[c] += 1
            for d in present:
                if d != c:
                    cooc[c][d] += 1
    meta = {"categories": sorted(cats.values()), "frequency": dict(freq),
            "cooccurrence": {c: dict(v) for c, v in cooc.items()}}
    json.dump({"images": records, **meta}, open(out / "labels.json", "w"))
    print(f"{len(records)} images, {sum(len(r['objects']) for r in records)} objects -> {out}")


if __name__ == "__main__":
    main()
