"""Sample COCO val2017 images with *all* their labels (classes, boxes, polygon masks) for the take-home.

    # one-time: get the annotations (~250 MB zip)
    curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip annotations_trainval2017.zip -d data

    python make_data.py --ann data/annotations/instances_val2017.json --n 1000 --out data/coco1k

Every image keeps every annotation, so you can tell which objects are truly absent (Q6) and where each
object is (Q1, Q4). Each image also gets a single "dominant" label: the supercategory of its largest
object (person, vehicle, animal, food, ...), used as the class label for the embedding questions (Q2).
"""
import argparse
import json
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="path to instances_val2017.json")
    ap.add_argument("--n", type=int, default=1000, help="number of images")
    ap.add_argument("--out", default="data/coco1k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coco = json.load(open(args.ann))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    supercat = {c["name"]: c["supercategory"] for c in coco["categories"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    images = [im for im in coco["images"] if anns[im["id"]]]          # at least one object
    random.Random(args.seed).shuffle(images)
    images = sorted(images[: args.n], key=lambda im: im["id"])

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    def fetch(im):
        dst = out / "images" / im["file_name"]
        if not dst.exists():
            urllib.request.urlretrieve(im["coco_url"], dst)

    with ThreadPoolExecutor(8) as pool:
        for i, _ in enumerate(pool.map(fetch, images)):
            print(f"\rdownloaded {i + 1}/{len(images)}", end="")
    print()

    records = []
    for im in images:
        objs = [{"category": cats[a["category_id"]], "supercategory": supercat[cats[a["category_id"]]],
                 "bbox": a["bbox"], "area": a["area"], "iscrowd": a["iscrowd"],
                 # polygons in pixel coordinates; crowd regions are RLE-encoded and skipped here
                 "polygons": a["segmentation"] if not a["iscrowd"] else []}
                for a in anns[im["id"]]]
        big = max((o for o in objs if not o["iscrowd"]), key=lambda o: o["area"], default=objs[0])
        records.append({"id": im["id"], "file": f"images/{im['file_name']}", "width": im["width"],
                        "height": im["height"], "objects": objs,
                        "classes": sorted({o["category"] for o in objs}),
                        "dominant": big["supercategory"],
                        "dominant_frac": big["area"] / (im["width"] * im["height"])})

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
    meta = {"categories": sorted(cats.values()), "supercategories": supercat, "frequency": dict(freq),
            "cooccurrence": {c: dict(v) for c, v in cooc.items()}}
    json.dump({"images": records, **meta}, open(out / "labels.json", "w"))
    print(f"{len(records)} images, {sum(len(r['objects']) for r in records)} objects -> {out}")


if __name__ == "__main__":
    main()
