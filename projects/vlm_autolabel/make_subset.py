"""Build a small COCO val2017 subset (images + ground-truth JSON) for the auto-labeling project.

    # one-time: get the annotations (~250 MB zip)
    curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip annotations_trainval2017.zip

    python make_subset.py --ann annotations/instances_val2017.json \
        --classes "person,car,dog,bicycle,traffic light" --n 200 --out data/coco_subset
"""
import argparse
import json
import random
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="path to instances_val2017.json")
    ap.add_argument("--classes", required=True, help="comma-separated COCO class names")
    ap.add_argument("--n", type=int, default=200, help="number of images")
    ap.add_argument("--out", default="data/coco_subset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coco = json.load(open(args.ann))
    wanted = [c.strip() for c in args.classes.split(",")]
    cats = [c for c in coco["categories"] if c["name"] in wanted]
    missing = set(wanted) - {c["name"] for c in cats}
    if missing:
        raise SystemExit(f"not COCO classes: {missing}")
    cat_ids = {c["id"] for c in cats}

    # Keep images that contain at least one wanted class.
    anns = [a for a in coco["annotations"] if a["category_id"] in cat_ids and not a["iscrowd"]]
    img_ids = sorted({a["image_id"] for a in anns})
    random.Random(args.seed).shuffle(img_ids)
    img_ids = set(img_ids[: args.n])

    images = [im for im in coco["images"] if im["id"] in img_ids]
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(images):
        dst = out / "images" / im["file_name"]
        if not dst.exists():
            urllib.request.urlretrieve(im["coco_url"], dst)
        print(f"\rdownloaded {i + 1}/{len(images)}", end="")
    print()

    gt = {
        "images": images,
        "categories": cats,
        "annotations": [a for a in anns if a["image_id"] in img_ids],
    }
    json.dump(gt, open(out / "gt.json", "w"))
    print(f"{len(images)} images, {len(gt['annotations'])} boxes -> {out}")


if __name__ == "__main__":
    main()
