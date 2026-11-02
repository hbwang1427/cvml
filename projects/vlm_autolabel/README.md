# Project: Foundation Models as Data Labelers

**Level:** senior undergraduate / early graduate · **Team:** 1–3 students · **Duration:** 3–4 weeks
**Compute:** a laptop (Apple Silicon / CPU) or free Colab T4 — **no large-scale training required**

Accompanies [Lecture 01 — Vision Foundation Models](../../lectures/01_foundation_models/README.md).

## Motivation

Labeling is the most expensive part of most vision projects. The Segment Anything data engine
showed that a model in the loop cuts mask time from ~91 s to 14 s per object, and that a model can
eventually label *on its own* at scale (1.1B masks). Today, open-vocabulary detectors (OWLv2,
Grounding DINO), promptable segmenters (SAM 2 / SAM 3) and open VLMs (Qwen2.5-VL, Florence-2,
Gemma 3, MedGemma) can be combined into a labeling pipeline with **zero training**.

**Research question:** *How good are foundation-model labels, which errors do they make, and how much
human effort is needed to turn them into training-quality data?*

## Core track (everyone)

### Week 1 — Zero-shot auto-labeling baseline

```bash
pip install torch transformers pillow pycocotools
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip

python make_subset.py --ann annotations/instances_val2017.json \
    --classes "person,car,bicycle,dog,traffic light" --n 200 --out data/coco_subset

python autolabel.py --images data/coco_subset/images --gt data/coco_subset/gt.json \
    --out preds_owlv2.json            # add --sam for instance masks
```

`autolabel.py` runs OWLv2 (any HF zero-shot detector works via `--model`), optionally prompts SAM
with each box, writes COCO-format predictions and prints COCO mAP.

Deliverables:
1. mAP@0.5 and mAP@[.5:.95], per class.
2. A precision–recall curve vs. score threshold; choose an operating threshold and justify it.
3. 20 annotated failure cases sorted into a taxonomy (missed small objects, wrong class,
   duplicate boxes, box too loose, hallucinated objects, ...).

### Week 2 — VLMs, model combinations and a labeling harness

`compare_pipelines.py` builds five labelers from the same frozen models and evaluates them on the
same images. Every model output is cached, so trying a new combination rule takes seconds.

```bash
python compare_pipelines.py --images data/coco_subset/images --gt data/coco_subset/gt.json \
    --n 50 --out results/
python visualize.py --images data/coco_subset/images --gt data/coco_subset/gt.json \
    --preds results/preds_A.json results/preds_E.json --names "A: OWLv2" "E: harness" --out compare.jpg
```

| | Strategy | Models | Idea |
|---|---|---|---|
| A | Detector only | OWLv2 | Open-vocabulary detector, class names as text queries |
| B | VLM only | Florence-2 (0.23B) | A VLM that *outputs boxes* (phrase grounding); no confidence scores |
| C | Detector + VLM verifier | OWLv2 → SmolVLM-500M | Crop each box, ask *"Is this a {class}?"*, use P(yes) to re-score |
| D | Ensemble | OWLv2 + Florence-2 | Fuse boxes; agreement between models boosts confidence |
| E | **Harness** | all three | VLM tags which classes are present → ensemble restricted to those → every box verified |

A **harness** is a small program that wraps several models and decides what to call next, much as
an agent framework wraps an LLM with tools. Strategy E is a fixed pipeline; your job in Week 2 is to
make it smarter. Implement **at least two** of the following and report the change in AP:

- **Better verification prompts** — add context ("a photo of a street; is the highlighted object a
  car?"), or show the full image with the box drawn instead of a crop.
- **Count consistency** — ask the VLM *"How many cars are there?"*; if it disagrees with the detector,
  lower the threshold or tile/zoom the image and detect again (small objects!).
- **Bigger or different VLMs** — swap SmolVLM for Qwen2.5-VL-3B, Gemma 3 4B or Florence-2-large;
  plot AP vs. seconds/image.
- **Learned fusion** — fit a logistic regression on (OWLv2 score, P(yes), Florence agreement) using 20
  labeled images; apply it to the rest. Is that still "no training"? Discuss.
- **SAM refinement** — tighten boxes with the SAM mask's bounding box; does AP@0.75 improve?
- **Agentic harness (extension)** — let a frontier VLM (Claude, Gemini or GPT via API) orchestrate:
  give it tools `detect(classes)`, `zoom(region)`, `verify(box, class)` and `submit(labels)`, and let it
  decide the order. Compare cost ($ / 100 images) and AP against E. Keep a budget cap!

### Week 3 — Human-in-the-loop & downstream value

1. **Correction cost.** Time yourself correcting auto-labels for 50 images (in
   [CVAT](https://www.cvat.ai/) or [Label Studio](https://labelstud.io/), both import COCO JSON)
   vs. labeling 50 images from scratch. Report seconds/object — compare with the SAM-paper numbers.
2. **Active review.** Rank images by uncertainty (low max score, VLM–detector disagreement). If a human
   only fixes the top 20% most uncertain images, how close do you get to GT-quality labels?
3. **Does it matter for training?** Fine-tune a *small* detector (e.g., YOLO-nano, ~10 min on a T4)
   on (a) GT labels, (b) raw auto-labels, (c) auto-labels + your Week-2 fixes. Evaluate all on the
   same held-out GT split. This is the headline result.

### Week 4 — Report & presentation

A 4-page report (CVPR-style template) + 8-minute talk: pipeline diagram, quantitative tables,
failure taxonomy, cost–quality trade-off plot (human seconds vs. mAP), and recommendations.

## Domain tracks (pick one to replace COCO in weeks 2–3)

| Track | Data (small, public) | Twist |
|---|---|---|
| **Healthcare** | [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) polyps (1k images), ISIC skin lesions, or a chest X-ray subset | Compare SAM vs. **MedSAM**; general VLM vs. **MedGemma** as verifier. Expect a big domain gap. |
| **Self-driving** | [nuImages](https://www.nuscenes.org/nuimages) mini, BDD100K val subset | Night / rain / small distant objects; per-condition mAP |
| **Robotics** | Your own 100 tabletop photos, or Open X-Embodiment frames | Label graspable objects + affordance points (VLM pointing, e.g. Molmo) |
| **Video** | DAVIS 2017 val | Label frame 0 only, propagate with **SAM 2**; how many re-prompts per video? |
| **Your own** | Lab/industry images with permission | Define 3–5 classes; build a mini labeled dataset you'll actually use |

## Alternative projects (same "no big training" spirit)

1. **Linear-probe shoot-out** — freeze CLIP, SigLIP 2, DINOv2, DINOv3 (small variants); train only a
   logistic-regression head on a 10-class subset of a domain dataset (e.g., MedMNIST, EuroSAT).
   Which representation transfers best per domain, and with how few labels (1, 5, 10, 100 per class)?
2. **VLM robustness audit** — build a 200-question VQA set on counting, left/right, and small objects;
   evaluate 3 open VLMs; analyze hallucinations.
3. **Data curation with embeddings** — use CLIP/DINO embeddings to deduplicate, find near-duplicates
   across train/test, and surface mislabeled images in a public dataset.
4. **Synthetic data probe** — generate 500 images of a rare class with an open diffusion model;
   does adding them help a few-shot linear probe?

## Grading rubric

| Component | Weight |
|---|---|
| Working, reproducible pipeline (README + scripts) | 20% |
| Quantitative evaluation (mAP, PR, cost) done correctly | 25% |
| Week-2 improvements with ablations | 20% |
| Failure analysis & insight | 20% |
| Report and presentation clarity | 15% |

## Tips & pitfalls

- Fix the random seed and keep a **held-out GT split** you never look at while tuning prompts or thresholds.
- COCO bbox format is `[x, y, width, height]`; most detectors output `[x0, y0, x1, y1]`.
- OWLv2 base takes a few seconds per image on a laptop (Apple MPS) and less on a Colab T4 — 200 images
  is a coffee break, not a cluster job. Run once, save the JSON, and iterate on the saved predictions.
- Report licenses of the models and datasets you use; medical data must stay de-identified.
