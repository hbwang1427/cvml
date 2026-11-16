# Take-home: What Has a Vision Model Learned?

**Level:** senior undergraduate / early graduate · **Time limit:** 6 hours · **Team:** individual
**Compute:** a laptop (Apple Silicon / CPU) or a free Colab T4. **No model training** except linear probes.

Accompanies [Lecture 02 — Interpretability](../../lectures/02_interpretability/README.md).

## The research question

Five image encoders with similar sizes were trained in very different ways:

| Model | Training signal | Architecture |
|---|---|---|
| `resnet50` | supervised: 1.28M ImageNet images with 1,000 class labels | CNN, 4 stages / 16 blocks |
| `vit_sup` | supervised: 14M ImageNet-21k images with 21,000 class labels | ViT-B/16 |
| `clip` | contrastive: 400M web images paired with their captions | ViT-B/32 |
| `dinov2` | self-supervised: 142M images, no labels and no text | ViT-S/14 |
| `clip_random`, `resnet50_random` | none: the same architectures with random weights | |

**What did each one learn, and how much of that comes from the training signal rather than the
architecture?** You will look at the representation at four scales: single units (Q1), the final
embedding space (Q2), every layer (Q3), and the invariances and spatial structure the training objective
produced (Q4). Then you will test the most popular explanation tool, the saliency map (Q5), and follow
the representation into a vision-language model (Q6).

At each step, ask how strong your evidence is. The lecture's **evidence ladder** has four levels:
*look* (a picture), *decode* (a probe can read it), *intervene* (changing it changes the output) and
*explain* (a mechanism that predicts new cases). A random-weight model, shuffled labels, or a trivial
baseline will often reproduce half of a "finding". Run those controls before you draw conclusions.

> **Do not trust the starter code blindly.** It makes choices (thresholds, pooling, crops, metrics, label
> definitions) that change the answers. If you find a bad default or a bug, showing what changes when you
> fix it counts as a result.

## Setup (≈ 20 min)

```bash
pip install torch torchvision transformers pillow matplotlib scikit-learn scipy
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d data

# 1,000 random COCO val2017 images with all labels and polygon masks (≈ 160 MB, a few minutes)
python make_data.py --ann data/annotations/instances_val2017.json --n 1000 --out data/coco1k
```

Check that everything runs. Times are for an M-series MacBook running **one script at a time**: scripts
that share the GPU slow each other down a lot. The first run also downloads ≈ 2.5 GB of models.

| Question | Command | Time | Main outputs (in `results/`) |
|---|---|---|---|
| Q1 | `python q1_units.py --model resnet50` | ~3 min | `units/filters_*.png`, `examples_*_layer{1-4}.png`, `dissection_*.json` |
| Q1 | `python q1_units.py --model resnet50_random` | ~10 min | same, for the control; `units/dissection.png` compares both |
| Q2 | `python q2_embeddings.py` | ~2 min | `embed/tsne.png`, `retrieval_*.png`, `embed_summary.json` |
| Q3 | `python q3_layers.py` | ~4 min | `layers/probe_by_layer_cls.png`, `cka.png` |
| Q4 | `python q4_invariance.py invariance` | ~2 min | `invariance/invariance.png`, `invariance.json` |
| Q4 | `python q4_invariance.py patches` | ~1 min | `invariance/patch_pca.png`, `patch_pca.json` |
| Q5 | `python q5_saliency.py --n 30` | ~15 min | `saliency/saliency_checks.png`, `saliency_summary.json` |
| Q6 | `python q6_vlm.py knockout --n 40` | ~4 min | `vlm/knockout.png` |
| Q6 | `python q6_vlm.py pope --n 150` | ~5 min | `vlm/pope_summary.json`, `pope_hidden.npy` |

All shared code (data, masks, the `Encoder` wrapper, CKA, saliency, VLM hooks) lives in
[`interp_utils.py`](interp_utils.py). Read it first. Features are cached in `results/`, so re-running an
analysis is fast.

## How to answer

Every question has two parts:

- **A — prediction (before running anything).** Write down what you expect and why, in a few sentences.
  Put a timestamp on it or commit it to git *before* running the matching experiment, and do not edit
  it afterwards. Being wrong is fine. Being wrong and noticing why is often the most interesting result.
- **B — experiment.** Run the experiment, report numbers with uncertainty (CI or standard error) and n,
  compare them with your prediction, and name the main threat to your conclusion.

## Deliverables

1. **Slides** (at most 15, PDF or HTML). Suggested structure: 1 title slide, about 2 slides per question,
   1 slide on what you would do next.
2. **Presentation**: a 15-minute video *or* a live talk, as if giving a research update to your advisor.
   Spend the time on surprising results and on your reasoning; don't read the slides aloud.
3. **Code ZIP** (≤ 100 MB): everything needed to reproduce every plot and table. No model weights, cached
   features or image data.
4. **A short note on AI use** (a few lines): if you used an AI assistant, say what for, and give one
   example of something it got wrong that you caught. AI use is allowed.

## Questions

Suggested time split: Q1–Q4 45 min each · Q5 40 min · Q6 30 min · Q7 20 min · slides 1 h.

### Part A — What has the encoder learned?

#### Q1 — What do individual units detect, and how does that change with depth?

Background: a **unit** is one channel of a layer's output, so one feature map. Three classic tools:
the first-layer filters (the only weights you can look at directly), **dataset examples** (the image
regions where a unit fires most), and **network dissection** (how well the unit's top-1% activation
region overlaps the mask of a labelled concept, as an IoU).

**Q1A.** What do you expect the first-layer filters of a trained ResNet-50 to look like? What should the
top dataset examples show for units in `layer1` versus `layer4`? Roughly what fraction of units in each
stage do you expect to be "object detectors"? And for the random-weight ResNet?

**Q1B.** Run `q1_units.py` for both models. Describe the filters and 3–4 units per stage in your own
words. Then look critically at the dissection numbers:

- Does the random network also have "detectors"? For which concepts, and why those? (Look at what the
  images of that concept have in common.)
- The default IoU threshold (0.04) comes from Bau et al. (2017), who used a dataset with many concept
  types. Is it sensible for COCO object masks, and for 7 × 7 `layer4` maps upsampled to 56 × 56? Vary
  `--iou` or `--quantile` and report how the per-stage curve changes.
- COCO only labels objects. Which units are "unlabelled" just because their concept (colour, texture,
  part) has no mask? Give two examples from the dataset-example figures.

#### Q2 — How is the final embedding space organised?

Background: an embedding is good for a task if images that should be treated alike are close together.
Three standard ways to check this: **k-nearest-neighbour** classification (is the neighbourhood
consistent?), a **linear probe** (is the label linearly decodable?), and a **2-D map** (t-SNE / UMAP)
for looking.

**Q2A.** Rank the five encoders by expected kNN accuracy on the image's dominant supercategory (person,
animal, vehicle, food, ...). Should kNN and linear probe accuracy agree? What will the random-weight model
score, and what will its nearest neighbours have in common?

**Q2B.** Run `q2_embeddings.py`. Are the differences between trained models larger than the confidence
intervals? Use the retrieval figures to describe *what kind* of similarity each model uses (object,
scene, colour, layout, viewpoint). Then go one step further. Options:

- the label is noisy (the largest object is not always what the image is "about"): change `--min-frac`
  and see whether the ranking changes;
- t-SNE can create clusters that are not there and hide distances between them. Re-run it with two
  other perplexities, or with PCA, and say which of your visual impressions survive;
- find the 10 test images that every model gets wrong. What do they have in common?

#### Q3 — How does the representation change with depth, and do different models learn similar ones?

Background: a **layer-wise probe** asks where information becomes linearly decodable. **CKA** (centred
kernel alignment) compares two representations of the same images: it is 1 if they are the same up to
a rotation and a uniform scaling, and near 0 if unrelated. It needs no labels.

**Q3A.** Sketch the probe AUROC-vs-depth curves you expect for a large class (person) and a small one
(dog). Will CKA between CLIP and DINOv2 be highest at matching depths? Between CLIP and its random-weight
version?

**Q3B.** Run `q3_layers.py`.

- The ViT curves start at exactly 0.5 at layer 0 with `--pool cls`, but the ResNet does not. Explain
  why. (What does the [CLS] token contain before any attention?) Re-run with `--pool mean`.
- Are differences between models larger than the CIs and the shuffled-label band?
- The `vit_sup` CKA maps show a sharp block: its last six layers look unlike every layer of every other
  model. Find out why. One hypothesis: a few dimensions with very large activations dominate linear CKA.
  Test it, for example by standardising each dimension, or by removing the top few dimensions by
  variance before computing CKA.
- **Climb one level of the ladder:** a probe shows that information is *present*, not that the model
  *uses* it. Pick one class and test use, e.g. remove the probe direction from CLIP's final embedding
  and measure the change in zero-shot P(class), compared with removing random directions. The starter
  code does not do this.

#### Q4 — What did the training objective teach the model to ignore, and where are the objects?

Background: models become **invariant** to what their training treats as "the same". Supervised models
see random crops and flips. DINOv2 is trained to give the same embedding to two augmented views (crops,
colour jitter, grey-scale, blur), and CLIP uses almost no augmentation, only text. Patch features of
ViTs also carry spatial structure: a PCA of one image's patch features often separates the object from
the background.

**Q4A.** For each transformation in `q4_invariance.py` (flip, crop, grey-scale, hue shift, blur,
rotation, 4 × 4 patch shuffle), predict which model is most and least invariant, using what you know
about its training. Which model's patch PCA should match object masks best?

**Q4B.** Run both sub-commands.

- Did the training augmentations predict the invariances?
- The random-weight model looks *very* invariant to blur and rotation. Is it? Look at how the normalised
  score is defined, and at the retrieval column.
- Patch shuffle destroys shape but keeps local texture. What does each model's score say about shape
  versus texture (Geirhos et al. 2019)?
- For patch PCA, the default scores are only a little above a centred-square baseline. Is that a model
  failure or a metric failure? (Look at what the "ground truth" is: the union of every COCO object.)
  Propose and test a better evaluation.

### Part B — Where does the model look?

#### Q5 — Do saliency maps explain CLIP?

Background: lecture section 3. A saliency map should depend on the model (a randomisation test) and
should point at pixels that matter (a deletion test).

**Q5A.** Of gradient, gradient × input and integrated gradients, which do you expect to (i) change most
when CLIP's top blocks are randomised and (ii) give the steepest deletion curve? Why?

**Q5B.** Run `q5_saliency.py` on at least 30 correctly classified images and report both tests with
uncertainty. Then test **one methodological choice** that could change the conclusion: the deletion fill
(`--fill grey` vs `blur`), the integrated-gradients baseline, or pixel- vs patch-level aggregation of
the map. Which method would you trust, and for what purpose?

### Part C — From pixels to words

#### Q6 — Where does a vision-language model read the image, and does it know when it hallucinates?

Background: lecture section 4. A VLM feeds image tokens into a language model. **Attention knockout**
blocks text→image attention in chosen layers. POPE asks "Is there a {object}?" for present and absent
objects. Answer **one** of the two options.

**Q6A.** *Option 1:* the lecture shows that blocking layers 15–17 of SmolVLM-256M removes the visual
evidence, based on 9 questions. Will this hold on 40+ images, and for absent objects? *Option 2:* give
two testable hypotheses for why VLMs say "yes" more often to absent objects that **co-occur** with the
scene, with your confidence in each.

**Q6B.** *Option 1:* run `q6_vlm.py knockout` on at least 40 images and address one confounder (does
blocking any window just break the model? does the result depend on object size?). *Option 2:* run
`q6_vlm.py pope`, report false-positive rates per split with CIs, and fit layer-wise probes on the saved
hidden states (`pope_hidden.npy`) for "the object is present" and "the model's answer is wrong".
Split by image, not by question.

### Q7 — Synthesis and an experiment you would run next

1. Make one table: for each of Q1–Q6, give your strongest claim, the highest ladder level it reached
   (look / decode / intervene / explain), and the main threat to its validity.
2. Across Q1–Q4, what did the **training signal** change, and what did the **architecture** alone give
   you (the random-weight controls)?
3. Design one experiment, possible with today's tools, that would check whether a vision model used for
   a real task (e.g. chest X-ray triage, or pedestrian detection) relies on a **shortcut** such as a
   hospital marker or the image background. Say what you would measure, what control you would use, and
   what result would change your mind.

## Grading rubric

| Weight | Criterion | What earns it |
|---|---|---|
| 70% | Research answers (10% per question) | Reasoned predictions; results compared with predictions; claims scoped to the evidence; controls and baselines; correct code |
| 20% | Communication | Labelled plots with CIs and n, terms defined, plain writing in your own words, nothing overclaimed |
| 10% | Judgment | From the talk and the AI note: informative experiments chosen, time spent well, errors caught |

## Tips and pitfalls

- **Small n.** About 300 test images. A class with 15 test positives (dog) has a very wide CI. Report n
  for every number.
- **Split by image.** Q6 asks several questions per image, and features leak between them.
- **Baselines everywhere:** a random-weight model, shuffled labels, random directions, a centred square,
  chance level (1/K). A number without a baseline is hard to interpret.
- **The crop matters.** Every encoder sees the centre 224 × 224 square, so objects near the edges of
  wide images are cut off. `object_masks` uses the same crop, but `rec["classes"]` lists every object in
  the full image.
- **Know what your hook does.** The VLM knockout blocks text→image attention only. Check on one example
  that a hook fires and changes what you expect.
- **Cache once, iterate fast.** Change the analysis, not the forward passes.

## Optional extensions

1. **Feature visualisation.** Optimise an input image to maximise one ResNet-50 unit (activation
   maximisation, with jitter and blur regularisation as in Olah et al. 2017). Compare it with that
   unit's dataset examples from Q1.
2. **Concept directions (TCAV).** Build a "striped" concept from 50 crops, fit its direction in `layer3`,
   and measure how much the zebra logit depends on it.
3. **Language models.** [`ext_llm_ioi.py`](ext_llm_ioi.py) runs activation patching on GPT-2 small for
   the indirect-object identification task (lecture section 5). Does the circuit replicate when you
   change the corrupted prompt (`--corruption swap` vs `abc`) or the template?
4. **Medical images.** Repeat Q2 and Q4 on a small chest X-ray or dermatology set. Which invariances
   learned from natural images are wrong for medical images (colour, rotation, laterality)?
