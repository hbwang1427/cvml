# Take-home: Tracing a Concept from Pixels to Words

**Level:** senior undergraduate / early graduate · **Time limit:** 6 hours · **Team:** individual
**Compute:** a laptop (Apple Silicon / CPU) or a free Colab T4. **No model training** except linear probes.

Accompanies [Lecture 02 — Interpretability](../../lectures/02_interpretability/README.md).

This take-home uses the format of AI-safety research take-homes such as the Anthropic Fellows and
Constellation programs. You get a fixed time budget, a set of research questions, starter code you
should not trust blindly, and **pre-registered predictions** written *before* each experiment. Your work
is graded on whether your claims match your evidence, not on whether you guessed right.

## The research question

Every model in this course can answer *"is there a dog in this image?"*. We follow that one concept
through three kinds of models:

| Part | Model | What you measure |
|---|---|---|
| A | Vision encoders: CLIP ViT-B/32, DINOv2-S, and a random-weight CLIP | Where the model "looks" (saliency) and where object presence is encoded (probes) |
| B | Vision-language model: SmolVLM-256M (and -500M) | Where the LLM reads the image, and whether its hidden states know when it hallucinates |
| C | Language model: GPT-2 small | Which heads move information to the answer (activation patching on a known circuit) |

At each step, a result that only *looks* convincing (level 1–2 on the lecture's evidence ladder) is not
enough. Try to reach level 3: show with a control or an intervention that the information is actually
**used**.

> **Important.** Do not assume the lecture figures, the cited papers or the starter code are correct.
> The lecture figures come from 1–9 examples each. The starter code makes choices (baselines, splits,
> metrics, prompts) that you should question. If you find a bug or a bad default, fixing it and showing
> what changed counts as a result.

## Setup (≈ 20 min)

```bash
pip install torch transformers pillow matplotlib scikit-learn scipy
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d data

# 400 random COCO val2017 images with ALL their labels (≈ 60 MB, a few minutes)
python make_data.py --ann data/annotations/instances_val2017.json --n 400 --out data/coco400
```

Check that everything runs. Times are for an M-series MacBook running one script at a time (scripts that
share the GPU slow each other down a lot); the first run also downloads ≈ 2 GB of models.

| Command | Time | Output |
|---|---|---|
| `python part_a_probe.py` | ~1 min | `results/probe/probe_by_layer.png`, `probe_results.json` |
| `python part_a_saliency.py --n 30` | ~15 min | `results/saliency/saliency_checks.png`, `saliency_summary.json` |
| `python part_b_vlm.py pope --n 150` | ~5 min | `results/vlm/pope.csv`, `pope_hidden.npy`, `pope_summary.json` |
| `python part_b_vlm.py knockout --n 40` | ~4 min | `results/vlm/knockout.png`, `knockout.npy` |
| `python part_c_llm.py --corruption swap` | ~1 min | `results/llm/patching_swap_t0.png`, `summary_swap_t0.json` |

All shared code (data loading, hooks, saliency, knockout, patching) lives in
[`interp_utils.py`](interp_utils.py). Read it first.

## Deliverables

1. **Slides** (at most 15, PDF or HTML). Suggested structure: 1 title slide, 2 slides per question, 1
   slide on what you would do next. Only results on the slides are graded.
2. **Presentation**: a 15-minute video *or* a live talk. Present it as a weekly update to a mentor who
   has read the lecture. Spend the time on surprising or confusing results and on your reasoning; don't
   read the slides aloud.
3. **Code ZIP** (≤ 100 MB): everything needed to reproduce every plot and table, including your
   prompts. No model weights, cached features or image data.
4. **AI session logs** for any AI assistant you used (for Claude Code, the JSONL transcripts from
   `~/.claude/projects/`). AI use is allowed and expected. The logs let us credit your own judgment:
   choosing directions, catching mistakes, deciding what to run.

**Write the "A" answers first and do not edit them later.** Put a timestamp on each one or commit it to
git before running the matching experiment.

## Questions

Each question is worth the same. Try to give some answer to all seven, then use the remaining time to
improve the most interesting ones. Suggested time split: Part A 1.5 h · Part B 1.5 h · Part C 1 h ·
Q6–Q7 0.5 h · slides 1 h.

### Q1 — Do saliency maps explain CLIP? (Part A1)

**Q1A (pre-register).** Of gradient, gradient × input and integrated gradients, which do you expect to
(i) change most when CLIP's top blocks are randomised and (ii) give the steepest deletion curve? Why?

**Q1B.** Run `part_a_saliency.py` on at least 30 correctly classified images and report both tests with
uncertainty. Then pick **one methodological choice** that could change the conclusion and test it.
Options: the deletion fill (`--fill grey` vs `blur`), the integrated-gradients baseline, or pixel-level
vs patch-level aggregation of the map. Which method would you trust, and for what purpose?

### Q2 — Is object presence encoded, and is it used? (Part A2)

**Q2A (pre-register).** Sketch the probe AUROC-vs-layer curves you expect for CLIP, DINOv2 and the
random-weight CLIP. Which of the five classes should be easiest to decode, and why?

**Q2B.** Run `part_a_probe.py`. Explain any result that looks odd. One hint: look at layer 0 and ask what
the [CLS] token contains before any attention. Are the differences between models larger than the
confidence intervals? Then **climb one level of the ladder**: test whether CLIP *uses* the direction your
probe found. One option: remove the probe direction from the final [CLS] embedding before the projection
and measure the change in zero-shot P(class). Compare against removing random directions of the same
norm. The starter code does not do this for you.

### Q3 — Where does the VLM read the image? (Part B)

**Q3A (pre-register).** The lecture shows that blocking text→image attention in layers 15–17 of
SmolVLM-256M removes the visual evidence, based on 9 questions. Do you expect this to hold on 40+ images?
For absent objects? For the 500M model, which has a different depth?

**Q3B.** Run `part_b_vlm.py knockout` on at least 40 images, then on `SmolVLM-500M-Instruct`, and compare
the two models by *relative* depth. Address at least one confounder. Examples: does blocking any window
just break the model (try a control question that needs no image)? Does the result depend on object
size, or on the question wording?

### Q4 — Does the VLM know when it hallucinates? (Part B)

**Q4A (pre-register).** Give at least two testable hypotheses for why a VLM says "yes" more often to
absent objects that *co-occur* with the scene (the POPE "adversarial" split). Give your credence for
each.

**Q4B.** Run `part_b_vlm.py pope` and report the false-positive rate per split with confidence intervals.
Using the saved hidden states (`pope_hidden.npy`, one vector per LLM layer at the answer position), fit
layer-wise probes for (a) ground-truth presence and (b) "the model's answer is wrong". Can you decode
the truth in cases where the model answers wrongly? Compare with a baseline that only uses P(yes). Split
by image, not by question.

### Q5 — Does the IOI circuit replicate, and is it robust? (Part C)

**Q5A (pre-register).** Do you expect the same important heads when the corruption *swaps* the two names
as when it *replaces* them with new names (`--corruption abc`)? Across different templates?

**Q5B.** Run both corruptions on at least two templates. Then test **necessity**: mean-ablate the top-k
heads you found (you write this; average each head's output over a reference set of prompts) and measure
the logit difference. Compare with ablating k random heads. Which parts of Wang et al. (2023) replicate,
and which depend on your choices?

### Q6 — Synthesis

Make one table: for each part, give your strongest claim, the highest ladder level it reached (look /
decode / intervene / explain) and the main threat to its validity. Which method transferred well from
vision to VLMs to LLMs, and which one broke? Did you see the same "read the evidence, then decide" pattern
at more than one level?

### Q7 — Dream experiment

Design an interpretability-based **monitor** for a deployed VLM in a high-stakes domain, such as
radiology report generation or driving. The monitor should flag answers that come from the language prior
instead of the image. It should be ambitious but possible today with a lab's support. Include:

- the signal you would monitor and why it should generalise beyond your training questions;
- how you get ground truth for training and evaluation;
- how you show the signal is causal, not merely correlated with errors;
- what false-positive and false-negative rates would make it useful, and who acts on a flag;
- how the monitor could fail (distribution shift, a model optimised against it) and how you would detect that.

## Grading rubric

| Weight | Criterion | What earns it |
|---|---|---|
| 70% | Research answers (10% per question) | Reasoned pre-registrations; results reconciled with predictions; claims scoped to the evidence; confounders addressed; interventions for causal claims; correct code |
| 20% | Communication | Labelled plots with error bars or CIs, terms defined, plain writing in your own words, nothing overclaimed |
| 10% | Judgment | From the logs and the talk: you chose informative experiments, spent time well, and caught errors (yours or the AI's) |

## Tips and pitfalls

- **Small n.** 400 images give ~120 test images per probe. A class with 15 positives has a very wide CI.
  Report the n for every number.
- **Split by image.** POPE produces 4 questions per image, and image features leak between questions.
- **Baselines everywhere.** Random-weight model, shuffled labels, random directions, random heads,
  random pixel order. A number without a baseline is hard to interpret.
- **Know what your hook does.** The knockout blocks text→image attention only; image tokens still
  attend to each other. `run_patched` overwrites one position. Check on a single example that the hook
  fires and changes what you expect.
- **Prompt details matter.** Some COCO class names read oddly in a question ("Is there a dining table
  ...", "a apple"). Check how often this happens and whether it matters.
- **Cache once, iterate fast.** Features, hidden states and P(yes) values are saved to `results/`.
  Change the analysis, not the forward passes.

## Optional extensions

1. **Sparse autoencoder features.** Load a public SAE for GPT-2 small (for example the `sae_lens`
   release `gpt2-small-res-jb`). Find the features active on the S2 token in IOI and test one with an
   intervention.
2. **Steering the VLM.** Add your Q4 probe direction to the residual stream during the yes/no answer.
   Can you reduce hallucinations without hurting accuracy on present objects?
3. **Medical track.** Repeat Part B on a small chest-X-ray or dermatology set with MedGemma or a general
   VLM. Does the image get read in the same layers when the domain is far from the training data?
