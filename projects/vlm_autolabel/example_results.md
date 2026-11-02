| Method | boxes | AP@[.5:.95] | AP@.5 | AR@100 |
|---|---:|---:|---:|---:|
| A detector (OWLv2) | 499 | 0.426 | 0.622 | 0.507 |
| B VLM grounding (Florence-2) | 176 | 0.094 | 0.131 | 0.257 |
| C detector + VLM verifier | 499 | 0.417 | 0.590 | 0.507 |
| D ensemble (OWLv2 + Florence-2) | 573 | 0.373 | 0.535 | 0.509 |
| E harness (tag → ensemble → verify) | 316 | 0.320 | 0.407 | 0.355 |

Per-class AP@.5:

| Method | person | bicycle | car | traffic light | dog |
|---|---:|---:|---:|---:|---:|
| A | 0.65 | 0.50 | 0.68 | 0.48 | 0.79 |
| B | 0.38 | 0.01 | 0.13 | 0.14 | 0.00 |
| C | 0.67 | 0.50 | 0.53 | 0.50 | 0.74 |
| D | 0.69 | 0.25 | 0.52 | 0.42 | 0.79 |
| E | 0.66 | 0.50 | 0.24 | 0.21 | 0.43 |
