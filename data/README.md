# Data — NVIDIA Nemotron Model Reasoning Challenge

The CSVs in this directory are unmodified copies from the Kaggle competition data release.

## Source

- **Competition:** NVIDIA Nemotron Model Reasoning Challenge
- **URL:** https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- **Provider:** NVIDIA, hosted on Kaggle

## License

These data files are released under the **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)** license per the competition's data license.

**Attribution required when redistributing or building on these files:** credit NVIDIA as the source, link to the competition page above, link to the CC BY 4.0 license, and indicate whether changes were made. The files in this directory are **unmodified** copies of the original competition release.

## Files

| File | Size | Rows | Schema | Purpose |
|---|---|---|---|---|
| `train.csv` | 3.0 MB | 9,500 | `id,prompt,answer` | Full training set with ground-truth answers |
| `test.csv` | 1.5 KB | 3 | `id,prompt` | **Sample preview only** — the real test set is hidden, replaced at scoring time by several hundred unseen puzzles |

**Important:** `test.csv` in the repo is the 3-row preview that ships with the competition. Do not assume it is representative of scoring distribution. The real test set is held out and applied by Kaggle's grader at submission time.

## Puzzle categories

Every puzzle is **few-shot rule induction**: the model sees examples of a hidden rule and must generalize to a new case. Six categories, evenly balanced across `train.csv` (~1,555–1,602 each):

| Category | Count | Few-shot examples | Answer format | Example answer |
|---|---|---|---|---|
| `bit_ops` | 1,602 | 7–10 binary pairs | 8-char binary string | `10010111` |
| `cipher` | 1,576 | 2–4 text pairs | 3–5 lowercase words | `cat imagines book` |
| `gravity` | 1,597 | 3–5 time/distance points | Decimal (mostly 2dp) | `154.62` |
| `numeral` | 1,576 | 2–4 number/Roman pairs | Roman numerals (1–100) | `XXXVIII` |
| `symbol` | 1,555 | 2–4 symbol equations | 1–4 chars from 36 unique symbols | `@&` |
| `unit_conv` | 1,594 | 3–5 measurement pairs | Decimal (always 2dp) | `16.65` |

All puzzles are framed as "In Alice's Wonderland..." scenarios. See `prepare.py` for the validation-split logic and `program.md` § "What you can modify" for the autoresearch-loop framing of these categories.

## Modifications

None. The files committed here match what `train.csv` and `test.csv` contain when downloaded from the Kaggle competition page on 2026-04-03 (per commit `a2382ba`, "Add competition data files").

If a future change derives synthetic data, expanded splits, or filtered subsets from these files, that derived data belongs in a separate file (e.g., `data/synthetic.csv`, `data/val_split.json`) with its own provenance note — not in `train.csv` / `test.csv`. Keeping the originals untouched preserves the unmodified-copies guarantee that simplifies CC BY 4.0 attribution.

## Validation split

`prepare.py` derives a 30-puzzle held-out validation split (5 per category) from `train.csv` and writes it to `data/val_split.json`. That file is run-output, not source data; it is regenerable from `train.csv` and is therefore allowed to live in `data/` without affecting the redistribution story.
