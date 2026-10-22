# Synthetic Data Generation with Diffusion Models

Starter repository for the course
[Synthetic Data Generation with Diffusion Models](https://patrykbart.github.io/teaching/synthetic-data-generation-with-diffusion-models/).
Every student answers the same question on the same data:

> Can images made by diffusion models improve recognition of Egyptian Mau, an
> underrepresented cat breed, without harming the other seven breeds?

The repository already downloads the data, fine-tunes the classifier, and runs
the diffusion models. Your work is to run controlled experiments, look at the
images, and explain what changes the classifier and why.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), create
your own repository with **Use this template**, clone it, and run:

```bash
uv sync --locked
uv run synth-data
uv run pytest
```

`synth-data` downloads Oxford-IIIT Pet (about 800 MB) into `data/` and checks
the course split, and `pytest` checks the starter code.

### Hardware

Measured on an RTX 5070 Ti laptop GPU (12 GB); slower GPUs take
proportionally longer.

| Task                                  |  Time | GPU memory | First download |
| ------------------------------------- | ----: | ---------: | -------------: |
| Train one run (5 seeds)               | 1 min |          – |          21 MB |
| Text-to-image, SD 1.5, 110 images     | 3 min |     3.2 GB |     about 3 GB |
| Text-to-image, SDXL-Turbo, 110 images | 1 min |     7.7 GB |     about 7 GB |
| img2img (strength 0.5), 110 images    | 2 min |     3.2 GB |  (uses SD 1.5) |
| IP-Adapter, 110 images                | 3 min |     4.4 GB |   + about 3 GB |
| Inpainting or outpainting, 110 images | 3 min |     3.2 GB |     about 3 GB |

Models download into `~/.cache/huggingface` the first time you use them. Add
`--low-vram` to `synth-generate` on GPUs with less than 6 GB. Training also runs
on a CPU, only slower; generating images on a CPU is impractically slow.

Without a suitable GPU, use the faculty computers or a free cloud GPU such as
Google Colab or Kaggle. In a notebook cell there, run:

```text
!git clone <your-repository-url> project
%cd project
!pip install -e .
!synth-data
```

and prefix every later command with `!`, for example
`!synth-train --name m1-real --method real`. Commit and push your results from
the notebook, or download them, before the session ends: its disk is erased.

## The Fixed Experiment

- **Data:** eight cat breeds from Oxford-IIIT Pet, split in `data/split.csv`.
  Each breed has 70 training photos except Egyptian Mau, which has 15. Every
  breed has 20 validation and 30 test photos.
- **Budget:** every method adds exactly 55 Egyptian Mau training examples, so
  Egyptian Mau reaches the 70 photos of the other breeds.
- **Classifier:** EfficientNet-B0 pretrained on ImageNet, fine-tuned for 8
  epochs. Each experiment trains 5 models with seeds 0–4.
- **Metrics:** Egyptian Mau recall (the share of Egyptian Mau photos recognised)
  and macro-F1 over all eight breeds (to catch harm to the other breeds).
- **Validation and test:** Meetings 1–4 evaluate on the validation photos. The
  test photos are used once, by `synth-final`, after Meeting 4.

## Commands

| Command | What it does |
| --- | --- |
| `uv run synth-train --name NAME --method METHOD` | trains 5 seeds, saves `runs/NAME/` |
| `uv run synth-generate --name NAME [options]` | makes a pool of images in `pools/NAME/` |
| `uv run synth-final --condition ...` | the single evaluation on the test photos |

`--method` is `real` (no added examples), `oversample`, `augment`, or
`synthetic` with `--pool NAME`. Each pool has a `manifest.jsonl` line for every
attempt, a `pool.json` with its settings, and a `contact-sheet.jpg` of its first
24 images. Run any command with `--help` for all options.

## Results

Every `synth-train` run writes `runs/NAME/results.json` with the validation
metrics of every seed. The tables and plots are yours to make from these files,
for example with pandas and matplotlib, which are installed. Save them in
`evidence/`.

Describe a run with your own variables using `--tag KEY=VALUE`, so that you can
later group and plot runs by them:

```bash
uv run synth-train --name m3-strength-0.5 --method synthetic --pool m3-strength-0.5 \
    --tag mode=img2img --tag strength=0.5
```

Numbers stay numbers, so `strength` can go on a plot axis. Running the same
command again with other tags adds them without training again, and you can
also edit `results.json` by hand. The file looks like this (the values are an
example):

```json
{
  "name": "m3-strength-0.5",
  "method": "synthetic",
  "pool": "m3-strength-0.5",
  "model": "efficientnet_b0",
  "epochs": 8,
  "batch_size": 64,
  "budget": 55,
  "split": "val",
  "tags": {"mode": "img2img", "strength": 0.5},
  "seeds": [
    {"seed": 0, "egyptian_mau_recall": 0.75, "macro_f1": 0.9, "accuracy": 0.9,
     "recall": {"Abyssinian": 0.95, "Bengal": 0.85}}
  ]
}
```

`recall` lists all eight breeds. Each `runs/NAME/seed-K/` folder also has
`metrics.json` with the confusion matrix, `predictions.csv` with the prediction
for every validation photo, and `run.json` with the exact training data.

## Meeting 1 — Real-Data Baselines

Write `oversample()` and `augment()` in `src/synthdata/balance.py`, then:

```bash
uv run synth-train --name m1-real --method real
uv run synth-train --name m1-oversample --method oversample
uv run synth-train --name m1-augment --method augment
```

Then compare the three runs, seed by seed, in your own table and plot.

## Meeting 2 — Text-to-Image Generation

```bash
uv run synth-generate --name m2-base
uv run synth-train --name m2-base --method synthetic --pool m2-base
```

Then make new pools that change one setting each, and tag each run with the
setting you changed:

```bash
uv run synth-generate --name m2-guidance-3 --guidance 3
uv run synth-train --name m2-guidance-3 --method synthetic --pool m2-guidance-3 \
    --tag guidance=3
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--model` | `sd15` (Stable Diffusion 1.5) or `sdxl-turbo` | `sd15` |
| `--prompt` | text prompt; repeat the option to cycle through several | four varied prompts |
| `--negative-prompt` | what the image should not contain | drawings, text, extra limbs… |
| `--steps` | denoising steps of the sampler | 30 (SDXL-Turbo: 4) |
| `--guidance` | how strongly the image follows the prompt | 7.5 (SDXL-Turbo: 0) |
| `--scheduler` | sampler: `dpm`, `ddim`, `euler`, or `default` | `dpm` |
| `--seed` | seed of the first image; image *i* uses seed + *i* | 1000 |
| `--count` | images in the pool | 110 |

## Meeting 3 — Generation from Training Photos

Both modes start from the 15 Egyptian Mau training photos, used in turn.

```bash
uv run synth-generate --name m3-strength-0.5 --mode img2img --strength 0.5
uv run synth-generate --name m3-ip-0.5 --mode ip-adapter --ip-adapter-scale 0.5
uv run synth-train --name m3-ip-0.5 --method synthetic --pool m3-ip-0.5 \
    --tag mode=ip-adapter --tag scale=0.5
```

- `--strength` (img2img): how much of the photo is replaced by noise. Near 0
  the result is almost a copy; near 1 it is almost a new image. The model runs
  only steps × strength of the denoising steps.
- `--ip-adapter-scale` (ip-adapter): how strongly the photo steers a new
  image. The pose and background may change; the cat should look similar.

## Meeting 4 — Editing Training Photos

```bash
uv run synth-generate --name m4-backgrounds --mode inpaint
uv run synth-generate --name m4-outpaint-1.5 --mode outpaint --outpaint-scale 1.5
```

- `inpaint` keeps the real cat and paints a new background. The mask comes from
  the Oxford-IIIT Pet annotations; the contact sheet shows it in red. Use
  `--prompt` to choose the scene; by default six scenes are used in turn. The
  annotation of `Egyptian_Mau_165` marks no cat, so inpainting skips that photo
  and starts from the other 14.
- `outpaint` shrinks the photo into the centre and paints the surroundings, as
  if the camera moved back. `--outpaint-scale` sets how far: 1.25 to 2.

## After Meeting 4 — Final Comparison

Choose the best pool of each method using the validation results, freeze the
choice in `EXPERIMENTS.md`, and only then run:

```bash
uv run synth-final --condition real --condition oversample --condition augment \
    --condition "text-to-image=m2-base" --condition "img2img=m3-strength-0.5" \
    --condition "ip-adapter=m3-ip-0.5" --condition "background=m4-backgrounds" \
    --condition "outpaint=m4-outpaint-1.5"
```

It trains every condition with the same seeds, evaluates them once on the test
photos, and writes `final/results.json`: the test metrics of every seed, and the
difference of every condition from oversampling and augmentation with a 95%
interval that accounts for both the seeds and the choice of test photos. Make
your final table and plot from it.

## What to Commit

Commit `EXPERIMENTS.md`, your tables and plots in `evidence/`, `runs/`,
`final/`, and each pool's
`manifest.jsonl`, `pool.json`, and `contact-sheet.jpg`. Generated images,
downloaded data, and model weights stay local; `.gitignore` already excludes
them, and the manifests record how to recreate every image.

## Rules

- Change one setting at a time and keep the budget at 55.
- Do not delete pools or images you do not like: describe them.
- Generation uses only the 15 Egyptian Mau training photos; the tools enforce
  this.
- Use `synth-final` once. A second run stops unless the instructor agrees.

## Coding Agent

If you already use Codex or Claude Code, nothing else is required. Otherwise,
install [Antigravity CLI](https://antigravity.google/docs/cli/install/). You
must be able to explain every command, setting, and result you submit.
