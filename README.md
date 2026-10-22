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
Download also the `model.pt` files of the runs you will test; they are not
committed.

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
- **Validation and test:** every run is evaluated on the validation photos,
  which you use to compare settings and choose. At the end of each meeting,
  `synth-test` evaluates the runs you chose on the test photos, and these test
  results summarise the meeting.

## Commands

| Command | What it does |
| --- | --- |
| `uv run synth-train --name NAME --method METHOD` | trains 5 seeds, saves `runs/NAME/` |
| `uv run synth-generate --name NAME --prompt PROMPT [options]` | makes a pool of images in `pools/NAME/` |
| `uv run synth-test --run NAME` | evaluates the saved models of a run on the test photos |

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
for every validation photo, `run.json` with the exact training data, and
`model.pt`, the trained model (about 16 MB, not committed) that `synth-test`
evaluates on the test photos.

## Meeting 1 — Real-Data Baselines

Write `oversample()` and `augment()` in `src/synthdata/balance.py`, then:

```bash
uv run synth-train --name m1-real --method real
uv run synth-train --name m1-oversample --method oversample
uv run synth-train --name m1-augment --method augment
```

Then compare the three runs, seed by seed, in your own table and plot, and
write down what the validation results show. Only then test the three
baselines:

```bash
uv run synth-test --run m1-real
uv run synth-test --run m1-oversample
uv run synth-test --run m1-augment
```

`synth-test` evaluates the saved model of every seed on the 240 test photos and
writes `runs/NAME/test-results.json` in the same format as `results.json`, plus
`test-metrics.json` and `test-predictions.csv` in every seed folder. Add
`--seeds 0` to test only the model of seed 0. Every meeting ends the same way:
choose on the validation results, write the choice down, then test the chosen
runs and report their test values. The baselines' test results are the
reference for every later meeting.

## Meeting 2 — Text-to-Image Generation

`--prompt` has no default: write the description yourself, and repeat the option
to cycle through several prompts.

```bash
uv run synth-generate --name m2-base --prompt "<your prompt>"
uv run synth-train --name m2-base --method synthetic --pool m2-base
```

Then make new pools that change one setting each, and tag each run with the
setting you changed:

```bash
uv run synth-generate --name m2-guidance-3 --prompt "<your prompt>" --guidance 3
uv run synth-train --name m2-guidance-3 --method synthetic --pool m2-guidance-3 \
    --tag guidance=3
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--model` | `sd15` (Stable Diffusion 1.5) or `sdxl-turbo` | `sd15` |
| `--prompt` | text prompt; repeat the option to cycle through several | none: required |
| `--negative-prompt` | what the image should not contain | drawings, text, extra limbs… |
| `--steps` | denoising steps of the sampler | 30 (SDXL-Turbo: 4) |
| `--guidance` | how strongly the image follows the prompt | 7.5 (SDXL-Turbo: 0) |
| `--scheduler` | sampler: `dpm`, `ddim`, `euler`, or `default` | `dpm` |
| `--seed` | seed of the first image; image *i* uses seed + *i* | 1000 |
| `--count` | images in the pool | 110 |

Choose the pool you take forward on the validation results, write the choice
down, and test its run with `uv run synth-test --run NAME`.

## Meeting 3 — Generation from Training Photos

You do not pass a photo: both modes start from the 15 Egyptian Mau training
photos, used in turn, and the command prints which ones. Add
`--source Egyptian_Mau_117` (repeat it for more) to start only from the
training photos you choose; any other photo is refused. Both modes still need
`--prompt`: keep it the same within a sweep, for example the prompt of the pool
you took forward from Meeting 2.

```bash
uv run synth-generate --name m3-strength-0.5 --mode img2img --strength 0.5 \
    --prompt "<your prompt>"
uv run synth-generate --name m3-ip-0.5 --mode ip-adapter --ip-adapter-scale 0.5 \
    --prompt "<your prompt>"
uv run synth-train --name m3-ip-0.5 --method synthetic --pool m3-ip-0.5 \
    --tag mode=ip-adapter --tag scale=0.5
```

- `--strength` (img2img): how much of the photo is replaced by noise. Near 0
  the result is almost a copy; near 1 it is almost a new image. The model runs
  only steps × strength of the denoising steps.
- `--ip-adapter-scale` (ip-adapter): how strongly the photo steers a new
  image. The pose and background may change; the cat should look similar.

Choose the best run of each mode on the validation results, write the choice
down, and test both runs with `synth-test`.

## Meeting 4 — Editing Training Photos

```bash
uv run synth-generate --name m4-backgrounds --mode inpaint \
    --prompt "<scene 1>" --prompt "<scene 2>" --prompt "<scene 3>"
uv run synth-generate --name m4-outpaint-1.5 --mode outpaint --outpaint-scale 1.5 \
    --prompt "<your prompt>"
```

- `inpaint` keeps the real cat and paints a new background. The mask comes from
  the Oxford-IIIT Pet annotations; the contact sheet shows it in red. Describe
  the whole photo in `--prompt`, the cat and its new surroundings, and repeat
  the option to use several scenes in turn. The annotation of
  `Egyptian_Mau_165` marks no cat, so inpainting skips that photo and starts
  from the other 14.
- `outpaint` shrinks the photo into the centre and paints the surroundings, as
  if the camera moved back. `--outpaint-scale` sets how far: 1.25 to 2.

As in Meeting 3, both modes start from the training photos themselves and print
which ones; `--source` chooses some of them. As in every meeting, choose the
best run of each mode on the validation results, write the choice down, and
test both runs.

## Meeting 5 — Final Comparison

No new tests are needed. Collect the test results of the runs you chose in
Meetings 1–4 into your final table and plot, compute the differences between
methods yourself, and compare the ranking with the validation results.

## What to Commit

Commit `EXPERIMENTS.md`, your tables and plots in `evidence/`, `runs/`, and each
pool's `manifest.jsonl`, `pool.json`, and `contact-sheet.jpg`. Generated images,
downloaded data, and model weights, including the `model.pt` of every run, stay
local; `.gitignore` already excludes them, and the manifests record how to
recreate every image.

## Rules

- Change one setting at a time and keep the budget at 55.
- Do not delete pools or images you do not like: describe them.
- Generation uses only the 15 Egyptian Mau training photos; the tools enforce
  this.
- Choose on the validation results and write the choice down before testing.
  Test only the chosen runs, and do not revise a choice after its test result.

## Coding Agent

If you already use Codex or Claude Code, nothing else is required. Otherwise,
install [Antigravity CLI](https://antigravity.google/docs/cli/install/). You
must be able to explain every command, setting, and result you submit.
