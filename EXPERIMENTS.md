# Experiment Log

Keep one section per meeting. Paste the tables that `synth-report` writes to
`evidence/*.md`, link its plots and the pools' contact sheets, and explain what
you see. A result that shows no effect is still a result: say so and say why.

## Meeting 1 — Real-data baselines

- **Question:** How much do oversampling and ordinary augmentation help Egyptian
  Mau recall compared with the imbalanced data?
- **How `oversample()` works:**
- **How `augment()` works, and why these transforms:**
- **Commands:**
- **Results:** (table and `evidence/m1-*.png`)
- **What I see:**

## Meeting 2 — Text-to-image generation

- **Question:**
- **Setting changed in each pool (one at a time):**
- **Commands:**
- **Contact sheets:** (good and bad images, with the failure types you name)
- **Results:** (table and plot)
- **What I see:**
- **Pool I take forward, and why:**

## Meeting 3 — Generation from training photos

- **Question:** How close to the real photos should the synthetic images be?
- **img2img strength values and IP-Adapter scale values tried:**
- **Commands:**
- **Contact sheets:**
- **Results:**
- **What I see:**

## Meeting 4 — Editing training photos

- **Question:** Does the classifier need new cats or new backgrounds?
- **Background prompts and outpainting scales tried:**
- **Commands:**
- **Contact sheets:**
- **Results:**
- **What I see:**

## Final comparison

- **Conditions frozen before using the test photos, and why each was chosen:**
- **Command:**
- **Results:** (`final/summary.md`, `final/comparison.png`, `final/differences.png`)
- **Recommendation:** use, revise, or reject synthetic data for this task.
