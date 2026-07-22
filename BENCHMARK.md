# NCCT→CECT benchmark — models & integration

This repo holds the **competing models** for the NCCT→CECT synthesis benchmark.
Scoring lives in the sibling repo `../synthetic_CECT` (`benchmark.py` +
`metrics.py` + `orgFeatXGB_CTPhase/phase_eval.py`).

**Integration contract:** every model, however it trains, must emit synthetic CECT
volumes as **NIfTI on each test case's source grid** (same shape as the real CECT,
values in **HU**) and a manifest CSV with columns
`gen_path,real_path,mask_path,target_phase`. `benchmark.py` then scores all models
identically. Nothing about a model's internals matters to the harness — only that
it produces those volumes for the shared test cases.

**Fairness:** all models train and evaluate on the *same* case-level split
(`../synthetic_CECT/benchmark/split.json`, seed 42) and are scored on the same
test cases, masks, and HU window. Retraining these repos on our data at our scale
does **not** reproduce their papers' headline numbers — this is a controlled
same-data comparison, not a reproduction.

---

## Models

### Pre-existing (GAN / CNN family)
- `Ea-GANs/` — edge-aware GAN (cross-modality synthesis).
- `UnpairedImageTranslation/` — CycleGAN-family unpaired translation.
- `VCE_CESM/` — virtual contrast enhancement.
- `pix2pix3D-CT/` — 3D pix2pix for CT.
- ⚠️ `Ea_GANs/` — appears to be a broken nested clone (contains only
  `Ea_GANs/Ea_GANs`); `Ea-GANs/` (hyphen) is the real one. Recommend removing.

### Added for SOTA + diffusion coverage (git submodules)

| dir | model | type | paper | entrypoints | input format |
|---|---|---|---|---|---|
| `SynDiff/` | SynDiff | adversarial **diffusion** | TMI'23, [icon-lab](https://github.com/icon-lab/SynDiff) | `train.py`, `test.py` | `.mat`, `(#img,W,H)`, values 0–1 |
| `CFPS-Diff/` | CFPS-Diff | conditional **diffusion** (NCCT→multiphase CECT) | MICCAI'25, [Kindyz](https://github.com/Kindyz/CFPS-Diff) | `main/train_CFPS-Diff_gen.py`, `main/Inference.py` | see `main/`; ships `Error_troubleshooting.txt` |
| `ResViT/` | ResViT | transformer-GAN | TMI'22, [icon-lab](https://github.com/icon-lab/ResViT) | `train.py`, `test.py` | pix2pix-style aligned pairs |
| `CyTran/` | CyTran | cycle **transformer** | Neurocomputing'23, [ristea](https://github.com/ristea/cycle-transformer) | `train.py`, `test.py`, `options/base_options.py` (data path) | CycleGAN/pix2pix-style; ships Coltea-Lung-CT-100W |

Notes:
- **ResViT and CyTran share the pix2pix/CycleGAN codebase** (options/, aligned-pair
  loaders) — one adapter pattern covers both. CyTran's README states it "is similar
  with CycleGAN-and-pix2pix".
- **SynDiff wants `.mat`** inputs (`(#images, W, H)`, [0,1]) — its adapter must
  convert NCCT/CECT slices to `.mat` and convert generated slices back to HU NIfTI.
- **CFPS-Diff** is the roughest repo (Python 3.8, torch 2.0+cu118, troubleshooting
  file). Do it last. It targets multiphase output — we score only the venous phase.
- Deps per repo stay inside each subdir; do not merge into a global env (SynDiff
  torch>=1.7 vs CFPS-Diff torch==2.0 conflict).

### Watch-list — no usable public code yet (add when released)
- **PHASOR** (arXiv'26) — volumetric video-diffusion, "code on acceptance". The
  current SOTA frontier; strongest candidate to add next.
- **SC-BBDM** (Brownian-bridge CTA, arXiv'25) — no code link.
- **MAN-GAN** (SciRep'25), **IEEE 10838281**, **MLMI'25 §14**, **Acad.Radiology
  '24** — no public code; would require reimplementation (out of scope).

---

## Per-model adapter (to be added: `<model>/adapter_<model>.py`)

Each adapter does three things:
1. Read `../synthetic_CECT/benchmark/split.json` → the train/val/test cases.
2. Convert our NIfTI cases to the repo's expected input, train, run inference.
3. Export synthetic CECT **NIfTI on the source grid (HU)** + a manifest CSV.

Reuse `../synthetic_CECT/infer_volume.py`'s tiling/stitching where a repo emits
slices or patches rather than whole volumes.

## Scoring (after any model produces a manifest)

```bash
cd ../synthetic_CECT
python benchmark.py --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --manifest ours=../out_synthesis_train/literature_baseline_l1_organ_curriculum/phase_infer/manifest.csv \
    --manifest resvit=../ncct2cect/ResViT/<out>/manifest.csv \
    --baseline ours --out analysis/benchmark
# → analysis/benchmark/master_table.md
```
