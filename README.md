# MedGemma CT Reader

Feed a 3D CT volume to **MedGemma 1.5** and ask it questions.

[![tests](https://github.com/rekalantar/MedGemma-CT-Reader/actions/workflows/tests.yml/badge.svg)](https://github.com/rekalantar/MedGemma-CT-Reader/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

MedGemma 1.5 is the first open multimodal LLM that interprets volumetric CT and MRI. Its
vision encoder still only accepts 2D RGB images, so a volume becomes a sequence of axial
slices — each windowed into three channels, each costing 256 vision tokens, capped at 85
slices to stay inside the context budget.

This package is the preprocessing that makes that work, extracted and tested.

---

## 📖 Read the full walkthrough

**[MedGemma 1.5 Reads a CT Volume](ARTICLE_URL)**

How a 2D vision encoder ends up reading 3D data, why three Hounsfield windows go into the
three RGB channels, what 85 slices costs in tokens, and how much of it actually fits on a
free Colab GPU.

The [notebook](tutorial/medgemma_ct.ipynb) measures it. The article explains it.

---

## Three windows, three channels

A CT holds thousands of Hounsfield values; the encoder takes 256 levels per channel. The
usual approach picks one window and copies it across R, G and B — spending three channels
on one view. MedGemma gives each channel a different window:

| Channel | HU window | Carries |
|---|---|---|
| Red | −1024 to 1024 | morphological boundaries, bone, lung |
| Green | −135 to 215 | soft tissue |
| Blue | 0 to 80 | subtle low-contrast detail |

What that buys you, on real voxel values:

```
     HU   R(wide)   G(soft)  B(low-contrast)   single greyscale
  -1000         3         0                0                  0
     40       132       128              128                128
     60       135       142              191                140
    400       177       255              255                255  <- saturated
   1000       252       255              255                255  <- saturated
```

Two soft tissues 20 HU apart are **63 levels apart in the blue channel** against 12 in a
single greyscale window. And where greyscale saturates on bone, red still resolves it.

## Install

```bash
pip install git+https://github.com/rekalantar/MedGemma-CT-Reader.git
```

Volume loading and the model itself are optional extras:

```bash
pip install "medgemma-ct[io,model] @ git+https://github.com/rekalantar/MedGemma-CT-Reader.git"
```

MedGemma is **gated** — you need a Hugging Face account, acceptance of the Health AI
Developer Foundations terms on the
[model page](https://huggingface.co/google/medgemma-1.5-4b-it), and a token.

## Use

```python
from medgemma_ct import window_rgb, sample_indices, build_messages

volume = window_rgb(volume_hu)                    # (D,H,W) HU -> (D,H,W,3) uint8
keep = sample_indices(len(volume))                # at most 85, evenly spread
messages = build_messages(volume[keep], "Describe the liver.")

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=dtype)
```

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rekalantar/MedGemma-CT-Reader/blob/main/tutorial/medgemma_ct.ipynb)

Self-contained — every function is defined in the notebook.

**Sections 1–5 need no GPU, no account and no model download.** They load a CT case from the
MedSAM2 demo set (NIfTI, public, unlike MedGemma's own CT notebook which needs Google Cloud
credentials), show what each window channel contributes, and build the prompt. Pure NumPy.

**Section 6 runs the model**, which needs a GPU runtime and a gated-model token. It tries
16-bit, falls back to 4-bit, then sweeps slice counts until the GPU runs out of memory — so
you learn what your hardware actually holds, which the documentation does not say.

## API

| Function | Does |
|---|---|
| `window_rgb(volume_hu, windows)` | Hounsfield → uint8 RGB, one window per channel |
| `window_grey(volume_hu, width, level)` | single-window greyscale, for comparison |
| `normalise(volume, low, high)` | clip to a window, rescale to 0–255 |
| `sample_indices(depth, max_slices)` | up to 85 indices spread evenly through a volume |
| `vision_tokens(n_slices)` | token cost of a slice count |
| `encode_slice(slice_rgb)` | one RGB slice → base64 data URI |
| `build_messages(slices, question)` | the `messages` list for `apply_chat_template` |
| `count_images(messages)` | how many image blocks a prompt carries |

## Verified against upstream

The windowing and slice-sampling here reproduce
[MedGemma's own CT notebook](https://github.com/google-health/medgemma/blob/main/notebooks/high_dimensional_ct_hugging_face.ipynb)
exactly — two tests assert byte-for-byte equality with the reference implementations rather
than trusting that the constants were copied correctly.

```bash
git clone https://github.com/rekalantar/MedGemma-CT-Reader.git
cd MedGemma-CT-Reader
pip install -e ".[dev]"
pytest
```

21 tests, no GPU required.

## What this does not do

- **It does not fine-tune.** Inference only.
- **It is not a diagnostic tool.** MedGemma's CT-RATE Macro F1 is 26.9 — better than
  general-purpose models, nowhere near clinical reliability.
- **85 slices is a cap, not a guarantee.** Whether your GPU holds that many is what the
  notebook measures. On a free Colab T4 it does not — section 6 reports the real ceiling.

## Reference

MedGemma 1.5 Technical Report, [arXiv:2604.05081](https://arxiv.org/abs/2604.05081).
Weights: [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it),
Health AI Developer Foundations licence.

MIT licensed. No model weights or patient data are redistributed here.
