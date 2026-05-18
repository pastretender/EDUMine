# Biomedical AI Tutorials — Contributor Onboarding Guide

> **Project Mission:** Build an open, aggregated library of high-quality, self-contained Jupyter notebook tutorials covering the intersection of AI and biomedical imaging — from classical signal processing to state-of-the-art deep learning.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Topic Taxonomy](#2-topic-taxonomy)
3. [Tutorial Structure Standards](#3-tutorial-structure-standards)
4. [Code Standards](#4-code-standards)
5. [Content & Pedagogy Standards](#5-content--pedagogy-standards)
6. [Evaluation & Metrics Standards](#6-evaluation--metrics-standards)
7. [Clinical & Ethical Compliance](#7-clinical--ethical-compliance)
8. [Dataset & Reproducibility Standards](#8-dataset--reproducibility-standards)
9. [Visualization Standards](#9-visualization-standards)
10. [Submission Checklist](#10-submission-checklist)

---

## 1. Project Overview

This project aggregates peer-quality educational notebooks that teach practitioners — from upper-undergraduates to early-career researchers — how to apply AI to biomedical imaging problems. Each tutorial is a standalone learning unit: a reader should be able to open it in Google Colab, run every cell top-to-bottom, and emerge with a working mental model of the topic.

### Who We Write For

| Audience | Background Assumed |
|---|---|
| **Primary** | Graduate students in biomedical engineering, computer science, or medicine with Python fluency |
| **Secondary** | Senior undergraduates with a scientific computing background |
| **Stretch** | Clinical researchers with minimal coding experience |

Tutorials should be accessible to the primary audience without hand-holding, while providing enough contextual explanation that secondary and stretch audiences can follow along.

### Platform Target

All tutorials must run on **Google Colab with a T4 GPU (16 GB VRAM)** unless the tutorial is explicitly CPU-only and labelled as such. This is the lowest common denominator that ensures broad accessibility.

---

## 2. Topic Taxonomy

The project is organized into six topic pillars. Contributors should position their tutorial clearly within one primary pillar and may note secondary connections.

### Pillar A — Signal & Image Processing Foundations

Classical and signal-processing approaches that underpin all higher-level methods. Entry points for readers new to the field.

**In-scope topics:**
- Noise modelling in specific modalities (Poisson/shot noise, Rician MRI noise, speckle)
- Contrast Transfer Function (CTF) estimation and correction
- Fourier analysis, filtering, and the Fourier Ring Correlation (FRC) resolution criterion
- SNR improvement strategies (averaging, Wiener filtering, bandpass filtering)
- Image quality metrics: PSNR, SSIM, NCC

**Reference notebooks:** `cryoem_low_snr_tutorial.ipynb`, `cryoem_reconstruction_tutorial.ipynb`

---

### Pillar B — Medical Image Segmentation

Delineating anatomical structures or pathological regions from volumetric medical images.

**In-scope topics:**
- 2D and 3D U-Net architectures, SegResNet, SwinUNETR
- Loss functions for imbalanced segmentation (Dice, DiceCE, Focal)
- Patch-based training and sliding-window inference
- MONAI framework: transforms, CacheDataset, post-processing
- Evaluation: Dice score, Hausdorff distance, 95th-percentile HD

**Reference notebooks:** `3D_Medical_Image_Segmentation_MONAI.ipynb`

---

### Pillar C — Image Registration

Aligning two or more images — across time, viewpoint, or imaging modality — into a common coordinate frame.

**In-scope topics:**
- Classical optimization: rigid, affine, deformable B-spline (SimpleITK, ANTsPy)
- Information-theoretic similarity metrics: Mutual Information (MI), Normalized MI
- Deep learning registration: VoxelMorph-style deformation field networks
- Evaluation: Dice overlap after registration, Target Registration Error (TRE), Jacobian determinant (folding penalty)

**Reference notebooks:** `multimodal_image_registration.ipynb`, `multimodal_image_registration_tutorial.ipynb`

---

### Pillar D — Image Synthesis & Generative Models

Generating missing or alternative imaging modalities, augmenting rare pathology data, or learning transport maps between data distributions.

**In-scope topics:**
- Conditional Flow Matching and Schrödinger Bridge (SB-CFM)
- Diffusion models for medical image synthesis
- CycleGAN and unpaired cross-modal translation
- Evaluation: PSNR, SSIM, MAE on held-out pairs; FID for unpaired settings
- Clinical motivation: data augmentation for rare pathology, replacing unavailable modalities

**Reference notebooks:** `SB_CFM_Medical_Synthesis.ipynb`, `Cross-Modal_Medical_Image_Synthesis_using_Schrödinger_Bridge_Flow_Matching__SB-CFM_.ipynb`

---

### Pillar E — Multimodal AI & Vision-Language Models

Integrating vision and language for clinical decision support, report generation, and visual question answering.

**In-scope topics:**
- Zero-shot classification with CLIP-style models (BiomedCLIP)
- Image captioning and report generation (BLIP-2)
- Medical Visual Question Answering (Med-VQA): VQA-RAD dataset
- Parameter-efficient fine-tuning: LoRA, 4-bit quantisation (bitsandbytes)
- NLP evaluation: BLEU, token-F1, BERTScore, ROUGE

**Reference notebooks:** `MedVQA_ReportGeneration_Tutorial.ipynb`, `multimodal_medical_imaging_tutorial.ipynb`

---

### Pillar F — Microscopy & Cell Biology

AI-assisted analysis of fluorescence microscopy, live-cell imaging, and morphological phenotyping.

**In-scope topics:**
- Instance segmentation for cell detection (Cellpose)
- Multi-frame cell tracking (nearest-neighbour, IoU assignment, Hungarian algorithm)
- Morphological feature extraction: shape descriptors, intensity statistics, trajectory analysis
- Self-supervised denoising without ground-truth clean images (Noise2Self, Noise2Void)
- Modality-specific noise modelling: Poisson, Gaussian, Rician, speckle

**Reference notebooks:** `cell_tracking_morphological_analysis.ipynb`, `self_supervised_denoising_medical.ipynb`

---

### Cross-Cutting Topics (any pillar)

The following themes may appear in any tutorial:
- Uncertainty quantification and probabilistic outputs
- Model interpretability and saliency mapping (Grad-CAM, attention visualization)
- Transfer learning and domain adaptation
- Efficient inference: AMP (Automatic Mixed Precision), GradScaler, model quantisation
- 3D data handling: NIfTI format, voxel spacing, orientation metadata

---

## 3. Tutorial Structure Standards

Every tutorial must follow this canonical section order. Sections may be subdivided but must not be reordered or omitted.

### Required Sections (in order)

#### Header Block
The very first cell must be a Markdown cell containing:
- A descriptive title (use a level-1 heading)
- A one-line clinical or scientific motivation sentence
- A hardware/runtime badge: `**Runtime:** Google Colab · T4 GPU (16 GB VRAM) · Python 3.x`
- A framework badge listing the core libraries used

```markdown
# 🧠 [Topic]: [Specific Technique or Task]
### [One-sentence clinical context]

**Runtime:** Google Colab · T4 GPU (16 GB VRAM) · Python 3.x
**Frameworks:** PyTorch · MONAI · [other]
```

#### Section 0 or 1 — Learning Objectives
A table listing 3–6 concrete, measurable learning objectives. Frame each as an action verb (understand, implement, evaluate, compare, apply).

```markdown
### 🎯 Learning Objectives
| # | After this tutorial you will be able to… |
|---|---|
| 1 | Explain the physical origin of [noise type] in [modality] |
| 2 | Implement [algorithm] from scratch in PyTorch |
| 3 | Evaluate results using [metric] and interpret what the numbers mean |
```

#### Section 1 — Environment Setup & GPU Verification
- Install all dependencies with `!pip install -q`
- Verify GPU availability with a descriptive diagnostic printout
- Set all random seeds for reproducibility
- Print library versions

#### Section 2 — Dataset & Data Pipeline
- Load or generate data (see §8 for dataset rules)
- Show representative samples with visualizations
- Explain preprocessing choices with reasoning

#### Section 3 — Theoretical Background / Mathematical Foundations
- Introduce the key equations using LaTeX in Markdown cells
- Provide intuitive explanations alongside formal definitions
- Use comparison tables when contrasting multiple methods

#### Sections 4–N — Core Implementation Modules
- One section per major algorithmic component
- Each section: theory → code → visualization → interpretation
- Include "Why?" callout boxes for non-obvious design choices

#### Final Section — Evaluation & Results
- Quantitative metrics on held-out data or a synthetic benchmark
- Side-by-side visualizations (input / method A / method B / ground truth)
- A brief interpretation: what do the numbers mean clinically or scientifically?

#### Optional: Reflection Questions
Include 3–5 open-ended questions that encourage deeper thinking. Label clearly so readers know they are optional.

---

### Notebook Overview Table

Every tutorial must include an overview table immediately after the header block:

```markdown
| Section | Topic | Key Concepts |
|---|---|---|
| 1 | Environment Setup | pip, seeds, device detection |
| 2 | Dataset & Preprocessing | [modality], [transforms] |
| 3 | Mathematical Foundations | [key equations] |
| … | … | … |
```

---

## 4. Code Standards

### Environment & Dependencies

- **Pin major versions** for all non-trivial packages: `monai==1.3.0` not `monai`
- Install in a single cell at the top; use `-q` to suppress verbose output
- Follow the install cell with a single imports cell; do not scatter imports throughout
- Always suppress warnings globally after imports: `warnings.filterwarnings('ignore')`

```python
# ── Install dependencies ──────────────────────────────────────────────────────
!pip install -q \
    "monai[nibabel,tqdm]==1.3.0" \
    torch>=2.0.0 \
    matplotlib

# ── Imports ───────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
...
```

### Reproducibility

Every tutorial must set all random seeds in a dedicated, clearly commented block:

```python
# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
import random, numpy as np, torch
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
```

Use `monai.utils.set_determinism(seed=42)` when using MONAI.

### GPU Detection

Always use a standardised GPU diagnostic block — do not assume GPU availability:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"✅ GPU: {gpu.name}  |  VRAM: {gpu.total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detected — running on CPU (some cells will be slow)")
```

### Code Style

- **Line length:** 88 characters maximum (Black formatter compatible)
- **Section separators:** Use `# ── Section Name ────────` comment banners to delimit logical blocks within cells
- **Docstrings:** All functions longer than 10 lines must have a one-line docstring
- **Type hints:** Use type hints for function signatures where they aid clarity
- **Magic numbers:** Name all constants — `PATCH_SIZE = 96`, not `96` inline
- **No silent failures:** Wrap I/O operations and downloads in try/except with informative error messages

### Naming Conventions

| Object | Convention | Example |
|---|---|---|
| Constants | UPPER_SNAKE | `NUM_EPOCHS`, `PATCH_SIZE` |
| Functions | lower_snake | `compute_dice_score` |
| Classes | PascalCase | `UNetEncoder` |
| Variables | lower_snake | `train_loader`, `pred_mask` |
| Loop indices | descriptive | `for epoch in range(NUM_EPOCHS)` |

### Memory & Performance

- Use `Automatic Mixed Precision (AMP)` for all training loops that target the T4:

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

- Cache datasets when the preprocessing pipeline is deterministic (`CacheDataset` in MONAI)
- Use `ThreadDataLoader` in preference to `DataLoader` for MONAI pipelines
- Document expected peak VRAM usage at the start of any cell that may OOM

---

## 5. Content & Pedagogy Standards

### The Explain-Before-Implement Rule

Every new concept must be explained in a Markdown cell **before** the code cell that implements it. The explanation should answer three questions:
1. **What** is this? (one sentence)
2. **Why** do we need it? (clinical or mathematical motivation)
3. **How** does it work? (intuition, then formalism)

### Mathematical Notation

- Write all equations using LaTeX inside Markdown cells
- Introduce notation inline: "where $\Delta z$ is the defocus in nanometres"
- For key equations, add a parameter table below:

```markdown
$$\text{CTF}(s) = -A\sin[\chi(s)] + \sqrt{1-A^2}\cos[\chi(s)]$$

| Symbol | Meaning | Typical Value |
|---|---|---|
| $s$ | Spatial frequency (cycles/Å) | 0 – 0.5 |
| $A$ | Amplitude contrast | 0.07–0.10 |
| $\Delta z$ | Defocus | 0.5–3 µm |
```

### Callout Boxes

Use blockquotes to create pedagogically important callouts:

```markdown
> **Why MONAI?**
> MONAI provides 3D-aware transforms that respect voxel spacing and orientation
> metadata — a common source of subtle bugs when using generic image libraries.

> ⚠️ **Common Pitfall:** Never normalize 3D volumes using 2D statistics
> (per-slice mean/std). Always compute statistics over the full volume.

> 💡 **Key Insight:** Averaging N identical noisy copies improves SNR by √N.
> This is the theoretical basis for class averaging in cryoEM.
```

### Comparison Tables for Methods

When introducing a new method, always compare it to at least one alternative:

```markdown
| Method | Transport Path | Inference Steps | Requires Pairs | Stability |
|---|---|---|---|---|
| CycleGAN | Adversarial (implicit) | 1 | ❌ | ⚠️ Mode collapse |
| DDPM | Gaussian noise → data | 50–1000 | Optional | ✅ |
| **SB-CFM** ⭐ | Entropy-regularised OT | 5–20 | ✅ | ✅ Optimal |
```

### Progressive Complexity

Tutorials should follow a deliberate complexity arc:
1. Start with a minimal, readable implementation that clearly maps to the theory
2. Add complexity in explicitly labelled steps ("Now let's add data augmentation…")
3. Never introduce two new concepts simultaneously in a single code cell

---

## 6. Evaluation & Metrics Standards

### Required Evaluation Coverage

Every tutorial that trains or applies a model must report at least one quantitative metric. The table below maps pillars to their expected metrics.

| Pillar | Primary Metrics | Secondary Metrics |
|---|---|---|
| A — Signal Processing | SNR (dB), PSNR, FRC resolution estimate | Wiener filter MSE, visual comparison |
| B — Segmentation | Dice score, Hausdorff Distance (HD95) | Sensitivity, Specificity |
| C — Registration | Dice (after warp), NMI, TRE | Jacobian determinant (% folding) |
| D — Synthesis | PSNR, SSIM, MAE | FID (unpaired), perceptual metrics |
| E — Vision-Language | BLEU-4, token-F1, BERTScore | ROUGE-L, qualitative examples |
| F — Microscopy/Denoising | PSNR, SSIM | FRC, cell count accuracy |

### Metric Implementation Guidelines

- Always implement metrics in a clearly named, reusable function
- Show the metric formula in a Markdown cell before the implementation
- Report mean ± standard deviation over the test set, never a single sample
- Provide an interpretation sentence: "A Dice score of 0.90+ is considered clinically acceptable for spleen segmentation."

### Plotting Evaluation Results

- For segmentation/registration: show axial, coronal, and sagittal slices side-by-side
- For generative models: show a grid of source / generated / target triplets
- For training curves: plot both training and validation loss on the same axes
- Always label axes with units; never leave axis labels as variable names

---

## 7. Clinical & Ethical Compliance

### Mandatory Disclaimer

Every tutorial that uses or trains a model on medical imaging data must include the following disclaimer, verbatim, in the first or second Markdown cell:

```markdown
> ⚠️ **Disclaimer:** This notebook is for **educational and research purposes only**.
> Model outputs must **never** be used for real clinical diagnosis or treatment decisions.
> Always consult a qualified and licensed healthcare professional for medical advice.
> Models shown here are **not** FDA-cleared or CE-marked medical devices.
```

### Data Privacy

- Never use real patient data in tutorials. Use one of:
  - Publicly released, IRB-exempt research datasets (MSD, TCIA, IXI, VQA-RAD)
  - Synthetically generated phantoms
  - De-identified public benchmarks with explicit open licenses
- Always cite the dataset license and access conditions

### Model Claims

- Do not make performance claims that extrapolate beyond the tutorial's synthetic or benchmark setting
- Use hedged language: "on this synthetic benchmark," not "this model achieves clinical-grade performance"
- When citing published results, include the original reference

---

## 8. Dataset & Reproducibility Standards

### Self-Contained Data Policy

Tutorials must be runnable without manual data download steps wherever possible. Prefer this hierarchy:

1. **Synthetic generation** — generate phantom data programmatically in the notebook (most preferred)
2. **Automatic download** — use library APIs that handle download and caching (e.g., `monai.apps.DecathlonDataset`, HuggingFace `datasets`)
3. **Public URL download** — use `wget` or `requests` with a stable, versioned URL
4. **Manual download** — only as a last resort; provide exact instructions including the expected file size and MD5 hash

### Synthetic Data Quality

When generating synthetic data, it must reflect the real modality's key properties:

| Modality | Must Simulate |
|---|---|
| MRI | Tissue contrast differences (T1 vs T2 intensity relationships), Rician noise |
| CT | Hounsfield-like intensity ranges, Poisson + Gaussian noise |
| Fluorescence microscopy | Poisson shot noise, background fluorescence, point-spread function blur |
| CryoEM | CTF modulation, very low SNR (SNR < 0.1), amorphous ice background |

### Versioning & Caching

- Pin dataset versions where APIs support it
- When using `DecathlonDataset` or similar, set `cache_num` appropriately for T4 VRAM
- Store downloaded data in `/tmp/` or a clearly named directory under the working path

---

## 9. Visualization Standards

### Color & Style

All tutorials must configure `matplotlib` with a consistent, readable style. Use the following base configuration:

```python
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'font.size': 10,
    'image.cmap': 'gray',       # default for medical images
})
```

For dark-themed notebooks (signal processing / cryoEM style), use consistent dark backgrounds. Do not mix dark and light themes within a single tutorial.

### Figure Anatomy

Every figure must have:
- A descriptive `fig.suptitle()` or per-axes `ax.set_title()`
- Labelled axes with units (e.g., `ax.set_xlabel("Spatial Frequency (cycles/Å)")`)
- A colorbar for any image displayed with `imshow` that is not greyscale
- Sufficient resolution: `figsize` should be at least `(12, 4)` for multi-panel figures

### Visualization Timing

Place visualizations:
- After data loading (show raw samples)
- After each preprocessing step (show the effect)
- After training (show training curves)
- After inference (show predictions vs. ground truth)
- At the end (summary comparison)

Never go more than three major processing steps without a visualization checkpoint.

### Colormaps for Medical Images

| Content | Recommended Colormap |
|---|---|
| Greyscale anatomical image | `gray` |
| Segmentation mask overlay | `jet` or custom label colors with legend |
| Error / residual map | `RdBu_r` (diverging, centred at zero) |
| Probability / confidence map | `viridis` |
| Fourier / frequency domain | `inferno` or `hot` |

---

## 10. Submission Checklist

Before submitting or merging a tutorial, verify every item below.

### Structure
- [ ] Header block with title, clinical motivation, runtime, and framework badges
- [ ] Notebook overview table
- [ ] Learning objectives table (3–6 items)
- [ ] Sections in canonical order (Setup → Data → Theory → Implementation → Evaluation)
- [ ] Clinical disclaimer present (if applicable)

### Code Quality
- [ ] All dependencies installed in a single top cell with pinned versions
- [ ] Random seeds set with the standard block (SEED = 42)
- [ ] GPU detection with informative diagnostic output
- [ ] No hardcoded magic numbers — all named constants
- [ ] No imports scattered mid-notebook
- [ ] All cells run top-to-bottom without error on a fresh T4 Colab instance
- [ ] Peak VRAM stays below 14 GB (leaving 2 GB headroom)

### Pedagogy
- [ ] Every concept has a Markdown explanation before its code cell
- [ ] Key equations written in LaTeX with a parameter table
- [ ] At least one "Why?" callout box per major design decision
- [ ] At least one comparison table for the primary method vs. alternatives

### Evaluation
- [ ] At least one quantitative metric reported as mean ± std over a test set
- [ ] Metric formula explained in a Markdown cell
- [ ] Results interpreted with a domain-relevant benchmark sentence
- [ ] Side-by-side visualizations of input / output / (ground truth if available)

### Data & Reproducibility
- [ ] Data is either synthetic, auto-downloaded, or has explicit download instructions
- [ ] Dataset license cited
- [ ] Running the notebook twice with the same seed produces identical outputs

### Visualization
- [ ] Every figure has a title, labelled axes with units, and a colorbar where applicable
- [ ] Visualizations placed at every major pipeline stage
- [ ] `matplotlib` style configured consistently at the top of the notebook

### Ethics & Compliance
- [ ] No real patient data used
- [ ] No unsupported performance claims
- [ ] Clinical disclaimer included for any model that produces outputs resembling diagnoses

---

*This document is a living standard. Contributors are encouraged to raise issues and propose amendments via the project's discussion board. When in doubt, look at the reference notebooks listed in §2 — they embody these standards in practice.*
