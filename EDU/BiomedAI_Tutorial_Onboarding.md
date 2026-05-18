# 📝 Tutorial Notebook Format Specification Template

## 1. Header & Overview
Each Notebook must begin with a clear Markdown header accompanied by a concise project description.
* **Main Title:** Ideally including a relevant Emoji (e.g., `# 🧠 3D Medical Image Segmentation` or `# 🏥 Multimodal AI`) to enhance identifiability.
* **Subtitle:** Clarify the specific task or experimental content.
* **Key Information:** Explicitly list the foundational details required to run the tutorial, such as **Hardware Target** (e.g., Google Colab T4 GPU), **Core Frameworks** (e.g., PyTorch, MONAI), and **Prerequisites** (prior knowledge or necessary packages).
* **Platform:** All tutorials should be able to run on google colab free tier (T4 GPU at most).
## 2. Learning Objectives & TOC
Ensure readers know what they will achieve before looking at the code.
* **Overview / What You Will Learn:** List the key milestones covered in this tutorial using a structured bulleted list or a table (e.g., 1. Environment Setup, 2. Dataset, 3. Model Architecture, etc.).
* **Important Notes:** For most circumstances, a clear disclaimer must be included: *"For research and educational purposes only, not to be used for clinical diagnosis."*

## 3. Environment Setup & Global Imports
Standardize the initial code blocks to guarantee all users run the notebook in an identical environment.
* **Hardware Verification:** Provide a code block to inspect the GPU status (e.g., `nvidia-smi` and `torch.cuda.is_available()`).
* **Dependency Installation:** Use `!pip install -q ...` to centrally install all required third-party libraries.
* **Global Configurations:** Consolidate package imports in a single cell, and initialize global random seeds (`SEED`), plotting styles (`plt.rcParams`), and device placement variables (`DEVICE`).

## 4. Data Loading & Exploration
* **Data Acquisition:** Provide code to automatically download open-source sample datasets (e.g., download and loading logic for medical images), preferably equipped with error-handling/fallback mechanisms (e.g., generating synthetic placeholder images upon download failure).

## 5. Core Content: Theory + Code
This is the core body where theory and practice must be interwoven.
* **Markdown Theoretical Background:** Before implementing complex algorithms (e.g., CTF Correction, Schrödinger Bridge, or U-Net Segmentation), use Markdown combined with LaTeX equations (e.g., $$I(\mathbf{x}) = ...$$) to explain core mathematical or physical principles, rather than just dumping raw code blocks.
* **Modular Code & Commenting:** Each code block should serve a single, dedicated purpose, and functions must include descriptive Docstrings. It is highly recommended to separate major sections using comments at the top of code blocks formatted as `# ── Section Name ──`.

## 6. Bonus / Questions
To enhance the educational value, open-ended questions or code modification challenges should be placed at the end of the Notebook.
* **Questions / Discussion:** Introduce theoretical or practical questions that encourage readers to contemplate the impact of parameter adjustments or model limitations.
* **Bonus - Going Further:** Guide advanced readers to explore next steps, such as replacing the backbone model, or evaluating alternative metrics.