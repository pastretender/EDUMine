# Presentation Speech — Multimodal Image Registration Tutorial
*Estimated delivery time: 25–35 minutes depending on pace and Q&A*

---

## Opening

Good morning / afternoon everyone. Thanks for having me. Today I'm going to walk you through a hands-on tutorial I put together on **multimodal image registration** — a topic that sits right at the intersection of classical medical imaging and modern deep learning. By the end of this session, you should have a clear picture of why this problem is hard, how both classical and learning-based methods tackle it, and how to evaluate whether your registration actually worked.

The tutorial is structured as a Jupyter notebook you can run on Google Colab with a T4 GPU. Everything is self-contained — synthetic data, training, evaluation, and a real-world example. Let me walk you through it part by part.

---

## Part 1 — Setup & Synthetic Data

We start with the setup, which installs four key libraries: **SimpleITK** for classical registration, **ANTsPy** for the gold-standard ANTs framework, **DIPY** for diffusion-specific metrics, and **nilearn/nibabel** for accessing real neuroimaging datasets.

The most important thing to flag here is the **GPU check**. Parts 4 and onward — the deep learning section — really benefit from a CUDA GPU. If you're running this at home and you see the CPU warning, just switch your Colab runtime to a T4 before you proceed.

The core of Part 1 is the **synthetic brain phantom**. Rather than jumping straight to real MRI data, which is messy and hard to ground-truth, we generate a 2D brain-like phantom with five tissue classes: background, CSF, grey matter, white matter, and a small lesion. The key insight is that we apply two completely different contrast tables to the same underlying label map — one mimicking T1 MRI where white matter is bright and CSF is dark, and another mimicking T2 MRI where those contrasts are essentially *inverted*. We then apply a known ground-truth rigid misalignment — 8 degrees rotation, 18 pixels translation in X, minus 12 in Y — to the T2 image to create our moving image. Because we know the ground truth exactly, we can measure registration accuracy with precision.

This is a really important design choice. In real-world neuroimaging, you rarely have a perfect ground truth to evaluate against. The synthetic phantom lets us do honest, quantitative benchmarking before we move to messy real data.

---

## Part 2 — Similarity Metrics

This is arguably the most conceptually important section. The core question is: **what does it even mean for two images to be "aligned"?**

In monomodal registration — say, aligning two T1 scans — the answer is simple. Identical anatomy should have similar intensities, so you can just minimise Mean Squared Error. But look at what happens when we plot the MSE landscape over rotation angles between T1 and T2. The minimum of MSE does *not* occur at the ground-truth alignment. In fact, MSE can actively mislead the optimizer, because the same brain tissue genuinely looks different in the two modalities.

The solution is **Mutual Information**. MI asks: how much does knowing the intensity of a pixel in the fixed image reduce my uncertainty about the corresponding pixel in the moving image? When the images are well aligned, there's a strong statistical relationship between intensities — even if they're not numerically similar. That relationship is captured by the joint histogram. At perfect alignment, the joint histogram has tight, structured clusters. At misalignment, it smears out into noise.

We also implement **Normalized Mutual Information**, which divides by the joint entropy. NMI is more robust when the images have different amounts of overlap — which matters in practice when a moving image is cropped or partially outside the field of view.

The metric landscape plots are worth spending a moment on. You can see MI and NMI both peak cleanly at the zero-offset position — that's our ground truth. MSE has no such peak. This single figure is often enough to convince someone why you can't just use pixel-wise error for multimodal work.

---

## Part 3 — Classical Registration with SimpleITK

Now we actually register images. SimpleITK's registration framework is built around three interchangeable components: a **transform** that defines the space of possible mappings, a **metric** that measures image similarity, and an **optimizer** that searches for the best transform parameters.

### 3.1 — Rigid Registration

Rigid registration recovers rotation and translation only — three degrees of freedom in 2D. This is appropriate whenever you can assume the anatomy itself doesn't change shape, which is a reasonable assumption for brain MRI.

We initialize the transform using the image geometry centroids — this is crucial, because a bad initialization can send the optimizer into a local minimum it never escapes from. We then run a **multi-resolution pyramid**: starting at quarter resolution, then half resolution, then full resolution. At each level we smooth the images with Gaussians to suppress fine-grained detail that would create spurious local optima.

The optimizer convergence plot is satisfying to look at — you can see the Mattes Mutual Information metric improving steadily across iterations, with the big gains coming in the early coarse levels and the fine levels polishing the solution.

At the end we print a comparison table: ground truth was 8 degrees and 18 pixels; the estimated parameters come out very close. The angular error is typically under half a degree.

One detail worth noting: Mattes MI only samples 15% of voxels randomly at each iteration. This sounds like it would hurt accuracy, but it actually helps the optimizer escape local minima by introducing controlled stochasticity — similar in spirit to mini-batch SGD in deep learning.

### 3.2 — Affine Registration

Affine adds scaling and shearing on top of rigid, giving us six degrees of freedom in 2D. This is useful when images come from different scanners with slightly different voxel sizes, or when there are field distortions. We apply a ground-truth affine misalignment — a small shear, anisotropic scaling, plus rotation and translation — and demonstrate that the rigid model fails to fully correct it while the affine model does.

### 3.3 — Deformable B-Spline Registration

For non-rigid anatomy — inter-subject differences, tumour deformation, respiratory motion — we need a much richer transform. B-splines parameterize a smooth displacement field using a sparse grid of control points, where:

u(x) = sum over control points of c_ij times B_i(x) B_j(y)

The key message here is the **pipeline order**: always start with rigid, then refine with affine if needed, then add B-splines on top. Starting deformable registration from scratch without a good rigid initialization almost always fails. The displacement field magnitude visualization at the end gives a nice visual sense of where the deformations are largest — typically around tissue boundaries.

---

## Part 4 — Deep Learning Registration

Classical registration optimizes for each image pair independently, which takes hundreds of milliseconds to seconds per pair. Deep learning takes a completely different approach: we train a network to **learn the registration function** once, and then apply it to new pairs in milliseconds.

### Architecture

We implement a **VoxelMorph-style symmetric U-Net encoder-decoder**. The input is a two-channel image: fixed and moving concatenated along the channel dimension. The network outputs a dense 2D displacement field — one displacement vector per pixel. The encoder downsamples through four convolutional blocks, building progressively abstract representations. The decoder upsamples back to full resolution with skip connections from the encoder at each scale. Skip connections are critical: without them, fine-grained spatial detail is lost in the bottleneck.

At the very end, a small convolutional head predicts the displacement field. Crucially, its weights are initialized near zero, so the network starts close to the identity transform and learns to deviate from it only as needed.

The **Spatial Transformer Network** then applies this displacement field to the moving image via differentiable bilinear interpolation. This is what makes the whole system end-to-end trainable: gradients flow through the warping operation back into the displacement field and then all the way through the U-Net.

### Loss Function

The training loss has two terms. The **similarity term** uses Local Normalized Cross-Correlation — LNCC — computed in sliding windows. LNCC is differentiable and handles contrast differences across modalities because it measures local correlation rather than intensity equality. The **regularization term** penalizes spatial gradients of the displacement field, encouraging smooth, physically plausible deformations. The relative weight lambda — set to 1.5 here — controls the smoothness-vs-accuracy trade-off.

### Training

We generate 400 synthetic training pairs on the fly, each with a different random rigid and non-rigid misalignment. This data augmentation is essential: the network needs to see the full range of misalignments it will encounter at test time. Training runs for 25 epochs with a cosine annealing learning rate schedule and gradient clipping.

The loss curves show a clean decrease in both the LNCC and smoothness terms. After training, inference on the original misaligned pair runs in single-digit milliseconds on GPU — compare that to the hundreds of milliseconds for classical rigid registration.

---

## Part 5 — Evaluation Metrics

This section is where a lot of tutorials fall short, and I want to spend some time on it because it matters enormously in practice.

We evaluate registration quality across five complementary metrics:

**NMI** tells us whether the overall intensity statistics improved. It's fast to compute and requires nothing beyond the images themselves.

**Dice Score** tells us whether anatomical structures actually overlap correctly after registration. You warp the moving image's segmentation labels using the recovered transform and compare to the fixed image's segmentation. Dice can reveal failures that NMI misses entirely — for instance, if the optimizer finds a degenerate solution where NMI is high but anatomy is misaligned.

**Target Registration Error (TRE)** measures the Euclidean distance between manually annotated corresponding landmarks in fixed and moved images. It's the gold standard for focal anatomical accuracy and is required for regulatory submissions in clinical applications.

**Jacobian Determinant** is a deformation quality metric. We compute the determinant of the Jacobian matrix of the displacement field at every pixel. A value of 1 means no volume change. Positive values below 1 mean compression. Values above 1 mean expansion. Negative values mean **folding** — the deformation has physically implausible topology, where tissue from one location has been mapped to the same location as tissue from another. In clinical imaging, negative Jacobian determinants are unacceptable.

The bar chart comparison shows all three methods — before registration, after rigid, and after DL — across all tissue classes. The dual-axis NMI/Dice plot shows that while both methods improve NMI substantially, they differ in Dice performance across tissue classes. Neither method is universally superior; the right choice depends on your specific application.

---

## Part 6 — Speed and Practical Comparison

The benchmark here is instructive. Classical rigid registration on this 256×256 image takes on the order of hundreds of milliseconds per pair. The DL network, once trained, runs in single-digit milliseconds — a speedup of one to two orders of magnitude, depending on hardware.

The summary table compares four methods: SimpleITK rigid, SimpleITK B-spline, VoxelMorph, and ANTs SyN. The key dimensions are inference time, whether training is required, generalization across subjects, and whether the method handles non-rigid deformations. There's a clear trade-off: classical methods require no training and can adapt fully to each image pair, but are slow. DL methods are fast at inference but require a representative training set and may generalize poorly when test images differ significantly from training data.

---

## Part 7 — Real-World Best Practices

Part 7 is a practical checklist rather than executable code. I'll hit the highlights:

On **data preprocessing**: always skull-strip brain MRI before registration. The scalp and skull have variable shapes across subjects and confound the optimizer. Intensity normalize each image independently — z-score or percentile-based normalization works well. Resample to isotropic voxel spacing, and always check that image orientation headers match before you start.

On **classical registration**: multi-resolution is non-negotiable. Always run rigid before affine before deformable — the pipeline order is not arbitrary. Use Mattes MI for CT-to-MRI and NMI for T1-to-T2.

On **deep learning**: make sure your training data covers the full range of expected misalignments. Use diffeomorphic integration — scaling-and-squaring of a stationary velocity field — if you need guaranteed invertibility. And tune lambda carefully: too small leads to folding, too large leads to under-registration.

On **evaluation**: never report only NMI. Always evaluate Dice on held-out segmentations. Check the Jacobian histogram. And for clinical deployment, expert visual inspection remains essential regardless of what your automated metrics say.

---

## Part 8 — Reflection Questions

The final part of the notebook I added is a set of ten reflection questions organized by theme. I won't go through all of them now, but I want to flag a few that I think are particularly worth sitting with.

**Question 5** on the smoothness regularizer is important for anyone who wants to use this code in practice. Understanding what happens at lambda equals zero versus lambda equals 100 — and what the Jacobian determinant tells you about deformation quality — is essential before you deploy anything clinically.

**Question 7** on evaluation metrics is something I feel strongly about. The question asks you to rank NMI, Dice, TRE, and Jacobian determinant in order of clinical importance for a neurosurgery planning system. There's no single right answer, but thinking through it forces you to understand what each metric actually measures and what it can miss.

**Question 10** is an open-ended design challenge — build a production pipeline to co-register T1 and FLAIR MRI for 10,000 patients across five hospital sites. It covers preprocessing, method choice, automatic quality control, failure recovery, and validation for clinical sign-off. That's the kind of holistic systems thinking that separates a working research prototype from something you can actually deploy.

---

## Closing

To summarise what we've covered:

We started with **why multimodal registration is hard** — different imaging physics means different intensity relationships, and standard pixel-wise metrics fail completely.

We then built up from **classical approaches** — rigid, affine, and deformable B-spline registration using Mutual Information in SimpleITK — to **deep learning** — a VoxelMorph-style U-Net trained with Local NCC and gradient smoothness regularization.

We showed that **evaluation requires multiple metrics** — NMI alone is not sufficient, and the Jacobian determinant is a critical sanity check for deformation quality.

And we grounded everything in **practical guidance** for real-world deployment.

The notebook is fully runnable on Colab. I'd encourage you to work through the reflection questions, especially if you're planning to apply any of this to your own data. The references in Part 7 — VoxelMorph, ANTs SyN, and the Mattes and Rueckert papers — are the canonical reading if you want to go deeper.

Happy to take any questions.

---

*[END OF SPEECH]*

---

## Speaker Notes

- **Estimated total time:** 25–35 min depending on audience questions mid-talk.
- **Slide suggestions:** Use the checkerboard visualizations from Parts 1 and 3, the metric landscape plots from Part 2, the U-Net architecture diagram from Part 4, and the Dice bar chart from Part 5 as your key slides.
- **Potential audience questions:**
  - *"Why not just use a pre-trained model like SynthMorph?"* — Good question; SynthMorph is modality-agnostic and works well, but training from scratch here is pedagogically important.
  - *"How does this extend to 3D?"* — All the SimpleITK code works identically in 3D. The U-Net needs Conv3D layers and substantially more GPU memory; roughly 8× more parameters for a 3D volume.
  - *"What about diffeomorphic registration?"* — Scaling-and-squaring of a stationary velocity field guarantees invertibility; ANTs SyN is the gold standard.
  - *"How do you choose the smoothness weight lambda?"* — Cross-validate on held-out Dice scores; start at 1.0 and sweep over [0.1, 0.5, 1.0, 2.0, 5.0].
