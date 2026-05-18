# Presentation Speech
## Self-Supervised Denoising in Medical Image Processing
### Group Meeting — Full Script

---

> **Speaker notes format**
> - *[Stage directions in italics and brackets]*
> - `~XX min` estimates at each section heading
> - Total estimated delivery time: **35–45 minutes** (adjust pace to your group's background)

---

## Opening `~2 min`

Good morning, everyone. Today I want to walk you through a topic that sits right at the intersection of deep learning and medical imaging — **self-supervised denoising**. I have a hands-on Jupyter notebook that we can work through together, and by the end of this session you should be able to run it yourself on a free Google Colab T4 GPU in about 25 to 30 minutes.

Let me start with the motivating problem. When we want to train a neural network to denoise images — whether that is an MRI scan, a CT slice, or a fluorescence microscopy image — the supervised approach is the obvious first choice. You collect pairs of noisy and clean images, minimize the pixel-wise reconstruction error, and you are done. The trouble is that in most medical imaging contexts, a truly clean ground-truth image simply does not exist. Acquiring it would often mean applying a dose of radiation so high it is clinically unacceptable, or spending days averaging thousands of acquisitions of the same biological sample that has long since moved or decayed. So we are stuck with noisy images only, and the question becomes: **can a network learn to denoise from noisy images alone, without ever seeing a clean reference?** The answer, perhaps surprisingly, is yes — and that is exactly what this tutorial demonstrates.

---

## Section 0 — Environment Setup `~2 min`

*[Point to the setup cell]*

The notebook opens with a minimal setup cell. We install scikit-image, tqdm, matplotlib, NumPy, PyTorch and torchvision — all of which are available on a standard Colab instance with no special configuration. We fix a global random seed of 42 for reproducibility, and then detect the GPU. When you run this on a T4, you will see roughly 15 gigabytes of VRAM reported. One practical note: you do need to switch the Colab runtime to GPU before running — Runtime → Change runtime type → T4 — otherwise training will be painfully slow.

---

## Section 1 — Noise Models in Medical Imaging `~5 min`

*[Point to the noise model table and the visualisation output]*

Before we can denoise anything intelligently, we need to understand what the noise actually *is*, because different imaging modalities produce fundamentally different noise statistics, and those statistics determine which algorithms are appropriate.

The notebook implements four canonical noise models. The first is **additive white Gaussian noise**, which you can think of as independent Gaussian random values added to every pixel. This is the dominant noise model in MRI, where it arises from thermal fluctuations in the receiver electronics. In code it is just `img + numpy.random.normal(0, sigma, img.shape)` — extremely simple.

The second is **Poisson noise**, also called shot noise or photon noise. Here the variance is proportional to the signal itself, not constant, because it reflects the quantum nature of photon detection. This is what you encounter in fluorescence microscopy and in low-dose CT. The implementation samples from a Poisson distribution parameterised by a "peak" intensity value — the lower the peak, the more severe the noise.

The third model is **Rician noise**, which is specific to MRI magnitude images. When you compute the magnitude of a complex-valued MRI signal contaminated by Gaussian noise in both the real and imaginary channels, the resulting distribution is Rician, not Gaussian. Notice the implementation — we add independent Gaussian noise to both the real and imaginary parts and then take the square root of the sum of squares. At high SNR this is well-approximated by Gaussian, but at low SNR the Rician distribution has a characteristic positive bias that matters for quantitative MRI.

The fourth is **multiplicative speckle noise**, dominant in ultrasound. Instead of adding an independent term, the noise is scaled by the local signal intensity, which means bright regions suffer more noise. This is implemented as `img + img * normal(0, std)`.

*[Pause on the five-panel visualisation]*

What you see on screen is a uniform-intensity test image corrupted by each of these four models, with their corresponding PSNR values. The visual difference is subtle but instructive. Gaussian and Rician noise look granular and uniform. Poisson noise is more pronounced in the brighter areas — you can see that the variance tracks the signal. Speckle creates a characteristic textured pattern.

Now, the **theoretical backbone** of everything that follows is written in the middle of this section as two equations. We assume that the observed image x-tilde equals the true signal x plus independent zero-mean noise n. The key insight is that a function f that predicts the value of a pixel using *only the surrounding pixels* — not the pixel itself — minimises a self-supervised loss that is provably equivalent to the supervised loss up to a noise-dependent constant. That equivalence is the mathematical justification for this entire class of methods, and we will come back to it when we discuss each algorithm. The key requirements are that the noise has zero mean and that the noise at different pixels is statistically independent.

---

## Section 2 — Synthetic Medical Phantom Generation `~4 min`

*[Show the three-panel phantom output]*

Since we want to rigorously evaluate our denoising methods — to compute true PSNR and SSIM against a known clean reference — we need an image for which we actually have the ground truth. Rather than downloading a large public dataset, the notebook generates a synthetic brain-slice-like phantom programmatically using `skimage.draw`.

Let me walk you through the construction. We start with a 256-by-256 blank canvas. We draw a large outer ellipse at high intensity to represent the skull outline. Inside that we add a cerebrospinal fluid layer at low intensity — that is the thin dark ring. Then a gray matter ellipse, then a white matter region at the centre. We carve out two symmetric ventricles as darker ellipses flanking the midline. On top of that we scatter five small hyper-intense disks — those are the simulated lesions, the kind of bright spots a radiologist might look for. Finally we draw eight thin radial lines emanating from the centre to simulate vessel-like structures.

*[Point to the third panel — the absolute noise map]*

The result is on the left: a 256-by-256 phantom that captures the most important qualitative features of a brain MRI — layered tissue contrasts, dark fluid-filled spaces, bright lesions, and fine linear structures. We then add Gaussian noise with sigma equals 0.10 to produce the noisy version in the centre panel, and the rightmost panel shows the absolute difference — that is the noise itself, displayed as a heat map. The reported PSNR of the noisy image is around 20 decibels, which represents a meaningful level of degradation — realistic for an MRI acquired at a reduced number of averages.

The reason we generate five independent noisy realisations of the same clean phantom, rather than just one, is to increase the *diversity* of the training patches. This matters because self-supervised training can overfit if the network sees the same noise pattern many times.

---

## Section 3 — Noise2Self: J-Invariant Masking `~7 min`

*[Point to the theory markdown cell]*

We are now ready for the first algorithm. Noise2Self, published by Batson and Royer at ICML 2019, formalises the idea of a **J-invariant function**. The definition is simple: a function is J-invariant if, for each pixel j, the output at position j does not depend on the *input* at position j — it only depends on all the *other* pixels. If the network is J-invariant, and the noise satisfies the zero-mean independence assumption, then training the network to predict its own input is mathematically equivalent to training it against the unknown clean image.

The question then becomes how to enforce J-invariance in practice without designing a special architecture. The answer used here is **masking**: at each training step we randomly select around 5% of pixels in the input patch, replace each of them with the value of a randomly chosen neighbour within a 5-by-5 window, and then compute the MSE loss *only at those masked locations*. The network cannot use the original value of a masked pixel as part of its input — because we have replaced it — so it is forced to rely purely on the surrounding context to predict what that pixel should be. J-invariance is enforced by construction.

*[Show the three-panel masking visualisation]*

You can see this in the masking visualisation. On the left is the original noisy patch. In the centre, after masking, you might not even see the difference at this scale — 5% of 256-by-256 is only around 3,000 pixels. The right panel highlights those pixels in white. The loss only flows back through those highlighted locations.

Now let me walk through the code structure. We have a **PatchDataset** that samples random 64-by-64 patches from the noisy image, with augmentation — horizontal flip, vertical flip, and four rotations. These augmentations are safe because the noise is isotropic, so flipping and rotating do not change its statistical properties. We build a **ConcatDataset** from all five noisy realisations, giving us 2,000 patches in total per epoch.

The backbone network is a lightweight **U-Net** with four resolution levels and a base channel count of 32. The channel progression is 32, 64, 128, 256 — this comes to about 3.1 million trainable parameters, well within the T4's memory budget. Skip connections between the encoder and decoder are essential here because they allow the network to propagate fine structural information across resolution scales, which is critical for preserving tissue boundaries during denoising.

Training runs for 40 epochs with the Adam optimiser at a learning rate of 3 times 10 to the minus 4, with a cosine annealing schedule that decays the learning rate to near zero by the final epoch. After training, we call `full_image_denoise`, which applies the network to the full 256-by-256 image using a **sliding-window tile inference** strategy. We process overlapping 128-by-128 tiles, blend their predictions using a 2D Hanning window, and accumulate a weighted average. The Hanning window is important — without it, you would see visible seam artefacts at the tile boundaries because the network has reduced context at the edges of each tile.

---

## Section 4 — Noise2Void: Structured Blind-Spot Masking `~6 min`

*[Point to the comparison table between N2S and N2V]*

The second algorithm is Noise2Void by Krull et al., published at CVPR 2019, and it is currently the most widely adopted self-supervised denoising method in bioimaging. While Noise2Self and Noise2Void are conceptually very similar, there is a subtle but important difference in how masking is implemented.

In Noise2Self, we mask 5% of pixels and compute loss there. The fraction is relatively large, which provides a good gradient signal per step but potentially introduces more bias because replacing pixels with neighbours slightly distorts the input distribution. In Noise2Void, the masking fraction is much smaller — just 1.6% — and the replacement is drawn from a **structured neighbourhood** defined by a radius of 2, meaning a 5-by-5 window excluding the centre pixel. This exclusion of the centre is the blind-spot idea: the replacement value is always taken from outside the mask location, ensuring the network genuinely cannot access the original pixel value even indirectly.

The intuition behind Noise2Void is: if you ask the network to predict the original noisy value at a pixel it cannot see in its input, the optimal strategy is to predict the *signal*, not the *noise*, because the noise is uncorrelated with everything else in the image. The network cannot cheat by memorising the noise pattern — the noise it sees at neighbouring pixels tells it nothing about the noise at the masked location.

*[Show training cell]*

Architecturally, the N2V model uses the exact same U-Net as Noise2Self. The only difference is the masking strategy. This is an intentional design choice in the notebook — we want to isolate the effect of the masking strategy from any architectural differences. The training loop is identical: Adam optimiser, cosine annealing over 40 epochs, MSE loss on masked pixels only.

You will notice that the Noise2Void training loss is typically a bit lower than Noise2Self's because it masks fewer pixels per step, so the loss is computed over fewer, more independently selected points with less replacement bias. Whether that translates to better final image quality depends on the specific image and noise level — that is actually something the FRC curves will help us assess.

---

## Section 5 — Probabilistic Noise2Void: Uncertainty Estimation `~6 min`

*[Point to the NLL loss equation]*

The third variant is Probabilistic Noise2Void, introduced by Krull et al. in 2020 in Frontiers in Computer Science. It takes the same blind-spot masking idea from N2V and makes one significant change to the output: instead of predicting a single intensity value at each pixel, the network predicts the **parameters of a Gaussian distribution** — a mean mu-theta and a log-variance log sigma-squared-theta.

The training loss is then the **negative log-likelihood** under this Gaussian, evaluated at the masked pixel locations. If you expand the NLL expression — which is shown explicitly in the notebook — you get two terms: the squared prediction error normalised by the predicted variance, and a regularisation term on the log-variance itself. Intuitively, the network can reduce the first term by making sigma large, but that is penalised by the second term. So it is incentivised to predict a small sigma where the image is smooth and predictable, and a large sigma where the image is highly noisy or uncertain.

*[Show the ProbUNet code]*

The architecture change is minimal. We replace the single output convolution of the standard U-Net with two parallel output heads — `out_mean` and `out_logvar` — both 1-by-1 convolutions operating on the final decoder feature map. The log-variance output is clamped between minus 10 and plus 10 for numerical stability. The resulting `uncertainty_map` is what we see rendered in the inferno colourmap in Section 7 — warmer colours indicate higher predicted uncertainty.

This uncertainty output is clinically important. After training, the denoised prediction is simply `mu_theta` — the predicted mean — which you use exactly like any other denoised image. But the sigma map gives you something extra: a per-pixel measure of how confident the network is. In practice, you would expect high uncertainty at tissue boundaries, at the edges of the lesions, and at vessel locations — precisely the places where small estimation errors matter most diagnostically. A radiologist reviewing the denoised image alongside its uncertainty map has qualitatively more information than one seeing the denoised image alone.

---

## Section 6 — Evaluation: PSNR, SSIM, and FRC `~5 min`

*[Point to the metrics table and FRC plot]*

Once all three models have been trained and have produced denoised images, we evaluate them with three complementary metrics.

**PSNR** — peak signal-to-noise ratio — is the most familiar. It is a logarithmic measure of the mean squared error between the denoised image and the clean reference. Higher is better. The noisy input starts at around 20 dB, and we are hoping our methods push this up into the mid-to-high 20s.

**SSIM** — structural similarity index — is more perceptually motivated. Rather than measuring absolute pixel differences, it compares local statistics of luminance, contrast, and structure between windows of the two images. It ranges from 0 to 1, and it is known to correlate better with human visual quality judgement than PSNR does.

**Fourier Ring Correlation** is the gold-standard resolution metric in cryo-EM and fluorescence microscopy. You may have encountered it in the context of the 0.143 half-bit criterion for determining the resolution limit of a 3D reconstruction. Here we adapt it as a *signal recovery* metric: we compute the normalised cross-correlation between the denoised image and the clean reference in each ring of Fourier space, as a function of spatial frequency. A value of 1.0 means perfect recovery at that frequency; 0.0 means no correlation at all.

*[Point to the FRC plot]*

The FRC plot is instructive. Look at the noisy input curve — it starts near 1 at low frequencies, meaning the large-scale contrast is well preserved even in the noisy image, and it drops off toward higher frequencies where the noise dominates. All three denoised curves sit above the noisy curve, confirming that denoising genuinely recovers signal at mid and high spatial frequencies. Where each curve crosses the 0.5 threshold line gives you a practical resolution limit for each method. The 0.143 threshold, shown as a dotted grey line, is a stricter criterion used in cryo-EM. If you are presenting results to a microscopy audience, quoting the FRC resolution at these thresholds is much more informative than PSNR alone.

---

## Section 7 — Visual Comparison and Training Curves `~3 min`

*[Display the training loss curves followed by the full comparison grid]*

Section 7 brings everything together visually. The training loss curves confirm that all three methods converge within 40 epochs on the T4 — you can see the cosine annealing progressively flattening each curve. Note that the PN2V loss is on a different scale from the MSE losses because it is a negative log-likelihood rather than a raw mean squared error, so you cannot directly compare the absolute values — only the convergence shape is informative across methods.

The comparison grid shows, in two rows, the full 256-by-256 images in the top row and tight central crops in the bottom row. The red rectangle on the full image marks the crop region. Look at the crop panels when assessing fine detail recovery. In the clean reference you can clearly see the vessel lines and the sharp boundaries between tissue layers. The noisy input blurs and obscures all of this. Each denoised result progressively recovers structure, with PSNR and SSIM values annotated directly in the panel titles. The rightmost panel — rendered in the inferno colourmap — is the PN2V uncertainty map, which shows higher values near boundaries and lesions as expected.

---

## Section 8 — Experiments and Extensions `~5 min`

*[Point to the three sub-experiments]*

Section 8 contains three deeper experiments that push the notebook beyond the standard comparison.

**Experiment 8.1** sweeps the input noise level sigma across six values — from a very mild 0.03 all the way up to a severe 0.30 — and measures the PSNR of both the noisy input and the N2V output at each level. The already-trained model is applied without retraining — this tests the model's generalisation. The gap between the two curves, shown as the green shaded region, is the denoising gain. You will observe that the gain grows roughly linearly as noise increases, up to a point, then the model starts to saturate — it simply cannot recover signal that has been buried under very heavy noise.

**Experiment 8.2** addresses Poisson noise, relevant for fluorescence microscopy and low-dose CT. There is a practical challenge: our N2V model was trained assuming Gaussian noise, but Poisson noise has signal-dependent variance, which violates that assumption. The standard solution is the **Anscombe variance-stabilising transform**: f of x equals 2 times the square root of x plus 3-eighths. The key property is that for large x, the transform makes the variance of a Poisson-distributed variable approximately constant and equal to 1, regardless of the mean. So we apply the Anscombe transform before passing the image to the Gaussian-trained network, and then apply the inverse transform afterward. You can see this working in the side-by-side comparison — the N2V-plus-VST output is substantially cleaner than the raw Poisson-noisy image.

**Experiment 8.3** is deliberately a failure case. We add horizontal stripe noise — rows of the image are offset by random amounts, simulating the kind of readout artefact that appears in EPI MRI sequences or in some CT scanner configurations. This noise has a critical property: it is **correlated along rows**, which means the independence assumption is violated. When we apply N2V to this image, the denoiser makes limited improvement on the stripes, even though it recovers some of the pixel-level Gaussian component. The printed warning at the bottom of the cell is intentional: it points the reader toward methods like Noise2Fast or Structurally Blind Spot Networks, which are specifically designed to handle structured, correlated noise.

---

## Section 9 — Summary and Method Guide `~2 min`

*[Point to the method comparison table]*

The summary table at the end codifies the practical takeaways. All three methods — Noise2Self, Noise2Void, and Probabilistic N2V — require only noisy images for training. They differ in their loss function and whether they provide an uncertainty output. Noise2Void is the community standard for fluorescence microscopy and is the one I would recommend as a starting point. Probabilistic N2V is the natural extension when uncertainty quantification matters. Noise2Self is more flexible in its masking strategy and is worth considering if you need to tune the masking hyperparameters for an unusual noise model.

The applicability guide below the table is worth reading carefully. The methods work well for MRI thermal noise, low-dose CT, fluorescence and cryo-EM microscopy, and multi-scanner harmonisation. They are less appropriate for ultrasound speckle — which is multiplicative and signal-dependent — or for MRI artefacts that are spatially correlated, as we just saw in Section 8.3.

---

## Section 10 — Reflection Questions `~3 min`

*[Show the final section briefly]*

The notebook closes with twelve reflection questions organised into four tiers of difficulty. Parts A and B can be answered purely from what you have just seen — they test conceptual understanding of J-invariance, the masking strategies, the metrics, and the uncertainty maps. Parts C and D go deeper. Part C asks you to engage carefully with the implementation — for example, deriving the Anscombe transform's variance-stabilising property using the delta method, or working out the minimum overlap required for artefact-free tiling. Part D is open-ended and research-oriented: proposing modifications for structured noise, thinking through the clinical validation requirements for a hospital deployment, and connecting the self-supervised denoising objective to the denoising score-matching loss that underpins modern diffusion models.

I would suggest using these questions as a study guide after you run the notebook yourself. Questions Q1 through Q6 make for good discussion in a reading group. Questions Q10 through Q12 could form the basis of a follow-up project.

---

## Closing `~1 min`

To summarise: we have built three self-supervised denoising methods from scratch in PyTorch, trained all of them on a single GPU in under 30 minutes without any clean training data, evaluated them with PSNR, SSIM, and Fourier Ring Correlation, explored where they succeed and where they break down, and framed the whole thing within the broader mathematical principle of J-invariant functions. The core message is that the gap between supervised and self-supervised denoising is much smaller than it might appear — and for modalities where clean ground truth simply does not exist, self-supervised methods are not just a compromise, they are the right tool for the job.

The notebook is self-contained and runs end-to-end on a free Colab T4 in about 25 minutes. I am happy to go through any section in more detail, or to discuss extensions to 3D volumetric data or to real clinical image collections. Thank you.

---

## Appendix: Quick Reference for Q&A

| Question | Key point |
|---|---|
| Why not just train on pairs from a denoising filter? | Filter-generated pseudo-targets introduce systematic bias — the network learns to mimic the filter's artefacts, not to recover the true signal. |
| Why does the MSE loss on masked pixels equal supervised MSE? | Because noise at masked pixels is independent of noise at unmasked pixels; the cross-term in the bias-variance expansion vanishes under independence. |
| Can N2V handle 3D MRI volumes? | Yes — replace 2D convolutions with 3D, extend the masking to sample 3D neighbourhoods, and tile along all three spatial axes during inference. Memory is the main constraint on a T4 at 15 GB. |
| How does N2V compare to BM3D or NLM? | BM3D and NLM are hand-crafted algorithms that do not require any training; N2V tends to outperform them at moderate to high noise levels once trained, and generalises better to non-Gaussian noise models. |
| What about diffusion model-based denoising? | Score-based / diffusion models achieve state-of-the-art results but require either paired data or many noisy realisations and substantially more compute. N2V remains competitive when training data is truly limited to a single image. |
| Is there a ready-made implementation to use in research? | Yes — the `n2v` Python package from the Jug lab (available on PyPI) and the napari-n2v plugin are widely used in biological imaging. |
