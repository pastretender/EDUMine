# Presentation Speech: Multimodal Image Registration Tutorial
### Group Meeting — Estimated Duration: 35–50 minutes

---
> **How to use this document**
> Text in *[square brackets in italics]* are stage directions — notes for you as the presenter on what to do or what should be visible on screen. They are not spoken aloud. Everything else is your spoken script, written in a natural register you can adapt to your own voice.

---

## OPENING

Good morning/afternoon everyone. Today I want to walk you through a tutorial I put together on multimodal image registration. The whole thing runs as a Jupyter notebook on Google Colab, so everything you see today is completely reproducible — you can run it yourself on a free T4 GPU in under five minutes.

I designed this to be a self-contained end-to-end treatment of the topic, starting from the very basics and building all the way up to a working deep learning registration network. Whether you've never touched registration before or you've used it as a black box and always wanted to understand what's happening under the hood, I hope there's something useful here for everyone.

Let me give you a quick roadmap of what we'll cover. We have seven parts: setup and data generation, a deep-dive on similarity metrics, classical registration using SimpleITK — that's three methods: rigid, affine, and deformable B-spline — then a full deep learning registration network built from scratch in PyTorch, a proper evaluation section with multiple metrics, a speed comparison, and finally a best-practices guide for real-world use.

Let's get into it.

---

## PART 1 — Setup, Installation & Synthetic Data

*[Scroll to the title cell and the Part 1 header in the notebook.]*

The first thing the notebook does is install everything it needs. We bring in SimpleITK, which is the workhorse for classical registration, ANTsPy, DIPY, nilearn, nibabel, and of course numpy, matplotlib, and PyTorch. If you're running this for the first time, this cell takes about a minute. After that, all subsequent sessions are instant because Colab caches the packages.

The imports cell also does an important GPU check. *[Point to the GPU check block.]* It queries PyTorch for the CUDA device name and total VRAM. On a T4 you'll see roughly 15 gigabytes, which is more than enough for everything we do here. If you forget to switch the runtime to GPU, you'll get a warning and a suggestion of exactly where to go to fix it.

Now, rather than immediately downloading a large clinical dataset — which would slow things down and add a dependency — we generate our own synthetic brain phantom from scratch. This turns out to be a really useful pedagogical choice, because it gives us ground truth. We know exactly what the correct alignment is, which means we can measure registration accuracy objectively.

*[Run the phantom cell and let the three-panel figure render.]*

Here's what we get. On the left, you can see the tissue-label map — five classes: background, CSF, grey matter, white matter, and a small synthetic lesion. The middle image is our T1-like modality. Notice the contrast: white matter is bright, CSF is dark, grey matter sits somewhere in between. On the right, the T2-like modality. Same anatomy — exactly the same tissue boundaries — but completely different contrast. White matter is now dark. CSF is bright. The lesion, which was moderately dark on T1, appears very bright on T2.

This is the fundamental challenge of multimodal registration. The images look completely different even though they show the same underlying structure. We generate these using tissue-specific intensity distributions — a mean and standard deviation per tissue class per modality — and then apply a mild Gaussian blur to simulate the point spread function of a real scanner.

*[Run the misalignment cell.]*

Next we apply a known ground-truth misalignment to create our problem. We rotate the T2 image by 8 degrees and translate it by 18 pixels horizontally and minus 12 pixels vertically. This becomes our "moving" image. The T1 stays fixed. The goal of every registration method we try will be to recover exactly this transform.

The checkerboard visualisation on the right is a standard tool in registration. You interleave tiles from the fixed and moving images in an alternating grid pattern. When registration is good, the anatomy should look continuous as you cross tile boundaries. Right now, you can clearly see discontinuities — edges don't line up, the ventricles jump. We'll come back to this checkerboard after each registration step to see how we're doing.

---

## PART 2 — Similarity Metrics: Why MSE Fails, and What to Use Instead

*[Scroll to the Part 2 header.]*

Before we can register anything, we need to understand what we're optimising. This is the part of multimodal registration that trips people up most often, and I think it's worth spending a few minutes on it carefully.

In monomodal registration — say, aligning two T1 brain scans from the same scanner — the simplest possible approach works: just minimise the mean squared error between pixel intensities. If the anatomy is aligned, the intensities should match, so MSE goes down. Makes sense.

In multimodal registration, this logic completely breaks down. The same anatomical location has a high intensity on one modality and a low intensity on the other. If you tried to drive MSE down, you'd actually be pushing the images apart. The metric is anti-correlated with alignment.

What we need instead is a metric that captures the *statistical relationship* between the two images, rather than direct intensity equality. The classic choice is Mutual Information, or MI. The formula is on screen here *[point to the MI formula]* — it measures how much knowing the intensity at a pixel in one image reduces your uncertainty about the intensity at the corresponding pixel in the other image. When images are well-aligned, there's a strong, consistent statistical relationship between intensities: every time you see a bright T1 value, you consistently see a dark T2 value at the same location. That predictability is high mutual information.

We also introduce Normalized Mutual Information, or NMI, which divides the sum of the marginal entropies by the joint entropy. This makes the metric more robust when the two images don't fully overlap — for instance, near image borders where there's field of view truncation.

*[Run the metrics cell and wait for the figure to render.]*

This figure is the key result of Part 2, and I want to walk through it carefully. The top left shows the joint intensity histogram when the images are perfectly aligned. The axes are T1 intensity and T2 intensity. What you see is a few tight, bright diagonal streaks — each one corresponding to a tissue class. White matter is always bright T1 and dark T2, so all the white matter pixels cluster in the upper-left. CSF always gives dark T1 and bright T2, so it clusters in the lower-right. The histogram is compact and structured. That's a high NMI — around 1.7 or so.

The top right shows the same histogram when the images are misaligned. The streaks have blurred and merged. The statistical relationship has broken down because pixels that were previously corresponding the same tissue type are now spatially offset. NMI drops significantly.

Now look at the three landscape plots below. We're sweeping the rotation angle over a range from minus 20 to plus 20 degrees offset around the ground truth, and plotting each metric as a function of that sweep.

The MSE landscape is on the top right. Notice that it is completely flat, noisy, and does *not* have a minimum at zero — the ground truth alignment. MSE is genuinely useless here. If you tried to optimise it you'd get garbage.

The MI and NMI landscapes tell a completely different story. Both have a clear, sharp peak at zero — exactly at the ground truth alignment. The NMI curve is slightly smoother and more well-defined than the raw MI curve, which is why in practice we prefer NMI for implementations where overlap may vary.

This is the theoretical justification for everything that follows. We will use Mutual Information — specifically Mattes Mutual Information as implemented in SimpleITK — as our similarity metric for all the classical registration experiments.

---

## PART 3 — Classical Registration with SimpleITK

*[Scroll to the Part 3 header.]*

Now let's actually register something. We're going to use SimpleITK, which is the Python interface to the ITK registration framework. This is production-quality software that has been used in tens of thousands of clinical and research pipelines. It's not a toy — it's what you'd use if you needed to register images in a real study.

The registration framework in SimpleITK has three components you always need to specify: a transform model, a metric, and an optimizer. We built a helper function called `setup_registration` that lets us swap any of these out cleanly with a single argument.

One architectural decision that applies to all three classical methods is multi-resolution registration. We never optimise at full resolution from the start, because the optimisation landscape at full resolution is extremely noisy — full of local minima. Instead, we start at one-eighth resolution, find an approximate solution, then pass that as the initialisation to the half-resolution step, and so on up to full resolution. This coarse-to-fine strategy is one of the most important practical choices in classical registration.

### 3.1 — Rigid Registration

*[Run the rigid registration cell.]*

The first and simplest transform model is rigid: rotation plus translation. In 2D that's three degrees of freedom. We initialise using a centred transform initialiser, which aligns the image centres of mass as a starting point — this alone gets you most of the way there for brain images.

The optimizer is gradient descent with line search, stepping in the direction that maximally increases Mattes Mutual Information. We log the metric value at every iteration so we can plot the convergence curve.

Look at the printed results when this cell runs. The notebook reports both the ground truth parameters we applied — 8 degrees, 18 pixels, minus 12 pixels — and the estimated parameters recovered by registration. You'll typically see the angle error is less than one degree and the translation error is a pixel or two. That's essentially perfect recovery of a known transform.

The convergence plot shows the Mattes MI value, reported as a negative number because the optimizer minimises. You can see it starts relatively flat during the coarse-resolution phase, then drops more sharply as it refines at higher resolutions, and eventually levels off. That S-shaped convergence curve is typical and healthy.

And the checkerboard on the right — compare it to the one from Part 1. The anatomy now lines up across tile boundaries. The ventricles are continuous. The cortex edge is smooth. Registration worked.

### 3.2 — Affine Registration

*[Run the affine registration cell.]*

Rigid registration is appropriate when the two images come from the same subject and you only need to correct for head movement. But many real scenarios introduce additional distortions. Different scanners have different geometric distortions. Multi-site studies may have slight scale differences. Affine registration adds scaling and shearing to the rigid transform, giving us 6 degrees of freedom in 2D and 12 in 3D.

For this experiment, we create a more complex misalignment: rotation, a small shear of 5%, anisotropic scaling of 3% difference between axes, and translation. This represents something like the scanner-induced distortions you'd see in a multi-site study.

The setup is identical — same Mattes MI metric, same multi-resolution pyramid — but the transform model is now an AffineTransform. The results show that NMI improves significantly, and the checkerboard confirms much better alignment than before the registration.

A practical note here: affine registration is always a better initialisation for deformable registration than rigid alone, because it eliminates all the global-scale distortions before you try to find local ones. Always do rigid, then affine, then deformable — never skip steps.

### 3.3 — Deformable B-Spline Registration

*[Run the B-spline cell.]*

For inter-subject registration, or any scenario where the two images show genuinely different anatomy — different brain shapes, tumours, organ deformation due to breathing — we need a deformable transform. The classic choice in SimpleITK is B-spline deformation.

The idea is that we place a sparse grid of control points over the image, and the displacement field at any pixel is computed as a weighted sum of B-spline basis functions centred on nearby control points. The formula is on screen. This gives us a smooth displacement field that can capture complex local shape differences, while the sparsity of the control point grid acts as an implicit regulariser — it prevents the field from becoming physically implausible.

For this experiment, we add a random smooth deformation on top of the rigid misalignment using a Gaussian-smoothed random field. This simulates the kind of local shape differences you'd see between two different subjects.

The strategy here is a two-stage pipeline. First we run rigid registration to remove the global offset. Then we refine with B-spline to correct the remaining local deformations. You should never start with B-spline directly — without a good rigid initialisation, it will get lost.

The fourth panel in the result figure shows the displacement field magnitude. You can see that it's not uniform — the field is larger in regions where the random deformation was stronger. The NMI progression is printed below: baseline, then after rigid correction, then after B-spline refinement. Each step meaningfully improves the score.

---

## PART 4 — Deep Learning Registration with a VoxelMorph-style U-Net

*[Scroll to the Part 4 header.]*

So classical registration works, but it has a fundamental limitation: every single image pair requires its own iterative optimisation, which takes anywhere from a few hundred milliseconds to several minutes depending on the transform model. In a clinical workflow where you might need to process thousands of patients, or in an application like real-time surgical guidance, that's completely impractical.

Deep learning registration takes a different approach. Instead of optimising per image pair, we train a neural network once on a large set of image pairs. At inference time, the network takes a fixed and moving image as input and predicts the displacement field in a single forward pass — no iterative optimisation at all. The cost of the training is amortised across all the image pairs you'll ever register.

The architecture we implement is inspired by VoxelMorph, the landmark 2019 paper from Balakrishnan et al. at MIT. It's a U-Net encoder-decoder. Both the fixed and moving images are concatenated along the channel dimension, so the network receives 2-channel input. The encoder progressively downsamples, learning increasingly abstract representations of the anatomical structure. The decoder upsamples back to full resolution, using skip connections to recover fine-grained spatial detail. At the output, a small convolutional head produces a 2-channel displacement field — one channel for displacement in x, one for y.

At the very end of the network sits a Spatial Transformer Network. This is the differentiable warping module: it takes the moving image and the predicted displacement field, and produces the warped "moved" image. Because it's differentiable — we use bilinear interpolation, which is smooth and differentiable everywhere — the entire pipeline from input images to warped output is end-to-end trainable with gradient descent.

*[Run the U-Net cell and show the parameter count.]*

Our implementation has roughly half a million parameters. That's deliberately lightweight — we want something that trains in a few minutes on Colab's T4. A production VoxelMorph would have more capacity, but the principles are identical.

Now let's talk about the loss function, because this is where the multimodal aspect comes back.

*[Run the losses cell.]*

We have two terms. The first is the similarity term: we want the moved image to look like the fixed image. But we can't use MSE — we've established that. We also can't directly use Mutual Information in a standard neural network training loop, because MI is computed via histogram estimation, which isn't differentiable in the standard sense.

Instead we use Local Normalized Cross-Correlation, or LNCC. The idea is to compute a local correlation coefficient in a sliding window across the image, rather than globally. In each local patch, the contrast relationship between the two modalities tends to be locally monotonic even if it varies globally. So measuring how well a local patch in the fixed image predicts the corresponding patch in the moved image captures the same information as MI, but in a form that's fully differentiable via standard backpropagation. You can see the implementation here — it uses standard 2D convolutions with a box filter kernel to compute local means and standard deviations, then produces the correlation coefficient map.

The second loss term is the smoothness regulariser. We penalise the squared spatial gradient of the displacement field — the differences between neighbouring displacement vectors. This is Tikhonov regularisation, and it prevents the network from learning unrealistic, folded deformations. The hyperparameter lambda controls the trade-off: too low and you get sharp, physically implausible deformations; too high and the network won't deform enough to correct the misalignment.

*[Run the dataset cell.]*

For training data, we use an on-the-fly data generator. At each epoch, for each sample, we generate a new random rigid plus non-rigid misalignment of our phantom. This gives us effectively unlimited training data while exposing the network to the full distribution of possible misalignments. Each misalignment is different, so the network can't memorise; it must learn a general registration function.

*[Run the training cell and let it run. While it trains, continue speaking.]*

The training loop is running now. It will take about two to three minutes on the T4. We're training for 25 epochs with a batch size of 4, using the Adam optimizer with a cosine annealing learning rate schedule. You can see the loss printed every five epochs — total loss, the LNCC similarity component, and the smoothness regularisation component.

What you should see is the total loss decreasing monotonically, the LNCC term driving most of that decrease, and the smoothness term staying relatively stable. The cosine schedule gradually reduces the learning rate toward the end of training to allow fine-grained convergence.

*[Wait for training to complete, then run the visualisation cell.]*

Now let's look at the result. The training curves on the left confirm healthy convergence — no plateaus, no oscillations, clean monotonic decrease.

On the right, the displacement field visualisation is particularly illuminating. The colour map shows the magnitude of the predicted displacement at every pixel, and the white arrows show the direction. You can see that the displacement is concentrated in the regions where the misalignment was strongest — around the edges of the brain where the rotation effect is most pronounced — and it's smooth and continuous everywhere. There are no sharp discontinuities, which is exactly what we want from a physically plausible deformation.

The inference time printed in the title is the key number. The forward pass through the trained network takes on the order of single-digit milliseconds on the T4. Compare that to even the fast rigid registration in SimpleITK, which takes hundreds of milliseconds. That's a speedup we'll quantify precisely in Part 6.

---

## PART 5 — Evaluation

*[Scroll to the Part 5 header.]*

A common mistake in registration papers is to evaluate only using the image similarity metric — in our case, NMI. This is problematic for a subtle but important reason: you can increase NMI while actually making the anatomical alignment worse, if the network learns to hallucinate texture that matches the fixed image rather than genuinely moving anatomy into alignment. NMI alone does not guarantee anatomical accuracy.

We implement a comprehensive evaluation suite with five different metrics, each measuring a different aspect of quality.

NMI we've discussed — intensity alignment.

Dice score measures anatomical overlap. We take the warped label map from the moving image and compare it to the ground truth label map from the fixed image. For each tissue class, Dice is two times the intersection divided by the sum of areas. A score of 1 means perfect overlap, 0 means no overlap at all.

Target Registration Error, or TRE, measures the mean distance between corresponding anatomical landmarks after registration. It's the gold standard in clinical registration validation — if a radiologist manually identifies the same anatomical point in both images, TRE tells you how far apart those points end up after registration.

The Jacobian determinant of the displacement field tells us whether the deformation is physically plausible. At every pixel, the Jacobian determinant describes the local volume change. If it's positive everywhere, the deformation is bijective — there's a unique correspondence between every point in the fixed and moving image. If the Jacobian determinant is negative anywhere, the deformation has folded — meaning two different source locations map to the same target location, which is physically impossible. We report the percentage of pixels with negative Jacobian determinant as a measure of deformation quality.

*[Run the evaluation cell.]*

The summary table is the result. Let me walk through it column by column. The "Before" column is our baseline — no registration, just the raw misalignment. The "Rigid" column is SimpleITK rigid registration. The "DL" column is our trained network.

For CSF, grey matter, and white matter, both methods achieve substantial improvements over baseline. Dice for white matter, for instance, typically jumps from around 0.6 or 0.7 at baseline to above 0.9 after registration. The lesion class tends to be the hardest — it's small and structurally ambiguous — so expect lower absolute Dice there, but still a meaningful improvement.

The NMI row confirms the pattern we'd expect: both methods improve intensity alignment significantly.

The Jacobian determinant stats, printed below, show that our DL deformation field has very few or zero negative values, meaning the smoothness regulariser is doing its job. A well-regularised network should produce less than one percent negative Jacobians as a rule of thumb.

*[Run the evaluation visualisation cell.]*

The bar chart makes the comparison visually immediate. Each tissue class on the x-axis, grouped bars for before, rigid, and DL registration. Both registration methods are clearly better than no registration across all classes. The DL and rigid methods are competitive on this synthetic phantom — which makes sense, since the true misalignment here is predominantly rigid. On real data with genuine inter-subject shape variation, the deformable approach would show a larger advantage.

---

## PART 6 — Speed Comparison & Method Summary

*[Scroll to the Part 6 header and run the speed benchmark cell.]*

Let's put numbers on the speed difference. We time both the classical rigid registration and the DL inference using Python's timeit module, averaging over 10 repetitions to get stable estimates.

*[Wait for the benchmark to complete.]*

The numbers speak for themselves. Classical rigid registration in SimpleITK, even with a minimal two-level pyramid, takes several hundred milliseconds per image pair. DL inference on the T4 takes single-digit milliseconds. That's a speedup of somewhere between 50x and 200x depending on the exact setup.

The table summarises the key trade-offs across all four approaches we've touched on. SimpleITK rigid is fast among classical methods, requires no training, and gives clean linear transforms. SimpleITK B-Spline is deformable but slow — five to thirty seconds per pair for the 2D case, much more for 3D volumes. VoxelMorph is fast at inference, does deformable registration, but requires a training phase. ANTs SyN is the current gold standard for neuroimaging — diffeomorphic, invertible, highly accurate — but it's the slowest of all, often taking minutes per pair.

The right choice depends entirely on your application. If you're doing a one-off study with fifty subjects and accuracy is paramount, use ANTs SyN. If you're processing a thousand patients in a clinical pipeline where each registration needs to happen in under a second, a trained DL model is the only viable option.

---

## PART 7 — Best Practices & Real-World Tips

*[Scroll to the Part 7 header.]*

I want to close with a practical checklist, because registration is one of those areas where the theory and the practice diverge in ways that bite people.

On the **data preprocessing** side: always skull-strip your brain MRI before registration. The scalp and skull have very different appearances across subjects and modalities, and they confuse both the metric and the transform model. Intensity normalise each image independently — not together. Resample both images to isotropic voxel spacing. And critically, check that the orientation headers match — RAS versus LAS is a common source of mysterious registration failures that's embarrassing to track down.

For **classical registration**: multi-resolution pyramids are non-negotiable. And the ordering rule is absolute: always rigid before affine before deformable. Using Mattes Mutual Information for CT-to-MRI is standard; for within-MRI registration like T1 to T2, NMI is often more stable. You can also safely sample only 10 to 20 percent of voxels per iteration without meaningfully reducing accuracy — this gives a large speed improvement.

For **deep learning registration**: your training augmentation must cover the full distribution of misalignments you expect to see at test time. A network trained only on small rotations will fail on large ones. Use diffeomorphic integration — replacing the direct displacement field with a stationary velocity field and integrating it — if you need guaranteed invertibility. Tune the lambda smoothness weight carefully on a validation set; it's one of the most influential hyperparameters.

For **evaluation**: never report only NMI. Compute Dice on segmentations whenever you have them. Report TRE if you have landmarks. Always check the Jacobian determinant histogram — it's a quick sanity check that catches catastrophically bad deformations that look fine in NMI. And for clinical applications, expert visual inspection remains essential and irreplaceable.

*[Run the final visualisation cell.]*

This last figure gives a complete side-by-side summary of the whole pipeline. The top row shows the intensity images: fixed T1, misaligned T2, rigid-registered result, B-spline result, and DL result. The middle row overlays the corresponding segmentation labels on each image. The bottom row shows the absolute difference from the fixed image — brighter means larger error. You can trace the improvement across methods visually and see how the error maps collapse toward zero as registration quality improves.

---

## CLOSING

So to summarise what we've built today: a complete multimodal image registration pipeline implemented from scratch in Python, covering the theoretical foundations — why MSE fails and why we need information-theoretic metrics — through three levels of classical registration using SimpleITK, and a full deep learning approach implemented in PyTorch with a U-Net, spatial transformer, LNCC loss, and GPU-accelerated training. We evaluated rigorously across multiple metrics and put hard numbers on the speed difference.

All of this runs in a single Colab notebook on a free T4 GPU. The further reading section at the bottom of the notebook points to the four foundational papers — Mattes et al. 2003, Rueckert et al. 1999, Balakrishnan et al. 2019, and Avants et al. 2008 — as well as four production-grade libraries and four publicly available datasets if you want to apply any of this to real data.

The natural next steps from here are: applying this to real 3D brain MRI — SimpleITK and the DL architecture both extend to 3D with minimal changes; implementing diffeomorphic registration using the scaling-and-squaring trick for guaranteed invertibility; and trying ANTs SyN for a baseline comparison on real data.

I'm happy to take questions on any part of this — the theory, the implementation details, or the practical choices.

*[Open for questions.]*

---

## APPENDIX: Likely Questions & Suggested Answers

**Q: Why use LNCC in the DL model rather than just computing NMI differentiably?**
Differentiable NMI estimators exist — they typically use Parzen window density estimation or soft histogram approximations. The challenge is that these are computationally expensive and often numerically unstable during early training. LNCC is a well-understood, fast, and stable surrogate that captures the same local correlation structure. For most practical purposes, LNCC and differentiable NMI produce similar registration quality.

**Q: Why InstanceNorm in the U-Net rather than BatchNorm?**
Registration networks are often applied to single image pairs at inference time, so batch statistics are unreliable. InstanceNorm normalises within each individual feature map, making it well-suited for the one-at-a-time inference pattern and for variable-contrast images. LeakyReLU is used rather than standard ReLU because the displacement field values can be negative, and we want gradients to flow even in the negative regime.

**Q: What happens with very large deformations that push tissue boundaries significantly?**
Large deformations are inherently harder for both classical and DL approaches. For classical methods, the multi-resolution strategy helps by finding large-scale alignment at coarse resolution first. For DL methods, the key is that the training distribution must include examples of comparably large deformations. Architecturally, using a stationary velocity field with diffeomorphic integration constrains the network to produce physically valid large deformations by construction.

**Q: Can this be extended to 3D volumetric data?**
Yes. Every component generalises directly. In SimpleITK you simply pass 3D images and the same `setup_registration` function works. In the U-Net, you replace `nn.Conv2d` with `nn.Conv3d`, `InstanceNorm2d` with `InstanceNorm3d`, and adjust the Spatial Transformer grid accordingly. The main practical challenge is memory — a 3D volume at full resolution may not fit in a single forward pass on 16 GB VRAM, so patch-based inference or mixed-precision training is often required.

**Q: What's the state of the art for neuroimaging registration today?**
For atlas-based registration and segmentation propagation in neuroimaging, ANTs SyN remains the empirical gold standard for accuracy. For speed-sensitive applications, the VoxelMorph family and its extensions — including TransMorph, which replaces the U-Net with a Swin Transformer — are the most actively developed. The latest work focuses on combining the accuracy of ANTs with the speed of DL using distillation: training a network to mimic ANTs outputs.
