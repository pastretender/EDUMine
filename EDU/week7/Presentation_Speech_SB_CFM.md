# Presentation Speech
## Cross-Modal Medical Image Synthesis via Schrödinger Bridge Flow Matching
### Group Meeting — Full Walkthrough

---

> **Estimated delivery time:** 35–50 minutes (depending on Q&A pauses)
> **Delivery notes:** Sections marked *[PAUSE]* are natural points to invite brief questions or comments. Sections marked *[POINT TO NOTEBOOK]* mean you should scroll to or display the corresponding cell on screen.

---

## Opening

Good morning / afternoon, everyone. Today I want to walk you through a tutorial I've put together on a generative modelling technique that I think has real practical relevance for anyone working with medical imaging data — and, honestly, for anyone thinking about distribution transport problems more broadly.

The title is **Cross-Modal Medical Image Synthesis via Schrödinger Bridge Flow Matching**, and the concrete task we use as a running example is synthesising T2-weighted MRI from T1-weighted MRI — with a specific motivation around rare pathology data augmentation.

I'll go through the notebook section by section, covering both the intuition behind each design choice and the specific implementation. The whole thing is written to run end-to-end on a free Google Colab T4 GPU, so if you want to reproduce any of this after the meeting, you can do so in about 30 to 40 minutes of compute time.

Let me start with the motivation, because I think it's important to establish *why* this problem is worth solving before we get into how.

---

## Opening: The Core Problem

Imagine you're working with a hospital that has collected five hundred T1-MRI scans of patients with a rare brain tumour — say, an oligodendroglioma. Now, a radiologist needs a T2 scan to properly characterise the lesion, because T1 and T2 capture fundamentally different tissue contrast. The problem is that for rare pathologies, you might have only eight or ten T2 scans in the entire dataset. You cannot train a reliable segmentation model on eight examples.

The traditional answer has been CycleGAN — you learn a mapping between the two modalities using adversarial training, without requiring paired data. And while CycleGAN works reasonably well in clean settings, it suffers from mode collapse, it's notoriously unstable to train, and — critically — it has no principled notion of *how* it transports information from one modality to the other. It's a black box.

The other option is diffusion models, DDPM-style. These are stable and high quality, but they start from Gaussian noise, not from the source image. That means they are not naturally suited to *paired* synthesis — they need hundreds or even thousands of denoising steps, and they have no inherent mechanism to preserve the anatomical structure of the source scan in the output.

What we want is something that starts *at the source image* and ends *at the target modality* — following the most efficient path between them, with as few steps as possible, and with strong structural guarantees. That's exactly what the Schrödinger Bridge gives us.

*[PAUSE — invite any initial reactions before diving in]*

---

## §1 · Setup and Dependencies

*[POINT TO NOTEBOOK — §1 cells]*

Let's begin at the beginning. The setup section installs four libraries that may not be familiar to everyone.

The first is **MONAI** — the Medical Open Network for AI. If you haven't used it, think of it as a PyTorch extension purpose-built for medical imaging. It provides NIfTI-aware data loaders, a rich transform pipeline, and network architectures like `BasicUNet` that are designed around the conventions of medical imaging. We'll use it heavily throughout.

The second is **torchcfm**, which is the reference implementation of Conditional Flow Matching from Lipman et al. and Tong et al. Crucially, it ships with a `SchrodingerBridgeConditionalFlowMatcher` class that handles the SB-specific sampling and vector field computation for us — we don't need to implement the bridge math from scratch.

We also install **nibabel** for reading NIfTI `.nii.gz` files, which is the standard volume format for MRI, and **einops** for clean tensor manipulation.

In the configuration cell, I've gathered all the key hyperparameters in one place so they're easy to modify: the image resolution is 256 by 256 pixels; the batch size is 8, which fits comfortably in the T4's 16 gigabytes; we train for 60 epochs; and we set sigma-min — the Schrödinger Bridge noise parameter — to 0.01. I'll explain what that last one means when we get to the math section.

One important note on reproducibility: we fix all random seeds — Python's `random`, NumPy, and PyTorch — but we intentionally leave `cudnn.benchmark` enabled for speed. If you need bitwise-identical results across runs, you'd set `cudnn.deterministic = True`, but for a tutorial the benchmark mode is fine.

---

## §2 · The Dataset and MONAI Pipeline

*[POINT TO NOTEBOOK — §2 cells, sample_pairs.png output]*

This is where we set up our data. Now, the *ideal* dataset for a tutorial like this would be something like SynthRAD 2023 — real paired MRI and CT from multiple hospitals — but that requires institutional data access. Instead, we use a **synthetic data generator** that I've designed to capture the essential biophysical properties of paired T1 and T2 MRI.

Here's the physics intuition behind the contrast model. In T1-weighted MRI, tissues are ordered from bright to dark roughly as: white matter, grey matter, and then cerebrospinal fluid, which appears nearly black. In T2, that order essentially *inverts*: cerebrospinal fluid becomes very bright, grey matter moderately bright, and white matter darker. This inversion is what makes T1-to-T2 synthesis non-trivial — you're not just performing a global intensity rescaling, you're learning a tissue-type-specific mapping.

Our synthetic generator creates this by building three components: first, a smooth ellipsoidal brain mask using a Gaussian-blurred binary ellipse to mimic the skull boundary; second, a tissue label map with three classes — CSF, grey matter, and white matter — using spatially correlated Gaussian noise thresholded at two quantile levels; and third, lookup tables mapping those tissue labels to the appropriate T1 and T2 intensities, with a small amount of spatially correlated noise added on top.

The output is always a pair of float32 arrays in the range negative one to positive one, which is the standard normalisation for generative models. We generate 950 slices total — 800 for training, 100 for validation, 50 for test.

I want to emphasise one key design decision here: we intentionally work in **2D slices**, not 3D volumes. On a T4 GPU, a batch of eight 256-by-256 float16 slices fits comfortably in memory, whereas equivalent 3D patches would require multi-GPU setups. The 2D approach is also clinically motivated — radiologists read 2D slices, and most clinical segmentation models are 2D or 2.5D.

*[POINT TO NOTEBOOK — sample_pairs.png]*

Looking at the visualisation output, you can see the characteristic contrast reversal between the T1 top row and the T2 bottom row. White matter regions appear bright in T1 and dark in T2; the CSF in the ventricles does the opposite. This is exactly the kind of nonlinear paired relationship that our flow matching model needs to learn.

The `PairedSliceDataset` class wraps these arrays into PyTorch tensors with a single channel dimension, yielding pairs of shape one-by-256-by-256. During training we apply a random horizontal flip — critically, to *both* modalities simultaneously so they stay aligned. This is a subtlety that's easy to get wrong: any spatial augmentation must be applied identically to source and target to preserve the pairing.

The DataLoaders use `pin_memory=True` and non-blocking device transfers, which on the T4 gives noticeably faster data throughput during training.

---

## §3 · Mathematical Foundations

*[POINT TO NOTEBOOK — §3 Markdown cell]*

Now for the theory. I want to spend some real time here because the mathematical foundation is what distinguishes this approach from everything that came before it.

**Conditional Flow Matching** — the parent framework — was introduced by Lipman, Le, Nesterov, and colleagues in 2022. The core idea is to learn a neural network vector field that, when integrated as an ODE from time zero to time one, maps samples from a source distribution to a target distribution. The key innovation over score-based methods is that instead of minimising an intractable marginal objective over all of probability space, we condition on individual data pairs and minimise what the paper calls the conditional flow matching objective.

The loss is: the expected mean-squared error between the network's predicted velocity and the ground-truth target vector field, where expectation is taken over time, over data pairs, and over the conditional path distribution. Written out, it looks like this:

$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t,\, q(x_0, x_1),\, q_t(x \mid x_0, x_1)} \left[\| v_\theta(x_t, t) - u_t(x \mid x_0, x_1) \|^2\right]$$

The **conditional probability path** for a pair of images x-zero and x-one is the linear interpolant:

$$x_t = (1 - t)\, x_0 + t\, x_1$$

And the **target vector field** — the ground-truth velocity that the network is trying to learn — is simply:

$$u_t = x_1 - x_0$$

This is the beautiful insight at the heart of flow matching: the target velocity is *constant in time*. For a given pair of images, the optimal transport is to move in a straight line from source to target, at constant speed. The network doesn't need to learn an evolving, time-varying velocity — it just needs to learn the direction. This is why flow matching trains so much faster than diffusion models, which must learn to denoise at every noise level.

*[PAUSE — let this sink in before continuing to the SB extension]*

Now, where does the **Schrödinger Bridge** come in? Standard conditional flow matching with linear paths works well, but there's a subtlety: it uses the independent coupling between the two distributions, meaning it doesn't explicitly use the fact that our data is *paired*. We can do better.

The Schrödinger Bridge is the solution to an entropy-regularised optimal transport problem. Formally, you're looking for the coupling pi-star between two distributions that minimises the expected squared distance between samples, plus an epsilon times the KL divergence from the product measure. This entropy regularisation introduces a small amount of stochasticity into the paths — they are no longer perfectly straight lines, but rather stochastic bridges — while still being far straighter than anything you'd get from DDPM.

The SB path distribution is Gaussian with mean at the linear interpolant and variance that is proportional to t-times-one-minus-t. Notice that the variance is *zero at both endpoints* — t equals zero and t equals one — which means the bridge is pinned to x-zero at the start and to x-one at the end. The variance peaks at the midpoint t equals 0.5, where the bridge has the most freedom to explore. The magnitude of this exploration is controlled by our sigma-min parameter.

The corresponding SB vector field is:

$$u_t^{SB}(x \mid x_0, x_1) = \frac{x_1 - x}{1 - t}$$

This is slightly more complex than the constant-velocity OT-CFM field — it's time-varying, and it points the current state toward x-one with increasing urgency as t approaches one. But the important point is that **torchcfm's `SchrodingerBridgeConditionalFlowMatcher` handles all of this automatically**. You give it a batch of paired (x-zero, x-one) tensors, and it returns the sampled time t, the interpolated state x-t, and the target velocity u-t. From the training loop's perspective, the interface is identical to standard CFM.

Finally, at inference, once the network is trained, we synthesise a target image from a new source by integrating the probability flow ODE forward from t equals zero to t equals one, starting from the source image. We implement both a first-order Euler method and a second-order midpoint method for this.

---

## §4 · Model Architecture — Time-Conditional U-Net

*[POINT TO NOTEBOOK — §4 cells]*

With the math established, let's talk about the neural network that learns to predict this velocity field.

The backbone is **MONAI's `BasicUNet`** — an encoder–decoder architecture with skip connections that is well-tested in medical imaging. I've adapted it for flow matching with three key modifications.

The first is a **sinusoidal time embedding**. Because the network needs to sense where it is along the integration path — is t close to zero, meaning we look like the source? Or is t close to one, meaning we should already look like the target? — we need to give it access to t. We do this using a fixed Fourier encoding borrowed from the original Transformer paper. The time scalar t is mapped through a bank of sine and cosine functions at logarithmically-spaced frequencies, then passed through a small two-layer MLP to produce a 256-dimensional embedding vector. This high-dimensional representation captures both low-frequency and high-frequency structure of the time axis.

One implementation detail worth noting: we scale t by 1000 before computing the Fourier features. This is because raw t values in the range zero to one would give a very narrow frequency range — the sinusoids would all look nearly constant. Multiplying by 1000 spreads the values across a much wider range, giving the embedding genuine multi-frequency content.

The second modification is **time conditioning at the bottleneck**. We inject the time embedding into the U-Net's deepest feature representation using a PyTorch forward hook. The hook intercepts the bottleneck layer's output, projects the time embedding to match the feature channel count — 256 channels in our architecture — and adds it elementwise, broadcast across the spatial dimensions. This is sometimes called AdaGN-style conditioning, though here we're doing pure addition rather than adaptive group normalisation.

The third modification is **source image concatenation**. The network's input is not just x-t — the current interpolated state — but the concatenation of x-t and x-zero along the channel axis. This gives the network explicit access to the source image at every step of inference, which is critical for *conditional* synthesis. Without x-zero as input, the network would be solving an unconditional generation problem, and there would be no guarantee that the output bears any structural relationship to the input. With x-zero concatenated, the network knows both where it is right now and where it started, which gives it all the information it needs to predict the correct direction.

The channel progression through the encoder is 32, 64, 128, 256, 256 — a fairly standard pyramidal structure that gives the network multi-scale receptive fields. We use Instance Normalization rather than Batch Normalization, which performs better for small batches and single-image inference. The final activation is a Tanh, which clips the output to negative one to positive one — consistent with our data normalisation.

The total parameter count is approximately 7 to 8 million, which is well within the T4's capacity.

---

## §5 · Training Loop

*[POINT TO NOTEBOOK — §5 cells and training curve output]*

The training section is where everything comes together. Let me walk through the exact sequence of operations in a single training step.

We draw a batch of paired slices (x-zero, x-one) — T1 and T2 respectively — and move them to the GPU with non-blocking transfers. We pass this pair to the `SchrodingerBridgeConditionalFlowMatcher`'s `sample_location_and_conditional_flow` method. This single call does three things: it samples a time t uniformly from zero to one for each element in the batch, it constructs the SB interpolated state x-t using the bridge noise schedule, and it computes the target velocity u-t.

We then pass x-t, t, and x-zero through our `FlowMatchingUNet` to get the predicted velocity v-hat. The loss is the mean-squared error between v-hat and u-t — a simple regression objective with no adversarial components, no KL divergence terms, no perceptual loss. Just MSE. This is what makes flow matching so stable compared to GAN training.

The backward pass uses **Automatic Mixed Precision** — `torch.cuda.amp.autocast` runs the forward pass in float16, and a `GradScaler` dynamically scales the loss before the backward pass to prevent gradient underflow. On the T4, this gives approximately a two-times speedup over float32 training. We also apply gradient clipping at a norm of 1.0, which prevents occasional large gradient spikes early in training when the network is far from convergence.

The optimizer is AdamW with a learning rate of 2e-4 and a weight decay of 1e-4. The learning rate follows a cosine annealing schedule that smoothly decays from the initial value down to one percent of that value over the full training run. This is important — the cosine schedule prevents the loss from plateauing prematurely and gives the final epochs a chance to refine fine-grained details.

Best-checkpoint saving compares validation loss at the end of each epoch and saves the full model state, optimizer state, epoch number, and hyperparameter configuration to disk. This means you can resume training or load the model for inference without re-running the entire notebook.

*[POINT TO NOTEBOOK — training curve plot]*

Looking at the training curve, you'll see the expected shape: rapid decrease in the first ten to fifteen epochs as the network learns the broad structure of the modality transformation, followed by a slower, smoother improvement as it refines texture and boundary details. The gap between training and validation loss should be small — this is a regression task, not a generative task with mode collapse risks, so overfitting is relatively mild.

---

## §6 · Inference — ODE Integration

*[POINT TO NOTEBOOK — §6 cells]*

Once training is complete, inference is the process of integrating the learned velocity field forward in time to transport a new, unseen source image to the target modality.

We initialise the state at x-t-equals-zero equal to x-zero — our input T1 slice. We then take N small steps, at each step querying the network for the velocity at the current state and time, and nudging the state in that direction. This is numerically the same as integrating any ODE, just with a neural network as the right-hand side.

The first-order **Euler method** is:

$$x_{t+h} = x_t + h \cdot v_\theta(x_t, t, x_0)$$

where h is one divided by the number of steps. Simple, interpretable, and fast.

We also implement the **midpoint method**, which is second-order accurate. It evaluates the velocity at the midpoint of each step, which costs two network forward passes per step but achieves the same accuracy as Euler with roughly half the number of steps. For latency-sensitive applications, this is a useful trade-off.

The cell at the bottom of this section runs a step-count sweep: we evaluate PSNR for Euler at 5, 10, 20, and 50 steps, and for the midpoint method at 5 and 10 steps. The key finding — which is a direct consequence of the Schrödinger Bridge's near-straight paths — is that quality improves very quickly with step count in the low-step regime, but plateaus well before 50 steps. In practice, 20 Euler steps or 10 midpoint steps gives near-optimal synthesis quality on this task.

This is in sharp contrast to DDPM, where you genuinely need 50 to 1000 steps to get high-quality samples. The reason is that DDPM paths curve through the high-dimensional Gaussian ball, accumulating integration error unless you take very small steps. SB-CFM paths are nearly straight, so even a few large steps give accurate integration.

---

## §7 · Evaluation and Visualisation

*[POINT TO NOTEBOOK — synthesis_results.png and flow_trajectory.png]*

Section 7 covers both quantitative metrics and visual diagnostics.

For quantitative evaluation, we use three standard metrics. **PSNR** — Peak Signal-to-Noise Ratio — is the most common, measuring the mean squared error between the synthesised and ground-truth images on a log scale. A PSNR above 30 dB is generally considered reasonable for medical image synthesis; above 35 dB is good.

**SSIM** — the Structural Similarity Index — is arguably more clinically meaningful than PSNR. It decomposes image similarity into three components: luminance, contrast, and structure, computed over local windows. A radiologist looking at an image doesn't care about individual pixel values — they care about whether edges are sharp, whether tissue boundaries are in the right place, whether the image *looks* like the correct modality. SSIM captures much more of this than PSNR.

**MAE** — Mean Absolute Error — is the simplest metric, directly interpretable in the same units as the image intensities.

*[POINT TO NOTEBOOK — synthesis_results.png]*

The side-by-side visualisation panel is the most informative output in the notebook. For each test sample, you see four columns: the T1 source on the left, our synthesised T2, the ground-truth T2, and the absolute error map on the right.

Looking at these panels, a few things stand out. First, the global contrast reversal is learned correctly: white matter regions that appear bright in T1 consistently appear dark in the synthesised T2, and vice versa for CSF. Second, the anatomy is preserved at the pixel level — the brain boundary, the major sulci, the ventricle outlines all line up correctly between source and synthesis. Third, the error map is concentrated at tissue boundaries, which is the expected pattern — these are precisely the regions where the contrast difference between T1 and T2 is largest and the network has the hardest prediction problem.

*[POINT TO NOTEBOOK — flow_trajectory.png]*

The trajectory visualisation is something I'm particularly pleased with. We store the intermediate states x-t at six evenly-spaced time points as we integrate from t equals 0 to t equals 1. What you see is a smooth, progressive transition from T1 contrast at the left to T2 contrast at the right. The path is near-linear in pixel space — there's no dramatic morphological change or artifact at any intermediate time. This is the visual signature of the Schrödinger Bridge's near-optimal transport paths, and it contrasts sharply with what you'd see in a DDPM trajectory, where intermediate states pass through a noisy, largely uninterpretable Gaussian intermediate.

---

## §8 · Rare Pathology Augmentation

*[POINT TO NOTEBOOK — §8 cells and pathology_augmentation.png]*

Section 8 brings us back to the original motivation: data augmentation for rare pathologies.

We generate a T1 slice with a synthetic lesion — a circular region that is hypointense on T1 and hyperintense on T2, which is the typical appearance of many inflammatory or oedematous lesions. We then push this slice through the trained flow and examine what happens.

The key question is whether the model *preserves* the pathological structure or *hallucinates* it. And the answer, predictably, is: it does some of both. Healthy tissue regions are transported accurately — the global T1-to-T2 contrast reversal is correct, the anatomy is preserved. But the lesion region, which the model has never seen during training, is handled with uncertainty — the network applies its learned healthy-tissue transport rules to an anomalous input and produces a plausible but potentially incorrect T2 appearance for the lesion.

This is not a failure — it is a known and manageable limitation. We quantify it using an **ensemble uncertainty map**. We run the same source image through the model eight times, each time adding a tiny amount of Gaussian noise to the input to simulate acquisition uncertainty. The variance across these eight runs gives us a per-pixel uncertainty map. And critically, the uncertainty map shows *higher variance precisely over the lesion region*. The model knows what it doesn't know.

*[POINT TO NOTEBOOK — pathology_augmentation.png]*

Looking at the five-panel output — source T1, synthesised T2 mean, ground-truth T2, absolute error, and uncertainty map — you can see the lesion is localised in the uncertainty map. This is exactly the kind of output you would want before presenting synthesised images to a radiologist: not just a synthesis, but a confidence signal that flags which regions the model is uncertain about.

The `augment_dataset` function at the end of this section is the practical takeaway. It takes a NumPy array of N source slices, runs them through the model in batches, and returns N synthesised target slices. In a real workflow, you would apply this to all 500 T1 lesion scans to generate 500 pseudo-T2 scans, dramatically expanding the training set for your downstream segmentation model.

---

## §9 · Summary Comparison Table

*[POINT TO NOTEBOOK — §9 Markdown cell]*

Section 9 collects the comparison we've been building implicitly throughout the notebook into a single table.

Let me highlight the three most important rows.

On **inference steps**: CycleGAN needs one forward pass — fast, but quality is limited and structural preservation is fragile. DDPM needs 50 to 1000 steps — high quality, but slow. OT-CFM needs 5 to 50 steps. SB-CFM needs 5 to 20. The difference from OT-CFM is modest in absolute terms, but the SB paths are provably optimal, which means the quality-per-step ratio is higher.

On **uncertainty quantification**: CycleGAN and OT-CFM produce deterministic outputs — you get a single synthesis with no principled way to assess confidence. DDPM is stochastic by design, so you can generate multiple samples and measure variance. SB-CFM also has this property through its bridge noise — at inference, you can optionally run multiple stochastic samples through the bridge to get an uncertainty estimate, as we demonstrated in section 8.

On **theoretical grounding**: CycleGAN rests on adversarial game theory; DDPM rests on stochastic differential equations and score matching; CFM methods rest on optimal transport theory. The Schrödinger Bridge in particular has a beautiful connection to classical statistical mechanics — it's the most likely diffusion process to connect two distributions, subject to entropy constraints. This theoretical solidity gives you much more confidence in the model's behaviour outside the training distribution.

---

## §10 · Further Reading and Extensions

*[POINT TO NOTEBOOK — §10 Markdown table]*

I want to briefly flag the extensions table, because this is where I see the most promising directions for taking this work further.

The most impactful near-term extension is **latent SB-CFM**. Right now we're running the flow directly in pixel space — 256 by 256 images. If you first encode each image with a variational autoencoder or a VQ-VAE into a compact latent representation, you can run the flow in that lower-dimensional space. The synthesis is roughly ten times faster, and the model can focus on semantic content rather than pixel-level texture.

A second important extension is **3D patch-based inference** using MONAI's `SlidingWindowInferer`. The current notebook is deliberately 2D for pedagogical clarity, but in production you'd want to synthesise full 3D volumes. MONAI's sliding window infrastructure handles the patch extraction, network inference, and reconstruction with overlap blending automatically.

Third, for any real clinical application, you'd replace our synthetic dataset with real paired data — the IXI dataset for T1/T2, or the SynthRAD 2023 challenge data for MRI-to-CT. The code structure is identical; you just swap out the dataset class.

---

## §11 · Reflection Questions

*[POINT TO NOTEBOOK — §11 Markdown cell]*

The final section of the notebook is a set of twelve reflection questions, organised into four tiers.

The first four questions are conceptual and theoretical. They include things like: why do we need multiple ODE steps if the target velocity is constant? What happens to the SB bridge as sigma-min approaches zero or infinity? Why does it matter that we use the paired data coupling rather than the independent coupling?

The next three questions are implementation-focused: how does AMP interact with the gradient stability of the velocity regression loss? What breaks if you don't scale the sinusoidal time embedding? How would you extend this to 3D with a single GPU?

Questions eight and nine deal with evaluation and clinical validity — specifically, how to design masked metrics that focus on clinically relevant regions, and how to calibrate ensemble uncertainty estimates.

The final three questions are research-level and open-ended. Question ten formalises the hallucination problem as a distribution shift and asks how you would detect hallucinated pathology without access to ground truth. Question eleven covers the regulatory and ethical dimensions — FDA clearance, HIPAA, and whether synthesised images count as de-identified data. And question twelve asks you to identify a domain outside radiology where this pipeline would be applicable — which I think is a useful exercise for anyone in this group.

*[PAUSE — open floor for questions on the reflection section]*

---

## Closing

Let me close with a few thoughts on where this fits in the broader landscape.

Generative modelling for medical imaging is moving very fast. The field has essentially gone through three generations in five years: first CycleGAN-style adversarial methods, then diffusion-based methods, and now flow matching. Each generation has been faster to train, faster to sample from, more theoretically grounded, and more stable. The Schrödinger Bridge represents the current frontier of that progression.

What I find most compelling about this particular approach for our purposes is not the PSNR numbers — it's the *controllability*. Because we're working with paired data and a deterministic conditional transport, we have a very direct handle on what the model learns. We're not asking a neural network to hallucinate a plausible T2 image from scratch — we're asking it to learn the specific physical relationship between T1 and T2 contrast as it exists in our training population. That is a much more constrained problem, and the constraint is what makes it safe to use for augmentation.

The notebook is available for everyone to use and extend. If you run it and find issues, or if you adapt it for a different modality pair or a different application domain, I'd love to hear about it.

Thank you. I'm happy to take questions on any section.

---

## Q&A Prompt Guide

*[Keep these handy during the question period]*

- **"Why not just use CycleGAN? It doesn't need paired data."** → Unpaired methods produce lower-quality results and have no principled way to preserve lesion structure. For augmentation, a hallucinated lesion in the wrong location is worse than no augmentation at all.

- **"How many real paired scans do you need to train this well?"** → In our experience with real datasets, 100–200 paired volumes (each providing 60–80 slices) is enough to get clinically reasonable results. Below 50 volumes, performance degrades sharply.

- **"Can this work for MRI to CT, not just T1 to T2?"** → Yes, the framework is modality-agnostic. MRI-to-CT is harder because the modalities differ more drastically — bone appears as black on MRI and bright on CT — but the SB-CFM approach handles it via the same loss. The SynthRAD 2023 dataset is the benchmark for that task.

- **"Why 2D slices and not 3D patches?"** → Memory and pedagogical clarity. A 3D patch of 64-by-64-by-64 at batch size 2 already fills a T4. The 2D approach is a practical choice for a tutorial, and many published methods use it in production.

- **"What's the difference between sigma-min equals zero and sigma-min equals 0.01?"** → Zero gives you pure OT-CFM with deterministic straight-line paths. A small positive sigma adds a small amount of Gaussian noise at the midpoint of the bridge, which acts as a regulariser and improves generalisation very slightly. The difference in practice is small, but the theoretical properties of the stochastic bridge are cleaner.

---

*End of speech document.*
