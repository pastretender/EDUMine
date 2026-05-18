# 🔬 Presentation Speech
## Cell Tracking & Bimorphological Analysis
### Instance Segmentation + Long-Term Sequence Tracking on Dynamic Cell Microscopy Images

---

> **Delivery guide:** ~25–35 minutes total. Each section is marked with an estimated
> duration. Pause cues are marked **[PAUSE]**. Slide/screen cues are marked
> **[SHOW: ...]**. Phrases in *italics* are optional elaborations to add if time allows.

---

## Opening (2–3 min)

Good morning, everyone. Today I want to walk you through a complete computational
pipeline that I've put together as a Jupyter notebook — one that takes us all the
way from raw microscopy images to quantitative biological insight, without writing a
single line of manual annotation.

The title is **Cell Tracking and Bimorphological Analysis**, and what that means in
practice is this: given a time-lapse video of cells under a microscope, we want to
automatically answer three questions that experimentalists care about deeply.

First — *where* is each individual cell, separated from every other cell, in every
frame? That's the segmentation problem.

Second — *which* cell in frame five is the same cell we saw in frame four, and
frame three, all the way back to frame one? That's the tracking problem.

And third — *how is each cell changing over time* — its shape, its size, its
brightness, its migration pattern? That's the morphological analysis problem.

These three problems are deeply coupled, and solving them together is what allows
us to do genuinely longitudinal single-cell biology from imaging data.

The entire notebook runs on **Google Colab with a free T4 GPU**, so everything
I show you today is reproducible by anyone in this room in about fifteen minutes.
Let me take you through it section by section.

**[PAUSE]**

---

## Section 1 — Installation & Environment Setup (2 min)

**[SHOW: Section 1 code cells]**

The first section handles the plumbing. We pin specific versions of every major
library — **Cellpose 3.0.11** for segmentation, **trackpy 0.6.4** for particle
tracking utilities, **scikit-image 0.24** for morphological measurement, and
**OpenCV** for image preprocessing, among others. Version pinning matters here
because Cellpose in particular has had breaking API changes between major releases,
and the last thing you want on a shared Colab session is a silent incompatibility.

After installation, we immediately verify the GPU. The notebook checks for a CUDA
device and prints both the GPU name and the available VRAM. On a T4 you typically
see 15 gigabytes. If no GPU is found, the notebook degrades gracefully and runs on
CPU — segmentation will be slower, but everything else is unaffected.

One small detail worth noting: we set `numpy.random.seed(42)` early. This makes
the synthetic data generation and all random operations fully reproducible across
runs, which is important when you want to compare results after changing a
downstream parameter.

---

## Section 2 — Synthetic Cell Image Sequence Generation (4–5 min)

**[SHOW: Section 2 — the SimulatedCell class and rendered frames]**

Now, rather than immediately jumping to a real dataset — which would require
acquisition time and data-sharing agreements — we build a **synthetic
fluorescence microscopy time-lapse** from scratch. This is one of the most
underrated practices in computational biology, and I want to spend a moment
explaining why we bother.

When you have synthetic data, you have **ground truth**. You know exactly where
every cell was at every frame, its exact radius, its parent cell if it divided,
and its true intensity before noise was added. That ground truth lets you
rigorously evaluate every downstream step — segmentation accuracy, tracking
identity preservation, feature measurement error — in a way that would require
months of manual annotation on real data.

So what does our simulator produce? We model each cell as a **software agent**
— a Python object called `SimulatedCell`. Each agent carries a position, a
radius, an intensity, and a preferred migration direction. At every frame, the
agent advances its position by a combination of two motion components: a small
**directed drift** — think chemotaxis or haptotaxis — and a **Brownian noise
term** representing random cytoskeletal fluctuations. The parameters
`MOTION_DRIFT = 0.4` and `MOTION_BROWN = 3.0` pixels per frame give us a
realistic mixture of directed and diffusive behaviour.

We also model **cell division**. At each frame, every cell has a 1.5% chance
of dividing. When division occurs, a daughter cell is spawned nearby with
75% of the parent's radius and slightly reduced intensity — mimicking the
well-known size reduction and GFP dilution that accompany mitosis.

The rendering pipeline converts these agent states into a realistic image. Each
cell is drawn as a **Gaussian-shaped intensity profile** — approximating the
point-spread function of a real fluorescence microscope. We then add a
directional background gradient, apply Gaussian blurring to simulate optical
diffraction, add **Poisson shot noise** — which is the dominant noise source in
fluorescence imaging because photon arrival is a counting process — and finally
add Gaussian readout noise from the camera electronics.

One more subtle detail: **photobleaching**. Each cell's intensity decays by 0.7%
per frame, modelling the irreversible photodestruction of fluorophores under
continuous illumination. You'll see this show up clearly later when we fit an
exponential decay curve to the population intensity.

**[SHOW: the five-frame preview plot]**

The result is a 512 by 512 pixel, 40-frame sequence starting with 20 cells and
growing as divisions accumulate. *The inferno colourmap here is purely aesthetic
— we're mapping intensity to colour just for visualisation.*

---

## Section 3 — Instance Segmentation with Cellpose (5 min)

**[SHOW: Section 3 — model loading and segmentation overlay]**

Now we arrive at the core algorithmic step: **instance segmentation**. Let me be
precise about what we mean. Semantic segmentation says "this pixel belongs to a
cell, that pixel belongs to background." Instance segmentation goes further — it
says "this pixel belongs to *cell number seven*, and that pixel belongs to
*cell number twelve*." The distinction is critical for tracking, because we need
unique identities, not just a foreground mask.

We use **Cellpose**, which is currently the most widely adopted general-purpose
cell segmentation tool in the field, published in Nature Methods by Stringer and
colleagues in 2021. Cellpose's key innovation is a **flow-based representation**.
Rather than predicting a binary mask directly, the network is trained to predict
a 2D spatial flow field — think of it as every pixel pointing towards the centre
of the cell it belongs to. Those flow vectors are then numerically integrated to
cluster pixels into individual cells. This approach is far more robust to touching
or overlapping cells than methods based on watershed or connected-component
labelling alone.

We load the `cyto3` model, which is the third-generation cytoplasm model trained
on over a million annotated cells spanning dozens of cell types and imaging
modalities.

**[SHOW: the flow visualisation panel — four images side by side]**

Look at this four-panel figure from a single frame. On the far left is the raw
image. The second panel shows the **spatial flows rendered in HSV colour** — the
hue encodes direction and the saturation encodes magnitude. You can see that
within each cell, all flows point inward toward the cell's centre, while
background pixels have near-zero flow. The third panel shows the **cell
probability map** — the logit-transformed likelihood that each pixel belongs to
any cell. And the rightmost panel shows the final instance masks after integrating
those flows.

Before passing each frame to Cellpose, we apply **CLAHE** — Contrast-Limited
Adaptive Histogram Equalization — from OpenCV. This locally boosts contrast in
regions where the signal is low, which helps Cellpose detect cells in the
periphery of the field where background haze can wash out weaker signals. The
clip limit of 3.0 prevents over-amplification of noise.

Three key parameters govern the segmentation quality. The `diameter` — set to
36 pixels here, matching our simulated cell size — tells the model the expected
scale of objects. The `flow_threshold` of 0.4 controls how connected the flow
vectors need to be for a region to be accepted as a cell. And the
`cellprob_threshold` of negative 1.5 makes the detector more permissive,
catching dim cells near the detection limit that might otherwise be missed.

---

## Section 4 — Bimorphological Feature Extraction (4 min)

**[SHOW: Section 4 — the feature table header, the feature category table in the markdown]**

With instance masks in hand for every frame, we now extract a **quantitative
fingerprint** for each cell at each time point. The term "bimorphological" refers
to the dual nature of the features we extract: geometric shape descriptors on one
hand, and intensity-based measurements on the other.

We use `scikit-image`'s `regionprops` function, which operates on a labelled
integer mask alongside the corresponding intensity image. For each segmented
region it computes a comprehensive set of measurements.

Let me walk through the four feature categories. **Shape features** include area
in pixels squared, perimeter, equivalent diameter — the diameter of a circle with
the same area — eccentricity, which ranges from zero for a circle to one for a
line, solidity, which is the ratio of the cell's area to its convex hull area,
and circularity, defined as four pi times area divided by perimeter squared, equal
to exactly one for a perfect circle and less than one for any deviation.

**Orientation features** give us the lengths of the major and minor axes of the
best-fit ellipse, the orientation angle of that ellipse in radians, and the
elongation ratio, which is major axis over minor axis.

**Intensity features** capture the mean pixel intensity inside the mask, the
maximum, the standard deviation — which reflects how heterogeneous the
fluorescence distribution is — and the integrated intensity, which is mean
times area, often used as a proxy for total protein amount.

One engineering note: we compute the convex hull perimeter separately using
`morphology.convex_hull_image` rather than relying solely on regionprops. This
gives us a **convexity** measure — the ratio of convex perimeter to actual
perimeter — that is sensitive to membrane ruffles and protrusions that simpler
descriptors might miss.

The output of this section is a pandas DataFrame with one row per cell per frame
— the raw material for everything that follows.

---

## Section 5 — Long-Term Cell Tracking (5 min)

**[SHOW: Section 5 — cost matrix construction and tracking loop]**

This is arguably the most technically involved section of the notebook, so let me
walk through the design carefully.

The tracking problem is fundamentally an **assignment problem**: given a set of
detected cells in frame t-1 and a set in frame t, find the best bijective mapping
between them. We solve this with the **Hungarian algorithm** — also known as the
Munkres algorithm — which finds the globally optimal minimum-cost assignment
in cubic time.

The cost between a previous detection and a current detection is a weighted sum
of two terms. The first is **normalised centroid distance** — how far apart the
two cells are in pixels, divided by `MAX_DIST = 55`. The second is
**one minus the Intersection-over-Union** of their pixel masks — where IoU
equals zero when the masks don't overlap at all and one when they overlap
perfectly. The final cost is 0.6 times the distance term plus 0.4 times the
IoU term. Assignments where the centroid distance exceeds 55 pixels are
forbidden by setting the cost to one billion, effectively creating a hard
exclusion zone.

Why use both distance and IoU rather than just distance? Because in dense cell
populations, two neighbouring cells can have similar centroid distances to a
given detection, and the mask overlap breaks the tie far more reliably than
distance alone.

**Gap closing** handles the case where a cell is transiently missed by the
segmenter — perhaps because it temporarily moved out of focus, or two cells were
merged in one frame. We allow the tracker to match a current detection to any
active track that was last seen within `MAX_GAP = 3` frames, not just the
immediately preceding frame. This dramatically improves track continuity without
introducing many false identity links.

After tracking, we filter out any track shorter than four frames, since very short
tracks are overwhelmingly spurious detections rather than real cells. The final
result is a `track_df` DataFrame where every row has a persistent, globally unique
`track_id` that follows the same cell across its entire observable lifetime.

**[SHOW: the trajectory map and displacement histogram]**

Here we can see all tracked cell paths overlaid on the final frame, colour-coded
by track identity. The histogram on the right shows the distribution of net
displacement — the straight-line distance from each cell's starting position to
its ending position. The non-zero median confirms that cells are not purely
diffusing; there is a genuine directed component to their motion, exactly as we
built into the simulator.

---

## Section 6 — Morphological Feature Analysis over Time (3–4 min)

**[SHOW: the six-panel morphological dynamics plot]**

Now we connect tracking with morphology. Because each measurement in `track_df`
carries both a track identity and a frame number, we can ask: how does the
population of cells, as a whole, change its shape profile over the course of the
experiment?

For each of six key features — area, circularity, eccentricity, elongation,
solidity, and mean intensity — we plot the **median across all cells at each
frame**, with the interquartile range shaded. This is more informative than the
mean because it's robust to outliers from occasional segmentation errors.

**[SHOW: correlation heatmap]**

The correlation heatmap reveals the intrinsic redundancy structure of
morphological features. Unsurprisingly, area, perimeter, equivalent diameter, and
integrated intensity are all strongly correlated — they all measure something
related to cell size. Similarly, eccentricity and elongation are highly correlated
because both measure how stretched the cell is. This has a direct implication: if
you were planning to use these features as inputs to a classifier or a clustering
algorithm, you would want to apply dimensionality reduction first, because feeding
in all fifteen features is essentially feeding in maybe five to six independent
axes of variation.

**[SHOW: violin plots and the single-cell deep-dive grid]**

The violin plots break the same features down by time quartile, making it easy
to spot distributional shifts — changes in the median, changes in spread, or the
emergence of bimodality — that might not be visible in the time-series plots.

The single-cell deep-dive panel picks the three longest-tracked cells and shows
their individual spatial trajectories alongside time series of their area,
circularity, and intensity. This is where the value of long-term tracking really
becomes clear — you can see the *biography* of a single cell, not just a
population average.

---

## Section 7 — Mean Squared Displacement Analysis (3–4 min)

**[SHOW: the two MSD panels — linear and log-log]**

Section 7 addresses a fundamental question about cell motility: is the migration
we observe random diffusion, directed movement, or something in between?

The standard tool for answering this is the **Mean Squared Displacement**, or
MSD. For a given lag time τ — meaning we look at pairs of positions separated by
τ frames — the MSD is the average squared distance the cell has moved. The
theoretical prediction is that MSD scales as a power law in τ: MSD proportional
to τ to the power α.

If α equals one, the cells are performing **pure Brownian diffusion** — random,
memoryless steps in all directions equally.

If α equals two, the motion is **perfectly directed** — the cell is moving in a
straight line at constant speed.

Values between one and two indicate **super-diffusion**, also called a persistent
random walk — the cell has a tendency to keep moving in its current direction for
some time before changing course. This is the most biologically realistic model
for actively migrating cells.

We compute the MSD for each valid track, average across the ensemble, and fit a
power law by performing a linear regression on the log-log plot. The slope of
that fit is α. We also extract an **effective diffusion coefficient** D, which
characterises the overall magnitude of movement regardless of its directionality.

On the right panel, the log-log plot makes the power law immediately visible as a
straight line. The two dotted reference lines show where a perfectly Brownian
(α = 1) and perfectly directed (α = 2) population would sit, so you can read off
the motility classification at a glance.

*The exponent we recover is consistent with the motion model we put into the
simulator — confirming that the pipeline is measuring what we intended.*

---

## Section 8 — Population Statistics & Cell Count Dynamics (2–3 min)

**[SHOW: the four-panel population dashboard]**

Section 8 brings together the key population-level readouts into a single
dashboard.

The top-left panel compares two things: the number of masks the segmenter
detected in each frame versus the number of tracks we actually maintained. The
gap between these two curves tells us something important — some detections in
every frame are fragmented cells, debris, or merge errors that the tracker
correctly rejects.

The top-right panel shows how cell size evolves over time, split into four time
quartiles. You can see whether the cell population is growing or shrinking on
average — driven in our case by the interplay of division and photobleaching
affecting apparent size.

The bottom-left scatter plot maps every cell-frame observation in eccentricity
versus circularity space, colour-coded by time. A cloud that drifts over time
would indicate systematic morphological change across the population.

And the bottom-right panel fits an exponential decay model to the mean intensity
trajectory — directly recovering the photobleaching rate constant k that we
specified in Section 2. This is a built-in validation: if the recovered k
matches our input parameter, we know the intensity measurement pipeline is
working correctly end to end.

---

## Section 9 — Annotated GIF Export (1–2 min)

**[SHOW: the animated GIF playing]**

Section 9 produces the most immediately interpretable output of the entire
pipeline: an annotated animated GIF where each frame shows the segmentation
overlay, the track ID label for every active track, and a short trailing path
showing where each cell came from in the previous eight frames.

This is not just eye candy. Animated visualisations of this kind are invaluable
for **quality control** — within a few seconds of watching the animation, an
experienced biologist can spot systematic tracking errors, cells that are being
merged by the segmenter, or unusual migration patterns that warrant closer
inspection. I strongly encourage anyone adapting this pipeline to real data to
always inspect the GIF before drawing biological conclusions from the numbers.

---

## Section 10 — Export & Summary (1 min)

**[SHOW: Section 10 output summary table]**

Section 10 packages all results for downstream use. The full `track_df` DataFrame
is saved as a CSV — one row per cell per frame, with all 15 morphological features
and the track ID. The MSD results are saved separately. All figures are saved as
high-resolution PNGs. And a convenience cell zips everything and triggers a
download from Colab directly to your local machine.

The summary block at the end prints the key numbers: total frames analysed, total
segmented instances, number of valid tracks, features extracted, and the MSD
parameters. This is the "abstract" of the computational experiment.

---

## Section 11 — Reflection Questions (2–3 min)

**[SHOW: Section 11 markdown]**

The final section is a set of twenty reflection questions grouped into six
thematic areas. I'll highlight just a few that I think are particularly
generative.

**Question A2** asks about CLAHE and its limits. We apply CLAHE to boost contrast
before segmentation, but CLAHE performs local histogram equalization. If one cell
genuinely expresses twice as much GFP as its neighbour, CLAHE will compress that
biological difference into a narrower local contrast range. So the preprocessing
step that helps detection can simultaneously destroy the quantitative signal
you wanted to measure. The question pushes you to think about when a
preprocessing heuristic is a net positive.

**Question C3** asks how you would detect cell division from the output of this
pipeline. The tracker currently treats daughters as new tracks. But if you look at
the features — a sudden appearance of a new track adjacent to an existing one,
accompanied by a reduction in both cells' areas — you have a detectable signature.
The ground-truth `gt_df` from Section 2 stores the parent IDs, so you could
build and validate a division classifier entirely within this notebook.

**Question E3** asks you to implement the **velocity autocorrelation function** —
a complementary motility descriptor that captures directional persistence directly
in the time domain rather than the frequency domain. This is a natural next step
for anyone wanting to go deeper than MSD analysis.

And **Question F3** poses a drug-screening design problem — how would you use
this pipeline across 96 wells, each with a different compound, to identify
morphological hits? That's genuinely open-ended and maps directly onto how
high-content screening is done in industry.

---

## Closing (1–2 min)

To summarise what we've built: an end-to-end, GPU-accelerated, fully reproducible
pipeline that takes a raw fluorescence time-lapse — real or synthetic — and
produces quantitative single-cell trajectories annotated with fifteen
morphological features, analysed for motility regime, and visualised at both the
population and individual cell level.

The design philosophy throughout has been **modularity and transparency**. Each
section does one thing and does it explicitly, so you can swap out any component
— replace Cellpose with SAM or Stardist, replace the Hungarian tracker with a
graph neural network tracker, replace regionprops with your own texture features —
without touching anything else.

The most immediate path to applying this to real data is the one-liner in the
conclusion section: load a TIFF stack, assign it to `frames_raw`, and re-run from
Section 3 onwards. Everything else adapts automatically.

I'm happy to go deeper on any section — particularly the cost matrix design in
the tracker, the MSD ergodicity discussion, or how one would extend the
morphological feature set for a specific biological application.

Thank you.

---

## Appendix: Quick Reference for Q&A

| Topic | Key point to remember |
|---|---|
| Why Cellpose over classical watershed? | Flow-based representation separates touching cells; watershed tends to merge them |
| Why synthetic data? | Provides exact ground truth for validating every pipeline stage |
| What does α > 1 mean? | Persistent random walk — cell has directional memory between steps |
| Why filter tracks < 4 frames? | Short tracks are overwhelmingly segmentation artefacts |
| CLAHE trade-off | Boosts contrast for detection, but may compress true intensity differences |
| Gap closing | Reconnects tracks broken by missed detections up to 3 frames apart |
| Real data adaptation | Replace `frames_raw` list with any (T, H, W) image stack |
| IOU in cost matrix | Breaks ties between equidistant neighbours in dense populations |
| Photobleaching fit | Validates the intensity measurement pipeline against known ground truth |
| Feature redundancy | Area, perimeter, diameter, and integrated intensity are collinear — reduce before ML |
