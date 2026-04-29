Watched [until 4.1 ](https://www.youtube.com/playlist?list=PLRqNpJmSRfar_z87-oa5W421_HP1ScB25)

# Lec1 Introduction

In the world of computer vision and geometry, a **homography** is a transformation (a 2D matrix) that maps points in one plane to points in another plane.

It is most commonly used to describe the relationship between two different images of the same flat surface viewed from different angles.

---

## 1. How It Works

Mathematically, a homography is represented by a $3 \times 3$ matrix, often denoted as $H$. If you have a point $(x, y)$ in the first image and its corresponding point $(x', y')$ in the second image, the homography relates wthem as:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Because it uses **homogeneous coordinates**, the matrix has 8 degrees of freedom (even though there are 9 elements, the scale is arbitrary). This means you need at least **four pairs of matching points** between two images to calculate the homography matrix.

---

## 2. Key Characteristics

- **Planar Requirement:** Homographies only work perfectly if the scene is **flat** (like a wall, a floor, or a piece of paper) OR if the camera is only **rotating** around its center without moving its position.
- **Straight Lines Stay Straight:** One of the defining features of a homography is that it preserves collinearity. If three points form a line in the first image, they will still form a line in the second image.
- **Non-Parallelism:** Unlike simpler transformations (like shifting or rotating), a homography can account for perspective. Parallel lines in one image might "converge" toward a vanishing point in the other.

---

## 3. Real-World Applications

### Image Stitching (Panoramas)

When you take a panorama on your phone, the software finds matching points between overlapping photos, calculates the homography to align them, and warps the images so they "stitch" together seamlessly.

### Perspective Correction (Document Scanning)

If you take a photo of a whiteboard or a document at an angle, a homography can be used to "rectify" the image, making it look like you were looking at it perfectly top-down.

### Augmented Reality (AR)

AR apps use homographies to track flat surfaces (like a table or a book cover). Once the homography is known, the app can "paste" a digital 3D object onto that surface so it looks like it belongs in the physical space.

### Camera Calibration and Navigation

Robots and self-driving cars use homographies to map the ground plane, helping them estimate distances to obstacles or identify lane markings on the road.

---

# Lec2 Filtering

![alt text](image.png)

#### Q: 2D kernel can be written as 1D convolve 1D. Then what about 3D or higher kernel?

**kernel separability**.

Just as a separable 2D kernel can be expressed as the outer product of two 1D vectors, a strictly separable 3D kernel (or $N$-dimensional kernel) can be expressed as the outer product of three (or $N$) 1D vectors.

### 1. The Math of 3D Separability

$$V * K = V * (v_z \otimes v_y \otimes v_x) = (((V * v_z) * v_y) * v_x)$$

### 2. The Computational Advantage

Let's assume a cubic kernel of size $k \times k \times k$.

- **Standard 3D Convolution:** To compute a single output voxel, you must perform $k^3$ multiplications.
- **Separable 3D Convolution:** To compute a single output voxel, you perform $k$ multiplications for each of the 3 axes, totaling $3k$ multiplications.

### 3. The Catch: Rank-1 Tensors

The critical caveat is that **not all multidimensional kernels are separable**.

For a kernel to be strictly separable into a sequence of 1D convolutions, it must be a **Rank-1 Tensor**.

- **In 2D:** We represent the kernel as a matrix. We can use Singular Value Decomposition (SVD) to check its rank. If the matrix is Rank-1, it is perfectly separable. If it is higher rank, we can sometimes use SVD to approximate it as a sum of a few separable filters.
- **In 3D and higher:** The kernel is a multi-way array (a tensor). To determine if it is separable, we look to **Tensor Decomposition** (specifically CP decomposition, also known as CANDECOMP/PARAFAC). A 3D kernel is separable only if its CP rank is 1.

### 4. Common Separable Kernels in Higher Dimensions

- **The Gaussian Kernel:** $e^{-(x^2+y^2+z^2)} = e^{-x^2} \cdot e^{-y^2} \cdot e^{-z^2}$
- **Box Filters / Average Pooling:** A 3D average pool is just three 1D average pools.
- **Sobel Operators (Approximations):** Edge detection filters in 3D can often be constructed by combining a 1D derivative filter along one axis and 1D smoothing filters along the others.

#### 2D kernel Utilities: Smooth, Sharpening

Q: How would you go about detecting edges in an image (i.e., discontinuities in a function)?
✓ You take derivatives: derivatives are large at discontinuities.
Q: How do you differentiate a discrete image (or any other discrete signal)?
✓ You use finite differences.(definition of a derivative using forward/central... difference)

#### Sobel filter

![alt text](image-1.png)
![alt text](image-3.png)
![alt text](image-4.png)

#### Differentiation is very sensitive to noise

#### When using derivative filters, it is critical to blur first!

### Zero crossings are more accurate at localizing edges (but not very convenient)

Zero crossings mean the second derivative

### Why it is "Not Very Convenient"

1. **The Double-Noise Problem:** We already saw that the first derivative massively amplifies high-frequency noise. The second derivative amplifies it _even worse_. If you don't heavily smooth the image first (usually using a filter called the Laplacian of Gaussian, or LoG), the second derivative will cross zero thousands of times just from random static.
2. **Phantom Edges:** A second derivative will cross zero anytime the signal stops accelerating and starts decelerating, even if it's just a tiny, insignificant ripple in the image intensity. To use zero crossings effectively, you have to add extra, inconvenient logical steps to your code, such as checking if the slope at the zero crossing is steep enough to be considered a "real" edge, rather than just random variance.

In modern computer vision, first-derivative methods (like the Canny edge detector, which looks for peaks) are generally preferred because they are more robust to noise, even if zero-crossings are theoretically more mathematically precise.

Quick mnemonic
Convolution: flips one signal → commutative + associative
Correlation: no flip → neither commutative nor associative

![alt text](image-2.png)![alt text](image-5.png)
Question 1: How much smoothing is needed to avoid aliasing?
Question 2: How many samples are needed to avoid aliasing?
Answer to both: Enough to reach the Nyquist limit. (We’ll see what this means soon.)

### The Laplacian Pyramid (The Detail Extractor)

A Laplacian pyramid stores the edges and textures that are removed at each step of the Gaussian pyramid. It acts as a series of **band-pass filters**, capturing features of a specific size at each level.

To construct the Laplacian level $L_i$, you calculate the difference between the Gaussian image at that level ($G_i$) and the Gaussian image at the _next_ level ($G_{i+1}$).

Because $G_{i+1}$ is smaller, you first have to scale it back up to the size of $G_i$ (often using interpolation).

The mathematical relationship is:
$$L_i = G_i - \text{upsample}(G_{i+1})$$

- **What it looks like:** If you look at a level of a Laplacian pyramid, it looks mostly gray (values near zero), with bright and dark outlines around the edges and textures of the image.

Fourier Transform (FT) 、

### 1. Amplitude: "How Much?" (Strength/Energy)

The amplitude tells you the **strength, size, or volume** of a specific frequency component within the overall signal.

- **In Audio:** If you take the FT of a musical chord, the amplitude of the 440 Hz frequency tells you exactly how loudly the "A" note is being played compared to the other notes.
- **In Image Processing:** Amplitude represents the contrast of a specific pattern. A high amplitude for a low-frequency component means there are strong, broad gradients (like a smooth sky), while high amplitude for high frequencies means sharp, stark edges.

### 2. Phase: "When?" (Timing/Alignment)

The phase tells you the **starting position or time delay** of that specific frequency component. It dictates how the different waves align with each other in time or space.

- **In Audio:** Phase determines exactly when the peak of the sound wave hits your ear. While human ears aren't incredibly sensitive to absolute phase, the _relative_ phase between two speakers is what dictates whether the sound waves constructively combine or destructively cancel each other out (noise-canceling headphones rely entirely on phase shifting).
- **In Structural Biology (Crystallography/Cryo-EM):** This is where phase becomes critical. When X-rays scatter off a protein crystal, the **amplitude** tells you how strongly a certain spatial frequency scattered, but the **phase** tells you exactly _where_ the electron density (and thus the atoms) is located in 3D space. If you lose the phase information (the famous "phase problem"), you cannot reconstruct the protein's structure, even if you have perfect amplitudes.
  ![alt text](image-6.png)

## Hough Transform

![alt text](image-7.png)

### 1. The Ballot Box: `Initialize accumulator H`

The "accumulator" ($H$) is the digital ballot box. In code, this is a 2D matrix or grid (represented by the drawing on the right).

- One axis represents all possible angles ($\theta$), typically from $0^\circ$ to $180^\circ$.
- The other axis represents all possible distances from the origin ($\rho$, labeled as $d$ in the diagram).
- At the very start, the grid is completely empty. Every cell has a value of 0.

### 2. The Voting Process (The nested loops)

This is the core engine of the algorithm.

- **`For each edge point (x,y)`:** You don't run this on every single pixel in the image. You first run an edge detector (like the derivative filters we discussed earlier) and only look at the pixels that are bright (meaning they are part of an edge).
- **`For \theta = 0 to 180`:** For every single edge pixel, you pretend a line is passing through it at $0^\circ$, then $1^\circ$, then $2^\circ$, all the way to $180^\circ$.
- **`\rho = x \cos \theta + y \sin \theta`:** For each of those angles, you calculate exactly how far that specific line is from the origin ($\rho$).
- **`H(\theta, \rho) = H(\theta, \rho) + 1`:** Now that you have a specific angle ($\theta$) and a specific distance ($\rho$), you go to that exact cell in your accumulator grid and add 1 to its score. You are casting a vote for that line.

### 3. Tallying the Votes: `Find local maximum`

Once every edge pixel has cast its votes, you look at your accumulator grid. Most cells will have 0, 1, or 2 votes from random noise. However, if there is a real straight line in the image, all the pixels on that line will have voted for the exact same ($\theta$, $\rho$) combination.

By finding the cells with the highest numbers (the "local maxima"), you find the lines that actually exist in the image.

### 4. Drawing the Line

Once you have the winning ($\theta$, $\rho$) pairs from step 3, you plug them back into the standard equation to draw the continuous line across your image screen.
![alt text](image-8.png)
![alt text](image-9.png)
![alt text](image-10.png)
![alt text](image-11.png)
![alt text](image-12.png)
![alt text](image-13.png)
![alt text](image-14.png)

## Hough Transform Rating

### 1. Deals with occlusion well? (Yes 👍)

**Why:** The Hough Transform does not require a line to be continuous. Because every individual pixel casts its own independent vote for a line, it doesn't matter if there is a gap in the middle of the line or if an object is blocking part of it (occlusion).

As long as the visible fragments of the line lie on the same infinite trajectory, all their pixels will still cast votes for the exact same $(\theta, \rho)$ bin in the accumulator space. If enough visible pixels vote, the peak will still be found, and the algorithm will mathematically "fill in" the occluded gaps.

### 2. Detects multiple instances? (Yes 👍)

**Why:** It can find as many lines as you want in a single pass.
Finding lines simply means finding the brightest points (local maxima) in the Hough Space. If there is a grid of ten intersecting lines in your image, there will simply be ten distinct, bright peaks in the accumulator array. You don't have to run the algorithm multiple times or tell it how many lines to look for in advance; you just extract all the peaks that cross a certain threshold.

### 3. Robust to noise? (Yes 👍)

**Why:** Random noise pixels are uncoordinated.
If an image has random static or scattered noise pixels, those pixels will cast votes for random, scattered lines. Their sine waves will smear across the Hough space without consistently intersecting at any single point.

Noise essentially raises the "background fuzz" of the accumulator array, but it rarely creates a concentrated peak. The true geometric lines, which have dozens or hundreds of perfectly aligned pixels, will easily out-vote the random noise and create peaks that rise clearly above the static.

### 4. Good computational complexity? (No 👎)

**Why:** It is extremely computationally expensive and memory-heavy.
As shown in the implementation slide earlier, there is a nested loop: for _every single edge pixel_, the computer must calculate the math for _every possible angle_.

- **Time Complexity:** If you have a high-resolution image with 100,000 edge pixels, and you check 180 different angles, that is 18,000,000 trig calculations just to find simple 2D lines.
- **Space/Memory Complexity (The Curse of Dimensionality):** Finding lines requires a 2D accumulator array $(\theta, \rho)$. If you want to find circles, a circle is defined by 3 parameters $(x, y, radius)$, which requires a massive 3D accumulator array. If you want to find ellipses, you need a 5D array. The memory and computation time explode exponentially as the shapes get more complex, making the standard Hough Transform impractically slow for higher-level 3D structural detection.

## Assignment 1

### Hough Transform Line Parametrization:

Q: Why do we parametrize the line in terms (ρ, θ) instead of the slope and intercept(m, c)? Express the slope and intercept in terms of (ρ, θ).

The fundamental reason we abandon the slope-intercept form ($y = mx + c$) for the Hough Transform is the infinite parameter space problem caused by vertical lines.
The flaw with $(m, c)$: For a completely vertical line, the slope $m$ approaches infinity ($m \to \infty$), and the y-intercept $c$ also becomes undefined or infinite.

The accumulator resolution needs to be selected carefully. If the resolution is set too
low, the estimated line parameters might be inaccurate. If resolution is too high, run
time will increase and votes for one line might get split into multiple cells in the array.

# 05_corners_slides

![alt text](image-15.png)
![alt text](image-16.png)
![alt text](image-17.png)
![alt text](image-18.png)
![alt text](image-19.png)
By computing the gradient covariance matrix above, we are fitting a quadratic to the gradients over a small image region
![alt text](image-20.png)
![alt text](image-21.png)
![alt text](image-22.png)
![alt text](image-23.png)
A true corner exists only when both $\lambda_1$ and $\lambda_2$ are large.

### 1. Harris & Stephens (1988)

$$R = \det(M) - \kappa \text{trace}^2(M)$$
_(Or in eigenvalues: $R = \lambda_1 \lambda_2 - \kappa (\lambda_1 + \lambda_2)^2$)_

### 2. Kanade & Tomasi / Shi-Tomasi (1994)

$$R = \min(\lambda_1, \lambda_2)$$

### 3. Nobel / Brown et al. (1998)

$$R = \frac{\det(M)}{\text{trace}(M) + \epsilon}$$
_(Or in eigenvalues: $R = \frac{\lambda_1 \lambda_2}{\lambda_1 + \lambda_2 + \epsilon}$)_

### Summary Comparison

| Feature                          | Harris (1988)              | Kanade & Tomasi (1994)        | Nobel (1998)             |
| :------------------------------- | :------------------------- | :---------------------------- | :----------------------- |
| **Requires $\kappa$ parameter?** | Yes                        | No                            | No                       |
| **Requires Square Roots?**       | No                         | Yes                           | No                       |
| **Primary calculation**          | Subtraction                | Minimum                       | Division                 |
| **Historical Significance**      | The foundational baseline. | Better accuracy for tracking. | Fast and parameter-free. |

![alt text](image-24.png)
![alt text](image-25.png)
Harris corner response is invariant to rotation
Partial invariance to affine intensity change
Only derivatives are used => invariance to intensity
shift I → I + b
Intensity scale: I → a I

**Why it is NOT invariant (and what the graphs show):**
If you multiply the image by $a$, the derivatives also get multiplied by $a$: $\frac{d}{dx}(aI) = a\frac{dI}{dx}$.
Because the Harris matrix $M$ is built using the _squares_ of these derivatives ($I_x^2$, $I_y^2$, etc.), the matrix $M$ scales by $a^2$. Consequently, the corner response score $R$ scales by a factor of $a^4$.

**The Conclusion:**
Harris is invariant to _shifts_ in lighting, but if the _contrast_ changes significantly, you will either gain fake corners (if contrast increases) or lose real corners (if contrast decreases) unless you dynamically adjust your threshold.

### How can we make a feature detector scale-invariant?

### How can we automatically select the scale?

Multi-scale blob detection
![alt text](image-26.png)

## 06_features_slides

![alt text](image-27.png)
![alt text](image-28.png)
![alt text](image-29.png)

SIFT(Scale Invariant Feature Transform)
SIFT describes both a detector and descriptor

1. Multi-scale extrema detection
2. Keypoint localization
3. Orientation assignment
4. Keypoint descriptor

![alt text](image-30.png)
![alt text](image-31.png)
