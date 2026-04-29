# defocus 是什么
**defocus** 直译是**散焦、失焦**，在不同领域里意思略有差别，但核心都是：**让焦点不清晰、故意模糊**。

---

## 1. 摄影 / 摄像 / 镜头里的 defocus
- 意思：**失焦、背景虚化**
- 画面里主体清晰，背景模糊朦胧的效果，就是 defocus
- 常见说法：**defocus blur（散焦模糊）、out of focus**

### 作用
- 突出主体，弱化杂乱背景
- 营造氛围感、柔和感、电影感
- 人像、静物拍摄常用，让画面更高级

---

## 2. 计算机视觉 / 图像处理
- 算法里的 **defocus** 一般指**模拟散焦模糊**，或**检测/去除散焦模糊**
### 作用
- 做背景虚化特效
- 修复拍糊的照片（去模糊）
- 用散焦信息估算景深、3D 信息

---

## 3. 光学 / 显微镜 / 激光
- 指**故意偏离焦点**，让光束散开
### 作用
- 避免激光能量过于集中，损伤样品
- 获得均匀的照明光斑
- 控制光斑大小和能量分布

---

## 4. 游戏 / 渲染
- 比如 **depth of field (DOF) / defocus** 效果
### 作用
- 模拟真实相机的虚化，提升画面质感
- 让远景模糊，聚焦在角色或关键物体上

---

### 简单总结
**defocus = 散焦 / 失焦 / 虚化**
作用主要就是：**突出主体、美化画面、控制光线、营造氛围**。

如果你是在某个软件（PR、AE、Blender、相机菜单）里看到这个词，我可以按具体场景帮你解释用法。


mechanism of the Contrast Transfer Function (CTF), **interference pattern** created by electron waves.

Here is the step-by-step mechanism of how the CTF emerges and why it distorts the image.

### 1. The Phase Shift (The Invisible Signal)
Electrons travel as waves. When the electron beam passes through the thin layer of ice and strikes a protein, the electrons do not stop or bounce back. Instead, the electrostatic potential of the protein's atoms slightly slows the electrons down, causing a **phase shift** in the wave.

If we had a "perfect" microscope focused perfectly on the sample, we would capture this exact wave. However, current detectors only measure **amplitude** (intensity of the electrons hitting the sensor), not phase. A perfectly focused image of a pure phase object actually looks completely blank. 

### 2. Defocusing to Create Contrast
To see the protein, we are forced to "defocus" the microscope—deliberately taking a blurry picture. 

By pushing the focal plane slightly below the sample, we force the scattered electron waves (which hit the protein) to travel a slightly different distance than the unscattered waves (which passed only through empty ice). When these waves meet at the detector, they **interfere** with one another.

### 3. Constructive and Destructive Interference (The CTF)
Because the waves are now slightly out of sync due to the defocus, their interference creates the contrast we see. However, this interference is entirely dependent on the **spatial frequency** (the size of the details you are trying to resolve):

* **Constructive Interference (Peaks/Valleys of CTF):** For certain details, the waves align perfectly. This boosts the signal, making those details highly visible. (The CTF equals $+1$ or $-1$).
* **Destructive Interference (Zero Crossings):** For other details, the waves arrive exactly out of phase and cancel each other out. The microscope is entirely blind to details at these specific sizes. (The CTF equals $0$).

This rhythmic oscillation between capturing data and losing data is the Contrast Transfer Function. 

### 4. The Mathematical Mechanism
The physical interference pattern can be perfectly described by a sine wave formula:

$CTF(k) = -\sin(\pi \lambda k^2 \Delta f - \frac{1}{2} \pi C_s \lambda^3 k^4)$

* **$k$ (Spatial Frequency):** Moving left to right on the curve, going from large blurry shapes to tiny high-resolution atomic details.
* **$\Delta f$ (Defocus):** The deeper your defocus, the faster the waves go in and out of sync. High defocus gives great contrast for large shapes but scrambles high-resolution details with rapid oscillations.
* **$\lambda$ (Wavelength):** Dictated by the voltage of the microscope (e.g., 300kV). 
* **$C_s$ (Spherical Aberration):** An unavoidable physical flaw in the electromagnetic lenses of the microscope.

### Explore the Mechanism
Use the simulator below to adjust the microscope's settings. Notice how increasing the **Defocus** creates a stronger signal early on, but causes the high-resolution information to oscillate wildly, creating "zero crossings" where data is entirely lost.

---

### 1. The Center vs. The Edges (Level of Detail)
The image is organized by how "fine" or detailed the information is.
* **The Dead Center (Low Frequency):** Represents the largest, grossest shapes in the image (like the overall boundary of a protein).
* **Moving Outward (High Frequency):** As you move radially away from the center toward the edges, you are looking at increasingly tiny, fine details (like internal helices, and eventually, individual atoms).

### 2. The Rings (Thon Rings)
The alternating concentric circles (called Thon rings) represent the oscillating nature of the electron wave interference.
* **Bright White / Solid Black Rings:** These are areas of **constructive interference**. If a detail in your protein happens to be the exact size that falls on one of these rings, the microscope captures it perfectly with maximum contrast. 
* **The Grey Zones (Zero Crossings):** The dull grey areas between the stark white and black rings are areas of **destructive interference**. If a detail in your protein falls into this frequency zone, the waves cancel out, and the microscope is completely blind to it. The information is physically erased from the raw image.

### 3. Radial Symmetry
In a well-aligned microscope, the CTF image is perfectly circular. This means the distortion is happening equally in all directions (up, down, left, right). 
* **Astigmatism:** If the microscope lenses are slightly warped, the rings won't be perfect circles—they will stretch into ovals or ellipses. Correcting this astigmatism is a major part of the AI and algorithmic preprocessing pipeline.

### What about a "Perfect" Original Picture?
If you had a mathematically perfect optical system with no defocus and no lens aberrations (which is physically impossible), the CTF image would not have rings at all. 

It would just be a solid, uniform plane where $CTF = 1$ everywhere. This would mean that every single detail, from the massive macro-structures to the tiniest atomic distances, is transferred to the sensor perfectly without any phase flipping or information loss.

---
---
## 10. Key Takeaways & Next Steps

| Concept | Key Insight |
|---|---|
| **Forward model** | CTF + Poisson + Gaussian noise |
| **CTF correction** | Phase-flip first; Wiener CTF for amplitude |
| **Classical filters** | Bandpass & Wiener best for mild denoising |
| **Class averaging** | SNR scales as √N — the gold standard |
| **FRC** | 0.143 half-bit criterion for resolution |
| **Self-supervised** | Noise2Void / cryoCARE — no clean targets needed |

---
$$
\text{average} = \frac{1}{N}\sum_{i=1}^{N}(\text{signal}+\text{noise}_i)
= \text{signal} + \frac{1}{N}\sum_{i}\text{noise}_i
$$

This is why modern cryoEM datasets collect millions of particles 

### FRC — "0.143 half-bit criterion for resolution"

This answers the question: *how do you know when your class average is actually showing real structure vs just correlated noise?*

The procedure is to split your particles into two independent halves, compute a class average from each half separately, then measure how similar those two averages are frequency-by-frequency:

$$\text{FRC}(r) = \frac{\sum_{\mathbf{k} \in r} F_1(\mathbf{k}) \cdot F_2^*(\mathbf{k})}{\sqrt{\sum|F_1|^2 \cdot \sum|F_2|^2}}$$

This gives a correlation value between 0 and 1 at each spatial frequency ring. The logic is:

- If both halves agree at a given frequency (FRC ≈ 1) → that frequency contains **reproducible signal**
- If both halves disagree (FRC ≈ 0) → that frequency is **dominated by noise**, which is different in each half

The **0.143 threshold** comes from information theory — it's the point where the two half-maps together contain exactly 1 bit of information per Fourier coefficient (the "half-bit" criterion). Below this threshold you are essentially fitting noise. It's conservative by design — better to underreport resolution than to claim structure that isn't there.

The critical insight is that **FRC measures reproducibility, not visual quality**. A blurred, over-smoothed map might look beautiful but fail FRC at low resolution because the smoothing is hiding the fact that the two halves never agreed at high frequencies. This is why FRC is the standard and not a subjective sharpness score.
---

### Connection to Robotics Perception
The low-SNR challenge here maps directly onto robotic sensing:
- Depth sensor noise profiles mirror cryoEM shot noise
- SDF-based collision guidance must handle uncertain, noisy point clouds
- FRC generalises to evaluating LiDAR/stereo depth reconstruction quality
