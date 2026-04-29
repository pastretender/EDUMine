# Image Super-Resolution (SR) Methods

## Summary Sheet

| Method Category | Representative Models | Key Advantages | Key Disadvantages | Ideal Use Cases |
| :--- | :--- | :--- | :--- | :--- |
| **1. Traditional Methods** | • Bicubic Interpolation<br>• Sparse Representation | • Extremely fast<br>• Requires no GPU/training<br>• Zero memory overhead | • Results are often blurry or jagged<br>• Cannot reconstruct missing high-frequency details | • Quick image previews<br>• Basic zooming<br>• Highly resource-constrained devices |
| **2. CNN-Based Methods** | • SRCNN (Pioneer)<br>• EDSR (ResNet-based)<br>• RCAN (Channel Attention) | • High objective metrics (PSNR/SSIM)<br>• Stable and predictable results<br>• Good balance of speed and performance | • Images can look overly smooth or "soft" to the human eye<br>• Struggles to invent new textures | • Medical imaging<br>• Satellite imagery<br>• Video surveillance where fidelity is critical |
| **3. GAN-Based Methods** | • SRGAN (Pioneer)<br>• ESRGAN<br>• Real-ESRGAN | • Excellent perceptual quality<br>• Generates sharp, highly realistic textures | • Lower PSNR scores<br>• Prone to generating "hallucinated" artifacts or fake details | • Restoring old photographs<br>• Anime/cartoon upscaling<br>• Enhancing highly degraded web images |
| **4. Transformer-Based** | • SwinIR<br>• HAT (Hybrid Attention Transformer) | • Extremely high performance ceiling<br>• Strong global feature modeling (understands context better than CNNs) | • High computational cost<br>• Large memory footprint during inference | • High-end image restoration<br>• Academic benchmarks<br>• Tasks requiring complex spatial understanding |
| **5. Diffusion Models** | • SRDiff<br>• Stable Diffusion Upscaler<br>• ControlNet Tile | • Unmatched ability to generate rich, logical, and hyper-realistic details (e.g., skin pores, fabric) | • Extremely slow inference speed<br>• Requires massive computing power<br>• Less faithful to original ground truth | • Artistic creation<br>• Extreme detail recovery<br>• AI art generation pipelines |


Applying image enhancement and Super-Resolution (SR) techniques to **Cryo-Electron Microscopy (Cryo-EM)** is a fundamentally different beast compared to standard computer vision tasks. 

In standard CV, SR usually deals with simple downsampling, compression artifacts, or Gaussian blur. In Cryo-EM, you are fighting against the limits of physics. If you apply standard models (like Real-ESRGAN or SwinIR) directly to a raw micrograph, they will likely "hallucinate" incorrect biological structures, which is disastrous for downstream tasks like protein orientation estimation or drug discovery.

Here are the major differences and necessary additions when processing Cryo-EM images:

### 1. The Physics Forward Model & CTF Correction
Standard SR assumes a simple degradation model ($LR = HR * blur + noise$). Cryo-EM degradation is dictated by a complex physical forward model. 
* **The Addition:** You must account for the **Contrast Transfer Function (CTF)**. The electron beam creates a phase contrast that modulates the image frequencies, causing phase reversals (some frequencies are inverted, others are lost entirely).
* **The Difference:** Before or during any neural network enhancement, CTF computation and correction are mandatory. Generating synthetic training data for Cryo-EM SR often requires programming a rigorous forward model that simulates particle phantoms, applies the CTF mathematically, and adds realistic detector noise. 

### 2. Extreme Low SNR & The Denoising Prerequisite
To prevent the electron beam from destroying the biological sample (radiation damage), Cryo-EM uses extremely low electron doses. This results in a Signal-to-Noise Ratio (SNR) that is incredibly low—often around 0.1 or worse.
* **The Difference:** Standard CNNs or Transformer SR models will simply amplify this noise. Therefore, **denoising takes precedence over pure super-resolution** in the 2D micrograph stage.
* **The Addition:** Self-supervised denoising algorithms are heavily utilized because they don't require clean targets. Techniques like **Noise2Void** (which predicts a pixel's value based on its surroundings without needing a clean ground truth) or specialized tools like Topaz are integrated into the pipeline to rescue the signal before higher-resolution structural details can be resolved. 

### 3. The Lack of Ground Truth (HR) Data
Most modern SR models (like EDSR or RCAN) rely on supervised learning, requiring perfectly aligned Low-Res and High-Res image pairs.
* **The Difference:** In Cryo-EM, there is no physical way to capture a "clean, high-resolution" 2D micrograph of the exact same sample state to use as ground truth. 
* **The Addition:** Researchers must use alternative training paradigms:
    * **Simulated Data:** Using known 3D protein structures (from the PDB) to generate thousands of 2D projections, adding simulated CTF and noise to train the network.
    * **Unpaired/Self-Supervised Learning:** Using CycleGANs or masking techniques to learn distributions rather than exact pixel-to-pixel mappings.

### 4. 2D Projections vs. 3D Volumes
Standard SR enhances a 2D image to look better. Cryo-EM image processing uses 2D projections to reconstruct a 3D density map.
* **The Difference:** Applying aggressive GAN-based SR to 2D micrographs can introduce subtle biases that ruin the final 3D reconstruction.
* **The Addition:** Much of the "Super-Resolution" in Cryo-EM actually happens *after* the 3D reconstruction. Deep learning models like **DeepEMhancer** or **EMReady** operate on the reconstructed 3D volume (often using ResNet or U-Net architectures) to sharpen the map, recover high-frequency details lost during the reconstruction averaging, and resolve local resolution variations. 

---

### Summary of the Cryo-EM AI Pipeline

| Stage | Standard SR Counterpart | Cryo-EM Specific Method / Addition |
| :--- | :--- | :--- |
| **Degradation Modeling** | Gaussian Blur / Bicubic | **CTF Simulation & Forward Modeling** |
| **Initial Enhancement** | CNN/Transformer Upscaling | **Self-supervised Denoising (e.g., Noise2Void)** |
| **Target Data** | Real HR Images | **Simulated Particle Phantoms / 3D Priors** |
| **Final Resolution Boost** | 2D Image Sharpening | **3D Density Map Sharpening (e.g., DeepEMhancer)** |


制造社交媒体上这些极具视觉冲击力的“赛博萝莉”，其底层技术本质上是一套高度工程化的**实时计算机视觉（CV）与计算机图形学（CG）流水线**。

从信息工程和算法的视角来看，这并非简单的“贴图”，而是对输入图像在空间域、频域以及高维特征空间进行了多层次的解构与重组。我们可以将这个流水线拆解为以下三个核心技术模块：

### 1. 几何拓扑重组：基于关键点的形态学映射 (Morphological Mapping)

生物学上的“幼态持续（Neoteny）”具有极其明确的形态学特征：颅面比例大、中庭（眉心到鼻底）极短、眼距偏宽且眼裂巨大、下颌骨发育不完全（幼态圆脸）。工程上的第一步，就是用数学模型强制逼近这些生物特征。

* **高精度面部特征点检测 (Facial Landmark Detection)：** 系统首先通过轻量级的深度卷积网络（如 MobileNet 等骨干网络）提取人脸的 106 个或 240 个高精度 2D/3D 关键点，建立面部的拓扑图结构。
* **非线性空间形变矩阵 (Non-linear Spatial Warping)：** 获取关键点后，系统会构建一个基于移动最小二乘法 (MLS) 或薄板样条插值 (TPS) 的形变场。算法会强行缩短中庭关键点之间的欧氏距离，放大眼部特征点的凸包面积，并平滑下颌骨边缘的曲率。这相当于在数学上强制修改了面部的骨骼动力学结构，使其完美符合理想化的“幼态比例”。

### 2. 频域滤波与信号重构：纹理的极度提纯 (Texture Denoising & Filtering)

真实的皮肤包含了大量的物理信息：毛孔、细纹、微小的色素沉积等。在图像处理系统中，这些物理细节都属于**高频空间噪声**。为了制造“幼态”那种像陶瓷一样纯净的肌肤质感，系统需要进行暴力的信号分离。

* **边缘保留滤波 (Edge-preserving Filtering)：** 传统的均值滤波会导致整张脸变得模糊（即低频结构和高频细节一起丢失）。现代算法通常采用双边滤波器 (Bilateral Filter) 或导向滤波 (Guided Filter)。它们能够在计算像素邻域权重的过程中，引入颜色梯度的考量。
* **信噪分离与重构：** 算法将图像拆分，剥离掉高频的“皮肤噪声”，只保留低频的“面部光影结构”和边缘边界（如眼睛、嘴唇的清晰轮廓）。这种处理逻辑与处理科学成像（如降低低信噪比图像中的散粒噪声）有异曲同工之妙，目的都是提取纯粹的结构信号，丢弃干扰项。最终呈现出的，就是毫无瑕疵的“磨皮”效果。

### 3. 生成式特征注入：隐空间的语义操控 (Latent Space Manipulation)

如果说前两步只是“修改”现有的脸，那么引入生成式模型（如 StyleGAN 系列或 Diffusion 模型）则是完成了跨越式的“语义重写”。

[Image of Generative Adversarial Network architecture]


* **隐向量空间遍历 (Latent Space Traversal)：** 在生成模型的理解中，每一张脸都可以被编码为一个高维空间中的隐向量 $z$。研究人员可以通过训练，在这个高维空间中找到代表特定语义的“方向向量”，例如代表“年龄减小”、“二次元化”或“无辜感”的法向量。
* **特征融合与解码：** 将原始图像编码到隐空间后，使其沿着“幼态化”的向量方向移动一段距离，然后再解码回像素空间。这时的模型不仅改变了像素的位置，更是**直接生成了原本不存在的幼态特征**（例如强行渲染出更加水润的眼球高光、改变虹膜颜色、自动生成二次元风格的睫毛和腮红投影）。

### 4. 实时 3D 渲染与追踪：动态的数字面具 (3DMM & Tracking)

为了让这些效果在视频流和直播中不穿帮，系统还需要极强的动态稳定性。

* **3D 形变模型 (3D Morphable Models, 3DMM)：** 现代的滤镜已经不再仅仅是在 2D 图像上揉捏，而是会在后台实时拟合一个 3D 的参数化人脸模型。通过解耦身份参数（Identity）、表情参数（Expression）和光照/姿态参数，系统能够确保人物在转头、做大表情时，贴在脸上的“幼态面具”在三维透视关系上依然严丝合缝，不会出现几何崩溃。

综上所述，我们在社交媒体上看到的“萝莉”，很多时候已经不再是一个具体的物理人类，而是一组经过精确计算的形态学参数、被高度提纯的图像信号，以及生成式模型在隐空间中计算出的最优解。它是一个完美迎合算法推荐机制与人类视觉本能的**工程学输出结果**。

---
CTF求解精度会影响高分辨率的结构解析
CTF correction: 1. 最简单, phase flipping(只能restore一半信息)
2. 维纳滤波 (需要准确估计信噪比)
S(Spectrum)SNR

成像时要选择不同的前交量

在冷冻电镜 (Cryo-EM) 的图像获取和处理过程中，**欠焦量 (Defocus，通常用 $\Delta z$ 表示)** 是一个极其核心的物理参数。以下是它的具体含义和作用原理：

### 1. 什么是欠焦量？
在显微镜成像中，生物大分子（如蛋白质）主要由碳、氢、氧、氮等轻元素组成。它们对高能电子束的散射极弱，被称为**弱相位体 (Weak Phase Objects)**。这意味着电子穿过样本时，只会发生相位的改变，而几乎没有振幅（明暗）的变化。

如果我们在冷冻电镜下对样本进行绝对精确的对焦 (Perfect Focus)，图像将几乎是一片空白。为了让样本的轮廓变得可见，操作员必须故意将物镜稍微偏离完美焦平面。这种人为引入的焦距偏移量，就叫做**欠焦量**。

### 2. 欠焦量如何产生对比度？
通过故意欠焦，显微镜能够将不可见的电子**相位差**转化为探测器上可见的**振幅差**。这被称为相位对比成像 (Phase Contrast Imaging)。欠焦量越大，图像在肉眼看来的黑白对比度就越强，蛋白质颗粒在背景中就越显眼。

### 3. 欠焦量在 CTF 中的数学地位
正如前面提到的，CTF 会在傅里叶空间中对图像信号进行调制，其正弦振荡公式如下：

$$\text{CTF}(k) = -\sin(\pi \lambda \Delta z k^2 - \frac{\pi}{2} C_s \lambda^3 k^4)$$

在这个公式中，**$\Delta z$ 就是欠焦量**。可以看出，欠焦量直接决定了 CTF 曲线振荡的剧烈程度：



* **高欠焦 (Large Defocus，例如 -2.5 μm 到 -3.0 μm)**：
    * **优势**：在低分辨率区域提供极高的对比度。颗粒在显微照片中非常清晰，极大地降低了早期使用传统 CV 算法或 AI 模型进行**颗粒挑选 (Particle Picking)** 的难度。
    * **劣势**：CTF 曲线在频率空间振荡得极其剧烈。频繁的零交叉点 (Zero-crossings) 会导致高频（高分辨率）信息严重丢失或相位反转，不利于后期解析原子级结构。
* **低欠焦 (Small Defocus，例如 -0.5 μm 到 -1.0 μm)**：
    * **优势**：CTF 曲线振荡缓慢，第一个零交叉点被推向更高的频率区间。这意味着更多的高分辨率结构细节（如氨基酸侧链的信号）得以无损保留。
    * **劣势**：低分辨率对比度极差，信噪比极低，图像看起来像充满噪声的灰色背景，很难直接用肉眼分辨出颗粒。

### 4. 实际处理中的互补策略
为了兼顾“看得见颗粒”和“保得住高分辨率细节”，并且填补 CTF 零交叉点带来的信息黑洞，冷冻电镜在实际采集数据时，**绝不会只使用一个固定的欠焦量**。

现代自动化采集系统会设置一个**欠焦梯度 (Defocus Range)**（例如让显微镜在 -0.8 μm 到 -2.5 μm 之间随机变化采集）。这样，在某一张高欠焦照片中因为 CTF 跌零而丢失的空间频率信息，就可以通过另一张低欠焦照片（其零交叉点位置不同）在后期的 3D 重建中完美互补填补回来。因此，精确估计每张照片的 $\Delta z$ 才是后续所有三维重建算法能够成立的基石。

2D iamge
先rotate摆正同一取向(依赖信噪比)再加和平均

Rotation function或Translation function 准确度都依赖信噪比
之前成像理论可知，欠焦量越大,Δz越大，contrast越好，但高分辨率信息丢的越多。小欠焦量，contrast不好。于是发明Image Pairs: 高低欠焦量各拍一张照片，高欠焦量用于alignment，低欠焦量用于三维重构。 但现在更多直接使用电子探测相机

Maximum Likelihood Method: 不断旋转raw image并与reference image相似度比较，同取向时相似度最大

---

ControlNet 和 T2I-Adapter 都是为冻结的文本到图像（Text-to-Image）扩散模型（如 Stable Diffusion）引入空间条件控制的里程碑式架构。从深度学习的网络设计角度来看，两者的核心思想同源，但在特征提取与注入机制上存在显著分歧。

以下是它们的核心联系与架构区别：

### 一、 核心联系 (Connections)

1. **目标一致**：两者都旨在解决标准 Diffusion 模型缺乏空间结构控制能力的问题，通过引入额外的条件（如 Canny 边缘、Depth 深度图、OpenPose 骨骼、Semantic Mask 等）来引导图像生成。
2. **冻结基座模型 (Frozen Base Model)**：在训练和推理阶段，两者都会“冻结”原始 SD 模型的权重（包括 UNet 和 VAE），只训练额外添加的辅助网络。这种设计避免了微调大型模型带来的灾难性遗忘，保留了原模型强大的先验生成能力。
3. **即插即用与多条件融合**：它们都可以作为外挂模块与不同的基座微调模型结合使用，并且都支持同时输入多个控制条件（如同时使用线稿和深度图进行约束）。

---

### 二、 架构与机制的区别 (Architectural Differences)

这正是两者产生性能、资源占用与效果差异的根本原因：

#### 1. 网络设计与特征注入路径
* **ControlNet** ：采用了**“权重克隆”**的重型架构。它直接复制了 SD UNet 的整个 Encoder（编码器）和 Middle Block（中间层）。在去噪过程中，条件输入与带噪潜变量融合，经过克隆的 Encoder 进行特征提取。提取出的多尺度特征通过**零卷积（Zero Convolutions）**，逐层注入到冻结的 SD UNet 的 **Decoder（解码器）**中。
* **T2I-Adapter** ：采用了**“轻量级解耦”**架构。它没有复制 UNet，而是从头设计了一个由 Pixel Unshuffle（像素重组）和残差块（ResBlock）组成的轻量级特征提取网络。条件图像经过该网络提取出多尺度特征后，直接与冻结的 SD UNet 的 **Encoder（编码器）**中的对应特征进行相加（Add）融合。

#### 2. 推理机制与时间开销
* **ControlNet**：在 Diffusion 采样的**每一个时间步（Timestep）**，ControlNet 分支都需要与主 UNet 一起运行。相当于在每次去噪迭代时，多计算了一个庞大的 Encoder，因此生成速度会明显变慢（通常会增加数倍的推理时间）。
* **T2I-Adapter**：条件特征的提取是**一次性（One-time）**的。Adapter 网络只在推理最开始运行一次，提取出的特征图会被缓存。在随后的几十个去噪时间步中，模型只需将缓存的特征图直接加到 UNet Encoder 的特征上。因此，T2I-Adapter 对推理速度的负面影响几乎为零（生成速度比 ControlNet 快约 3 倍）。

#### 3. 参数量与显存占用
* **ControlNet**：由于复制了半个 UNet，参数量非常庞大。其条件特征提取器的参数量远大于 T2I-Adapter，模型文件通常在几百 MB 到 1GB 左右。
* **T2I-Adapter**：极度轻量化，参数量仅在 77M 左右，模型文件大小约 300MB。在显存受限的环境下，或需要同时加载多个控制条件（Multi-Control）时，T2I-Adapter 的资源优势极其显著。

#### 4. 控制精度与生成质量
* **ControlNet**：通常具有更好的**细粒度控制力**和特征对齐能力。因为它继承了 SD 预训练的权重，且模型容量大，加上零卷积的保护，其生成的细节更锐利，对复杂条件（如复杂手部姿态、精细线稿）的遵循度更高。
* **T2I-Adapter**：在整体空间结构控制上非常优秀，但在极其微小的细节映射上可能略逊于 ControlNet（因为它的特征提取器是从零开始训练的小型网络）。不过，对于深度图（Depth）、颜色网格（Color）这种偏宏观布局的控制，两者的肉眼差距并不明显。

### 总结选型

* **选择 ControlNet**：当任务需要极致的控制精度（例如人体姿态骨骼、极度精细的线稿上色），且算力/显存充裕时。
* **选择 T2I-Adapter**：当任务注重推理速度、需要组合极多控制条件，或者在显存受限的设备上部署应用时。从工程落地视角来看，它是一个高性价比、更优雅的高效方案。


CryoEM single particle 3D reconstruction encounter 2 main problems: 1. composition heterogeneity 2. conformational heterogeneity

在冷冻电镜（Cryo-EM）单颗粒分析中，核心思想是通过对成千上万个**完全相同**的分子颗粒的二维投影进行“平均化”，从而计算出高分辨率的三维结构。

然而，真实的生物大分子在溶液中是动态且复杂的。样本中的颗粒往往并不完全一致，这种不一致性被称为**异质性（Heterogeneity）**。它主要分为两类：组成异质性（Compositional heterogeneity）和构象异质性（Conformational heterogeneity）。理解它们是获得高分辨率结构以及解析大分子动态机制的关键。



## 组成异质性 (Compositional Heterogeneity)

**是什么：**
组成异质性是指样本中复合物的**化学成分或分子组成**存在差异。即使是你非常纯的样品，在微观层面也可能有不同的“零部件”组装状态。
常见的例子包括：
* 蛋白质复合体在纯化或冷冻制样过程中，部分颗粒丢失了边缘的某个亚基（Subunit）。
* 只有一部分受体蛋白结合了小分子配体（Ligand）或药物分子，另一部分是空载状态（Apo state）。
* 样本中混入了不同组装阶段的中间态分子。

**有什么影响：**
如果在重构时没有将这些成分不同的颗粒区分开，而是强行将它们叠加平均，会导致那些有差异的区域（例如配体结合位点、容易脱落的亚基）的电子密度被严重“稀释”。
在最终获得的三维密度图中，分子的核心稳定区域可能分辨率很高，但发生组成变化的区域密度会非常模糊、呈现“半透明”状态，甚至由于密度阈值过低而完全看不见。这不仅会降低整体分辨率，还可能导致研究者误判配体是否结合，或者错误地搭建原子模型。



## 构象异质性 (Conformational Heterogeneity)

**是什么：**
构象异质性是指样本中的颗粒**化学组成完全一致，但空间形状或物理形态不同**。蛋白质是动态的纳米机器，它们在行使功能时会发生形变。这种异质性通常分为两种：
* **离散构象异质性：** 蛋白质在几个特定的稳定状态之间切换，比如离子通道的“完全开放态”、“闭合态”和“失活态”。
* **连续构象异质性：** 蛋白质的某个结构域（如柔性环区、铰链结构）在空间中进行连续的、无规律的摆动或弯曲。

**有什么影响：**
构象异质性对三维重构的影响极其显著。算法在将这些形状不同的颗粒进行三维空间对齐（Alignment）时会发生错位。
强行对存在构象差异的颗粒进行平均，会导致运动频繁或发生位移的结构域在最终的密度图中被“抹平”（Smearing effect）。其结果是，刚性较强的核心区域可能达到 2-3 Å 的原子级分辨率，但发生构象变化的柔性区域分辨率极低，甚至像一团模糊的云，完全无法追踪多肽主链的走向。

---

## 异质性带来的整体挑战与解决策略

在 Cryo-EM 领域，异质性通常被认为是阻碍分辨率突破的最大“敌人”，但同时也是获取生物大分子**动态机理和功能全貌**的巨大金矿。

为了消除这两种异质性带来的负面影响，目前的常规处理策略包括：
* **3D 分类 (3D Classification)：** 算法通过迭代计算，将数百万个颗粒按照形态或密度的细微差异，聚类到不同的“盒子”里。这样可以将丢失配体的颗粒与结合配体的颗粒分开（解决组成异质性），或者将开放态和闭合态分开（解决离散构象异质性），从而分别获得高分辨率的三维重构。
* **多体精修 (Multi-body Refinement) / 局部精修 (Local Refinement)：** 针对具有柔性连接但自身相对刚性的结构域。通过在图像对齐时给大分子加上“遮罩”（Mask），忽略其他部分的干扰，专门针对目标柔性区域进行高精度的对齐。
* **深度学习与连续构象解析：** 针对最棘手的连续构象异质性，近年来涌现了诸如 CryoDRGN、3DVA 等基于神经网络和流形学习的先进算法。它们不再强行进行离散分类，而是将颗粒映射到连续的隐空间中，从而直接生成蛋白质连续运动的三维动画。