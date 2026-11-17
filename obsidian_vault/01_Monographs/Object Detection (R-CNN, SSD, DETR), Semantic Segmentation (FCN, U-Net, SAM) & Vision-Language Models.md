---
type: monograph
domain: "D03"
parent_topic: "[[PT 03 - Computer Vision, Image Formation & 3D Scene Understanding]]"
sub_topic: "[[ST 03 - Object Detection, Segmentation & Vision-Language Understanding]]"
cheatsheet: "[[Cheatsheet - Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]"
part_of: 
  - "[[ST 03 - Object Detection, Segmentation & Vision-Language Understanding]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
concepts: 
  - "[[CLIP & Vision-Language Alignment]]"
  - "[[IoU, Anchor Boxes & NMS]]"
  - "[[Object Detection (R-CNN, SSD & DETR)]]"
  - "[[Semantic & Instance Segmentation (FCN, U-Net & SAM)]]"
prerequisites: 
  - "[[CNN & Equivariance]]"
  - "[[Gaussian & Laplacian Pyramids]]"
  - "[[IoU, Anchor Boxes & NMS]]"
  - "[[Residual Connections]]"
  - "[[Transformer Architecture]]"
  - "[[Vision Transformer (ViT)]]"
enables: 
  - "[[Object Detection (R-CNN, SSD & DETR)]]"
related: 
  - "[[Classifier-Free Guidance (CFG)]]"
  - "[[DDPM]]"
  - "[[Gaussian & Laplacian Pyramids]]"
  - "[[IoU, Anchor Boxes & NMS]]"
  - "[[Object Detection (R-CNN, SSD & DETR)]]"
  - "[[Semantic & Instance Segmentation (FCN, U-Net & SAM)]]"
  - "[[Sparse vs Dense Retrieval & RRF]]"
  - "[[Transformer Architecture]]"
sources: 
  - "[[Source - Antonio Torralba, Phillip Isola, and William Freeman]]"
  - "[[Source - Dive into Deep Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-03
---

# Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models

> [!abstract] Monograph Navigation
> - **Domain:** [[D03 - Neural Architectures, Representations & Regularization|Neural Architectures, Representations & Regularization]]
> - **Parent / Sub-Topic:** [[PT 03 - Computer Vision, Image Formation & 3D Scene Understanding|Computer Vision, Image Formation & 3D Scene Understanding]] → [[ST 03 - Object Detection, Segmentation & Vision-Language Understanding|Object Detection, Segmentation & Vision-Language Understanding]]
> - **Visual Cheatsheet:** [[Cheatsheet - Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]
> - **Description:** Bounding-box IoU & NMS, anchor-box multiscale detection (SSD, Faster R-CNN) vs. bipartite Hungarian matching (DETR), transposed convolution, FCN/U-Net/SAM segmentation, neural style transfer, and contrastive CLIP vision-language alignment.

## Problem Formulation & Unified Notation

\begin{sloppypar}
Dense visual scene understanding extends image-level classification $f(\boldsymbol{I}) \to y$ to structured spatial outputs: *object detection* predicts a variable-size set $\{(c_i, \boldsymbol{b}_i)\}_{i=1}^M$ of categories and bounding boxes $\boldsymbol{b}_i = (x, y, w, h)^{\top}$, while *semantic/instance segmentation* assigns pixel-level masks $M \in \{1,\dots,C\}^{H \times W}$ [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 14, pp. 646–705), [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 50, pp. 982–1014). Modern vision systems further align visual features with natural language embeddings via contrastive pretraining (CLIP) and promptable mask decoders (SAM) [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 51, pp. 1015–1032).
\end{sloppypar}

> [!info] Cross-Book Notation: Detection, Anchor Boxes & Segmentation
> | **Concept** | **D2L (Zhang et al.)** | **MIT Vision / Prince** | **Unified Monograph** |
> | — | — | — | — |
> | Bounding Box | $(x_1,y_1,x_2,y_2)$ or $(c_x,c_y,w,h)$ | $\boldsymbol{b} \in \mathbb{R}^4$ | $\boldsymbol{b} = (c_x, c_y, w, h)^{\top}$ |
> | Box Overlap | $\mathrm{IoU}(A,B)$ (Jaccard) | IoU / GIoU | $\mathrm{IoU}(A,B) = \frac{\mid A \cap B\mid }{\mid A \cup B\mid }$ |
> | Anchor Parameterization | Scale $s \in (0,1]$, aspect $r > 0$ | Multi-scale FPN anchors | $w_a = s\sqrt{r}$, $h_a = s/\sqrt{r}$ |
> | Upsampling | Transposed conv $\boldsymbol{W}^{\top}$ | Skip-connected U-Net / FCN | Transposed conv or bilinear + conv |

## Core Mathematical Derivations

### Bounding-Box Geometry, Anchor Boxes, NMS & Bipartite Matching (DETR)

> [!important] IoU, Box Offset Regression, Focal Loss & Hungarian Set Matching
> Given predicted box $A$ and ground-truth box $B$, *Intersection over Union (IoU)* measures localization quality [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 14, pp. 649–653):
>
> $$
> \mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} \in [0, 1], \qquad \mathcal{L}_{\mathrm{GIoU}} = 1 - \mathrm{IoU}(A,B) + \frac{|C \setminus (A \cup B)|}{|C|},
>
> $$
>
> where $C$ is the smallest enclosing box. Anchor-based detectors (SSD, Faster R-CNN, YOLO) tile each feature map cell $(i,j)$ with $K$ priors of scale $s$ and aspect ratio $r$ ($w_a = s\sqrt{r}, h_a = s/\sqrt{r}$) and regress scale-invariant offsets [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 14, pp. 650–675):
>
> $$
> t_x = \frac{c_x - c_{x,a}}{w_a}, \quad t_y = \frac{c_y - c_{y,a}}{h_a}, \quad t_w = \log\frac{w}{w_a}, \quad t_h = \log\frac{h}{h_a}.
>
> $$
>
> To suppress extreme easy-negative background dominance in dense one-stage detectors, *Focal Loss* down-weights well-classified anchors: $\mathcal{L}_{\mathrm{FL}}(p_t) = -\alpha_t (1-p_t)^\gamma \log p_t$ ($\gamma \approx 2$) [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 50, pp. 995–1002). Overlapping predictions are pruned via *Non-Maximum Suppression (NMS)* or eliminated entirely in *DETR*, which treats detection as direct set prediction via optimal bipartite *Hungarian matching* $\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^N \mathcal{L}_{\mathrm{match}}(y_i, \hat{y}_{\sigma(i)})$ [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 50, pp. 1004–1010).

### Semantic/Instance Segmentation, Style Transfer & Contrastive Vision-Language

> [!important] Transposed Convolution, FCN/U-Net, Style Gram Matrices & CLIP InfoNCE
> Dense pixel prediction recovers spatial resolution from downsampled backbone features using *transposed convolution* (whose forward and backward passes multiply by $\boldsymbol{W}^{\top}$ and $\boldsymbol{W}$ respectively) or bilinear upsampling followed by $3\times 3$ refinement [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 14, pp. 694–704). Because deep layers encode semantics at coarse resolution while shallow layers retain crisp boundaries, *FCN* and *U-Net* fuse multi-scale feature maps via skip connections [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 10, pp. 185–192), [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 14, pp. 699–705).
>
> In *Neural Style Transfer*, content is preserved via deep activation MSE $\lVert \boldsymbol{F}^l - \boldsymbol{P}^l \rVert_F^2$, while texture statistics are matched via channel-wise *Gram matrices* $G_{ij}^l = \sum_k F_{ik}^l F_{jk}^l$ [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 14, pp. 706–712). Finally, *CLIP* aligns image encoder $f_I(\boldsymbol{I})$ and text encoder $f_T(\boldsymbol{t})$ over $B$ paired captions using symmetric InfoNCE contrastive loss [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 51, pp. 1015–1028):
>
> $$
> \mathcal{L}_{\mathrm{CLIP}} = -\frac{1}{2B}\sum_{i=1}^B \left[\log\frac{\exp(\langle \boldsymbol{u}_i, \boldsymbol{v}_i \rangle/\tau)}{\sum_j \exp(\langle \boldsymbol{u}_i, \boldsymbol{v}_j \rangle/\tau)} + \log\frac{\exp(\langle \boldsymbol{u}_i, \boldsymbol{v}_i \rangle/\tau)}{\sum_j \exp(\langle \boldsymbol{u}_j, \boldsymbol{v}_i \rangle/\tau)}\right].
>
> $$

## Visual Diagnostics & Algorithmic Pipeline

> [!example] Algorithm: Greedy Non-Maximum Suppression (NMS) for Object Detection
> **Input:** Candidate boxes $\mathcal{B} = \{b_1, \dots, b_M\}$, confidence scores $\{s_1, \dots, s_M\}$, IoU threshold $\tau_{\mathrm{nms}}$
> **Output:** Filtered detection set $\mathcal{D}$
> Sort $\mathcal{B}$ in descending order of confidence $s_i$
> $\mathcal{D} \gets \emptyset$
> **while** $\mathcal{B} \neq \emptyset$:
>
>   Pop highest-scoring box $b^*$ from $\mathcal{B}$ and append $b^*$ to $\mathcal{D}$
>   **for each** remaining box $b \in \mathcal{B}$:
>
>     **if** $\mathrm{IoU}(b^*, b) > \tau_{\mathrm{nms}}$:
> Remove $b$ from $\mathcal{B}$
>
>
> **return** $\mathcal{D}$

> [!warning] Checkerboard Artifacts, Small-Object Collapse & NMS Crowding
> Transposed convolutions whose kernel size is not divisible by stride create uneven overlap and *checkerboard artifacts*; prefer `Resize + Conv` or divisible kernels [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 14, pp. 694–700). In single-scale detectors, downsampling by $32\times$ erases small objects; Feature Pyramid Networks (FPN) resolve this by routing high-resolution shallow maps with top-down semantic context [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 14, pp. 663–666). Standard hard NMS deletes valid neighboring objects in dense crowds—use Soft-NMS or end-to-end DETR set prediction [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 50, pp. 1004–1008).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (4):** [[CLIP & Vision-Language Alignment]], [[IoU, Anchor Boxes & NMS]], [[Object Detection (R-CNN, SSD & DETR)]], [[Semantic & Instance Segmentation (FCN, U-Net & SAM)]]
- **Prerequisites (Directed $\leftarrow$):** [[CNN & Equivariance]], [[Gaussian & Laplacian Pyramids]], [[IoU, Anchor Boxes & NMS]], [[Residual Connections]], [[Transformer Architecture]], [[Vision Transformer (ViT)]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Object Detection (R-CNN, SSD & DETR)]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Classifier-Free Guidance (CFG)]], [[DDPM]], [[Gaussian & Laplacian Pyramids]], [[IoU, Anchor Boxes & NMS]], [[Object Detection (R-CNN, SSD & DETR)]], [[Semantic & Instance Segmentation (FCN, U-Net & SAM)]], [[Sparse vs Dense Retrieval & RRF]], [[Transformer Architecture]]
- **Synthesized Sources:** [[Source - Antonio Torralba, Phillip Isola, and William Freeman]], [[Source - Dive into Deep Learning]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
