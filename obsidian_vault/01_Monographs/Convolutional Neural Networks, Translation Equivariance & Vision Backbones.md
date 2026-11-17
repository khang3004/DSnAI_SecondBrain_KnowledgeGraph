---
type: monograph
domain: "D03"
parent_topic: "[[PT 03 - Convolutional, Graph, Recurrent & State-Space Architectures]]"
sub_topic: "[[ST 03 - Convolutional Networks & Graph Neural Networks]]"
cheatsheet: "[[Cheatsheet - Convolutional Neural Networks, Translation Equivariance & Vision Backbones]]"
part_of: 
  - "[[ST 03 - Convolutional Networks & Graph Neural Networks]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
concepts: 
  - "[[CNN & Equivariance]]"
  - "[[Vision Transformer (ViT)]]"
prerequisites: 
  - "[[Self-Attention & Multi-Head Attention]]"
enables: 
  - "[[CLIP & Vision-Language Alignment]]"
  - "[[Image Filtering & 2D Fourier Transform]]"
  - "[[Object Detection (R-CNN, SSD & DETR)]]"
  - "[[Semantic & Instance Segmentation (FCN, U-Net & SAM)]]"
related: 
  - "[[Residual Connections]]"
  - "[[Transformer Architecture]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-03
---

# Convolutional Neural Networks, Translation Equivariance & Vision Backbones

> [!abstract] Monograph Navigation
> - **Domain:** [[D03 - Neural Architectures, Representations & Regularization|Neural Architectures, Representations & Regularization]]
> - **Parent / Sub-Topic:** [[PT 03 - Convolutional, Graph, Recurrent & State-Space Architectures|Convolutional, Graph, Recurrent & State-Space Architectures]] → [[ST 03 - Convolutional Networks & Graph Neural Networks|Convolutional Networks & Graph Neural Networks]]
> - **Visual Cheatsheet:** [[Cheatsheet - Convolutional Neural Networks, Translation Equivariance & Vision Backbones]]
> - **Description:** Formalizes discrete/1D/2D convolution vs. cross-correlation, sparse Toeplitz weight sharing, translation equivariance vs. pooling invariance, receptive field growth, dilated/depthwise-separable convolutions, and U-Net/ResNet vision designs.

## Problem Formulation & Unified Notation

Convolutional neural networks (CNNs) are specialized architectures for data defined on regular discrete lattices, such as 1D temporal signals ($x \in \mathbb{R}^{C_{\mathrm{in}} \times N}$) and 2D spatial images ($\boldsymbol{X} \in \mathbb{R}^{C_{\mathrm{in}} \times H \times W}$) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 9, PDF p. 345). Unlike dense multilayer perceptrons (MLPs), which assign an independent weight to every input–output pair and scale quadratically as $O((C_{\mathrm{in}} H W)^2)$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 10, p. 175), convolutional layers exploit three structural inductive biases: *sparse local connectivity*, *spatial parameter sharing*, and *translation equivariance* [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 9, PDF pp. 350–354), [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 10, p. 306).

Let $\boldsymbol{X} \in \mathbb{R}^{C_{\mathrm{in}} \times H \times W}$ denote an input feature tensor with $C_{\mathrm{in}}$ channels, height $H$, and width $W$, and let $\boldsymbol{W} \in \mathbb{R}^{C_{\mathrm{out}} \times C_{\mathrm{in}} \times k_h \times k_w}$ denote a bank of $C_{\mathrm{out}}$ learnable kernels of spatial extent $k_h \times k_w$ with bias $\boldsymbol{b} \in \mathbb{R}^{C_{\mathrm{out}}}$. Under symmetric zero-padding $(p_h, p_w)$, stride $(s_h, s_w)$, and dilation rate $(d_h, d_w)$, the output feature map $\boldsymbol{Z} \in \mathbb{R}^{C_{\mathrm{out}} \times H_{\mathrm{out}} \times W_{\mathrm{out}}}$ has spatial dimensions [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 14, Eq. (14.10), PDF p. 502):

$$
H_{\mathrm{out}} = \left\lfloor \frac{H + 2p_h - d_h(k_h - 1) - 1}{s_h} \right\rfloor + 1, \qquad
W_{\mathrm{out}} = \left\lfloor \frac{W + 2p_w - d_w(k_w - 1) - 1}{s_w} \right\rfloor + 1.
$$

Setting $d_h = 1$, $s_h = 1$, and $p_h = (k_h - 1)/2$ for odd $k_h$ yields *same* convolution ($H_{\mathrm{out}} = H$), whereas $p_h = 0$ yields *valid* convolution ($H_{\mathrm{out}} = H - k_h + 1$) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 10, p. 178), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 14, PDF p. 501).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Because CNN literature bridges signal processing, numerical linear algebra, and deep learning, the four primary sources use distinct symbols and conventions for the same underlying tensor operations.
>
>
>
> | **Source Textbook** | **Unified Tensor Symbols, Conv vs. Cross-Corr. & Upsampling** |
> | — | — |
> | Goodfellow et al. | Uses $I, K, S$ (PDF p. 347) and $*$ for both conv and cross-corr., noting DL omits kernel flip (Eq. 9.6); upsampling via transposed conv / unpooling [[Source - Deep Learning]] (Ch. 9). |
> | [3pt] Bishop & Bishop | Uses $\boldsymbol{x}, \boldsymbol{w}, z$ (p. 306) with unflipped weighted sum; calls upsampling "fractionally strided" or transpose conv (p. 334) [[Source - Foundations]] (Ch. 10). |
> | [3pt] Prince | Uses $\boldsymbol{x}, \boldsymbol{\Omega}, \boldsymbol{h}$ with channels $C_i, C_o$ (pp. 179, 188); unflipped sum (Eq. 10.4); transposed conv $\boldsymbol{W}^{\top}$ and max unpooling (Fig. 10.12–13) [[Source - Understanding Deep Learning]] (Ch. 10). |
> | [3pt] Murphy | Uses $\boldsymbol{X}, \boldsymbol{W}, \boldsymbol{Z}$ with channels $C, D$ (PDF pp. 500, 518); distinguishes $\circledast$ (flipped) vs. $*$ (cross-corr.), adopts unflipped; warns "deconvolution" is a misnomer (PDF p. 517) [[Source - Probabilistic Machine Learning]] (Ch. 14). |

## Core Mathematical Derivations & Theorems

### Cross-Correlation, Convolution, and Toeplitz Factorization

> [!important] Discrete Cross-Correlation vs. Convolution & Toeplitz Linear Operator
> In pure mathematics and signal processing, discrete 2D convolution flips the kernel along both spatial axes to preserve commutativity:
>
> $$
> (\boldsymbol{X} *_{\mathrm{flip}} \boldsymbol{W})[i, j] = \sum_{u=0}^{k_h - 1} \sum_{v=0}^{k_w - 1} \boldsymbol{X}[i - u, j - v]  \boldsymbol{W}[u, v].
> $$
>
> Deep learning frameworks instead implement *discrete cross-correlation* (omitting the flip), which is equivalent up to a learned re-indexing of $\boldsymbol{W}$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 9, PDF p. 348), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 14, PDF p. 499):
>
> $$
> \boldsymbol{Z}[d, i, j] = b_d + \sum_{c=0}^{C_{\mathrm{in}}-1} \sum_{u=0}^{k_h - 1} \sum_{v=0}^{k_w - 1} \boldsymbol{X}[c, s_h i + d_h u - p_h,  s_w j + d_w v - p_w]  \boldsymbol{W}[d, c, u, v],
> $$
>
> where out-of-bounds indices evaluate to zero under zero-padding.
>
> **Toeplitz Matrix Equivalence.** Because `(eq:cnn:xcorr_2d)` is linear in $\boldsymbol{X}$ (ignoring bias), flattening $\boldsymbol{X} \in \mathbb{R}^{H \times W}$ into $\boldsymbol{x} = \mathrm{vec}(\boldsymbol{X}) \in \mathbb{R}^{HW}$ expresses single-channel 2D convolution as matrix–vector multiplication $\boldsymbol{z} = \boldsymbol{C}_{\boldsymbol{W}} \boldsymbol{x}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. (14.8), PDF p. 501), [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Fig. 10.4, p. 180). In 1D (valid, stride $1$, kernel $\boldsymbol{w} = [w_0, \dots, w_{k-1}]^{\top}$), $\boldsymbol{C}_{\boldsymbol{w}} \in \mathbb{R}^{(N - k + 1) \times N}$ is a banded *Toeplitz matrix* whose entries satisfy $(\boldsymbol{C}_{\boldsymbol{w}})_{i, j} = w_{j - i}$ for $0 \le j - i < k$ and $0$ otherwise:
>
> $$
> \boldsymbol{C}_{\boldsymbol{w}} =
> \begin{bmatrix}
> w_0 & w_1 & \cdots & w_{k-1} & 0 & \cdots & 0 \\
> 0 & w_0 & w_1 & \cdots & w_{k-1} & \ddots & \vdots \\
> \vdots & \ddots & \ddots & \ddots & & \ddots & 0 \\
> 0 & \cdots & 0 & w_0 & w_1 & \cdots & w_{k-1}
> \end{bmatrix}.
> $$
>
> In 2D, $\boldsymbol{C}_{\boldsymbol{W}}$ is a *doubly block Toeplitz matrix* (or doubly block circulant under periodic boundary conditions) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 9, PDF p. 348). Consequently, backpropagation through a convolutional layer is simply multiplication by $\boldsymbol{C}_{\boldsymbol{W}}^{\top}$, which corresponds to a *transposed convolution* [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (PDF p. 517), [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Fig. 10.13, p. 187). Parameter storage drops from $H W H_{\mathrm{out}} W_{\mathrm{out}}$ (dense MLP) to $k_h k_w$ per channel pair.

> [!note] Source Figure
> *Figure:* Sparse Toeplitz connectivity in 1D convolutional layers versus dense fully connected layers [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 10, Fig. 10.4, p. 180). Panels (a)–(b) show a dense MLP weight matrix; (c)–(d) show a stride-1 convolution with banded, tied weights; (e)–(f) show a stride-2 convolution reducing spatial resolution by half. (`understandingdl_simonprince_p180_fig1.png`)

### Translation Equivariance and Pooling Local Invariance

> [!important] Translation Equivariance Theorem & Pooling Invariance
> Let $T_{\Delta}$ be a spatial translation operator on $\mathbb{Z}^2$ defined by $(T_{\Delta} \boldsymbol{X})[c, y, x] = \boldsymbol{X}[c, y - \Delta_y, x - \Delta_x]$ for integer shift $\Delta = (\Delta_y, \Delta_x) \in \mathbb{Z}^2$ (assuming infinite support or circular padding). Let $\mathcal{F}_{\boldsymbol{W}}(\boldsymbol{X}) = \boldsymbol{X} * \boldsymbol{W}$ denote a stride-$1$ cross-correlation layer. Then $\mathcal{F}_{\boldsymbol{W}}$ is *translation equivariant*:
>
> $$
> \mathcal{F}_{\boldsymbol{W}}(T_{\Delta} \boldsymbol{X}) = T_{\Delta} \mathcal{F}_{\boldsymbol{W}}(\boldsymbol{X}).
> $$
>
> *Proof.* Evaluating the left-hand side at output coordinate $(i, j)$ for channel $d$:
>
> $$
> \begin{aligned}
> \big(\mathcal{F}_{\boldsymbol{W}}(T_{\Delta}\boldsymbol{X})\big)[d, i, j]
> &= \sum_{c, u, v} (T_{\Delta}\boldsymbol{X})[c, i + u, j + v]  \boldsymbol{W}[d, c, u, v] \\
> &= \sum_{c, u, v} \boldsymbol{X}[c, (i - \Delta_y) + u, (j - \Delta_x) + v]  \boldsymbol{W}[d, c, u, v] \\
> &= \big(\mathcal{F}_{\boldsymbol{W}}(\boldsymbol{X})\big)[d, i - \Delta_y, j - \Delta_x] = \big(T_{\Delta}\mathcal{F}_{\boldsymbol{W}}(\boldsymbol{X})\big)[d, i, j]. \quad \blacksquare
> \end{aligned}
> $$
>
> Pointwise nonlinearities $\sigma(\cdot)$ (e.g., ReLU) commute with $T_{\Delta}$, so any composition of stride-$1$ convolutions and pointwise activations preserves exact translation equivariance [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 9, PDF pp. 353–354), [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 10, Eq. (10.2), p. 176).
>
> **Local Invariance via Pooling.** A spatial pooling operator $P_k$ (e.g., max pooling $\max_{(u,v) \in \mathcal{N}_k(i,j)} \boldsymbol{H}[c, u, v]$ or average pooling) over a $k \times k$ neighborhood introduces *approximate local translation invariance*: if an activation peak lies strictly inside the interior of $\mathcal{N}_k(i, j)$, small shifts $\delta$ keep the maximum unchanged, yielding $P_k(T_{\delta}\boldsymbol{H}) \approx P_k(\boldsymbol{H})$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 9, Fig. 9.8, PDF p. 358). Global average pooling (GAP) over the entire spatial domain $(H, W)$ achieves exact global translation invariance (up to boundary effects) for image-level classification [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 10, Eq. (10.1), p. 176).

> [!tip] Convolution and Pooling as Infinitely Strong Priors
> From a Bayesian perspective, a convolutional layer is equivalent to a fully connected layer equipped with an *infinitely strong prior* over its weight matrix: all weights outside the local $k_h \times k_w$ receptive field have zero variance around zero, and weights at identical relative offsets across spatial locations are constrained to be equal with probability $1$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 9, Sec. 9.4, PDF p. 360). Similarly, pooling imposes an infinitely strong prior that features must be invariant to small local translations. When the data respects spatial locality and stationarity (natural images, spectrograms), these priors dramatically reduce sample complexity; when absolute pixel coordinates carry semantic meaning (e.g., aligned facial landmarks or CoordConv tasks), global weight sharing must be relaxed or augmented with positional encodings.

### Receptive Field Growth, Dilated Convolutions, and Depthwise Factorization

> [!important] Receptive Field Recursion & Depthwise-Separable Efficiency Ratio
> Consider a chain of $L$ convolutional or pooling layers, where layer $l \in \{1, \dots, L\}$ has kernel size $k_l$, stride $s_l$, and dilation rate $d_l$. The effective kernel span of layer $l$ is $k_l^{\mathrm{eff}} = d_l(k_l - 1) + 1$. Let $J_l = \prod_{i=1}^{l} s_i$ (with $J_0 = 1$) denote the cumulative stride (jump) at layer $l$. The theoretical receptive field size $R_l$ in the input image satisfied by a unit at layer $l$ obeys the exact linear recurrence [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 10, Fig. 10.6, p. 182), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 14, PDF p. 516):
>
> $$
> R_l = R_{l-1} + (k_l^{\mathrm{eff}} - 1) J_{l-1} = 1 + \sum_{l'=1}^{l} d_{l'}(k_{l'} - 1) \prod_{i=1}^{l'-1} s_i, \qquad R_0 = 1.
> $$
>
> Thus, stacking $m$ stride-$1$ layers of $3 \times 3$ kernels ($d_l = 1$) grows the receptive field linearly as $R_m = 2m + 1$ while requiring $m(9 C^2)$ parameters—whereas a single $(2m+1)\times(2m+1)$ kernel requires $(2m+1)^2 C^2$ parameters (e.g., two $3\times 3$ layers match a $5\times 5$ receptive field with $18 C^2$ vs. $25 C^2$ weights, a $28\%$ reduction, plus an extra nonlinearity). Exponentially increasing dilation $d_l = 2^{l-1}$ with $s_l = 1$ achieves exponential receptive field growth $R_L = 2^{L+1} - 1$ without losing spatial resolution [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 14.4.1, PDF p. 516).
>
> **Depthwise-Separable Convolution.** Standard 2D convolution couples spatial filtering and cross-channel mixing into a single $k \times k \times C_{\mathrm{in}} \times C_{\mathrm{out}}$ tensor. Depthwise-separable convolution [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 14.4.3, Eq. (14.26), PDF p. 518) factorizes this into:
> - *Depthwise spatial filtering*: one $k \times k$ filter per input channel, $\hat{\boldsymbol{Z}}[c, i, j] = \sum_{u,v} \boldsymbol{X}[c, i+u, j+v]  \boldsymbol{W}_{\mathrm{dw}}[c, u, v]$ ($k^2 C_{\mathrm{in}}$ parameters).
> - *Pointwise $1 \times 1$ channel projection*: $\boldsymbol{Z}[d, i, j] = b_d + \sum_{c=0}^{C_{\mathrm{in}}-1} \hat{\boldsymbol{Z}}[c, i, j]  \boldsymbol{W}_{\mathrm{pw}}[d, c]$ ($C_{\mathrm{in}} C_{\mathrm{out}}$ parameters) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 10.4.3, p. 188).
>
> For an output spatial grid of $H_{\mathrm{out}} \times W_{\mathrm{out}}$, the parameter and multiply–accumulate (MAC) reduction ratio relative to standard convolution is identical:
>
> $$
> \frac{\mathrm{Params}_{\mathrm{DS}}}{\mathrm{Params}_{\mathrm{Std}}}
> = \frac{\mathrm{MACs}_{\mathrm{DS}}}{\mathrm{MACs}_{\mathrm{Std}}}
> = \frac{k^2 C_{\mathrm{in}} + C_{\mathrm{in}} C_{\mathrm{out}}}{k^2 C_{\mathrm{in}} C_{\mathrm{out}}}
> = \frac{1}{C_{\mathrm{out}}} + \frac{1}{k^2}.
> $$
>
> For $k = 3$ and $C_{\mathrm{out}} = 256$, this reduces compute and parameters by $\approx 8.7\times$ ($11.5\%$ of standard cost).

> [!note] Source Figure
> *Figure:* Dilated (à trous) $3 \times 3$ convolution with dilation rates $d \in \{1, 2, 3\}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 14, Fig. 14.23, PDF p. 516). Inserting $d - 1$ zeros between kernel taps expands the effective receptive field from $3 \times 3$ to $5 \times 5$ ($d=2$) and $7 \times 7$ ($d=3$) with zero additional parameters. (`probabilisticml_kevinmurphy_introduction_p516_fig1.png`)

## Algorithmic Implementation & Comparative Synthesis

{ Modern GPU and TPU kernels avoid nested spatial loops by lowering 2D convolution into dense General Matrix Multiplication (GEMM) via `im2col` (image-to-column) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 9), or by executing factorized depthwise-separable passes [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 14.4.3, PDF p. 518). Algorithm `alg:cnn:ds_im2col` formalizes both mechanisms.
}

> [!example] Algorithm: Im2Col GEMM Lowering vs. Depthwise-Separable 2D Convolution
> **Input:** Input $\boldsymbol{X} \in \mathbb{R}^{C_{\mathrm{in}} \times H \times W}$, kernel size $k$, stride $s$, padding $p$, mode $\in \{Std, DS\}$.
> **Output:** Output feature map $\boldsymbol{Z} \in \mathbb{R}^{C_{\mathrm{out}} \times H_{\mathrm{out}} \times W_{\mathrm{out}}}$.
> Compute $H_{\mathrm{out}}, W_{\mathrm{out}}$ via `(eq:cnn:output_shape)` with $d_h = d_w = 1$
> **if** mode = Std:
>
>    *// Standard Convolution via Im2Col GEMM Lowering*
>   Construct patch matrix $\boldsymbol{M}_{\mathrm{col}} \in \mathbb{R}^{(C_{\mathrm{in}} k^2) \times (H_{\mathrm{out}} W_{\mathrm{out}})}$ by extracting each $C_{\mathrm{in}} \times k \times k$ receptive field as a column
>   Reshape kernel bank $\boldsymbol{W} \in \mathbb{R}^{C_{\mathrm{out}} \times C_{\mathrm{in}} \times k \times k}$ into $\boldsymbol{W}_{\mathrm{row}} \in \mathbb{R}^{C_{\mathrm{out}} \times (C_{\mathrm{in}} k^2)}$
>   Execute dense BLAS GEMM: $\boldsymbol{Z}_{\mathrm{flat}} \leftarrow \boldsymbol{W}_{\mathrm{row}} \boldsymbol{M}_{\mathrm{col}} + \boldsymbol{b} \boldsymbol{1}^{\top}$
>   Reshape $\boldsymbol{Z}_{\mathrm{flat}} \in \mathbb{R}^{C_{\mathrm{out}} \times (H_{\mathrm{out}} W_{\mathrm{out}})}$ to $\boldsymbol{Z} \in \mathbb{R}^{C_{\mathrm{out}} \times H_{\mathrm{out}} \times W_{\mathrm{out}}}$
>
> **else**:
>
>    *// Depthwise-Separable Factorization*
>   **for** $c \leftarrow 0$ to $C_{\mathrm{in}} - 1$:
>
>     Compute spatial slice $\hat{\boldsymbol{Z}}[c, i, j] \leftarrow \sum_{u,v=0}^{k-1} \boldsymbol{X}[c, si + u - p, sj + v - p]  \boldsymbol{W}_{\mathrm{dw}}[c, u, v]$
>
>   Flatten spatial axes of $\hat{\boldsymbol{Z}}$ to $\hat{\boldsymbol{Z}}_{\mathrm{flat}} \in \mathbb{R}^{C_{\mathrm{in}} \times (H_{\mathrm{out}} W_{\mathrm{out}})}$
>   Execute $1 \times 1$ pointwise GEMM: $\boldsymbol{Z}_{\mathrm{flat}} \leftarrow \boldsymbol{W}_{\mathrm{pw}} \hat{\boldsymbol{Z}}_{\mathrm{flat}} + \boldsymbol{b} \boldsymbol{1}^{\top}$ where $\boldsymbol{W}_{\mathrm{pw}} \in \mathbb{R}^{C_{\mathrm{out}} \times C_{\mathrm{in}}}$
>   Reshape $\boldsymbol{Z}_{\mathrm{flat}}$ to $\boldsymbol{Z} \in \mathbb{R}^{C_{\mathrm{out}} \times H_{\mathrm{out}} \times W_{\mathrm{out}}}$
>
> **return** $\boldsymbol{Z}$

Table `tab:cnn:ops_compare` summarizes the parameter complexity, arithmetic intensity, receptive-field contribution, and primary failure modes across convolutional operators.

**Table:** Computational and structural comparison of 2D convolutional operators ($k \times k$ kernel, $C_{\mathrm{in}} \to C_{\mathrm{out}}$ channels, output size $H_{\mathrm{out}} \times W_{\mathrm{out}}$).

| **Operator** | **Parameters** | **MACs ($\times H_{\mathrm{out}} W_{\mathrm{out}}$)** | **Primary Role & Failure Mode** |
| — | — | — | — |
| Standard $k \times k$ | $k^2 C_{\mathrm{in}} C_{\mathrm{out}}$ | $k^2 C_{\mathrm{in}} C_{\mathrm{out}}$ | General feature extractor; quadratic scaling in $k$ and $C$. |
| Stacked $2 \times (3 \times 3)$ | $18 C^2$ ($C_{\mathrm{in}}=C_{\mathrm{out}}=C$) | $18 C^2$ | Matches $5 \times 5$ RF ($25 C^2$) with $28\%$ fewer params + extra ReLU. |
| Pointwise $1 \times 1$ | $C_{\mathrm{in}} C_{\mathrm{out}}$ | $C_{\mathrm{in}} C_{\mathrm{out}}$ | Channel bottleneck/expansion; zero spatial mixing. |
| Dilated ($d > 1$) | $k^2 C_{\mathrm{in}} C_{\mathrm{out}}$ | $k^2 C_{\mathrm{in}} C_{\mathrm{out}}$ | Exponential RF growth; gridding artifacts if $d$ shares factors. |
| Depthwise-Separable | $C_{\mathrm{in}}(k^2 + C_{\mathrm{out}})$ | $C_{\mathrm{in}}(k^2 + C_{\mathrm{out}})$ | Edge/mobile backbones; lower arithmetic intensity on GPUs. |
| Transposed ($s > 1$) | $k^2 C_{\mathrm{in}} C_{\mathrm{out}}$ | $k^2 C_{\mathrm{in}} C_{\mathrm{out}}$ (at $H_{\mathrm{in}} W_{\mathrm{in}}$) | Learnable upsampling; checkerboard artifacts if $k \bmod s \neq 0$. |

> [!info] Cross-Book Synthesis: Vision Backbone Evolution (AlexNet/VGG $\to$ ResNet $\to$ U-Net)
> Across [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 10.5, pp. 188–191), [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 10.2.8 & 10.5.4, pp. 315–336), and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 14.3, pp. 509–515), modern vision architectures evolved through three structural transitions:
> - **AlexNet (2012) $\to$ VGG (2014)**: AlexNet demonstrated ImageNet scalability using large $11 \times 11$ and $5 \times 5$ kernels with ReLU and max pooling, whereas VGG standardized homogeneous stacks of $3 \times 3$ stride-$1$ convolutions interleaved with $2 \times 2$ pooling, proving that depth with small kernels outperforms shallow wide kernels.
> - **GoogLeNet / ResNet (2015)**: To scale beyond $\sim 20$ layers without degradation or excessive FLOPs, GoogLeNet introduced $1 \times 1$ dimensionality-reduction bottlenecks, and ResNet introduced additive residual skip connections $\boldsymbol{H}_{l+1} = \boldsymbol{H}_l + \mathcal{F}(\boldsymbol{H}_l)$ (with $1 \times 1$ projection shortcuts when channel dimensions change) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 14.20, PDF p. 513). Bottleneck blocks ($1\times 1$ compress $\to 3\times 3$ spatial $\to 1\times 1$ expand) preserve representational capacity while slashing $3\times 3$ FLOPs.
> - **FCN $\to$ U-Net (2015)**: Classification backbones discard spatial coordinates via repeated striding. For dense pixel-level semantic segmentation, U-Net [[Source - Foundations|deep_learning_foundations_concepts]] (Fig. 10.31, p. 335) pairs a contracting encoder with a symmetric expanding decoder, concatenating high-resolution encoder feature maps directly into decoder stages via skip connections to recover fine boundary localization.

> [!note] Source Figure
> *Figure:* The U-Net encoder–decoder architecture for semantic segmentation [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 10, Fig. 10.31, p. 335). Contracting encoder blocks (red) extract semantic context at coarser resolutions, while skip concatenations (horizontal arrows) inject high-resolution spatial features into the expanding decoder (blue). (`deep_learning_foundations_concepts_p335_fig2.png`)

## Visual Diagnostics & Geometric Structure

Figure `fig:cnn:tikz_factorization` illustrates the structural contrast between standard 2D convolution, depthwise-separable factorization, and ResNet bottleneck/U-Net multi-scale skip topologies. Figure `fig:cnn:pgfplots_rf_eff` quantifies the receptive field growth dynamics `(eq:cnn:rf_recursion)` and the depthwise-separable parameter/compute efficiency ratio `(eq:cnn:ds_ratio)`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Factorized and residual convolutional motifs: (a) replacing coupled $k \times k \times C_{\mathrm{in}} \times C_{\mathrm{out}}$ filters with depthwise spatial filtering plus $1 \times 1$ pointwise mixing; (b) ResNet bottleneck block compressing channels by $4\times$ before spatial $3 \times 3$ filtering while preserving gradient flow via the identity skip.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Quantitative scaling of CNN operators: (a) theoretical receptive field $R_l$ across depth $l$ under standard stride-$1$, periodic striding, and exponential dilation `(eq:cnn:rf_recursion)`; (b) parameter and MAC ratio $\frac{1}{C_{\mathrm{out}}} + \frac{1}{k^2}$ of depthwise-separable vs. standard convolution `(eq:cnn:ds_ratio)`.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Stride-Induced Aliasing Breaking Shift-Invariance
> **Failure Mode.** Standard strided convolutions and pooling layers ($s \ge 2$) sub-sample feature maps without an anti-aliasing low-pass filter. By the Nyquist–Shannon sampling theorem, high-frequency spatial components fold into low frequencies, destroying exact translation equivariance: a $1$-pixel input shift can produce markedly different output activations and class predictions.
> **Remedy.** Apply a spatial low-pass blur kernel (e.g., $[1, 2, 1]/4$ or $[1, 4, 6, 4, 1]/16$ separable Gaussian) immediately before sub-sampling (BlurPool), or use stride-$1$ pooling followed by anti-aliased downsampling.

> [!warning] Transposed Convolution Checkerboard Artifacts
> **Failure Mode.** In transposed convolution with stride $s > 1$ and kernel size $k$, each input pixel stamps a $k \times k$ kernel onto the output grid at spacing $s$ [[Source - Foundations|deep_learning_foundations_concepts]] (Fig. 10.30, p. 335), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 14.24, PDF p. 517). Whenever $k$ is not divisible by $s$ ($k \bmod s \neq 0$, e.g., $k=3, s=2$), adjacent output pixels receive unequal numbers of overlapping kernel stamps, creating periodic high-frequency "checkerboard" grid artifacts in generated or segmented images.
> **Remedy.** Either choose $k$ divisible by $s$ (e.g., $k=4, s=2$), or replace transposed convolution with parameter-free bilinear/nearest-neighbor upsampling followed by a standard stride-$1$ $3 \times 3$ convolution (resize-convolution).

> [!warning] Theoretical vs. Effective Receptive Field (ERF) Decay
> **Failure Mode.** Although `(eq:cnn:rf_recursion)` predicts a uniform $R_l \times R_l$ support, gradient sensitivity $\partial \boldsymbol{Z}_L[0,0] / \partial \boldsymbol{X}[u,v]$ is heavily concentrated at the center. Under random or typical learned weights, the effective receptive field (ERF) follows a 2D Gaussian profile whose standard deviation grows only as $O(\sqrt{L})$ rather than $O(L)$, leaving peripheral boundary pixels in $R_l$ nearly ignored.
> **Remedy.** Use residual skip connections (which preserve shorter paths), dilated convolutions ($d_l > 1$) in later stages, or hybridize deep CNN stages with global self-attention blocks when long-range spatial context is critical.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[CNN & Equivariance]], [[Vision Transformer (ViT)]]
- **Prerequisites (Directed $\leftarrow$):** [[Self-Attention & Multi-Head Attention]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[CLIP & Vision-Language Alignment]], [[Image Filtering & 2D Fourier Transform]], [[Object Detection (R-CNN, SSD & DETR)]], [[Semantic & Instance Segmentation (FCN, U-Net & SAM)]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Residual Connections]], [[Transformer Architecture]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
