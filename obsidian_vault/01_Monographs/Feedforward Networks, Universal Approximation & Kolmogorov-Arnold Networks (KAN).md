---
type: monograph
domain: "D03"
parent_topic: "[[PT 03 - MLPs, Kolmogorov-Arnold Networks, Residuals & Regularization]]"
sub_topic: "[[ST 03 - Feedforward Networks, Normalization, Residuals & KANs]]"
cheatsheet: "[[Cheatsheet - Feedforward Networks, Universal Approximation & Kolmogorov-Arnold Networks (KAN)]]"
part_of: 
  - "[[ST 03 - Feedforward Networks, Normalization, Residuals & KANs]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
concepts: 
  - "[[Kolmogorov-Arnold Networks (KAN)]]"
  - "[[MLP & Universal Approximation]]"
prerequisites: 
  - "[[Automatic Differentiation (AD)]]"
enables: 
  - "[[Factorization Machines & DeepFM]]"
  - "[[He & Xavier Initialization]]"
related: 
  - "[[Automatic Differentiation (AD)]]"
  - "[[Radiance Fields (NeRF & 3DGS)]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Math behind KAN]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-03
---

# Feedforward Networks, Universal Approximation & Kolmogorov-Arnold Networks (KAN)

> [!abstract] Monograph Navigation
> - **Domain:** [[D03 - Neural Architectures, Representations & Regularization|Neural Architectures, Representations & Regularization]]
> - **Parent / Sub-Topic:** [[PT 03 - MLPs, Kolmogorov-Arnold Networks, Residuals & Regularization|MLPs, Kolmogorov-Arnold Networks, Residuals & Regularization]] → [[ST 03 - Feedforward Networks, Normalization, Residuals & KANs|Feedforward Networks, Normalization, Residuals & KANs]]
> - **Visual Cheatsheet:** [[Cheatsheet - Feedforward Networks, Universal Approximation & Kolmogorov-Arnold Networks (KAN)]]
> - **Description:** Connects piecewise-linear ReLU polyhedral regions and the Cybenko/Hornik Universal Approximation Theorem with Kolmogorov-Arnold superposition and learnable B-spline edge activations in KANs.

## Problem Formulation & Unified Notation

Supervised learning seeks to approximate an unknown target mapping $f^* : \mathcal{X} \subseteq \mathbb{R}^{D_i} \to \mathbb{R}^{D_o}$ from finite observations $\{(\boldsymbol{x}_n, \boldsymbol{y}_n)\}_{n=1}^N$. Classical linear basis-function models parameterize predictions as $y(\boldsymbol{x}, \boldsymbol{w}) = \sum_{j=1}^M w_j \varphi_j(\boldsymbol{x}) + w_0$ using *fixed*, data-independent basis functions $\varphi_j$ [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 6.1, p. 172). For multivariate polynomials of degree $M$ in $D_i$ inputs, the number of independent coefficients grows as $O(D_i^M)$, triggering Bellman's curse of dimensionality [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 6.1.1, p. 173). Feedforward neural networks—specifically Multilayer Perceptrons (MLPs)—overcome this bottleneck by making the basis functions themselves *data-adaptive* through compositions of affine maps and elementwise nonlinearities. Kolmogorov-Arnold Networks (KANs) offer a dual paradigm: instead of learning linear weights on fixed node activations, KANs place learnable univariate spline functions directly on the graph edges.

Let an $L$-layer architecture have layer widths $[n_0, n_1, \dots, n_L]$ with $n_0 = D_i$ and $n_L = D_o$, and set $\boldsymbol{h}^{(0)} = \boldsymbol{x}$.
- **Multilayer Perceptron (MLP):** Each layer $\ell \in \{1,\dots,L-1\}$ applies an affine transformation followed by a fixed scalar activation $\sigma : \mathbb{R} \to \mathbb{R}$ evaluated elementwise:
  
$$
\boldsymbol{a}^{(\ell)} = \boldsymbol{W}^{(\ell)} \boldsymbol{h}^{(\ell-1)} + \boldsymbol{b}^{(\ell)}, \qquad
    \boldsymbol{h}^{(\ell)} = \sigma\!\left(\boldsymbol{a}^{(\ell)}\right),
$$

  where $\boldsymbol{W}^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$ and $\boldsymbol{b}^{(\ell)} \in \mathbb{R}^{n_\ell}$, culminating in a linear (or task-specific) readout $\hat{\boldsymbol{y}} = \boldsymbol{W}^{(L)} \boldsymbol{h}^{(L-1)} + \boldsymbol{b}^{(L)}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 13.2.2, p. 428), [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 4.4.1, p. 49).
- **Kolmogorov-Arnold Network (KAN):** Each layer $\ell \in \{1,\dots,L\}$ replaces scalar edge weights with a matrix of learnable univariate functions $\boldsymbol{\Phi}^{(\ell)} = [\phi_{\ell,q,p}]_{q=1..n_\ell,  p=1..n_{\ell-1}}$, and each node performs pure summation:
  
$$
h^{(\ell)}_q = \sum_{p=1}^{n_{\ell-1}} \phi_{\ell,q,p}\!\left(h^{(\ell-1)}_p\right), \qquad q = 1, \dots, n_\ell,
$$

  where each $\phi_{\ell,q,p}$ is parameterized as a residual SiLU term plus a degree-$k$ B-spline on $G$ grid intervals [[Source - Math behind KAN|math_behind_kan]] (PDF p. 6).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
>
> | **Concept** | **Unified Monograph** | **Prince / Bishop / Goodfellow** | **Murphy / KAN Monograph** |
> | --- | --- | --- | --- |
> | Input / Output dim. | $D_i = n_0,  D_o = n_L$ | Prince: $D_i, D_o$; Goodfellow: $d$ inputs; Bishop: $D$ inputs | Murphy: $D$ inputs; KAN: $n_0, n_L$ |
> | Layer weights | $\boldsymbol{W}^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$ | Prince: $\boldsymbol{\Omega}_{k-1}$; Bishop/Goodfellow: $\boldsymbol{W}^{(\ell)}$ | Murphy: $\boldsymbol{W}_l$; KAN: $c_m$ (spline coeffs) |
> | Layer bias | $\boldsymbol{b}^{(\ell)} \in \mathbb{R}^{n_\ell}$ | Prince: $\boldsymbol{\beta}_{k-1}$; Goodfellow: $\boldsymbol{c}, \boldsymbol{b}$ | Murphy: $\boldsymbol{b}_l$; KAN: absorbed in $\phi_{\ell,q,p}$ |
> | Pre- / Post-activation | $\boldsymbol{a}^{(\ell)},  \boldsymbol{h}^{(\ell)}$ | Prince: $\boldsymbol{h}_k$; Bishop: $a, z$; Goodfellow: $\boldsymbol{h}$ | Murphy: $\boldsymbol{a}_l, \boldsymbol{z}_l$; KAN: $x_p, \Phi_q$ |
> | Activation function | $\sigma(\cdot)$ (MLP), $\phi_{\ell,q,p}(\cdot)$ (KAN) | Prince: $a[\cdot]$; Bishop: $h(\cdot)$; Goodfellow: $g(\cdot)$ | Murphy: $\varphi_l(\cdot)$; KAN: $w_b b(x) + w_s \mathrm{spline}(x)$ |
> | Depth / Width | $L$ layers, $n_\ell$ units | Prince: $K$ hidden layers, $D$ units; Goodfellow: depth $l$, width $n$ | Murphy: $L$ layers, $K_l$ units; KAN: $[n_0,\dots,n_L]$ |

## Core Mathematical Derivations & Theorems

### Universal Approximation Theorems for Shallow Networks

As established in [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 6.4.1, p. 194) and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 13.2.5, p. 434), a single hidden layer suffices for uniform approximation on compact domains whenever the activation function is non-polynomial.

> [!important] Cybenko–Hornik Universal Approximation Theorem (UAT)
> Let $K \subset \mathbb{R}^{D_i}$ be compact and let $f \in C(K, \mathbb{R})$ be any continuous target function. Suppose $\sigma : \mathbb{R} \to \mathbb{R}$ is continuous and non-polynomial (e.g., logistic sigmoid, $\tanh$, or $\mathrm{ReLU}$). Then for every tolerance $\epsilon > 0$, there exist a finite width $N \in \mathbb{N}$, output weights $\alpha_j \in \mathbb{R}$, input weights $\boldsymbol{w}_j \in \mathbb{R}^{D_i}$, and biases $\theta_j \in \mathbb{R}$ ($j = 1, \dots, N$) such that
>
> $$
> \sup_{\boldsymbol{x} \in K} \lvert f(\boldsymbol{x}) - \sum_{j=1}^N \alpha_j \sigma\!\left(\boldsymbol{w}_j^{\top} \boldsymbol{x} + \theta_j\right) \rvert < \epsilon.
> $$
>

> [!tip] Geometric Partitioning vs. the Width Curse
> In 1D with $\mathrm{ReLU}(z) = \max(0, z)$, each hidden unit $h_j(x) = \mathrm{ReLU}(\theta_{j0} + \theta_{j1}x)$ introduces a single "joint" (hinge) at $x = -\theta_{j0}/\theta_{j1}$, so $D$ hidden units partition $\mathbb{R}$ into at most $D + 1$ linear segments [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 3.2, p. 29). Adding units refines the piecewise-linear interpolant to arbitrary precision. In $D_i$ dimensions, however, approximating a generic Lipschitz function to accuracy $\epsilon$ with a shallow network can require $N = \Omega(\epsilon^{-D_i})$ hidden units—an exponential bottleneck unless the target exhibits compositional structure.

### Piecewise-Linear Polyhedral Regions & Exponential Depth Separation

A ReLU network composes continuous piecewise-linear (CPWL) maps, so the full network $f(\boldsymbol{x}; \boldsymbol{\theta})$ is itself CPWL over convex polyhedral cells in $\mathbb{R}^{D_i}$, each corresponding to a fixed binary activation pattern $\boldsymbol{s} \in \{0,1\}^{\sum_\ell n_\ell}$.

> [!important] Linear Region Counting & Depth Separation
> - **1D Shallow vs. Deep Exact Bounds [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 4.5.2, p. 50):** A scalar-input ($D_i = 1$) shallow ReLU network with $D > 2$ hidden units has $3D + 1$ parameters and creates at most $D + 1$ linear regions. A deep ReLU network with $K$ hidden layers of width $D > 2$ has $3D + 1 + (K-1)D(D+1)$ parameters and can create up to
>
> $$
> R_{\max}(1, K, D) = (D + 1)^K \quad \text{linear regions.}
> $$
>
>   For example, $K=5, D=10$ uses only $471$ parameters yet produces up to $11^5 = 161{,}051$ linear regions.
> - **Multivariate Depth Separation (Mont\'ufar et al.) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 6.4.1, Eq. 6.42, p. 200):** For a ReLU network with $d$ inputs, depth $l$, and $n \ge d$ units per hidden layer, the maximum number of linear regions scales as
>
> $$
> R_{\max}(d, l, n) = \Omega\!\left(\left(\frac{n}{d}\right)^{(l-1)d} n^d\right),
> $$
>
>   growing exponentially in depth $l$, whereas a shallow ($l=1$) network is bounded by Zaslavsky's hyperplane arrangement count $\sum_{j=0}^{d} \binom{n}{j} = O(n^d)$.

> [!tip] Input-Space Folding
> Each hidden layer in a deep ReLU (or absolute-value) network folds the input space along the hyperplanes $\boldsymbol{w}_j^{\top} \boldsymbol{h} + b_j = 0$, mapping distinct regions onto overlapping images [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Fig. 6.5, p. 200), [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Fig. 4.3, p. 44). Subsequent layers then slice all folded sheets simultaneously, multiplying the region count across layers rather than adding regions linearly.

### Smooth & Piecewise-Linear Activation Taxonomy

As noted in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 13.2.3, p. 428), if $\sigma$ is linear then Eq. `(eq:mlpkan:mlp_layer)` collapses to a single affine map $\boldsymbol{W}' \boldsymbol{x} + \boldsymbol{b}'$. Classical sigmoids ($\sigma(a) = (1+e^{-a})^{-1}$ and $\tanh(a)$) saturate exponentially as $\lvert a \rvert \to \infty$, causing vanishing gradients. Modern non-saturating activations—detailed in [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 6.2.3, pp. 184–185) and [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Fig. 3.13, p. 37)—include:

$$
\begin{aligned}
\mathrm{ReLU}(a) &= \max(0, a), &
  \mathrm{LeakyReLU}_\alpha(a) &= \max(0, a) + \alpha \min(0, a),  \\
  \mathrm{Softplus}(a) &= \ln(1 + e^a), &
  \mathrm{GELU}(a) &= a \Phi(a),  \\
  \mathrm{SiLU}_\beta(a) &= \frac{a}{1 + e^{-\beta a}}, &
  \sigma(a) &= \frac{1}{1 + e^{-a}},
\end{aligned}
$$

where $\Phi(a)$ is the standard Gaussian CDF and $\beta=1$ gives the standard SiLU/Swish activation used inside KAN residual branches. Unlike ReLU, GELU, SiLU, and Softplus are $C^\infty$ smooth and non-monotonic near $a \in [-2, 0)$ (for GELU/SiLU), improving gradient flow for negative pre-activations.

### Kolmogorov–Arnold Superposition & B-Spline KANs

Rather than approximating functions via sums of ridge functions $\sigma(\boldsymbol{w}_j^{\top} \boldsymbol{x} + \theta_j)$, Kolmogorov and Arnold proved an *exact* decomposition into univariate functions [[Source - Math behind KAN|math_behind_kan]] (PDF p. 2).

> [!important] Kolmogorov–Arnold Representation Theorem (1957)
> Every continuous multivariate function $f : [0,1]^n \to \mathbb{R}$ can be represented exactly as a two-layer superposition of univariate continuous functions:
>
> $$
> f(x_1, \dots, x_n) = \sum_{q=0}^{2n} \Phi_q\!\left(\sum_{p=1}^n \phi_{q,p}(x_p)\right),
> $$
>
> where $\phi_{q,p} : [0,1] \to \mathbb{R}$ and $\Phi_q : \mathbb{R} \to \mathbb{R}$ are continuous univariate functions.

Classical attempts to apply Eq. `(eq:mlpkan:ka_rep)` directly failed because the inner functions $\phi_{q,p}$ can be fractal and non-differentiable, rendering gradient-based optimization intractable [[Source - Math behind KAN|math_behind_kan]] (PDF p. 2). Modern KANs resolve this by (i) generalizing the fixed $[n, 2n+1, 1]$ depth-2 template to arbitrary layer widths $[n_0, n_1, \dots, n_L]$, and (ii) parameterizing each edge $\phi_{\ell,q,p}$ as a smooth combination of a residual basis $b(x) = \mathrm{SiLU}(x)$ and a degree-$k$ B-spline [[Source - Math behind KAN|math_behind_kan]] (PDF p. 6):

$$
\phi(x) = w_b   b(x) + w_s   \mathrm{spline}(x), \qquad
  \mathrm{spline}(x) = \sum_{m=0}^{G+k-1} c_m B_{m,k}(x),
$$

where $w_b, w_s$ control the global residual magnitude and $\{c_m\}$ are learnable spline control coefficients. Given an augmented non-decreasing knot vector $t_0 \le t_1 \le \dots \le t_{G+2k}$ spanning $[a, b]$ with $G$ interior intervals, the B-spline basis functions $B_{m,k}(x)$ are defined recursively via the **Cox–de Boor formula**:

$$
\begin{aligned}
B_{m,0}(x) &= \mathbb{I}[t_m \le x < t_{m+1}],  \\
  B_{m,r}(x) &= \frac{x - t_m}{t_{m+r} - t_m} B_{m,r-1}(x) + \frac{t_{m+r+1} - x}{t_{m+r+1} - t_{m+1}} B_{m+1,r-1}(x), \quad r = 1, \dots, k.
\end{aligned}
$$

Crucially, $B_{m,k}(x)$ has *local support* $[t_m, t_{m+k+1})$ (at most $k+1$ basis functions are nonzero at any $x$), is $C^{k-1}$ smooth on uniform knots, and satisfies the partition-of-unity identity $\sum_{m=0}^{G+k-1} B_{m,k}(x) = 1$ for $x \in [t_k, t_{G+k}]$.

## Algorithmic Implementation & Comparative Synthesis

Unlike MLPs—which require retraining from scratch when increasing hidden width—KANs support **Grid Extension** [[Source - Math behind KAN|math_behind_kan]] (PDF pp. 7–8): a model trained on a coarse grid $G_1$ can be refined to a finer grid $G_2 > G_1$ by solving a local least-squares projection $\min_{\boldsymbol{c}'} \mathbb{E}_{x}\!\left[\left(\sum_j c'_j B_{j,k}^{\text{fine}}(x) - \sum_m c_m B_{m,k}^{\text{coarse}}(x)\right)^2\right]$ on each edge, preserving the learned function while unlocking higher spatial resolution for fine-tuning. Algorithm `alg:mlpkan:kan_forward` details the forward evaluation and grid-extension step.

> [!example] Algorithm: KAN Layer Forward Pass (Cox–de Boor) & Grid Extension
> **Input:** Input $\boldsymbol{h}^{(\ell-1)} \in \mathbb{R}^{n_{\ell-1}}$, knot vector $\boldsymbol{t} \in \mathbb{R}^{G+2k+1}$, spline coefficients $\boldsymbol{C} \in \mathbb{R}^{n_\ell \times n_{\ell-1} \times (G+k)}$, residual weights $\boldsymbol{W}_b, \boldsymbol{W}_s \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$, spline degree $k$.
> **Output:** Layer output $\boldsymbol{h}^{(\ell)} \in \mathbb{R}^{n_\ell}$.
>
> **for** $p \leftarrow 1$ to $n_{\ell-1}$:
>
>   $x \leftarrow h^{(\ell-1)}_p$
>   Initialize $B_{m,0}(x) \leftarrow \mathbb{I}[t_m \le x < t_{m+1}]$ for $m = 0, \dots, G+2k-1$
>   **for** $r \leftarrow 1$ to $k$:
>
>     **for** $m \leftarrow 0$ to $G+2k-1-r$:
>
>       Compute $B_{m,r}(x)$ via Cox–de Boor recursion in Eq. `(eq:mlpkan:cdbr)`
>
>   $b_p \leftarrow x / (1 + e^{-x})$  *// SiLU residual basis*
>
> **for** $q \leftarrow 1$ to $n_\ell$:
>
>   $h^{(\ell)}_q \leftarrow \sum_{p=1}^{n_{\ell-1}} \left[ W_{b,q,p}  b_p + W_{s,q,p} \sum_{m=0}^{G+k-1} C_{q,p,m}  B_{m,k}\!\left(h^{(\ell-1)}_p\right) \right]$
>
> **Optional Grid Extension ($G \to G_{\text{new}}$):** Construct fine knots $\boldsymbol{t}'$, sample collocation points $\{x_s\}_{s=1}^S \subset [t_k, t_{G+k}]$, and solve $\boldsymbol{C}'_{q,p} \leftarrow \arg\min_{\boldsymbol{c}'} \sum_{s=1}^S \left(\sum_j c'_j B^{\text{fine}}_{j,k}(x_s) - \sum_m C_{q,p,m} B_{m,k}(x_s)\right)^2$ via QR/normal equations

**Table:** Per-layer parameter and computational complexity comparison between Shallow MLPs, Deep MLPs, and B-spline KANs ($N = n_{\ell-1} = n_\ell$, spline degree $k$, grid intervals $G$).

| **Architecture** | **Params / Layer** | **FLOPs / Sample** | **Hardware & Inductive Trade-off** |
| --- | --- | --- | --- |
| Shallow MLP ($L=2$) | $N(D_i + D_o + 1)$ | $O(N(D_i + D_o))$ | Dense GEMM; $O(\epsilon^{-D_i})$ width curse |
| Deep MLP ($L$ layers) | $N^2 + N$ | $O(N^2)$ | Dense GEMM; $O((N/D_i)^{L D_i})$ regions |
| KAN ($L$ layers, $G, k$) | $N^2(G + k + 2)$ | $O(N^2 k^2)$ (local) | Symbolic interpretability & grid extension; non-GEMM overhead |

> [!info] Cross-Book Synthesis: Compositional Priors vs. Univariate Edge Plasticity
> [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 6.3.1, p. 187) and [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 6.4.1, p. 201) frame deep MLPs as encoding a *compositional program prior*: intermediate hidden layers extract reusable motifs (such as solving XOR via AND/OR sub-circuits [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 13.2.1, p. 427)) that exponentially multiply linear regions per parameter [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 4.5.2, p. 50). By contrast, [[Source - Math behind KAN|math_behind_kan]] (PDF pp. 6–7) shifts representational capacity from depth/width scaling to *edge-level B-spline plasticity*. When the target admits low-dimensional compositional or physical-law structure, KANs achieve lower approximation error with far fewer nodes and allow post-hoc symbolic regression on individual edges; when high-dimensional unstructured data (e.g., images or token embeddings) dominate, dense GEMM-friendly MLPs remain orders of magnitude faster per FLOP.

## Visual Diagnostics & Geometric Structure

Figure `fig:mlpkan:mlp_vs_kan_tikz` contrasts the computational graphs of an MLP and a KAN, highlighting the duality between fixed node nonlinearities with scalar edge weights and linear node summations with spline-parameterized edges.

> [!example] Visual Diagram (LaTeX / PDF)
> *Architectural duality between an MLP (fixed scalar node activations $\sigma$, linear scalar edge weights) and a KAN (linear summation at nodes, learnable univariate B-spline activations $\phi_{q,p}$ on edges).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:mlpkan:activations_bsplines` plots the canonical activation functions alongside cubic B-spline basis functions $B_{m,3}(x)$ on uniform knots, illustrating both the smooth non-monotonicity of GELU/SiLU and the local support and partition-of-unity property of B-splines.

> [!example] Visual Diagram (LaTeX / PDF)
> ***Left:** Comparison of ReLU, Softplus, SiLU ($\beta=1$), and GELU activations. **Right:** Cubic ($k=3$) B-spline basis functions on $[0,1]$ with clamped boundary knots, demonstrating non-negativity and partition of unity ($\sum_{m=0}^3 B_{m,3}(x) = 1$).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Empirical and theoretical region-counting behavior is captured directly in the source literature: Figure `fig:mlpkan:prince_regions` demonstrates the exponential growth in linear regions per parameter as depth $K$ increases, Figure `fig:mlpkan:murphy_2d_regions` visualizes the resulting polyhedral tessellation of $\mathbb{R}^2$, and Figure `fig:mlpkan:kan_arch` illustrates a two-layer KAN with learned univariate edge splines.

> [!note] Source Figure
> *Figure:* Maximum number of linear regions versus total parameter count for ReLU networks of depth $K \in \{1,\dots,5\}$ with $D_i=1$ and $D_i=10$ inputs, illustrating exponential depth efficiency [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Fig. 4.7, p. 51). (`understandingdl_simonprince_p65_fig1.png`)

> [!note] Source Figure
> *Figure:* Decomposition of $\mathbb{R}^2$ into convex linear decision regions produced by a ReLU MLP with (a) one hidden layer of $25$ units and (b) two hidden layers [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 13.7, p. 435). (`probabilisticml_kevinmurphy_introduction_p465_fig1.png`)

> [!note] Source Figure
> *Figure:* Two-layer KAN architecture ($[2, 5, 1]$): each edge learns a univariate B-spline activation $\phi_{\ell,q,p}$, while hidden and output nodes perform pure summation [[Source - Math behind KAN|math_behind_kan]] (PDF p. 7). (`math_behind_kan_p7_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Dying ReLU & Dead-Zone Collapse
> If a ReLU unit's pre-activation satisfies $\boldsymbol{w}_j^{\top} \boldsymbol{x}_n + b_j < 0$ for every training sample $\boldsymbol{x}_n$ (often triggered by an overly large learning rate or negative bias drift), then $\frac{\partial \mathcal{L}}{\partial \boldsymbol{w}_j} = \boldsymbol{0}$ identically and the unit permanently ceases learning.
> **Remedy:** Use He/Kaiming variance-preserving initialization, replace ReLU with LeakyReLU ($\alpha \in [0.01, 0.2]$), GELU, or SiLU, and apply gradient norm clipping.

> [!warning] Spectral Bias & Shallow Width Explosion
> Standard ReLU MLPs exhibit a strong inductive bias toward low-frequency functions on low-dimensional coordinate domains, failing to capture fine oscillations without sinusoidal activations or positional encodings [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 13.2.3, p. 429). Conversely, relying on shallow UAT widths without compositional depth incurs $O(\epsilon^{-D_i})$ parameter blowup [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 6.1.1, p. 173).

> [!warning] KAN Boundary Extrapolation & GPU Throughput Overhead
> Because B-spline basis functions $B_{m,k}(x)$ vanish identically outside the knot support $[t_k, t_{G+k}]$, any out-of-distribution activation collapses $\mathrm{spline}(x) = 0$, leaving only the residual branch $w_b \mathrm{SiLU}(x)$ [[Source - Math behind KAN|math_behind_kan]] (PDF pp. 4–6).
> **Remedy:** Periodically execute adaptive grid updates (`update_grid`) aligned to empirical activation quantiles during training, and reserve KANs for low-dimensional scientific/symbolic tasks where their $5\times$–$10\times$ non-GEMM wall-clock overhead per parameter is offset by compact node counts.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[Kolmogorov-Arnold Networks (KAN)]], [[MLP & Universal Approximation]]
- **Prerequisites (Directed $\leftarrow$):** [[Automatic Differentiation (AD)]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Factorization Machines & DeepFM]], [[He & Xavier Initialization]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Automatic Differentiation (AD)]], [[Radiance Fields (NeRF & 3DGS)]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Math behind KAN]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
