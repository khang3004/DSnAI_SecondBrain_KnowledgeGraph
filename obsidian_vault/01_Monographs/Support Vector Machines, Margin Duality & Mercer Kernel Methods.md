---
type: monograph
domain: "D02"
parent_topic: "[[PT 02 - Linear Models, Generalization, Kernels & Ensembles]]"
sub_topic: "[[ST 02 - Kernel Methods, Support Vector Machines & Tree Ensembles]]"
cheatsheet: "[[Cheatsheet - Support Vector Machines, Margin Duality & Mercer Kernel Methods]]"
part_of: 
  - "[[ST 02 - Kernel Methods, Support Vector Machines & Tree Ensembles]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
concepts: 
  - "[[Mercer Kernels & RKHS]]"
  - "[[Support Vector Machines (SVM)]]"
prerequisites: 
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Inner Product & Norms]]"
  - "[[Lagrange Duality & KKT]]"
enables: 
  - "[[Gaussian Processes]]"
related: 
  - "[[Gaussian Processes]]"
  - "[[Lagrange Duality & KKT]]"
  - "[[Mercer Kernels & RKHS]]"
  - "[[Support Vector Machines (SVM)]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - monograph
  - domain-02
---

# Support Vector Machines, Margin Duality & Mercer Kernel Methods

> [!abstract] Monograph Navigation
> - **Domain:** [[D02 - Classical, Kernel & Probabilistic Machine Learning|Classical, Kernel & Probabilistic Machine Learning]]
> - **Parent / Sub-Topic:** [[PT 02 - Linear Models, Generalization, Kernels & Ensembles|Linear Models, Generalization, Kernels & Ensembles]] → [[ST 02 - Kernel Methods, Support Vector Machines & Tree Ensembles|Kernel Methods, Support Vector Machines & Tree Ensembles]]
> - **Visual Cheatsheet:** [[Cheatsheet - Support Vector Machines, Margin Duality & Mercer Kernel Methods]]
> - **Description:** Derives maximum-margin hard/soft SVMs via hinge loss and Lagrangian duality, characterizes support vectors via KKT complementary slackness, proves the Representer Theorem, and connects RKHS Mercer kernels.

## Problem Formulation & Unified Notation

Consider a binary classification dataset $\mathcal{D} = \{(\boldsymbol{x}_i, y_i)\}_{i=1}^N$ with inputs $\boldsymbol{x}_i \in \mathcal{X} \subseteq \mathbb{R}^D$ and bipolar class labels $y_i \in \{-1, +1\}$. A linear discriminant partitions $\mathbb{R}^D$ via an affine hyperplane $\mathcal{P} = \{\boldsymbol{x} \in \mathbb{R}^D : f(\boldsymbol{x}) = 0\}$, where the decision function is parameterized by a normal vector $\boldsymbol{w} \in \mathbb{R}^D$ and scalar offset (bias) $b \in \mathbb{R}$:

$$
f(\boldsymbol{x}) = \langle \boldsymbol{w}, \boldsymbol{x} \rangle + b, \qquad \hat{y}(\boldsymbol{x}) = \mathrm{sign}\!\big(f(\boldsymbol{x})\big).
$$

{
Unlike probabilistic classifiers that model $p(y \mid \boldsymbol{x})$ directly, the Support Vector Machine (SVM) maximizes the geometric separation between classes, yielding a sparse dual governed by inner products $\langle \boldsymbol{x}_i, \boldsymbol{x}_j \rangle$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 12.1, p. 372) and [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 5.7.2, p. 141). Replacing Euclidean inner products with positive-definite Mercer kernels $k(\boldsymbol{x}_i, \boldsymbol{x}_j) = \langle \phi(\boldsymbol{x}_i), \phi(\boldsymbol{x}_j) \rangle_{\mathcal{H}_k}$ lifts this linear maximum-margin geometry into an implicit Reproducing Kernel Hilbert Space (RKHS) $\mathcal{H}_k$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.1, p. 567).
}

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> | **Concept** | **Deisenroth et al.** | **Murphy** | **Tiep Vu** | **Unified Monograph Convention** |
> | — | — | — | — | — |
> | Class label | $y_n \in \{-1, +1\}$ | $\tilde{y}_n \in \{-1, +1\}$ | $y_n \in \{-1, +1\}$ | $y_i \in \{-1, +1\}$ |
> | Affine offset | $b$ | $w_0$ | $b$ | $b \in \mathbb{R}$ |
> | Dual multiplier | $\alpha_n \ge 0$ | $\alpha_n \ge 0$ | $\lambda_n \ge 0$ | $\alpha_i \in [0, C]$ |
> | Slack multiplier | $\gamma_n \ge 0$ | $\mu_n \ge 0$ | $\mu_n \ge 0$ | $\gamma_i \ge 0$ |
> | Regularization | $C$ (slack weight) | $C$ or $\lambda = (2C)^{-1}$ | $C$ (slack weight) | $C > 0$; $\lambda = (2C)^{-1}$ in ERM |
> | Feature map | $\phi(\boldsymbol{x})$ | $\phi(\boldsymbol{x})$ | $\Phi(\boldsymbol{x})$ | $\phi : \mathcal{X} \to \mathcal{H}_k$ |
> | Kernel function | $k(\boldsymbol{x}_i, \boldsymbol{x}_j)$ | $K(\boldsymbol{x}, \boldsymbol{x}')$ | $k(\boldsymbol{x}, \boldsymbol{z})$ | $k(\boldsymbol{x}, \boldsymbol{x}')$; Gram matrix $\boldsymbol{K}$ |
>
> 
>
>
> **Pedagogical Synthesis:** [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 12, pp. 372–392) unifies the geometric margin, hinge-loss ERM, convex-hull duality, and standard quadratic programming (QP) matrix forms. [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 17, pp. 567–595) embeds SVMs inside Mercer kernel theory, comparing them with Gaussian Processes, Kernel Ridge Regression (KRR), and $\nu$-SVMs. [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Part VIII, pp. 329–378) provides step-by-step algebraic KKT and subgradient derivations, while [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 5.7.2, pp. 141–143) frames the kernel trick as template matching and contrasts its $\mathcal{O}(N^2)$ scaling against learned deep representations.

## Core Mathematical Derivations & Theorems

### Hard-Margin & Soft-Margin Primal Formulations

> [!important] Hard-Margin Primal & Canonical Hyperplane Normalization
> Let $\mathcal{D}$ be linearly separable. For any point $\boldsymbol{x}_i$, its orthogonal projection $\boldsymbol{x}_{\perp}$ onto $\mathcal{P} = \{\boldsymbol{x} : \langle \boldsymbol{w}, \boldsymbol{x} \rangle + b = 0\}$ satisfies $\boldsymbol{x}_i = \boldsymbol{x}_{\perp} + r_i \frac{\boldsymbol{w}}{\lVert \boldsymbol{w} \rVert_2}$, where the signed distance is $r_i = \frac{\langle \boldsymbol{w}, \boldsymbol{x}_i \rangle + b}{\lVert \boldsymbol{w} \rVert_2}$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 12.2.1, p. 375) [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 26.2, p. 330). Because $y_i$ matches the sign of $f(\boldsymbol{x}_i)$ for correctly classified points, the geometric margin is
>
> $$
> \rho(\boldsymbol{w}, b) = \min_{i=1,\dots,N} \frac{y_i (\langle \boldsymbol{w}, \boldsymbol{x}_i \rangle + b)}{\lVert \boldsymbol{w} \rVert_2}.
> $$
>
> Since $(\boldsymbol{w}, b)$ and $(\kappa\boldsymbol{w}, \kappa b)$ ($\kappa > 0$) define the identical hyperplane, we fix the scale by requiring the closest training points to satisfy $\min_i y_i(\langle \boldsymbol{w}, \boldsymbol{x}_i \rangle + b) = 1$. Maximizing $\rho = 1/\lVert \boldsymbol{w} \rVert_2$ is then equivalent to the strictly convex quadratic program:
>
> $$
> \min_{\boldsymbol{w}, b}   \frac{1}{2}\lVert \boldsymbol{w} \rVert_2^2 \qquad \text{subject to} \quad y_i(\langle \boldsymbol{w}, \boldsymbol{x}_i \rangle + b) \ge 1, \quad \forall i \in \{1,\dots,N\}.
> $$
>
> The slab between the two canonical margin hyperplanes $\langle \boldsymbol{w}, \boldsymbol{x} \rangle + b = \pm 1$ has total width $2 / \lVert \boldsymbol{w} \rVert_2$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.3.1, p. 585).

When $\mathcal{D}$ is non-separable or contains label noise, `(eq:svm:hard_primal)` is infeasible. Introducing non-negative slack variables $\xi_i \ge 0$ relaxes the margin constraint to $y_i(\langle \boldsymbol{w}, \boldsymbol{x}_i \rangle + b) \ge 1 - \xi_i$, yielding the $C$-SVM soft-margin primal [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 12.2.4, p. 379) [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 27.2, p. 340):

$$
\min_{\boldsymbol{w}, b, \boldsymbol{\xi}}   \frac{1}{2}\lVert \boldsymbol{w} \rVert_2^2 + C \sum_{i=1}^N \xi_i
  \qquad \text{subject to} \quad y_i(\langle \boldsymbol{w}, \boldsymbol{x}_i \rangle + b) \ge 1 - \xi_i,    \xi_i \ge 0.
$$

> [!important] Hinge Loss Equivalence & $\nu$-SVM Parameterization
> For fixed $(\boldsymbol{w}, b)$, minimizing `(eq:svm:soft_primal)` over $\xi_i$ sets $\xi_i^* = \max\big(0, 1 - y_i(\langle \boldsymbol{w}, \boldsymbol{x}_i \rangle + b)\big)$. Substituting $\xi_i^*$ eliminates the constraints and reveals that soft-margin SVM is $\ell_2$-regularized empirical risk minimization under the **hinge loss** $\ell_{\mathrm{hinge}}(t) = \max(0, 1 - t)$ with $t = y_i f(\boldsymbol{x}_i)$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 12.2.5, p. 381) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.3.6, p. 591):
>
> $$
> \min_{\boldsymbol{w}, b}   \frac{1}{2C}\lVert \boldsymbol{w} \rVert_2^2 + \sum_{i=1}^N \max\!\big(0, 1 - y_i(\langle \boldsymbol{w}, \boldsymbol{x}_i \rangle + b)\big).
> $$
>
> Because $\xi_i > 1$ whenever $y_i f(\boldsymbol{x}_i) < 0$, $\sum_{i=1}^N \xi_i$ is a convex upper bound on the number of training misclassifications. An alternative parameterization, the $\nu$-SVM ($\nu \in (0, 1]$), replaces $C$ by optimizing $\frac{1}{2}\lVert \boldsymbol{w} \rVert_2^2 - \nu \rho + \frac{1}{N}\sum_{i=1}^N \xi_i$ subject to $y_i f(\boldsymbol{x}_i) \ge \rho - \xi_i$, where $\nu$ simultaneously upper-bounds the fraction of margin errors and lower-bounds the fraction of support vectors [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.3.3, p. 590).

### Wolfe Lagrangian Dual & KKT Complementary Slackness

> [!important] Primal-to-Wolfe Dual Derivation & KKT Classification
> Introducing Lagrange multipliers $\alpha_i \ge 0$ for the margin constraints and $\gamma_i \ge 0$ for $\xi_i \ge 0$, the Lagrangian of `(eq:svm:soft_primal)` is [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 12.3.1, p. 383):
>
> $$
> \mathcal{L}(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\gamma}) = \frac{1}{2}\lVert \boldsymbol{w} \rVert_2^2 + C\sum_{i=1}^N \xi_i - \sum_{i=1}^N \alpha_i \big[y_i(\langle \boldsymbol{w}, \boldsymbol{x}_i \rangle + b) - 1 + \xi_i\big] - \sum_{i=1}^N \gamma_i \xi_i.
> $$
>
> Since `(eq:svm:soft_primal)` is convex with affine constraints, Slater's condition holds (take $\boldsymbol{w} = \boldsymbol{0}, b = 0, \xi_i = 2$), guaranteeing strong duality. Setting primal gradients to zero gives the stationarity conditions:
>
> $$
> \begin{aligned}
> \nabla_{\boldsymbol{w}} \mathcal{L} = \boldsymbol{0} &\implies \boldsymbol{w}^* = \sum_{i=1}^N \alpha_i y_i \boldsymbol{x}_i,  \\
>   \frac{\partial \mathcal{L}}{\partial b} = 0 &\implies \sum_{i=1}^N \alpha_i y_i = 0,  \\
>   \frac{\partial \mathcal{L}}{\partial \xi_i} = 0 &\implies C - \alpha_i - \gamma_i = 0 \implies 0 \le \alpha_i \le C \quad (\text{since } \gamma_i \ge 0).
> \end{aligned}
> $$
>
> Substituting `(eq:svm:kkt_w)`–`(eq:svm:kkt_xi)` back into `(eq:svm:lagrangian)` eliminates all primal variables and yields the **Wolfe dual QP** [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Eq. (12.41), p. 385) [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 27.3, p. 343):
>
> $$
> \max_{\boldsymbol{\alpha} \in [0, C]^N}   \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j \langle \boldsymbol{x}_i, \boldsymbol{x}_j \rangle
>   \qquad \text{subject to} \quad \sum_{i=1}^N \alpha_i y_i = 0.
> $$
>
> At optimality, the Karush–Kuhn–Tucker (KKT) complementary slackness conditions require
>
> $$
> \alpha_i \big[y_i(\langle \boldsymbol{w}^*, \boldsymbol{x}_i \rangle + b^*) - 1 + \xi_i^*\big] = 0, \qquad (C - \alpha_i)\xi_i^* = 0.
> $$
>
> These partition the dataset into three disjoint regimes [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.3.3, p. 589) [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 27.3, p. 344):
> - $\alpha_i = 0$ (**Non-support vectors**): $\gamma_i = C > 0 \implies \xi_i^* = 0$ and $y_i f(\boldsymbol{x}_i) \ge 1$. These points lie strictly outside or on the safe margin and do not enter $\boldsymbol{w}^*$.
> - $0 < \alpha_i < C$ (**Free / unbounded support vectors**, index set $\mathcal{M}$): $\gamma_i > 0 \implies \xi_i^* = 0$, and $\alpha_i > 0 \implies y_i f(\boldsymbol{x}_i) = 1$. They lie *exactly* on the margin hyperplanes and determine the optimal bias via averaging for numerical stability:
>
> $$
> b^* = \frac{1}{\lvert \mathcal{M} \rvert} \sum_{m \in \mathcal{M}} \left( y_m - \sum_{i \in \mathcal{S}} \alpha_i y_i \langle \boldsymbol{x}_i, \boldsymbol{x}_m \rangle \right), \qquad \mathcal{S} = \{i : \alpha_i > 0\}.
> $$
> - $\alpha_i = C$ (**Bounded support vectors**): $\gamma_i = 0$, permitting $\xi_i^* \ge 0$ and $y_i f(\boldsymbol{x}_i) = 1 - \xi_i^* \le 1$. Points with $0 < \xi_i^* \le 1$ fall inside the margin yet remain correctly classified; points with $\xi_i^* > 1$ are misclassified.

> [!tip] Dual Convex Hulls & Reduced Hulls
> Geometrically, the hard-margin dual minimizes $\frac{1}{2}\lVert \sum_{y_i=+1}\alpha_i \boldsymbol{x}_i - \sum_{y_j=-1}\alpha_j \boldsymbol{x}_j \rVert_2^2$ subject to $\sum_{y_i=+1}\alpha_i = \sum_{y_j=-1}\alpha_j = \text{const}$, which is equivalent to finding the closest pair of points between the convex hulls of the positive and negative classes [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 12.3.2, p. 386). Bounding $\alpha_i \le C$ in the soft-margin dual shrinks each convex hull to a *reduced convex hull*, preventing outlier points from dominating the separating hyperplane.

### Mercer's Theorem, RKHS & The Representer Theorem

Because `(eq:svm:dual_qp)` and `(eq:svm:bias_recovery)` depend on inputs solely through pairwise inner products $\langle \boldsymbol{x}_i, \boldsymbol{x}_j \rangle$, replacing $\boldsymbol{x}_i \mapsto \phi(\boldsymbol{x}_i) \in \mathcal{H}_k$ allows us to evaluate $k(\boldsymbol{x}_i, \boldsymbol{x}_j) = \langle \phi(\boldsymbol{x}_i), \phi(\boldsymbol{x}_j) \rangle_{\mathcal{H}_k}$ directly without ever instantiating $\phi(\boldsymbol{x}_i)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.3.4, p. 590) [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 28.2, p. 358).

> [!important] Mercer's Theorem & Positive-Semidefinite Gram Matrices
> A symmetric function $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a valid **Mercer kernel** (positive-definite kernel) if and only if for every finite set $\{\boldsymbol{x}_1, \dots, \boldsymbol{x}_N\} \subset \mathcal{X}$, the $N \times N$ Gram matrix $\boldsymbol{K}$ with entries $K_{ij} = k(\boldsymbol{x}_i, \boldsymbol{x}_j)$ is symmetric positive semidefinite (PSD):
>
> $$
> \boldsymbol{c}^{\top} \boldsymbol{K} \boldsymbol{c} = \sum_{i=1}^N \sum_{j=1}^N c_i c_j k(\boldsymbol{x}_i, \boldsymbol{x}_j) \ge 0 \qquad \forall \boldsymbol{c} \in \mathbb{R}^N.
> $$
>
> On a compact domain $\mathcal{X}$ with finite measure $\mu$, Mercer's theorem guarantees a spectral expansion $k(\boldsymbol{x}, \boldsymbol{x}') = \sum_{m=1}^\infty \lambda_m \psi_m(\boldsymbol{x}) \psi_m(\boldsymbol{x}')$ with eigenvalues $\lambda_m \ge 0$ and orthonormal eigenfunctions $\{\psi_m\}$, inducing the explicit feature map $\phi(\boldsymbol{x}) = (\sqrt{\lambda_m}\psi_m(\boldsymbol{x}))_{m=1}^\infty \in \ell_2$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.1.1, p. 568). Crucially, PSDness of $\boldsymbol{K}$ implies $\boldsymbol{Y}\boldsymbol{K}\boldsymbol{Y}$ (with $\boldsymbol{Y} = \mathrm{diag}(\boldsymbol{y})$) is PSD, ensuring the dual QP `(eq:svm:dual_qp)` is concave in $\boldsymbol{\alpha}$ and globally solvable [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 28.3.1, p. 359).

Canonical Mercer kernels include the polynomial kernel $k(\boldsymbol{x}, \boldsymbol{x}') = (\langle \boldsymbol{x}, \boldsymbol{x}' \rangle + c_0)^d$ ($c_0 \ge 0, d \in \mathbb{N}$) and the Gaussian Radial Basis Function (RBF) / squared-exponential kernel:

$$
k_{\mathrm{RBF}}(\boldsymbol{x}, \boldsymbol{x}') = \exp\!\left(-\frac{\lVert \boldsymbol{x} - \boldsymbol{x}' \rVert_2^2}{2\ell^2}\right) = \exp\!\left(-\gamma \lVert \boldsymbol{x} - \boldsymbol{x}' \rVert_2^2\right), \qquad \gamma = \frac{1}{2\ell^2},
$$

whose implicit feature space is infinite-dimensional [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. (17.3), p. 568) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Eq. (5.84), p. 142).

> [!important] The Representer Theorem (Sch\"olkopf–Herbrich–Smola)
> Let $\mathcal{H}_k$ be the unique RKHS associated with Mercer kernel $k$ satisfying the reproducing property $\langle f, k(\cdot, \boldsymbol{x}) \rangle_{\mathcal{H}_k} = f(\boldsymbol{x})$ for all $f \in \mathcal{H}_k$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 12.4, p. 389). Consider any regularized empirical risk functional with strictly increasing penalty $\Omega : [0, \infty) \to \mathbb{R}$:
>
> $$
> \min_{f \in \mathcal{H}_k}   J(f) = \sum_{i=1}^N \ell\big(y_i, f(\boldsymbol{x}_i)\big) + \Omega\!\left(\lVert f \rVert_{\mathcal{H}_k}\right).
> $$
>
> Decompose $f = f_{\parallel} + f_{\perp}$, where $f_{\parallel} \in \mathrm{span}\{k(\cdot, \boldsymbol{x}_1), \dots, k(\cdot, \boldsymbol{x}_N)\}$ and $f_{\perp} \perp f_{\parallel}$ in $\mathcal{H}_k$. By the reproducing property,
>
> $$
> f(\boldsymbol{x}_i) = \langle f_{\parallel} + f_{\perp}, k(\cdot, \boldsymbol{x}_i) \rangle_{\mathcal{H}_k} = \langle f_{\parallel}, k(\cdot, \boldsymbol{x}_i) \rangle_{\mathcal{H}_k} = f_{\parallel}(\boldsymbol{x}_i),
> $$
>
> so the empirical loss is invariant to $f_{\perp}$. Meanwhile, $\lVert f \rVert_{\mathcal{H}_k}^2 = \lVert f_{\parallel} \rVert_{\mathcal{H}_k}^2 + \lVert f_{\perp} \rVert_{\mathcal{H}_k}^2 \ge \lVert f_{\parallel} \rVert_{\mathcal{H}_k}^2$, with strict inequality whenever $f_{\perp} \ne 0$. Hence any minimizer $f^*$ of `(eq:svm:representer_obj)` lies strictly in the finite-dimensional span of the training kernels:
>
> $$
> f^*(\boldsymbol{x}) = \sum_{i=1}^N \beta_i k(\boldsymbol{x}_i, \boldsymbol{x}), \qquad \beta_i \in \mathbb{R}.
> $$
>
> For the SVM, $\beta_i = \alpha_i y_i$ and KKT sparsity zeroes out all $\alpha_i$ outside the support set $\mathcal{S}$.

## Algorithmic Implementation & Comparative Synthesis

Solving the dual QP `(eq:svm:dual_qp)` via generic interior-point methods requires storing the dense $N \times N$ matrix $\boldsymbol{Y}\boldsymbol{K}\boldsymbol{Y}$ in $\mathcal{O}(N^2)$ memory and $\mathcal{O}(N^3)$ time [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.3.4, p. 590). Because the linear equality constraint $\sum_{i=1}^N \alpha_i y_i = 0$ prevents updating a single $\alpha_i$ independently, Platt's **Sequential Minimal Optimization (SMO)** decomposes the QP into analytical two-variable subproblems $(\alpha_i, \alpha_j)$ that preserve $\alpha_i y_i + \alpha_j y_j = \text{const}$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 12.5, p. 392).

> [!example] Algorithm: Sequential Minimal Optimization (SMO) Two-Variable Dual Solver
> **Input:** Dataset $\{(\boldsymbol{x}_i, y_i)\}_{i=1}^N$, Mercer kernel $k(\cdot,\cdot)$, box bound $C > 0$, tolerance $\varepsilon > 0$.
> **Output:** Dual multipliers $\boldsymbol{\alpha} \in [0, C]^N$, bias $b \in \mathbb{R}$.
> Initialize $\boldsymbol{\alpha} \leftarrow \boldsymbol{0}$, $b \leftarrow 0$, error cache $E_i \leftarrow -y_i$ for all $i$
> **while** any $(\boldsymbol{x}_i, \alpha_i)$ violates KKT conditions within $\varepsilon$:
>
>   Select first multiplier $i$ violating KKT (prioritizing $0 < \alpha_i < C$)
>   Select second multiplier $j \ne i$ maximizing $\lvert E_i - E_j \rvert$
>   Compute box bounds $[L, H]$ for $\alpha_j^{\mathrm{new}}$ preserving $\alpha_i y_i + \alpha_j y_j = \text{const}$:
>   \[
>     L = \begin{cases} \max(0, \alpha_j - \alpha_i) & y_i \ne y_j \\ \max(0, \alpha_i + \alpha_j - C) & y_i = y_j \end{cases}, \qquad
>     H = \begin{cases} \min(C, C + \alpha_j - \alpha_i) & y_i \ne y_j \\ \min(C, \alpha_i + \alpha_j) & y_i = y_j \end{cases};
>   \]
>   Compute curvature $\eta \leftarrow k(\boldsymbol{x}_i, \boldsymbol{x}_i) + k(\boldsymbol{x}_j, \boldsymbol{x}_j) - 2k(\boldsymbol{x}_i, \boldsymbol{x}_j)$
>   **if** $\eta > 0$:
>
>     $\alpha_j^{\mathrm{unc}} \leftarrow \alpha_j + \frac{y_j (E_i - E_j)}{\eta}$
>     $\alpha_j^{\mathrm{new}} \leftarrow \min\big(H, \max(L, \alpha_j^{\mathrm{unc}})\big)$
>     $\alpha_i^{\mathrm{new}} \leftarrow \alpha_i + y_i y_j (\alpha_j - \alpha_j^{\mathrm{new}})$
>     Update $b$ via KKT margin condition on $(i, j)$ and refresh error cache $E_k = f(\boldsymbol{x}_k) - y_k$

**Table:** Computational Complexity & Structural Comparison Across Kernel and Linear Solvers

| **Method** | **Train Time** | **Memory** | **Inference** | **Sparsity** | **Primary Regime** |
| — | — | — | — | — | — |
| Primal Subgradient (Linear) | $\mathcal{O}(T N D)$ | $\mathcal{O}(D)$ | $\mathcal{O}(D)$ | Dense $\boldsymbol{w}$ | Large $N$, linear $k$ |
| Dual Interior-Point QP | $\mathcal{O}(N^3)$ | $\mathcal{O}(N^2)$ | $\mathcal{O}(N_{\mathrm{sv}} D)$ | Sparse $\boldsymbol{\alpha}$ | Small $N$ ($N \lesssim 5\times 10^3$) |
| SMO Decomposition (LIBSVM) | $\mathcal{O}(N_{\mathrm{sv}} N)$–$\mathcal{O}(N^2)$ | $\mathcal{O}(N_{\mathrm{cache}})$ | $\mathcal{O}(N_{\mathrm{sv}} D)$ | Sparse $\boldsymbol{\alpha}$ | Medium $N$, nonlinear $k$ |
| Kernel Ridge Regression | $\mathcal{O}(N^3)$ | $\mathcal{O}(N^2)$ | $\mathcal{O}(N D)$ | Dense $\boldsymbol{\alpha}$ | Closed-form regression |

> [!info] Cross-Book Synthesis: Kernel Ridge Regression vs. SVM vs. Deep Feature Learning
> - **KRR vs. SVM Sparsity** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.3.9–17.3.10, pp. 594–595): Kernel Ridge Regression solves $\boldsymbol{\alpha} = (\boldsymbol{K} + \lambda \boldsymbol{I}_N)^{-1}\boldsymbol{y}$ in closed form, but the squared loss produces a completely *dense* $\boldsymbol{\alpha}$ requiring $\mathcal{O}(N)$ kernel evaluations at test time. Replacing squared loss with hinge loss (classification) or $\epsilon$-insensitive loss (SVR) introduces a flat zero-gradient region that induces exact KKT sparsity ($\alpha_i = 0$ for well-classified points).
> - **Generic Kernels vs. Learned Representations** [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 5.7.2, pp. 142–143): An RBF kernel SVM behaves like smooth template matching over training exemplars. When the target function varies rapidly across high-dimensional manifolds (e.g., natural images), local Euclidean similarity struggles to generalize without exponentially many templates, motivating deep neural networks that learn task-adaptive feature maps $\phi_{\theta}(\boldsymbol{x})$ directly.

## Visual Diagnostics & Geometric Structure

Figure `fig:svm:primal_rkhs` illustrates the geometric duality between the input-space soft-margin slab and its kernel-induced linear separation in $\mathcal{H}_k$, while Figure `fig:svm:losses_and_complexity` contrasts the hinge loss against alternative classification surrogates and plots the RBF bandwidth–regularization trade-off.

> [!example] Visual Diagram (LaTeX / PDF)
> *Geometric duality of the soft-margin hyperplane in input space $\mathcal{X}$ (with free support vectors on $\pm 1$ and bounded support vectors incurring slack $\xi_i > 0$) lifted via $\phi(\boldsymbol{x})$ into an RKHS $\mathcal{H}_k$ where classes become linearly separable.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *(a) Comparison of 0–1 loss, hinge loss, squared hinge loss, and scaled logistic log-loss; only hinge-based losses vanish identically for $t \ge 1$, producing sparse support sets. (b) Increasing RBF precision $\gamma$ shrinks the template radius $\ell$, driving $N_{\mathrm{sv}}/N \to 1$ and creating a U-shaped generalization curve.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

{
The extracted source diagrams below reinforce three geometric diagnostics from Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.3, pp. 586–593): Figure `fig:svm:murphy_geom` visualizes the orthogonal projection $r = f(\boldsymbol{x})/\lVert \boldsymbol{w} \rVert_2$ and slack regimes ($\xi = 0$, $0 < \xi < 1$, $\xi > 1$); Figure `fig:svm:murphy_scaling` demonstrates how unscaled features distort the Euclidean margin; and Figure `fig:svm:murphy_rbf_grid` illustrates the joint $(C, \gamma)$ regularization surface on non-linearly separable moons data.
}

> [!note] Source Figure
> *Figure:* Geometric interpretation of the linear decision boundary $f(\boldsymbol{x}) = \boldsymbol{w}^{\top}\boldsymbol{x} + w_0$ with orthogonal distance $r = f(\boldsymbol{x})/\lVert \boldsymbol{w} \rVert$ (left) and soft-margin slack regimes $\xi = 0$, $\xi < 1$, and $\xi > 1$ (right). Reproduced from [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 17.13, p. 586). (`probabilisticml_kevinmurphy_introduction_p616_fig1.png`)

> [!note] Source Figure
> *Figure:* Sensitivity of the $\ell_2$ margin to feature scaling: when $x_1$ spans $[0, 80]$ and $x_2$ spans $[0, 6]$, the unscaled SVM ignores $x_2$ (left), whereas standardization restores the balanced maximum-margin separator (right). Reproduced from [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 17.14, p. 587). (`probabilisticml_kevinmurphy_introduction_p617_fig1.png`)

> [!note] Source Figure
> *Figure:* Joint effect of RBF precision $\gamma = (2\sigma^2)^{-1}$ and slack penalty $C$ on the two-moons dataset: small $\gamma$ underfits, while large $\gamma$ with large $C$ forms tight local islands around individual support vectors. Reproduced from [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 17.17, p. 593). (`probabilisticml_kevinmurphy_introduction_p623_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Quadratic Memory & $\mathcal{O}(N^3)$ / $\mathcal{O}(N_{\mathrm{sv}} N)$ Scaling Bottleneck
> Standard dual QP and SMO solvers scale between $\mathcal{O}(N_{\mathrm{sv}} N)$ and $\mathcal{O}(N^3)$ in training time and require $\mathcal{O}(N^2)$ memory if the full Gram matrix $\boldsymbol{K}$ is materialized [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.3.4, p. 590) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 5.7.2, p. 143). When $N \gg 10^5$, kernel SVMs become computationally prohibitive. **Remedy:** For linear kernels, optimize the unconstrained primal `(eq:svm:hinge_erm)` directly via stochastic subgradient descent (e.g., Pegasos) or dual coordinate descent in $\mathcal{O}(ND)$; for nonlinear kernels, approximate $\boldsymbol{K}$ via low-rank Nystr\"om landmarks or Random Fourier Features.

> [!warning] Unscaled Features & RBF Bandwidth Collapse
> Both the primal regularizer $\lVert \boldsymbol{w} \rVert_2^2$ and stationary kernels $k(\boldsymbol{x}, \boldsymbol{x}') = \exp(-\gamma \lVert \boldsymbol{x} - \boldsymbol{x}' \rVert_2^2)$ assume isotropic Euclidean geometry. If one feature has variance $10^4\times$ larger than others, it dominates $\lVert \boldsymbol{x} - \boldsymbol{x}' \rVert_2^2$, forcing $\gamma$ to either collapse all kernel similarities to $0$ or ignore informative low-variance coordinates (Figure `fig:svm:murphy_scaling`). **Remedy:** Always standardize features to zero mean and unit variance (or $[0, 1]$ range) on the training fold before fitting, and tune $(C, \gamma)$ jointly on a logarithmic grid [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.3.8, p. 593).

> [!warning] Non-PSD (Indefinite) Kernels & Non-Convex Duals
> Using a similarity function that violates Mercer's condition `(eq:svm:mercer_psd)`—such as the hyperbolic tangent (sigmoid) kernel $k(\boldsymbol{x}, \boldsymbol{x}') = \tanh(\kappa \langle \boldsymbol{x}, \boldsymbol{x}' \rangle + c_0)$ for certain $(\kappa, c_0)$—produces a Gram matrix $\boldsymbol{K}$ with negative eigenvalues [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 28.3.1, p. 359). Consequently, the dual objective `(eq:svm:dual_qp)` ceases to be concave, strong duality fails, and SMO may stall or diverge. **Remedy:** Restrict to proven Mercer kernels (Linear, Polynomial with $c_0 \ge 0$, RBF, Mat\'ern) or verify $\lambda_{\min}(\boldsymbol{K}) \ge -\epsilon_{\mathrm{mach}}$.

> [!warning] Uncalibrated Margins vs. Platt Scaling
> The raw SVM margin $f(\boldsymbol{x})$ is an uncalibrated geometric score, not a log-odds ratio $\log \frac{p(y=1\mid\boldsymbol{x})}{p(y=-1\mid\boldsymbol{x})}$, and often exhibits severe distortion even when the decision boundary is Bayes-optimal [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.3.5, p. 591). Applying Platt scaling $p(y=1 \mid \boldsymbol{x}) = \sigma(a f(\boldsymbol{x}) + b)$ directly on the training set causes severe overfitting because training margins are biased away from $[-1, 1]$. **Remedy:** Fit $(a, b)$ via maximum likelihood exclusively on a held-out validation set or via out-of-fold cross-validation.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[Mercer Kernels & RKHS]], [[Support Vector Machines (SVM)]]
- **Prerequisites (Directed $\leftarrow$):** [[Eigendecomposition & Spectral Theorem]], [[Inner Product & Norms]], [[Lagrange Duality & KKT]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Gaussian Processes]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Gaussian Processes]], [[Lagrange Duality & KKT]], [[Mercer Kernels & RKHS]], [[Support Vector Machines (SVM)]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
