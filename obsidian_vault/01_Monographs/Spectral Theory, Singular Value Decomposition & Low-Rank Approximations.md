---
type: monograph
domain: "D01"
parent_topic: "[[PT 01 - Linear Algebra, Matrix Decompositions & Vector Calculus]]"
sub_topic: "[[ST 01 - Vector Spaces, Spectral Theory & Matrix Factorizations]]"
cheatsheet: "[[Cheatsheet - Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]"
part_of: 
  - "[[ST 01 - Vector Spaces, Spectral Theory & Matrix Factorizations]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
concepts: 
  - "[[Condition Number]]"
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Low-Rank Approximation (Eckart-Young)]]"
  - "[[Moore-Penrose Pseudoinverse]]"
  - "[[Singular Value Decomposition (SVD)]]"
prerequisites: 
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Inner Product & Norms]]"
  - "[[Orthogonal Projection]]"
  - "[[Singular Value Decomposition (SVD)]]"
  - "[[Vector Space & Subspaces]]"
enables: 
  - "[[Camera Calibration & Homography]]"
  - "[[Condition Number]]"
  - "[[Double Descent]]"
  - "[[Gradient Descent]]"
  - "[[Linear & Ridge Regression]]"
  - "[[LoRA & QLoRA]]"
  - "[[Low-Rank Approximation (Eckart-Young)]]"
  - "[[Matrix Factorization (ALS & SGD)]]"
  - "[[Mercer Kernels & RKHS]]"
  - "[[Momentum & Nesterov NAG]]"
  - "[[Moore-Penrose Pseudoinverse]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
  - "[[Principal Component Analysis (PCA)]]"
  - "[[Singular Value Decomposition (SVD)]]"
related: 
  - "[[Condition Number]]"
  - "[[Double Descent]]"
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Gradient Descent]]"
  - "[[Linear & Ridge Regression]]"
  - "[[LoRA & QLoRA]]"
  - "[[Low-Rank Approximation (Eckart-Young)]]"
  - "[[Momentum & Nesterov NAG]]"
  - "[[Moore-Penrose Pseudoinverse]]"
  - "[[Principal Component Analysis (PCA)]]"
  - "[[QR Factorization]]"
  - "[[Singular Value Decomposition (SVD)]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - E D I T I O N]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - monograph
  - domain-01
---

# Spectral Theory, Singular Value Decomposition & Low-Rank Approximations

> [!abstract] Monograph Navigation
> - **Domain:** [[D01 - Mathematical Foundations for Machine Intelligence|Mathematical Foundations for Machine Intelligence]]
> - **Parent / Sub-Topic:** [[PT 01 - Linear Algebra, Matrix Decompositions & Vector Calculus|Linear Algebra, Matrix Decompositions & Vector Calculus]] → [[ST 01 - Vector Spaces, Spectral Theory & Matrix Factorizations|Vector Spaces, Spectral Theory & Matrix Factorizations]]
> - **Visual Cheatsheet:** [[Cheatsheet - Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
> - **Description:** Unifies eigendecomposition of symmetric matrices, spectral theorem, full/thin SVD, Eckart-Young-Mirsky best rank-k approximation, and data-compression interpretations.

## Problem Formulation & Unified Notation

Matrix factorizations expose the invariant geometric and algebraic structure of linear operators independent of the canonical coordinate axes in which data or weights happen to be recorded. While a square non-defective matrix $\boldsymbol{A} \in \mathbb{R}^{n \times n}$ admits an eigendecomposition (EVD) $\boldsymbol{A} = \boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}$ over $\mathbb{C}$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 4.4, p. 116), general matrices in machine learning—rectangular feature tables $\boldsymbol{X} \in \mathbb{R}^{N \times D}$, weight layers $\boldsymbol{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, Jacobian operators, and defective transition matrices—need neither be square nor possess $n$ linearly independent eigenvectors. The Spectral Theorem resolves this restriction for real symmetric matrices ($\boldsymbol{A} = \boldsymbol{A}^{\top}$), guaranteeing real eigenvalues and an orthonormal eigenbasis $\boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\top}$. The Singular Value Decomposition (SVD) generalizes orthogonal diagonalization to *every* real rectangular matrix $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ by decoupling the domain rotation $\boldsymbol{V} \in \mathbb{R}^{n \times n}$ from the codomain rotation $\boldsymbol{U} \in \mathbb{R}^{m \times m}$, yielding the canonical factorization $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$ and enabling optimal low-rank approximation via the Eckart–Young–Mirsky theorem [[Source - E D I T I O N|linear_algebra]] (Sec. 7.4, p. 419) [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 4.5–4.6, pp. 119–132).

Let $\mathbb{S}^n = \{\boldsymbol{A} \in \mathbb{R}^{n \times n} : \boldsymbol{A} = \boldsymbol{A}^{\top}\}$ denote the subspace of real symmetric matrices, and $\mathbb{S}_+^n$ ($\mathbb{S}_{++}^n$) the cone of positive semidefinite (positive definite) matrices. Unless stated otherwise, eigenvalues of $\boldsymbol{A} \in \mathbb{S}^n$ are ordered algebraically as $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n$, whereas singular values of $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ are ordered non-increasingly as $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0 = \sigma_{r+1} = \dots = \sigma_p$, where $p = \min(m,n)$ and $r = \mathrm{rank}(\boldsymbol{A})$.

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Across the five synthesized sources, symbol collisions occur between EVD eigenvectors, SVD singular matrices, diagonal eigenvalue/singular-value matrices, and approximation norms. Table `tab:svd:notation` establishes the unified convention adopted throughout this leaf monograph: $\boldsymbol{P}, \boldsymbol{Q}$ denote general and orthogonal EVD eigenvector matrices; $\boldsymbol{U}, \boldsymbol{\Sigma}, \boldsymbol{V}$ are reserved strictly for the SVD; and $\boldsymbol{A}^{+}$ denotes the Moore–Penrose pseudoinverse. Notice one important typographical discrepancy in the literature: while [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Thm. 20.1, p. 250) proves the squared Frobenius error $\lVert \boldsymbol{A}-\boldsymbol{A}_k \rVert_F^2 = \sum_{i=k+1}^r \sigma_i^2$, [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. (7.201), p. 264) prints $\lVert \boldsymbol{A}-\hat{\boldsymbol{A}} \rVert_F = \sum_{k=K+1}^r \sigma_k$, omitting the square on both sides.

**Table:** Unified Notation and Cross-Book Convention Reconciliation.

| **Concept** | **Unified Symbol** | **Source Textbook Conventions & Reconciliation** |
| --- | --- | --- |
| General EVD | $\boldsymbol{A} = \boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}$ | $\boldsymbol{P}\boldsymbol{D}\boldsymbol{P}^{-1}$ in [[Source - E D I T I O N]] (Sec. 7.1) and [[Source - Mathematics for Machine Learning]] (Sec. 4.4); $\boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{-1}$ in [[Source - Probabilistic Machine Learning]] (Sec. 7.4.2); $\boldsymbol{V}\mathrm{diag}(\boldsymbol{\lambda})\boldsymbol{V}^{-1}$ in [[Source - Deep Learning]] (Sec. 2.7). |
| Symmetric EVD | $\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\top}$ | $\boldsymbol{P}\boldsymbol{D}\boldsymbol{P}^{\top}$ in [[Source - E D I T I O N]] (Sec. 7.1) and [[Source - Mathematics for Machine Learning]] (Sec. 4.2); $\boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top}$ in [[Source - Probabilistic Machine Learning]] (Sec. 7.4.3); $\boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\top}$ in [[Source - Deep Learning]] (Eq. (2.41)). |
| Full SVD | $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$ | $\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$ in [[Source - E D I T I O N]] (Sec. 7.4), [[Source - Mathematics for Machine Learning]] (Sec. 4.5), and [[Source - Machine Learning co ban]] (Ch. 20); $\boldsymbol{U}\boldsymbol{S}\boldsymbol{V}^{\top}$ in [[Source - Probabilistic Machine Learning]] (Sec. 7.5.1); $\boldsymbol{U}\boldsymbol{D}\boldsymbol{V}^{\top}$ in [[Source - Deep Learning]] (Sec. 2.8). |
| Thin / Compact / Truncated SVD | $\boldsymbol{U}_p\boldsymbol{\Sigma}_p\boldsymbol{V}_p^{\top}$,  $\boldsymbol{U}_r\boldsymbol{\Sigma}_r\boldsymbol{V}_r^{\top}$,  $\boldsymbol{A}_k = \boldsymbol{U}_k\boldsymbol{\Sigma}_k\boldsymbol{V}_k^{\top}$ | Economy/thin uses $p = \min(m,n)$ [[Source - Probabilistic Machine Learning]] (Fig. 7.8); compact/reduced uses $r = \mathrm{rank}(\boldsymbol{A})$ [[Source - E D I T I O N]] (Ex. 7, p. 424) [[Source - Machine Learning co ban]] (Sec. 20.2.4); truncated keeps $k < r$ modes [[Source - Mathematics for Machine Learning]] (Eq. (4.92)). |
| Pseudoinverse | $\boldsymbol{A}^{+} = \boldsymbol{V}_r\boldsymbol{\Sigma}_r^{-1}\boldsymbol{U}_r^{\top}$ | $\boldsymbol{A}^+$ in [[Source - E D I T I O N]] (Eq. (10), p. 424) and [[Source - Deep Learning]] (Eq. (2.47)); $\boldsymbol{A}^\dagger$ in [[Source - Probabilistic Machine Learning]] (Sec. 7.5.3). |
| Eckart–Young Frobenius error | $\lVert \boldsymbol{A}-\boldsymbol{A}_k \rVert_F =$  $\bigl(\sum_{i=k+1}^r \sigma_i^2\bigr)^{1/2}$ | Squared norm $\sum_{i=k+1}^r \sigma_i^2$ proved in [[Source - Machine Learning co ban]] (Thm. 20.1); spectral norm $\lVert \boldsymbol{A}-\boldsymbol{A}_k \rVert_2 = \sigma_{k+1}$ proved in [[Source - Mathematics for Machine Learning]] (Thm. 4.25). |

## Core Mathematical Derivations & Theorems

### Spectral Theorem for Real Symmetric Matrices

General square matrices over $\mathbb{R}$ may exhibit complex conjugate eigenvalues (e.g., rotation matrices) or fail to span $\mathbb{R}^n$ with eigenvectors when geometric multiplicity falls strictly below algebraic multiplicity (defective matrices, such as $\begin{bsmallmatrix} 2 & 1 \\ 0 & 2 \end{bsmallmatrix}$; [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ex. 4.6, p. 108)). Real symmetry eliminates both pathologies simultaneously.

> [!important] Theorem: Real Spectral Theorem & Orthogonal Spectral Decomposition
> Let $\boldsymbol{A} \in \mathbb{S}^n$ be a real symmetric matrix. Then [[Source - E D I T I O N|linear_algebra]] (Sec. 7.1, Thms. 1–3, pp. 397–400) [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Thm. 4.15, p. 111):
> - Every eigenvalue $\lambda_i$ of $\boldsymbol{A}$ is real ($\lambda_i \in \mathbb{R}$).
> - Eigenvectors corresponding to distinct eigenvalues are orthogonal.
> - For every eigenvalue $\lambda$, its geometric multiplicity equals its algebraic multiplicity, yielding an orthonormal eigenbasis $\{\boldsymbol{q}_1,\dots,\boldsymbol{q}_n\}$ of $\mathbb{R}^n$.
> - With $\boldsymbol{Q} = [\boldsymbol{q}_1  \cdots  \boldsymbol{q}_n] \in \mathbb{R}^{n \times n}$ ($\boldsymbol{Q}^{\top}\boldsymbol{Q} = \boldsymbol{Q}\boldsymbol{Q}^{\top} = \boldsymbol{I}_n$) and $\boldsymbol{\Lambda} = \mathrm{diag}(\lambda_1,\dots,\lambda_n)$, $\boldsymbol{A}$ admits the orthogonal diagonalization and rank-1 spectral expansion:
>
> $$
> \boldsymbol{A} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\top} = \sum_{i=1}^n \lambda_i \boldsymbol{q}_i \boldsymbol{q}_i^{\top}.
> $$
>
> **Proof.**
> *(1) Real eigenvalues:* Suppose $\boldsymbol{A}\boldsymbol{z} = \lambda \boldsymbol{z}$ with $\lambda \in \mathbb{C}$ and nonzero $\boldsymbol{z} \in \mathbb{C}^n$. Taking the conjugate transpose $\boldsymbol{z}^H = \overline{\boldsymbol{z}}^{\top}$ and using $\boldsymbol{A}^H = \boldsymbol{A}^{\top} = \boldsymbol{A}$ (since $\boldsymbol{A}$ is real symmetric),
>
> $$
>   \lambda \lVert \boldsymbol{z} \rVert_2^2 = \boldsymbol{z}^H (\boldsymbol{A}\boldsymbol{z}) = (\boldsymbol{z}^H \boldsymbol{A}\boldsymbol{z}) = (\boldsymbol{A}\boldsymbol{z})^H \boldsymbol{z} = \overline{\lambda}\lVert \boldsymbol{z} \rVert_2^2 \implies (\lambda - \overline{\lambda})\lVert \boldsymbol{z} \rVert_2^2 = 0.
> $$
>
> Since $\lVert \boldsymbol{z} \rVert_2^2 > 0$, $\lambda = \overline{\lambda} \in \mathbb{R}$. Thus a real eigenvector $\boldsymbol{q}_1 \in \mathbb{R}^n$ with $\lVert \boldsymbol{q}_1 \rVert_2 = 1$ exists for $\lambda_1$.
>
> *(2) Orthogonality across distinct eigenspaces:* If $\boldsymbol{A}\boldsymbol{v}_1 = \lambda_1 \boldsymbol{v}_1$ and $\boldsymbol{A}\boldsymbol{v}_2 = \lambda_2 \boldsymbol{v}_2$ with $\lambda_1 \neq \lambda_2$, then
>
> $$
>   \lambda_1 (\boldsymbol{v}_1^{\top} \boldsymbol{v}_2) = (\boldsymbol{A}\boldsymbol{v}_1)^{\top} \boldsymbol{v}_2 = \boldsymbol{v}_1^{\top} \boldsymbol{A}^{\top} \boldsymbol{v}_2 = \boldsymbol{v}_1^{\top} \boldsymbol{A}\boldsymbol{v}_2 = \lambda_2 (\boldsymbol{v}_1^{\top} \boldsymbol{v}_2).
> $$
>
> Hence $(\lambda_1 - \lambda_2)(\boldsymbol{v}_1^{\top} \boldsymbol{v}_2) = 0 \implies \boldsymbol{v}_1^{\top} \boldsymbol{v}_2 = 0$.
>
> *(3)–(4) Existence of an orthonormal eigenbasis by induction on $n$:* The base case $n=1$ is immediate. Assume the claim holds for dimension $n-1$. By (1), choose a unit eigenvector $\boldsymbol{q}_1 \in \mathbb{R}^n$ with $\boldsymbol{A}\boldsymbol{q}_1 = \lambda_1\boldsymbol{q}_1$, and extend $\{\boldsymbol{q}_1\}$ to an orthonormal basis $[\boldsymbol{q}_1, \boldsymbol{W}_1]$ of $\mathbb{R}^n$ where $\boldsymbol{W}_1 \in \mathbb{R}^{n \times (n-1)}$ spans $\boldsymbol{q}_1^\perp$. For any $\boldsymbol{w} \in \boldsymbol{q}_1^\perp$, symmetry implies $\boldsymbol{q}_1^{\top} (\boldsymbol{A}\boldsymbol{w}) = (\boldsymbol{A}\boldsymbol{q}_1)^{\top} \boldsymbol{w} = \lambda_1 (\boldsymbol{q}_1^{\top} \boldsymbol{w}) = 0$, so $\boldsymbol{q}_1^\perp$ is $\boldsymbol{A}$-invariant. In this basis,
>
> $$
>   [\boldsymbol{q}_1, \boldsymbol{W}_1]^{\top} \boldsymbol{A} [\boldsymbol{q}_1, \boldsymbol{W}_1] =
>   \begin{bmatrix} \lambda_1 & \boldsymbol{0}^{\top} \\ \boldsymbol{0} & \boldsymbol{W}_1^{\top} \boldsymbol{A} \boldsymbol{W}_1 \end{bmatrix},
> $$
>
> where $\boldsymbol{B} = \boldsymbol{W}_1^{\top} \boldsymbol{A} \boldsymbol{W}_1 \in \mathbb{S}^{n-1}$ is symmetric. By the induction hypothesis, $\boldsymbol{B} = \tilde{\boldsymbol{Q}}\tilde{\boldsymbol{\Lambda}}\tilde{\boldsymbol{Q}}^{\top}$ with $\tilde{\boldsymbol{Q}} \in \mathbb{R}^{(n-1)\times(n-1)}$ orthogonal. Setting $\boldsymbol{Q} = [\boldsymbol{q}_1, \boldsymbol{W}_1\tilde{\boldsymbol{Q}}]$ yields $\boldsymbol{Q}^{\top}\boldsymbol{Q} = \boldsymbol{I}_n$ and $\boldsymbol{Q}^{\top}\boldsymbol{A}\boldsymbol{Q} = \mathrm{diag}(\lambda_1,\lambda_2,\dots,\lambda_n) = \boldsymbol{\Lambda}$. Expanding $\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\top}$ via the column–row outer product gives `(eq:svd:spectral)`, where each $\boldsymbol{P}_i = \boldsymbol{q}_i\boldsymbol{q}_i^{\top}$ satisfies $\boldsymbol{P}_i^2 = \boldsymbol{P}_i = \boldsymbol{P}_i^{\top}$ and $\boldsymbol{P}_i\boldsymbol{P}_j = \boldsymbol{0}$ for $i \neq j$.

Spectral decomposition immediately diagonalizes any quadratic form $Q(\boldsymbol{x}) = \boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x}$ with $\boldsymbol{A} \in \mathbb{S}^n$. Under the orthogonal change of coordinates $\boldsymbol{y} = \boldsymbol{Q}^{\top} \boldsymbol{x}$ ($\lVert \boldsymbol{y} \rVert_2 = \lVert \boldsymbol{x} \rVert_2$), the Principal Axes Theorem [[Source - E D I T I O N|linear_algebra]] (Sec. 7.2, pp. 404–407) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.4.3–7.4.4, pp. 255–256) eliminates all cross-product terms $x_i x_j$:

$$
Q(\boldsymbol{x}) = \boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x} = \boldsymbol{y}^{\top} \boldsymbol{\Lambda} \boldsymbol{y} = \sum_{i=1}^n \lambda_i y_i^2.
$$

Consequently, $\boldsymbol{A} \in \mathbb{S}_{++}^n$ (positive definite: $Q(\boldsymbol{x}) > 0$ for all $\boldsymbol{x} \neq \boldsymbol{0}$) iff $\lambda_i > 0$ for all $i$; $\boldsymbol{A} \in \mathbb{S}_+^n$ iff $\lambda_i \ge 0$ for all $i$; and indefinite iff $\boldsymbol{A}$ has both positive and negative eigenvalues ([[Source - E D I T I O N|linear_algebra]] (Thm. 5, p. 407); [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 2.7, p. 44)). When $\boldsymbol{A} \in \mathbb{S}_{++}^n$, the level set $\{ \boldsymbol{x} : \boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x} = c \}$ ($c > 0$) defines an $n$-dimensional hyper-ellipsoid aligned with eigenvectors $\boldsymbol{q}_i$ and semi-axis lengths $\sqrt{c/\lambda_i}$.

> [!tip] Rotate–Scale–Rotate for Symmetric Operators
> In `(eq:svd:spectral)`, applying $\boldsymbol{A}$ to $\boldsymbol{x}$ decomposes into three rigid/axis-aligned steps [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Fig. 4.7, p. 117) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.4.3, p. 255): (i) $\boldsymbol{Q}^{\top} \boldsymbol{x}$ orthogonally rotates/reflects $\boldsymbol{x}$ into the principal eigenbasis coordinates $\boldsymbol{y}$, (ii) $\boldsymbol{\Lambda}\boldsymbol{y}$ scales each principal axis independently by $\lambda_i$, and (iii) $\boldsymbol{Q}(\boldsymbol{\Lambda}\boldsymbol{y})$ rotates back into canonical coordinates. For a general non-symmetric diagonalizable matrix $\boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}$, $\boldsymbol{P}$ shears coordinates rather than rotating rigidly, and $\boldsymbol{P}^{-1}$ is not $\boldsymbol{P}^{\top}$.

### Rayleigh Quotient & Courant–Fischer Min-Max Theorem

In statistical learning (PCA, Fisher discriminant analysis, spectral clustering) and numerical linear algebra, eigenvalues arise directly as constrained stationary values of quadratic forms.

> [!important] Variational Characterization: Rayleigh Quotient & Courant–Fischer
> For $\boldsymbol{A} \in \mathbb{S}^n$ with eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n$ and orthonormal eigenvectors $\boldsymbol{q}_1, \dots, \boldsymbol{q}_n$, the *Rayleigh quotient* of any nonzero vector $\boldsymbol{x} \in \mathbb{R}^n \setminus \{\boldsymbol{0}\}$ is defined by [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. (7.172), p. 258):
>
> $$
> R(\boldsymbol{A}, \boldsymbol{x}) \triangleq \frac{\boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x}}{\boldsymbol{x}^{\top} \boldsymbol{x}}.
> $$
>
> Then:
> - **Sequential Constrained Extrema [[Source - E D I T I O N|linear_algebra]] (Thms. 6–8, pp. 412–414):** Stationary points of $R(\boldsymbol{A},\boldsymbol{x})$ on the unit sphere $\{\boldsymbol{x} : \lVert \boldsymbol{x} \rVert_2 = 1\}$ are unit eigenvectors of $\boldsymbol{A}$. Moreover,
>
> $$
> \lambda_k = \max_{\substack{\lVert \boldsymbol{x} \rVert_2 = 1 \\ \boldsymbol{x} \perp \mathrm{span}(\boldsymbol{q}_1,\dots,\boldsymbol{q}_{k-1})}} \boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x}, \qquad k = 1,\dots,n,
> $$
>
>   with maximum attained at $\boldsymbol{x} = \boldsymbol{q}_k$, and $\lambda_n = \min_{\lVert \boldsymbol{x} \rVert_2 = 1} \boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x} \le R(\boldsymbol{A},\boldsymbol{x}) \le \lambda_1$.
> - **Courant–Fischer Min-Max Principle:** Without explicit reference to prior eigenvectors, the $k$-th eigenvalue satisfies
>
> $$
> \lambda_k = \max_{\dim(\mathcal{V}) = k}   \min_{\boldsymbol{x} \in \mathcal{V} \setminus \{\boldsymbol{0}\}} R(\boldsymbol{A}, \boldsymbol{x})
>     = \min_{\dim(\mathcal{W}) = n - k + 1}   \max_{\boldsymbol{x} \in \mathcal{W} \setminus \{\boldsymbol{0}\}} R(\boldsymbol{A}, \boldsymbol{x}).
> $$
>
> **Proof.**
> *Lagrangian stationarity:* For maximizing $\boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x}$ subject to $\boldsymbol{x}^{\top}\boldsymbol{x} = 1$, form $\mathcal{L}(\boldsymbol{x},\lambda) = \boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x} + \lambda(1 - \boldsymbol{x}^{\top}\boldsymbol{x})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.4.8, p. 259). Setting $\nabla_{\boldsymbol{x}}\mathcal{L} = 2\boldsymbol{A}\boldsymbol{x} - 2\lambda\boldsymbol{x} = \boldsymbol{0}$ yields $\boldsymbol{A}\boldsymbol{x} = \lambda\boldsymbol{x}$, where $R(\boldsymbol{A},\boldsymbol{x}) = \lambda$.
>
> *Sequential extrema via eigenbasis expansion:* Write any $\boldsymbol{x}$ satisfying $\lVert \boldsymbol{x} \rVert_2=1$ and $\boldsymbol{x}^{\top}\boldsymbol{q}_1 = \dots = \boldsymbol{x}^{\top}\boldsymbol{q}_{k-1} = 0$ in the eigenbasis $\boldsymbol{y} = \boldsymbol{Q}^{\top}\boldsymbol{x}$, so $y_1 = \dots = y_{k-1} = 0$ and $\sum_{i=k}^n y_i^2 = 1$. Since $\lambda_k \ge \lambda_i$ for all $i \ge k$,
>
> $$
>   \boldsymbol{x}^{\top}\boldsymbol{A}\boldsymbol{x} = \sum_{i=k}^n \lambda_i y_i^2 \le \lambda_k \sum_{i=k}^n y_i^2 = \lambda_k,
> $$
>
> with equality achieved at $\boldsymbol{x} = \boldsymbol{q}_k$ ($y_k = 1$, $y_{i \neq k} = 0$).
>
> *Courant–Fischer min-max:* Let $\mathcal{W}_0 = \mathrm{span}(\boldsymbol{q}_k,\dots,\boldsymbol{q}_n)$ (dimension $n-k+1$) and $\mathcal{V}_0 = \mathrm{span}(\boldsymbol{q}_1,\dots,\boldsymbol{q}_k)$ (dimension $k$). For any subspace $\mathcal{V}$ with $\dim(\mathcal{V}) = k$, Grassmann's dimension formula gives
>
> $$
>   \dim(\mathcal{V} \cap \mathcal{W}_0) = \dim(\mathcal{V}) + \dim(\mathcal{W}_0) - \dim(\mathcal{V} + \mathcal{W}_0) \ge k + (n-k+1) - n = 1.
> $$
>
> Hence there exists a unit vector $\boldsymbol{z} \in \mathcal{V} \cap \mathcal{W}_0$. Since $\boldsymbol{z} \in \mathcal{W}_0$, $\boldsymbol{z} = \sum_{i=k}^n y_i \boldsymbol{q}_i$ satisfies $R(\boldsymbol{A},\boldsymbol{z}) = \sum_{i=k}^n \lambda_i y_i^2 \le \lambda_k$, so $\min_{\boldsymbol{x} \in \mathcal{V} \setminus \{\boldsymbol{0}\}} R(\boldsymbol{A},\boldsymbol{x}) \le \lambda_k$ for every $k$-dimensional $\mathcal{V}$. Conversely, taking $\mathcal{V} = \mathcal{V}_0$ yields $\min_{\boldsymbol{x} \in \mathcal{V}_0 \setminus \{\boldsymbol{0}\}} R(\boldsymbol{A},\boldsymbol{x}) = \sum_{i=1}^k \lambda_i y_i^2 \ge \lambda_k$, establishing the $\max\min$ identity. The $\min\max$ identity follows analogously by intersecting any $(n-k+1)$-dimensional $\mathcal{W}$ with $\mathcal{V}_0$.

Applying Courant–Fischer to the symmetric positive semidefinite Gram matrix $\boldsymbol{A}^{\top}\boldsymbol{A} \in \mathbb{S}_+^n$ gives an immediate variational characterization of singular values:

$$
\sigma_k(\boldsymbol{A}) = \sqrt{\lambda_k(\boldsymbol{A}^{\top}\boldsymbol{A})}
  = \max_{\dim(\mathcal{V}) = k}   \min_{\substack{\boldsymbol{x} \in \mathcal{V} \\ \lVert \boldsymbol{x} \rVert_2 = 1}} \lVert \boldsymbol{A}\boldsymbol{x} \rVert_2,
  \qquad \lVert \boldsymbol{A} \rVert_2 \triangleq \max_{\lVert \boldsymbol{x} \rVert_2 = 1} \lVert \boldsymbol{A}\boldsymbol{x} \rVert_2 = \sigma_1(\boldsymbol{A}).
$$

### Constructive Existence of the SVD, Fundamental Subspaces & Pseudoinverse

Every matrix $\boldsymbol{A} \in \mathbb{R}^{m \times n}$, regardless of aspect ratio or rank deficiency, factorizes into an orthogonal domain transformation, a diagonal non-negative scaling, and an orthogonal codomain transformation.

> [!important] Theorem: Existence of Full, Reduced, and Compact SVD
> Let $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ have rank $r \le p = \min(m,n)$. Then there exist orthogonal matrices $\boldsymbol{U} = [\boldsymbol{u}_1  \cdots  \boldsymbol{u}_m] \in \mathbb{R}^{m \times m}$ and $\boldsymbol{V} = [\boldsymbol{v}_1  \cdots  \boldsymbol{v}_n] \in \mathbb{R}^{n \times n}$, and an $m \times n$ diagonal-structured matrix
>
> $$
>   \boldsymbol{\Sigma} = \begin{bmatrix} \boldsymbol{\Sigma}_r & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{0} \end{bmatrix} \in \mathbb{R}^{m \times n},
>   \qquad
>   \boldsymbol{\Sigma}_r = \mathrm{diag}(\sigma_1,\dots,\sigma_r), \quad \sigma_1 \ge \dots \ge \sigma_r > 0,
> $$
>
> such that $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$ [[Source - E D I T I O N|linear_algebra]] (Thm. 10, p. 419) [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Thm. 4.22, p. 119). Equivalently, with $p = \min(m,n)$:
>
> $$
> \begin{aligned}
> \boldsymbol{A} &= \boldsymbol{U}_p \boldsymbol{\Sigma}_p \boldsymbol{V}_p^{\top} \quad \text{(Economy / Thin SVD: }\boldsymbol{U}_p \in \mathbb{R}^{m\times p},  \boldsymbol{\Sigma}_p \in \mathbb{R}^{p\times p},  \boldsymbol{V}_p \in \mathbb{R}^{n\times p}\text{)},  \\
>   \boldsymbol{A} &= \boldsymbol{U}_r \boldsymbol{\Sigma}_r \boldsymbol{V}_r^{\top} = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} \quad \text{(Compact SVD / Rank-1 Expansion)}.
> \end{aligned}
> $$
>
> **Constructive Proof.** Following [[Source - E D I T I O N|linear_algebra]] (pp. 418–420) and [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 4.5.2, pp. 123–125), consider the symmetric matrix $\boldsymbol{G} = \boldsymbol{A}^{\top}\boldsymbol{A} \in \mathbb{S}^n$. For every $\boldsymbol{x} \in \mathbb{R}^n$, $\boldsymbol{x}^{\top} \boldsymbol{G} \boldsymbol{x} = \lVert \boldsymbol{A}\boldsymbol{x} \rVert_2^2 \ge 0$, so $\boldsymbol{G} \in \mathbb{S}_+^n$ is positive semidefinite [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Thm. 4.14, p. 111). By Theorem `thm:svd:spectral`, $\boldsymbol{G}$ has an orthonormal eigenbasis $\{\boldsymbol{v}_1,\dots,\boldsymbol{v}_n\}$ of $\mathbb{R}^n$ with real non-negative eigenvalues $\lambda_1 \ge \dots \ge \lambda_n \ge 0$. Because $\lVert \boldsymbol{A}\boldsymbol{v}_i \rVert_2^2 = \boldsymbol{v}_i^{\top} (\boldsymbol{A}^{\top}\boldsymbol{A})\boldsymbol{v}_i = \lambda_i$, define $\sigma_i = \sqrt{\lambda_i} \ge 0$, and let $r$ be the index count of strictly positive singular values ($\sigma_1 \ge \dots \ge \sigma_r > 0$, $\sigma_{r+1}=\dots=\sigma_n=0$).
>
> For $i \neq j$, orthogonality of $\boldsymbol{v}_i, \boldsymbol{v}_j$ induces orthogonality of their images under $\boldsymbol{A}$:
>
> $$
> (\boldsymbol{A}\boldsymbol{v}_i)^{\top} (\boldsymbol{A}\boldsymbol{v}_j) = \boldsymbol{v}_i^{\top} (\boldsymbol{A}^{\top}\boldsymbol{A}) \boldsymbol{v}_j = \lambda_j (\boldsymbol{v}_i^{\top} \boldsymbol{v}_j) = 0.
> $$
>
> For $i \in \{1,\dots,r\}$, define unit left-singular vectors $\boldsymbol{u}_i = \frac{1}{\sigma_i}\boldsymbol{A}\boldsymbol{v}_i \in \mathbb{R}^m$, so that $\boldsymbol{A}\boldsymbol{v}_i = \sigma_i\boldsymbol{u}_i$ and $\boldsymbol{u}_i^{\top} \boldsymbol{u}_j = \delta_{ij}$ by `(eq:svd:img_orth)`. For $j > r$, $\lVert \boldsymbol{A}\boldsymbol{v}_j \rVert_2 = \sigma_j = 0 \implies \boldsymbol{A}\boldsymbol{v}_j = \boldsymbol{0}$. Extend the orthonormal set $\{\boldsymbol{u}_1,\dots,\boldsymbol{u}_r\}$ to an orthonormal basis $\{\boldsymbol{u}_1,\dots,\boldsymbol{u}_m\}$ of $\mathbb{R}^m$ (for instance via Gram–Schmidt). Assembling $\boldsymbol{U} = [\boldsymbol{u}_1  \cdots  \boldsymbol{u}_m]$ and $\boldsymbol{V} = [\boldsymbol{v}_1  \cdots  \boldsymbol{v}_n]$ yields
>
> $$
>   \boldsymbol{A}\boldsymbol{V} = [\boldsymbol{A}\boldsymbol{v}_1  \cdots  \boldsymbol{A}\boldsymbol{v}_r    \boldsymbol{0}  \cdots  \boldsymbol{0}] = [\sigma_1\boldsymbol{u}_1  \cdots  \sigma_r\boldsymbol{u}_r    \boldsymbol{0}  \cdots  \boldsymbol{0}] = \boldsymbol{U}\boldsymbol{\Sigma}.
> $$
>
> Right-multiplying by $\boldsymbol{V}^{\top} = \boldsymbol{V}^{-1}$ yields $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$. Truncating the zero rows/columns of $\boldsymbol{\Sigma}$ directly gives `(eq:svd:thin)` and `(eq:svd:compact)`.

Multiplying $\boldsymbol{A}^{\top}\boldsymbol{A}$ and $\boldsymbol{A}\boldsymbol{A}^{\top}$ directly exposes the symmetric eigen-relations linking EVD and SVD [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.5.2, pp. 260–261):

$$
\boldsymbol{A}^{\top}\boldsymbol{A} = \boldsymbol{V}(\boldsymbol{\Sigma}^{\top}\boldsymbol{\Sigma})\boldsymbol{V}^{\top}, \qquad \boldsymbol{A}\boldsymbol{A}^{\top} = \boldsymbol{U}(\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{\top})\boldsymbol{U}^{\top}.
$$

Thus the right-singular vectors $\boldsymbol{V}$ diagonalize $\boldsymbol{A}^{\top}\boldsymbol{A}$, the left-singular vectors $\boldsymbol{U}$ diagonalize $\boldsymbol{A}\boldsymbol{A}^{\top}$, and both symmetric Gram matrices share the exact same $r$ nonzero eigenvalues $\sigma_1^2, \dots, \sigma_r^2$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 2.8, p. 45). Crucially, for any symmetric positive semidefinite matrix $\boldsymbol{S} \in \mathbb{S}_+^n$, $\boldsymbol{S} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\top}$ coincides directly with its SVD (with singular values $\sigma_i = \lambda_i$); for an arbitrary symmetric matrix $\boldsymbol{A} \in \mathbb{S}^n$, $\sigma_i = \lvert \lambda_i \rvert$ and $\boldsymbol{u}_i = \mathrm{sign}(\lambda_i)\boldsymbol{v}_i$ (for $\lambda_i \neq 0$).

The SVD simultaneously constructs orthonormal bases for all four fundamental subspaces associated with $\boldsymbol{A}$ and $\boldsymbol{A}^{\top}$ ([[Source - E D I T I O N|linear_algebra]] (Sec. 7.4, Eqs. (5)–(8), pp. 422–423); see also [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.5.4, p. 262)):

$$
\begin{aligned}
  \mathrm{col}(\boldsymbol{A}) &= \mathrm{span}(\boldsymbol{u}_1,\dots,\boldsymbol{u}_r), &
  \mathrm{null}(\boldsymbol{A}^{\top}) &= \mathrm{col}(\boldsymbol{A})^\perp = \mathrm{span}(\boldsymbol{u}_{r+1},\dots,\boldsymbol{u}_m), \\
  \mathrm{row}(\boldsymbol{A}) &= \mathrm{col}(\boldsymbol{A}^{\top}) = \mathrm{span}(\boldsymbol{v}_1,\dots,\boldsymbol{v}_r), &
  \mathrm{null}(\boldsymbol{A}) &= \mathrm{row}(\boldsymbol{A})^\perp = \mathrm{span}(\boldsymbol{v}_{r+1},\dots,\boldsymbol{v}_n).
\end{aligned}
$$

Hence $\dim\mathrm{col}(\boldsymbol{A}) + \dim\mathrm{null}(\boldsymbol{A}) = r + (n-r) = n$ (Rank–Nullity Theorem) and $\mathrm{rank}(\boldsymbol{A})$ equals the count of strictly positive singular values. Inverting the $r \times r$ core $\boldsymbol{\Sigma}_r$ yields the *Moore–Penrose pseudoinverse* ([[Source - E D I T I O N|linear_algebra]] (Ex. 7–8, p. 424); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.5.3, p. 261); [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 2.9, p. 46)):

$$
\boldsymbol{A}^{+} = \boldsymbol{V}_r \boldsymbol{\Sigma}_r^{-1} \boldsymbol{U}_r^{\top} = \sum_{i=1}^r \frac{1}{\sigma_i} \boldsymbol{v}_i \boldsymbol{u}_i^{\top} \in \mathbb{R}^{n \times m}.
$$

By direct substitution using $\boldsymbol{U}_r^{\top}\boldsymbol{U}_r = \boldsymbol{V}_r^{\top}\boldsymbol{V}_r = \boldsymbol{I}_r$, $\boldsymbol{A}^{+}$ uniquely satisfies all four Moore–Penrose axioms: $\boldsymbol{A}\boldsymbol{A}^{+}\boldsymbol{A} = \boldsymbol{A}$, $\boldsymbol{A}^{+}\boldsymbol{A}\boldsymbol{A}^{+} = \boldsymbol{A}^{+}$, $(\boldsymbol{A}\boldsymbol{A}^{+})^{\top} = \boldsymbol{A}\boldsymbol{A}^{+}$, and $(\boldsymbol{A}^{+}\boldsymbol{A})^{\top} = \boldsymbol{A}^{+}\boldsymbol{A}$, where $\boldsymbol{A}\boldsymbol{A}^{+} = \boldsymbol{U}_r\boldsymbol{U}_r^{\top}$ is the orthogonal projector onto $\mathrm{col}(\boldsymbol{A})$ and $\boldsymbol{A}^{+}\boldsymbol{A} = \boldsymbol{V}_r\boldsymbol{V}_r^{\top}$ is the orthogonal projector onto $\mathrm{row}(\boldsymbol{A})$. Therefore, for any linear system $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$, $\hat{\boldsymbol{x}} = \boldsymbol{A}^{+} \boldsymbol{b}$ solves $\min_{\boldsymbol{x} \in \mathbb{R}^n} \lVert \boldsymbol{A}\boldsymbol{x} - \boldsymbol{b} \rVert_2$ with minimum Euclidean norm $\lVert \hat{\boldsymbol{x}} \rVert_2$:

$$
  \lVert \boldsymbol{A}\boldsymbol{x} - \boldsymbol{b} \rVert_2^2
  = \lVert \boldsymbol{\Sigma}_r \boldsymbol{V}_r^{\top} \boldsymbol{x} - \boldsymbol{U}_r^{\top} \boldsymbol{b} \rVert_2^2 + \lVert \boldsymbol{U}_{m-r}^{\top} \boldsymbol{b} \rVert_2^2,
$$

because orthogonal transformations preserve $\ell_2$ norm; setting $\boldsymbol{V}_r^{\top} \hat{\boldsymbol{x}} = \boldsymbol{\Sigma}_r^{-1} \boldsymbol{U}_r^{\top} \boldsymbol{b}$ annihilates the first term while setting $\boldsymbol{V}_{n-r}^{\top} \hat{\boldsymbol{x}} = \boldsymbol{0}$ minimizes $\lVert \boldsymbol{x} \rVert_2^2 = \lVert \boldsymbol{V}_r^{\top}\boldsymbol{x} \rVert_2^2 + \lVert \boldsymbol{V}_{n-r}^{\top}\boldsymbol{x} \rVert_2^2$ [[Source - E D I T I O N|linear_algebra]] (Ex. 8, p. 424) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 2.9, p. 46).

> [!tip] Geometric Action of SVD: Rotate, Scale/Embed, Rotate
> Every linear map $\Phi(\boldsymbol{x}) = \boldsymbol{A}\boldsymbol{x} = \boldsymbol{U}(\boldsymbol{\Sigma}(\boldsymbol{V}^{\top}\boldsymbol{x}))$ sends the unit sphere $S^{n-1} = \{\boldsymbol{x} \in \mathbb{R}^n : \lVert \boldsymbol{x} \rVert_2 = 1\}$ onto an $r$-dimensional hyper-ellipsoid in $\mathbb{R}^m$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Fig. 4.8, p. 120) [[Source - E D I T I O N|linear_algebra]] (Ex. 1, p. 417). First, $\boldsymbol{V}^{\top}$ rotates the unit sphere so the right-singular basis $\{\boldsymbol{v}_i\}$ aligns with canonical axes. Second, $\boldsymbol{\Sigma}$ scales each coordinate $i \le r$ by $\sigma_i$, collapses the nullspace coordinates $i > r$ to zero, and embeds/projects from $\mathbb{R}^n$ to $\mathbb{R}^m$. Third, $\boldsymbol{U}$ rotates the canonical axes into the left-singular basis $\{\boldsymbol{u}_i\}$, leaving the semi-axes of the image hyper-ellipsoid aligned with $\sigma_1\boldsymbol{u}_1, \dots, \sigma_r\boldsymbol{u}_r$.

> [!note] Source Figure
> *Figure:* Geometric action of a rectangular matrix $\boldsymbol{A} \in \mathbb{R}^{2 \times 3}$ mapping the unit sphere $S^2 \subset \mathbb{R}^3$ onto an ellipse in $\mathbb{R}^2$ whose major and minor semi-axes are $\sigma_1\boldsymbol{u}_1 = (18,6)^{\top}$ ($\sigma_1 = 6\sqrt{10}$) and $\sigma_2\boldsymbol{u}_2 = (3,-9)^{\top}$ ($\sigma_2 = 3\sqrt{10}$), while collapsing $\mathrm{null}(\boldsymbol{A}) = \mathrm{span}(\boldsymbol{v}_3)$ to the origin [[Source - E D I T I O N|linear_algebra]] (Ch. 7, Sec. 7.4, Ex. 1–2, pp. 417–418 (PDF p. 434)). (`linear_algebra_p434_fig1.png`)

### Eckart–Young–Mirsky Best Rank-$k$ Approximation Theorem

Truncating the compact outer-product sum `(eq:svd:compact)` after $k < r$ terms produces the rank-$k$ matrix $\boldsymbol{A}_k = \sum_{i=1}^k \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} = \boldsymbol{U}_k \boldsymbol{\Sigma}_k \boldsymbol{V}_k^{\top}$. Remarkably, $\boldsymbol{A}_k$ is simultaneously optimal across *all* unitarily invariant matrix norms (and specifically both spectral and Frobenius norms).

> [!important] Theorem: Eckart–Young–Mirsky Optimality (Spectral & Frobenius Norms)
> Let $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ have rank $r$ with SVD $\boldsymbol{A} = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top}$. For any integer $1 \le k < r$, define the truncated SVD $\boldsymbol{A}_k = \boldsymbol{U}_k\boldsymbol{\Sigma}_k\boldsymbol{V}_k^{\top} = \sum_{i=1}^k \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top}$. Then $\boldsymbol{A}_k$ solves the constrained non-convex rank-$k$ approximation problem in both the spectral ($\ell_2$ operator) norm and the Frobenius norm [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Thm. 4.25, p. 131) [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Thm. 20.1 & Sec. 20.2.6, pp. 250–251):
>
> $$
> \begin{aligned}
> \min_{\mathrm{rank}(\boldsymbol{B}) \le k} \lVert \boldsymbol{A} - \boldsymbol{B} \rVert_2 &= \lVert \boldsymbol{A} - \boldsymbol{A}_k \rVert_2 = \sigma_{k+1},  \\
>   \min_{\mathrm{rank}(\boldsymbol{B}) \le k} \lVert \boldsymbol{A} - \boldsymbol{B} \rVert_F &= \lVert \boldsymbol{A} - \boldsymbol{A}_k \rVert_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}.
> \end{aligned}
> $$
>
> **Proof.**
> *Part 1 (Spectral Norm [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (pp. 131–132)):* Since $\boldsymbol{A} - \boldsymbol{A}_k = \sum_{i=k+1}^r \sigma_i \boldsymbol{u}_i\boldsymbol{v}_i^{\top}$ has largest singular value $\sigma_{k+1}$, `(eq:svd:var_sv)` yields $\lVert \boldsymbol{A} - \boldsymbol{A}_k \rVert_2 = \sigma_{k+1}$. Now let $\boldsymbol{B} \in \mathbb{R}^{m \times n}$ be any matrix with $\mathrm{rank}(\boldsymbol{B}) \le k$. By Rank–Nullity, $\dim(\mathrm{null}(\boldsymbol{B})) = n - \mathrm{rank}(\boldsymbol{B}) \ge n - k$. Let $\mathcal{V}_{k+1} = \mathrm{span}(\boldsymbol{v}_1,\dots,\boldsymbol{v}_{k+1})$, which has $\dim(\mathcal{V}_{k+1}) = k + 1$. Because $(n-k) + (k+1) = n + 1 > n$, the intersection $\mathrm{null}(\boldsymbol{B}) \cap \mathcal{V}_{k+1}$ contains a unit vector $\boldsymbol{z} = \sum_{i=1}^{k+1} c_i \boldsymbol{v}_i$ ($\sum_{i=1}^{k+1} c_i^2 = 1$) with $\boldsymbol{B}\boldsymbol{z} = \boldsymbol{0}$. Hence
>
> $$
>   \lVert \boldsymbol{A} - \boldsymbol{B} \rVert_2^2 \ge \lVert (\boldsymbol{A} - \boldsymbol{B})\boldsymbol{z} \rVert_2^2 = \lVert \boldsymbol{A}\boldsymbol{z} \rVert_2^2 = \Bigl\lVert \sum_{i=1}^{k+1} c_i \sigma_i \boldsymbol{u}_i \Bigr\rVert_2^2 = \sum_{i=1}^{k+1} c_i^2 \sigma_i^2 \ge \sigma_{k+1}^2 \sum_{i=1}^{k+1} c_i^2 = \sigma_{k+1}^2.
> $$
>
> Taking square roots gives $\lVert \boldsymbol{A} - \boldsymbol{B} \rVert_2 \ge \sigma_{k+1} = \lVert \boldsymbol{A} - \boldsymbol{A}_k \rVert_2$.
>
> *Part 2 (Frobenius Norm [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Thm. 20.1, p. 250)):* First verify the error of $\boldsymbol{A}_k$ via $\lVert \boldsymbol{X} \rVert_F^2 = \mathrm{Tr}(\boldsymbol{X}^{\top}\boldsymbol{X})$ and cyclic trace invariance:
>
> $$
>   \lVert \boldsymbol{A} - \boldsymbol{A}_k \rVert_F^2
>   = \mathrm{Tr}\!\Bigl(\sum_{i=k+1}^r \sigma_i \boldsymbol{v}_i\boldsymbol{u}_i^{\top} \sum_{j=k+1}^r \sigma_j \boldsymbol{u}_j\boldsymbol{v}_j^{\top}\Bigr)
>   = \sum_{i=k+1}^r \sigma_i^2 \mathrm{Tr}(\boldsymbol{v}_i\boldsymbol{v}_i^{\top})
>   = \sum_{i=k+1}^r \sigma_i^2,
> $$
>
> using $\boldsymbol{u}_i^{\top}\boldsymbol{u}_j = \delta_{ij}$ and $\mathrm{Tr}(\boldsymbol{v}_i\boldsymbol{v}_i^{\top}) = \boldsymbol{v}_i^{\top}\boldsymbol{v}_i = 1$. In particular, setting $k=0$ gives $\lVert \boldsymbol{A} \rVert_F^2 = \sum_{i=1}^r \sigma_i^2$. Next, for any $\boldsymbol{B}$ with $\mathrm{rank}(\boldsymbol{B}) \le k$, Mirsky's singular-value inequality (a consequence of Weyl's interlacing inequalities and orthogonal invariance of $\lVert \cdot \rVert_F$) states that
>
> $$
>   \lVert \boldsymbol{A} - \boldsymbol{B} \rVert_F^2 \ge \sum_{i=1}^p \bigl(\sigma_i(\boldsymbol{A}) - \sigma_i(\boldsymbol{B})\bigr)^2.
> $$
>
> (Alternatively, write $\boldsymbol{B} = \boldsymbol{P}_B\boldsymbol{B}$ where $\boldsymbol{P}_B \in \mathbb{R}^{m\times m}$ is the orthogonal projector onto $\mathrm{col}(\boldsymbol{B})$ of rank $\le k$: among matrices sharing column space $\mathrm{col}(\boldsymbol{B})$, orthogonal projection $\hat{\boldsymbol{B}} = \boldsymbol{P}_B\boldsymbol{A}$ minimizes $\lVert \boldsymbol{A}-\boldsymbol{B} \rVert_F^2 = \lVert (\boldsymbol{I}-\boldsymbol{P}_B)\boldsymbol{A} \rVert_F^2 + \lVert \boldsymbol{P}_B\boldsymbol{A}-\boldsymbol{B} \rVert_F^2$, leaving $\lVert \boldsymbol{A}-\hat{\boldsymbol{B}} \rVert_F^2 = \sum_{j=1}^p \sigma_j^2((\boldsymbol{I}-\boldsymbol{P}_B)\boldsymbol{A}) \ge \sum_{i=k+1}^r \sigma_i^2$ by `(eq:svd:var_sv)` and interlacing.) Since $\mathrm{rank}(\boldsymbol{B}) \le k$ forces $\sigma_i(\boldsymbol{B}) = 0$ for all $i \ge k+1$,
>
> $$
>   \lVert \boldsymbol{A} - \boldsymbol{B} \rVert_F^2 \ge \sum_{i=k+1}^p \bigl(\sigma_i(\boldsymbol{A}) - 0\bigr)^2 = \sum_{i=k+1}^r \sigma_i^2 = \lVert \boldsymbol{A} - \boldsymbol{A}_k \rVert_F^2. 
> $$
>

Normalizing `(eq:svd:ey_frob)` by $\lVert \boldsymbol{A} \rVert_F^2 = \sum_{j=1}^r \sigma_j^2$ yields the exact relative reconstruction error and cumulative explained spectral energy for selecting truncation rank $k$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Eqs. (20.18)–(20.19), p. 251):

$$
\frac{\lVert \boldsymbol{A} - \boldsymbol{A}_k \rVert_F^2}{\lVert \boldsymbol{A} \rVert_F^2} = \frac{\sum_{i=k+1}^r \sigma_i^2}{\sum_{j=1}^r \sigma_j^2} = 1 - \rho(k),
  \qquad \rho(k) \triangleq \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{j=1}^r \sigma_j^2}.
$$

## Algorithmic Implementation & Comparative Synthesis

Exact computation of the full SVD of $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ via Golub–Kahan Householder bidiagonalization followed by implicit QR iterations (or divide-and-conquer) requires $O(\min(mn^2, m^2n))$ floating-point operations [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.5.1, p. 260) [[Source - E D I T I O N|linear_algebra]] (p. 424). When $m, n \sim 10^5$–$10^9$ (e.g., recommender matrices, web graphs, or high-dimensional embeddings) and only the leading $k \ll \min(m,n)$ singular triplets are required, direct deterministic SVD is prohibitive [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 20.4, p. 253).

When $k$ is small and $\boldsymbol{A}$ is large or sparse, a classical iterative alternative is *power iteration with deflation* [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.4.6–7.4.7, pp. 258–259): repeatedly applying $\boldsymbol{v}^{(t)} \propto (\boldsymbol{A}^{\top}\boldsymbol{A})\boldsymbol{v}^{(t-1)}$ (implemented sequentially as $\boldsymbol{w} \leftarrow \boldsymbol{A}\boldsymbol{v}^{(t-1)}$, $\boldsymbol{v}^{(t)} \leftarrow \boldsymbol{A}^{\top}\boldsymbol{w}/\lVert \boldsymbol{A}^{\top}\boldsymbol{w} \rVert_2$ to avoid forming $\boldsymbol{A}^{\top}\boldsymbol{A}$) converges geometrically at rate $O((\sigma_2/\sigma_1)^{2t})$, after which rank-1 deflation $\boldsymbol{A}^{(2)} = \boldsymbol{A} - \sigma_1\boldsymbol{u}_1\boldsymbol{v}_1^{\top}$ reveals subsequent modes. A modern, cache-friendly, single-pass-capable extension of power iteration is *Randomized Subspace Truncated SVD* (Algorithm `alg:svd:rsvd`), which probes the dominant $k$-dimensional range of $\boldsymbol{A}$ simultaneously using $k + p$ Gaussian test vectors ($p \approx 5$–$10$ oversampling) and $q \in \{1,2\}$ subspace power iterations before running a tiny deterministic SVD on a $(k+p) \times n$ sketch.

> [!example] Algorithm: Randomized Truncated SVD with Power Subspace Iteration
> **Input:** Matrix $\boldsymbol{A} \in \mathbb{R}^{m \times n}$, target rank $k \ll \min(m,n)$, oversampling $p \ge 5$, power steps $q \ge 0$
> **Output:** Approximate rank-$k$ factors $\boldsymbol{U}_k \in \mathbb{R}^{m \times k}$, $\boldsymbol{\Sigma}_k \in \mathbb{R}^{k \times k}$, $\boldsymbol{V}_k \in \mathbb{R}^{n \times k}$
> $\ell \leftarrow \min(k + p,  \min(m, n))$
> Draw Gaussian test matrix $\boldsymbol{\Omega} \in \mathbb{R}^{n \times \ell}$ with i.i.d. entries $\Omega_{ij} \sim \mathcal{N}(0, 1)$
> $\boldsymbol{Y} \leftarrow \boldsymbol{A}\boldsymbol{\Omega} \in \mathbb{R}^{m \times \ell}$  *// Initial range sketch: $O(mn\ell)$ or $O(\mathrm{nnz}(\boldsymbol{A})\ell)$*
> Compute thin QR factorization $\boldsymbol{Q}, \boldsymbol{R} \leftarrow \mathrm{qr}(\boldsymbol{Y})$  *// Orthonormalize to prevent overflow/collapse*
> **for** $j = 1$ to $q$:
>
>   $\boldsymbol{Z}, \sim \leftarrow \mathrm{qr}(\boldsymbol{A}^{\top} \boldsymbol{Q})$  *// Power iteration $\approx (\boldsymbol{A}\boldsymbol{A}^{\top})^q \boldsymbol{A}\boldsymbol{\Omega}$ sharpens spectral gap*
>   $\boldsymbol{Q}, \sim \leftarrow \mathrm{qr}(\boldsymbol{A} \boldsymbol{Z})$
>
> $\boldsymbol{B} \leftarrow \boldsymbol{Q}^{\top} \boldsymbol{A} \in \mathbb{R}^{\ell \times n}$  *// Project $\boldsymbol{A}$ onto low-dimensional subspace $\mathrm{col}(\boldsymbol{Q})$*
> Compute thin deterministic SVD $\tilde{\boldsymbol{U}}, \tilde{\boldsymbol{\Sigma}}, \tilde{\boldsymbol{V}}^{\top} \leftarrow \mathrm{svd}(\boldsymbol{B})$  *// Only $O(\ell^2 n)$ flops*
> $\boldsymbol{U}_k \leftarrow \boldsymbol{Q}\tilde{\boldsymbol{U}}_{:, 1:k}$,   $\boldsymbol{\Sigma}_k \leftarrow \tilde{\boldsymbol{\Sigma}}_{1:k, 1:k}$,   $\boldsymbol{V}_k \leftarrow \tilde{\boldsymbol{V}}_{:, 1:k}$
> **return** $(\boldsymbol{U}_k, \boldsymbol{\Sigma}_k, \boldsymbol{V}_k)$

Table `tab:svd:complexity` compares the asymptotic computational cost, memory footprint, and numerical accuracy trade-offs across exact, iterative, and randomized decompositions.

**Table:** Computational and Storage Complexity of SVD Solvers for $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ ($p=\min(m,n)$, $r=\mathrm{rank}(\boldsymbol{A})$, target rank $k \ll p$).

| **Method** | **Time Complexity** | **Output Storage** | **Precision** | **Primary ML Use Case** |
| --- | --- | --- | --- | --- |
| Full SVD (Bidiag. QR) | $O(mn^2 + m^2n)$ | $m^2 + n^2 + p$ | Machine $\epsilon$ | Small/medium dense operators, full nullspaces |
| Thin / Economy SVD | $O(\min(mn^2, m^2n))$ | $p(m + n + 1)$ | Machine $\epsilon$ | Least-squares $\boldsymbol{A}^{+}$, whitening, medium PCA |
| Power Method + Deflation | $O(T k \mathrm{nnz}(\boldsymbol{A}))$ | $k(m + n + 1)$ | $O((\sigma_{k+1}/\sigma_k)^{2T})$ | Sparse graph/PageRank top-$k$ eigenpairs |
| Randomized Truncated SVD | $O((q{+}1) \mathrm{nnz}(\boldsymbol{A})(k{+}p)$ $\quad +  (k{+}p)^2 n)$ | $k(m + n + 1)$ | Near-optimal EYM | Large-scale embeddings, streaming/GPU PCA |

> [!info] Cross-Book Synthesis: From Matrix Phylogeny to Low-Rank ML Systems
> The five source books organize spectral decompositions along complementary axes that converge cleanly at low-rank approximation:
> - **Matrix Phylogeny [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Fig. 4.13, Sec. 4.7, p. 134):** Every real matrix $\boldsymbol{A} \in \mathbb{R}^{m\times n}$ admits an SVD and Moore–Penrose pseudoinverse $\boldsymbol{A}^{+}$. Specializing to square $\mathbb{R}^{n\times n}$ unlocks determinants and traces; non-defective matrices admit general EVD $\boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}$; symmetric matrices guarantee orthogonal EVD $\boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\top}$; symmetric positive definite matrices admit Cholesky $\boldsymbol{L}\boldsymbol{L}^{\top}$ and equal eigenvalues/singular values.
> - **Statistical Covariance, PCA & Whitening [[Source - E D I T I O N|linear_algebra]] (Sec. 7.5, pp. 426–430) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.4.5, pp. 256–258):** Given centered observations $\boldsymbol{X} \in \mathbb{R}^{N \times D}$ with empirical covariance $\boldsymbol{S} = \frac{1}{N}\boldsymbol{X}^{\top}\boldsymbol{X} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\top}$ (note $\frac{1}{N-1}\boldsymbol{B}\boldsymbol{B}^{\top}$ in Lay), the right-singular vectors of $\boldsymbol{X}$ are the principal components with $\sigma_i^2 = N\lambda_i$. PCA whitening scales uncorrelated projections via $\boldsymbol{W}_{\text{PCA}} = \boldsymbol{\Lambda}^{-1/2}\boldsymbol{Q}^{\top}$, whereas ZCA (Mahalanobis) whitening $\boldsymbol{W}_{\text{ZCA}} = \boldsymbol{Q}\boldsymbol{\Lambda}^{-1/2}\boldsymbol{Q}^{\top}$ rotates back to minimize $\lVert \boldsymbol{W}\boldsymbol{x} - \boldsymbol{x} \rVert_2^2$, preserving spatial image structure [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 7.7, p. 257).
> - **Compression & Spectral Decay [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 20.3, pp. 252–253) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.5.5, pp. 263–264):** Storing rank-$k$ factors $(\boldsymbol{U}_k, \mathrm{diag}(\boldsymbol{\Sigma}_k), \boldsymbol{V}_k)$ requires $k(m+n+1)$ scalars compared to $mn$ for $\boldsymbol{A}$ (or $4k(m+n+1)/(mn)$ bytes when compressing 8-bit grayscale images into 32-bit floats [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Eq. (20.23))). Natural images and structured feature matrices exhibit rapid singular decay where $k \ll \min(m,n)$ captures $>90\%$–$99\%$ of $\lVert \boldsymbol{A} \rVert_F^2$, whereas randomly permuted entries flatten the spectrum toward a broad bulk.

## Visual Diagnostics & Geometric Structure

To cement the algebraic–geometric correspondence, Figure `fig:svd:blocks` contrasts full, thin/economy, compact, and truncated rank-$k$ matrix factorizations, while Figure `fig:svd:subspace_action` traces the sequential operator action $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$ across the four fundamental subspaces.

> [!example] Visual Diagram (LaTeX / PDF)
> *Block dimensions across (a) Full SVD, (b) Thin/Economy SVD (shown for tall $m > n$, $p=n$), and (c) Truncated Rank-$k$ SVD ($k < r$). Compact SVD replaces $k$ by exact rank $r = \mathrm{rank}(\boldsymbol{A})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 7.8, p. 260) [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Fig. 20.2, p. 250).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Top: sequential decomposition of $\Phi(\boldsymbol{x}) = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}\boldsymbol{x}$ into domain rotation ($\boldsymbol{V}^{\top}$), axiswise scaling/dimension mapping ($\boldsymbol{\Sigma}$), and codomain rotation ($\boldsymbol{U}$). Bottom: orthogonal splitting of $\mathbb{R}^n$ and $\mathbb{R}^m$ into the four fundamental subspaces, showing that $\boldsymbol{A}$ acts invertibly between $\mathrm{row}(\boldsymbol{A})$ and $\mathrm{col}(\boldsymbol{A})$ via $\boldsymbol{A}^{+}$ [[Source - E D I T I O N|linear_algebra]] (Fig. 4, p. 423).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:svd:spectrum_error` quantifies the spectral decay and Eckart–Young error behavior for a rank-$40$ structured low-rank signal ($\sigma_k = 100 k^{-1.1} + 4 e^{-0.08(k-1)}$) contrasted against a shuffled/white-noise spectrum ($\sigma_k^{\text{noise}} = 16 \sqrt{1 - 0.65(k-1)/39}$), reproducing the separation between structured natural data and randomized entries discussed in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 7.10, p. 263). Finally, Figure `fig:svd:image_compression` demonstrates empirical low-rank image reconstruction via truncated SVD.

> [!example] Visual Diagram (LaTeX / PDF)
> *(a) Comparison of rapid algebraic singular-value decay for structured low-rank data against a flat noise/shuffled bulk [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.5.5, Fig. 7.10, p. 263). (b) Relative Frobenius reconstruction error $\lVert \boldsymbol{A}-\boldsymbol{A}_k \rVert_F/\lVert \boldsymbol{A} \rVert_F = \sqrt{1-\rho(k)}$, relative spectral error $\sigma_{k+1}/\sigma_1$, and cumulative captured energy $\rho(k)$ from `(eq:svd:ey_spec)`–`(eq:svd:rel_energy)` [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 20.2.5, p. 251).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!note] Source Figure
> *Figure:* Truncated SVD image compression of a $200 \times 320$ grayscale image (original rank $200$, $64{,}000$ pixels) using ranks $K \in \{2, 5, 20\}$. At $K=20$, storage drops to $K(N+D+1) = 10{,}420$ parameters ($16.3\%$ of raw pixels) while faithfully preserving facial structure [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.5.5, Fig. 7.9, pp. 263–264). (`probabilisticml_kevinmurphy_introduction_p293_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Pitfall 1: Squaring the Condition Number via Normal Equations $\boldsymbol{A}^{\top}\boldsymbol{A}$
> Constructing the SVD by explicitly forming $\boldsymbol{A}^{\top}\boldsymbol{A}$ and calling a symmetric eigensolver (the naive textbook hand-calculation recipe) squares the spectral condition number:
>
> $$
>   \kappa_2(\boldsymbol{A}^{\top}\boldsymbol{A}) = \frac{\lambda_1(\boldsymbol{A}^{\top}\boldsymbol{A})}{\lambda_r(\boldsymbol{A}^{\top}\boldsymbol{A})} = \left(\frac{\sigma_1}{\sigma_r}\right)^2 = \bigl[\kappa_2(\boldsymbol{A})\bigr]^2.
> $$
>
> In IEEE-754 double precision ($\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$), any matrix with $\kappa_2(\boldsymbol{A}) \gtrsim 10^8$ causes $\sigma_r^2/\sigma_1^2 < \epsilon_{\text{mach}}$, completely obliterating small singular values and corrupting their singular vectors [[Source - E D I T I O N|linear_algebra]] (Numerical Note, p. 424) [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 4.5.2, p. 126). **Remedy:** Compute the SVD directly via Householder bidiagonalization + implicit QR (LAPACK `dgesdd`/`dgesvd`) or QR-stabilized subspace iterations (Algorithm `alg:svd:rsvd`).

> [!warning] Pitfall 2: Naive Pseudoinverse vs. Truncated SVD under Ill-Conditioned Spectra
> Consider solving a noisy linear inverse problem $\boldsymbol{b} = \boldsymbol{A}\boldsymbol{x}_\star + \boldsymbol{\eta}$ ($\lVert \boldsymbol{\eta} \rVert_2 \le \delta$) using the exact Moore–Penrose pseudoinverse `(eq:svd:pinv)` versus naive normal inversion $(\boldsymbol{A}^{\top}\boldsymbol{A})^{-1}\boldsymbol{A}^{\top}\boldsymbol{b}$. Expanding the reconstruction error in the singular bases gives
>
> $$
>   \hat{\boldsymbol{x}} - \boldsymbol{x}_\star = \boldsymbol{A}^{+}\boldsymbol{b} - \boldsymbol{V}_r\boldsymbol{V}_r^{\top}\boldsymbol{x}_\star = \sum_{i=1}^r \frac{\boldsymbol{u}_i^{\top} \boldsymbol{\eta}}{\sigma_i} \boldsymbol{v}_i
>   \implies \lVert \hat{\boldsymbol{x}} - \boldsymbol{x}_\star \rVert_2 \le \frac{\delta}{\sigma_r}.
> $$
>
> When $\boldsymbol{A}$ is severely ill-conditioned ($\sigma_r \ll \delta$), unregularized inversion amplifies measurement noise and roundoff error by $1/\sigma_r \gg 1$, producing catastrophic oscillatory solutions; naive matrix inversion fails outright whenever $m \neq n$ or $\mathrm{rank}(\boldsymbol{A}) < n$. **Remedy:** Use *Truncated SVD (TSVD)* regularization $\boldsymbol{A}_k^{+} = \sum_{\sigma_i > \tau} \sigma_i^{-1} \boldsymbol{v}_i\boldsymbol{u}_i^{\top}$ with cutoff $\tau \ge \max(m,n)\epsilon_{\text{mach}}\sigma_1$ (the `rcond` tolerance in numerical libraries) or *Tikhonov/Ridge damping* $\sum_{i=1}^r \frac{\sigma_i}{\sigma_i^2 + \alpha}\boldsymbol{v}_i\boldsymbol{u}_i^{\top}$ aligned with $\lim_{\alpha \to 0^+} (\boldsymbol{A}^{\top}\boldsymbol{A}+\alpha\boldsymbol{I})^{-1}\boldsymbol{A}^{\top}$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Eq. (2.46), p. 46).

> [!warning] Pitfall 3: Rank Determination by Pivots & Defective Non-Symmetric Matrices
> Gaussian elimination / row-echelon pivot counting is numerically unstable for rank determination: tiny roundoff perturbations of order $\epsilon_{\text{mach}}$ turn exact zeros into nonzero pivots, falsely reporting full rank [[Source - E D I T I O N|linear_algebra]] (Numerical Note, p. 419). Likewise, applying symmetric spectral formulas to a non-symmetric matrix fails when eigenvalues are complex or the matrix is defective (fewer than $n$ linearly independent eigenvectors; [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 4.7, p. 134)). **Remedy:** Always determine *effective numerical rank* by counting singular values satisfying $\sigma_i > \tau$, and use SVD rather than EVD whenever $\boldsymbol{A} \neq \boldsymbol{A}^{\top}$.

> [!warning] Pitfall 4: Sign Ambiguity, Degenerate Subspaces & API Transpose Traps
> Even for distinct singular values, $(\boldsymbol{u}_i, \boldsymbol{v}_i)$ and $(-\boldsymbol{u}_i, -\boldsymbol{v}_i)$ satisfy $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$ identically; when singular values coincide ($\sigma_i = \sigma_{i+1}$), singular vectors are unique only up to an arbitrary orthogonal rotation inside the degenerate subspace [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 20.2.1, p. 247) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.4.1, p. 254). Furthermore, `numpy.linalg.svd(A)` returns $(\boldsymbol{U}, \boldsymbol{\sigma}, \boldsymbol{V}^{\top})$—returning the transpose $\boldsymbol{V}^{\top}$ rather than $\boldsymbol{V}$ and a 1D array $\boldsymbol{\sigma}$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 20.2.2, p. 249)—whereas `eig` does not guarantee sorted eigenvalues [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.5.2, p. 260). **Remedy:** Canonicalize signs (e.g., force the largest-magnitude entry of each $\boldsymbol{u}_i$ positive) for deterministic downstream features, and verify transpose conventions before matrix multiplication.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (5):** [[Condition Number]], [[Eigendecomposition & Spectral Theorem]], [[Low-Rank Approximation (Eckart-Young)]], [[Moore-Penrose Pseudoinverse]], [[Singular Value Decomposition (SVD)]]
- **Prerequisites (Directed $\leftarrow$):** [[Eigendecomposition & Spectral Theorem]], [[Inner Product & Norms]], [[Orthogonal Projection]], [[Singular Value Decomposition (SVD)]], [[Vector Space & Subspaces]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Camera Calibration & Homography]], [[Condition Number]], [[Double Descent]], [[Gradient Descent]], [[Linear & Ridge Regression]], [[LoRA & QLoRA]], [[Low-Rank Approximation (Eckart-Young)]], [[Matrix Factorization (ALS & SGD)]], [[Mercer Kernels & RKHS]], [[Momentum & Nesterov NAG]], [[Moore-Penrose Pseudoinverse]], [[Multivariate Gaussian & Schur Complement]], [[Principal Component Analysis (PCA)]], [[Singular Value Decomposition (SVD)]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Condition Number]], [[Double Descent]], [[Eigendecomposition & Spectral Theorem]], [[Gradient Descent]], [[Linear & Ridge Regression]], [[LoRA & QLoRA]], [[Low-Rank Approximation (Eckart-Young)]], [[Momentum & Nesterov NAG]], [[Moore-Penrose Pseudoinverse]], [[Principal Component Analysis (PCA)]], [[QR Factorization]], [[Singular Value Decomposition (SVD)]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - E D I T I O N]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
