---
type: monograph
domain: "D01"
parent_topic: "[[PT 01 - Linear Algebra, Matrix Decompositions & Vector Calculus]]"
sub_topic: "[[ST 01 - Vector Spaces, Spectral Theory & Matrix Factorizations]]"
cheatsheet: "[[Cheatsheet - Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]"
part_of: 
  - "[[ST 01 - Vector Spaces, Spectral Theory & Matrix Factorizations]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
concepts: 
  - "[[Cauchy-Schwarz Inequality]]"
  - "[[Inner Product & Norms]]"
  - "[[Orthogonal Projection]]"
  - "[[QR Factorization]]"
  - "[[Vector Space & Subspaces]]"
prerequisites: 
  - "[[Inner Product & Norms]]"
  - "[[Orthogonal Projection]]"
  - "[[Vector Space & Subspaces]]"
enables: 
  - "[[Cauchy-Schwarz Inequality]]"
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Inner Product & Norms]]"
  - "[[Linear & Ridge Regression]]"
  - "[[Mahalanobis Distance]]"
  - "[[Matrix Calculus & Jacobians]]"
  - "[[Mercer Kernels & RKHS]]"
  - "[[Orthogonal Projection]]"
  - "[[QR Factorization]]"
  - "[[Rademacher Complexity & VC Dimension]]"
  - "[[Singular Value Decomposition (SVD)]]"
related: 
  - "[[Cauchy-Schwarz Inequality]]"
  - "[[Condition Number]]"
  - "[[Inner Product & Norms]]"
  - "[[Linear & Ridge Regression]]"
  - "[[Mahalanobis Distance]]"
  - "[[Orthogonal Projection]]"
  - "[[QR Factorization]]"
  - "[[Rademacher Complexity & VC Dimension]]"
  - "[[Vector Space & Subspaces]]"
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

# Vector Spaces, Norms, Orthogonality & Least-Squares Projections

> [!abstract] Monograph Navigation
> - **Domain:** [[D01 - Mathematical Foundations for Machine Intelligence|Mathematical Foundations for Machine Intelligence]]
> - **Parent / Sub-Topic:** [[PT 01 - Linear Algebra, Matrix Decompositions & Vector Calculus|Linear Algebra, Matrix Decompositions & Vector Calculus]] → [[ST 01 - Vector Spaces, Spectral Theory & Matrix Factorizations|Vector Spaces, Spectral Theory & Matrix Factorizations]]
> - **Visual Cheatsheet:** [[Cheatsheet - Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]
> - **Description:** Synthesizes vector subspaces, inner products, Lp/Frobenius norms, orthogonal projections, Gram-Schmidt, and Moore-Penrose pseudoinverses for overdetermined systems.

## Problem Formulation & Unified Notation

Modern machine learning parameterizes representations, predictions, and loss functionals inside finite-dimensional vector spaces equipped with inner products and norms. Whether fitting linear predictors via empirical risk minimization, projecting high-dimensional embeddings onto low-rank principal subspaces, or solving overdetermined linear systems $\boldsymbol{A}\boldsymbol{x} \approx \boldsymbol{b}$ when exact solutions do not exist, the core mathematical operation is *orthogonal projection onto a linear or affine subspace* [[Source - E D I T I O N|linear_algebra]] (Ch. 6, pp. 349–367), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, pp. 81–90).

### Vector Spaces, Subspaces, and Matrix Operators

A *real vector space* $(V, +, \cdot)$ is a nonempty set $V$ closed under vector addition $(\boldsymbol{u}, \boldsymbol{v}) \mapsto \boldsymbol{u} + \boldsymbol{v}$ and scalar multiplication $(c, \boldsymbol{u}) \mapsto c\boldsymbol{u}$ ($c \in \mathbb{R}$), satisfying the ten axioms of associativity, commutativity of addition, identity elements ($\boldsymbol{0} \in V$ and $1 \in \mathbb{R}$), additive inverses, and distributivity [[Source - E D I T I O N|linear_algebra]] (Ch. 4, Sec. 4.1, p. 192), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 2, Sec. 2.4, pp. 35–39). A subset $W \subseteq V$ is a *subspace* if and only if $\boldsymbol{0} \in W$ and $W$ is closed under linear combinations: $\alpha\boldsymbol{u} + \beta\boldsymbol{v} \in W$ for all $\boldsymbol{u}, \boldsymbol{v} \in W$ and $\alpha, \beta \in \mathbb{R}$. Shifting a subspace $U \subseteq V$ by a translation vector $\boldsymbol{x}_0 \in V$ produces an *affine subspace* $L = \boldsymbol{x}_0 + U = \{\boldsymbol{x}_0 + \boldsymbol{u} : \boldsymbol{u} \in U\}$, which models solution sets of consistent inhomogeneous systems and separating hyperplanes [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 2, Sec. 2.8, pp. 61–63).

Given a finite collection $\{\boldsymbol{a}_1, \dots, \boldsymbol{a}_n\} \subset V$, its *span* is $\mathrm{span}\{\boldsymbol{a}_1, \dots, \boldsymbol{a}_n\} := \{\sum_{j=1}^n x_j \boldsymbol{a}_j : x_j \in \mathbb{R}\}$. The vectors are *linearly independent* if $\sum_{j=1}^n x_j \boldsymbol{a}_j = \boldsymbol{0}$ forces $x_1 = \dots = x_n = 0$; otherwise they are linearly dependent [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 2, Sec. 2.4, p. 38), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 1, Sec. 1.7, p. 18). A linearly independent spanning set forms a *basis* of $V$, assigning every $\boldsymbol{v} \in V$ unique coordinates, and every basis has the same cardinality $\dim(V)$ [[Source - E D I T I O N|linear_algebra]] (Ch. 4, Sec. 4.5, pp. 227–230).

Viewing a matrix $\boldsymbol{A} = [\boldsymbol{a}_1, \dots, \boldsymbol{a}_n] \in \mathbb{R}^{m \times n}$ as a linear map $\boldsymbol{x} \mapsto \boldsymbol{A}\boldsymbol{x}$ from $\mathbb{R}^n$ to $\mathbb{R}^m$ defines four fundamental subspaces:

$$
\begin{aligned}
\mathrm{col}(\boldsymbol{A}) &:= \{\boldsymbol{A}\boldsymbol{x} : \boldsymbol{x} \in \mathbb{R}^n\} = \mathrm{span}\{\boldsymbol{a}_1, \dots, \boldsymbol{a}_n\} \subseteq \mathbb{R}^m, &
  \mathrm{null}(\boldsymbol{A}) &:= \{\boldsymbol{x} \in \mathbb{R}^n : \boldsymbol{A}\boldsymbol{x} = \boldsymbol{0}\} \subseteq \mathbb{R}^n,  \\
  \mathrm{Row}(\boldsymbol{A}) &:= \mathrm{col}(\boldsymbol{A}^{\top}) \subseteq \mathbb{R}^n, &
  \mathrm{null}(\boldsymbol{A}^{\top}) &:= \{\boldsymbol{y} \in \mathbb{R}^m : \boldsymbol{A}^{\top}\boldsymbol{y} = \boldsymbol{0}\} \subseteq \mathbb{R}^m.
\end{aligned}
$$

An arbitrary linear system $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ is consistent if and only if $\boldsymbol{b} \in \mathrm{col}(\boldsymbol{A})$. In overdetermined regression ($m > n$), noisy measurements generically force $\boldsymbol{b} \notin \mathrm{col}(\boldsymbol{A})$, motivating the least-squares objective $\min_{\boldsymbol{x} \in \mathbb{R}^n} \lVert \boldsymbol{A}\boldsymbol{x} - \boldsymbol{b} \rVert_2^2$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, Sec. 7.7.3, p. 268).

### Unified Notation & Cross-Book Conventions

Throughout this leaf, column vectors are denoted by bold lowercase letters ($\boldsymbol{x} \in \mathbb{R}^n$), matrices by bold uppercase letters ($\boldsymbol{A} \in \mathbb{R}^{m \times n}$), inner products by $\langle \boldsymbol{x}, \boldsymbol{y} \rangle$ (defaulting to $\boldsymbol{x}^{\top}\boldsymbol{y}$ in Euclidean space), vector $\ell_p$ norms by $\lVert \boldsymbol{x} \rVert_p$, orthogonal projectors onto subspace $W$ by $\boldsymbol{P}_W$, the Moore–Penrose pseudoinverse by $\boldsymbol{A}^{+}$, and the $\ell_2$ condition number by $\kappa_2(\boldsymbol{A})$. Table `tab:vecnorm:notation` reconciles the divergent symbols across all five synthesized textbooks.

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Every source adopts slightly different typographical and domain conventions across linear algebra, machine learning, and deep learning. The table below maps each convention onto our unified monograph standard.
>
>
>
> **Table:** Cross-book notation mapping across [[Source - E D I T I O N|linear_algebra]] (Ch. 4, 6), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 2, 3), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, 11), [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 2), and [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 1).
>
> | **Concept** | **Lay et al. (2016)** | **Deisenroth et al. (2020)** | **Murphy (2022)** | **Goodfellow et al. (2016)** | **Vũ (2018) / Unified** |
> | — | — | — | — | — | — |
> | Column vector / Matrix | $\mathbf{x}, \mathbf{u}$; $A$ (italic) | $\boldsymbol{x}$; $\boldsymbol{A}$ (bold italic) | $\mathbf{x}$; $\mathbf{A}$ (bold upright) | $\boldsymbol{x}$; $\boldsymbol{A}$ (bold italic) | $\boldsymbol{x}$, $\boldsymbol{A}$ ($\boldsymbol{A}^{\mathsf{H}}$ in $\mathbb{C}$) |
> | Column / Null space | $\mathrm{Col} A$, $\mathrm{Nul} A$ | $\mathrm{Im}(\Phi)$, $\ker(\Phi)$ | $\mathrm{range}(\mathbf{A})$, $\mathrm{nullspace}(\mathbf{A})$ | Column space / range | $\mathrm{col}(\boldsymbol{A})$, $\mathrm{null}(\boldsymbol{A})$ |
> | Inner product | $\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^{\top}\mathbf{v}$ | $\langle \boldsymbol{x}, \boldsymbol{y}\rangle = \hat{\boldsymbol{x}}^{\top}\boldsymbol{A}\hat{\boldsymbol{y}}$ | $\langle \mathbf{x}, \mathbf{y}\rangle \triangleq \mathbf{x}^{\top}\mathbf{y}$ | $\boldsymbol{x}^{\top}\boldsymbol{y}$ | $\langle \boldsymbol{x}, \boldsymbol{y} \rangle$; $\boldsymbol{x}^{\top}\boldsymbol{y}$ in $\mathbb{R}^n$ |
> | Vector $\ell_p$ / Matrix norm | $\lVert \mathbf{v} \rVert$ ($\ell_2$ only) | $\lVert \boldsymbol{x} \rVert_1$, $\lVert \boldsymbol{x} \rVert_2$ | $\lVert \mathbf{x} \rVert_p$, $\lVert \mathbf{A} \rVert_p$, $\lVert \mathbf{A} \rVert_F$, $\lVert \mathbf{A} \rVert_*$ | $\lVert \boldsymbol{x} \rVert_p$ ($L^p$), $\lVert \boldsymbol{A} \rVert_F$ | $\lVert \boldsymbol{x} \rVert_p$, $\lVert \boldsymbol{A} \rVert_2$, $\lVert \boldsymbol{A} \rVert_F$ |
> | Orthogonal projection | $\hat{\mathbf{y}} = \mathrm{proj}_W\mathbf{y} = UU^{\top}\mathbf{y}$ | $\pi_U(\boldsymbol{x}) = \boldsymbol{P}_\pi\boldsymbol{x}$, $\boldsymbol{B}(\boldsymbol{B}^{\top}\boldsymbol{B})^{-1}\boldsymbol{B}^{\top}$ | $\mathrm{Proj}(\mathbf{y}; \mathbf{A}) = \mathbf{A}(\mathbf{A}^{\top}\mathbf{A})^{-1}\mathbf{A}^{\top}\mathbf{y}$ | $\boldsymbol{r}(\boldsymbol{x}) = \boldsymbol{D}\boldsymbol{D}^{\top}\boldsymbol{x}$ (PCA) | $\boldsymbol{P}_W\boldsymbol{y}$ ($= \boldsymbol{Q}\boldsymbol{Q}^{\top}\boldsymbol{y}$ for ONB) |
> | Pseudoinverse | $(A^{\top} A)^{-1}A^{\top}$ (full col. rank) | $(\boldsymbol{B}^{\top}\boldsymbol{B})^{-1}\boldsymbol{B}^{\top}$ | $\mathbf{A}^\dagger = \mathbf{V}\mathbf{S}^{-1}\mathbf{U}^{\top}$ | $\boldsymbol{A}^+ = \boldsymbol{V}\boldsymbol{D}^+\boldsymbol{U}^{\top}$ | $\boldsymbol{A}^{+} = \boldsymbol{V}\boldsymbol{\Sigma}^{+}\boldsymbol{U}^{\top}$ |
> | Least-squares model | $A\mathbf{x} \approx \mathbf{b}$; $X\boldsymbol{\beta} = \mathbf{y} + \boldsymbol{\varepsilon}$ | $\boldsymbol{B}\boldsymbol{\lambda} \approx \boldsymbol{x}$ (subspace fit) | $\mathbf{X}\mathbf{w} \approx \mathbf{y}$ ($\mathrm{RSS}(\mathbf{w})$) | $\min_{\boldsymbol{D}}\lVert \boldsymbol{X} - \boldsymbol{X}\boldsymbol{d}\boldsymbol{d}^{\top} \rVert_F^2$ | $\min_{\boldsymbol{x}}\lVert \boldsymbol{A}\boldsymbol{x} - \boldsymbol{b} \rVert_2^2$ |
>
>
> **Four Crucial Conceptual Reconciliations:**
> - **"$\ell_0$ Norm" Misnomer:** Goodfellow et al. explicitly caution that counting nonzero entries $\sum_i \mathbb{I}(x_i \neq 0)$ is *not* a norm because scaling $\boldsymbol{x}$ by $\alpha \neq 0$ leaves the count unchanged, violating $\lVert \alpha\boldsymbol{x} \rVert = |\alpha|\lVert \boldsymbol{x} \rVert$; Murphy labels $\lVert \boldsymbol{x} \rVert_0$ a "pseudo norm" for the same reason [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 2, Sec. 2.5, p. 40), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, Sec. 7.1.3, p. 235).
> - **"Orthogonal Matrix" Terminology:** All five texts warn that an *orthogonal matrix* $\boldsymbol{Q} \in \mathbb{R}^{n \times n}$ must be square with *orthonormal* columns ($\boldsymbol{Q}^{\top}\boldsymbol{Q} = \boldsymbol{Q}\boldsymbol{Q}^{\top} = \boldsymbol{I}_n$, so $\boldsymbol{Q}^{-1} = \boldsymbol{Q}^{\top}$). A tall matrix $\hat{\boldsymbol{Q}} \in \mathbb{R}^{m \times n}$ ($m > n$) with orthonormal columns satisfies $\hat{\boldsymbol{Q}}^{\top}\hat{\boldsymbol{Q}} = \boldsymbol{I}_n$ (left inverse), whereas $\hat{\boldsymbol{Q}}\hat{\boldsymbol{Q}}^{\top} \neq \boldsymbol{I}_m$ is instead the rank-$n$ orthogonal projector onto $\mathrm{col}(\hat{\boldsymbol{Q}})$ [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Sec. 6.2, p. 346), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, Def. 3.8, p. 78), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, Eq. 7.59, p. 241), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 1, Sec. 1.9, p. 21).
> - **General Inner Products vs. Dot Product:** Lay, Goodfellow et al., and Vũ largely specialize to the canonical Euclidean dot product $\boldsymbol{x}^{\top}\boldsymbol{y}$, whereas Deisenroth et al. treat general symmetric positive definite (SPD) bilinear forms $\langle \boldsymbol{x}, \boldsymbol{y} \rangle_{\boldsymbol{M}} = \boldsymbol{x}^{\top}\boldsymbol{M}\boldsymbol{y}$ ($\boldsymbol{M} \succ 0$), demonstrating that angles and orthogonality depend critically on the chosen inner product [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, Ex. 3.7, pp. 77–78).
> - **Algebraic vs. Numerical Least Squares:** While Lay introduces the normal equations $\boldsymbol{A}^{\top}\boldsymbol{A}\hat{\boldsymbol{x}} = \boldsymbol{A}^{\top}\boldsymbol{b}$ first and defers numerical stability warnings to a remark, Murphy and Goodfellow et al. stress that directly inverting $\boldsymbol{A}^{\top}\boldsymbol{A}$ or $\boldsymbol{A}$ is numerically hazardous and recommend QR or SVD solvers [[Source - E D I T I O N|linear_algebra]] (Ch. 6, p. 366), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 11, Sec. 11.2.2.3, p. 375), [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 2, Sec. 2.3, p. 37).

## Core Mathematical Derivations & Theorems

### Norms, SPD Inner Products, and Cauchy–Schwarz

> [!important] Vector Norms, $\ell_p$ Limit, and Matrix Operator/Frobenius Norms
> A function $\lVert \cdot \rVert : V \to \mathbb{R}$ is a *norm* if for all $\boldsymbol{x}, \boldsymbol{y} \in V$ and $\alpha \in \mathbb{R}$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, Def. 3.1, p. 71), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, Sec. 7.1.3, pp. 234–235):
> - **Positive definiteness:** $\lVert \boldsymbol{x} \rVert \ge 0$ and $\lVert \boldsymbol{x} \rVert = 0 \iff \boldsymbol{x} = \boldsymbol{0}$;
> - **Absolute homogeneity:** $\lVert \alpha \boldsymbol{x} \rVert = \lvert \alpha \rvert \lVert \boldsymbol{x} \rVert$;
> - **Triangle inequality:** $\lVert \boldsymbol{x} + \boldsymbol{y} \rVert \le \lVert \boldsymbol{x} \rVert + \lVert \boldsymbol{y} \rVert$.
>
> On $\mathbb{R}^n$, for any $p \in [1, \infty)$, the *$\ell_p$ norm* is defined as $\lVert \boldsymbol{x} \rVert_p := \bigl(\sum_{i=1}^n \lvert x_i \rvert^p\bigr)^{1/p}$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 2, Eq. 2.30, p. 39). Setting $m = \max_{1 \le j \le n} \lvert x_j \rvert > 0$, factoring out $m$ yields the $\ell_\infty$ limit [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 1, Eq. 1.39–1.41, p. 28):
>
> $$
> m \le \lVert \boldsymbol{x} \rVert_p = m \left( \sum_{j=1}^n \left(\frac{\lvert x_j \rvert}{m}\right)^p \right)^{1/p} \le m  n^{1/p}
>   \quad\Longrightarrow\quad
>   \lim_{p \to \infty} \lVert \boldsymbol{x} \rVert_p = \max_{1 \le j \le n} \lvert x_j \rvert = \lVert \boldsymbol{x} \rVert_\infty,
> $$
>
> since $\lim_{p\to\infty} n^{1/p} = 1$. Moreover, any two norms on $\mathbb{R}^n$ are topologically equivalent; for the canonical $\ell_1, \ell_2, \ell_\infty$ triple, Cauchy–Schwarz gives the sharp chain
>
> $$
> \lVert \boldsymbol{x} \rVert_\infty \le \lVert \boldsymbol{x} \rVert_2 \le \lVert \boldsymbol{x} \rVert_1 \le \sqrt{n} \lVert \boldsymbol{x} \rVert_2 \le n \lVert \boldsymbol{x} \rVert_\infty.
> $$
>
>
> For matrices $\boldsymbol{A} \in \mathbb{R}^{m \times n}$, two principal norms govern learning and stability [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, Sec. 7.1.3.2, pp. 235–236), [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 2, Eq. 2.33 & 2.49, pp. 40, 47):
>
> $$
> \begin{aligned}
> \lVert \boldsymbol{A} \rVert_F &:= \sqrt{\sum_{i=1}^m \sum_{j=1}^n A_{ij}^2} = \sqrt{\mathrm{Tr}(\boldsymbol{A}^{\top}\boldsymbol{A})} = \lVert \mathrm{vec}(\boldsymbol{A}) \rVert_2 = \sqrt{\sum_{k=1}^r \sigma_k(\boldsymbol{A})^2},  \\
>   \lVert \boldsymbol{A} \rVert_p &:= \sup_{\boldsymbol{x} \neq \boldsymbol{0}} \frac{\lVert \boldsymbol{A}\boldsymbol{x} \rVert_p}{\lVert \boldsymbol{x} \rVert_p} = \max_{\lVert \boldsymbol{x} \rVert_p = 1} \lVert \boldsymbol{A}\boldsymbol{x} \rVert_p, \qquad \lVert \boldsymbol{A} \rVert_2 = \sqrt{\lambda_{\max}(\boldsymbol{A}^{\top}\boldsymbol{A})} = \sigma_{\max}(\boldsymbol{A}),
> \end{aligned}
> $$
>
> where $\sigma_k(\boldsymbol{A})$ are the singular values of $\boldsymbol{A}$. Both $\lVert \cdot \rVert_F$ and $\lVert \cdot \rVert_2$ are *orthogonally invariant*: for any orthogonal $\boldsymbol{U} \in \mathbb{R}^{m \times m}$ and $\boldsymbol{V} \in \mathbb{R}^{n \times n}$, $\lVert \boldsymbol{U}\boldsymbol{A}\boldsymbol{V} \rVert_F = \lVert \boldsymbol{A} \rVert_F$ (via trace cyclicity $\mathrm{Tr}(\boldsymbol{V}^{\top}\boldsymbol{A}^{\top}\boldsymbol{U}^{\top}\boldsymbol{U}\boldsymbol{A}\boldsymbol{V}) = \mathrm{Tr}(\boldsymbol{A}^{\top}\boldsymbol{A})$) and $\lVert \boldsymbol{U}\boldsymbol{A}\boldsymbol{V} \rVert_2 = \lVert \boldsymbol{A} \rVert_2$.

While every inner product induces a norm via $\lVert \boldsymbol{x} \rVert = \sqrt{\langle \boldsymbol{x}, \boldsymbol{x} \rangle}$, the converse is false: a norm is induced by an inner product if and only if it satisfies the *parallelogram law* $\lVert \boldsymbol{u}+\boldsymbol{v} \rVert^2 + \lVert \boldsymbol{u}-\boldsymbol{v} \rVert^2 = 2\lVert \boldsymbol{u} \rVert^2 + 2\lVert \boldsymbol{v} \rVert^2$ [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Ex. 24, p. 339). Hence $\ell_2$ arises from an inner product, whereas $\ell_1$ and $\ell_\infty$ (for $n \ge 2$) do not [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, Sec. 3.3, p. 75).

> [!important] SPD Inner Products, Cauchy–Schwarz Inequality, and Pythagorean Theorem
> Let $V$ be an $n$-dimensional real vector space with ordered basis $\mathcal{B} = (\boldsymbol{b}_1, \dots, \boldsymbol{b}_n)$ and coordinate map $\boldsymbol{x} \mapsto \hat{\boldsymbol{x}} \in \mathbb{R}^n$. By bilinearity, symmetry, and positive definiteness, $\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$ is an inner product if and only if there exists a symmetric positive definite (SPD) Gram matrix $\boldsymbol{M} \in \mathbb{R}^{n \times n}$ ($M_{ij} = \langle \boldsymbol{b}_i, \boldsymbol{b}_j \rangle$, $\boldsymbol{M} = \boldsymbol{M}^{\top} \succ 0$) such that [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, Thm. 3.5, p. 74)
>
> $$
> \langle \boldsymbol{x}, \boldsymbol{y} \rangle_{\boldsymbol{M}} = \hat{\boldsymbol{x}}^{\top}\boldsymbol{M}\hat{\boldsymbol{y}}, \qquad \lVert \boldsymbol{x} \rVert_{\boldsymbol{M}} = \sqrt{\hat{\boldsymbol{x}}^{\top}\boldsymbol{M}\hat{\boldsymbol{x}}}.
> $$
>
>
> **Theorem (Cauchy–Schwarz Inequality).** *For any $\boldsymbol{u}, \boldsymbol{v}$ in an inner-product space $(V, \langle \cdot, \cdot \rangle)$, $\lvert \langle \boldsymbol{u}, \boldsymbol{v} \rangle \rvert \le \lVert \boldsymbol{u} \rVert\lVert \boldsymbol{v} \rVert$, with equality if and only if $\boldsymbol{u}$ and $\boldsymbol{v}$ are linearly dependent* [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, Eq. 3.17, p. 75).
>
> *Proof.* If $\boldsymbol{v} = \boldsymbol{0}$, both sides equal $0$. Otherwise $\lVert \boldsymbol{v} \rVert^2 > 0$; for any $t \in \mathbb{R}$, positive definiteness yields
> \[
>   0 \le \lVert \boldsymbol{u} - t\boldsymbol{v} \rVert^2 = \langle \boldsymbol{u} - t\boldsymbol{v}, \boldsymbol{u} - t\boldsymbol{v} \rangle = \lVert \boldsymbol{u} \rVert^2 - 2t\langle \boldsymbol{u}, \boldsymbol{v} \rangle + t^2\lVert \boldsymbol{v} \rVert^2.
> \]
> Setting $t = \langle \boldsymbol{u}, \boldsymbol{v} \rangle / \lVert \boldsymbol{v} \rVert^2$ (the orthogonal projection coordinate) simplifies the inequality to
> \[
>   0 \le \lVert \boldsymbol{u} \rVert^2 - \frac{\langle \boldsymbol{u}, \boldsymbol{v} \rangle^2}{\lVert \boldsymbol{v} \rVert^2} \quad\Longleftrightarrow\quad \langle \boldsymbol{u}, \boldsymbol{v} \rangle^2 \le \lVert \boldsymbol{u} \rVert^2\lVert \boldsymbol{v} \rVert^2,
> \]
> with equality iff $\boldsymbol{u} - t\boldsymbol{v} = \boldsymbol{0}$. Taking square roots gives $\lvert \langle \boldsymbol{u}, \boldsymbol{v} \rangle \rvert \le \lVert \boldsymbol{u} \rVert\lVert \boldsymbol{v} \rVert$ and immediately proves the triangle inequality $\lVert \boldsymbol{u}+\boldsymbol{v} \rVert^2 = \lVert \boldsymbol{u} \rVert^2 + 2\langle \boldsymbol{u}, \boldsymbol{v} \rangle + \lVert \boldsymbol{v} \rVert^2 \le (\lVert \boldsymbol{u} \rVert + \lVert \boldsymbol{v} \rVert)^2$, as well as defining the unique angle $\omega \in [0, \pi]$ via $\cos\omega = \langle \boldsymbol{u}, \boldsymbol{v} \rangle / (\lVert \boldsymbol{u} \rVert\lVert \boldsymbol{v} \rVert)$ for nonzero vectors. Two vectors are orthogonal ($\boldsymbol{u} \perp \boldsymbol{v}$) iff $\langle \boldsymbol{u}, \boldsymbol{v} \rangle = 0$, which yields the **Pythagorean Theorem** [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Thm. 2, p. 336):
>
> $$
> \lVert \boldsymbol{u} + \boldsymbol{v} \rVert^2 = \lVert \boldsymbol{u} \rVert^2 + \lVert \boldsymbol{v} \rVert^2 \iff \langle \boldsymbol{u}, \boldsymbol{v} \rangle = 0.
> $$

### The Four Fundamental Subspaces and Orthogonal Complements

For any subspace $W \subseteq \mathbb{R}^d$, its *orthogonal complement* is $W^\perp := \{\boldsymbol{z} \in \mathbb{R}^d : \langle \boldsymbol{z}, \boldsymbol{w} \rangle = 0\ \forall \boldsymbol{w} \in W\}$. Notice $W^\perp$ is itself a subspace satisfying $W \cap W^\perp = \{\boldsymbol{0}\}$ (since $\boldsymbol{v} \in W \cap W^\perp \implies \lVert \boldsymbol{v} \rVert^2 = \langle \boldsymbol{v}, \boldsymbol{v} \rangle = 0 \implies \boldsymbol{v} = \boldsymbol{0}$) [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Sec. 6.1, p. 336), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, Sec. 3.6, pp. 79–80).

> [!important] Fundamental Theorem of Linear Algebra (Rank–Nullity & Orthogonal Duality)
> Let $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ have rank $r = \mathrm{rank}(\boldsymbol{A})$. Then [[Source - E D I T I O N|linear_algebra]] (Ch. 4, Thm. 14, p. 235; Ch. 6, Thm. 3, p. 337), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 1, Sec. 1.7–1.8, pp. 19–20):
> - **Dimension equality:** $\dim\mathrm{col}(\boldsymbol{A}) = \dim\mathrm{Row}(\boldsymbol{A}) = r$, with $\dim\mathrm{null}(\boldsymbol{A}) = n - r$ and $\dim\mathrm{null}(\boldsymbol{A}^{\top}) = m - r$.
> - **Orthogonal complement duality (under the standard dot product):**
>
> $$
> \mathrm{Row}(\boldsymbol{A})^\perp = \mathrm{null}(\boldsymbol{A}) \subseteq \mathbb{R}^n, \qquad \mathrm{col}(\boldsymbol{A})^\perp = \mathrm{null}(\boldsymbol{A}^{\top}) \subseteq \mathbb{R}^m.
> $$
> - **Direct-sum decompositions:** $\mathbb{R}^n = \mathrm{Row}(\boldsymbol{A}) \oplus \mathrm{null}(\boldsymbol{A})$ and $\mathbb{R}^m = \mathrm{col}(\boldsymbol{A}) \oplus \mathrm{null}(\boldsymbol{A}^{\top})$.
>
>
> *Proof.* By the row–column rule, $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{0}$ holds iff every row $\boldsymbol{A}_{i,:}$ satisfies $\boldsymbol{A}_{i,:}\boldsymbol{x} = 0$. Since the rows span $\mathrm{Row}(\boldsymbol{A})$, $\boldsymbol{x} \in \mathrm{null}(\boldsymbol{A})$ iff $\boldsymbol{x}$ is orthogonal to every vector in $\mathrm{Row}(\boldsymbol{A})$, establishing $\mathrm{Row}(\boldsymbol{A})^\perp = \mathrm{null}(\boldsymbol{A})$. Applying the identical argument to $\boldsymbol{A}^{\top}$ and using $\mathrm{Row}(\boldsymbol{A}^{\top}) = \mathrm{col}(\boldsymbol{A})$ gives $\mathrm{col}(\boldsymbol{A})^\perp = \mathrm{null}(\boldsymbol{A}^{\top})$. Counting pivots and free columns in reduced row echelon form yields $r + \dim\mathrm{null}(\boldsymbol{A}) = n$; via the SVD $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$ (Section `sec:vecnorm:lsq_pinv`), the first $r$ columns of $\boldsymbol{V}$ and $\boldsymbol{U}$ span $\mathrm{Row}(\boldsymbol{A})$ and $\mathrm{col}(\boldsymbol{A})$, while the remaining $n-r$ columns of $\boldsymbol{V}$ and $m-r$ columns of $\boldsymbol{U}$ form orthonormal bases for $\mathrm{null}(\boldsymbol{A})$ and $\mathrm{null}(\boldsymbol{A}^{\top})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, Sec. 7.5.4, p. 262).

> [!tip] Action of $\boldsymbol{A}$ Across the Four Subspaces
> Think of $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ as splitting both its input and output spaces along perpendicular fault lines. Every input $\boldsymbol{x} \in \mathbb{R}^n$ decomposes uniquely as $\boldsymbol{x} = \boldsymbol{x}_r + \boldsymbol{x}_n$ with $\boldsymbol{x}_r \in \mathrm{Row}(\boldsymbol{A})$ and $\boldsymbol{x}_n \in \mathrm{null}(\boldsymbol{A})$. The operator $\boldsymbol{A}$ annihilates $\boldsymbol{x}_n$ ($\boldsymbol{A}\boldsymbol{x}_n = \boldsymbol{0}$) while mapping $\mathrm{Row}(\boldsymbol{A})$ *bijectively* (and invertibly!) onto $\mathrm{col}(\boldsymbol{A}) \subseteq \mathbb{R}^m$. Conversely, any target $\boldsymbol{b} \in \mathbb{R}^m$ decomposes into an attainable shadow $\hat{\boldsymbol{b}} \in \mathrm{col}(\boldsymbol{A})$ and an orthogonal residual $\boldsymbol{e} = \boldsymbol{b} - \hat{\boldsymbol{b}} \in \mathrm{null}(\boldsymbol{A}^{\top})$ that no choice of $\boldsymbol{x}$ can ever reach.

### Orthogonal Decomposition, Best Approximation, and Projector Algebra

> [!important] Orthogonal Decomposition, Best Approximation, and Projector Matrix
> Let $W \subseteq \mathbb{R}^m$ be a subspace with basis matrix $\boldsymbol{B} = [\boldsymbol{b}_1, \dots, \boldsymbol{b}_r] \in \mathbb{R}^{m \times r}$ (columns linearly independent).
>
> **Theorem (Orthogonal Decomposition & Best Approximation).** *Every $\boldsymbol{y} \in \mathbb{R}^m$ decomposes uniquely as $\boldsymbol{y} = \hat{\boldsymbol{y}} + \boldsymbol{z}$ with $\hat{\boldsymbol{y}} \in W$ and $\boldsymbol{z} \in W^\perp$. Furthermore, $\hat{\boldsymbol{y}}$ is the unique point in $W$ minimizing Euclidean distance to $\boldsymbol{y}$:*
>
> $$
> \hat{\boldsymbol{y}} = \arg\min_{\boldsymbol{v} \in W} \lVert \boldsymbol{y} - \boldsymbol{v} \rVert_2, \qquad \lVert \boldsymbol{y} - \boldsymbol{v} \rVert_2^2 = \lVert \boldsymbol{y} - \hat{\boldsymbol{y}} \rVert_2^2 + \lVert \hat{\boldsymbol{y}} - \boldsymbol{v} \rVert_2^2 > \lVert \boldsymbol{y} - \hat{\boldsymbol{y}} \rVert_2^2 \quad \forall \boldsymbol{v} \in W \setminus \{\hat{\boldsymbol{y}}\}.
> $$
>
> *Proof.* Writing $\hat{\boldsymbol{y}} = \boldsymbol{B}\boldsymbol{\lambda} \in W$, the orthogonality condition $\boldsymbol{y} - \hat{\boldsymbol{y}} \in W^\perp$ requires $\boldsymbol{b}_j^{\top}(\boldsymbol{y} - \boldsymbol{B}\boldsymbol{\lambda}) = 0$ for all $j = 1,\dots,r$, i.e., $\boldsymbol{B}^{\top}\boldsymbol{B}\boldsymbol{\lambda} = \boldsymbol{B}^{\top}\boldsymbol{y}$. Because $\boldsymbol{B}$ has linearly independent columns, the Gram matrix $\boldsymbol{B}^{\top}\boldsymbol{B} \in \mathbb{R}^{r \times r}$ is symmetric positive definite ($\boldsymbol{u}^{\top}\boldsymbol{B}^{\top}\boldsymbol{B}\boldsymbol{u} = \lVert \boldsymbol{B}\boldsymbol{u} \rVert_2^2 > 0$ for $\boldsymbol{u} \neq \boldsymbol{0}$) and invertible, giving $\boldsymbol{\lambda} = (\boldsymbol{B}^{\top}\boldsymbol{B})^{-1}\boldsymbol{B}^{\top}\boldsymbol{y}$ and the linear projection operator $\hat{\boldsymbol{y}} = \boldsymbol{P}_W\boldsymbol{y}$ [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Thm. 8–9, pp. 350–352), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, Eq. 3.55–3.59, pp. 86–87):
>
> $$
> \boldsymbol{P}_W = \boldsymbol{B}(\boldsymbol{B}^{\top}\boldsymbol{B})^{-1}\boldsymbol{B}^{\top}.
> $$
>
> For uniqueness, if $\boldsymbol{y} = \hat{\boldsymbol{y}}_1 + \boldsymbol{z}_1 = \hat{\boldsymbol{y}}_2 + \boldsymbol{z}_2$, then $\hat{\boldsymbol{y}}_1 - \hat{\boldsymbol{y}}_2 = \boldsymbol{z}_2 - \boldsymbol{z}_1 \in W \cap W^\perp = \{\boldsymbol{0}\}$. For any $\boldsymbol{v} \in W \setminus \{\hat{\boldsymbol{y}}\}$, writing $\boldsymbol{y} - \boldsymbol{v} = (\boldsymbol{y} - \hat{\boldsymbol{y}}) + (\hat{\boldsymbol{y}} - \boldsymbol{v})$ couples a vector in $W^\perp$ with a nonzero vector in $W$; applying the Pythagorean Theorem `(eq:vecnorm:pythagoras)` yields `(eq:vecnorm:best_approx)`.
>
> **Orthonormal Simplification & Algebraic Characterization.** If $\boldsymbol{Q} = [\boldsymbol{q}_1, \dots, \boldsymbol{q}_r] \in \mathbb{R}^{m \times r}$ is an orthonormal basis ($\boldsymbol{Q}^{\top}\boldsymbol{Q} = \boldsymbol{I}_r$), then `(eq:vecnorm:proj_general)` collapses without matrix inversion to [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Thm. 10, p. 353), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, Eq. 3.65, p. 88)
>
> $$
> \boldsymbol{P}_W = \boldsymbol{Q}\boldsymbol{Q}^{\top} = \sum_{j=1}^r \boldsymbol{q}_j \boldsymbol{q}_j^{\top}, \qquad \boldsymbol{P}_{W^\perp} = \boldsymbol{I}_m - \boldsymbol{Q}\boldsymbol{Q}^{\top}.
> $$
>
> A square matrix $\boldsymbol{P} \in \mathbb{R}^{m \times m}$ is an orthogonal projector onto $\mathrm{col}(\boldsymbol{P})$ if and only if it is *idempotent* ($\boldsymbol{P}^2 = \boldsymbol{P}$) and *symmetric* ($\boldsymbol{P}^{\top} = \boldsymbol{P}$). Its eigenvalues lie strictly in $\{0, 1\}$ ($\mathrm{rank}(\boldsymbol{P}) = \mathrm{Tr}(\boldsymbol{P}) = r$) and $\lVert \boldsymbol{P} \rVert_2 = 1$ when $r \ge 1$. For an affine subspace $L = \boldsymbol{x}_0 + U$, shifting to the origin gives $\pi_L(\boldsymbol{x}) = \boldsymbol{x}_0 + \boldsymbol{P}_U(\boldsymbol{x} - \boldsymbol{x}_0)$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, Eq. 3.72, p. 90).

Note the crucial distinction between *oblique* projections ($\boldsymbol{P}^2 = \boldsymbol{P}$, $\boldsymbol{P}^{\top} \neq \boldsymbol{P}$) and *orthogonal* projections ($\boldsymbol{P}^2 = \boldsymbol{P} = \boldsymbol{P}^{\top}$): only symmetry forces $\langle \boldsymbol{y} - \boldsymbol{P}\boldsymbol{y}, \boldsymbol{P}\boldsymbol{v} \rangle = \boldsymbol{y}^{\top}(\boldsymbol{I} - \boldsymbol{P})^{\top}\boldsymbol{P}\boldsymbol{v} = \boldsymbol{y}^{\top}(\boldsymbol{P} - \boldsymbol{P}^2)\boldsymbol{v} = 0$, guaranteeing minimum $\ell_2$ error. Notice also that in PCA reconstruction [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 2, Sec. 2.12, pp. 48–51), encoding $\boldsymbol{c}^* = \boldsymbol{D}^{\top}\boldsymbol{x}$ and decoding $\boldsymbol{r}(\boldsymbol{x}) = \boldsymbol{D}\boldsymbol{D}^{\top}\boldsymbol{x}$ for orthonormal $\boldsymbol{D} \in \mathbb{R}^{n \times \ell}$ ($\boldsymbol{D}^{\top}\boldsymbol{D} = \boldsymbol{I}_\ell$) is literally the orthogonal projector `(eq:vecnorm:proj_onb)` minimizing $\lVert \boldsymbol{X} - \boldsymbol{X}\boldsymbol{D}\boldsymbol{D}^{\top} \rVert_F^2$.

### Least Squares, Minimum-Norm Systems, and Moore–Penrose Pseudoinverse

> [!important] Normal Equations, Underdetermined Least-Norm, and Moore–Penrose Pseudoinverse
> Consider $\boldsymbol{A}\boldsymbol{x} \approx \boldsymbol{b}$ with $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ and $\boldsymbol{b} \in \mathbb{R}^m$.
>
> **Case 1: Overdetermined / General Least Squares ($\min_{\boldsymbol{x}} \frac{1}{2}\lVert \boldsymbol{A}\boldsymbol{x} - \boldsymbol{b} \rVert_2^2$).**
> Let $f(\boldsymbol{x}) = \frac{1}{2}(\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b})^{\top}(\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}) = \frac{1}{2}\boldsymbol{x}^{\top}\boldsymbol{A}^{\top}\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}^{\top}\boldsymbol{A}\boldsymbol{x} + \frac{1}{2}\boldsymbol{b}^{\top}\boldsymbol{b}$. Because $f$ is convex with $\nabla^2 f(\boldsymbol{x}) = \boldsymbol{A}^{\top}\boldsymbol{A} \succeq 0$, a vector $\hat{\boldsymbol{x}}$ minimizes $f$ if and only if $\nabla f(\hat{\boldsymbol{x}}) = \boldsymbol{0}$, yielding the *normal equations* [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Thm. 13, p. 363), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, Eq. 7.234–7.235, p. 269; Ch. 11, Eq. 11.7–11.8, p. 373):
>
> $$
> \boldsymbol{A}^{\top}\boldsymbol{A}\hat{\boldsymbol{x}} = \boldsymbol{A}^{\top}\boldsymbol{b} \quad\Longleftrightarrow\quad \boldsymbol{A}^{\top}(\boldsymbol{b} - \boldsymbol{A}\hat{\boldsymbol{x}}) = \boldsymbol{0} \quad\Longleftrightarrow\quad \boldsymbol{b} - \boldsymbol{A}\hat{\boldsymbol{x}} \in \mathrm{null}(\boldsymbol{A}^{\top}) = \mathrm{col}(\boldsymbol{A})^\perp.
> $$
>
> Equivalently, $\boldsymbol{A}\hat{\boldsymbol{x}} = \hat{\boldsymbol{b}} = \boldsymbol{P}_{\mathrm{col}(\boldsymbol{A})}\boldsymbol{b}$ is always unique, while $\hat{\boldsymbol{x}}$ is unique iff $\mathrm{null}(\boldsymbol{A}) = \{\boldsymbol{0}\}$ ($\mathrm{rank}(\boldsymbol{A}) = n$), in which case $\boldsymbol{A}^{\top}\boldsymbol{A} \succ 0$ and $\hat{\boldsymbol{x}} = (\boldsymbol{A}^{\top}\boldsymbol{A})^{-1}\boldsymbol{A}^{\top}\boldsymbol{b}$ [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Thm. 14, p. 365).
>
> **Case 2: Underdetermined Least-Norm ($\min_{\boldsymbol{x}} \lVert \boldsymbol{x} \rVert_2$ s.t. $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$, $m < n$, $\mathrm{rank}(\boldsymbol{A}) = m$).**
> Every solution has the affine form $\boldsymbol{x}_p + \boldsymbol{z}$ ($\boldsymbol{z} \in \mathrm{null}(\boldsymbol{A})$). Forming the Lagrangian $\mathcal{L}(\boldsymbol{x}, \boldsymbol{\lambda}) = \frac{1}{2}\boldsymbol{x}^{\top}\boldsymbol{x} + \boldsymbol{\lambda}^{\top}(\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b})$ gives $\nabla_{\boldsymbol{x}}\mathcal{L} = \boldsymbol{x} + \boldsymbol{A}^{\top}\boldsymbol{\lambda} = \boldsymbol{0} \implies \boldsymbol{x} = -\boldsymbol{A}^{\top}\boldsymbol{\lambda} \in \mathrm{Row}(\boldsymbol{A}) = \mathrm{null}(\boldsymbol{A})^\perp$. Substituting into $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ yields $\boldsymbol{\lambda} = -(\boldsymbol{A}\boldsymbol{A}^{\top})^{-1}\boldsymbol{b}$, hence the unique minimum-norm solution [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, Eq. 7.222–7.228, p. 268):
>
> $$
> \hat{\boldsymbol{x}}_{\mathrm{LN}} = \boldsymbol{A}^{\top}(\boldsymbol{A}\boldsymbol{A}^{\top})^{-1}\boldsymbol{b}.
> $$
>
>
> **Case 3: Unified Moore–Penrose Pseudoinverse ($\boldsymbol{A}^{+}$).**
> For *any* $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ of arbitrary rank $r \le \min(m, n)$, the Moore–Penrose pseudoinverse $\boldsymbol{A}^{+} \in \mathbb{R}^{n \times m}$ is the unique matrix satisfying the four Penrose equations [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, Eq. 7.186, p. 261):
>
> $$
> \boldsymbol{A}\boldsymbol{A}^{+}\boldsymbol{A} = \boldsymbol{A}, \qquad \boldsymbol{A}^{+}\boldsymbol{A}\boldsymbol{A}^{+} = \boldsymbol{A}^{+}, \qquad (\boldsymbol{A}\boldsymbol{A}^{+})^{\top} = \boldsymbol{A}\boldsymbol{A}^{+}, \qquad (\boldsymbol{A}^{+}\boldsymbol{A})^{\top} = \boldsymbol{A}^{+}\boldsymbol{A}.
> $$
>
> Using the thin SVD $\boldsymbol{A} = \boldsymbol{U}_r \boldsymbol{\Sigma}_r \boldsymbol{V}_r^{\top}$ ($\sigma_1 \ge \dots \ge \sigma_r > 0$), $\boldsymbol{A}^{+}$ equals both the Tikhonov limit and the explicit spectral formula [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 2, Eq. 2.46–2.47, p. 46), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, Eq. 7.191, p. 261):
>
> $$
> \boldsymbol{A}^{+} = \lim_{\alpha \searrow 0} (\boldsymbol{A}^{\top}\boldsymbol{A} + \alpha\boldsymbol{I}_n)^{-1}\boldsymbol{A}^{\top} = \boldsymbol{V}_r \boldsymbol{\Sigma}_r^{-1} \boldsymbol{U}_r^{\top} = \sum_{k=1}^r \frac{1}{\sigma_k} \boldsymbol{v}_k \boldsymbol{u}_k^{\top}.
> $$
>
> **Theorem (Minimum-Norm Least-Squares Property).** *For any $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ and $\boldsymbol{b} \in \mathbb{R}^m$, $\hat{\boldsymbol{x}}^* = \boldsymbol{A}^{+}\boldsymbol{b}$ is the unique vector minimizing $\lVert \boldsymbol{x} \rVert_2$ among all minimizers of $\lVert \boldsymbol{A}\boldsymbol{x} - \boldsymbol{b} \rVert_2$.*
>
> *Proof.* By `(eq:vecnorm:penrose_axioms)`, $\boldsymbol{A}\boldsymbol{A}^{+} = \boldsymbol{U}_r\boldsymbol{U}_r^{\top} = \boldsymbol{P}_{\mathrm{col}(\boldsymbol{A})}$ and $\boldsymbol{A}^{+}\boldsymbol{A} = \boldsymbol{V}_r\boldsymbol{V}_r^{\top} = \boldsymbol{P}_{\mathrm{Row}(\boldsymbol{A})}$. Any least-squares minimizer $\boldsymbol{x}$ satisfies $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{P}_{\mathrm{col}(\boldsymbol{A})}\boldsymbol{b} = \boldsymbol{A}\hat{\boldsymbol{x}}^*$, so $\boldsymbol{x} - \hat{\boldsymbol{x}}^* \in \mathrm{null}(\boldsymbol{A})$. Since $\hat{\boldsymbol{x}}^* = \boldsymbol{V}_r\boldsymbol{\Sigma}_r^{-1}\boldsymbol{U}_r^{\top}\boldsymbol{b} \in \mathrm{Row}(\boldsymbol{A}) = \mathrm{null}(\boldsymbol{A})^\perp$, Pythagoras `(eq:vecnorm:pythagoras)` gives $\lVert \boldsymbol{x} \rVert_2^2 = \lVert \hat{\boldsymbol{x}}^* \rVert_2^2 + \lVert \boldsymbol{x} - \hat{\boldsymbol{x}}^* \rVert_2^2 \ge \lVert \hat{\boldsymbol{x}}^* \rVert_2^2$, with strict inequality whenever $\boldsymbol{x} \neq \hat{\boldsymbol{x}}^*$.

## Algorithmic Implementation & Comparative Synthesis

Solving `(eq:vecnorm:normal_eq)` numerically requires transforming $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ ($m \ge n$) into an orthonormal basis without squaring the condition number. If $\boldsymbol{A} = \hat{\boldsymbol{Q}}\hat{\boldsymbol{R}}$ is an economy QR factorization where $\hat{\boldsymbol{Q}} \in \mathbb{R}^{m \times n}$ satisfies $\hat{\boldsymbol{Q}}^{\top}\hat{\boldsymbol{Q}} = \boldsymbol{I}_n$ and $\hat{\boldsymbol{R}} \in \mathbb{R}^{n \times n}$ is upper triangular with $R_{kk} > 0$, substituting into `(eq:vecnorm:normal_eq)` yields $\hat{\boldsymbol{R}}^{\top}\hat{\boldsymbol{Q}}^{\top}\hat{\boldsymbol{Q}}\hat{\boldsymbol{R}}\hat{\boldsymbol{x}} = \hat{\boldsymbol{R}}^{\top}\hat{\boldsymbol{Q}}^{\top}\boldsymbol{b}$, which simplifies (since $\hat{\boldsymbol{R}}^{\top}$ is nonsingular) directly to the triangular system [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Thm. 15 & Eq. 7, p. 367), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 11, Eq. 11.19–11.21, p. 375):

$$
\hat{\boldsymbol{R}}\hat{\boldsymbol{x}} = \hat{\boldsymbol{Q}}^{\top}\boldsymbol{b},
$$

solved cleanly by back-substitution in $\mathcal{O}(n^2)$ operations.

### Modified Gram–Schmidt vs. Householder Reflection QR

Gram–Schmidt constructs $\hat{\boldsymbol{Q}}$ column-by-column via successive projections onto $\mathrm{span}\{\boldsymbol{q}_1,\dots,\boldsymbol{q}_{k-1}\}^\perp$ [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Thm. 11–12, pp. 357–359), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, Sec. 3.8.3, p. 89). Crucially, **Classical Gram–Schmidt (CGS)** computes all inner products $r_{jk} = \boldsymbol{q}_j^{\top}\boldsymbol{a}_k$ against the original column $\boldsymbol{a}_k$ in batch, allowing floating-point roundoff to destroy orthogonality; **Modified Gram–Schmidt (MGS)** (Algorithm `alg:vecnorm:mgs`) immediately subtracts each rank-1 projection from the *remaining* columns $\boldsymbol{v}_j$ so rounding errors from earlier directions are re-projected out.

> [!example] Algorithm: Modified Gram–Schmidt (MGS) Economy QR Factorization
> **Input:** Full-column-rank matrix $\boldsymbol{A} = [\boldsymbol{a}_1, \dots, \boldsymbol{a}_n] \in \mathbb{R}^{m \times n}$ ($m \ge n$).
> **Output:** Orthonormal $\hat{\boldsymbol{Q}} \in \mathbb{R}^{m \times n}$ ($\hat{\boldsymbol{Q}}^{\top}\hat{\boldsymbol{Q}} = \boldsymbol{I}_n$) and upper triangular $\hat{\boldsymbol{R}} \in \mathbb{R}^{n \times n}$ with $\boldsymbol{A} = \hat{\boldsymbol{Q}}\hat{\boldsymbol{R}}$.
> Initialize $\boldsymbol{V} \leftarrow \boldsymbol{A}$, $\hat{\boldsymbol{R}} \leftarrow \boldsymbol{0}_{n \times n}$
> **for** $k = 1, \dots, n$:
>
>   $R_{kk} \leftarrow \lVert \boldsymbol{v}_k \rVert_2$
>   $\boldsymbol{q}_k \leftarrow \boldsymbol{v}_k / R_{kk}$  *// Normalize residual; CGS would keep $\boldsymbol{a}_k$ fixed*
>   **for** $j = k+1, \dots, n$:
>
>     $R_{kj} \leftarrow \boldsymbol{q}_k^{\top} \boldsymbol{v}_j$  *// Project updated column $\boldsymbol{v}_j$, not $\boldsymbol{a}_j$*
>     $\boldsymbol{v}_j \leftarrow \boldsymbol{v}_j - R_{kj}\boldsymbol{q}_k$  *// Remove component along $\boldsymbol{q}_k$ immediately*
>
>
> **return** $\hat{\boldsymbol{Q}} = [\boldsymbol{q}_1, \dots, \boldsymbol{q}_n]$, $\hat{\boldsymbol{R}}$

Although MGS improves orthogonality over CGS, finite-precision orthogonality of MGS still degrades proportionally to $\kappa_2(\boldsymbol{A})$. For unconditional machine-precision orthogonality $\lVert \boldsymbol{Q}^{\top}\boldsymbol{Q} - \boldsymbol{I} \rVert_2 = \mathcal{O}(\epsilon_{\mathrm{mach}})$, numerical linear algebra libraries (LAPACK `DGEQRF` / `DGELS`) left-multiply $\boldsymbol{A}$ by $n$ orthogonal **Householder reflectors** $\boldsymbol{H}_k = \boldsymbol{I}_m - \beta_k \tilde{\boldsymbol{v}}_k \tilde{\boldsymbol{v}}_k^{\top}$ (with $\beta_k = 2/\lVert \tilde{\boldsymbol{v}}_k \rVert_2^2$, $\boldsymbol{H}_k^{\top} = \boldsymbol{H}_k$, $\boldsymbol{H}_k^2 = \boldsymbol{I}_m$) to annihilate subdiagonal entries directly into $\boldsymbol{R} = \boldsymbol{H}_n \cdots \boldsymbol{H}_1 \boldsymbol{A}$ [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Sec. 6.4, p. 360). Choosing $s = \mathrm{sign}(z_1)\lVert \boldsymbol{z} \rVert_2$ and $v_1 = z_1 + s$ in Algorithm `alg:vecnorm:householder` (equivalently mapping $\boldsymbol{z} \mapsto -s \boldsymbol{e}_1$) enforces same-sign addition and prevents catastrophic cancellation.

> [!example] Algorithm: Householder QR Factorization & Back-Substitution Least-Squares Solver
> **Input:** Full-column-rank matrix $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ ($m \ge n$) and target vector $\boldsymbol{b} \in \mathbb{R}^m$.
> **Output:** Least-squares solution $\hat{\boldsymbol{x}} = \arg\min_{\boldsymbol{x}}\lVert \boldsymbol{A}\boldsymbol{x}-\boldsymbol{b} \rVert_2$ and residual norm $\lVert \boldsymbol{r} \rVert_2$.
> Initialize $\boldsymbol{R} \leftarrow \boldsymbol{A}$, $\boldsymbol{y} \leftarrow \boldsymbol{b}$
> **for** $k = 1, \dots, n$:
>
>   $\boldsymbol{z} \leftarrow \boldsymbol{R}_{k:m,  k}$  *// Subcolumn from diagonal $k$ to row $m$*
>   $s \leftarrow \mathrm{sign}(z_1)\lVert \boldsymbol{z} \rVert_2$ (taking $\mathrm{sign}(0) = +1$)
>   $\boldsymbol{v} \leftarrow \boldsymbol{z}$; \quad $v_1 \leftarrow z_1 + s$  *// Same-sign sum avoids cancellation; maps $\boldsymbol{z} \mapsto -s\boldsymbol{e}_1$*
>   $\beta \leftarrow 2 / (\boldsymbol{v}^{\top}\boldsymbol{v})$
>   $\boldsymbol{R}_{k:m,  k:n} \leftarrow \boldsymbol{R}_{k:m,  k:n} - \beta \boldsymbol{v}\bigl(\boldsymbol{v}^{\top} \boldsymbol{R}_{k:m,  k:n}\bigr)$  *// Apply $\boldsymbol{H}_k$ implicitly in $\mathcal{O}((m-k)(n-k))$*
>   $\boldsymbol{y}_{k:m} \leftarrow \boldsymbol{y}_{k:m} - \beta \boldsymbol{v}(\boldsymbol{v}^{\top} \boldsymbol{y}_{k:m})$  *// Accumulate $\boldsymbol{Q}^{\top}\boldsymbol{b}$ without forming $\boldsymbol{Q}$*
>
> Solve $\boldsymbol{R}_{1:n,  1:n}\hat{\boldsymbol{x}} = \boldsymbol{y}_{1:n}$ via back-substitution: $\hat{x}_i \leftarrow \bigl(y_i - \sum_{j=i+1}^n R_{ij}\hat{x}_j\bigr) / R_{ii}$ for $i = n, \dots, 1$
> **return** $\hat{\boldsymbol{x}}$ and $\lVert \boldsymbol{r} \rVert_2 = \lVert \boldsymbol{y}_{n+1:m} \rVert_2$

### Complexity, Accuracy, and Solver Selection Synthesis

Table `tab:vecnorm:solvers` summarizes the computational complexity and backward-stability hierarchy across direct least-squares methods for $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ ($m \ge n$).

**Table:** Comparison of direct least-squares solvers for $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ ($m \ge n$), where $\rho = \lVert \boldsymbol{b} - \boldsymbol{A}\hat{\boldsymbol{x}} \rVert_2 / (\lVert \boldsymbol{A} \rVert_2\lVert \hat{\boldsymbol{x}} \rVert_2)$ measures residual-to-signal scale. Derived from [[Source - E D I T I O N|linear_algebra]] (Ch. 6, pp. 360, 366–367) and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, p. 265; Ch. 11, p. 375).

| **Method** | **FLOPs ($m \ge n$)** | **Orthogonality Loss $\lVert \hat{\boldsymbol{Q}}^{\top}\hat{\boldsymbol{Q}} - \boldsymbol{I} \rVert_2$** | **Relative Forward Error ($\epsilon = \epsilon_{\mathrm{mach}}$)** | **Rank-Deficient / Regime Suitability** |
| — | — | — | — | — |
| Normal Eq. + Cholesky | $mn^2 + \tfrac{1}{3}n^3$ | N/A ($\boldsymbol{A}^{\top}\boldsymbol{A}$ formed) | $\mathcal{O}\bigl(\epsilon \kappa_2(\boldsymbol{A})^2\bigr)$ | Fails if $\mathrm{rank}(\boldsymbol{A}) < n$; fast only when $\kappa_2(\boldsymbol{A})$ is small |
| Classical GS (CGS) | $2mn^2$ | $\mathcal{O}\bigl(\epsilon \kappa_2(\boldsymbol{A})^2\bigr)$ | Unreliable (orthogonality lost) | Avoid in floating point without reorthogonalization |
| Modified GS (MGS) | $2mn^2$ | $\mathcal{O}\bigl(\epsilon \kappa_2(\boldsymbol{A})\bigr)$ | $\mathcal{O}\bigl(\epsilon [\kappa_2(\boldsymbol{A}) + \kappa_2(\boldsymbol{A})^2\rho]\bigr)$ | Good if $\boldsymbol{Q}$ needed explicitly; needs pivoting if rank-deficient |
| Householder QR | $2mn^2 - \tfrac{2}{3}n^3$ | $\mathcal{O}(\epsilon)$ (unconditional) | $\mathcal{O}\bigl(\epsilon [\kappa_2(\boldsymbol{A}) + \kappa_2(\boldsymbol{A})^2\rho]\bigr)$ | Preferred for full-rank tall/skinny systems ($m \gg n$) |
| Thin SVD ($\boldsymbol{A}^{+}\boldsymbol{b}$) | $\approx 2mn^2 + 11n^3$ | $\mathcal{O}(\epsilon)$ (for $\boldsymbol{U}, \boldsymbol{V}$) | $\mathcal{O}\bigl(\epsilon [\kappa_2(\boldsymbol{A}) + \kappa_2(\boldsymbol{A})^2\rho]\bigr)$ | Gold standard for rank-deficient or nearly singular $\boldsymbol{A}$ |

> [!info] Cross-Book Synthesis: Normal Equations, QR, SVD, and Iterative Solvers
> Synthesizing perspectives across the five sources yields a clear computational decision hierarchy for linear systems and least squares:
> - **Analytical vs. Numerical Viewpoint:** Lay et al. prioritize the normal equations $\boldsymbol{A}^{\top}\boldsymbol{A}\hat{\boldsymbol{x}} = \boldsymbol{A}^{\top}\boldsymbol{b}$ for pedagogical clarity and small exact examples, then highlight QR back-substitution $\hat{\boldsymbol{R}}\boldsymbol{x} = \hat{\boldsymbol{Q}}^{\top}\boldsymbol{b}$ as the reliable numerical alternative [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Sec. 6.5, pp. 363–367). Vũ pairs Gram matrices $\boldsymbol{B}^{\top}\boldsymbol{B} \succeq 0$ with Cholesky factorization $\boldsymbol{L}\boldsymbol{L}^{\top}$ when $\boldsymbol{A}^{\top}\boldsymbol{A}$ is strictly SPD [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 1, Sec. 1.13, p. 26).
> - **Production ML Solvers:** Murphy notes that `scipy.linalg.lstsq` (used inside `scikit-learn`) defaults to LAPACK's SVD driver `DGELSD` for rank robustness, recommends economy QR when $m \gg n$ (tall-skinny design matrices), and turns to matrix-free Krylov methods (Conjugate Gradient for SPD systems, GMRES for general sparse $\boldsymbol{A}$) when $n$ is too large for dense factorizations [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 11, Sec. 11.2.2.3, pp. 375–376).
> - **Regularized Stabilization:** Both Deisenroth et al. and Goodfellow et al. note that adding a small ridge/jitter term $\alpha\boldsymbol{I}$ to $\boldsymbol{A}^{\top}\boldsymbol{A}$ restores positive definiteness and yields the Tikhonov approximation whose limit as $\alpha \searrow 0$ recovers $\boldsymbol{A}^{+}$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, Sec. 3.8.2, pp. 86–87), [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 2, Eq. 2.46, p. 46).

## Visual Diagnostics & Geometric Structure

Figure `fig:vecnorm:four_subspaces` and Figure `fig:vecnorm:lsq_lay` anchor the geometric bridge between the four fundamental subspaces of $\boldsymbol{A}$ and least-squares projection, while Figure `fig:vecnorm:qr_murphy` contrasts full and economy QR shapes across tall and wide matrices.

> [!note] Source Figure
> *Figure:* The four fundamental subspaces determined by $\boldsymbol{A} \in \mathbb{R}^{m \times n}$: orthogonal complements $\mathrm{Row}(\boldsymbol{A}) \perp \mathrm{null}(\boldsymbol{A})$ in the domain $\mathbb{R}^n$ and $\mathrm{col}(\boldsymbol{A}) \perp \mathrm{null}(\boldsymbol{A}^{\top})$ in the codomain $\mathbb{R}^m$. Reproduced from [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Sec. 6.1, Fig. 8, p. 337). (`linear_algebra_p354_fig1.png`)

> [!note] Source Figure
> *Figure:* Geometric interpretation of the least-squares solution $\hat{\boldsymbol{x}} \in \mathbb{R}^n$ mapping via $\boldsymbol{A}$ to the closest point $\hat{\boldsymbol{b}} = \boldsymbol{A}\hat{\boldsymbol{x}} \in \mathrm{col}(\boldsymbol{A}) \subseteq \mathbb{R}^m$, with orthogonal residual $\boldsymbol{b} - \boldsymbol{A}\hat{\boldsymbol{x}} \in \mathrm{null}(\boldsymbol{A}^{\top})$. Reproduced from [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Sec. 6.5, Fig. 2, p. 363). (`linear_algebra_p380_fig2.png`)

> [!note] Source Figure
> *Figure:* Full vs. economy (reduced) $\boldsymbol{Q}\boldsymbol{R}$ decomposition $\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{R}$: (a) tall-skinny ($m > n$), where the shaded $m - n$ columns of $\boldsymbol{Q}$ span $\mathrm{null}(\boldsymbol{A}^{\top})$ and multiply zero rows of $\boldsymbol{R}$; (b) short-wide ($m < n$). Reproduced from [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, Sec. 7.6.2, Fig. 7.11, p. 265). (`probabilisticml_kevinmurphy_introduction_p295_fig1.png`)

Figure `fig:vecnorm:tikz_proj` unifies the geometric projection triangle in $\mathbb{R}^m$ (showing why $\boldsymbol{e} = \boldsymbol{b} - \boldsymbol{A}\hat{\boldsymbol{x}} \perp \mathrm{col}(\boldsymbol{A})$) with the algebraic four-subspace action of $\boldsymbol{A}$ and $\boldsymbol{A}^{+}$.

> [!example] Visual Diagram (LaTeX / PDF)
> *Geometric and algebraic structure of least-squares projection. (a) Any alternative $\boldsymbol{A}\boldsymbol{x} \in \mathrm{col}(\boldsymbol{A})$ satisfies $\lVert \boldsymbol{b} - \boldsymbol{A}\boldsymbol{x} \rVert_2^2 = \lVert \boldsymbol{e} \rVert_2^2 + \lVert \boldsymbol{A}\hat{\boldsymbol{x}} - \boldsymbol{A}\boldsymbol{x} \rVert_2^2$. (b) Across the four fundamental subspaces, $\boldsymbol{A}$ restricts to a bijection between $\mathrm{Row}(\boldsymbol{A})$ and $\mathrm{col}(\boldsymbol{A})$ whose inverse is $\boldsymbol{A}^{+}$, while collapsing $\mathrm{null}(\boldsymbol{A})$ and $\mathrm{null}(\boldsymbol{A}^{\top})$ to zero.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:vecnorm:pgf_norms` visualizes the geometry of $\ell_p$ unit balls in $\mathbb{R}^2$ alongside the quadratic-form level-set distortion induced by squaring $\boldsymbol{A}$ into the normal-equation Hessian $\boldsymbol{A}^{\top}\boldsymbol{A}$.

> [!example] Visual Diagram (LaTeX / PDF)
> *Norm geometry and conditioning. (a) Unit spheres $\lVert \boldsymbol{x} \rVert_p = 1$ transition from non-convex ($p = 0.5$) to sparse diamond corners ($p = 1$), isotropic Euclidean circle ($p = 2$), and hypercube ($p = \infty$) [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, Fig. 3.3, p. 71), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 1, Fig. 1.2, p. 27). (b) For $\sigma_1 = 1.5, \sigma_2 = 0.7$, forming $\boldsymbol{A}^{\top}\boldsymbol{A}$ squares the eccentricity from $\kappa_2(\boldsymbol{A}) = 2.14$ to $\kappa_2(\boldsymbol{A}^{\top}\boldsymbol{A}) = 4.59$, shrinking the semi-axis along $\boldsymbol{v}_1$ to $1/\sigma_1^2$ and elongating $\boldsymbol{v}_2$ to $1/\sigma_2^2$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Normal Equations Square the Condition Number: $\kappa_2(\boldsymbol{A}^{\top}\boldsymbol{A}) = \kappa_2(\boldsymbol{A})^2$
> From $\kappa_2(\boldsymbol{A}) = \lVert \boldsymbol{A} \rVert_2 \lVert \boldsymbol{A}^{+} \rVert_2 = \sigma_{\max}(\boldsymbol{A}) / \sigma_{\min}(\boldsymbol{A})$ and $\boldsymbol{A}^{\top}\boldsymbol{A} = \boldsymbol{V}(\boldsymbol{\Sigma}^{\top}\boldsymbol{\Sigma})\boldsymbol{V}^{\top}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, Eq. 7.51 & 7.180, pp. 238, 260), the singular values square:
>
> $$
> \kappa_2(\boldsymbol{A}^{\top}\boldsymbol{A}) = \frac{\sigma_{\max}(\boldsymbol{A})^2}{\sigma_{\min}(\boldsymbol{A})^2} = \kappa_2(\boldsymbol{A})^2.
> $$
>
> **Failure Example (L\"auchli Matrix):** Take $\varepsilon = 10^{-8}$ in IEEE-754 double precision ($\epsilon_{\mathrm{mach}} \approx 2.22 \times 10^{-16}$):
> \[
>   \boldsymbol{A} = \begin{bmatrix} 1 & 1 \\ \varepsilon & 0 \\ 0 & \varepsilon \end{bmatrix}
>   \quad\Longrightarrow\quad
>   \boldsymbol{A}^{\top}\boldsymbol{A} = \begin{bmatrix} 1 + \varepsilon^2 & 1 \\ 1 & 1 + \varepsilon^2 \end{bmatrix}
>    \xrightarrow{\mathrm{fl}(1 + 10^{-16}) = 1} 
>   \mathrm{fl}(\boldsymbol{A}^{\top}\boldsymbol{A}) = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}.
> \]
> Although $\mathrm{rank}(\boldsymbol{A}) = 2$ with $\kappa_2(\boldsymbol{A}) = \sqrt{2 + \varepsilon^2}/\varepsilon \approx 1.41 \times 10^8$, rounding destroys independence; Cholesky on $\boldsymbol{A}^{\top}\boldsymbol{A}$ breaks near $\kappa_2(\boldsymbol{A}) \gtrsim \epsilon_{\mathrm{mach}}^{-1/2} \approx 10^8$, whereas Householder QR or SVD holds up to $\kappa_2(\boldsymbol{A}) \lesssim \epsilon_{\mathrm{mach}}^{-1} \approx 10^{16}$ [[Source - E D I T I O N|linear_algebra]] (Ch. 6, p. 366), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 11, Sec. 11.2.2.3, p. 375).

> [!warning] Classical vs. Modified Gram–Schmidt Loss of Orthogonality
> In exact arithmetic, CGS and MGS (Algorithm `alg:vecnorm:mgs`) agree, yet in finite precision CGS evaluates $R_{jk} = \boldsymbol{q}_j^{\top}\boldsymbol{a}_k$ against the unreduced column $\boldsymbol{a}_k$ and subtracts $\sum_{j=1}^{k-1} R_{jk}\boldsymbol{q}_j$ simultaneously. For nearly collinear columns, cancellation leaves residual components along $\boldsymbol{q}_1,\dots,\boldsymbol{q}_{k-1}$, degrading orthogonality to $\lVert \hat{\boldsymbol{Q}}^{\top}\hat{\boldsymbol{Q}} - \boldsymbol{I}_n \rVert_2 \sim \mathcal{O}(\epsilon_{\mathrm{mach}}\kappa_2(\boldsymbol{A})^2)$. MGS reduces this drift to $\mathcal{O}(\epsilon_{\mathrm{mach}}\kappa_2(\boldsymbol{A}))$ by projecting updated vectors sequentially, while Householder QR (Algorithm `alg:vecnorm:householder`) guarantees $\mathcal{O}(\epsilon_{\mathrm{mach}})$ orthogonality unconditionally [[Source - E D I T I O N|linear_algebra]] (Ch. 6, Numerical Note 1, p. 360).

> [!warning] Rank Deficiency, Explicit Inversion, and Inner-Product Mismatch
> - **Near-Zero Singular Values:** When $\sigma_k \approx 0$ (collinear features or $m < n$), $(\boldsymbol{A}^{\top}\boldsymbol{A})^{-1}$ diverges and `(eq:vecnorm:pinv_svd)` amplifies noise by $\sigma_k^{-1}$. Libraries truncate $\sigma_k < \max(m, n) \epsilon_{\mathrm{mach}} \sigma_1$ (LAPACK `DGELSD` / `scipy.linalg.lstsq`) or apply $\ell_2$ ridge $(\boldsymbol{A}^{\top}\boldsymbol{A} + \lambda\boldsymbol{I}_n)^{-1}\boldsymbol{A}^{\top}\boldsymbol{b}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 11, Sec. 11.2.2.3 & 11.3, pp. 375, 381).
> - **Never Materialize $\boldsymbol{R}^{-1}$ or $(\boldsymbol{A}^{\top}\boldsymbol{A})^{-1}$:** Explicit inversion wastes $\mathcal{O}(n^3)$ work and loses accuracy compared to triangular back-substitution [[Source - E D I T I O N|linear_algebra]] (Ch. 6, p. 367), [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 2, Sec. 2.3, p. 37), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 7, Sec. 7.7.1, p. 267).
> - **General SPD Inner-Product Mismatch:** Under metric $\boldsymbol{M} \succ 0$ ($\langle \boldsymbol{u}, \boldsymbol{v} \rangle_{\boldsymbol{M}} = \boldsymbol{u}^{\top}\boldsymbol{M}\boldsymbol{v}$), $\boldsymbol{A}(\boldsymbol{A}^{\top}\boldsymbol{A})^{-1}\boldsymbol{A}^{\top}$ is *not* $\boldsymbol{M}$-orthogonal; enforcing $\boldsymbol{A}^{\top}\boldsymbol{M}(\boldsymbol{b} - \boldsymbol{A}\hat{\boldsymbol{x}}) = \boldsymbol{0}$ gives $\boldsymbol{P}_{\boldsymbol{M}} = \boldsymbol{A}(\boldsymbol{A}^{\top}\boldsymbol{M}\boldsymbol{A})^{-1}\boldsymbol{A}^{\top}\boldsymbol{M}$, reducing to Weighted Least Squares for $\boldsymbol{M} = \mathrm{diag}(1/\sigma_1^2, \dots, 1/\sigma_m^2)$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 3, Sec. 3.4 & 3.8, pp. 77, 86), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 11, Eq. 11.24, p. 376).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (5):** [[Cauchy-Schwarz Inequality]], [[Inner Product & Norms]], [[Orthogonal Projection]], [[QR Factorization]], [[Vector Space & Subspaces]]
- **Prerequisites (Directed $\leftarrow$):** [[Inner Product & Norms]], [[Orthogonal Projection]], [[Vector Space & Subspaces]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Cauchy-Schwarz Inequality]], [[Eigendecomposition & Spectral Theorem]], [[Inner Product & Norms]], [[Linear & Ridge Regression]], [[Mahalanobis Distance]], [[Matrix Calculus & Jacobians]], [[Mercer Kernels & RKHS]], [[Orthogonal Projection]], [[QR Factorization]], [[Rademacher Complexity & VC Dimension]], [[Singular Value Decomposition (SVD)]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Cauchy-Schwarz Inequality]], [[Condition Number]], [[Inner Product & Norms]], [[Linear & Ridge Regression]], [[Mahalanobis Distance]], [[Orthogonal Projection]], [[QR Factorization]], [[Rademacher Complexity & VC Dimension]], [[Vector Space & Subspaces]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - E D I T I O N]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
