---
type: monograph
domain: "D01"
parent_topic: "[[PT 01 - Convex, Smooth & Stochastic Optimization]]"
sub_topic: "[[ST 01 - Convex Duality & Deterministic Gradient Descent Theory]]"
cheatsheet: "[[Cheatsheet - Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]"
part_of: 
  - "[[ST 01 - Convex Duality & Deterministic Gradient Descent Theory]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
concepts: 
  - "[[Bregman & Legendre Duality]]"
  - "[[Convexity & Subgradients]]"
  - "[[Lagrange Duality & KKT]]"
prerequisites: 
  - "[[Convexity & Subgradients]]"
enables: 
  - "[[Bregman & Legendre Duality]]"
  - "[[Gradient Descent]]"
  - "[[Kantorovich Duality]]"
  - "[[Lagrange Duality & KKT]]"
  - "[[Proximal & Subgradient Methods]]"
  - "[[Support Vector Machines (SVM)]]"
  - "[[TRPO]]"
related: 
  - "[[Convexity & Subgradients]]"
  - "[[Gradient Descent]]"
  - "[[Kantorovich Duality]]"
  - "[[Lagrange Duality & KKT]]"
  - "[[Proximal & Subgradient Methods]]"
  - "[[Sinkhorn Algorithm]]"
  - "[[Support Vector Machines (SVM)]]"
  - "[[TRPO]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - monograph
  - domain-01
---

# Convex Sets, Fenchel/Lagrange Duality & KKT Conditions

> [!abstract] Monograph Navigation
> - **Domain:** [[D01 - Mathematical Foundations for Machine Intelligence|Mathematical Foundations for Machine Intelligence]]
> - **Parent / Sub-Topic:** [[PT 01 - Convex, Smooth & Stochastic Optimization|Convex, Smooth & Stochastic Optimization]] → [[ST 01 - Convex Duality & Deterministic Gradient Descent Theory|Convex Duality & Deterministic Gradient Descent Theory]]
> - **Visual Cheatsheet:** [[Cheatsheet - Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]
> - **Description:** Presents convex sets/functions, first/second-order convexity characterizations, Lagrangian duality, weak vs. strong duality (Slater's condition), and KKT complementary slackness.

## Problem Formulation & Unified Notation

Constrained optimization underpins nearly every structural estimator in statistical machine learning and modern artificial intelligence, from maximum-margin Support Vector Machines (SVMs) and $\ell_1$-constrained Lasso regression to simplex-constrained exponential-family inference and trust-region policy updates. We consider the canonical finite-dimensional optimization program in $\mathbb{R}^n$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 24, p. 303), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 7.2, p. 235), and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.5, p. 302):

$$
\begin{aligned}
    \mathrm{minimize}_{\boldsymbol{x} \in \mathcal{D}} \quad & f_0(\boldsymbol{x}) \\
    \text{subject to} \quad & f_i(\boldsymbol{x}) \le 0, \qquad i = 1, \dots, m, \\
                             & h_j(\boldsymbol{x}) = 0, \qquad j = 1, \dots, p,
  \end{aligned}
$$

where $\mathcal{D} := \bigl(\bigcap_{i=0}^m \operatorname{dom} f_i\bigr) \cap \bigl(\bigcap_{j=1}^p \operatorname{dom} h_j\bigr) \subseteq \mathbb{R}^n$ is the nonempty *problem domain*, $f_0 : \mathcal{D} \to \mathbb{R}$ is the *objective (loss) function*, $f_i : \mathcal{D} \to \mathbb{R}$ are *inequality constraint functions*, and $h_j : \mathcal{D} \to \mathbb{R}$ are *equality constraint functions*. The *feasible set* (constraint set) is

$$
\mathcal{X} := \{\boldsymbol{x} \in \mathcal{D} : f_i(\boldsymbol{x}) \le 0 \forall i \in \{1,\dots,m\},   h_j(\boldsymbol{x}) = 0 \forall j \in \{1,\dots,p\}\},
$$

and the *primal optimal value* is $p^* := \inf_{\boldsymbol{x} \in \mathcal{X}} f_0(\boldsymbol{x}) \in [-\infty, +\infty]$, with $p^* = +\infty$ when $\mathcal{X} = \emptyset$ (infeasible) and $p^* = -\infty$ when $f_0$ is unbounded below on $\mathcal{X}$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 24, Sec. 24.2.1, p. 304). A point $\boldsymbol{x}^* \in \mathcal{X}$ is *primal optimal* if $f_0(\boldsymbol{x}^*) = p^*$. An inequality constraint $f_i(\boldsymbol{x}) \le 0$ is *active* at $\boldsymbol{x} \in \mathcal{X}$ if $f_i(\boldsymbol{x}) = 0$ and *inactive* if $f_i(\boldsymbol{x}) < 0$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 4, p. 95).

### Convex Optimization Subclasses and Canonical Transformations

Problem `(eq:convex:primal_std)` is a **convex optimization problem** if and only if (i) $f_0, f_1, \dots, f_m$ are convex functions on convex domain $\mathcal{D}$ and (ii) all equality constraints are affine, $h_j(\boldsymbol{x}) = \boldsymbol{a}_j^{\top} \boldsymbol{x} - b_j = 0$ (written compactly as $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ with $\boldsymbol{A} \in \mathbb{R}^{p \times n}$) [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 24, Def. 24.1, p. 305), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, Eq. 7.38, p. 239). Standard canonical subclasses include:
- **Linear Programs (LP):** $f_0(\boldsymbol{x}) = \boldsymbol{c}^{\top}\boldsymbol{x} + d$, $\boldsymbol{G}\boldsymbol{x} \preceq \boldsymbol{h}$, $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$, whose feasible set forms a convex polyhedron [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 24, p. 308) and whose optima occur at extreme vertices or boundary faces [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.5, p. 305).
- **Quadratic Programs (QP):** $f_0(\boldsymbol{x}) = \frac{1}{2}\boldsymbol{x}^{\top}\boldsymbol{Q}\boldsymbol{x} + \boldsymbol{c}^{\top}\boldsymbol{x} + r$ with symmetric positive semidefinite $\boldsymbol{Q} \succeq \boldsymbol{0}$ subject to affine constraints $\boldsymbol{G}\boldsymbol{x} \preceq \boldsymbol{h}$, $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, Sec. 7.3.2, p. 241), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8, Sec. 8.5.4, p. 306).
- **Geometric Programs (GP):** Problems with posynomial objectives/inequalities $f_i(\boldsymbol{x}) = \sum_{k=1}^{K_i} c_{ik}\prod_{\ell=1}^n x_\ell^{a_{ik\ell}} \le 1$ ($c_{ik}>0$) and monomial equalities on $\boldsymbol{x} \succ \boldsymbol{0}$; while non-convex in $\boldsymbol{x}$, the logarithmic change of variables $y_\ell = \ln x_\ell$ and log-sum-exp composition $\tilde{f}_i(\boldsymbol{y}) = \ln \sum_{k} \exp(\boldsymbol{a}_{ik}^{\top}\boldsymbol{y} + \ln c_{ik}) \le 0$ converts GP exactly into a convex program [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 24, Sec. 24.6, pp. 313–315).

### Unified Notation & Cross-Book Conventions

Throughout this leaf, we adopt the unified primal-dual notation $(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ with $f_0$ as objective, $f_i(\boldsymbol{x}) \le 0$ as inequalities ($\lambda_i \ge 0$), $h_j(\boldsymbol{x}) = 0$ as equalities ($\nu_j \in \mathbb{R}$), $L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ as the Lagrangian, $g(\boldsymbol{\lambda}, \boldsymbol{\nu})$ as the dual function, and $f^*(\boldsymbol{y})$ as the Fenchel–Legendre conjugate. Table `tab:convex:notation` reconciles the divergent symbols across all five synthesized textbooks.

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Because optimization texts, probabilistic ML references, and deep learning monographs inherit conventions from different mathematical communities, three symbol-role inversions and sign flips require explicit disambiguation before chaining derivations.
>
> **Table:** Cross-book notation reconciliation across [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 24–25), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8), [[Source - Foundations|deep_learning_foundations_concepts]] (App. C), and [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 4).
>
> | **Concept** | **Vũ (2018)** | **Deisenroth et al. (2020)** | **Murphy (2022)** | **Bishop & Bishop (2024)** | **Goodfellow et al. (2017) / Unified** |
> | --- | --- | --- | --- | --- | --- |
> | Primal objective | $\min f_0(\boldsymbol{x})$ | $\min f(\boldsymbol{x})$ | $\min \mathcal{L}(\boldsymbol{\theta})$ | $\max f(\boldsymbol{x})$ | $\min f(\boldsymbol{x})$ / $\min f_0(\boldsymbol{x})$ |
> | Inequality constraint | $f_i(\boldsymbol{x}) \le 0$ | $g_i(\boldsymbol{x}) \le 0$ | $g_j(\boldsymbol{\theta}) \le 0$ | $h_k(\boldsymbol{x}) \ge 0$ (flipped!) | $h^{(j)}(\boldsymbol{x}) \le 0$ / $f_i(\boldsymbol{x}) \le 0$ |
> | Equality constraint | $h_j(\boldsymbol{x}) = 0$ | $h_j(\boldsymbol{x}) = 0$ | $h_i(\boldsymbol{\theta}) = 0$ | $g_j(\boldsymbol{x}) = 0$ | $g^{(i)}(\boldsymbol{x}) = 0$ / $h_j(\boldsymbol{x}) = 0$ |
> | Dual multipliers | $\boldsymbol{\lambda} \succeq \boldsymbol{0}$ (ineq.), $\boldsymbol{\nu}$ (eq.) | $\boldsymbol{\lambda} \ge \boldsymbol{0}$ (ineq.) | $\boldsymbol{\mu} \ge \boldsymbol{0}$ (ineq.), $\boldsymbol{\lambda}$ (eq.) | $\mu_k \ge 0$ (ineq.), $\lambda_j$ (eq.) | $\boldsymbol{\alpha} \ge \boldsymbol{0}$ (ineq.), $\boldsymbol{\lambda}$ (eq.) / $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ |
> | Lagrangian | $L(\boldsymbol{x},\boldsymbol{\lambda},\boldsymbol{\nu})$ | $\mathfrak{L}(\boldsymbol{x},\boldsymbol{\lambda})$ | $L(\boldsymbol{\theta},\boldsymbol{\mu},\boldsymbol{\lambda})$ | $L(\boldsymbol{x},\{\lambda_j\},\{\mu_k\})$ | $L(\boldsymbol{x},\boldsymbol{\lambda},\boldsymbol{\alpha})$ / $L(\boldsymbol{x},\boldsymbol{\lambda},\boldsymbol{\nu})$ |
> | Dual function | $g(\boldsymbol{\lambda},\boldsymbol{\nu})$ | $\mathfrak{D}(\boldsymbol{\lambda})$ | Implicit in dual QP/LP | Not formulated | Implicit via $\min_{\boldsymbol{x}}\max_{\boldsymbol{\lambda},\boldsymbol{\alpha}\ge \boldsymbol{0}}L$ |
> | Fenchel conjugate | N/A (referenced) | $f^*(\boldsymbol{s}) = \sup_{\boldsymbol{x}}(\langle \boldsymbol{s}, \boldsymbol{x} \rangle - f(\boldsymbol{x}))$ | $f^*(\lambda) = \max_x \lambda x - f(x)$ | N/A | $f^*(\boldsymbol{y}) = \sup_{\boldsymbol{x}}(\langle \boldsymbol{y}, \boldsymbol{x} \rangle - f(\boldsymbol{x}))$ |
>
> **Four Critical Convention Reconciliations:**
> - **Objective Sense and Inequality Orientation:** Bishop & Bishop formulate *maximization* of $f(\boldsymbol{x})$ subject to $h_k(\boldsymbol{x}) \ge 0$, writing $L = f + \sum_j \lambda_j g_j + \sum_k \mu_k h_k$ with $\mu_k \ge 0$ so that $\nabla f = -\mu_k \nabla h_k$ points away from the feasible interior $h_k > 0$; when switching to *minimization* with $h_k(\boldsymbol{x}) \ge 0$, they explicitly flip the sign to $L = f - \mu_k h_k$ ($\mu_k \ge 0$) [[Source - Foundations|deep_learning_foundations_concepts]] (App. C, pp. 623–624). In our standard minimization form `(eq:convex:primal_std)` with $f_i(\boldsymbol{x}) \le 0$, replacing $f_i = -h_k$ yields $L = f_0 + \sum_i \lambda_i f_i$ ($\lambda_i \ge 0$), fully unifying all signs.
> - **$g$ vs. $h$ Role Inversion:** Deisenroth et al. and Murphy denote inequality constraints by $g$ and equalities by $h$, whereas Goodfellow et al. and Bishop & Bishop assign $g$ to equalities and $h$ to inequalities, and Vũ reserves $g(\boldsymbol{\lambda}, \boldsymbol{\nu})$ exclusively for the Lagrange dual function while labeling primal inequalities $f_i$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (p. 235), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (p. 304), [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (p. 94), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (p. 318).
> - **$\mathcal{L}$ vs. $L$ Overloading:** Murphy denotes the scalar objective (loss) as $\mathcal{L}(\boldsymbol{\theta})$ and the Lagrangian as $L(\boldsymbol{\theta}, \boldsymbol{\mu}, \boldsymbol{\lambda})$; we reserve $f_0$ for the primal objective and $L$ for the Lagrangian to prevent conflating unconstrained loss values with saddle objectives.
> - **Convex Conjugate Sign Convention:** Both Deisenroth et al. ($f^*(\boldsymbol{s})$) and Murphy ($f^*(\lambda)$) define the Fenchel–Legendre conjugate as $\sup_{\boldsymbol{x}}(\langle \boldsymbol{s}, \boldsymbol{x} \rangle - f(\boldsymbol{x}))$, which equals the *negated* intercept $-c$ of a supporting affine hyperplane $t = \langle \boldsymbol{s}, \boldsymbol{x} \rangle + c$ lying below $\operatorname{epi} f$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 7.3.3, pp. 242–243), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 8.4, p. 279).

## Core Mathematical Derivations & Theorems

### Convex Sets, Epigraphs, and Hyperplane Separation

A set $C \subseteq \mathbb{R}^n$ is *convex* if the closed line segment joining any two elements lies entirely within $C$: for all $\boldsymbol{x}_1, \boldsymbol{x}_2 \in C$ and all $\theta \in [0, 1]$, $\theta \boldsymbol{x}_1 + (1 - \theta)\boldsymbol{x}_2 \in C$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 23, Def. 23.1, p. 284), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, Def. 7.2, p. 236), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8, Eq. 8.6, p. 277). More generally, a point $\boldsymbol{x} = \sum_{k=1}^K \theta_k \boldsymbol{x}_k$ with $\theta_k \ge 0$ and $\sum_{k=1}^K \theta_k = 1$ is a *convex combination*, and the set of all finite convex combinations of $S \subseteq \mathbb{R}^n$ defines its *convex hull* $\operatorname{conv}(S)$, the smallest convex set containing $S$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 23, Def. 23.5, p. 287).

Fundamental convex primitives preserved under arbitrary intersections $\bigcap_{\alpha} C_\alpha$ and affine maps $\boldsymbol{x} \mapsto \boldsymbol{A}\boldsymbol{x} + \boldsymbol{b}$ include:
- **Hyperplanes and Halfspaces:** $\mathcal{H} = \{\boldsymbol{x} \in \mathbb{R}^n : \boldsymbol{a}^{\top}\boldsymbol{x} = b\}$ ($\boldsymbol{a} \neq \boldsymbol{0}$) and $\mathcal{H}_- = \{\boldsymbol{x} \in \mathbb{R}^n : \boldsymbol{a}^{\top}\boldsymbol{x} \le b\}$.
- **Polyhedra:** Finite intersections of halfspaces and hyperplanes $\mathcal{P} = \{\boldsymbol{x} \in \mathbb{R}^n : \boldsymbol{G}\boldsymbol{x} \preceq \boldsymbol{h},  \boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}\}$, forming the canonical feasible domains of linear programs [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 23, p. 285), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.5, p. 305).
- **Norm Balls:** $B_{\lVert \cdot \rVert}(\boldsymbol{x}_c, r) = \{\boldsymbol{x} \in \mathbb{R}^n : \lVert \boldsymbol{x} - \boldsymbol{x}_c \rVert \le r\}$ for any valid norm ($\ell_p$ with $p \ge 1$), by the triangle inequality; by contrast, $\ell_p$ pseudo-norm balls with $0 < p < 1$ violate the triangle inequality and are non-convex [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 23, Fig. 23.3, p. 286).

> [!important] Hyperplane Separation, Supporting Hyperplanes & Epigraph Characterization
> **1. Strict Point–Set Separation & Separating Hyperplane Theorem.** Let $C \subseteq \mathbb{R}^n$ be nonempty, closed, and convex, and let $\boldsymbol{x}_0 \notin C$. Define the Euclidean projection $\boldsymbol{p} := \arg\min_{\boldsymbol{z} \in C} \lVert \boldsymbol{z} - \boldsymbol{x}_0 \rVert_2^2$, which exists because $C$ is closed and $r(\boldsymbol{z}) = \lVert \boldsymbol{z} - \boldsymbol{x}_0 \rVert_2^2$ is coercive. Set $\boldsymbol{a} := \boldsymbol{x}_0 - \boldsymbol{p} \neq \boldsymbol{0}$. For any $\boldsymbol{z} \in C$ and $t \in (0, 1]$, convexity of $C$ gives $\boldsymbol{p} + t(\boldsymbol{z} - \boldsymbol{p}) = (1-t)\boldsymbol{p} + t\boldsymbol{z} \in C$, so optimality of $\boldsymbol{p}$ implies
>
> $$
> \lVert \boldsymbol{p} - \boldsymbol{x}_0 \rVert_2^2 \le \lVert \boldsymbol{p} + t(\boldsymbol{z} - \boldsymbol{p}) - \boldsymbol{x}_0 \rVert_2^2
>   = \lVert \boldsymbol{p} - \boldsymbol{x}_0 \rVert_2^2 + 2t \langle \boldsymbol{p} - \boldsymbol{x}_0, \boldsymbol{z} - \boldsymbol{p} \rangle + t^2 \lVert \boldsymbol{z} - \boldsymbol{p} \rVert_2^2.
> $$
>
> Subtracting $\lVert \boldsymbol{p} - \boldsymbol{x}_0 \rVert_2^2$, dividing by $2t > 0$, and letting $t \searrow 0$ yields $\langle \boldsymbol{a}, \boldsymbol{z} - \boldsymbol{p} \rangle \le 0$ for all $\boldsymbol{z} \in C$. Since $\langle \boldsymbol{a}, \boldsymbol{x}_0 - \boldsymbol{p} \rangle = \lVert \boldsymbol{a} \rVert_2^2 > 0$, choosing $b = \langle \boldsymbol{a}, \boldsymbol{p} \rangle + \frac{1}{2}\lVert \boldsymbol{a} \rVert_2^2$ strictly separates $\boldsymbol{x}_0$ from $C$: $\langle \boldsymbol{a}, \boldsymbol{z} \rangle < b < \langle \boldsymbol{a}, \boldsymbol{x}_0 \rangle$ for all $\boldsymbol{z} \in C$. Applying this separation to the Minkowski difference $C - D$ and $\boldsymbol{0}$ (with closure/boundary approximation when sets touch) establishes the **Separating Hyperplane Theorem**: two nonempty convex sets $C, D \subseteq \mathbb{R}^n$ with $C \cap D = \emptyset$ admit $(\boldsymbol{a} \neq \boldsymbol{0}, b)$ satisfying $\boldsymbol{a}^{\top}\boldsymbol{x} \le b  \forall \boldsymbol{x} \in C$ and $\boldsymbol{a}^{\top}\boldsymbol{x} \ge b  \forall \boldsymbol{x} \in D$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 23, Thm. 23.1, p. 288). Taking a sequence $\boldsymbol{x}_0^{(k)} \notin C$ converging to a boundary point $\boldsymbol{x}_0 \in \partial C$ and normalizing $\boldsymbol{a}^{(k)} / \lVert \boldsymbol{a}^{(k)} \rVert_2$ gives a convergent subsequence $\boldsymbol{a}^* \neq \boldsymbol{0}$ establishing the **Supporting Hyperplane Theorem**: at every boundary point $\boldsymbol{x}_0 \in \partial C$ of a convex set $C$, there exists a hyperplane supporting $C$, i.e., $\langle \boldsymbol{a}^*, \boldsymbol{x} \rangle \le \langle \boldsymbol{a}^*, \boldsymbol{x}_0 \rangle$ for all $\boldsymbol{x} \in C$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, p. 242), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 24, p. 307).
>
> **2. Epigraph & Sublevel Characterization of Convex Functions.** Given $f : \operatorname{dom} f \to \mathbb{R}$, its *epigraph* and *$\alpha$-sublevel set* are [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, p. 236), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8, Sec. 8.1.3.2, p. 278), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 23, Def. 23.8, p. 294)
>
> $$
> \operatorname{epi} f := \{(\boldsymbol{x}, t) \in \operatorname{dom} f \times \mathbb{R} : f(\boldsymbol{x}) \le t\}, \qquad
>   \mathcal{C}_\alpha(f) := \{\boldsymbol{x} \in \operatorname{dom} f : f(\boldsymbol{x}) \le \alpha\}.
> $$
>
> *Theorem.* A function $f$ is convex if and only if $\operatorname{epi} f \subseteq \mathbb{R}^{n+1}$ is a convex set.
>
> *Proof.* ($\Rightarrow$) Let $(\boldsymbol{x}, t), (\boldsymbol{y}, s) \in \operatorname{epi} f$ and $\theta \in [0, 1]$. Since $\operatorname{dom} f$ is convex, $\theta\boldsymbol{x} + (1-\theta)\boldsymbol{y} \in \operatorname{dom} f$, and convexity of $f$ yields $f(\theta\boldsymbol{x} + (1-\theta)\boldsymbol{y}) \le \theta f(\boldsymbol{x}) + (1-\theta)f(\boldsymbol{y}) \le \theta t + (1-\theta)s$, so $\theta(\boldsymbol{x}, t) + (1-\theta)(\boldsymbol{y}, s) \in \operatorname{epi} f$. ($\Leftarrow$) Take $t = f(\boldsymbol{x})$ and $s = f(\boldsymbol{y})$; then $(\boldsymbol{x}, f(\boldsymbol{x})), (\boldsymbol{y}, f(\boldsymbol{y})) \in \operatorname{epi} f \implies (\theta\boldsymbol{x} + (1-\theta)\boldsymbol{y},  \theta f(\boldsymbol{x}) + (1-\theta)f(\boldsymbol{y})) \in \operatorname{epi} f$, proving $f(\theta\boldsymbol{x} + (1-\theta)\boldsymbol{y}) \le \theta f(\boldsymbol{x}) + (1-\theta)f(\boldsymbol{y})$. Moreover, intersecting $\operatorname{epi} f$ with the hyperplane $t = \alpha$ shows that every $\alpha$-sublevel set $\mathcal{C}_\alpha(f)$ of a convex function is convex, whereas the converse fails: functions with convex sublevel sets are *quasiconvex*, such as $f(x, y) = -e^{x+y}$ (concave, yet every sublevel set is a convex halfspace) [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 23, Thm. 23.2 & Def. 23.9, pp. 295–296).

### First/Second-Order Conditions, Jensen's Inequality & Optimality

A function $f : \operatorname{dom} f \to \mathbb{R}$ on a convex domain is *convex* if, for all $\boldsymbol{x}, \boldsymbol{y} \in \operatorname{dom} f$ and $\theta \in [0, 1]$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 23, p. 289), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, p. 236), and [[Source - Foundations|deep_learning_foundations_concepts]] (Eq. 2.101, p. 52),

$$
f(\theta\boldsymbol{x} + (1 - \theta)\boldsymbol{y}) \le \theta f(\boldsymbol{x}) + (1 - \theta) f(\boldsymbol{y}).
$$

It is *strictly convex* if strict inequality holds for $\boldsymbol{x} \neq \boldsymbol{y}$ and $\theta \in (0, 1)$, and *$m$-strongly convex* ($m > 0$) if $f(\boldsymbol{x}) - \frac{m}{2}\lVert \boldsymbol{x} \rVert_2^2$ is convex [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8, Eq. 8.10, p. 281). A function $f$ is (strictly) concave iff $-f$ is (strictly) convex. Convexity is preserved under nonnegative weighted sums $\sum_k \alpha_k f_k$ ($\alpha_k \ge 0$), pointwise supremum $\sup_{\alpha \in \mathcal{A}} f_\alpha(\boldsymbol{x})$, and composition with affine maps $\boldsymbol{x} \mapsto f(\boldsymbol{A}\boldsymbol{x} + \boldsymbol{b})$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 23, Sec. 23.3.2, pp. 289–290), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, Ex. 7.4, p. 238).

> [!important] Jensen's Inequality, Differential Convexity Criteria & Global Optimality
> **1. Jensen's Inequality.** By induction on $K \ge 2$ from `(eq:convex:chord_def)` (writing $\sum_{i=1}^K \theta_i \boldsymbol{x}_i = \theta_1 \boldsymbol{x}_1 + (1-\theta_1)\sum_{i=2}^K \frac{\theta_i}{1-\theta_1}\boldsymbol{x}_i$ when $\theta_1 < 1$), any convex $f$ satisfies $f\bigl(\sum_{i=1}^K \theta_i \boldsymbol{x}_i\bigr) \le \sum_{i=1}^K \theta_i f(\boldsymbol{x}_i)$ for $\theta_i \ge 0$, $\sum_i \theta_i = 1$. Extending to integrable random vectors $\boldsymbol{X}$ (via supporting hyperplanes at $\mathbb{E}[\boldsymbol{X}]$) gives [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, p. 239), [[Source - Foundations|deep_learning_foundations_concepts]] (Eqs. 2.102–2.104, p. 52):
>
> $$
> f(\mathbb{E}[\boldsymbol{X}]) \le \mathbb{E}[f(\boldsymbol{X})],
> $$
>
> with strict inequality when $f$ is strictly convex and $\boldsymbol{X}$ is non-degenerate. Applying `(eq:convex:jensen)` to the strictly convex function $-\ln u$ ($u > 0$) yields Gibbs' non-negativity inequality for Kullback–Leibler divergence: $D_{\mathrm{KL}}(p \parallel q) = \mathbb{E}_p[-\ln(q(X)/p(X))] \ge -\ln \mathbb{E}_p[q(X)/p(X)] = -\ln 1 = 0$, with equality iff $p = q$ almost everywhere [[Source - Foundations|deep_learning_foundations_concepts]] (Eq. 2.105, pp. 52–53).
>
> **2. First- and Second-Order Differential Conditions.** Suppose $\operatorname{dom} f$ is open and convex.
> - *First-Order ($f \in C^1$):* $f$ is convex if and only if every first-order Taylor approximation is a global underestimator [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, Eq. 7.31, p. 237), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 23, Eq. 23.3, p. 296):
>
> $$
> f(\boldsymbol{y}) \ge f(\boldsymbol{x}) + \nabla f(\boldsymbol{x})^{\top}(\boldsymbol{y} - \boldsymbol{x}) \qquad \forall \boldsymbol{x}, \boldsymbol{y} \in \operatorname{dom} f.
> $$
>
>   *Proof.* ($\Rightarrow$) Rearranging `(eq:convex:chord_def)` for $t \in (0, 1]$ gives $f(\boldsymbol{y}) - f(\boldsymbol{x}) \ge \frac{f(\boldsymbol{x} + t(\boldsymbol{y}-\boldsymbol{x})) - f(\boldsymbol{x})}{t} \xrightarrow{t \searrow 0} \nabla f(\boldsymbol{x})^{\top}(\boldsymbol{y}-\boldsymbol{x})$. ($\Leftarrow$) Let $\boldsymbol{z} = \theta\boldsymbol{x} + (1-\theta)\boldsymbol{y}$. Applying `(eq:convex:first_order)` at $\boldsymbol{z}$ toward $\boldsymbol{x}$ and $\boldsymbol{y}$, multiplying by $\theta$ and $1-\theta$, and summing cancels $\nabla f(\boldsymbol{z})^{\top}[\theta(\boldsymbol{x}-\boldsymbol{z}) + (1-\theta)(\boldsymbol{y}-\boldsymbol{z})] = 0$, leaving $\theta f(\boldsymbol{x}) + (1-\theta)f(\boldsymbol{y}) \ge f(\boldsymbol{z})$.
> - *Second-Order ($f \in C^2$):* Restricting $g(t) = f(\boldsymbol{x} + t\boldsymbol{d})$ gives $g"(t) = \boldsymbol{d}^{\top}\nabla^2 f(\boldsymbol{x} + t\boldsymbol{d})\boldsymbol{d}$. Hence $f$ is convex iff $\nabla^2 f(\boldsymbol{x}) \succeq \boldsymbol{0}$ everywhere; strictly convex if $\nabla^2 f(\boldsymbol{x}) \succ \boldsymbol{0}$ everywhere; and $m$-strongly convex iff $\nabla^2 f(\boldsymbol{x}) \succeq m\boldsymbol{I}_n$ everywhere [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8, Thm. 8.1.1 & Sec. 8.1.3.4, pp. 280–281), [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 4, p. 93), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 23, pp. 297–298).
>
> **3. Local $=$ Global & Variational Optimality.** Let $\boldsymbol{x}_0 \in \mathcal{X}$ be locally optimal for convex problem `(eq:convex:primal_std)` within radius $R > 0$. If there existed $\boldsymbol{y} \in \mathcal{X}$ with $f_0(\boldsymbol{y}) < f_0(\boldsymbol{x}_0)$, choosing $\theta \in (0, 1]$ small enough that $\boldsymbol{z} = (1-\theta)\boldsymbol{x}_0 + \theta\boldsymbol{y}$ satisfies $\lVert \boldsymbol{z}-\boldsymbol{x}_0 \rVert_2 < R$ preserves $\boldsymbol{z} \in \mathcal{X}$ (by convexity of $\mathcal{X}$) while yielding $f_0(\boldsymbol{z}) \le (1-\theta)f_0(\boldsymbol{x}_0) + \theta f_0(\boldsymbol{y}) < f_0(\boldsymbol{x}_0)$, contradicting local optimality [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 24, Sec. 24.3.2, p. 306). When $f_0 \in C^1$, $\boldsymbol{x}^* \in \mathcal{X}$ is globally optimal iff [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 24, Eq. 24.10, p. 306):
>
> $$
> \nabla f_0(\boldsymbol{x}^*)^{\top}(\boldsymbol{x} - \boldsymbol{x}^*) \ge 0 \quad \forall \boldsymbol{x} \in \mathcal{X}
>   \quad\Longleftrightarrow\quad
>   -\nabla f_0(\boldsymbol{x}^*) \in \mathcal{N}_{\mathcal{X}}(\boldsymbol{x}^*),
> $$
>
> where $\mathcal{N}_{\mathcal{X}}(\boldsymbol{x}^*) := \{\boldsymbol{v} \in \mathbb{R}^n : \langle \boldsymbol{v}, \boldsymbol{x} - \boldsymbol{x}^* \rangle \le 0 \forall \boldsymbol{x} \in \mathcal{X}\}$ is the *normal cone* of $\mathcal{X}$ at $\boldsymbol{x}^*$. For nonsmooth convex $f$, the *subdifferential* $\partial f(\boldsymbol{x}) := \{\boldsymbol{g} \in \mathbb{R}^n : f(\boldsymbol{z}) \ge f(\boldsymbol{x}) + \boldsymbol{g}^{\top}(\boldsymbol{z}-\boldsymbol{x}) \forall \boldsymbol{z} \in \operatorname{dom} f\}$ generalizes gradients, and $\boldsymbol{x}^*$ minimizes $f$ over $\mathbb{R}^n$ iff $\boldsymbol{0} \in \partial f(\boldsymbol{x}^*)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8, Eq. 8.13, p. 282).

> [!tip] Supporting Hyperplanes as First-Order Optimality Certificates
> Inequality `(eq:convex:variational_opt)` carries an immediate geometric meaning (visualized later in Figure `fig:convex:tiep_support`): if $\nabla f_0(\boldsymbol{x}^*) \neq \boldsymbol{0}$, the negative gradient $-\nabla f_0(\boldsymbol{x}^*)$ points in the steepest descent direction, while every feasible displacement $\boldsymbol{x} - \boldsymbol{x}^*$ forms an obtuse or right angle with $-\nabla f_0(\boldsymbol{x}^*)$. Thus the affine hyperplane $\{\boldsymbol{x} : \nabla f_0(\boldsymbol{x}^*)^{\top}(\boldsymbol{x} - \boldsymbol{x}^*) = 0\}$ supports the convex feasible set $\mathcal{X}$ at $\boldsymbol{x}^*$, preventing any feasible step from decreasing $f_0$.

### Fenchel–Legendre Conjugate & Biconjugation Duality

Whereas Lagrangian duality dualizes explicit problem constraints, the *Fenchel–Legendre transform* (convex conjugate) dualizes a function's epigraph directly through its family of supporting affine hyperplanes, as developed in [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 7.3, p. 242) and illustrated in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 8.4, p. 279).

> [!important] Fenchel–Legendre Conjugate, Fenchel–Young Inequality & Fenchel–Moreau
> For any function $f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ (not necessarily convex or differentiable), its **Fenchel–Legendre conjugate** $f^* : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ is defined by [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, Eq. 7.53, p. 242):
>
> $$
> f^*(\boldsymbol{y}) := \sup_{\boldsymbol{x} \in \operatorname{dom} f} \bigl(\langle \boldsymbol{y}, \boldsymbol{x} \rangle - f(\boldsymbol{x})\bigr).
> $$
>
> **1. Universal Convexity of $f^*$.** For each fixed $\boldsymbol{x} \in \operatorname{dom} f$, the map $\boldsymbol{y} \mapsto \langle \boldsymbol{y}, \boldsymbol{x} \rangle - f(\boldsymbol{x})$ is affine in $\boldsymbol{y}$. Because the pointwise supremum of affine functions is convex and lower semi-continuous (closed), $f^*$ is *always convex and closed*, even when $f$ is non-convex [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 23, p. 290), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, p. 242).
>
> **2. Fenchel–Young Inequality.** By definition `(eq:convex:fenchel_def)`, for all $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^n$:
>
> $$
> f(\boldsymbol{x}) + f^*(\boldsymbol{y}) \ge \langle \boldsymbol{y}, \boldsymbol{x} \rangle,
> $$
>
> with equality if and only if $\boldsymbol{x}$ attains the supremum, i.e., $\langle \boldsymbol{y}, \boldsymbol{z} \rangle - f(\boldsymbol{z}) \le \langle \boldsymbol{y}, \boldsymbol{x} \rangle - f(\boldsymbol{x})  \forall \boldsymbol{z} \iff f(\boldsymbol{z}) \ge f(\boldsymbol{x}) + \langle \boldsymbol{y}, \boldsymbol{z} - \boldsymbol{x} \rangle \iff \boldsymbol{y} \in \partial f(\boldsymbol{x})$ (reducing to $\boldsymbol{y} = \nabla f(\boldsymbol{x})$ when $f$ is differentiable and convex) [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, Eq. 7.58, p. 243).
>
> **3. Biconjugate & Fenchel–Moreau Theorem.** Applying `(eq:convex:fenchel_def)` a second time gives $f^{**}(\boldsymbol{x}) := \sup_{\boldsymbol{y} \in \operatorname{dom} f^*} \bigl(\langle \boldsymbol{x}, \boldsymbol{y} \rangle - f^*(\boldsymbol{y})\bigr)$. By `(eq:convex:fenchel_young)`, $f^{**}(\boldsymbol{x}) \le f(\boldsymbol{x})$ universally; specifically, $f^{**}$ is the *closed convex envelope* (greatest closed convex minorizer) of $f$. Under the mild condition that $f$ is *proper* ($f(\boldsymbol{x}) > -\infty$ and $\operatorname{dom} f \neq \emptyset$), convex, and lower semi-continuous, the **Fenchel–Moreau Theorem** guarantees exact involution $f^{**} = f$ (proved via strict hyperplane separation of any point $(\boldsymbol{x}_0, t_0) \notin \operatorname{epi} f$ from the closed convex set $\operatorname{epi} f$) [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, p. 243), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8, Fig. 8.4(b), p. 279).
>
> **4. Canonical ML Conjugates & Fenchel–Rockafellar Duality.**
> - *Kernel Quadratic Form* [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, Ex. 7.7, Eqs. 7.59–7.62, p. 244): For $f(\boldsymbol{y}) = \frac{\lambda}{2}\boldsymbol{y}^{\top}\boldsymbol{K}^{-1}\boldsymbol{y}$ ($\boldsymbol{K} \succ \boldsymbol{0}$, $\lambda > 0$), setting $\nabla_{\boldsymbol{y}}(\langle \boldsymbol{y}, \boldsymbol{\alpha} \rangle - f(\boldsymbol{y})) = \boldsymbol{\alpha} - \lambda\boldsymbol{K}^{-1}\boldsymbol{y} = \boldsymbol{0}$ gives $\boldsymbol{y} = \lambda \boldsymbol{K}\boldsymbol{\alpha}$ and $f^*(\boldsymbol{\alpha}) = \frac{1}{2\lambda}\boldsymbol{\alpha}^{\top}\boldsymbol{K}\boldsymbol{\alpha}$.
> - *Norms & Dual Norms:* If $f(\boldsymbol{x}) = \lVert \boldsymbol{x} \rVert$ with dual norm $\lVert \boldsymbol{y} \rVert_* := \sup_{\lVert \boldsymbol{z} \rVert\le 1}\langle \boldsymbol{y}, \boldsymbol{z} \rangle$, then $f^*(\boldsymbol{y}) = 0$ if $\lVert \boldsymbol{y} \rVert_* \le 1$ and $+\infty$ otherwise (the indicator $I_{\{\lVert \cdot \rVert_*\le 1\}}(\boldsymbol{y})$).
> - *Negative Entropy & Log-Sum-Exp:* On $\boldsymbol{x} \in \Delta^{n-1}$ (with $f(\boldsymbol{x}) = \sum_i x_i \ln x_i$ inside $\Delta^{n-1}$ and $+\infty$ outside), maximizing $\boldsymbol{y}^{\top}\boldsymbol{x} - \sum_i x_i \ln x_i$ gives the softmax-normalized Gibbs optimal $\hat{x}_i = e^{y_i}/\sum_k e^{y_k}$ and $f^*(\boldsymbol{y}) = \ln \sum_{i=1}^n e^{y_i}$.
> - *Separable Empirical Losses & Fenchel Duality* [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, Ex. 7.8–7.9, pp. 244–245): If $L(\boldsymbol{t}) = \sum_{i=1}^n \ell_i(t_i)$, then $L^*(\boldsymbol{z}) = \sum_{i=1}^n \ell_i^*(z_i)$. Introducing $\boldsymbol{y} = \boldsymbol{A}\boldsymbol{x}$ with Lagrange multiplier $\boldsymbol{u}$ into $\min_{\boldsymbol{x}} f(\boldsymbol{A}\boldsymbol{x}) + g(\boldsymbol{x})$ for convex $f, g$ swaps $\min_{\boldsymbol{x},\boldsymbol{y}}\max_{\boldsymbol{u}}$ to $\max_{\boldsymbol{u}}\min_{\boldsymbol{x},\boldsymbol{y}}$ (under regularity), yielding the **Fenchel–Rockafellar dual**:
>
> $$
> \min_{\boldsymbol{x} \in \mathbb{R}^n} f(\boldsymbol{A}\boldsymbol{x}) + g(\boldsymbol{x})
>     = \max_{\boldsymbol{u} \in \mathbb{R}^m} \Bigl[ -f^*(\boldsymbol{u}) - g^*(-\boldsymbol{A}^{\top}\boldsymbol{u}) \Bigr].
> $$
>

> [!tip] Geometric Intercept Interpretation of $f^*(\boldsymbol{y})$
> Consider a 1D function $f(x)$ and fix a slope $s \in \mathbb{R}$. Moving a line $t = s x + c$ vertically downward until it just touches the graph of $f$ from below sets the intercept $c = \inf_x (f(x) - sx) = -\sup_x (sx - f(x)) = -f^*(s)$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 7.3.3, Eqs. 7.54–7.58, p. 243). Hence $f^*(s)$ measures the exact *negative $t$-intercept* of the tightest affine lower bound of slope $s$. Reconstructing $f^{**}(x) = \sup_s (sx - f^*(s))$ takes the upper envelope of all such supporting halfspaces $\{(x,t) : t \ge sx - f^*(s)\}$, faithfully carving out $\operatorname{epi} f$ when $f$ is closed convex, or filling in non-convex "dents" to produce the convex hull of $\operatorname{epi} f$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 8.4(b), p. 279).

### Lagrangian Dual Function, Weak/Strong Duality & Slater's Condition

To convert the hard-constrained problem `(eq:convex:primal_std)` into a smooth saddle formulation, first note that adding infinite-step indicator barriers $\hat{L}(\boldsymbol{x}) = f_0(\boldsymbol{x}) + \sum_{i=1}^m I_{\le 0}(f_i(\boldsymbol{x})) + \sum_{j=1}^p I_{=0}(h_j(\boldsymbol{x}))$ (where $I_{\le 0}(z) = 0$ if $z \le 0$ and $+\infty$ if $z > 0$) preserves the exact primal optimum, yet is discontinuous [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, Eqs. 7.18–7.19, pp. 233–234), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8, Eq. 8.97, p. 304). Replacing infinite indicator steps by linear penalties yields the **Lagrangian** [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 25, Sec. 25.2.1, p. 318), [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 4, Eq. 4.14, p. 94):

$$
L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) := f_0(\boldsymbol{x}) + \sum_{i=1}^m \lambda_i f_i(\boldsymbol{x}) + \sum_{j=1}^p \nu_j h_j(\boldsymbol{x})
  = f_0(\boldsymbol{x}) + \boldsymbol{\lambda}^{\top} \boldsymbol{f}(\boldsymbol{x}) + \boldsymbol{\nu}^{\top} \boldsymbol{h}(\boldsymbol{x}),
$$

with Lagrange/KKT multiplier vectors $\boldsymbol{\lambda} \in \mathbb{R}^m_{\ge 0}$ and $\boldsymbol{\nu} \in \mathbb{R}^p$. Exactly recovering the barrier via $\sup_{\boldsymbol{\lambda} \succeq \boldsymbol{0},  \boldsymbol{\nu}} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = \hat{L}(\boldsymbol{x})$ expresses the primal problem in unconstrained minimax form: $p^* = \inf_{\boldsymbol{x} \in \mathcal{D}} \sup_{\boldsymbol{\lambda} \succeq \boldsymbol{0},  \boldsymbol{\nu}} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Eqs. 4.15–4.18, p. 94), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Eq. 7.26, p. 235).

> [!important] Lagrange Dual Problem, Weak Duality, Slater's Theorem & Saddle Points
> **1. Dual Function & Concavity.** Swapping $\inf$ and $\sup$ defines the **Lagrange dual function** [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 25, Eq. 25.5, p. 318), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Def. 7.1, p. 234):
>
> $$
> g(\boldsymbol{\lambda}, \boldsymbol{\nu}) := \inf_{\boldsymbol{x} \in \mathcal{D}} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}).
> $$
>
> Because $L(\boldsymbol{x}, \cdot, \cdot)$ is affine in $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ for every $\boldsymbol{x} \in \mathcal{D}$, $g(\boldsymbol{\lambda}, \boldsymbol{\nu})$ is a pointwise infimum of affine functions and is therefore *always concave* (possibly taking $-\infty$), regardless of whether $f_0, f_i, h_j$ are convex [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (p. 318), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (p. 235). The **Lagrange dual problem** is thus always a concave maximization (convex optimization) problem:
>
> $$
> d^* := \sup_{\boldsymbol{\lambda} \succeq \boldsymbol{0},  \boldsymbol{\nu} \in \mathbb{R}^p} g(\boldsymbol{\lambda}, \boldsymbol{\nu}),
> $$
>
> with dual feasibility requiring $\boldsymbol{\lambda} \succeq \boldsymbol{0}$ and $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) > -\infty$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 25, Eq. 25.11, p. 321).
>
> **2. Weak Duality ($d^* \le p^*$).** By the universal minimax inequality $\sup_y \inf_x \phi(x,y) \le \inf_x \sup_y \phi(x,y)$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Eqs. 7.23–7.24, pp. 234–235)—equivalently, for any $\boldsymbol{x} \in \mathcal{X}$ and $\boldsymbol{\lambda} \succeq \boldsymbol{0}$, $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \le f_0(\boldsymbol{x})$ since $\lambda_i f_i(\boldsymbol{x}) \le 0$ and $\nu_j h_j(\boldsymbol{x}) = 0$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Eq. 25.6, p. 319)—taking $\inf_{\boldsymbol{x} \in \mathcal{X}}$ and $\sup_{\boldsymbol{\lambda} \succeq \boldsymbol{0}, \boldsymbol{\nu}}$ establishes **weak duality**:
>
> $$
> d^* = \sup_{\boldsymbol{\lambda} \succeq \boldsymbol{0},  \boldsymbol{\nu}} \inf_{\boldsymbol{x} \in \mathcal{D}} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
>    \le  \inf_{\boldsymbol{x} \in \mathcal{D}} \sup_{\boldsymbol{\lambda} \succeq \boldsymbol{0},  \boldsymbol{\nu}} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = p^*,
> $$
>
> holding unconditionally for *all* optimization problems (convex or non-convex). The nonnegative quantity $\Delta_{\mathrm{gap}} := p^* - d^* \ge 0$ is the *optimal duality gap* [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 25.3.1, p. 322). When $d^* = p^*$, **strong duality** holds.
>
> **3. Slater's Constraint Qualification & Strong Duality Theorem.** Assume `(eq:convex:primal_std)` is convex ($f_0,\dots,f_m$ convex, $h_j(\boldsymbol{x}) = (\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b})_j$). *Slater's condition* holds if there exists a strictly feasible point $\tilde{\boldsymbol{x}} \in \operatorname{relint}(\mathcal{D})$ such that [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 25, Def. 25.1 & Thm. 25.1, p. 322):
>
> $$
> f_i(\tilde{\boldsymbol{x}}) < 0 \quad \text{for all non-affine } f_i \qquad \bigl(f_i(\tilde{\boldsymbol{x}}) \le 0 \text{ suffices if } f_i \text{ is affine}\bigr), \qquad \boldsymbol{A}\tilde{\boldsymbol{x}} = \boldsymbol{b}.
> $$
>
> *Theorem (Slater).* Under convexity and `(eq:convex:slater)`, strong duality $d^* = p^*$ holds; moreover, if $p^* > -\infty$, the dual optimum $(\boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$ is attained.
>
> *Proof Sketch via Hyperplane Separation.* Consider the convex epigraph-value set
>
> $$
>   \mathcal{A} := \{(\boldsymbol{u}, \boldsymbol{v}, t) \in \mathbb{R}^m \times \mathbb{R}^p \times \mathbb{R} : \exists \boldsymbol{x} \in \mathcal{D} \text{ with } \boldsymbol{f}(\boldsymbol{x}) \preceq \boldsymbol{u},  \boldsymbol{A}\boldsymbol{x}-\boldsymbol{b} = \boldsymbol{v},  f_0(\boldsymbol{x}) \le t\}.
> $$
>
> Assume $\operatorname{rank}(\boldsymbol{A}) = p$ and $\operatorname{int}(\mathcal{D}) \neq \emptyset$ (otherwise reduce to $\operatorname{aff}(\mathcal{D})$). Because $p^* = \inf \{t : (\boldsymbol{0}, \boldsymbol{0}, t) \in \mathcal{A}\}$, the point $(\boldsymbol{0}, \boldsymbol{0}, p^*)$ does not belong to $\operatorname{int}(\mathcal{A})$ (if $p^* > -\infty$). By the Supporting/Separating Hyperplane Theorem (Core Formulation `cm:convex:separation_epi`), there exists $(\tilde{\boldsymbol{\lambda}}, \tilde{\boldsymbol{\nu}}, \mu) \neq \boldsymbol{0}$ supporting $\mathcal{A}$ at $(\boldsymbol{0}, \boldsymbol{0}, p^*)$:
>
> $$
> \tilde{\boldsymbol{\lambda}}^{\top} \boldsymbol{u} + \tilde{\boldsymbol{\nu}}^{\top} \boldsymbol{v} + \mu t \ge \mu p^* \qquad \forall (\boldsymbol{u}, \boldsymbol{v}, t) \in \mathcal{A}.
> $$
>
> Since $\boldsymbol{u}$ and $t$ can be increased arbitrarily inside $\mathcal{A}$, necessarily $\tilde{\boldsymbol{\lambda}} \succeq \boldsymbol{0}$ and $\mu \ge 0$. Applying `(eq:convex:slater_sep)` to $(\boldsymbol{f}(\tilde{\boldsymbol{x}}), \boldsymbol{0}, f_0(\tilde{\boldsymbol{x}})) \in \mathcal{A}$ yields $\tilde{\boldsymbol{\lambda}}^{\top}\boldsymbol{f}(\tilde{\boldsymbol{x}}) + \mu(f_0(\tilde{\boldsymbol{x}}) - p^*) \ge 0$. If $\mu = 0$, then $\tilde{\boldsymbol{\lambda}}^{\top}\boldsymbol{f}(\tilde{\boldsymbol{x}}) \ge 0$, which forces $\tilde{\boldsymbol{\lambda}} = \boldsymbol{0}$ since $f_i(\tilde{\boldsymbol{x}}) < 0$; then `(eq:convex:slater_sep)` becomes $\tilde{\boldsymbol{\nu}}^{\top}(\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}) \ge 0$ for all $\boldsymbol{x} \in \mathcal{D}$ near $\tilde{\boldsymbol{x}} \in \operatorname{int}(\mathcal{D})$, forcing $\tilde{\boldsymbol{\nu}} = \boldsymbol{0}$ (contradicting $(\tilde{\boldsymbol{\lambda}}, \tilde{\boldsymbol{\nu}}, \mu) \neq \boldsymbol{0}$). Hence $\mu > 0$. Dividing `(eq:convex:slater_sep)` by $\mu$ and setting $\boldsymbol{\lambda}^* := \tilde{\boldsymbol{\lambda}}/\mu \succeq \boldsymbol{0}$, $\boldsymbol{\nu}^* := \tilde{\boldsymbol{\nu}}/\mu$ gives $L(\boldsymbol{x}, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*) \ge p^*$ for all $\boldsymbol{x} \in \mathcal{D}$, so $d^* \ge g(\boldsymbol{\lambda}^*, \boldsymbol{\nu}^*) \ge p^*$, combining with `(eq:convex:weak_duality)` to yield $d^* = p^*$.
>
> **4. Saddle-Point Characterization.** A pair $(\boldsymbol{x}^*, (\boldsymbol{\lambda}^*, \boldsymbol{\nu}^*))$ with $\boldsymbol{x}^* \in \mathcal{D}$, $\boldsymbol{\lambda}^* \succeq \boldsymbol{0}$ is primal-dual optimal with zero duality gap ($f_0(\boldsymbol{x}^*) = g(\boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$) if and only if it is a **saddle point** of $L$:
>
> $$
> L(\boldsymbol{x}^*, \boldsymbol{\lambda}, \boldsymbol{\nu})  \le  L(\boldsymbol{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)  \le  L(\boldsymbol{x}, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)
>   \qquad \forall \boldsymbol{x} \in \mathcal{D}, \forall \boldsymbol{\lambda} \succeq \boldsymbol{0}, \forall \boldsymbol{\nu} \in \mathbb{R}^p.
> $$
>

### Complementary Slackness & KKT Optimality Conditions

> [!important] Complementary Slackness & Karush–Kuhn–Tucker (KKT) Theorem
> **1. Derivation of Complementary Slackness.** Suppose strong duality holds ($p^* = d^*$) and is attained at primal-optimal $\boldsymbol{x}^*$ and dual-optimal $(\boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$ (no convexity required yet). Expanding the duality chain yields [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 25, Eqs. 25.13–25.16, p. 323):
>
> $$
> \begin{aligned}
> f_0(\boldsymbol{x}^*) = g(\boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)
>   &= \inf_{\boldsymbol{x} \in \mathcal{D}} \Bigl( f_0(\boldsymbol{x}) + \sum_{i=1}^m \lambda_i^* f_i(\boldsymbol{x}) + \sum_{j=1}^p \nu_j^* h_j(\boldsymbol{x}) \Bigr)  \\
>   &\le f_0(\boldsymbol{x}^*) + \sum_{i=1}^m \underbrace{\lambda_i^* f_i(\boldsymbol{x}^*)}_{\le 0} + \sum_{j=1}^p \underbrace{\nu_j^* h_j(\boldsymbol{x}^*)}_{= 0}
>   \le f_0(\boldsymbol{x}^*).
> \end{aligned}
> $$
>
> Because the two ends are identical, both inequalities collapse to equalities. First, $\boldsymbol{x}^*$ minimizes $L(\cdot, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$ over $\mathcal{D}$ (so $\nabla_{\boldsymbol{x}} L(\boldsymbol{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*) = \boldsymbol{0}$ when differentiable, or $\boldsymbol{0} \in \partial_{\boldsymbol{x}} L$ when subdifferentiable). Second, $\sum_{i=1}^m \lambda_i^* f_i(\boldsymbol{x}^*) = 0$ with each term non-positive forces termwise **complementary slackness** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. 8.106, p. 305), [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (p. 95), [[Source - Foundations|deep_learning_foundations_concepts]] (Eq. C.11, p. 624):
>
> $$
> \lambda_i^* f_i(\boldsymbol{x}^*) = 0 \quad \forall i=1,\dots,m
>   \quad\Longleftrightarrow\quad
>   \bigl(\lambda_i^* > 0 \implies f_i(\boldsymbol{x}^*) = 0\bigr)  \land  \bigl(f_i(\boldsymbol{x}^*) < 0 \implies \lambda_i^* = 0\bigr).
> $$
>
> **2. The Four KKT Conditions.** Assuming $f_0, f_i, h_j$ are differentiable, the KKT system at $(\boldsymbol{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$ reads [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Eqs. 25.19–25.23, p. 324), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eqs. 8.103–8.106, pp. 304–305), [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (pp. 95–96):
>
> $$
> \begin{alignedat}{2}
> \text{(Primal)}\quad && f_i(\boldsymbol{x}^*) &\le 0  (i=1,\dots,m),   h_j(\boldsymbol{x}^*) = 0  (j=1,\dots,p),  \\
>   \text{(Dual)}\quad && \lambda_i^* &\ge 0  (i=1,\dots,m),   \boldsymbol{\nu}^* \in \mathbb{R}^p,  \\
>   \text{(Slackness)}\quad && \lambda_i^* f_i(\boldsymbol{x}^*) &= 0  (i=1,\dots,m),  \\
>   \text{(Stationarity)}\quad && \nabla f_0(\boldsymbol{x}^*) &+ \sum_{i=1}^m \lambda_i^* \nabla f_i(\boldsymbol{x}^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(\boldsymbol{x}^*) = \boldsymbol{0}.
> \end{alignedat}
> $$
>
> **3. Necessity vs. Sufficiency.**
> - *Non-Convex (Necessity Only):* Under strong duality (or a regularity/constraint qualification such as LICQ), KKT `(eq:convex:kkt_primal)`–`(eq:convex:kkt_stat)` are *necessary* for optimality, not sufficient [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (p. 95), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (p. 324).
> - *Convex (Sufficiency & Equivalence under Slater):* If `(eq:convex:primal_std)` is convex and differentiable, any triple $(\tilde{\boldsymbol{x}}, \tilde{\boldsymbol{\lambda}}, \tilde{\boldsymbol{\nu}})$ satisfying KKT is *sufficient* for global primal-dual optimality and zero duality gap! Indeed, since $\tilde{\boldsymbol{\lambda}} \succeq \boldsymbol{0}$ and $h_j$ are affine, $\boldsymbol{x} \mapsto L(\boldsymbol{x}, \tilde{\boldsymbol{\lambda}}, \tilde{\boldsymbol{\nu}})$ is convex; hence stationarity `(eq:convex:kkt_stat)` implies $\tilde{\boldsymbol{x}}$ globally minimizes $L(\cdot, \tilde{\boldsymbol{\lambda}}, \tilde{\boldsymbol{\nu}})$, yielding $g(\tilde{\boldsymbol{\lambda}}, \tilde{\boldsymbol{\nu}}) = L(\tilde{\boldsymbol{x}}, \tilde{\boldsymbol{\lambda}}, \tilde{\boldsymbol{\nu}}) = f_0(\tilde{\boldsymbol{x}})$ by `(eq:convex:kkt_primal)` and `(eq:convex:kkt_cs)`. Conversely, under Slater's condition `(eq:convex:slater)`, strong duality holds and KKT is both *necessary and sufficient* [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (p. 324), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (p. 305).

> [!tip] Geometric Normal-Vector Balance at Active Constraints
> Why must $\lambda_i^* \ge 0$ in `(eq:convex:kkt_dual)`? As Bishop & Bishop and Murphy both illustrate [[Source - Foundations|deep_learning_foundations_concepts]] (App. C, Fig. C.3, p. 623), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.5.2, p. 305), at any point on an active boundary $f_i(\boldsymbol{x}^*) = 0$, the gradient $\nabla f_i(\boldsymbol{x}^*)$ points orthogonally *outward* into the infeasible zone $f_i(\boldsymbol{x}) > 0$ (since $f_i(\boldsymbol{x}^* + \boldsymbol{\epsilon}) \approx \boldsymbol{\epsilon}^{\top}\nabla f_i(\boldsymbol{x}^*) > 0$). For $\boldsymbol{x}^*$ to minimize $f_0$, the downhill direction $-\nabla f_0(\boldsymbol{x}^*)$ cannot point into the feasible interior; instead it must lie strictly inside the nonnegative *normal cone* generated by active outward normals: $-\nabla f_0(\boldsymbol{x}^*) = \sum_{i \in \mathcal{A}(\boldsymbol{x}^*)} \lambda_i^* \nabla f_i(\boldsymbol{x}^*) + \sum_j \nu_j^* \nabla h_j(\boldsymbol{x}^*)$ with $\lambda_i^* \ge 0$. Any inactive constraint ($f_i(\boldsymbol{x}^*) < 0$) exerts zero force ($\lambda_i^* = 0$).

**Worked Canonical ML Instantiations of KKT & Duals:**
- **Equality-Constrained KKT Linear System** [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 25, Eqs. 25.24–25.26, pp. 324–325): Minimizing $\frac{1}{2}\boldsymbol{x}^{\top}\boldsymbol{P}\boldsymbol{x} + \boldsymbol{q}^{\top}\boldsymbol{x} + r$ ($\boldsymbol{P} \succeq \boldsymbol{0}$) subject to $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ reduces KKT directly to the symmetric indefinite block saddle system:
  
$$
\begin{bmatrix} \boldsymbol{P} & \boldsymbol{A}^{\top} \\ \boldsymbol{A} & \boldsymbol{0} \end{bmatrix}
    \begin{bmatrix} \boldsymbol{x}^* \\ \boldsymbol{\nu}^* \end{bmatrix}
    = \begin{bmatrix} -\boldsymbol{q} \\ \boldsymbol{b} \end{bmatrix}.
$$
- **Inequality-Constrained QP Dual & Active-Set KKT on the $\ell_1$ Ball:** Minimizing $\frac{1}{2}\boldsymbol{x}^{\top}\boldsymbol{Q}\boldsymbol{x} + \boldsymbol{c}^{\top}\boldsymbol{x}$ ($\boldsymbol{Q} \succ \boldsymbol{0}$) subject to $\boldsymbol{A}\boldsymbol{x} \preceq \boldsymbol{b}$ gives $\nabla_{\boldsymbol{x}} L = \boldsymbol{Q}\boldsymbol{x} + \boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda} = \boldsymbol{0}$, so $\boldsymbol{x}(\boldsymbol{\lambda}) = -\boldsymbol{Q}^{-1}(\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda})$ and the concave dual QP becomes $\max_{\boldsymbol{\lambda} \succeq \boldsymbol{0}} -\frac{1}{2}(\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda})^{\top}\boldsymbol{Q}^{-1}(\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda}) - \boldsymbol{\lambda}^{\top}\boldsymbol{b}$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 7.3, pp. 241–242). In Murphy's 2D $\ell_1$-ball instantiation $\min_{\boldsymbol{\theta}} (\theta_1 - \frac{3}{2})^2 + (\theta_2 - \frac{1}{8})^2$ ($\boldsymbol{Q} = 2\boldsymbol{I}_2$, $\boldsymbol{c} = -(3, \frac{1}{4})^{\top}$) s.t. $\lvert \theta_1 \rvert+\lvert \theta_2 \rvert \le 1$ (four inequalities $\pm\theta_1 \pm\theta_2 \le 1$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.5, pp. 306–307), the two left faces are strictly inactive ($f_3, f_4 < 0 \implies \mu_3^* = \mu_4^* = 0$), while the active right faces enforce $\boldsymbol{\theta}^* = (1, 0)^{\top}$ and $\boldsymbol{\mu}^* = (0.625, 0.375, 0, 0)^{\top}$ (confirming $-\nabla f_0(\boldsymbol{\theta}^*) = (1, 0.25)^{\top} = 0.625 (1,1)^{\top} + 0.375 (1,-1)^{\top}$).
- **Rayleigh Quotient & Spectral Norm:** [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (App. A, pp. 380–381). Maximizing $\boldsymbol{x}^{\top}\boldsymbol{A}^{\top}\boldsymbol{A}\boldsymbol{x}$ s.t. $\boldsymbol{x}^{\top}\boldsymbol{x} = 1$ gives $L(\boldsymbol{x}, \lambda) = \boldsymbol{x}^{\top}\boldsymbol{A}^{\top}\boldsymbol{A}\boldsymbol{x} + \lambda(1 - \boldsymbol{x}^{\top}\boldsymbol{x}) \implies \boldsymbol{A}^{\top}\boldsymbol{A}\boldsymbol{x} = \lambda\boldsymbol{x}$, identifying the Lagrange multiplier $\lambda^*$ directly with the eigenvalue $\lambda_{\max}(\boldsymbol{A}^{\top}\boldsymbol{A}) = \lVert \boldsymbol{A} \rVert_2^2$.

## Algorithmic Implementation & Comparative Synthesis

Solving general constrained convex programs beyond closed-form KKT systems relies on two complementary algorithmic paradigms: (1) **Primal–Dual Interior-Point (Log-Barrier) Methods**, which smooth inequalities into a logarithmic barrier and follow the central path toward KKT optimality with superlinear/quadratic Newton precision [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8, Sec. 8.5.3.1, p. 306), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 24, p. 307), and (2) **Augmented Lagrangian Methods (Method of Multipliers)**, which add a quadratic penalty to the Lagrangian to enforce strong convexity/smoothness and achieve dual ascent convergence without strict convexity of $f_0$.

### Primal–Dual Interior-Point Barrier Method

For convex `(eq:convex:primal_std)` with strictly feasible interior ($\exists \tilde{\boldsymbol{x}} : f_i(\tilde{\boldsymbol{x}}) < 0$, $\boldsymbol{A}\tilde{\boldsymbol{x}} = \boldsymbol{b}$), replacing indicator steps by $-\frac{1}{t}\sum_{i=1}^m \ln(-f_i(\boldsymbol{x}))$ ($t > 0$) yields the central-path problem $\min_{\boldsymbol{x}} t f_0(\boldsymbol{x}) - \sum_{i=1}^m \ln(-f_i(\boldsymbol{x}))$ s.t. $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$. Defining $\lambda_i := -1 / (t f_i(\boldsymbol{x})) > 0$ reveals the perturbed KKT conditions:

$$
\boldsymbol{r}_t(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) :=
  \begin{bmatrix}
    \nabla f_0(\boldsymbol{x}) + \boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})^{\top} \boldsymbol{\lambda} + \boldsymbol{A}^{\top}\boldsymbol{\nu} \\
    -\mathrm{diag}(\boldsymbol{\lambda}) \boldsymbol{f}(\boldsymbol{x}) - \frac{1}{t}\boldsymbol{1}_m \\
    \boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}
  \end{bmatrix} = \boldsymbol{0},
$$

where $\boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x}) \in \mathbb{R}^{m \times n}$ has rows $\nabla f_i(\boldsymbol{x})^{\top}$. Because $\boldsymbol{\lambda} \succ \boldsymbol{0}$ and $\boldsymbol{f}(\boldsymbol{x}) \prec \boldsymbol{0}$, evaluating the dual bound along the central path yields an exact certificate on the suboptimality gap: $f_0(\boldsymbol{x}^*(t)) - p^* \le m / t$ (via $\sum_i \lambda_i^* (-f_i(\boldsymbol{x}^*(t))) = m/t$). Algorithm `alg:convex:ipm` solves `(eq:convex:central_path)` via damped Newton steps while maintaining strict primal–dual positivity ($\boldsymbol{f}(\boldsymbol{x}) \prec \boldsymbol{0}$, $\boldsymbol{\lambda} \succ \boldsymbol{0}$).

> [!example] Algorithm: Primal–Dual Path-Following Interior-Point Method for Convex Programs
> **Input:** Convex $f_0, \boldsymbol{f} \in C^2$, $\boldsymbol{A} \in \mathbb{R}^{p \times n}$, $\boldsymbol{b} \in \mathbb{R}^p$, strictly feasible $\boldsymbol{x}^{(0)}$ ($\boldsymbol{f}(\boldsymbol{x}^{(0)}) \prec \boldsymbol{0}$), $\boldsymbol{\lambda}^{(0)} \succ \boldsymbol{0}$, $\boldsymbol{\nu}^{(0)}$, barrier factor $\mu > 1$ (e.g., $10$), tolerance $\varepsilon > 0$.
> **Output:** Primal–dual $\varepsilon$-optimal tuple $(\boldsymbol{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$ with duality gap $\le \varepsilon$.
> Initialize $k \leftarrow 0$
> **repeat until** $\hat{\eta} \le \varepsilon$ **and** $\lVert \boldsymbol{r}_{\mathrm{dual}} \rVert_2 \le \varepsilon$ **and** $\lVert \boldsymbol{A}\boldsymbol{x}^{(k)} - \boldsymbol{b} \rVert_2 \le \varepsilon$:
>
>   Compute surrogate duality gap $\hat{\eta} \leftarrow -\boldsymbol{f}(\boldsymbol{x}^{(k)})^{\top} \boldsymbol{\lambda}^{(k)}$ and set $t \leftarrow \mu m / \hat{\eta}$
>   Evaluate KKT residual $\boldsymbol{r}_t(\boldsymbol{x}^{(k)}, \boldsymbol{\lambda}^{(k)}, \boldsymbol{\nu}^{(k)})$ via `(eq:convex:central_path)`, where $\boldsymbol{r}_{\mathrm{dual}} = \nabla f_0 + \boldsymbol{J}_{\boldsymbol{f}}^{\top}\boldsymbol{\lambda} + \boldsymbol{A}^{\top}\boldsymbol{\nu}$
>   Form Newton Jacobian $\boldsymbol{H}_{\mathrm{KKT}}$ of $\boldsymbol{r}_t$ and solve $\boldsymbol{H}_{\mathrm{KKT}} (\Delta\boldsymbol{x}, \Delta\boldsymbol{\lambda}, \Delta\boldsymbol{\nu})^{\top} = -\boldsymbol{r}_t$
>   Compute max feasible fraction-to-boundary step:
>   $\alpha_{\max} \leftarrow \min\!\bigl(1, 0.99 \min_{i : \Delta\lambda_i < 0} (-\lambda_i^{(k)} / \Delta\lambda_i)\bigr)$
>   Backtrack $\alpha \leftarrow \alpha_{\max}$ (halving $\alpha \leftarrow \beta\alpha$, $\beta \in (0,1)$) until $\boldsymbol{f}(\boldsymbol{x}^{(k)} + \alpha\Delta\boldsymbol{x}) \prec \boldsymbol{0}$ and $\lVert \boldsymbol{r}_t(\boldsymbol{x}^{(k)}+\alpha\Delta\boldsymbol{x}, \boldsymbol{\lambda}^{(k)}+\alpha\Delta\boldsymbol{\lambda}, \boldsymbol{\nu}^{(k)}+\alpha\Delta\boldsymbol{\nu}) \rVert_2 \le (1 - c \alpha)\lVert \boldsymbol{r}_t \rVert_2$
>   Update $(\boldsymbol{x}^{(k+1)}, \boldsymbol{\lambda}^{(k+1)}, \boldsymbol{\nu}^{(k+1)}) \leftarrow (\boldsymbol{x}^{(k)}, \boldsymbol{\lambda}^{(k)}, \boldsymbol{\nu}^{(k)}) + \alpha(\Delta\boldsymbol{x}, \Delta\boldsymbol{\lambda}, \Delta\boldsymbol{\nu})$, $k \leftarrow k + 1$
>
> **return** $(\boldsymbol{x}^{(k)}, \boldsymbol{\lambda}^{(k)}, \boldsymbol{\nu}^{(k)})$

### Augmented Lagrangian (Method of Multipliers)

When $f_0$ is non-strictly convex (e.g., linear or piecewise-linear) or high-dimensional ($\nabla^2 L$ too large to factorize), ordinary dual ascent $\boldsymbol{\nu}^{k+1} = \boldsymbol{\nu}^k + \alpha_k(\boldsymbol{A}\boldsymbol{x}^k - \boldsymbol{b})$ can oscillate or diverge because $g$ is non-differentiable or unbounded. Adding a quadratic augmentation $\frac{\rho}{2}\lVert \boldsymbol{A}\boldsymbol{x}-\boldsymbol{b} \rVert_2^2$ ($\rho > 0$) and Rockafellar–Wets smoothed inequality term yields the **Augmented Lagrangian** (Algorithm `alg:convex:alm`):

$$
\mathcal{L}_\rho(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
  := f_0(\boldsymbol{x}) + \boldsymbol{\nu}^{\top}(\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}) + \frac{\rho}{2}\lVert \boldsymbol{A}\boldsymbol{x} - \boldsymbol{b} \rVert_2^2
  + \frac{1}{2\rho}\sum_{i=1}^m \Bigl( \max\bigl(0,  \lambda_i + \rho f_i(\boldsymbol{x})\bigr)^2 - \lambda_i^2 \Bigr).
$$

Differentiating `(eq:convex:aug_lagrangian)` w.r.t. $\boldsymbol{x}$ shows that the dual update $\lambda_i^{k+1} = \max(0, \lambda_i^k + \rho f_i(\boldsymbol{x}^{k+1}))$, $\boldsymbol{\nu}^{k+1} = \boldsymbol{\nu}^k + \rho(\boldsymbol{A}\boldsymbol{x}^{k+1} - \boldsymbol{b})$ automatically enforces exact KKT stationarity $\nabla f_0(\boldsymbol{x}^{k+1}) + \sum_i \lambda_i^{k+1}\nabla f_i(\boldsymbol{x}^{k+1}) + \boldsymbol{A}^{\top}\boldsymbol{\nu}^{k+1} \approx \boldsymbol{0}$ at every outer iteration, without requiring $\rho \to \infty$ (avoiding quadratic-penalty ill-conditioning).

> [!example] Algorithm: Primal–Dual Augmented Lagrangian (Method of Multipliers)
> **Input:** Objective $f_0$, convex constraints $\boldsymbol{f}(\boldsymbol{x}) \preceq \boldsymbol{0}$, $\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}$, penalty $\rho > 0$, initial $(\boldsymbol{x}^{(0)}, \boldsymbol{\lambda}^{(0)}, \boldsymbol{\nu}^{(0)})$, tolerance $\varepsilon > 0$.
> **Output:** Approximate KKT tuple $(\boldsymbol{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$.
> **for** $k = 0, 1, \dots, K_{\max}-1$:
>
>   $\boldsymbol{x}^{(k+1)} \leftarrow \arg\min_{\boldsymbol{x} \in \mathcal{D}} \mathcal{L}_\rho(\boldsymbol{x}, \boldsymbol{\lambda}^{(k)}, \boldsymbol{\nu}^{(k)})$  *// Unconstrained/box-constrained inner solve*
>   $\boldsymbol{\nu}^{(k+1)} \leftarrow \boldsymbol{\nu}^{(k)} + \rho(\boldsymbol{A}\boldsymbol{x}^{(k+1)} - \boldsymbol{b})$  *// Exact dual ascent on equality residual*
>   $\lambda_i^{(k+1)} \leftarrow \max\bigl(0, \lambda_i^{(k)} + \rho f_i(\boldsymbol{x}^{(k+1)})\bigr)$ for $i = 1, \dots, m$  *// Projected dual ascent*
>   $r_{\mathrm{prim}} \leftarrow \lVert \boldsymbol{A}\boldsymbol{x}^{(k+1)} - \boldsymbol{b} \rVert_2 + \lVert \max(\boldsymbol{0}, \boldsymbol{f}(\boldsymbol{x}^{(k+1)})) \rVert_2$
>   $r_{\mathrm{cs}} \leftarrow \sum_{i=1}^m \lvert \lambda_i^{(k+1)} f_i(\boldsymbol{x}^{(k+1)}) \rvert$
>   **if** $r_{\mathrm{prim}} \le \varepsilon$ **and** $r_{\mathrm{cs}} \le \varepsilon$:
>
>     **break**
>
> **return** $(\boldsymbol{x}^{(k+1)}, \boldsymbol{\lambda}^{(k+1)}, \boldsymbol{\nu}^{(k+1)})$

### Complexity, Convergence & Cross-Book Methodological Synthesis

Table `tab:convex:solvers` compares the major algorithmic classes for constrained convex optimization discussed across the synthesized textbooks.

**Table:** Computational trade-offs among constrained convex optimization algorithms synthesized from [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8, pp. 305–309), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 24, pp. 307–316), and [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 4, pp. 93–96).

| **Method** | **Per-Iteration Cost** | **Iteration Rate** | **Feasibility** | **Suitability & Failure Mode** |
| --- | --- | --- | --- | --- |
| Projected GD (Murphy Sec. 8.6.1) | $\mathcal{O}(n)$ + $\operatorname{proj}_{\mathcal{X}}$ | $\mathcal{O}(1/\epsilon)$ smooth; $\mathcal{O}(\log(1/\epsilon))$ str. cvx | Strictly feasible at every step | Fast for simple boxes/balls/simplex; impractical if $\operatorname{proj}_{\mathcal{X}}$ is costly |
| Subgradient / Dual Ascent | $\mathcal{O}(n + m)$ (or $\min_{\boldsymbol{x}} L$) | $\mathcal{O}(1/\epsilon^2)$ nonsmooth | Dual feasible; primal only at limit | Decomposable across constraints; fails/slow if $L(\cdot,\boldsymbol{\lambda})$ is unbounded or non-strictly convex |
| Augmented Lagrangian (Alg. `alg:convex:alm`) | Inner L-BFGS / GD solve | Linear to superlinear (outer) | Asymptotic primal feasibility | Handles non-strictly convex $f_0$ without $\rho \to \infty$; relies on inner solver tolerance |
| Primal–Dual IPM (Alg. `alg:convex:ipm`) | $\mathcal{O}\bigl((n+p)^3\bigr)$ dense KKT solve | $\mathcal{O}(\sqrt{m}\log(m/\epsilon))$ Newton steps | Strictly interior ($\boldsymbol{f} \prec \boldsymbol{0}$) | Gold standard (e.g., `CVXOPT`) for small/medium LPs, QPs, GPs; memory-bound in huge $n$ |
| Simplex / Active-Set (Murphy Sec. 8.5.3) | $\mathcal{O}(mn)$ pivot / active KKT | Low poly typical; exponential worst-case | Vertex feasible at every step | Exact vertex solutions and warm starts for LPs/QPs; slower than IPM on large dense problems |

> [!info] Cross-Book Synthesis: Constrained Optimization Perspectives in ML
> Each of the five textbooks highlights a distinct facet of constrained convex theory tailored to its ML applications:
> - **Projected Gradient vs. Lagrangian Reparameterization:** Goodfellow et al. contrast small-step projected gradient descent (or tangent-space gradient projection) against unconstrained reparameterizations (e.g., $\boldsymbol{x}(\theta) = [\cos\theta, \sin\theta]^{\top}$ on $S^1$) and generalized KKT Lagrangians, emphasizing that deep networks rarely admit global convexity guarantees [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 4, pp. 93–96). Murphy unifies projected gradient descent rigorously as a *proximal gradient* update on the indicator function $I_{\mathcal{C}}(\boldsymbol{\theta})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8, Sec. 8.6.1, pp. 308–309).
> - **Normal-Vector Statics vs. Minimax Duality:** Bishop & Bishop derive KKT via local Taylor orthogonality ($\boldsymbol{\epsilon}^{\top}\nabla g = 0$) and boundary gradient opposition ($\nabla f = -\mu\nabla h$), building intuition for MaxEnt Gaussians and SVMs [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 2.5.4 & App. C, pp. 50–51, 621–624). Deisenroth et al. and Vũ complement this local view with global *minimax weak duality*, concave dual maximization, and Fenchel conjugation [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 7, pp. 233–245), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 25, pp. 317–325).
> - **From Convex Theory to Kernel Duals & Solvers:** Vũ demonstrates explicitly how Slater's condition verifies strong duality for both hard-margin and soft-margin SVMs (via scaling $(\boldsymbol{w}_1, b_1) = (2\boldsymbol{w}_0, 2b_0)$ or selecting slack $\xi_n$ sufficiently large) and pairs theory with practical `CVXOPT` interior-point LP/QP/GP solvers [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 24, pp. 307–316; Ch. 26–27, pp. 332, 342).

## Visual Diagnostics & Geometric Structure

We integrate three crisp 300 DPI source figures alongside bespoke TikZ geometric diagrams and PGFPlots curves to visualize supporting hyperplanes, active KKT constraints, Fenchel biconjugation, and primal–dual saddle/gap geometry.

> [!note] Source Figure
> *Figure:* Supporting hyperplane and first-order variational optimality condition `(eq:convex:variational_opt)` for a differentiable objective on a convex feasible set $\mathcal{X}$. At the optimal boundary point $\boldsymbol{x}_0$, every feasible step $\boldsymbol{x} - \boldsymbol{x}_0$ forms an obtuse angle with $-\nabla f_0(\boldsymbol{x}_0)$, so the level-set tangent plane supports $\mathcal{X}$. Reproduced from [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 24, Fig. 24.2, p. 307). (`machine_learning_co_ban_tiep_vu_p319_fig1.png`)

> [!note] Source Figure
> *Figure:* Active vs. inactive constraints in 2D constrained minimization: (a) equality constraint $\theta_1 + \theta_2 = 1$ forces parallel contour and constraint normals at $\boldsymbol{\theta}^* = (1/2, 1/2)$; (b) $\ell_1$-norm ball $\lvert \theta_1 \rvert + \lvert \theta_2 \rvert \le 1$ exhibits two active right faces ($\mu_1^*, \mu_2^* > 0$) and two inactive left faces ($\mu_3^* = \mu_4^* = 0$) at vertex $\boldsymbol{\theta}^* = (1, 0)^{\top}$. Reproduced from [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8, Fig. 8.20, p. 302). (`probabilisticml_kevinmurphy_introduction_p332_fig1.png`)

> [!note] Source Figure
> *Figure:* Lagrangian lower bounds and concave dual maximization for $\min_x x^2 + 10\sin(x) + 10$ subject to $(x-2)^2 \le 4$: (a) Lagrangian slices $f_0(x) + \lambda f_1(x)$ lie below $f_0(x)$ inside the feasible interval $[0, 4]$; (b) the dual function $g(\lambda)$ is concave and bounded above by $p^*$, here attaining zero duality gap despite non-convex $f_0$. Reproduced from [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 25, Fig. 25.1, p. 320). (`machine_learning_co_ban_tiep_vu_p332_fig1.png`)

Figure `fig:convex:tikz_epigraph_kkt` constructs the exact epigraph/Fenchel supporting hyperplane correspondence alongside the KKT normal-cone decomposition at an active vertex, while Figure `fig:convex:tikz_saddle` depicts primal–dual saddle geometry and the epigraph–value separation proof of Slater's theorem.

> [!example] Visual Diagram (LaTeX / PDF)
> *Geometric structure of Fenchel conjugation and KKT stationarity. (a) Any affine supporting tangent $t = sx - f^*(s)$ lies below $\operatorname{epi} f$, intersecting the vertical axis at $-f^*(s)$ with vertical Fenchel–Young gap $f(x) + f^*(s) - sx \ge 0$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 7.3.3, p. 243), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 8.4, p. 279). (b) At a KKT point $\boldsymbol{x}^*$, $-\nabla f_0(\boldsymbol{x}^*)$ lies inside the nonnegative normal cone $\mathcal{N}_{\mathcal{X}}(\boldsymbol{x}^*) = \operatorname{cone}\{\nabla f_1(\boldsymbol{x}^*), \nabla f_2(\boldsymbol{x}^*)\}$ spanned by active outward normals ($\lambda_1^*, \lambda_2^* > 0$), while inactive constraint $f_3$ receives $\lambda_3^* = 0$ [[Source - Foundations|deep_learning_foundations_concepts]] (App. C, p. 623), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (p. 307).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Primal–dual saddle geometry and Slater's theorem. (a) Any optimal primal–dual pair $(\boldsymbol{x}^*, \boldsymbol{\lambda}^*)$ with zero gap forms a saddle point of $L(\boldsymbol{x}, \boldsymbol{\lambda})$: convex-minimal along primal $\boldsymbol{x}$ and concave-maximal along dual $\boldsymbol{\lambda} \succeq \boldsymbol{0}$. (b) In $(u, t)$ constraint–objective space, Slater's strict feasibility ($\tilde{u} = f_1(\tilde{\boldsymbol{x}}) < 0$) forces the separating hyperplane supporting $\mathcal{A}$ at $(0, p^*)$ to be non-vertical ($\mu > 0$), ensuring $d^* = p^*$; non-convex dips extending below the supporting line for $u > 0$ create a strictly positive duality gap $p^* - d^* > 0$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:convex:pgf_fenchel_gap` presents exact PGFPlots diagnostics contrasting convex vs. non-convex Fenchel biconjugation ($f^{**} = f$ vs. $f^{**} < f$) alongside weak and strong Lagrangian duality curves $g(\lambda)$ vs. $p^*$.

> [!example] Visual Diagram (LaTeX / PDF)
> *Quantitative Fenchel conjugation and Lagrangian duality gaps. (a) For convex $f_{\mathrm{cvx}}(x) = \frac{1}{2}x^2$, supporting lines $y = sx - f^*(s)$ recover $f_{\mathrm{cvx}}^{**} = f_{\mathrm{cvx}}$ exactly; for the non-convex double well $f_{\mathrm{nc}}(x) = \frac{1}{2}(x^2-1)^2$, the Fenchel biconjugate $f_{\mathrm{nc}}^{**}$ flattens the central barrier into the convex hull ($f_{\mathrm{nc}}^{**} = 0$ on $[-1, 1]$). (b) For convex $\min_{x \ge 1} \frac{1}{2}x^2$, Slater's condition holds and $g_{\mathrm{cvx}}(\lambda) = \lambda - \frac{1}{2}\lambda^2$ attains $p_{\mathrm{cvx}}^* = 0.5$ at $\lambda^* = 1$; for non-convex $\min_{x \ge 0} (x^2-1)^2 + 0.4x$, $g_{\mathrm{nc}}(\lambda)$ remains concave with a subdifferential kink at $\lambda^* = 0.4$ ($d_{\mathrm{nc}}^* = 0$), leaving a strict duality gap $p_{\mathrm{nc}}^* - d_{\mathrm{nc}}^* \approx 0.389$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Non-Differentiable Kinks, Multi-Valued Duals & Subgradient Oscillation
> At non-differentiable kinks—such as $\ell_1$ regularizers ($\lvert x_j \rvert$ at $x_j = 0$), hinge losses, or dual functions $g(\lambda)$ where $\arg\min_{\boldsymbol{x}} L(\boldsymbol{x}, \lambda)$ jumps between distinct minimizers (e.g., $\lambda^* = 0.4$ in Figure `fig:convex:pgf_fenchel_gap`(b))—gradients do not exist [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8, Sec. 8.1.4, pp. 281–282). Standard fixed-step gradient descent oscillates perpetually across the kink because $\lVert \boldsymbol{g} \rVert_2$ need not vanish near $\boldsymbol{x}^*$.
> **Remedy:** Replace gradients with subgradients $\boldsymbol{g}^{(k)} \in \partial f(\boldsymbol{x}^{(k)})$ using square-summable diminishing step sizes ($\sum_k \alpha_k = \infty$, $\sum_k \alpha_k^2 < \infty$), or exploit composite splitting via proximal gradient operators $\boldsymbol{x}^{(k+1)} = \operatorname{prox}_{\eta L_r}(\boldsymbol{x}^{(k)} - \eta\nabla L_s(\boldsymbol{x}^{(k)}))$ (such as soft-thresholding for $\ell_1$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.6, pp. 308–310).

> [!warning] Failed Slater Qualification: Non-Zero Duality Gap & Missing KKT Multipliers
> Convexity alone does *not* guarantee strong duality or existence of KKT multipliers when Slater's condition `(eq:convex:slater)` fails [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 25, Sec. 25.3.2, p. 323):
> - **Convex Non-Zero Duality Gap:** Consider minimizing $f_0(x, y) = e^{-x}$ on $\mathcal{D} = \mathbb{R} \times \mathbb{R}_{>0}$ subject to $f_1(x, y) = x^2 / y \le 0$. Although $f_0$ and the quadratic-over-linear constraint $f_1$ are convex ($\nabla^2 f_1(x,y) = \frac{2}{y^3}[y,  -x]^{\top}[y,  -x] \succeq \boldsymbol{0}$), feasibility forces $x = 0$, leaving no strictly feasible point ($f_1(x, y) < 0$ is impossible). Hence $p^* = e^0 = 1$, whereas $g(\lambda) = \inf_{x \in \mathbb{R},  y > 0} (e^{-x} + \lambda x^2/y) = 0$ for all $\lambda \ge 0$, yielding $d^* = 0$ and a permanent duality gap $p^* - d^* = 1$.
> - **KKT Multiplier Non-Existence (Failed Constraint Qualification):** Minimizing $f_0(x) = x$ subject to $f_1(x) = x^2 \le 0$ is convex with unique optimum $x^* = 0$, yet Slater fails and $\nabla f_0(0) + \lambda \nabla f_1(0) = 1 + \lambda(0) = 1 \neq 0$ for every $\lambda \ge 0$: no KKT tuple exists at $x^* = 0$.
>
> **Remedy:** Always verify strict feasibility (e.g., as in Vũ's explicit Slater checks for hard/soft-margin SVMs [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (pp. 332, 342)) or replace degenerate constraints like $x^2 \le 0$ with linear equalities $x = 0$.

> [!warning] Spurious Saddle Points, Indefinite Hessians & Multiplier Sign Errors
> Outside convex programs, KKT stationarity `(eq:convex:kkt_stat)` holds at local maxima and saddle points as well as minima [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 4, pp. 92–95), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 8, pp. 277, 280).
> - **Indefinite Hessians:** At saddle points (e.g., $f(x, y) = x^2 - y^2$ in Vũ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Fig. 23.11, p. 293) and Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 8.6(d), p. 280)), $\nabla^2 L_{\boldsymbol{x}\boldsymbol{x}}$ possesses mixed-sign eigenvalues; second-order Newton steps blindly jump toward saddle points unless restricted to positive curvature on the tangent subspace $\ker(\boldsymbol{J}_{\mathrm{act}}(\boldsymbol{x}^*))$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (p. 92).
> - **Multiplier Sign Inversion:** Flipping $f_i(\boldsymbol{x}) \le 0$ vs. Bishop's $h_k(\boldsymbol{x}) \ge 0$ without adjusting the Lagrangian sign allows $\lambda_i < 0$, which reverses the normal-cone constraint and turns a constrained minimum into a boundary maximum [[Source - Foundations|deep_learning_foundations_concepts]] (App. C, pp. 623–624).

> [!warning] Barrier Conditioning Blowup Near Active Boundaries
> In primal log-barrier optimization $\phi_t(\boldsymbol{x}) = t f_0(\boldsymbol{x}) - \sum_{i=1}^m \ln(-f_i(\boldsymbol{x}))$, the barrier Hessian
>
> $$
>   \nabla^2 \phi_t(\boldsymbol{x}) = t\nabla^2 f_0(\boldsymbol{x}) + \sum_{i=1}^m \frac{1}{f_i(\boldsymbol{x})^2}\nabla f_i(\boldsymbol{x})\nabla f_i(\boldsymbol{x})^{\top} - \sum_{i=1}^m \frac{1}{f_i(\boldsymbol{x})}\nabla^2 f_i(\boldsymbol{x})
> $$
>
> exhibits rank-$1$ spikes scaling as $f_i(\boldsymbol{x})^{-2} = \mathcal{O}(t^2)$ perpendicular to active constraints, driving the Hessian condition number $\kappa_2(\nabla^2 \phi_t) \to \infty$ as $t \to \infty$. Solving the unscaled primal Newton system directly suffers severe floating-point cancellation; primal–dual interior-point formulations (Algorithm `alg:convex:ipm`) equalize scaling via the regularized complementarity block $-\mathrm{diag}(\boldsymbol{\lambda})\boldsymbol{f}(\boldsymbol{x}) = \frac{1}{t}\boldsymbol{1}_m$ and fraction-to-boundary step control.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[Bregman & Legendre Duality]], [[Convexity & Subgradients]], [[Lagrange Duality & KKT]]
- **Prerequisites (Directed $\leftarrow$):** [[Convexity & Subgradients]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Bregman & Legendre Duality]], [[Gradient Descent]], [[Kantorovich Duality]], [[Lagrange Duality & KKT]], [[Proximal & Subgradient Methods]], [[Support Vector Machines (SVM)]], [[TRPO]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Convexity & Subgradients]], [[Gradient Descent]], [[Kantorovich Duality]], [[Lagrange Duality & KKT]], [[Proximal & Subgradient Methods]], [[Sinkhorn Algorithm]], [[Support Vector Machines (SVM)]], [[TRPO]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
