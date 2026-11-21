---
type: monograph
domain: "D01"
parent_topic: "[[PT 01 - Probability Theory, Information Theory & Optimal Transport]]"
sub_topic: "[[ST 01 - Information Theory, Divergences & Optimal Transport]]"
cheatsheet: "[[Cheatsheet - Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]"
part_of: 
  - "[[ST 01 - Information Theory, Divergences & Optimal Transport]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
concepts: 
  - "[[Kantorovich Duality]]"
  - "[[Sinkhorn Algorithm]]"
  - "[[Wasserstein Distance]]"
prerequisites: 
  - "[[KL Divergence]]"
  - "[[Lagrange Duality & KKT]]"
  - "[[Wasserstein Distance]]"
enables: 
  - "[[Flow Matching & Rectified Flow]]"
  - "[[Sinkhorn Algorithm]]"
  - "[[Wasserstein GAN (WGAN)]]"
related: 
  - "[[Bregman & Legendre Duality]]"
  - "[[Flow Matching & Rectified Flow]]"
  - "[[Lagrange Duality & KKT]]"
  - "[[Sinkhorn Algorithm]]"
  - "[[Wasserstein Distance]]"
  - "[[Wasserstein GAN (WGAN)]]"
sources: 
  - "[[Source - Optimal Transport in Large-Scale]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
tags:
  - monograph
  - domain-01
---

# Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling

> [!abstract] Monograph Navigation
> - **Domain:** [[D01 - Mathematical Foundations for Machine Intelligence|Mathematical Foundations for Machine Intelligence]]
> - **Parent / Sub-Topic:** [[PT 01 - Probability Theory, Information Theory & Optimal Transport|Probability Theory, Information Theory & Optimal Transport]] → [[ST 01 - Information Theory, Divergences & Optimal Transport|Information Theory, Divergences & Optimal Transport]]
> - **Visual Cheatsheet:** [[Cheatsheet - Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]
> - **Description:** Unifies Monge maps, Kantorovich LP relaxation, dual Kantorovich-Rubinstein potential form, entropic regularization, Sinkhorn-Knopp matrix scaling complexity, and minibatch/sliced OT.

## Problem Formulation & Unified Notation

Whereas information-theoretic $f$-divergences quantify discrepancy vertically through pointwise density ratios $\mathrm{d}\mu / \mathrm{d}\nu$ and therefore blow up or saturate when measures have disjoint supports, optimal transport (OT) compares probability distributions *horizontally* by lifting a ground metric on the sample space to a variational cost over probability measures [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slides 46–49) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8, p. 314).

Let $(\mathcal{X}, d)$ be a Polish metric space (canonically Euclidean space $\mathbb{R}^d$ with $d(\boldsymbol{x}, \boldsymbol{y}) = \lVert \boldsymbol{x} - \boldsymbol{y} \rVert_2$), and let $\mathcal{P}(\mathcal{X})$ denote the space of Borel probability measures on $\mathcal{X}$. Given a measurable non-negative ground cost $c : \mathcal{X} \times \mathcal{X} \to \mathbb{R}_{\ge 0}$ (typically $c(\boldsymbol{x}, \boldsymbol{y}) = d(\boldsymbol{x}, \boldsymbol{y})^p$ for $p \ge 1$, or $\frac{1}{2}\lVert \boldsymbol{x}-\boldsymbol{y} \rVert_2^2$), we seek the least-cost transformation coupling a source measure $\mu \in \mathcal{P}(\mathcal{X})$ to a target measure $\nu \in \mathcal{P}(\mathcal{X})$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 7.2.1, pp. 194–195). For $p$-Wasserstein distances, we restrict to $\mathcal{P}_p(\mathcal{X}) \triangleq \{ \rho \in \mathcal{P}(\mathcal{X}) : \int_{\mathcal{X}} d(\boldsymbol{x}, \boldsymbol{x}_0)^p \mathrm{d}\rho(\boldsymbol{x}) < \infty \}$ for an arbitrary anchor $\boldsymbol{x}_0 \in \mathcal{X}$.

In computational and empirical machine learning settings, $\mu$ and $\nu$ are supported on finite point clouds $\{\boldsymbol{x}_i\}_{i=1}^n \subset \mathcal{X}$ and $\{\boldsymbol{y}_j\}_{j=1}^m \subset \mathcal{X}$ with probability weights $\boldsymbol{a} \in \Delta_n$ and $\boldsymbol{b} \in \Delta_m$ on the unit simplices ($\sum_{i=1}^n a_i = \sum_{j=1}^m b_j = 1$, $a_i, b_j > 0$):

$$
\mu = \sum_{i=1}^n a_i \delta_{\boldsymbol{x}_i}, \qquad
  \nu = \sum_{j=1}^m b_j \delta_{\boldsymbol{y}_j}, \qquad
  C_{ij} \triangleq c(\boldsymbol{x}_i, \boldsymbol{y}_j) \in \mathbb{R}_{\ge 0}^{n \times m}.
$$

Admissible joint couplings form the **transportation polytope** [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slide 27) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 6.192, p. 316):

$$
\Pi(\boldsymbol{a}, \boldsymbol{b}) \triangleq \bigl\{ \boldsymbol{\pi} \in \mathbb{R}_{\ge 0}^{n \times m} : \boldsymbol{\pi}\boldsymbol{1}_m = \boldsymbol{a},  \boldsymbol{\pi}^{\top}\boldsymbol{1}_n = \boldsymbol{b} \bigr\}.
$$

When $n = m$ and $\boldsymbol{a} = \boldsymbol{b} = \frac{1}{n}\boldsymbol{1}_n$, scaling by $n$ maps $\Pi(\frac{1}{n}\boldsymbol{1}_n, \frac{1}{n}\boldsymbol{1}_n)$ bijectively onto the **Birkhoff polytope** $\mathcal{B}_n$ of $n \times n$ doubly stochastic matrices, whose extreme points are the $n!$ permutation matrices $\boldsymbol{P}_\sigma$ ($\sigma \in S_n$) by the Birkhoff–von Neumann theorem [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8.1, p. 314).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> The three synthesized sources originate in statistical optimization, scalable ML systems, and continuous-time diffusion theory; the table below reconciles their distinct symbol sets into a single consistent monograph notation.
>
> | **Concept** | **Unified Monograph** | **Source Variants** | **Reconciliation** |
> | --- | --- | --- | --- |
> | Source / target measures | $\mu, \nu \in \mathcal{P}(\mathcal{X})$; weights $\boldsymbol{a}, \boldsymbol{b}$ | $\mu, \nu$ ($\boldsymbol{a}, \boldsymbol{b}$) in Murphy; $P, Q$ ($\boldsymbol{p}, \boldsymbol{q}$) in Ho; $p_{\mathrm{src}}, p_{\mathrm{tgt}}$ in Diffusion Models | Continuous $\mu, \nu$ generalize both densities $p_{\mathrm{src}}, p_{\mathrm{tgt}}$ and discrete $\sum a_i \delta_{\boldsymbol{x}_i}$. |
> | Support atoms | $\boldsymbol{x}_i, \boldsymbol{y}_j \in \mathbb{R}^d$ | $\boldsymbol{x}_i, \boldsymbol{y}_j$ in Murphy; $\eta_i, \theta_j$ (or $B_i, R_j$) in Ho; $\boldsymbol{x}^{(i)}, \boldsymbol{y}^{(j)}$ in Diffusion Models | Subscripts index support elements; boldface denotes vectors in $\mathbb{R}^d$. |
> | Coupling & feasible set | $\pi \in \Pi(\mu, \nu)$ (discrete $\boldsymbol{\pi} \in \Pi(\boldsymbol{a}, \boldsymbol{b})$) | $P \in \mathbb{R}_+^{n\times m}$ / $\pi \in \Pi(\mu,\nu)$ in Murphy; $\pi \in \mathcal{P}(\boldsymbol{p},\boldsymbol{q})$ in Ho; $\gamma \in \Gamma(p_{\mathrm{src}},p_{\mathrm{tgt}})$ in Diffusion Models | Avoids collision with probability densities $p, q$ and Euler–Maruyama steps $\gamma$. |
> | Entropic regularizer | $\varepsilon > 0$ | $\gamma > 0$ in Murphy; $\eta > 0$ in Ho; $\varepsilon > 0$ in Diffusion Models | Matches Brownian diffusivity $\mathrm{d}\boldsymbol{x}_t = \sqrt{\varepsilon} \mathrm{d}\boldsymbol{w}_t$ in Schrödinger bridges. |
> | Entropy penalty form | $\varepsilon D_{\mathrm{KL}}(\pi \parallel \mu \otimes \nu)$ | $\gamma D_{\mathrm{KL}}(\pi \parallel \mu \otimes \nu)$ in Murphy; $-\eta H(\pi)$ in Ho; $\varepsilon D_{\mathrm{KL}}(\gamma \parallel p_{\mathrm{src}} \otimes p_{\mathrm{tgt}})$ in Diffusion Models | Discrete identity $D_{\mathrm{KL}}(\boldsymbol{\pi} \parallel \boldsymbol{a}\boldsymbol{b}^{\top}) = -H(\boldsymbol{\pi}) + H(\boldsymbol{a}) + H(\boldsymbol{b})$ differs only by a marginal constant. |
> | Dual potentials | $(\boldsymbol{f}, \boldsymbol{g}) \in \mathbb{R}^n \times \mathbb{R}^m$; scaled $(u, v) = (e^{f/\varepsilon}, e^{g/\varepsilon})$ | $(f, g)$ with $\mathrm{diag}(e^{f/\gamma})\boldsymbol{K}\mathrm{diag}(e^{g/\gamma})$ in Murphy; dimensionless $(u, v) = (f/\eta, g/\eta)$ in Ho; $(\Psi, \widehat{\Psi})$ in Diffusion Models | Log-potentials $(f,g)$ carry cost units; dimensionless $(u,v) = (f/\varepsilon, g/\varepsilon)$ are log-scalings. |
> | Quadratic cost factor | $c(\boldsymbol{x}, \boldsymbol{y}) = \lVert \boldsymbol{x}-\boldsymbol{y} \rVert_2^2$ (for $W_2^2$); $\frac{1}{2}\lVert \boldsymbol{x}-\boldsymbol{y} \rVert_2^2$ in Brenier | Both forms in Murphy; $\lVert \boldsymbol{x}-\boldsymbol{y} \rVert_2^2$ in Ho; $\lVert \boldsymbol{x}-\boldsymbol{y} \rVert_2^2$ and $\frac{\lVert \boldsymbol{y}-\boldsymbol{x} \rVert^2}{2T}$ in Diffusion Models | Factor $\frac{1}{2}$ cancels the $2$ in $\nabla(\frac{1}{2}\lVert \cdot \rVert^2) = \mathrm{Id}$, yielding $T^\star = \nabla u$. |
>
> *Synthesized Sources:* [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 2.7.2, 6.8, 26.2.4, pp. 57–58, 314–322, 900–901), [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slides 16–39, 46–97), and [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Ch. 7, Sec. 7.1–7.5, pp. 191–223).

## Core Mathematical Derivations & Theorems

### From Monge Pushforward Maps to Kantorovich Relaxation

Consider first matching two unweighted clouds $\{\boldsymbol{x}_i\}_{i=1}^n$ and $\{\boldsymbol{y}_j\}_{j=1}^n$ of equal cardinality. Any one-to-one assignment corresponds to a permutation $\sigma \in S_n$ minimizing $\frac{1}{n}\sum_{i=1}^n C_{i,\sigma(i)}$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slide 17) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 6.191, p. 315). Exhaustive enumeration requires $n!$ evaluations, whereas the Hungarian algorithm solves this bipartite matching in $O(n^3)$ time [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slide 18). Extending this bijection to general measures requires the **pushforward** operator: a measurable map $T : \mathcal{X} \to \mathcal{X}$ satisfies $T_\sharp \mu = \nu$ iff $\mu(T^{-1}(A)) = \nu(A)$ for every Borel set $A \subseteq \mathcal{X}$, or equivalently $\int_{\mathcal{X}} \varphi(T(\boldsymbol{x})) \mathrm{d}\mu(\boldsymbol{x}) = \int_{\mathcal{X}} \varphi(\boldsymbol{y}) \mathrm{d}\nu(\boldsymbol{y})$ for all bounded measurable test functions $\varphi$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slides 20–21). When $T$ is a diffeomorphism on $\mathbb{R}^d$ and $\mu, \nu$ admit Lebesgue densities $p, q$, the change-of-variables formula enforces

$$
q\bigl(T(\boldsymbol{x})\bigr) \lvert \det \boldsymbol{J}_T(\boldsymbol{x}) \rvert = p(\boldsymbol{x}) \quad \text{almost everywhere,}
$$

where $\boldsymbol{J}_T(\boldsymbol{x}) = \nabla T(\boldsymbol{x}) \in \mathbb{R}^{d \times d}$ denotes the Jacobian of $T$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Eq. 7.2.4, p. 197) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 6.193, p. 317).

> [!important] Monge Problem, Obstructions & Kantorovich Linear Program
> **1. Monge's Deterministic Formulation.** Given $(\mu, \nu)$ and ground cost $c$, Monge's problem seeks a measurable transport map $T : \mathcal{X} \to \mathcal{X}$ achieving [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Eq. (2), Slide 21) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 6.194, p. 317):
>
> $$
> \mathrm{Monge}_c(\mu, \nu) \triangleq \inf_{T : T_\sharp \mu = \nu} \int_{\mathcal{X}} c\bigl(\boldsymbol{x}, T(\boldsymbol{x})\bigr) \mathrm{d}\mu(\boldsymbol{x}).
> $$
>
> Two structural obstructions limit Eq. `(eq:ot:monge)`: (i) *non-existence*—because $T$ cannot split mass at a single point, mapping $\mu = \delta_{\boldsymbol{x}_0}$ to $\nu = \frac{1}{2}\delta_{\boldsymbol{y}_1} + \frac{1}{2}\delta_{\boldsymbol{y}_2}$ ($\boldsymbol{y}_1 \ne \boldsymbol{y}_2$) yields an empty feasible set since $\mu(T^{-1}(\{\boldsymbol{y}_1\})) \in \{0, 1\} \ne \frac{1}{2}$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slide 22); and (ii) *non-convexity*—even when $T_\sharp \mu = \nu$ and $T'_\sharp \mu = \nu$, the convex combination $(\frac{1}{2}T + \frac{1}{2}T')_\sharp \mu \ne \nu$ in general [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8.2.2, p. 317).
>
> **2. Kantorovich's Probabilistic Relaxation.** Kantorovich replaced deterministic maps with probabilistic couplings $\pi \in \Pi(\mu, \nu) \subset \mathcal{P}(\mathcal{X} \times \mathcal{X})$ satisfying the marginal constraints $\pi(A \times \mathcal{X}) = \mu(A)$ and $\pi(\mathcal{X} \times B) = \nu(B)$ for all Borel $A, B \subseteq \mathcal{X}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. 6.195–6.196, p. 317) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Eq. 7.2.1, p. 195):
>
> $$
> \mathrm{OT}_c(\mu, \nu) \triangleq \inf_{\pi \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{X}} c(\boldsymbol{x}, \boldsymbol{y}) \mathrm{d}\pi(\boldsymbol{x}, \boldsymbol{y}).
> $$
>
> Unlike Monge's feasible set, $\Pi(\mu, \nu)$ is always non-empty (containing $\mu \otimes \nu$), convex, and weakly compact on Polish spaces when $c$ is lower semicontinuous and bounded below, guaranteeing attainment of an optimal coupling $\pi^\star$. Every Monge map $T$ induces a deterministic coupling $\pi_T = (\mathrm{Id}, T)_\sharp \mu \in \Pi(\mu, \nu)$ with identical cost, so $\mathrm{OT}_c(\mu, \nu) \le \mathrm{Monge}_c(\mu, \nu)$.
>
> **3. Discrete Linear Program & Birkhoff Vertices.** For discrete measures `(eq:ot:discrete_measures)`, Eq. `(eq:ot:kantorovich)` becomes the finite-dimensional linear program [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Eq. (5), Slide 27) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 6.192, p. 316):
>
> $$
> \mathrm{OT}_{\boldsymbol{C}}(\boldsymbol{a}, \boldsymbol{b}) = \min_{\boldsymbol{\pi} \in \Pi(\boldsymbol{a}, \boldsymbol{b})} \langle \boldsymbol{C}, \boldsymbol{\pi} \rangle \triangleq \sum_{i=1}^n \sum_{j=1}^m C_{ij}\pi_{ij}.
> $$
>
> When $n = m$ and $\boldsymbol{a} = \boldsymbol{b} = \frac{1}{n}\boldsymbol{1}_n$, the Birkhoff–von Neumann theorem characterizes the extreme points of $\Pi(\frac{1}{n}\boldsymbol{1}_n, \frac{1}{n}\boldsymbol{1}_n)$ as the scaled permutation matrices $\{\frac{1}{n}\boldsymbol{P}_\sigma : \sigma \in S_n\}$. Because a linear program attains its minimum at an extreme point of its feasible polytope, Eq. `(eq:ot:discrete_lp)` admits an optimal permutation vertex, proving exact equivalence between Monge assignment and Kantorovich relaxation in the balanced uniform discrete case.

### Kantorovich Duality, $c$-Transforms, and Wasserstein-$p$ Metrics

Because Eq. `(eq:ot:kantorovich)` is an infinite-dimensional linear program, strong duality transforms constrained mass transport over couplings $\pi$ into maximization over dual potential pairs $(f, g)$ satisfying a pointwise cost inequality [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8.3.1, p. 318).

> [!important] Kantorovich Duality, $c$-Concavity & Kantorovich–Rubinstein Theorem
> **1. Primal-to-Dual Lagrangian Derivation.** In discrete form `(eq:ot:discrete_lp)`, enforce the marginal constraints $\boldsymbol{\pi}\boldsymbol{1}_m = \boldsymbol{a}$ and $\boldsymbol{\pi}^{\top}\boldsymbol{1}_n = \boldsymbol{b}$ via unconstrained Lagrange multipliers $\boldsymbol{f} \in \mathbb{R}^n$ and $\boldsymbol{g} \in \mathbb{R}^m$ while retaining $\boldsymbol{\pi} \ge \boldsymbol{0}$:
>
> $$
> \min_{\boldsymbol{\pi} \ge \boldsymbol{0}} \sup_{\boldsymbol{f} \in \mathbb{R}^n,  \boldsymbol{g} \in \mathbb{R}^m}
>   \boldsymbol{f}^{\top}\boldsymbol{a} + \boldsymbol{g}^{\top}\boldsymbol{b} + \sum_{i=1}^n \sum_{j=1}^m \bigl(C_{ij} - f_i - g_j\bigr)\pi_{ij}.
> $$
>
> By Slater's condition ($\boldsymbol{\pi} = \boldsymbol{a}\boldsymbol{b}^{\top} > \boldsymbol{0}$ lies in the relative interior of the non-negative orthant), strong duality permits swapping $\min$ and $\sup$. Minimizing over $\pi_{ij} \ge 0$ yields $-\infty$ unless $f_i + g_j \le C_{ij}$ for all $(i,j)$, in which case the infimum occurs at $(C_{ij} - f_i - g_j)\pi_{ij} = 0$. Passing to continuous measures $(\mu, \nu)$ gives the **Kantorovich dual** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. 6.198–6.199, p. 318):
>
> $$
> \mathrm{OT}_c(\mu, \nu) = \sup_{\substack{f \in L^1(\mu),  g \in L^1(\nu) \\ f(\boldsymbol{x}) + g(\boldsymbol{y}) \le c(\boldsymbol{x},\boldsymbol{y})}}
>   \int_{\mathcal{X}} f(\boldsymbol{x}) \mathrm{d}\mu(\boldsymbol{x}) + \int_{\mathcal{X}} g(\boldsymbol{y}) \mathrm{d}\nu(\boldsymbol{y}),
> $$
>
> with complementary slackness $f^\star(\boldsymbol{x}) + g^\star(\boldsymbol{y}) = c(\boldsymbol{x}, \boldsymbol{y})$ holding $\pi^\star$-almost everywhere on $\mathrm{supp}(\pi^\star)$.
>
> **2. The $c$-Transform & Semi-Dual Reduction.** Fixing $f$, the pointwise tightest admissible $g$ in Eq. `(eq:ot:dual)` is its **$c$-transform** $f^c$ (and symmetrically $g^{\bar{c}}$ for fixed $g$, where $f^c = f^{\bar{c}}$ for symmetric $c$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. 6.200–6.203, p. 318):
>
> $$
> f^c(\boldsymbol{y}) \triangleq \inf_{\boldsymbol{x} \in \mathcal{X}} \bigl[ c(\boldsymbol{x}, \boldsymbol{y}) - f(\boldsymbol{x}) \bigr], \qquad
>   g^{\bar{c}}(\boldsymbol{x}) \triangleq \inf_{\boldsymbol{y} \in \mathcal{X}} \bigl[ c(\boldsymbol{x}, \boldsymbol{y}) - g(\boldsymbol{y}) \bigr].
> $$
>
> By definition, $f(\boldsymbol{x}) + f^c(\boldsymbol{y}) \le c(\boldsymbol{x},\boldsymbol{y}) \implies f(\boldsymbol{x}) \le (f^c)^{\bar{c}}(\boldsymbol{x})$. Taking the $c$-transform again flips order, giving $(f^{c\bar{c}})^c(\boldsymbol{y}) \le f^c(\boldsymbol{y})$, while substituting $g = f^c$ gives $(f^c)^{\bar{c}}(\boldsymbol{x}) + f^c(\boldsymbol{y}) \le c(\boldsymbol{x},\boldsymbol{y}) \implies f^c(\boldsymbol{y}) \le (f^{c\bar{c}})^c(\boldsymbol{y})$. Hence $f^{c\bar{c}c} = f^c$ identically: double conjugation is idempotent. Restricting to the class of **$c$-concave functions** $\mathcal{F}_c \triangleq \{ f : f = (f^c)^{\bar{c}} \}$ eliminates $g$ and yields the semi-dual problem:
>
> $$
> \mathrm{OT}_c(\mu, \nu) = \sup_{f \in \mathcal{F}_c} \int_{\mathcal{X}} f(\boldsymbol{x}) \mathrm{d}\mu(\boldsymbol{x}) + \int_{\mathcal{X}} f^c(\boldsymbol{y}) \mathrm{d}\nu(\boldsymbol{y}).
> $$
>
> **3. Wasserstein-$p$ Distance & Kantorovich–Rubinstein Duality ($p = 1$).** For any $p \in [1, \infty)$ and metric $d$, the **Wasserstein-$p$ distance** on $\mathcal{P}_p(\mathcal{X})$ is [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 6.197, p. 317):
>
> $$
> W_{p}(\mu, \nu) \triangleq \left( \inf_{\pi \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{X}} d(\boldsymbol{x}, \boldsymbol{y})^p \mathrm{d}\pi(\boldsymbol{x}, \boldsymbol{y}) \right)^{1/p}.
> $$
>
> Symmetry and definiteness ($W_{p}(\mu, \nu) = 0 \iff \mu = \nu$) follow directly from metric axioms of $d$, while the triangle inequality $W_{p}(\mu, \rho) \le W_{p}(\mu, \nu) + W_{p}(\nu, \rho)$ follows by gluing optimal plans $(\pi_{12}, \pi_{23})$ over the intermediate marginal $\nu$ (the *gluing lemma*) and applying Minkowski's inequality in $L^p$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8.2.4, p. 317).
>
> Specializing to $p = 1$ ($c(\boldsymbol{x}, \boldsymbol{y}) = d(\boldsymbol{x}, \boldsymbol{y})$), suppose $f$ is $d$-concave, so $f(\boldsymbol{x}) = \inf_{\boldsymbol{z}} [d(\boldsymbol{x}, \boldsymbol{z}) - g(\boldsymbol{z})]$. For any $\boldsymbol{x}, \boldsymbol{y} \in \mathcal{X}$, the triangle inequality $d(\boldsymbol{x}, \boldsymbol{z}) \le d(\boldsymbol{x}, \boldsymbol{y}) + d(\boldsymbol{y}, \boldsymbol{z})$ implies $f(\boldsymbol{x}) - f(\boldsymbol{y}) \le d(\boldsymbol{x}, \boldsymbol{y})$; by symmetry, $\lvert f(\boldsymbol{x}) - f(\boldsymbol{y}) \rvert \le d(\boldsymbol{x}, \boldsymbol{y})$, so $f$ is $1$-Lipschitz ($\lVert f \rVert_{\mathrm{Lip}} \le 1$). Conversely, if $\lVert f \rVert_{\mathrm{Lip}} \le 1$, then $f(\boldsymbol{x}) - f(\boldsymbol{y}) \le d(\boldsymbol{x}, \boldsymbol{y}) \implies -f(\boldsymbol{y}) \le d(\boldsymbol{x}, \boldsymbol{y}) - f(\boldsymbol{x})$ for all $\boldsymbol{x}$, with equality at $\boldsymbol{x} = \boldsymbol{y}$; thus $f^d(\boldsymbol{y}) = -f(\boldsymbol{y})$ identically! Substituting $f^d = -f$ into Eq. `(eq:ot:semidual)` proves the **Kantorovich–Rubinstein duality** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. 2.297, 6.204, 26.29, pp. 58, 319, 900) [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Eq. (11), Slide 51):
>
> $$
> W_{1}(\mu, \nu) = \sup_{\lVert f \rVert_{\mathrm{Lip}} \le 1} \left( \int_{\mathcal{X}} f(\boldsymbol{x}) \mathrm{d}\mu(\boldsymbol{x}) - \int_{\mathcal{X}} f(\boldsymbol{y}) \mathrm{d}\nu(\boldsymbol{y}) \right),
> $$
>
> identifying $W_{1}(\cdot, \cdot)$ as an Integral Probability Metric (IPM) over the unit Lipschitz ball.

Two tractable closed forms allow computing $W_{p}(\cdot, \cdot)$ explicitly without iterative optimization [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8.3.4, pp. 320–321):
- **Univariate Distributions ($d = 1$):** With generalized quantile functions $Q_\mu(u) = F_\mu^{-1}(u)$ and $Q_\nu(u) = F_\nu^{-1}(u)$ on $u \in [0, 1]$, any convex cost $h(x - y)$ yields $\mathrm{OT}_h(\mu, \nu) = \int_0^1 h(Q_\mu(u) - Q_\nu(u)) \mathrm{d}u$, so $W_{p}(\mu, \nu) = \lVert Q_\mu - Q_\nu \rVert_{L^p([0,1])}$ and the optimal Monge map (for atomless $\mu$) is $T^\star = Q_\nu \circ F_\mu$ (computable on $n$ uniform points in $O(n \log n)$ via sorting [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slide 18)).
- **Multivariate Gaussians (Bures–Wasserstein-2):** For $\mu = \mathcal{N}(\boldsymbol{m}_1, \boldsymbol{\Sigma}_1)$ and $\nu = \mathcal{N}(\boldsymbol{m}_2, \boldsymbol{\Sigma}_2)$ on $\mathbb{R}^d$, $W_{2}(\mu, \nu)^2$ splits into mean displacement plus the Bures metric on SPD matrices [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. 6.210–6.212, pp. 320–321):
  
$$
W_{2}(\mathcal{N}(\boldsymbol{m}_1, \boldsymbol{\Sigma}_1), \mathcal{N}(\boldsymbol{m}_2, \boldsymbol{\Sigma}_2))^2
    = \lVert \boldsymbol{m}_1 - \boldsymbol{m}_2 \rVert_2^2 + \mathrm{Tr}\!\left(\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2 - 2\bigl(\boldsymbol{\Sigma}_1^{1/2}\boldsymbol{\Sigma}_2\boldsymbol{\Sigma}_1^{1/2}\bigr)^{1/2}\right),
$$

  with linear Monge map $T^\star(\boldsymbol{x}) = \boldsymbol{A}(\boldsymbol{x} - \boldsymbol{m}_1) + \boldsymbol{m}_2$ where $\boldsymbol{A} = \boldsymbol{\Sigma}_1^{-1/2}\bigl(\boldsymbol{\Sigma}_1^{1/2}\boldsymbol{\Sigma}_2\boldsymbol{\Sigma}_1^{1/2}\bigr)^{1/2}\boldsymbol{\Sigma}_1^{-1/2} \succ \boldsymbol{0}$ whenever $\boldsymbol{\Sigma}_1 \succ \boldsymbol{0}$.

> [!tip] Horizontal Metric Geometry vs. Vertical $f$-Divergences
> Consider two point masses $\mu = \delta_0$ and $\nu_\theta = \delta_\theta$ on $\mathbb{R}$ (or parallel 1D uniform lines $P = (0, \mathcal{U}(0,1))$ and $Q_\theta = (\theta, \mathcal{U}(0,1))$ in $\mathbb{R}^2$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slides 46–49)). Whenever $\theta \ne 0$, the supports are completely disjoint, causing $D_{\mathrm{KL}}(\mu \parallel \nu_\theta) = +\infty$, $D_{\mathrm{TV}}(\mu, \nu_\theta) = 1$, and $D_{\mathrm{JS}}(\mu \parallel \nu_\theta) = \ln 2$ with zero gradient $\nabla_\theta D_{\mathrm{JS}}(\mu \parallel \nu_\theta) = 0$. In stark contrast, Wasserstein metrics scale smoothly with displacement: $W_{1}(\mu, \nu_\theta) = \lvert \theta \rvert$ (gradient $\mathrm{sgn}(\theta)$) and $W_{2}(\mu, \nu_\theta)^2 = \theta^2$ (gradient $2\theta$). Thus $W_{p}(\cdot, \cdot)$ metrizes weak convergence of measures plus convergence of $p$-th moments, furnishing informative gradients for generator learning on low-dimensional manifolds [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 26.2.4, p. 900).

### Brenier's Theorem, McCann Interpolation, and Benamou–Brenier Dynamics

When the ground cost is quadratic $c(\boldsymbol{x}, \boldsymbol{y}) = \frac{1}{2}\lVert \boldsymbol{x} - \boldsymbol{y} \rVert_2^2$ on $\mathbb{R}^d$, complementary slackness in Eq. `(eq:ot:dual)` collapses Kantorovich couplings onto the gradient of a convex scalar potential, bridging static transport to fluid dynamics and generative flows [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8.3.3, pp. 319–320) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 7.2.1 & 7.5, pp. 195–197, 217–223).

> [!important] Brenier's Theorem, Monge–Ampère PDE & Benamou–Brenier Fluid Action
> **1. Brenier's Polar Factorization.** Suppose $\mu, \nu \in \mathcal{P}_2(\mathbb{R}^d)$ and $\mu$ gives zero mass to $(d-1)$-dimensional sets (in particular, $\mu$ admits a Lebesgue density $p_{\mathrm{src}}$) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (p. 195, Thm. 7.1 p. 217). Let $(f^\star, (f^\star)^c)$ optimize Eq. `(eq:ot:semidual)` for $c(\boldsymbol{x},\boldsymbol{y}) = \frac{1}{2}\lVert \boldsymbol{x}-\boldsymbol{y} \rVert_2^2$. For any $(\boldsymbol{x}_0, \boldsymbol{y}_0) \in \mathrm{supp}(\pi^\star)$, complementary slackness gives $f^\star(\boldsymbol{x}_0) + (f^\star)^c(\boldsymbol{y}_0) = \frac{1}{2}\lVert \boldsymbol{x}_0 - \boldsymbol{y}_0 \rVert_2^2$, whereas $(f^\star)^c(\boldsymbol{y}_0) = \inf_{\boldsymbol{x}} [\frac{1}{2}\lVert \boldsymbol{x} - \boldsymbol{y}_0 \rVert_2^2 - f^\star(\boldsymbol{x})]$. Therefore $\boldsymbol{x}_0$ minimizes $\boldsymbol{x} \mapsto \frac{1}{2}\lVert \boldsymbol{x}-\boldsymbol{y}_0 \rVert_2^2 - f^\star(\boldsymbol{x})$, so first-order stationarity yields [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. 6.206–6.208, pp. 319–320):
>
> $$
> \boldsymbol{y}_0 - \boldsymbol{x}_0 - \nabla f^\star(\boldsymbol{x}_0) = \boldsymbol{0} \implies
>   \boldsymbol{y}_0 = T^\star(\boldsymbol{x}_0) = \nabla u(\boldsymbol{x}_0), \qquad
>   u(\boldsymbol{x}) \triangleq \frac{1}{2}\lVert \boldsymbol{x} \rVert_2^2 - f^\star(\boldsymbol{x}).
> $$
>
> Unfolding the $c$-transform shows $u^\ast(\boldsymbol{y}) = \sup_{\boldsymbol{x}} \{\langle \boldsymbol{x}, \boldsymbol{y} \rangle - u(\boldsymbol{x})\} = \frac{1}{2}\lVert \boldsymbol{y} \rVert_2^2 - (f^\star)^c(\boldsymbol{y})$, so $u = u^{\ast\ast}$ is convex iff $f^\star$ is $c$-concave. Hence the unique optimal coupling is deterministic, $\pi^\star = (\mathrm{Id}, \nabla u)_\sharp \mu$, and combining $\nabla T^\star = \nabla^2 u \succeq \boldsymbol{0}$ with Eq. `(eq:ot:pushforward_jacobian)` yields the nonlinear **Monge–Ampère PDE** [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Eq. 7.2.4, p. 197):
>
> $$
> p_{\mathrm{tgt}}\bigl(\nabla u(\boldsymbol{x})\bigr) \det\nabla^2 u(\boldsymbol{x}) = p_{\mathrm{src}}(\boldsymbol{x}).
> $$
>
> **2. McCann Displacement Interpolation & Benamou–Brenier Action.** Along $t \in [0, 1]$, particles following $T^\star$ trace constant-velocity straight lines $T_t^\star(\boldsymbol{x}) = (1 - t)\boldsymbol{x} + t T^\star(\boldsymbol{x})$, defining **McCann's displacement interpolation** $p_t^{\mathrm{OT}} = (T_t^\star)_\sharp p_{\mathrm{src}}$—the constant-speed geodesic in $(\mathcal{P}_2(\mathbb{R}^d), W_2)$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (p. 197). Benamou and Brenier (2000) proved that this geodesic minimizes kinetic energy over all Eulerian density–velocity paths $(p_t, \boldsymbol{v}_t)$ solving the continuity equation [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Eq. 7.2.3, p. 197):
>
> $$
> \begin{aligned}
>   W_{2}(p_{\mathrm{src}}, p_{\mathrm{tgt}})^2
>   &= \inf_{(p_t, \boldsymbol{v}_t)} \int_0^1 \int_{\mathbb{R}^d} \lVert \boldsymbol{v}_t(\boldsymbol{x}) \rVert_2^2  p_t(\boldsymbol{x}) \mathrm{d}\boldsymbol{x} \mathrm{d}t \\
>   &\quad \text{s.t.} \quad \partial_t p_t + \nabla \cdot (p_t \boldsymbol{v}_t) = 0,   p_0 = p_{\mathrm{src}},   p_1 = p_{\mathrm{tgt}}.
> \end{aligned}
> $$
>
> Optimal pairs satisfy $\boldsymbol{v}_t(\boldsymbol{x}) = \nabla \varphi_t(\boldsymbol{x})$ where $\varphi_t$ solves the Hamilton–Jacobi equation $\partial_t \varphi_t + \frac{1}{2}\lVert \nabla\varphi_t \rVert_2^2 = 0$, guaranteeing zero vorticity and non-crossing trajectories.
>
> **3. Conditional-Flow Upper Bound vs. Non-Optimality of Diffusion PF-ODEs.** For any coupling $\pi \in \Pi(p_{\mathrm{src}}, p_{\mathrm{tgt}})$ and conditional bridge $\Psi_t(\boldsymbol{x}_0, \boldsymbol{x}_1)$ with $\Psi_0 = \boldsymbol{x}_0, \Psi_1 = \boldsymbol{x}_1$, Jensen's inequality bounds Eq. `(eq:ot:benamou_brenier)` from above; uniquely minimizing the conditional energy yields straight interpolants $\Psi_t^\star(\boldsymbol{x}_0, \boldsymbol{x}_1) = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Prop. 7.5.1, p. 221):
>
> $$
> W_{2}(p_{\mathrm{src}}, p_{\mathrm{tgt}})^2
>   \le \mathbb{E}_{(\boldsymbol{X}_0, \boldsymbol{X}_1)\sim \pi}\!\left[\int_0^1 \lVert \Psi'_t(\boldsymbol{X}_0, \boldsymbol{X}_1) \rVert_2^2 \mathrm{d}t\right]
>   \ge \mathbb{E}_{(\boldsymbol{X}_0, \boldsymbol{X}_1)\sim \pi}\!\left[\lVert \boldsymbol{X}_1 - \boldsymbol{X}_0 \rVert_2^2\right].
> $$
>
> However, equality holds only when $\pi$ itself is optimal and trajectories induce a potential flow: in general, neither variance-preserving diffusion PF-ODEs (violating the Lavenant–Santambrogio commutation condition between $\nabla^2\ln p_0$ and $\nabla^2 u$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Eq. 7.5.1, p. 219)) nor iterative potential rectification $\mathrm{Rectify}^\perp$ (Hertrich et al. counterexamples [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (pp. 222–223)) guarantee recovery of the exact $W_2$ Brenier map.

### Entropic Regularization, Schrödinger Bridges, and Sinkhorn Divergences

Solving the unregularized LP `(eq:ot:discrete_lp)` on $n$ points via network simplex requires $O(n^3 \log n)$ operations, suffers from non-unique optimal faces under small cost perturbations, and exhibits slow $O(n^{-1/d})$ sample convergence in $\mathbb{R}^d$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8.3.5, p. 321) [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slide 28) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (p. 198). Adding an entropic barrier $\varepsilon D_{\mathrm{KL}}(\pi \parallel \mu \otimes \nu)$ ($\varepsilon > 0$) restores strict convexity, smoothness, and $O(n^2)$ GPU-parallelizable scaling [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8.3.6, p. 322).

> [!important] Entropic OT, Primal–Dual Structure, Schrödinger Bridges & Sinkhorn Divergences
> **1. Primal Formulation.** For $\varepsilon > 0$, define the **entropic optimal transport (EOT)** objective [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. 6.213–6.214, p. 322) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Eq. 7.2.5, p. 198):
>
> $$
> \mathrm{EOT}_\varepsilon(\mu, \nu) \triangleq \inf_{\pi \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{X}} c(\boldsymbol{x}, \boldsymbol{y}) \mathrm{d}\pi(\boldsymbol{x}, \boldsymbol{y}) + \varepsilon D_{\mathrm{KL}}(\pi \parallel \mu \otimes \nu).
> $$
>
> On discrete measures `(eq:ot:discrete_measures)`, $D_{\mathrm{KL}}(\boldsymbol{\pi} \parallel \boldsymbol{a}\boldsymbol{b}^{\top}) = -H(\boldsymbol{\pi}) + H(\boldsymbol{a}) + H(\boldsymbol{b})$ where $H(\boldsymbol{\pi}) = -\sum_{i,j} \pi_{ij}\ln \pi_{ij}$ (with $0\ln 0 \triangleq 0$), matching Ho's entropy-penalized form $\min_{\boldsymbol{\pi} \in \Pi(\boldsymbol{a},\boldsymbol{b})} \langle \boldsymbol{C}, \boldsymbol{\pi} \rangle - \varepsilon H(\boldsymbol{\pi})$ up to the constant $\varepsilon(H(\boldsymbol{a}) + H(\boldsymbol{b}))$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Eq. (6), Slide 30). Because $-H(\boldsymbol{\pi})$ is $1$-strongly convex in the $L^1$ norm on $\Pi(\boldsymbol{a},\boldsymbol{b})$ (by Pinsker's inequality), $\mathrm{EOT}_\varepsilon(\boldsymbol{a},\boldsymbol{b})$ is $\varepsilon$-strongly convex and admits a *unique* optimizer $\boldsymbol{\pi}_\varepsilon^\star$.
>
> **2. Unconstrained Dual & Gibbs Kernel Factorization.** Forming the Lagrangian of discrete Eq. `(eq:ot:eot_primal)` with multipliers $(\tilde{\boldsymbol{f}}, \tilde{\boldsymbol{g}}) \in \mathbb{R}^n \times \mathbb{R}^m$ and applying $\sum_{i,j}\pi_{ij} = \sum_{i,j} a_i b_j = 1$ gives
>
> $$
> \mathcal{L}_\varepsilon(\boldsymbol{\pi}, \tilde{\boldsymbol{f}}, \tilde{\boldsymbol{g}})
>   = \sum_{i,j} C_{ij}\pi_{ij} + \varepsilon \sum_{i,j}\pi_{ij}\!\left(\ln\frac{\pi_{ij}}{a_i b_j} - 1\right) + \varepsilon
>   + \tilde{\boldsymbol{f}}^{\top}(\boldsymbol{a} - \boldsymbol{\pi}\boldsymbol{1}_m) + \tilde{\boldsymbol{g}}^{\top}(\boldsymbol{b} - \boldsymbol{\pi}^{\top}\boldsymbol{1}_n).
> $$
>
> Setting $\partial_{\pi_{ij}}\mathcal{L}_\varepsilon = C_{ij} + \varepsilon\ln\frac{\pi_{ij}}{a_i b_j} - \tilde{f}_i - \tilde{g}_j = 0$ yields the strictly positive Gibbs-scaled coupling
>
> $$
> \pi_{\varepsilon, ij}^\star = a_i b_j \exp\!\left(\frac{\tilde{f}_i^\star + \tilde{g}_j^\star - C_{ij}}{\varepsilon}\right)
>   = \exp\!\left(\frac{f_i^\star}{\varepsilon}\right) K_{ij} \exp\!\left(\frac{g_j^\star}{\varepsilon}\right), \qquad
>   K_{ij} \triangleq \exp\!\left(-\frac{C_{ij}}{\varepsilon}\right),
> $$
>
> where absorbing marginal logs via $f_i \triangleq \tilde{f}_i + \varepsilon\ln a_i$ and $g_j \triangleq \tilde{g}_j + \varepsilon\ln b_j$ recovers Murphy's matrix-scaling form $\boldsymbol{\pi}_\varepsilon^\star = \mathrm{diag}(e^{\boldsymbol{f}^\star/\varepsilon})\boldsymbol{K}\mathrm{diag}(e^{\boldsymbol{g}^\star/\varepsilon})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 6.216, p. 322). Substituting Eq. `(eq:ot:gibbs_factorization)` back into Eq. `(eq:ot:eot_lagrangian)` yields the smooth unconstrained concave dual [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 6.215, p. 322) [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Eq. (7), Slide 33):
>
> $$
> \mathrm{EOT}_\varepsilon(\boldsymbol{a}, \boldsymbol{b})
>   = \max_{\boldsymbol{f} \in \mathbb{R}^n,  \boldsymbol{g} \in \mathbb{R}^m}
>   \boldsymbol{f}^{\top}\boldsymbol{a} + \boldsymbol{g}^{\top}\boldsymbol{b} - \varepsilon \bigl(e^{\boldsymbol{f}/\varepsilon}\bigr)^{\top} \boldsymbol{K} e^{\boldsymbol{g}/\varepsilon} + \varepsilon\bigl(1 + H(\boldsymbol{a}) + H(\boldsymbol{b})\bigr).
> $$
>
> Holding $\boldsymbol{g}$ fixed and setting $\nabla_{\boldsymbol{f}}(\cdot) = \boldsymbol{0}$ (and symmetrically for $\boldsymbol{g}$) yields block-coordinate ascent (**Sinkhorn updates**) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 6.217, p. 322) [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slide 34):
>
> $$
> f_i \leftarrow \varepsilon\ln a_i - \varepsilon\ln\sum_{j=1}^m \exp\!\left(\frac{g_j - C_{ij}}{\varepsilon}\right), \qquad
>   g_j \leftarrow \varepsilon\ln b_j - \varepsilon\ln\sum_{i=1}^n \exp\!\left(\frac{f_i - C_{ij}}{\varepsilon}\right).
> $$
>
> **3. Asymptotic Limits ($\varepsilon \to 0^+$ and $\varepsilon \to \infty$) & Dynamic Schrödinger Bridge Equivalence.**
> - *Zero-temperature limit ($\varepsilon \to 0^+$):* Because $D_{\mathrm{KL}}(\pi \parallel \mu \otimes \nu) \ge 0$, $\mathrm{EOT}_\varepsilon(\mu,\nu) \ge \mathrm{OT}_c(\mu,\nu)$; conversely, comparing against any optimal $\pi^\star$ with finite KL gives $\mathrm{EOT}_\varepsilon(\mu,\nu) \le \mathrm{OT}_c(\mu,\nu) + \varepsilon D_{\mathrm{KL}}(\pi^\star \parallel \mu\otimes\nu)$, establishing $\lim_{\varepsilon\to 0^+}\mathrm{EOT}_\varepsilon(\mu,\nu) = \mathrm{OT}_c(\mu,\nu)$ and weak convergence $\pi_\varepsilon^\star \rightharpoonup \arg\max_{\pi \in \Pi^\star(\mu,\nu)} H(\pi)$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Thm. 2(a), Slide 32) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Thm. 7.3.1, pp. 209–210).
> - *High-temperature limit ($\varepsilon \to \infty$):* As $\varepsilon \to \infty$, the KL penalty dominates, forcing $\pi_\varepsilon^\star \to \mu \otimes \nu$ and $\mathrm{EOT}_\varepsilon(\mu,\nu) \to \int c \mathrm{d}(\mu \otimes \nu)$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Thm. 2(b), Slide 32).
> - *Schrödinger Bridge (SB) duality:* Minimizing path-space KL $\mathrm{SB}_\varepsilon(\mu, \nu) \triangleq \min_{P : P_0 = \mu,  P_T = \nu} D_{\mathrm{KL}}(P \parallel R^\varepsilon)$ against a Brownian reference $R^\varepsilon$ ($\mathrm{d}\boldsymbol{x}_t = \sqrt{\varepsilon} \mathrm{d}\boldsymbol{w}_t$, transition kernel $-\varepsilon\ln p_T(\boldsymbol{y}|\boldsymbol{x}) = \frac{\lVert \boldsymbol{y}-\boldsymbol{x} \rVert_2^2}{2T} + \text{const}$) is affinely equivalent to $\mathrm{EOT}_\varepsilon(\mu,\nu)$ for $c(\boldsymbol{x},\boldsymbol{y}) = \frac{1}{2T}\lVert \boldsymbol{x}-\boldsymbol{y} \rVert_2^2$, with endpoint law $(X_0, X_T) \sim \pi_\varepsilon^\star$ and controlled drift $\boldsymbol{v}_t^\star(\boldsymbol{x}) = \varepsilon \nabla \ln \Psi(\boldsymbol{x}, t)$ solving the Benamou–Brenier problem via $\Gamma$-convergence as $\varepsilon \to 0^+$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 7.2.3 & 7.3, Eqs. 7.2.9–7.3.1, Thm. 7.3.2, pp. 202–211).
>
> **4. Debiased Sinkhorn Divergence.** Because entropic smoothing diffuses self-transport mass away from the diagonal $\boldsymbol{x}=\boldsymbol{y}$, $\mathrm{EOT}_\varepsilon(\mu, \mu) > 0$ whenever $\mu$ is non-atomic (in fact $\arg\min_\mu \mathrm{EOT}_\varepsilon(\mu, \nu)$ collapses $\mu$ toward a shrunk mean of $\nu$). Subtracting the symmetric self-transport biases defines the **Sinkhorn divergence** (Genevay et al., 2018; cited in [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (p. 205)):
>
> $$
> S_\varepsilon(\mu, \nu) \triangleq \mathrm{EOT}_\varepsilon(\mu, \nu) - \frac{1}{2}\mathrm{EOT}_\varepsilon(\mu, \mu) - \frac{1}{2}\mathrm{EOT}_\varepsilon(\nu, \nu).
> $$
>
> Debiasing restores $S_\varepsilon(\mu, \mu) = 0$, $S_\varepsilon(\mu, \nu) \ge 0$, and convexity/metrization properties for positive-definite Gibbs kernels $K = e^{-C/\varepsilon}$, while smoothly interpolating between $\mathrm{OT}_c(\mu, \nu)$ as $\varepsilon \to 0^+$ and squared MMD ($\frac{1}{2}\lVert \mu - \nu \rVert_{-c}^2$ for quadratic $c(\boldsymbol{x},\boldsymbol{y}) = \lVert \boldsymbol{x}-\boldsymbol{y} \rVert_2^2$, equal to $\lVert \mathbb{E}_\mu[X] - \mathbb{E}_\nu[Y] \rVert_2^2$) as $\varepsilon \to \infty$, achieving dimension-free $O(n^{-1/2}\varepsilon^{-d/2})$ empirical sample complexity [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8.3.6, p. 322).

## Algorithmic Implementation & Comparative Synthesis

Direct multiplicative Sinkhorn updates $\boldsymbol{u} \leftarrow \boldsymbol{a} \oslash (\boldsymbol{K}\boldsymbol{v})$ and $\boldsymbol{v} \leftarrow \boldsymbol{b} \oslash (\boldsymbol{K}^{\top}\boldsymbol{u})$ suffer catastrophic floating-point underflow in $\boldsymbol{K} = e^{-\boldsymbol{C}/\varepsilon}$ whenever $\varepsilon \ll \lVert \boldsymbol{C} \rVert_\infty$. Algorithm `alg:ot:sinkhorn` executes the dual updates `(eq:ot:sinkhorn_updates)` [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 6.217, p. 322) strictly in log-space via max-shifted $\mathrm{LogSumExp}$, followed by Altschuler–Weed–Rigollet (2017) feasibility rounding [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slides 34–37) to project $\tilde{\boldsymbol{\pi}}$ onto $\Pi(\boldsymbol{a}, \boldsymbol{b})$ when an exact plan is needed.

> [!example] Algorithm: Log-Domain Stabilized Sinkhorn–Knopp Scaling with Feasibility Rounding & Debiasing
> **Input:** Marginals $\boldsymbol{a} \in \Delta_n$, $\boldsymbol{b} \in \Delta_m$ ($a_i, b_j > 0$); cost $\boldsymbol{C} \in \mathbb{R}_{\ge 0}^{n \times m}$; regularization $\varepsilon > 0$; tolerance $\delta > 0$; max iterations $T_{\max}$.
> **Output:** Rounded plan $\bar{\boldsymbol{\pi}} \in \Pi(\boldsymbol{a}, \boldsymbol{b})$, entropic value $\widehat{\mathrm{EOT}}_\varepsilon(\boldsymbol{a},\boldsymbol{b})$, and (optionally) Sinkhorn divergence $\widehat{S}_\varepsilon(\boldsymbol{a},\boldsymbol{b})$.
> Define $\mathrm{LSE}_j(z_{ij}) \triangleq m_i + \ln\sum_{j=1}^m \exp(z_{ij} - m_i)$ where $m_i \triangleq \max_j z_{ij}$
> Initialize log-potentials $\boldsymbol{f}^{(0)} \leftarrow \boldsymbol{0}_n$, $\boldsymbol{g}^{(0)} \leftarrow \boldsymbol{0}_m$, $t \leftarrow 0$
> **while** $t < T_{\max}$:
>
>   **for** $i = 1, \dots, n$:
>
>     $f_i^{(t+1)} \leftarrow \varepsilon \ln a_i - \varepsilon \mathrm{LSE}_{j=1}^m\!\left(\dfrac{g_j^{(t)} - C_{ij}}{\varepsilon}\right)$  *// Row marginal projection*
>
>   **for** $j = 1, \dots, m$:
>
>     $g_j^{(t+1)} \leftarrow \varepsilon \ln b_j - \varepsilon \mathrm{LSE}_{i=1}^n\!\left(\dfrac{f_i^{(t+1)} - C_{ij}}{\varepsilon}\right)$  *// Column marginal projection*
>
>   Compute primal matrix $\tilde{\pi}_{ij}^{(t+1)} \leftarrow \exp\!\left(\dfrac{f_i^{(t+1)} + g_j^{(t+1)} - C_{ij}}{\varepsilon}\right)$
>   **if** $\lVert \tilde{\boldsymbol{\pi}}^{(t+1)}\boldsymbol{1}_m - \boldsymbol{a} \rVert_1 + \lVert (\tilde{\boldsymbol{\pi}}^{(t+1)})^{\top}\boldsymbol{1}_n - \boldsymbol{b} \rVert_1 \le \delta$:
>
>     **break**
>
>   $t \leftarrow t + 1$
>
>  *// Altschuler–Weed–Rigollet Feasibility Rounding onto $\Pi(\boldsymbol{a}, \boldsymbol{b})$*
> Set $\boldsymbol{r} \leftarrow \tilde{\boldsymbol{\pi}}\boldsymbol{1}_m$, $x_i \leftarrow \min\{a_i / \max(r_i, 10^{-30}),  1\}$, and $\boldsymbol{\pi}' \leftarrow \mathrm{diag}(\boldsymbol{x})\tilde{\boldsymbol{\pi}}$
> Set $\boldsymbol{c} \leftarrow (\boldsymbol{\pi}')^{\top}\boldsymbol{1}_n$, $y_j \leftarrow \min\{b_j / \max(c_j, 10^{-30}),  1\}$, and $\boldsymbol{\pi}" \leftarrow \boldsymbol{\pi}'\mathrm{diag}(\boldsymbol{y})$
> Compute deficits $\boldsymbol{e}_r \leftarrow \boldsymbol{a} - \boldsymbol{\pi}"\boldsymbol{1}_m \ge \boldsymbol{0}$, $\boldsymbol{e}_c \leftarrow \boldsymbol{b} - (\boldsymbol{\pi}")^{\top}\boldsymbol{1}_n \ge \boldsymbol{0}$, and $\bar{\boldsymbol{\pi}} \leftarrow \boldsymbol{\pi}" + \frac{\boldsymbol{e}_r \boldsymbol{e}_c^{\top}}{\max(\lVert \boldsymbol{e}_r \rVert_1,  10^{-30})}$
> Evaluate $\widehat{\mathrm{EOT}}_\varepsilon(\boldsymbol{a},\boldsymbol{b}) \leftarrow \langle \boldsymbol{C}, \bar{\boldsymbol{\pi}} \rangle + \varepsilon D_{\mathrm{KL}}(\bar{\boldsymbol{\pi}} \parallel \boldsymbol{a}\boldsymbol{b}^{\top})$ (or dual Eq. `(eq:ot:eot_dual)` at convergence)
>  *// Optional Sinkhorn divergence via two symmetric self-calls*
> Run symmetric updates on $(\boldsymbol{a}, \boldsymbol{a}, \boldsymbol{C}_{\mu\mu})$ and $(\boldsymbol{b}, \boldsymbol{b}, \boldsymbol{C}_{\nu\nu})$ to obtain $\widehat{\mathrm{EOT}}_\varepsilon(\boldsymbol{a},\boldsymbol{a})$, $\widehat{\mathrm{EOT}}_\varepsilon(\boldsymbol{b},\boldsymbol{b})$
> $\widehat{S}_\varepsilon(\boldsymbol{a}, \boldsymbol{b}) \leftarrow \widehat{\mathrm{EOT}}_\varepsilon(\boldsymbol{a},\boldsymbol{b}) - \frac{1}{2}\widehat{\mathrm{EOT}}_\varepsilon(\boldsymbol{a},\boldsymbol{a}) - \frac{1}{2}\widehat{\mathrm{EOT}}_\varepsilon(\boldsymbol{b},\boldsymbol{b})$
> **return** $\bar{\boldsymbol{\pi}}$, $\widehat{\mathrm{EOT}}_\varepsilon(\boldsymbol{a},\boldsymbol{b})$, $\widehat{S}_\varepsilon(\boldsymbol{a},\boldsymbol{b})$

Under the regularization choice $\varepsilon = \frac{\epsilon}{4\ln(\max\{n,m\})}$, Altschuler, Weed, and Rigollet (2017) guarantee primal error $\langle \boldsymbol{C}, \bar{\boldsymbol{\pi}} \rangle \le \mathrm{OT}_{\boldsymbol{C}}(\boldsymbol{a},\boldsymbol{b}) + \epsilon$ within $t = O\bigl(\lVert \boldsymbol{C} \rVert_\infty^2 \ln(\max\{n,m\}) / \epsilon^2\bigr)$ iterations, costing $O(nm)$ arithmetic operations per step [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Thm. 3, Slides 37–38). Table `tab:ot:complexity` contrasts classical exact solvers, entropic scaling, minibatch approximations, and sliced projections.

**Table:** Computational complexity, statistical sample rates ($N$ i.i.d. samples in $\mathbb{R}^d$), and structural trade-offs across optimal transport solvers [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slides 18, 28, 37–39, 62, 70, 84) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8.3.5–6.8.3.6, pp. 321–322).

| **Solver / Variant** | **Time Complexity** | **Sample Rate** | **Characteristics & Failure Regimes** |
| --- | --- | --- | --- |
| 1D Quantile Sort ($d=1$) | $O(n \log n)$ | $O(N^{-1/2})$ | Exact $W_p$ and Monge map $Q_\nu \circ F_\mu$; limited to univariate measures. |
| Hungarian Assignment | $O(n^3)$ | $O(N^{-1/d})$ | Exact bijection; restricted to $n = m$ with uniform weights. |
| Network Simplex LP | $O((n+m)nm\ln(n+m)\cdot$   $\ln((n+m)\lVert \boldsymbol{C} \rVert_\infty))$ | $O(N^{-1/d})$ | Exact vertex of $\Pi(\boldsymbol{a},\boldsymbol{b})$ ($O(n^3\ln n)$ for $n=m$); non-parallelizable and non-differentiable. |
| Sinkhorn–Knopp + Round ($\epsilon$-approx) | $O(\max\{n,m\}^2 \lVert \boldsymbol{C} \rVert_\infty^2 \cdot$   $\ln(\max\{n,m\}) / \epsilon^2)$ | $O(N^{-1/2}\varepsilon^{-d/2})$ | $O(nm)$ per step; GPU-parallel, smooth, and auto-differentiable; degrades as $\varepsilon \to 0$. |
| Greenkhorn / Accel. Sinkhorn | $O(nm)$ per greedy/mirror step | $O(N^{-1/2}\varepsilon^{-d/2})$ | Faster empirical convergence than cyclic Sinkhorn on sparse/skewed plans. |
| Minibatch OT (`m-OT`) / Partial (`m-POT`) | $O(k m^2)$ ($k$ batches of size $m \ll n$) | Biased wrt full OT | Scales to millions of points; `m-OT` induces cross-mode errors mitigated by `m-POT` / `BoMb-OT`. |
| Sliced Wasserstein $\mathrm{SW}_p$ | $O(L n \log n)$ ($L$ projections on $\mathbb{S}^{d-1}$) | $O(N^{-1/2})$ | Dimension-free statistical rate and $O(n\log n)$ speed; vectorization loses 2D spatial structure unless convolutional. |

> [!info] Cross-Book Synthesis: Static Geometry, Computational Scaling & Generative Bridges
> Uniting the three source perspectives reveals how static transport geometry, scalable matrix scaling, and continuous-time generative dynamics interlock:
> - **Duality & Implicit Generative Critics:** Murphy and Cuturi formulate $W_1$ as a $1$-Lipschitz IPM and $W_2$ through Brenier's convex potential $T^\star = \nabla u$, motivating Wasserstein GANs ($1$-Lipschitz critic regularization) and Input-Convex Neural Networks (ICNNs) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8.3.2–6.8.3.3, 26.2.4, pp. 319–320, 900). Ho emphasizes why WGANs avoid Jensen–Shannon gradient vanishing on disjoint manifolds, while exposing the $O(n^3 \ln n)$ combinatorial barrier of exact LP solvers and the curse of dimensionality ($O(n^{-1/d})$) [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slides 28, 46–58, 82).
> - **Minibatch & Sliced Approximations:** Rather than computing global $n \times n$ couplings or unreliable neural duals for general costs, Ho develops $O(k m^2)$ minibatch OT (`m-OT`), partial/hierarchical fixes (`m-POT`, `BoMb-OT`) to suppress wrong cross-mode matchings, and $O(L n \log n)$ Sliced and Convolution-Sliced Wasserstein metrics $\mathrm{SW}_p(\mu, \nu) = \bigl(\int_{\mathbb{S}^{d-1}} W_{p}(\theta_\sharp\mu, \theta_\sharp\nu)^p \mathrm{d}\theta\bigr)^{1/p}$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slides 60–91).
> - **Diffusion Models, Flow Matching & Schrödinger Bridges:** The Diffusion Models monograph establishes that $\mathrm{EOT}_\varepsilon$ is the static marginal projection of a Brownian Schrödinger Bridge $\mathrm{SB}_\varepsilon$, which collapses via $\Gamma$-convergence to Benamou–Brenier dynamic OT as $\varepsilon \to 0^+$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 7.3, Fig. 7.3, pp. 206–211). Standard score-based diffusion models correspond to the zero-control limit ($\Psi \equiv 1$) solving only a one-sided Schrödinger half-bridge ($p_T \approx p_{\mathrm{prior}}$ only as $T \to \infty$), whereas Flow Matching straight-line paths optimize a conditional upper bound on kinetic energy without guaranteeing global $W_2$ optimality [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 7.4–7.5, pp. 212–223).

## Visual Diagnostics & Geometric Structure

Figure `fig:ot:murphy_monge_kantorovich` contrasts the three canonical regimes of mass transport: one-to-one permutation matching on equal unweighted sets, mass-splitting Kantorovich couplings on unequal weighted discrete measures, and continuous pushforward maps [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 6.18, p. 315).

> [!note] Source Figure
> *Figure:* Unified taxonomy of static optimal transport formulations. **Left:** Optimal bijection $\sigma^\star \in S_n$ minimizing $\sum_i \lVert \boldsymbol{x}_i - \boldsymbol{y}_{\sigma(i)} \rVert$ over $n=5$ unweighted points (versus a suboptimal permutation $\sigma$). **Middle:** Kantorovich mass-splitting matrix $\boldsymbol{P}^\star \in \Pi(\boldsymbol{a}, \boldsymbol{b})$ coupling weighted Dirac clouds of unequal cardinalities ($n=6$, $m=4$). **Right:** Continuous Monge pushforward map $T^\star_\sharp \mu = \nu$ associating each $\boldsymbol{x} \in \mathrm{supp}(\mu)$ to a unique destination $T^\star(\boldsymbol{x})$. Reproduced from [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 6, Fig. 6.18, p. 315). (`probabilisticml_kevinmurphy_advancedtopics_p349_fig1.png`)

Figure `fig:ot:geometry_birkhoff_dynamics` illustrates the convex geometry of entropic regularization inside the transportation polytope alongside the dynamic transition from deterministic Benamou–Brenier geodesics ($\varepsilon = 0$) to controlled Schrödinger bridges ($\varepsilon > 0$).

> [!example] Visual Diagram (LaTeX / PDF)
> *Geometric structure of static and dynamic optimal transport. **(a)** Inside the transportation polytope $\Pi(\boldsymbol{a},\boldsymbol{b})$ (a scaled Birkhoff polytope for uniform weights), unregularized LP solutions sit at sparse permutation vertices $\boldsymbol{\pi}_0^\star$, whereas entropic regularization $-\varepsilon H(\boldsymbol{\pi})$ curves the unique optimum $\boldsymbol{\pi}_\varepsilon^\star$ smoothly through the interior toward the product measure $\boldsymbol{a}\boldsymbol{b}^{\top}$ as $\varepsilon \to \infty$. **(b)** In $\mathbb{R}^d$, $W_2$ transport evolves along non-crossing straight-line McCann geodesics driven by potential gradients $\boldsymbol{v}_t = \nabla\varphi_t$, whereas $\mathrm{SB}_\varepsilon$ adds Brownian noise $\sqrt{\varepsilon} \mathrm{d}\boldsymbol{w}_t$ whose endpoint law reproduces $\boldsymbol{\pi}_\varepsilon^\star$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:ot:pgf_divergences_sinkhorn` provides an exact quantitative verification of both core motivations for optimal transport and Sinkhorn debiasing. In Panel (a), comparing $\mu = \delta_0$ and $\nu_\theta = \delta_\theta$ across shift $\theta \in [-1.4, 1.4]$ confirms that $D_{\mathrm{TV}}$ and $D_{\mathrm{JS}}(\mu \parallel \nu_\theta)$ jump discontinuously at $\theta = 0$ and remain flat everywhere $\theta \ne 0$, whereas $W_{1}(\mu, \nu_\theta) = \lvert \theta \rvert$ and $W_{2}(\mu, \nu_\theta)^2 = \theta^2$ vary smoothly with non-vanishing gradients. In Panel (b), evaluating the closed-form two-point problem $\mu = \frac{1}{2}\delta_0 + \frac{1}{2}\delta_2$ and $\nu = \frac{1}{2}\delta_0 + \frac{1}{2}\delta_1$ under $c(x,y) = (x-y)^2$ yields cost matrix $\boldsymbol{C} = \bigl[\begin{smallmatrix} 0 & 1 \\ 4 & 1 \end{smallmatrix}\bigr]$ (with $\mathrm{OT}(\mu,\nu) = \frac{1}{2}$, mean cross-cost $\langle \boldsymbol{C}, \boldsymbol{a}\boldsymbol{b}^{\top} \rangle = \frac{3}{2}$, and $\boldsymbol{\pi}_\varepsilon^\star = \frac{1}{2}\bigl[\begin{smallmatrix} 1-p_\varepsilon & p_\varepsilon \\ p_\varepsilon & 1-p_\varepsilon \end{smallmatrix}\bigr]$ where $p_\varepsilon = (1 + e^{2/\varepsilon})^{-1}$), yielding exact closed-form curves:

$$
\mathrm{EOT}_\varepsilon(\mu,\nu) = \frac{1}{2} + \varepsilon \ln\!\left(\frac{2}{1 + e^{-2/\varepsilon}}\right), \qquad
  S_\varepsilon(\mu,\nu) = \mathrm{EOT}_\varepsilon(\mu,\nu) - \frac{\varepsilon}{2}\ln\!\left(\frac{4}{(1 + e^{-4/\varepsilon})(1 + e^{-1/\varepsilon})}\right).
$$

As $\varepsilon \to 0^+$, $\mathrm{EOT}_\varepsilon(\mu,\nu)$ deviates upward with slope $\ln 2$ while $S_\varepsilon(\mu,\nu)$ cancels this self-bias to track $\mathrm{OT}(\mu,\nu) = 0.5$ to high accuracy; as $\varepsilon \to \infty$, $\mathrm{EOT}_\varepsilon$ saturates at the independent coupling cost $1.5$ whereas $S_\varepsilon(\mu,\nu) \to (\mathbb{E}_\mu[X] - \mathbb{E}_\nu[Y])^2 = 0.25$.

> [!example] Visual Diagram (LaTeX / PDF)
> *Analytical behavior of Wasserstein distances and Sinkhorn regularization. **(a)** On disjoint Diracs $\delta_0, \delta_\theta$, $D_{\mathrm{TV}}$ and $D_{\mathrm{JS}}$ experience a jump discontinuity at $\theta=0$ with zero derivative elsewhere, whereas $W_1$ and $W_2^2$ remain continuous with informative gradients [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slides 46–49). **(b)** On $\mu = \frac{1}{2}\delta_0 + \frac{1}{2}\delta_2$ and $\nu = \frac{1}{2}\delta_0 + \frac{1}{2}\delta_1$ ($c(x,y) = (x-y)^2$), un-debiased $\mathrm{EOT}_\varepsilon(\mu,\nu)$ incurs linear self-entropy bias near $\varepsilon=0$ and blurs toward $\langle \boldsymbol{C}, \boldsymbol{a}\boldsymbol{b}^{\top} \rangle=1.5$ as $\varepsilon\to\infty$, while debiased $S_\varepsilon(\mu,\nu)$ recovers $\mathrm{OT}(\mu,\nu)=0.5$ as $\varepsilon \to 0^+$ and $\mathrm{MMD}^2=0.25$ as $\varepsilon \to \infty$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Complementing the static curves above, Figures `fig:ot:dynamic_mccann` and `fig:ot:minibatch_plans` showcase continuous-time Benamou–Brenier displacement interpolation and empirical coupling matrices in large-scale discrete transport.

> [!note] Source Figure
> *Figure:* Continuous-time Benamou–Brenier dynamic optimal transport and McCann displacement interpolation $p_t^{\mathrm{OT}} = (T_t^\star)_\sharp p_{\mathrm{src}}$ ($t \in [0, 1]$). Mass flows deterministically along non-crossing straight-line characteristics minimizing kinetic energy $\int_0^1 \mathbb{E}_{p_t}[\lVert \boldsymbol{v}_t \rVert_2^2] \mathrm{d}t$. Reproduced from [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Ch. 7, Fig. 7.1, p. 196). (`theprinciple_diffusion_models_p201_fig1.png`)

> [!note] Source Figure
> *Figure:* Empirical transportation plans between two 10-atom Gaussian mixtures ($n=10$). **Left:** Minibatch OT (`m-OT`, batch size $m=6$, $k=32$) creates spurious cross-mode couplings (off-block diagonals in heatmap) because unmatched batches force high-cost pairings. **Middle:** Minibatch Partial OT (`m-POT`, transport fraction $s=0.6$) prunes forced cross-cluster pairings. **Right:** Exact `full-OT` recovers the clean block-sparse permutation coupling. Reproduced from [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slide 73). (`optimal_transport_p73_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Small-$\varepsilon$ Kernel Underflow vs. Large-$\varepsilon$ Entropic Blurring Bias
> Evaluating the Gibbs kernel $\boldsymbol{K} = \exp(-\boldsymbol{C}/\varepsilon)$ directly in multiplicative Sinkhorn updates underflows to exact floating-point zero whenever $\min_{i,j} C_{ij} / \varepsilon \gtrsim 88$ (IEEE-754 float32) or $\gtrsim 709$ (float64), yielding $0 / 0 = \mathrm{NaN}$ in $\boldsymbol{a} \oslash (\boldsymbol{K}\boldsymbol{v})$. Conversely, choosing overly large $\varepsilon$ induces severe **entropic blurring**, pushing $\boldsymbol{\pi}_\varepsilon^\star \to \boldsymbol{a}\boldsymbol{b}^{\top}$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Thm. 2(b), Slide 32), while taking $\varepsilon \to 0^+$ requires $O(\lVert \boldsymbol{C} \rVert_\infty^2 \ln n / \epsilon^2)$ iterations as the Hilbert projective metric contraction factor approaches $1$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slide 37) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8.3.6, p. 322).
> **Remedy:** Always execute updates in log-space using max-shifted $\mathrm{LogSumExp}$ (Algorithm `alg:ot:sinkhorn`), rescale $\boldsymbol{C}$ by $\mathrm{median}(\boldsymbol{C})$ or $\lVert \boldsymbol{C} \rVert_\infty$ prior to selecting $\varepsilon$, and apply $\varepsilon$-scaling (annealing $\varepsilon$ geometrically from $\varepsilon_0 \sim \mathrm{median}(\boldsymbol{C})$ down to the target $\varepsilon$ while warm-starting $(\boldsymbol{f}, \boldsymbol{g})$).

> [!warning] Un-Debiased Self-Transport Bias, Mode Shrinking & Premature Stopping Asymmetry
> Although $\mathrm{EOT}_\varepsilon(\mu, \nu)$ is symmetric for symmetric costs, $\mathrm{EOT}_\varepsilon(\mu, \mu) > 0$ for every non-Dirac measure because entropic regularization spreads self-coupling mass onto off-diagonal neighbors (Figure `fig:ot:pgf_divergences_sinkhorn`(b)). Consequently, minimizing un-debiased $\mu \mapsto \mathrm{EOT}_\varepsilon(\mu, \nu)$ biases $\mu$ toward a collapsed Dirac spike at the barycenter of $\nu$ in order to drive its internal self-entropy cost down. Furthermore, halting Sinkhorn after finite $t$ steps leaves the primal iterate $\tilde{\boldsymbol{\pi}}^{(t)} \notin \Pi(\boldsymbol{a}, \boldsymbol{b})$ violating one marginal constraint [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slide 36), which breaks symmetry and biases gradients when differentiating through unrolled steps.
> **Remedy:** Use the debiased Sinkhorn divergence $S_\varepsilon(\mu, \nu)$ (Eq. `(eq:ot:sinkhorn_divergence)`) as any generative or barycentric loss, and apply the Altschuler–Weed–Rigollet feasibility rounding operator (Algorithm `alg:ot:sinkhorn`) whenever an exact marginal-preserving plan $\bar{\boldsymbol{\pi}} \in \Pi(\boldsymbol{a}, \boldsymbol{b})$ is required.

> [!warning] Discrete Support Mismatch, Unbalanced Marginals & Minibatch Misspecification
> Several structural failure modes arise when applying idealized OT formulations directly to raw empirical datasets:
> - *Monge Non-Existence & Unbalanced Mass:* Discrete measures with incompatible atom weights cannot satisfy $T_\sharp \mu = \nu$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slide 22), and Kantorovich LP constraints fail immediately if $\sum_i a_i \ne \sum_j b_j$ or if outliers distort marginal budgets.
> - *Curse of Dimensionality:* Unregularized empirical Wasserstein estimation $W_{p}(\hat{\mu}_n, \mu)$ converges only at rate $O(n^{-1/d})$ for $d > 2p$, so exact LP solves on high-dimensional point clouds overfit sample noise [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8.3.5, p. 321) [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slide 82).
> - *Minibatch Misspecified Matchings:* Evaluating OT independently on random $m$-sample minibatches (`m-OT`) forces pairings between far-apart modes whenever batch realizations imbalance cluster proportions (Figure `fig:ot:minibatch_plans`, left) [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slides 65–67).
>
> **Remedy:** Replace hard marginal constraints with KL-penalized Unbalanced OT or Partial OT (`m-POT` / hierarchical `BoMb-OT`, [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slides 68–79)) to reject cross-mode outliers, and adopt Sliced Wasserstein $\mathrm{SW}_p$ (or convolutional $\mathrm{SW}_p$ on images) to restore $O(n^{-1/2})$ sample efficiency at $O(L n \log n)$ cost [[Source - Optimal Transport in Large-Scale|optimal_transport]] (Slides 83–89).

> [!warning] Lipschitz Enforcement Fragility & Flow Non-Optimality in Deep Generative Models
> Approximating the $W_1$ Kantorovich–Rubinstein dual `(eq:ot:kr_duality)` with neural critics $D_\phi$ via naive parameter clipping $[-c, c]$ induces capacity collapse or gradient explosion, whereas gradient penalties enforce $\lVert \nabla D_\phi \rVert_2 \approx 1$ only along sampled line segments rather than globally on $\mathbb{R}^d$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 26.2.4, p. 900). Similarly, training continuous-time probability-flow ODEs or iteratively applying Rectified Flow ($\mathrm{Rectify}^\perp$) does not generally yield the global $W_2$ Brenier map $\nabla u$, failing under disconnected intermediate supports or non-commuting density Hessians [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 7.5, pp. 216–223).
> **Remedy:** Pair spectral normalization with two-sided gradient penalties for $W_1$ critics, parametrize $W_2$ potentials directly via Input-Convex Neural Networks (ICNNs, [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.8.3.3, p. 320)), or pair conditional Flow Matching with minibatch OT couplings rather than assuming uncontrolled Reflow converges to optimal transport.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[Kantorovich Duality]], [[Sinkhorn Algorithm]], [[Wasserstein Distance]]
- **Prerequisites (Directed $\leftarrow$):** [[KL Divergence]], [[Lagrange Duality & KKT]], [[Wasserstein Distance]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Flow Matching & Rectified Flow]], [[Sinkhorn Algorithm]], [[Wasserstein GAN (WGAN)]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Bregman & Legendre Duality]], [[Flow Matching & Rectified Flow]], [[Lagrange Duality & KKT]], [[Sinkhorn Algorithm]], [[Wasserstein Distance]], [[Wasserstein GAN (WGAN)]]
- **Synthesized Sources:** [[Source - Optimal Transport in Large-Scale]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]]
