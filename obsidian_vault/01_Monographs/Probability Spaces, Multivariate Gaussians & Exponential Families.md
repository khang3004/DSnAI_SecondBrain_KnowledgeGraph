---
type: monograph
domain: "D01"
parent_topic: "[[PT 01 - Probability Theory, Information Theory & Optimal Transport]]"
sub_topic: "[[ST 01 - Probability Spaces, Gaussians & Statistical Estimation]]"
cheatsheet: "[[Cheatsheet - Probability Spaces, Multivariate Gaussians & Exponential Families]]"
part_of: 
  - "[[ST 01 - Probability Spaces, Gaussians & Statistical Estimation]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
concepts: 
  - "[[Exponential Families]]"
  - "[[Mahalanobis Distance]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
prerequisites: 
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Inner Product & Norms]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
enables: 
  - "[[Exponential Families]]"
  - "[[Gaussian Mixture Models (GMM)]]"
  - "[[Gaussian Processes]]"
  - "[[Generalized Linear Models (GLM)]]"
  - "[[Kalman Filter]]"
  - "[[Mahalanobis Distance]]"
related: 
  - "[[Exponential Families]]"
  - "[[Gaussian Mixture Models (GMM)]]"
  - "[[Gaussian Processes]]"
  - "[[Generalized Linear Models (GLM)]]"
  - "[[Graphical Models (BN & MRF)]]"
  - "[[Inner Product & Norms]]"
  - "[[Kalman Filter]]"
  - "[[Mahalanobis Distance]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
  - "[[Variational Inference & ELBO]]"
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - monograph
  - domain-01
---

# Probability Spaces, Multivariate Gaussians & Exponential Families

> [!abstract] Monograph Navigation
> - **Domain:** [[D01 - Mathematical Foundations for Machine Intelligence|Mathematical Foundations for Machine Intelligence]]
> - **Parent / Sub-Topic:** [[PT 01 - Probability Theory, Information Theory & Optimal Transport|Probability Theory, Information Theory & Optimal Transport]] → [[ST 01 - Probability Spaces, Gaussians & Statistical Estimation|Probability Spaces, Gaussians & Statistical Estimation]]
> - **Visual Cheatsheet:** [[Cheatsheet - Probability Spaces, Multivariate Gaussians & Exponential Families]]
> - **Description:** Bridges Kolmogorov axioms, change-of-variables Jacobian densities, conditioning/marginalization of multivariate Gaussians (Schur complement), and canonical exponential-family sufficient statistics.

## Problem Formulation & Unified Notation

Machine learning models reason under uncertainty by placing probability measures over structured state and parameter spaces. Across probability theory, Bayesian inference, and deep generative modeling, three tightly coupled pillars govern tractable probabilistic computation: Kolmogorov probability spaces $(\Omega, \mathcal{F}, P)$ and Bayes' rule, the multivariate Gaussian (MVN) with its dual moment–precision geometry, and the canonical exponential family whose convex log-partition function generates moments, Fisher curvature, and closed-form conjugate updates.

Let $(\Omega, \mathcal{F}, P)$ denote a probability space comprising a sample space $\Omega$, an event $\sigma$-algebra $\mathcal{F} \subseteq 2^{\Omega}$ closed under complements and countable unions (specifically the Borel $\sigma$-field $\mathcal{B}(\mathbb{R}^D)$ on $\Omega = \mathbb{R}^D$), and a measure $P : \mathcal{F} \to [0,1]$ satisfying Kolmogorov's probability axioms [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 2.2, pp. 58–76):

$$
P(A) \ge 0   \forall A \in \mathcal{F}, \qquad
  P(\Omega) = 1, \qquad
  P\!\left(\bigsqcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i) \quad \text{for pairwise disjoint } A_i \in \mathcal{F},
$$

as formalized measure-theoretically in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 2.1, pp. 1–7). A random vector $\boldsymbol{X} : \Omega \to \mathcal{X} \subseteq \mathbb{R}^D$ is an $(\mathcal{F}, \mathcal{B}(\mathcal{X}))$-measurable map inducing a pushforward law $P_{\boldsymbol{X}}(B) = P(\boldsymbol{X}^{-1}(B))$. When $P_{\boldsymbol{X}}$ is absolutely continuous with respect to a reference measure $\nu$ (Lebesgue measure $\mathrm{d}\boldsymbol{x}$ on $\mathbb{R}^D$ or counting measure on discrete domains), the Radon–Nikodym derivative $p(\boldsymbol{x}) = \frac{\mathrm{d}P_{\boldsymbol{X}}}{\mathrm{d}\nu}(\boldsymbol{x})$ defines the probability density function (PDF) or probability mass function (PMF). Given joint observations over $(\boldsymbol{X}_a, \boldsymbol{X}_b)$, probabilistic inference rests on three identities [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 2.1, pp. 25–31) [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 6.2, pp. 177–180):

$$
\begin{aligned}
\text{Sum Rule (Marginalization):} && p(\boldsymbol{x}_a) &= \int_{\mathcal{X}_b} p(\boldsymbol{x}_a, \boldsymbol{x}_b) \mathrm{d}\nu(\boldsymbol{x}_b), \\
  \text{Product Rule (Chain Factor):} && p(\boldsymbol{x}_a, \boldsymbol{x}_b) &= p(\boldsymbol{x}_a \mid \boldsymbol{x}_b) p(\boldsymbol{x}_b), \\
  \text{Bayes' Rule (Posterior):} && p(\boldsymbol{x}_b \mid \boldsymbol{x}_a) &= \frac{p(\boldsymbol{x}_a \mid \boldsymbol{x}_b) p(\boldsymbol{x}_b)}{\int_{\mathcal{X}_b} p(\boldsymbol{x}_a \mid \boldsymbol{x}_b') p(\boldsymbol{x}_b') \mathrm{d}\nu(\boldsymbol{x}_b')}.
\end{aligned}
$$

Here we also invoke the Law of Total Probability for any countable measurable partition $\{A_i\}_{i\ge 1}$ of $\Omega$: $P(B) = \sum_{i} P(B \mid A_i) P(A_i)$ whenever $P(A_i) > 0$ [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 2.4.3, pp. 89–91). For finite-variance scalar random variables, the centered second moment $\mathrm{Cov}[X,Y] = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$ defines an inner product on the Hilbert space of zero-mean square-integrable random variables, inducing correlation $\mathrm{Corr}[X,Y] = \mathrm{Cov}[X,Y]/\sqrt{\mathrm{Var}[X]\mathrm{Var}[Y]} \in [-1,1]$ by Cauchy–Schwarz [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 6.4.1, pp. 188–189) [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 5.3, pp. 260–266).

Because the five primary references adopt distinct conventions for $\sigma$-algebras, densities, precision forms, exponential-family symbols, and Jacobian transforms, Table `tab:probdist:notation` establishes a unified notation used throughout this monograph.

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> | **Concept** | **Unified Symbol** | **Source-Book Conventions Reconciled** |
> | — | — | — |
> | Probability space & CDF | $(\Omega,\mathcal{F},P)$, $p(\boldsymbol{x})$, $F_{\boldsymbol{X}}(\boldsymbol{x})$ | Chan [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]] (pp. 58–76, 180) uses $(\Omega,\mathcal{F},P)$, $f_X(x)$, and $F_X(x)$; MML [[Source - Mathematics for Machine Learning]] (pp. 172–175) writes $(\Omega,\mathcal{A},\mathcal{P})$; Murphy [[Source - Probabilistic Machine Learning]] (pp. 2–7) writes $(\Omega,\mathcal{F},P)$ with uppercase $P_X(x)$ for CDF; Bishop [[Source - Foundations]] (pp. 25–41) omits $(\Omega,\mathcal{F},P)$ and denotes PDF/CDF by $p(x)/P(x)$. |
> | [3pt]   MVN moment vs. precision form | $\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$, $\boldsymbol{\Lambda}=\boldsymbol{\Sigma}^{-1}$, $(\boldsymbol{h},\boldsymbol{K})$ | Bishop [[Source - Foundations]] (pp. 71–83) uses block $(\boldsymbol{x}_a,\boldsymbol{x}_b)$ and precision $\boldsymbol{\Lambda}=\boldsymbol{\Sigma}^{-1}$; MML [[Source - Mathematics for Machine Learning]] (pp. 197–204) partitions $(\boldsymbol{x},\boldsymbol{y})$; Chan [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]] (pp. 290–294) writes $\mathrm{Gaussian}(\boldsymbol{\mu},\boldsymbol{\Sigma})$; Murphy [[Source - Probabilistic Machine Learning]] (pp. 22–32) partitions $(\boldsymbol{x}_1,\boldsymbol{x}_2)$ and introduces the canonical potential $(g,\boldsymbol{h},\boldsymbol{K})$ with $\boldsymbol{K}=\boldsymbol{\Sigma}^{-1}$ and $\boldsymbol{h}=\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}$. |
> | [3pt]   Canonical exponential family | $h(\boldsymbol{x})\exp\bigl(\langle \boldsymbol{\eta}, \boldsymbol{T}(\boldsymbol{x}) \rangle$ $\qquad - A(\boldsymbol{\eta})\bigr)$ | Murphy [[Source - Probabilistic Machine Learning]] (pp. 93–97) [[Source - Probabilistic Machine Learning]] (pp. 34–41) uses $(\boldsymbol{\eta}, \boldsymbol{T}(\boldsymbol{x}), A(\boldsymbol{\eta}))$ with mean parameters $\boldsymbol{m}=\nabla A(\boldsymbol{\eta})$; MML [[Source - Mathematics for Machine Learning]] (pp. 210–213) uses $(\boldsymbol{\theta},\boldsymbol{\phi}(\boldsymbol{x}),A(\boldsymbol{\theta}))$ and prior hyperparameters $\boldsymbol{\gamma}$; Bishop [[Source - Foundations]] (pp. 94–97) writes $h(\boldsymbol{x})g(\boldsymbol{\eta})\exp(\boldsymbol{\eta}^{\top}\boldsymbol{u}(\boldsymbol{x}))$ with $g(\boldsymbol{\eta}) = \exp(-A(\boldsymbol{\eta}))$. |
> | [3pt]   Change-of-variables Jacobian | $\lvert \det \boldsymbol{J}_{\boldsymbol{f}^{-1}}(\boldsymbol{y}) \rvert$ | For $\boldsymbol{y}=\boldsymbol{f}(\boldsymbol{x})$, MML [[Source - Mathematics for Machine Learning]] (p. 219) and Murphy [[Source - Probabilistic Machine Learning]] (p. 44) apply the inverse Jacobian $\boldsymbol{J}_{\boldsymbol{f}^{-1}}(\boldsymbol{y})$; Bishop [[Source - Foundations]] (pp. 42–45) parameterizes directly via $\boldsymbol{x}=\boldsymbol{g}(\boldsymbol{y})$ so $\boldsymbol{J}_{\boldsymbol{g}}(\boldsymbol{y}) = \boldsymbol{J}_{\boldsymbol{f}^{-1}}(\boldsymbol{y})$. |
> | [3pt]   Characteristic function sign | $\varphi_X(\omega) = \mathbb{E}[e^{i\omega X}]$ | Standard probability convention uses $+i\omega X$; Chan [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]] (p. 330) adopts the signal-processing Fourier convention $\Phi_X(j\omega) = \mathbb{E}[e^{-j\omega X}]$. |

## Core Mathematical Derivations & Theorems

### Transformations of Random Variables & Generating Functions

> [!important] Theorem: Change of Variables, Jacobian Scaling & Probability Integral Transform
> Let $U, V \subseteq \mathbb{R}^D$ be open sets and $\boldsymbol{f} : U \to V$ be a $C^1$-diffeomorphism with $\boldsymbol{y} = \boldsymbol{f}(\boldsymbol{x})$ and smooth inverse $\boldsymbol{x} = \boldsymbol{f}^{-1}(\boldsymbol{y})$. Conservation of probability mass requires that for every measurable region $B \subseteq U$,
>
> $$
> \int_{B} p_{\boldsymbol{x}}(\boldsymbol{x}) \mathrm{d}\boldsymbol{x}
>   = P(\boldsymbol{X} \in B)
>   = P(\boldsymbol{Y} \in \boldsymbol{f}(B))
>   = \int_{\boldsymbol{f}(B)} p_{\boldsymbol{y}}(\boldsymbol{y}) \mathrm{d}\boldsymbol{y}.
> $$
>
> Applying the multivariable change-of-variables formula $\mathrm{d}\boldsymbol{x} = \lvert \det \boldsymbol{J}_{\boldsymbol{f}^{-1}}(\boldsymbol{y}) \rvert \mathrm{d}\boldsymbol{y}$, where $(\boldsymbol{J}_{\boldsymbol{f}^{-1}}(\boldsymbol{y}))_{ij} = \frac{\partial x_i}{\partial y_j}$, and shrinking $B$ to a neighborhood around $\boldsymbol{x}$ yields the pushforward density [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 2.4, pp. 42–45) [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 6.7, pp. 214–220) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 2.6, pp. 43–45):
>
> $$
> p_{\boldsymbol{y}}(\boldsymbol{y})
>   = p_{\boldsymbol{x}}\!\left(\boldsymbol{f}^{-1}(\boldsymbol{y})\right) \lvert \det \boldsymbol{J}_{\boldsymbol{f}^{-1}}(\boldsymbol{y}) \rvert
>   = p_{\boldsymbol{x}}(\boldsymbol{x}) \lvert \det \boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x}) \rvert^{-1}.
> $$
>
> In one dimension ($D=1$), if $X$ has strictly increasing continuous CDF $F_X$, choosing $y = F_X(x)$ gives $\frac{\mathrm{d}x}{\mathrm{d}y} = \bigl(F_X'(x)\bigr)^{-1} = p_X(x)^{-1}$, hence $p_Y(y) = p_X(x)\cdot p_X(x)^{-1} = 1$ on $[0,1]$: the *probability integral transform (PIT)* $F_X(X) \sim \mathcal{U}(0,1)$, and conversely $F_X^{-1}(U) \sim p_X$ for $U \sim \mathcal{U}(0,1)$.

> [!tip] Local Volume Distortion and Non-Invariance of Modes
> The factor $\lvert \det \boldsymbol{J}_{\boldsymbol{f}^{-1}}(\boldsymbol{y}) \rvert$ measures the ratio of infinitesimal hypervolumes $\mathrm{d}\boldsymbol{x} / \mathrm{d}\boldsymbol{y}$. Where $\boldsymbol{f}$ compresses coordinates ($\lvert \det \boldsymbol{J}_{\boldsymbol{f}} \rvert < 1$), probability mass concentrates and the output density spikes upward so total integral mass remains unity. Because $\nabla_{\boldsymbol{y}} \log p_{\boldsymbol{y}}(\boldsymbol{y}) = \boldsymbol{J}_{\boldsymbol{f}^{-1}}(\boldsymbol{y})^{\top} \nabla_{\boldsymbol{x}}\log p_{\boldsymbol{x}}(\boldsymbol{x}) + \nabla_{\boldsymbol{y}}\log\lvert \det \boldsymbol{J}_{\boldsymbol{f}^{-1}}(\boldsymbol{y}) \rvert$, the mode of $p_{\boldsymbol{y}}$ differs from $\boldsymbol{f}(\arg\max p_{\boldsymbol{x}})$ whenever $\boldsymbol{f}$ is nonlinear [[Source - Foundations|deep_learning_foundations_concepts]] (Fig. 2.12, p. 44).

> [!important] Derivation: Moment & Cumulant Generating Functions and Convolution
> For a random variable $X$ on $\mathbb{R}$ (or vector $\boldsymbol{X}$ on $\mathbb{R}^D$), the *moment generating function (MGF)* and *cumulant generating function (CGF)* are defined wherever the expectation is finite in an open neighborhood $\mathcal{U}$ of the origin [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6.1, pp. 324–329):
>
> $$
> M_{\boldsymbol{X}}(\boldsymbol{s}) = \mathbb{E}\!\left[e^{\langle \boldsymbol{s}, \boldsymbol{X} \rangle}\right], \qquad
>   K_{\boldsymbol{X}}(\boldsymbol{s}) = \log M_{\boldsymbol{X}}(\boldsymbol{s}), \qquad \boldsymbol{s} \in \mathcal{U}.
> $$
>
> Differentiating under the integral sign (justified by dominated convergence on $\mathcal{U}^{\circ}$) yields raw moments from $M_X$ and cumulants $\kappa_k$ from $K_X$:
>
> $$
> \begin{aligned}
> \left.\frac{\mathrm{d}^k}{\mathrm{d}s^k} M_X(s)\right|_{s=0} &= \mathbb{E}[X^k], &
>   \kappa_1 &= K_X'(0) = \mathbb{E}[X], \\
>   \kappa_2 &= K_X"(0) = \mathrm{Var}[X], &
>   \kappa_4 &= K_X^{(4)}(0) = \mathbb{E}[(X-\mu)^4] - 3\sigma^4.
> \end{aligned}
> $$
>
> If $X_1, \dots, X_N$ are independent and $Z = \sum_{n=1}^N X_n$, factorization of $\mathbb{E}[e^{s\sum_n X_n}]$ converts convolution of densities into algebraic multiplication of MGFs and *addition* of CGFs [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Thm. 6.2–6.6, pp. 327–329):
>
> $$
> M_Z(s) = \prod_{n=1}^N M_{X_n}(s) \implies K_Z(s) = \sum_{n=1}^N K_{X_n}(s).
> $$
>
> For $\boldsymbol{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, completing the square in $\mathbb{E}[e^{\langle \boldsymbol{s}, \boldsymbol{X} \rangle}]$ gives $M_{\boldsymbol{X}}(\boldsymbol{s}) = \exp\!\bigl(\langle \boldsymbol{s}, \boldsymbol{\mu} \rangle + \tfrac{1}{2}\boldsymbol{s}^{\top}\boldsymbol{\Sigma}\boldsymbol{s}\bigr)$ and $K_{\boldsymbol{X}}(\boldsymbol{s}) = \langle \boldsymbol{s}, \boldsymbol{\mu} \rangle + \tfrac{1}{2}\boldsymbol{s}^{\top}\boldsymbol{\Sigma}\boldsymbol{s}$; every cumulant of order $k \ge 3$ vanishes identically, proving closure of Gaussians under independent summation and affine maps.

### Multivariate Gaussian Geometry, Whitening & Schur Conditioning

A random vector $\boldsymbol{x} \in \mathbb{R}^D$ follows a non-degenerate multivariate Gaussian distribution $\mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})$ with mean vector $\boldsymbol{\mu} \in \mathbb{R}^D$ and symmetric positive definite (SPD) covariance matrix $\boldsymbol{\Sigma} \in \mathbb{S}_{++}^D$ (precision $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1} \in \mathbb{S}_{++}^D$) when its density takes the quadratic exponential form [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 3.2, pp. 70–75):

$$
p(\boldsymbol{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})
  = \frac{1}{(2\pi)^{D/2}\lvert \boldsymbol{\Sigma} \rvert^{1/2}}
    \exp\!\left(-\frac{1}{2}\Delta^2(\boldsymbol{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})\right), \qquad
  \Delta^2(\boldsymbol{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})
  = (\boldsymbol{x}-\boldsymbol{\mu})^{\top} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x}-\boldsymbol{\mu}),
$$

as also analyzed via Mahalanobis geometry in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 3.2, pp. 81–83). Level sets of constant density are $D$-dimensional hyperellipsoids $\{\boldsymbol{x} : \Delta^2(\boldsymbol{x}) = c^2\}$ whose principal axes align with the orthonormal eigenvectors $\{\boldsymbol{u}_i\}_{i=1}^D$ of $\boldsymbol{\Sigma} = \boldsymbol{U}\boldsymbol{\Gamma}\boldsymbol{U}^{\top}$ ($\boldsymbol{\Gamma} = \mathrm{diag}(\gamma_1,\dots,\gamma_D)$, $\gamma_i > 0$) with semi-axis lengths $c\sqrt{\gamma_i}$. Applying the affine whitening transformation $\boldsymbol{z} = \boldsymbol{\Gamma}^{-1/2}\boldsymbol{U}^{\top}(\boldsymbol{x}-\boldsymbol{\mu})$ (PCA whitening) or $\boldsymbol{z}_{\mathrm{ZCA}} = \boldsymbol{U}\boldsymbol{\Gamma}^{-1/2}\boldsymbol{U}^{\top}(\boldsymbol{x}-\boldsymbol{\mu}) = \boldsymbol{\Sigma}^{-1/2}(\boldsymbol{x}-\boldsymbol{\mu})$ (ZCA whitening) yields $\lvert \det \boldsymbol{J}_{\boldsymbol{x}}(\boldsymbol{z}) \rvert = \prod_{i=1}^D \gamma_i^{1/2} = \lvert \boldsymbol{\Sigma} \rvert^{1/2}$ by `(eq:probdist:cov_density)`, proving $\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_D)$ and confirming normalization $\int_{\mathbb{R}^D} p(\boldsymbol{x}\mid\boldsymbol{\mu},\boldsymbol{\Sigma}) \mathrm{d}\boldsymbol{x} = 1$ [[Source - Foundations|deep_learning_foundations_concepts]] (pp. 73–75) [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 5.6.5, pp. 296–302).

> [!important] Derivation: Schur Complement Conditioning, Marginalization & Gaussian Bayes
> Partition $\boldsymbol{x} \in \mathbb{R}^{D_a + D_b}$ into subvectors $\boldsymbol{x} = \begin{bmatrix}\boldsymbol{x}_a \\ \boldsymbol{x}_b\end{bmatrix}$ with matching mean, covariance, and precision blocks [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 3.2.1–3.2.5, pp. 76–83) [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 6.5.1–6.5.2, pp. 199–201) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 3.2.3, 7.3, pp. 83–85, 249–252):
>
> $$
> \boldsymbol{\mu} = \begin{bmatrix}\boldsymbol{\mu}_a \\ \boldsymbol{\mu}_b\end{bmatrix}, \qquad
>   \boldsymbol{\Sigma} = \begin{bmatrix}\boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab} \\ \boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb}\end{bmatrix}, \qquad
>   \boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1} = \begin{bmatrix}\boldsymbol{\Lambda}_{aa} & \boldsymbol{\Lambda}_{ab} \\ \boldsymbol{\Lambda}_{ba} & \boldsymbol{\Lambda}_{bb}\end{bmatrix}.
> $$
>
> Block-LDU factorization of $\boldsymbol{\Sigma}$ isolates the Schur complement $\boldsymbol{\Sigma}/\boldsymbol{\Sigma}_{bb} := \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba} \succ \boldsymbol{0}$:
>
> $$
> \boldsymbol{\Sigma}
>   = \begin{bmatrix}\boldsymbol{I} & \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}
>     \begin{bmatrix}\boldsymbol{\Sigma}/\boldsymbol{\Sigma}_{bb} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{\Sigma}_{bb}\end{bmatrix}
>     \begin{bmatrix}\boldsymbol{I} & \boldsymbol{0} \\ \boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba} & \boldsymbol{I}\end{bmatrix}.
> $$
>
> Taking inverses and determinants immediately establishes $\lvert \boldsymbol{\Sigma} \rvert = \lvert \boldsymbol{\Sigma}_{bb} \rvert\lvert \boldsymbol{\Sigma}/\boldsymbol{\Sigma}_{bb} \rvert$ and the block-precision identities:
>
> $$
> \begin{aligned}
> \boldsymbol{\Lambda}
>   &= \begin{bmatrix}\boldsymbol{I} & \boldsymbol{0} \\ -\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba} & \boldsymbol{I}\end{bmatrix}
>      \begin{bmatrix}(\boldsymbol{\Sigma}/\boldsymbol{\Sigma}_{bb})^{-1} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{\Sigma}_{bb}^{-1}\end{bmatrix}
>      \begin{bmatrix}\boldsymbol{I} & -\boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}, \\
>   \boldsymbol{\Lambda}_{aa}
>   &= (\boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba})^{-1}, \qquad
>   \boldsymbol{\Lambda}_{ab} = -\boldsymbol{\Lambda}_{aa}\boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}.
> \end{aligned}
> $$
>
> Substituting `(eq:probdist:precision_blocks)` into the Mahalanobis quadratic form decomposes the joint exponent cleanly into a conditional quadratic term in $\boldsymbol{x}_a$ and a marginal quadratic term in $\boldsymbol{x}_b$:
>
> $$
> \begin{aligned}
> (\boldsymbol{x}-\boldsymbol{\mu})^{\top}\boldsymbol{\Lambda}(\boldsymbol{x}-\boldsymbol{\mu})
>   &= \bigl(\boldsymbol{x}_a - \boldsymbol{\mu}_{a\mid b}\bigr)^{\top} \boldsymbol{\Lambda}_{aa} \bigl(\boldsymbol{x}_a - \boldsymbol{\mu}_{a\mid b}\bigr)
>    + (\boldsymbol{x}_b - \boldsymbol{\mu}_b)^{\top} \boldsymbol{\Sigma}_{bb}^{-1} (\boldsymbol{x}_b - \boldsymbol{\mu}_b), \\
>   \boldsymbol{\mu}_{a\mid b}
>   &= \boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}(\boldsymbol{x}_b - \boldsymbol{\mu}_b)
>    = \boldsymbol{\mu}_a - \boldsymbol{\Lambda}_{aa}^{-1}\boldsymbol{\Lambda}_{ab}(\boldsymbol{x}_b - \boldsymbol{\mu}_b), \\
>   \boldsymbol{\Sigma}_{a\mid b}
>   &= \boldsymbol{\Lambda}_{aa}^{-1}
>    = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}.
> \end{aligned}
> $$
>
> Consequently, $p(\boldsymbol{x}_a \mid \boldsymbol{x}_b) = \mathcal{N}(\boldsymbol{x}_a \mid \boldsymbol{\mu}_{a\mid b}, \boldsymbol{\Sigma}_{a\mid b})$ (with precision $\boldsymbol{\Lambda}_{a\mid b} = \boldsymbol{\Lambda}_{aa}$) and integrating out $\boldsymbol{x}_a$ yields the marginal $p(\boldsymbol{x}_b) = \mathcal{N}(\boldsymbol{x}_b \mid \boldsymbol{\mu}_b, \boldsymbol{\Sigma}_{bb})$ (by symmetry, $p(\boldsymbol{x}_a) = \mathcal{N}(\boldsymbol{x}_a \mid \boldsymbol{\mu}_a, \boldsymbol{\Sigma}_{aa})$). Furthermore, for a linear-Gaussian system $p(\boldsymbol{z}) = \mathcal{N}(\boldsymbol{\mu}_z, \boldsymbol{\Sigma}_z)$ and $p(\boldsymbol{y}\mid\boldsymbol{z}) = \mathcal{N}(\boldsymbol{W}\boldsymbol{z}+\boldsymbol{b}, \boldsymbol{\Sigma}_y)$, completing the square in $\boldsymbol{z}$ (equivalently applying `(eq:probdist:schur_cond)` to the joint vector $[\boldsymbol{z}^{\top}, \boldsymbol{y}^{\top}]^{\top}$) yields the exact Gaussian Bayes posterior $p(\boldsymbol{z}\mid\boldsymbol{y}) = \mathcal{N}(\boldsymbol{\mu}_{z\mid y}, \boldsymbol{\Sigma}_{z\mid y})$:
>
> $$
> \boldsymbol{\Sigma}_{z\mid y}^{-1} = \boldsymbol{\Sigma}_z^{-1} + \boldsymbol{W}^{\top} \boldsymbol{\Sigma}_y^{-1} \boldsymbol{W}, \qquad
>   \boldsymbol{\mu}_{z\mid y} = \boldsymbol{\Sigma}_{z\mid y}\bigl(\boldsymbol{W}^{\top} \boldsymbol{\Sigma}_y^{-1}(\boldsymbol{y}-\boldsymbol{b}) + \boldsymbol{\Sigma}_z^{-1} \boldsymbol{\mu}_z\bigr).
> $$

> [!note] Source Figure
> *Figure:* Geometric comparison of MVN marginalization and Schur-complement conditioning [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 3.2, Fig. 3.5, p. 82). (a) Joint density contours $p(x_a, x_b)$ sliced along $x_b = 0.7$. (b) Conditional density $p(x_a \mid x_b = 0.7)$ has a shifted mean $\mu_{a\mid b}$ and strictly smaller variance $\Sigma_{a\mid b} = \Sigma_{aa} - \Sigma_{ab}^2/\Sigma_{bb} \le \Sigma_{aa}$ than the marginal $p(x_a)$. (`deep_learning_foundations_concepts_p101_fig1.png`)

> [!tip] Dual Computational Economy: Covariance vs. Precision
> As highlighted in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 2.3.1.4, p. 25), equation `(eq:probdist:schur_cond)` reveals a striking coordinate asymmetry: marginalization is trivial in moment form ($\boldsymbol{\Sigma} \mapsto \boldsymbol{\Sigma}_{aa}$) while conditioning requires a Schur complement, whereas conditioning is trivial in precision form ($\boldsymbol{\Lambda} \mapsto \boldsymbol{\Lambda}_{aa}$) while marginalization requires $\boldsymbol{\Lambda}_{aa} - \boldsymbol{\Lambda}_{ab}\boldsymbol{\Lambda}_{bb}^{-1}\boldsymbol{\Lambda}_{ba}$. Equivalently, in Murphy's canonical potential $\exp(g + \boldsymbol{h}^{\top}\boldsymbol{x} - \tfrac{1}{2}\boldsymbol{x}^{\top}\boldsymbol{K}\boldsymbol{x})$ where $\boldsymbol{K}=\boldsymbol{\Lambda}$ and $\boldsymbol{h}=\boldsymbol{\Lambda}\boldsymbol{\mu}$, multiplying Gaussian factors sums $(\boldsymbol{h}, \boldsymbol{K})$ directly, and $\Lambda_{ij} = 0 \iff X_i \perp\!\!\!\perp X_j \mid \boldsymbol{X}_{-\{i,j\}}$ characterizes conditional independence graphs [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 2.3.3, pp. 30–33).

### Canonical Exponential Families, Log-Partition Derivatives & Conjugacy

A parametric family $\{p(\boldsymbol{x} \mid \boldsymbol{\eta})\}$ belongs to the canonical *exponential family* if its density against reference measure $\nu$ takes the unified log-linear form ([[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 3.4, pp. 94–97)):

$$
p(\boldsymbol{x} \mid \boldsymbol{\eta})
  = h(\boldsymbol{x}) \exp\!\bigl(\langle \boldsymbol{\eta}, \boldsymbol{T}(\boldsymbol{x}) \rangle - A(\boldsymbol{\eta})\bigr), \qquad
  A(\boldsymbol{\eta}) = \log \int_{\mathcal{X}} h(\boldsymbol{x}) \exp\!\bigl(\langle \boldsymbol{\eta}, \boldsymbol{T}(\boldsymbol{x}) \rangle\bigr) \mathrm{d}\nu(\boldsymbol{x}),
$$

as developed in [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 6.6.3, pp. 210–213) and unified through convex duality in Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 3.4, pp. 93–97). Here $h(\boldsymbol{x}) \ge 0$ is the base measure, $\boldsymbol{T}(\boldsymbol{x}) \in \mathbb{R}^K$ are the sufficient statistics, $\boldsymbol{\eta} \in \Omega_\eta = \{\boldsymbol{\eta} \in \mathbb{R}^K : A(\boldsymbol{\eta}) < \infty\}$ are the natural (canonical) parameters, and $A(\boldsymbol{\eta}) = \log Z(\boldsymbol{\eta})$ is the log-partition (cumulant) function. The family is *minimal* when no nonzero $\boldsymbol{a} \in \mathbb{R}^K$ satisfies $\langle \boldsymbol{a}, \boldsymbol{T}(\boldsymbol{x}) \rangle = \text{const}$ $\nu$-a.e., and *curved* when $\boldsymbol{\eta}(\boldsymbol{\theta})$ maps from a lower-dimensional manifold into $\mathbb{R}^K$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 2.4, pp. 34–41).

> [!important] Theorem: Convexity, Log-Partition Moment/Variance Identities & Legendre Duality
> On the interior $\Omega_\eta^\circ$, $A(\boldsymbol{\eta})$ is $C^\infty$ and convex (by H\"older's inequality applied to $\int h(\boldsymbol{x})\exp(\langle \lambda\boldsymbol{\eta}_1 + (1-\lambda)\boldsymbol{\eta}_2, \boldsymbol{T}(\boldsymbol{x}) \rangle) \mathrm{d}\nu(\boldsymbol{x})$). Differentiating the normalization identity $\int h(\boldsymbol{x})\exp(\langle \boldsymbol{\eta}, \boldsymbol{T}(\boldsymbol{x}) \rangle - A(\boldsymbol{\eta})) \mathrm{d}\nu(\boldsymbol{x}) = 1$ twice with respect to $\boldsymbol{\eta}$ inside the integral yields [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 2.4.2.2, pp. 35–36):
>
> $$
> \begin{aligned}
> \nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta})
>   &= \frac{\int h(\boldsymbol{x}) \boldsymbol{T}(\boldsymbol{x})\exp(\langle \boldsymbol{\eta}, \boldsymbol{T}(\boldsymbol{x}) \rangle) \mathrm{d}\nu(\boldsymbol{x})}{Z(\boldsymbol{\eta})}
>    = \mathbb{E}_{p(\cdot\mid\boldsymbol{\eta})}[\boldsymbol{T}(\boldsymbol{x})] =: \boldsymbol{m}, \\
>   \nabla^2_{\boldsymbol{\eta}} A(\boldsymbol{\eta})
>   &= \mathbb{E}_{p(\cdot\mid\boldsymbol{\eta})}\!\left[\boldsymbol{T}(\boldsymbol{x})\boldsymbol{T}(\boldsymbol{x})^{\top}\right] - \mathbb{E}[\boldsymbol{T}(\boldsymbol{x})]\mathbb{E}[\boldsymbol{T}(\boldsymbol{x})]^{\top}
>    = \mathrm{Cov}_{p(\cdot\mid\boldsymbol{\eta})}[\boldsymbol{T}(\boldsymbol{x})]
>    = \boldsymbol{I}_F(\boldsymbol{\eta}) \succeq \boldsymbol{0},
> \end{aligned}
> $$
>
> where $\boldsymbol{I}_F(\boldsymbol{\eta}) = \mathbb{E}[-\nabla^2_{\boldsymbol{\eta}}\log p(\boldsymbol{x}\mid\boldsymbol{\eta})]$ is the Fisher information matrix (strictly positive definite $\nabla^2 A \succ \boldsymbol{0}$ whenever the representation is minimal). Moreover, the CGF of $\boldsymbol{T}(\boldsymbol{x})$ is simply $K_{\boldsymbol{T}}(\boldsymbol{s}) = \log \mathbb{E}[\exp(\langle \boldsymbol{s}, \boldsymbol{T}(\boldsymbol{x}) \rangle)] = A(\boldsymbol{\eta}+\boldsymbol{s}) - A(\boldsymbol{\eta})$. Thus $\boldsymbol{m} = \nabla A(\boldsymbol{\eta})$ maps $\Omega_\eta^\circ$ bijectively onto the interior $\mathcal{M}^\circ$ of the realizable mean-parameter set, with inverse $\boldsymbol{\eta} = \nabla A^*(\boldsymbol{m})$ given by the Fenchel–Legendre conjugate $A^*(\boldsymbol{m}) = \sup_{\boldsymbol{\eta}\in\Omega_\eta}\bigl(\langle \boldsymbol{\eta}, \boldsymbol{m} \rangle - A(\boldsymbol{\eta})\bigr) = -\mathrm{H}(p_{\boldsymbol{\eta}(\boldsymbol{m})}) + \mathbb{E}[\log h(\boldsymbol{x})]$.

To cast the multivariate Gaussian `(eq:probdist:mvn_pdf)` into canonical form `(eq:probdist:exp_family)`, expand the log-density in $\boldsymbol{x}$ and $\boldsymbol{x}\boldsymbol{x}^{\top}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 2.4.2.5, pp. 38–39):

$$
\log p(\boldsymbol{x}\mid\boldsymbol{\mu},\boldsymbol{\Sigma})
  = \log(2\pi)^{-D/2}
  + \langle \boldsymbol{\Lambda}\boldsymbol{\mu}, \boldsymbol{x} \rangle
  + \mathrm{Tr}\!\left(\!\left(-\frac{1}{2}\boldsymbol{\Lambda}\right)\!\boldsymbol{x}\boldsymbol{x}^{\top}\!\right)
  - \left(\frac{1}{2}\boldsymbol{\mu}^{\top}\boldsymbol{\Lambda}\boldsymbol{\mu} - \frac{1}{2}\log\lvert \boldsymbol{\Lambda} \rvert\right).
$$

Matching terms identifies $h(\boldsymbol{x}) = (2\pi)^{-D/2}$, sufficient statistics $\boldsymbol{T}(\boldsymbol{x}) = (\boldsymbol{x}, \boldsymbol{x}\boldsymbol{x}^{\top})$, natural parameters $\boldsymbol{\eta}_1 = \boldsymbol{\Lambda}\boldsymbol{\mu}$ and $\boldsymbol{\eta}_2 = -\tfrac{1}{2}\boldsymbol{\Lambda} \in -\mathbb{S}_{++}^D$, and log-partition

$$
A(\boldsymbol{\eta}_1, \boldsymbol{\eta}_2)
  = -\frac{1}{4}\boldsymbol{\eta}_1^{\top} \boldsymbol{\eta}_2^{-1} \boldsymbol{\eta}_1 - \frac{1}{2}\log\lvert -2\boldsymbol{\eta}_2 \rvert.
$$

Applying `(eq:probdist:grad_A)` via matrix calculus ($\frac{\partial \log\lvert \boldsymbol{M} \rvert}{\partial \boldsymbol{M}} = \boldsymbol{M}^{-\top}$ and $\frac{\partial \boldsymbol{v}^{\top}\boldsymbol{M}^{-1}\boldsymbol{v}}{\partial \boldsymbol{M}} = -\boldsymbol{M}^{-\top}\boldsymbol{v}\boldsymbol{v}^{\top}\boldsymbol{M}^{-\top}$) immediately recovers $\nabla_{\boldsymbol{\eta}_1} A = -\tfrac{1}{2}\boldsymbol{\eta}_2^{-1}\boldsymbol{\eta}_1 = \boldsymbol{\mu} = \mathbb{E}[\boldsymbol{x}]$ and $\nabla_{\boldsymbol{\eta}_2} A = \boldsymbol{\mu}\boldsymbol{\mu}^{\top} + \boldsymbol{\Sigma} = \mathbb{E}[\boldsymbol{x}\boldsymbol{x}^{\top}]$.

> [!important] Theorem: Maximum Likelihood Moment Matching, MaxEnt & Conjugate Priors
> Given i.i.d. observations $\mathcal{D} = \{\boldsymbol{x}_n\}_{n=1}^N$, the exponential-family log-likelihood $\ell(\boldsymbol{\eta};\mathcal{D}) = \sum_{n=1}^N \log h(\boldsymbol{x}_n) + \langle \boldsymbol{\eta}, \sum_{n=1}^N \boldsymbol{T}(\boldsymbol{x}_n) \rangle - N A(\boldsymbol{\eta})$ is concave in $\boldsymbol{\eta}$ (since $-\nabla^2_\eta \ell = N\nabla^2 A \succeq \boldsymbol{0}$). Setting $\nabla_{\boldsymbol{\eta}}\ell = \boldsymbol{0}$ yields the global MLE *moment-matching* equation [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 3.4, pp. 96–97) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 2.4.4–2.4.5, pp. 40–41):
>
> $$
> \nabla_{\boldsymbol{\eta}} A(\hat{\boldsymbol{\eta}}_{\mathrm{MLE}})
>   = \mathbb{E}_{p(\cdot\mid\hat{\boldsymbol{\eta}}_{\mathrm{MLE}})}[\boldsymbol{T}(\boldsymbol{x})]
>   = \frac{1}{N}\sum_{n=1}^N \boldsymbol{T}(\boldsymbol{x}_n) =: \bar{\boldsymbol{t}}_{\mathcal{D}}.
> $$
>
> Under fixed-support regularity, the Pitman–Koopman–Darmois theorem states that only exponential families admit a fixed-dimension sufficient statistic $\sum_{n=1}^N \boldsymbol{T}(\boldsymbol{x}_n)$ independent of $N$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 6.6.2, pp. 209–210); dually, maximizing differential entropy $-\int p\log p \mathrm{d}\nu$ subject to $\mathbb{E}_p[\boldsymbol{T}(\boldsymbol{x})] = \bar{\boldsymbol{t}}$ uniquely yields `(eq:probdist:exp_family)` (with $h(\boldsymbol{x})=1$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 3.4.4, p. 97). Finally, for any exponential-family likelihood `(eq:probdist:exp_family)`, the canonical *conjugate prior* parameterized by $(\boldsymbol{\tau}_0, \nu_0)$ is [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3.2.5, pp. 96–98) [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 6.6.1, pp. 205–208):
>
> $$
> p(\boldsymbol{\eta} \mid \boldsymbol{\tau}_0, \nu_0)
>   = \exp\!\bigl(\langle \boldsymbol{\eta}, \boldsymbol{\tau}_0 \rangle - \nu_0 A(\boldsymbol{\eta}) - \psi(\boldsymbol{\tau}_0, \nu_0)\bigr),
> $$
>
> where $\exp(\psi(\boldsymbol{\tau}_0, \nu_0)) = \int_{\Omega_\eta} \exp(\langle \boldsymbol{\eta}, \boldsymbol{\tau}_0 \rangle - \nu_0 A(\boldsymbol{\eta})) \mathrm{d}\boldsymbol{\eta}$. Multiplying by $p(\mathcal{D}\mid\boldsymbol{\eta}) = \bigl(\prod_{n=1}^N h(\boldsymbol{x}_n)\bigr)\exp(\langle \boldsymbol{\eta}, N\bar{\boldsymbol{t}}_{\mathcal{D}} \rangle - N A(\boldsymbol{\eta}))$ yields a posterior in the exact same family and a closed-form log marginal likelihood:
>
> $$
> \begin{aligned}
> p(\boldsymbol{\eta}\mid\mathcal{D}) &= p(\boldsymbol{\eta}\mid \boldsymbol{\tau}_N, \nu_N), &
>   \boldsymbol{\tau}_N &= \boldsymbol{\tau}_0 + \sum_{n=1}^N \boldsymbol{T}(\boldsymbol{x}_n), &
>   \nu_N &= \nu_0 + N, \\
>   \log p(\mathcal{D}) &= \sum_{n=1}^N \log h(\boldsymbol{x}_n) + \psi(\boldsymbol{\tau}_N, \nu_N) - \psi(\boldsymbol{\tau}_0, \nu_0).
> \end{aligned}
> $$
>
> Canonical instances include Beta–Bernoulli ($T(x)=x$), Dirichlet–Categorical ($T(\boldsymbol{x})=\boldsymbol{x}$ one-hot), Gamma–Poisson/Exponential, and Normal–Inverse-Wishart (NIW)–MVN.

## Algorithmic Implementation & Comparative Synthesis

In practice, computing Schur-complement Gaussian conditionals or conjugate exponential-family posteriors requires avoiding explicit matrix inversions and underflowing product densities. Algorithm `alg:probdist:unified_inference` combines Cholesky-stabilized Schur conditioning with recursive exponential-family conjugate updating in log space.

> [!example] Algorithm: Cholesky-Stabilized Gaussian Schur Conditioning & Exponential-Family Conjugate Update
> **Input:** Observed partition $\boldsymbol{x}_b \in \mathbb{R}^{D_b}$, MVN moment parameters $(\boldsymbol{\mu}_a,\boldsymbol{\mu}_b,\boldsymbol{\Sigma}_{aa},\boldsymbol{\Sigma}_{ab},\boldsymbol{\Sigma}_{bb})$, jitter $\epsilon > 0$; OR i.i.d. batch $\mathcal{D}=\{\boldsymbol{x}_n\}_{n=1}^N$ with prior hyperparameters $(\boldsymbol{\tau}_0, \nu_0)$, sufficient statistic map $\boldsymbol{T}$, and conjugate log-normalizer $\psi$.
>   **Output:** Conditional Gaussian $(\boldsymbol{\mu}_{a\mid b}, \boldsymbol{L}_{a\mid b})$ with $\boldsymbol{\Sigma}_{a\mid b} = \boldsymbol{L}_{a\mid b}\boldsymbol{L}_{a\mid b}^{\top}$, OR posterior hyperparameters $(\boldsymbol{\tau}_N, \nu_N)$ and marginal log-evidence $\log p(\mathcal{D})$.
>    *// Branch 1: Numerically stable Schur-complement MVN conditioning*
>   $\boldsymbol{L}_{bb} \gets \mathrm{Cholesky}\!\left(\boldsymbol{\Sigma}_{bb} + \epsilon \boldsymbol{I}_{D_b}\right)$  *// Lower triangular factor, $\mathcal{O}(D_b^3)$*
>   Solve $\boldsymbol{L}_{bb}\boldsymbol{\alpha} = \boldsymbol{x}_b - \boldsymbol{\mu}_b$ by forward substitution
>   Solve $\boldsymbol{L}_{bb}\boldsymbol{K}^{\top} = \boldsymbol{\Sigma}_{ba}$ by forward substitution  *// $\boldsymbol{K}^{\top} = \boldsymbol{L}_{bb}^{-1}\boldsymbol{\Sigma}_{ba} \in \mathbb{R}^{D_b \times D_a}$*
>   $\boldsymbol{\mu}_{a\mid b} \gets \boldsymbol{\mu}_a + \boldsymbol{K}\boldsymbol{\alpha}$  *// Avoids explicit $\boldsymbol{\Sigma}_{bb}^{-1}$: $\boldsymbol{K}\boldsymbol{\alpha} = \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}(\boldsymbol{x}_b-\boldsymbol{\mu}_b)$*
>   $\boldsymbol{\Sigma}_{a\mid b} \gets \tfrac{1}{2}\!\left((\boldsymbol{\Sigma}_{aa} - \boldsymbol{K}\boldsymbol{K}^{\top}) + (\boldsymbol{\Sigma}_{aa} - \boldsymbol{K}\boldsymbol{K}^{\top})^{\top}\right)$  *// Symmetrized Schur complement*
>   $\boldsymbol{L}_{a\mid b} \gets \mathrm{Cholesky}\!\left(\boldsymbol{\Sigma}_{a\mid b} + \epsilon \boldsymbol{I}_{D_a}\right)$
>    *// Branch 2: Recursive exponential-family conjugate Bayesian update*
>   $\boldsymbol{S}_{\mathcal{D}} \gets \boldsymbol{0}$;\quad $\ell_h \gets 0$
>   **for** $n \gets 1$ to $N$:
>
>     $\boldsymbol{S}_{\mathcal{D}} \gets \boldsymbol{S}_{\mathcal{D}} + \boldsymbol{T}(\boldsymbol{x}_n)$;\quad
>     $\ell_h \gets \ell_h + \log h(\boldsymbol{x}_n)$
>
>   $\boldsymbol{\tau}_N \gets \boldsymbol{\tau}_0 + \boldsymbol{S}_{\mathcal{D}}$;\quad $\nu_N \gets \nu_0 + N$
>   $\log p(\mathcal{D}) \gets \ell_h + \psi(\boldsymbol{\tau}_N, \nu_N) - \psi(\boldsymbol{\tau}_0, \nu_0)$
>   **return** $(\boldsymbol{\mu}_{a\mid b}, \boldsymbol{L}_{a\mid b})$ or $(\boldsymbol{\tau}_N, \nu_N, \log p(\mathcal{D}))$

Table `tab:probdist:complexity` contrasts the memory, inference, and statistical trade-offs across covariance structures and canonical exponential-family parameterizations [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 3.2–3.4, pp. 81–98) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 2.3–2.4, pp. 22–41).

**Table:** Asymptotic storage and operational complexity across MVN coordinate representations and conjugate exponential families ($D = D_a + D_b$, sufficient-statistic dimension $K$).

| **Parameterization** | **Parameters** | **Conditioning** | **Marginal** | **Primary Strength / Bottleneck** |
| — | — | — | — | — |
| Full Moment MVN $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | $D + \frac{D(D+1)}{2}$ | $\mathcal{O}(D_b^3 + D_a D_b^2)$ | $\mathcal{O}(1)$ slice | Expressive correlation; covariance MLE singular when $N \le D$. |
| Information MVN $(\boldsymbol{h}, \boldsymbol{K})$ | $D + \frac{D(D+1)}{2}$ | $\mathcal{O}(1)$ slice | $\mathcal{O}(D_b^3 + D_a D_b^2)$ | Additive Bayesian fusion & sparse Markov structure ($K_{ij}=0$). |
| Diagonal MVN $(\boldsymbol{\mu}, \mathrm{diag}(\boldsymbol{\sigma}^2))$ | $2D$ | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ | Linear scaling in $D$, but assumes conditional & marginal independence. |
| Spherical MVN $(\boldsymbol{\mu}, \sigma^2 \boldsymbol{I}_D)$ | $D + 1$ | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ | Minimal sample complexity; isotropic quadratic Euclidean distance. |
| Canonical Conjugate $(\boldsymbol{\tau}_0, \nu_0)$ | $K + 1$ | $\nabla A^*(\boldsymbol{m})$ | $\mathcal{O}(NK)$ update | Exact finite-dim. sufficiency; requires tractable $\psi(\boldsymbol{\tau},\nu)$. |

> [!info] Cross-Book Synthesis: Axiomatic Mechanics to Convex Exponential Duality
> Each textbook contributes a complementary layer to this unified framework. Chan [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 2, 5–6) anchors sample spaces $(\Omega,\mathcal{F},P)$, transformation densities, and MGF/characteristic-function convolution calculus cleanly from engineering first principles. Bishop & Bishop [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 2–3) develop the Gaussian workhorse algebraically via completing the square for conditional/marginal laws and linear-Gaussian Bayesian estimation, highlighting Pitman–Koopman–Darmois sufficiency. Deisenroth, Faisal, and Ong [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 6) supply geometric Hilbert-space inner products for covariance alongside explicit block-inverse and exponential-family proofs. Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 2–4, 7) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 2–3) elevates the hierarchy to Legendre duality ($\nabla A \leftrightarrow \nabla A^*$), canonical potential calculus $(g,\boldsymbol{h},\boldsymbol{K})$ for Gaussian graphical models, Fisher geometry $\nabla^2 A(\boldsymbol{\eta})$, and finite-sample covariance regularization in high dimensions.

## Visual Diagnostics & Geometric Structure

Figure `fig:probdist:duality_tikz` synthesizes the algebraic and convex-analytic relationships connecting moment coordinates $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, precision/natural coordinates $\boldsymbol{\eta} = (\boldsymbol{\Lambda}\boldsymbol{\mu}, -\tfrac{1}{2}\boldsymbol{\Lambda})$, and conjugate exponential-family hyperparameters $(\boldsymbol{\tau}, \nu)$.

> [!example] Visual Diagram (LaTeX / PDF)
> *Duality linking MVN moment parameters $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, precision/natural coordinates $\boldsymbol{\eta}$, and conjugate exponential-family inference via the Legendre pair $(A, A^*)$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:probdist:ellipses_pgfplots` visualizes two geometric hallmarks: (a) how covariance structure shapes bivariate Gaussian iso-density contours and regression lines $\mu_{2\mid 1}(x_1)$, and (b) how quadratic log-density decay in exponential families contrasts with polynomial heavy tails where the MGF diverges.

> [!example] Visual Diagram (LaTeX / PDF)
> *Geometric structure and tail behavior. (a) Iso-density contours at Mahalanobis radius $c=1.5$ comparing spherical, diagonal, and full covariances; the conditional mean line $\mu_{2\mid 1}(x_1)$ tilts below the major principal eigenvector $\boldsymbol{u}_1$ due to regression toward the mean. (b) Gaussian quadratic log-decay guarantees finite $M_X(s)$ everywhere, whereas polynomial tails in Student-$t$ and Cauchy distributions cause $M_X(s)=\infty$ for all $s \neq 0$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Correlation vs. Independence, Confounding & Simpson's Paradox
> Zero linear correlation ($\mathrm{Cov}[X,Y] = 0$) guarantees statistical independence ($X \perp\!\!\!\perp Y$) *only* when $(X,Y)$ are jointly Gaussian; two marginally Gaussian variables can be uncorrelated yet strictly dependent—for example, take $X \sim \mathcal{N}(0,1)$ and $Y = WX$ with $W \sim \mathcal{U}\{-1,1\}$ independent of $X$, giving $Y \sim \mathcal{N}(0,1)$ and $\mathrm{Cov}[X,Y]=0$ while $X^2 = Y^2$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 3.1.3–3.1.5, Ex. 3.5, pp. 78–81, 102). Similarly, observing correlation cannot separate direct causality from common confounders $Z \to X, Z \to Y$, and pooling marginal subgroups can invert correlation signs entirely (*Simpson's paradox*, Figure `fig:probdist:simpson`) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 3.3, p. 80). Always condition on (or intervene on) causal confounders before fitting linear or Gaussian summaries.

> [!note] Source Figure
> *Figure:* Simpson's paradox on the Iris dataset [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 3.1.5, Fig. 3.3, p. 80). Marginal regression pooling all species yields a negative correlation ($R = -0.12$, left), whereas conditioning on each species reveals positive within-group correlations ($R \in \{0.74, 0.53, 0.46\}$, right). (`probabilisticml_kevinmurphy_introduction_p110_fig1.png`)

> [!warning] High-Dimensional Covariance Singularity ($N \le D$) & Gaussian Typical Sets
> Given $N$ samples $\boldsymbol{x}_1, \dots, \boldsymbol{x}_N \in \mathbb{R}^D$, the sample scatter matrix $\boldsymbol{S} = \sum_{n=1}^N (\boldsymbol{x}_n - \bar{\boldsymbol{x}})(\boldsymbol{x}_n - \bar{\boldsymbol{x}})^{\top}$ has rank at most $\min(D, N-1)$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 3.2, pp. 72–76). Whenever $D \ge N$, the MLE covariance $\hat{\boldsymbol{\Sigma}}_{\mathrm{MLE}} = \frac{1}{N}\boldsymbol{S}$ is singular ($\det\hat{\boldsymbol{\Sigma}}_{\mathrm{MLE}} = 0$), its precision $\hat{\boldsymbol{\Lambda}}$ does not exist, and even for $N \approx D$ its condition number explodes ($\kappa \gg 10^{15}$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4.5.2, Fig. 4.4, pp. 122–123). **Remedy:** apply conjugate Inverse-Wishart / Ledoit–Wolf shrinkage $\hat{\boldsymbol{\Sigma}}_{\mathrm{MAP}} = \lambda \mathrm{diag}(\hat{\boldsymbol{\Sigma}}_{\mathrm{MLE}}) + (1-\lambda)\hat{\boldsymbol{\Sigma}}_{\mathrm{MLE}}$ or diagonal/low-rank factor covariance plus Cholesky jitter $\epsilon\boldsymbol{I}_D$. Simultaneously, in high dimension $D \gg 1$, random draws from $\mathcal{N}(\boldsymbol{\mu}, \sigma^2 \boldsymbol{I}_D)$ almost never lie near the density mode $\boldsymbol{\mu}$; because spherical volume elements scale as $r^{D-1} \mathrm{d}r$, probability mass concentrates in a thin "soap-bubble" annulus of radius $\sigma\sqrt{D}$ and thickness $\mathcal{O}(\sigma D^{1/4})$ (Figure `fig:probdist:shell`) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 2.3.1.2, Fig. 2.9, pp. 23–24).

> [!note] Source Figure
> *Figure:* High-dimensional Gaussian concentration of measure [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 2.3.1.2, Fig. 2.9, p. 24). (a) Exponentially decaying density balanced against $r^{D-1}$ shell volume creates a narrow typical set away from the mode. (b) Samples from $\mathcal{N}(\boldsymbol{\mu}, \sigma^2 \boldsymbol{I}_D)$ concentrate at radius $\sigma\sqrt{D}$ within an annulus of width $\mathcal{O}(\sigma D^{1/4})$. (`probabilisticml_kevinmurphy_advancedtopics_p58_fig1.png`)

> [!warning] Canonical Parameter Traps, Zero-Count Boundaries & Numerical Underflow
> Three recurring implementation errors arise in exponential-family and MVN pipelines:
> - **Overcomplete non-identifiability & domain violations:** If components of $\boldsymbol{T}(\boldsymbol{x})$ are linearly dependent (e.g., unconstrained categorical probabilities with $\sum_k T_k(\boldsymbol{x})=1$), $\nabla^2 A(\boldsymbol{\eta})$ is singular and $\boldsymbol{\eta}$ is non-unique; similarly, MVN natural matrix $\boldsymbol{\eta}_2 = -\tfrac{1}{2}\boldsymbol{\Sigma}^{-1}$ must remain strictly negative definite during gradient optimization (parameterize via Cholesky factors $\boldsymbol{L}$ with $\exp(L_{ii})$ on the diagonal) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 2.4.1–2.4.2, pp. 34–39).
> - **Boundary singularities at $\bar{\boldsymbol{t}}_{\mathcal{D}} \in \partial\mathcal{M}$:** When discrete counts contain zeros ($\hat{\mu}_k = 0$), canonical parameters diverge ($\eta_k = \log(\mu_k/\mu_K) \to -\infty$), assigning zero probability to unseen events (the black-swan trap). Conjugate smoothing $\boldsymbol{\tau}_N = \boldsymbol{\tau}_0 + N\bar{\boldsymbol{t}}_{\mathcal{D}}$ with $\boldsymbol{\tau}_0 > \boldsymbol{0}$ strictly forces $\bar{\boldsymbol{t}}_{\mathrm{post}} \in \mathcal{M}^\circ$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4.5.1, pp. 121–122).
> - **MGF divergence & log-space precision:** Heavy-tailed laws such as Student-$t$ and Cauchy possess $M_X(s) = \infty$ for $s \neq 0$ (Figure `fig:probdist:ellipses_pgfplots`b); always substitute the universally bounded characteristic function $\varphi_X(\omega) = \mathbb{E}[e^{i\omega X}]$ ($\lvert e^{i\omega X} \rvert = 1$) [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Thm. 6.7, pp. 331–332). Likewise, never evaluate `(eq:probdist:mvn_pdf)` directly via $\det(\boldsymbol{\Sigma})$ in high $D$ (risking overflow/underflow in IEEE-754 float64); evaluate $\log p(\boldsymbol{x}) = -\frac{D}{2}\log(2\pi) - \sum_{i=1}^D \log L_{ii} - \frac{1}{2}\lVert \boldsymbol{L}^{-1}(\boldsymbol{x}-\boldsymbol{\mu}) \rVert_2^2$ via Cholesky forward substitution.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[Exponential Families]], [[Mahalanobis Distance]], [[Multivariate Gaussian & Schur Complement]]
- **Prerequisites (Directed $\leftarrow$):** [[Eigendecomposition & Spectral Theorem]], [[Inner Product & Norms]], [[Multivariate Gaussian & Schur Complement]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Exponential Families]], [[Gaussian Mixture Models (GMM)]], [[Gaussian Processes]], [[Generalized Linear Models (GLM)]], [[Kalman Filter]], [[Mahalanobis Distance]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Exponential Families]], [[Gaussian Mixture Models (GMM)]], [[Gaussian Processes]], [[Generalized Linear Models (GLM)]], [[Graphical Models (BN & MRF)]], [[Inner Product & Norms]], [[Kalman Filter]], [[Mahalanobis Distance]], [[Multivariate Gaussian & Schur Complement]], [[Variational Inference & ELBO]]
- **Synthesized Sources:** [[Source - Foundations]], [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
