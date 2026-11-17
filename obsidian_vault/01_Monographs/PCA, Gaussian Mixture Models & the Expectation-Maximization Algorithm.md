---
type: monograph
domain: "D02"
parent_topic: "[[PT 02 - Latent Variables, Graphical Models, VI & MCMC Sampling]]"
sub_topic: "[[ST 02 - PCA, Factor Analysis, Mixture Models & EM]]"
cheatsheet: "[[Cheatsheet - PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]"
part_of: 
  - "[[ST 02 - PCA, Factor Analysis, Mixture Models & EM]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
concepts: 
  - "[[EM Algorithm]]"
  - "[[Gaussian Mixture Models (GMM)]]"
  - "[[Principal Component Analysis (PCA)]]"
prerequisites: 
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Gaussian Mixture Models (GMM)]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
  - "[[Singular Value Decomposition (SVD)]]"
  - "[[Variational Inference & ELBO]]"
enables: 
  - "[[EM Algorithm]]"
related: 
  - "[[Autoencoders]]"
  - "[[EM Algorithm]]"
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Gaussian Mixture Models (GMM)]]"
  - "[[Low-Rank Approximation (Eckart-Young)]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
  - "[[Singular Value Decomposition (SVD)]]"
  - "[[Variational Inference & ELBO]]"
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - monograph
  - domain-02
---

# PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm

> [!abstract] Monograph Navigation
> - **Domain:** [[D02 - Classical, Kernel & Probabilistic Machine Learning|Classical, Kernel & Probabilistic Machine Learning]]
> - **Parent / Sub-Topic:** [[PT 02 - Latent Variables, Graphical Models, VI & MCMC Sampling|Latent Variables, Graphical Models, VI & MCMC Sampling]] → [[ST 02 - PCA, Factor Analysis, Mixture Models & EM|PCA, Factor Analysis, Mixture Models & EM]]
> - **Visual Cheatsheet:** [[Cheatsheet - PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]
> - **Description:** Unifies maximum-variance and minimum-reconstruction PCA, probabilistic PCA/Factor Analysis, K-means as hard-EM, Gaussian Mixture Models, and the general E-step/M-step monotonic ELBO proof.

## Problem Formulation & Unified Notation

Unsupervised representation learning and density estimation address two dual questions for an unlabeled dataset $\mathcal{D} = \{\boldsymbol{x}_n\}_{n=1}^N$ with $\boldsymbol{x}_n \in \mathbb{R}^D$: (i) what low-dimensional continuous latent coordinates $\boldsymbol{z}_n \in \mathbb{R}^M$ ($M < D$) best preserve geometric structure, and (ii) what discrete latent cluster assignments $z_n \in \{1,\dots,K\}$ best explain multimodal density? Stacking observations row-wise yields the data matrix $\boldsymbol{X} = [\boldsymbol{x}_1, \dots, \boldsymbol{x}_N]^{\top} \in \mathbb{R}^{N \times D}$ with sample mean $\bar{\boldsymbol{x}} = \frac{1}{N}\sum_{n=1}^N \boldsymbol{x}_n$, centered matrix $\tilde{\boldsymbol{X}} = \boldsymbol{X} - \boldsymbol{1} \bar{\boldsymbol{x}}^{\top}$, and empirical covariance

$$
\boldsymbol{S} = \frac{1}{N}\tilde{\boldsymbol{X}}^{\top}\tilde{\boldsymbol{X}} = \frac{1}{N}\sum_{n=1}^N (\boldsymbol{x}_n - \bar{\boldsymbol{x}})(\boldsymbol{x}_n - \bar{\boldsymbol{x}})^{\top} = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top} \in \mathbb{R}^{D \times D},
$$

where $\boldsymbol{U} = [\boldsymbol{u}_1, \dots, \boldsymbol{u}_D] \in \mathbb{R}^{D \times D}$ is orthogonal ($\boldsymbol{U}^{\top}\boldsymbol{U} = \boldsymbol{I}_D$) and $\boldsymbol{\Lambda} = \mathrm{diag}(\lambda_1, \dots, \lambda_D)$ orders eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 10, PDF p. 327)[[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 16, p. 512). In probabilistic latent-variable models, observed data $\boldsymbol{X}$ are coupled to unobserved variables $\boldsymbol{Z}$ via a joint distribution $p(\boldsymbol{X}, \boldsymbol{Z} \mid \boldsymbol{\theta})$, and parameter estimation seeks $\arg\max_{\boldsymbol{\theta}} \log p(\boldsymbol{X} \mid \boldsymbol{\theta}) = \arg\max_{\boldsymbol{\theta}} \log \int p(\boldsymbol{X}, \boldsymbol{Z} \mid \boldsymbol{\theta}) \mathrm{d}\boldsymbol{Z}$ (with summation replacing integration for discrete $\boldsymbol{Z}$).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Source texts differ in latent-dimension symbols, loading-matrix letters, data orientation, and responsibility notation. We standardize on row-major $\boldsymbol{X} \in \mathbb{R}^{N \times D}$, latent dimension $M$, and $K$ mixture components:
> 
>
>
>
> | **Concept** | **Unified** | **Deisenroth** | **Bishop** | **Murphy** | **Tiep Vu / Notes** |
> | — | — | — | — | — | — |
> | Latent dim. | $M$ | $M$ | $M$ | $L$ | $K$ (PCA) vs. $K$ clusters |
> | Data matrix | $\boldsymbol{X} \in \mathbb{R}^{N \times D}$ | $\{\boldsymbol{x}_n\}$ | $\boldsymbol{X} \in \mathbb{R}^{N \times D}$ | $\boldsymbol{X} \in \mathbb{R}^{N \times D}$ | $\boldsymbol{X} \in \mathbb{R}^{D \times N}$ (col-major) |
> | Basis / Loadings | $\boldsymbol{B}, \boldsymbol{W} \in \mathbb{R}^{D \times M}$ | $\boldsymbol{B}$ | $\boldsymbol{U}_M, \boldsymbol{W}$ | $\boldsymbol{U}_L, \boldsymbol{W}$ | $\boldsymbol{U}_K \in \mathbb{R}^{D \times K}$ |
> | Covariance | $\boldsymbol{S}$ | $\boldsymbol{S}$ | $\boldsymbol{S}$ | $\hat{\boldsymbol{\Sigma}}$ | $\boldsymbol{S} = \frac{1}{N}\bar{\boldsymbol{X}}\bar{\boldsymbol{X}}^{\top}$ |
> | Responsibility | $r_{nk}$ | $r_{nk}$ | $\gamma(z_{nk})$ | $r_{nk}$ | Posterior $p(z_n{=}k \mid \boldsymbol{x}_n)$ |
> | Noise cov. | $\sigma^2\boldsymbol{I}_D$, $\boldsymbol{\Psi}$ | $\sigma^2\boldsymbol{I}_D$ | $\sigma^2\boldsymbol{I}_D$, $\boldsymbol{\Psi}$ | $\sigma^2\boldsymbol{I}_D$, $\boldsymbol{\Psi}$ | Isotropic (PPCA) vs. diag. (FA) |

## Core Mathematical Derivations & Theorems

### Equivalence of Maximum-Variance, Minimum-Error, and SVD PCA

Principal Component Analysis (PCA) admits two distinct geometric derivations—maximizing projected variance [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 10.2, PDF p. 327) and minimizing orthogonal reconstruction error [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 20.1.2, PDF p. 689)—that yield identical principal subspaces and coincide with the truncated Singular Value Decomposition (SVD).

> [!important] Equivalence of Maximum-Variance and Minimum-Reconstruction PCA
> Assume centered data $\tilde{\boldsymbol{x}}_n = \boldsymbol{x}_n - \bar{\boldsymbol{x}}$ with covariance $\boldsymbol{S} = \frac{1}{N}\sum_{n=1}^N \tilde{\boldsymbol{x}}_n\tilde{\boldsymbol{x}}_n^{\top}$.
> - **Maximum-Variance Formulation:** Seeking a unit direction $\boldsymbol{b}_1 \in \mathbb{R}^D$ ($\lVert \boldsymbol{b}_1 \rVert_2 = 1$) maximizing projected variance $V_1 = \frac{1}{N}\sum_{n=1}^N (\boldsymbol{b}_1^{\top} \tilde{\boldsymbol{x}}_n)^2 = \boldsymbol{b}_1^{\top} \boldsymbol{S} \boldsymbol{b}_1$ leads to the Lagrangian $\mathcal{L}(\boldsymbol{b}_1, \lambda_1) = \boldsymbol{b}_1^{\top} \boldsymbol{S} \boldsymbol{b}_1 + \lambda_1(1 - \boldsymbol{b}_1^{\top} \boldsymbol{b}_1)$. Setting $\nabla_{\boldsymbol{b}_1}\mathcal{L} = \boldsymbol{0}$ gives
>
> $$
> \boldsymbol{S}\boldsymbol{b}_1 = \lambda_1 \boldsymbol{b}_1, \qquad V_1 = \boldsymbol{b}_1^{\top} \boldsymbol{S} \boldsymbol{b}_1 = \lambda_1.
> $$
>
>   By induction with orthogonality constraints $\boldsymbol{b}_m^{\top} \boldsymbol{b}_j = 0$ ($j < m$), the $M$-dimensional variance-maximizing basis is $\boldsymbol{B} = \boldsymbol{U}_M = [\boldsymbol{u}_1, \dots, \boldsymbol{u}_M] \in \mathbb{R}^{D \times M}$ spanning the top-$M$ eigenspace of $\boldsymbol{S}$, capturing total variance $\sum_{m=1}^M \lambda_m$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 10.2, PDF p. 328).
> - **Minimum-Reconstruction Formulation:** Seeking an orthonormal basis $\boldsymbol{B} \in \mathbb{R}^{D \times M}$ ($\boldsymbol{B}^{\top}\boldsymbol{B} = \boldsymbol{I}_M$) and latent codes $\{\boldsymbol{z}_n\}_{n=1}^N \subset \mathbb{R}^M$ minimizing mean squared error
>
> $$
> J_M(\boldsymbol{B}, \boldsymbol{Z}) = \frac{1}{N}\sum_{n=1}^N \lVert \tilde{\boldsymbol{x}}_n - \boldsymbol{B}\boldsymbol{z}_n \rVert_2^2 = \frac{1}{N}\lVert \tilde{\boldsymbol{X}} - \boldsymbol{Z}\boldsymbol{B}^{\top} \rVert_F^2,
> $$
>
>   differentiating with respect to $\boldsymbol{z}_n$ gives $\frac{\partial J_M}{\partial \boldsymbol{z}_n} = -\frac{2}{N}\boldsymbol{B}^{\top}(\tilde{\boldsymbol{x}}_n - \boldsymbol{B}\boldsymbol{z}_n) = \boldsymbol{0} \implies \boldsymbol{z}_n^\star = \boldsymbol{B}^{\top}\tilde{\boldsymbol{x}}_n$ (since $\boldsymbol{B}^{\top}\boldsymbol{B} = \boldsymbol{I}_M$). Substituting $\hat{\boldsymbol{x}}_n = \boldsymbol{B}\boldsymbol{B}^{\top}\tilde{\boldsymbol{x}}_n$ back into `(eq:em:pca_recon)` via the Pythagorean theorem yields
>
> $$
> J_M(\boldsymbol{B}) = \frac{1}{N}\sum_{n=1}^N \tilde{\boldsymbol{x}}_n^{\top}(\boldsymbol{I}_D - \boldsymbol{B}\boldsymbol{B}^{\top})\tilde{\boldsymbol{x}}_n = \mathrm{Tr}(\boldsymbol{S}) - \mathrm{Tr}(\boldsymbol{B}^{\top}\boldsymbol{S}\boldsymbol{B}) = \sum_{i=M+1}^D \lambda_i,
> $$
>
>   which is minimized identically at $\boldsymbol{B} = \boldsymbol{U}_M$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 16.1.2, p. 513)[[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 20.1.2, PDF p. 690).
> - **SVD Equivalence:** Let $\tilde{\boldsymbol{X}} = \boldsymbol{U}_X \boldsymbol{\Sigma}_X \boldsymbol{V}_X^{\top}$ be the thin SVD. Then $\boldsymbol{S} = \frac{1}{N}\tilde{\boldsymbol{X}}^{\top}\tilde{\boldsymbol{X}} = \boldsymbol{V}_X \left(\frac{1}{N}\boldsymbol{\Sigma}_X^2\right) \boldsymbol{V}_X^{\top}$. Hence the right singular vectors $\boldsymbol{V}_{X,1:M}$ equal $\boldsymbol{U}_M$, the eigenvalues satisfy $\lambda_k = \sigma_k^2 / N$, the scores are $\boldsymbol{Z} = \tilde{\boldsymbol{X}}\boldsymbol{V}_{X,1:M} = \boldsymbol{U}_{X,1:M}\boldsymbol{\Sigma}_{X,1:M}$, and $\hat{\boldsymbol{X}} = \boldsymbol{U}_{X,1:M}\boldsymbol{\Sigma}_{X,1:M}\boldsymbol{V}_{X,1:M}^{\top}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 20.1.3, PDF p. 693)[[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 21, PDF p. 271).

> [!tip] Unsupervised Variance (PCA) vs. Supervised Discriminability (Fisher LDA)
> PCA maximizes total marginal variance $\boldsymbol{b}^{\top}\boldsymbol{S}\boldsymbol{b}$ without class labels. When between-class separation lies along a low-variance direction while within-class noise dominates the leading principal axis, PCA merges classes upon projection. Fisher's Linear Discriminant Analysis (LDA) instead maximizes the Rayleigh quotient $J(\boldsymbol{w}) = \frac{\boldsymbol{w}^{\top} \boldsymbol{S}_B \boldsymbol{w}}{\boldsymbol{w}^{\top} \boldsymbol{S}_W \boldsymbol{w}}$ (between-class scatter over within-class scatter), trading total variance for class separability [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 22, PDF p. 280–282)[[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 9.2, PDF p. 359).

### Probabilistic PCA, Factor Analysis, and the $\sigma^2 \to 0$ Limit

Conventional PCA lacks a likelihood function, cannot generate samples, and assigns zero reconstruction error anywhere inside the principal subspace regardless of distance from the training manifold [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 16.2, p. 520). Probabilistic PCA (PPCA) and Factor Analysis (FA) resolve this via a linear-Gaussian latent-variable model.

> [!important] PPCA Marginal, Woodbury Posterior, MLE, and Zero-Noise PCA Limit
> Define the generative process $\boldsymbol{z}_n \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_M)$ and $\boldsymbol{x}_n \mid \boldsymbol{z}_n \sim \mathcal{N}(\boldsymbol{W}\boldsymbol{z}_n + \boldsymbol{\mu}, \boldsymbol{\Psi})$, where $\boldsymbol{W} \in \mathbb{R}^{D \times M}$ is the factor loading matrix. Setting $\boldsymbol{\Psi} = \mathrm{diag}(\psi_1,\dots,\psi_D)$ gives **Factor Analysis**; restricting $\boldsymbol{\Psi} = \sigma^2 \boldsymbol{I}_D$ gives **Probabilistic PCA** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 28.3.2, PDF p. 978).
> - **Marginal & Posterior:** Integrating out $\boldsymbol{z}_n$ yields $p(\boldsymbol{x}_n) = \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}, \boldsymbol{C})$ with $\boldsymbol{C} = \boldsymbol{W}\boldsymbol{W}^{\top} + \sigma^2 \boldsymbol{I}_D$. Applying the Woodbury matrix identity reduces $\boldsymbol{C}^{-1}$ from $\mathcal{O}(D^3)$ to $\mathcal{O}(M^3)$:
>
> $$
> \boldsymbol{C}^{-1} = \sigma^{-2}\boldsymbol{I}_D - \sigma^{-2}\boldsymbol{W}\boldsymbol{M}^{-1}\boldsymbol{W}^{\top}, \qquad \boldsymbol{M} = \boldsymbol{W}^{\top}\boldsymbol{W} + \sigma^2\boldsymbol{I}_M \in \mathbb{R}^{M \times M},
> $$
>
>   and Bayes' rule for linear-Gaussian models yields the exact latent posterior [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 16.2.2, p. 523):
>
> $$
> p(\boldsymbol{z}_n \mid \boldsymbol{x}_n) = \mathcal{N}\big(\boldsymbol{z}_n  \big|  \boldsymbol{M}^{-1}\boldsymbol{W}^{\top}(\boldsymbol{x}_n - \boldsymbol{\mu}),  \sigma^2\boldsymbol{M}^{-1}\big).
> $$
> - **Closed-Form MLE (Tipping–Bishop):** Maximizing $\log p(\boldsymbol{X} \mid \boldsymbol{\mu}, \boldsymbol{W}, \sigma^2) = -\frac{N}{2}\big[D\log(2\pi) + \log\lvert \boldsymbol{C} \rvert + \mathrm{Tr}(\boldsymbol{C}^{-1}\boldsymbol{S})\big]$ yields $\hat{\boldsymbol{\mu}}_{\mathrm{ML}} = \bar{\boldsymbol{x}}$ and stationary points [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 16.2.3, p. 524)[[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 28.3.2, PDF p. 979):
>
> $$
> \boldsymbol{W}_{\mathrm{ML}} = \boldsymbol{U}_M(\boldsymbol{\Lambda}_M - \sigma^2\boldsymbol{I}_M)^{1/2}\boldsymbol{R}, \qquad \sigma^2_{\mathrm{ML}} = \frac{1}{D - M}\sum_{i=M+1}^D \lambda_i,
> $$
>
>   where $\boldsymbol{R} \in \mathbb{R}^{M \times M}$ is any orthogonal matrix ($\boldsymbol{R}\boldsymbol{R}^{\top} = \boldsymbol{I}_M$). All other eigenvector subsets are saddle points.
> - **Isotropic Limit $\sigma^2 \to 0$:** As $\sigma^2 \to 0$, $\boldsymbol{W}_{\mathrm{ML}} \to \boldsymbol{U}_M \boldsymbol{\Lambda}_M^{1/2}\boldsymbol{R}$, the marginal covariance becomes the rank-$M$ approximation $\boldsymbol{C} \to \boldsymbol{U}_M\boldsymbol{\Lambda}_M\boldsymbol{U}_M^{\top} = \boldsymbol{S}_M$, and the posterior covariance $\sigma^2\boldsymbol{M}^{-1} \to \boldsymbol{0}$ collapses onto a point mass at the orthogonal projection $\mathbb{E}[\boldsymbol{z}_n \mid \boldsymbol{x}_n] \to (\boldsymbol{W}^{\top}\boldsymbol{W})^{-1}\boldsymbol{W}^{\top}(\boldsymbol{x}_n - \bar{\boldsymbol{x}})$, recovering standard PCA [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 28.3.2.2, PDF p. 979).

### Gaussian Mixture Models, Non-Convexity, and $K$-Means as Hard EM

{
A $K$-component Gaussian Mixture Model (GMM) introduces a discrete latent indicator $z_n \in \{1,\dots,K\}$ with prior $\pi_k = p(z_n=k)$ ($\sum_{k=1}^K \pi_k = 1$, $\pi_k \ge 0$) and component likelihood $p(\boldsymbol{x}_n \mid z_n=k) = \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$. Marginalizing $z_n$ yields the incomplete-data log-likelihood [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 11.1, PDF p. 355) and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 28.2.1, PDF p. 964):

}

$$
\log p(\boldsymbol{X} \mid \boldsymbol{\theta}) = \sum_{n=1}^N \log \left(\sum_{k=1}^K \pi_k \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\right), \qquad \boldsymbol{\theta} = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^K.
$$

Because the logarithm sits outside the sum over $k$, `(eq:em:gmm_ll)` is non-convex and admits no closed-form global maximizer. By Bayes' rule, the posterior responsibility of component $k$ for observation $\boldsymbol{x}_n$ is

$$
r_{nk} := p(z_n = k \mid \boldsymbol{x}_n, \boldsymbol{\theta}) = \frac{\pi_k \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}.
$$

Restricting $\boldsymbol{\Sigma}_k = \epsilon \boldsymbol{I}_D$ and $\pi_k = 1/K$ for all $k$, the exponent in $\mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \epsilon\boldsymbol{I}_D)$ scales as $-\frac{1}{2\epsilon}\lVert \boldsymbol{x}_n - \boldsymbol{\mu}_k \rVert_2^2$. In the zero-variance limit $\epsilon \to 0^+$, the softmax in `(eq:em:resp)` sharpens to a hard indicator $r_{nk} \to \mathbb{I}\big[k = \arg\min_j \lVert \boldsymbol{x}_n - \boldsymbol{\mu}_j \rVert_2^2\big]$, and the expected complete-data log-likelihood reduces to minimizing the $K$-means distortion $J = \sum_{n=1}^N \sum_{k=1}^K r_{nk} \lVert \boldsymbol{x}_n - \boldsymbol{\mu}_k \rVert_2^2$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 15.3.2, p. 495). Thus $K$-means is precisely the hard-assignment limit of GMM-EM.

### General EM Algorithm: ELBO Decomposition & Monotone Ascent

The Expectation-Maximization (EM) algorithm maximizes $\log p(\boldsymbol{X} \mid \boldsymbol{\theta})$ iteratively by constructing and maximizing a tight variational lower bound (ELBO) [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 15.4, pp. 500–503)[[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 11.3, PDF p. 366).

> [!important] ELBO Decomposition and Monotonic Log-Likelihood Ascent
> For any variational distribution $q(\boldsymbol{Z})$ over the latent variables $\boldsymbol{Z}$, the marginal log-likelihood decomposes identically as
>
> $$
> \log p(\boldsymbol{X} \mid \boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + D_{\mathrm{KL}}(q(\boldsymbol{Z}) \parallel p(\boldsymbol{Z} \mid \boldsymbol{X}, \boldsymbol{\theta})),
> $$
>
> where the Evidence Lower Bound (ELBO) $\mathcal{L}(q, \boldsymbol{\theta})$ and Kullback–Leibler divergence are
>
> $$
> \begin{aligned}
> \mathcal{L}(q, \boldsymbol{\theta}) &= \sum_{\boldsymbol{Z}} q(\boldsymbol{Z}) \log \frac{p(\boldsymbol{X}, \boldsymbol{Z} \mid \boldsymbol{\theta})}{q(\boldsymbol{Z})} = \mathbb{E}_{q(\boldsymbol{Z})}[\log p(\boldsymbol{X}, \boldsymbol{Z} \mid \boldsymbol{\theta})] + \mathrm{H}(q),  \\
>   D_{\mathrm{KL}}(q(\boldsymbol{Z}) \parallel p(\boldsymbol{Z} \mid \boldsymbol{X}, \boldsymbol{\theta})) &= -\sum_{\boldsymbol{Z}} q(\boldsymbol{Z}) \log \frac{p(\boldsymbol{Z} \mid \boldsymbol{X}, \boldsymbol{\theta})}{q(\boldsymbol{Z})} \ge 0,
> \end{aligned}
> $$
>
> (with $\sum_{\boldsymbol{Z}}$ replaced by $\int \mathrm{d}\boldsymbol{Z}$ for continuous $\boldsymbol{Z}$). Non-negativity of KL divergence implies $\log p(\boldsymbol{X} \mid \boldsymbol{\theta}) \ge \mathcal{L}(q, \boldsymbol{\theta})$ for all $q$ and $\boldsymbol{\theta}$ (equivalently by Jensen's inequality). Alternating two coordinate-ascent steps from iterate $\boldsymbol{\theta}^{(t)}$:
> - **E-step:** Hold $\boldsymbol{\theta}^{(t)}$ fixed and maximize $\mathcal{L}(q, \boldsymbol{\theta}^{(t)})$ over $q$. Since $\log p(\boldsymbol{X} \mid \boldsymbol{\theta}^{(t)})$ is independent of $q$, maximizing $\mathcal{L}$ minimizes $D_{\mathrm{KL}}(q(\boldsymbol{Z}) \parallel p(\boldsymbol{Z} \mid \boldsymbol{X}, \boldsymbol{\theta}^{(t)}))$ to zero at
>
> $$
> q^{(t+1)}(\boldsymbol{Z}) = p(\boldsymbol{Z} \mid \boldsymbol{X}, \boldsymbol{\theta}^{(t)}), \qquad \mathcal{L}(q^{(t+1)}, \boldsymbol{\theta}^{(t)}) = \log p(\boldsymbol{X} \mid \boldsymbol{\theta}^{(t)}).
> $$
> - **M-step:** Hold $q^{(t+1)}$ fixed and maximize $\mathcal{L}(q^{(t+1)}, \boldsymbol{\theta})$ over $\boldsymbol{\theta}$. Dropping the entropy term $\mathrm{H}(q^{(t+1)})$ (independent of $\boldsymbol{\theta}$) reduces this to maximizing the expected complete-data log-likelihood $Q(\boldsymbol{\theta}; \boldsymbol{\theta}^{(t)})$:
>
> $$
> \boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta}} Q(\boldsymbol{\theta}; \boldsymbol{\theta}^{(t)}), \qquad Q(\boldsymbol{\theta}; \boldsymbol{\theta}^{(t)}) := \mathbb{E}_{p(\boldsymbol{Z} \mid \boldsymbol{X}, \boldsymbol{\theta}^{(t)})}[\log p(\boldsymbol{X}, \boldsymbol{Z} \mid \boldsymbol{\theta})].
> $$
>
>
> **Proof of Monotonic Ascent:** Combining `(eq:em:elbo_decomp)`, `(eq:em:estep_general)`, and `(eq:em:mstep_general)` gives
>
> $$
> \begin{aligned}
> \log p(\boldsymbol{X} \mid \boldsymbol{\theta}^{(t+1)}) &= \mathcal{L}(q^{(t+1)}, \boldsymbol{\theta}^{(t+1)}) + \underbrace{D_{\mathrm{KL}}(q^{(t+1)} \parallel p(\cdot \mid \boldsymbol{X}, \boldsymbol{\theta}^{(t+1)}))}_{\ge  0}  \\
>   &\ge \mathcal{L}(q^{(t+1)}, \boldsymbol{\theta}^{(t+1)}) \ge \mathcal{L}(q^{(t+1)}, \boldsymbol{\theta}^{(t)}) = \log p(\boldsymbol{X} \mid \boldsymbol{\theta}^{(t)}).
> \end{aligned}
> $$
>
> Moreover, at $\boldsymbol{\theta} = \boldsymbol{\theta}^{(t)}$, $\mathcal{L}(q^{(t+1)}, \boldsymbol{\theta})$ makes tangential contact with $\log p(\boldsymbol{X} \mid \boldsymbol{\theta})$ (matching both value and gradient) [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 15.4.1, p. 502).

### Closed-Form E/M Updates for GMM, PPCA/FA, and Kernel PCA

{

**GMM EM Updates..**  For i.i.d. data, $Q(\boldsymbol{\theta};\boldsymbol{\theta}^{(t)}) = \sum_{n=1}^N \sum_{k=1}^K r_{nk}^{(t)} \big[\log \pi_k + \log \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\big]$. Differentiating and applying a Lagrange multiplier for $\sum_k \pi_k = 1$ yields [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 11.3, PDF p. 367) and [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 15.3.1, p. 494):

}

$$
\begin{aligned}
N_k &= \sum_{n=1}^N r_{nk}^{(t)}, & \boldsymbol{\mu}_k^{(t+1)} &= \frac{1}{N_k}\sum_{n=1}^N r_{nk}^{(t)} \boldsymbol{x}_n,  \\
  \boldsymbol{\Sigma}_k^{(t+1)} &= \frac{1}{N_k}\sum_{n=1}^N r_{nk}^{(t)}(\boldsymbol{x}_n - \boldsymbol{\mu}_k^{(t+1)})(\boldsymbol{x}_n - \boldsymbol{\mu}_k^{(t+1)})^{\top}, & \pi_k^{(t+1)} &= \frac{N_k}{N}.
\end{aligned}
$$

**PPCA & EM-PCA Updates..**  For PPCA, the E-step computes the posterior moments from `(eq:em:ppca_post)`:

$$
\mathbb{E}[\boldsymbol{z}_n] = \boldsymbol{M}^{-1}\boldsymbol{W}^{\top}(\boldsymbol{x}_n - \bar{\boldsymbol{x}}), \qquad \mathbb{E}[\boldsymbol{z}_n\boldsymbol{z}_n^{\top}] = \sigma^2\boldsymbol{M}^{-1} + \mathbb{E}[\boldsymbol{z}_n]\mathbb{E}[\boldsymbol{z}_n]^{\top},
$$

and the M-step updates $\boldsymbol{W}_{\mathrm{new}}$ and $\sigma^2_{\mathrm{new}}$ in $\mathcal{O}(NDM)$ time without forming the $D\times D$ covariance matrix [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 16.3.1, pp. 532–533):

$$
\begin{aligned}
\boldsymbol{W}_{\mathrm{new}} &= \left(\sum_{n=1}^N (\boldsymbol{x}_n - \bar{\boldsymbol{x}})\mathbb{E}[\boldsymbol{z}_n]^{\top}\right)\left(\sum_{n=1}^N \mathbb{E}[\boldsymbol{z}_n\boldsymbol{z}_n^{\top}]\right)^{-1},  \\
  \sigma^2_{\mathrm{new}} &= \frac{1}{ND}\sum_{n=1}^N \Big\{\lVert \boldsymbol{x}_n - \bar{\boldsymbol{x}} \rVert_2^2 - 2\mathbb{E}[\boldsymbol{z}_n]^{\top}\boldsymbol{W}_{\mathrm{new}}^{\top}(\boldsymbol{x}_n - \bar{\boldsymbol{x}}) + \mathrm{Tr}\big(\mathbb{E}[\boldsymbol{z}_n\boldsymbol{z}_n^{\top}]\boldsymbol{W}_{\mathrm{new}}^{\top}\boldsymbol{W}_{\mathrm{new}}\big)\Big\}.
\end{aligned}
$$

In the zero-noise limit $\sigma^2 \to 0$, `(eq:em:ppca_estep)` and `(eq:em:ppca_mstep)` reduce to Roweis's **EM-PCA** algorithm; collecting $\mathbb{E}[\boldsymbol{z}_n]$ into $\boldsymbol{\Omega} \in \mathbb{R}^{M \times N}$ gives [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 16.3.2, p. 534):

$$
\text{E-step: } \boldsymbol{\Omega} = (\boldsymbol{W}^{\top}\boldsymbol{W})^{-1}\boldsymbol{W}^{\top}\tilde{\boldsymbol{X}}^{\top}, \qquad \text{M-step: } \boldsymbol{W}_{\mathrm{new}} = \tilde{\boldsymbol{X}}^{\top}\boldsymbol{\Omega}^{\top}(\boldsymbol{\Omega}\boldsymbol{\Omega}^{\top})^{-1}.
$$

{

**Dual Gram Trick & Kernel PCA..**  When $N \ll D$, forming $\boldsymbol{S} \in \mathbb{R}^{D \times D}$ costs $\mathcal{O}(ND^2)$ and eigendecomposition costs $\mathcal{O}(D^3)$. Solving the $N \times N$ Gram eigenproblem $\frac{1}{N}\tilde{\boldsymbol{X}}\tilde{\boldsymbol{X}}^{\top} \boldsymbol{v}_i = \lambda_i \boldsymbol{v}_i$ ($\lVert \boldsymbol{v}_i \rVert_2 = 1$) instead yields unit eigenvectors of $\boldsymbol{S}$ via $\boldsymbol{u}_i = (N\lambda_i)^{-1/2}\tilde{\boldsymbol{X}}^{\top}\boldsymbol{v}_i$ in $\mathcal{O}(N^3)$ time [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 16.1.5, p. 519) and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 20.1.3.2, PDF p. 693). Replacing inner products $\langle \boldsymbol{x}_i, \boldsymbol{x}_j \rangle$ with a positive-definite kernel $k(\boldsymbol{x}_i, \boldsymbol{x}_j) = \langle \phi(\boldsymbol{x}_i), \phi(\boldsymbol{x}_j) \rangle_{\mathcal{H}}$ defines **Kernel PCA** [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 10.8, PDF p. 353): centering $\boldsymbol{K}_{ij} = k(\boldsymbol{x}_i, \boldsymbol{x}_j)$ via $\tilde{\boldsymbol{K}} = \boldsymbol{H}\boldsymbol{K}\boldsymbol{H}$ ($\boldsymbol{H} = \boldsymbol{I}_N - \frac{1}{N}\boldsymbol{1}\boldsymbol{1}^{\top}$), we solve $\tilde{\boldsymbol{K}}\boldsymbol{\alpha}_m = N\lambda_m \boldsymbol{\alpha}_m$ with $\lVert \boldsymbol{\alpha}_m \rVert_2^2 = \frac{1}{N\lambda_m}$ and project $\boldsymbol{x}$ via $z_m(\boldsymbol{x}) = \sum_{n=1}^N \alpha_{mn}\tilde{k}(\boldsymbol{x}_n, \boldsymbol{x})$.

}

## Algorithmic Implementation & Comparative Synthesis

{
Practical GMM-EM combines three safeguards against numerical underflow, poor local optima, and covariance collapse: log-sum-exp stabilization of responsibilities [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 28.2.5, PDF p. 973), $k$-means++ seeding [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 21.3.4, PDF p. 755), and diagonal covariance regularization $\epsilon\boldsymbol{I}_D$.

}

> [!example] Algorithm: Log-Sum-Exp Stabilized EM for Gaussian Mixture Models
> **Input:** Data $\boldsymbol{X} = \{\boldsymbol{x}_n\}_{n=1}^N \subset \mathbb{R}^D$, components $K$, floor $\epsilon > 0$, tolerance $\tau > 0$
> **Output:** Parameters $\boldsymbol{\theta} = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^K$ and responsibilities $\{r_{nk}\}$
> Initialize $\{\boldsymbol{\mu}_k^{(0)}\}_{k=1}^K$ via $k$-means++; set $\pi_k^{(0)} \leftarrow 1/K$, $\boldsymbol{\Sigma}_k^{(0)} \leftarrow \mathrm{Cov}(\boldsymbol{X}) + \epsilon\boldsymbol{I}_D$
> $t \leftarrow 0$, $\ell^{(0)} \leftarrow -\infty$
> **repeat until** $\lvert \ell^{(t)} - \ell^{(t-1)} \rvert < \tau$:
>
>   **E-step (Log-Sum-Exp):** For each $n \in \{1,\dots,N\}$ and $k \in \{1,\dots,K\}$
>   \quad $\beta_{nk} \leftarrow \log \pi_k^{(t)} + \log \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})$
>   \quad $m_n \leftarrow \max_j \beta_{nj}$, \quad $\log Z_n \leftarrow m_n + \log \sum_{j=1}^K \exp(\beta_{nj} - m_n)$
>   \quad $r_{nk} \leftarrow \exp(\beta_{nk} - \log Z_n)$
>   $\ell^{(t+1)} \leftarrow \sum_{n=1}^N \log Z_n$
>   **M-step:** For each $k \in \{1,\dots,K\}$
>   \quad $N_k \leftarrow \sum_{n=1}^N r_{nk}$, \quad $\pi_k^{(t+1)} \leftarrow N_k / N$
>   \quad $\boldsymbol{\mu}_k^{(t+1)} \leftarrow \frac{1}{N_k}\sum_{n=1}^N r_{nk}\boldsymbol{x}_n$
>   \quad $\boldsymbol{\Sigma}_k^{(t+1)} \leftarrow \frac{1}{N_k}\sum_{n=1}^N r_{nk}(\boldsymbol{x}_n - \boldsymbol{\mu}_k^{(t+1)})(\boldsymbol{x}_n - \boldsymbol{\mu}_k^{(t+1)})^{\top} + \epsilon\boldsymbol{I}_D$
>   $t \leftarrow t + 1$

**Table:** Computational complexity and structural comparison across dimensionality reduction and mixture models ($N$: samples, $D$: ambient dim., $M$: latent dim., $K$: clusters, $T$: EM/iter steps).

| **Method** | **Time Complexity** | **Space** | **Latent / Noise** | **Key Trade-off** |
| — | — | — | — | — |
| PCA (Cov. Eig) | $\mathcal{O}(ND^2 + D^3)$ | $\mathcal{O}(D^2)$ | Deterministic / None | Exact global optimum; infeasible for huge $D$ |
| PCA (Dual Gram) | $\mathcal{O}(N^2 D + N^3)$ | $\mathcal{O}(N^2)$ | Deterministic / None | Optimal when $N \ll D$; enables Kernel PCA |
| Randomized SVD | $\mathcal{O}(NDM + M^3)$ | $\mathcal{O}(NM + DM)$ | Deterministic / None | Fast top-$M$ subspace approximation |
| EM-PCA / PPCA | $\mathcal{O}(T N D M)$ | $\mathcal{O}(NM + DM)$ | Cont. / Isotropic $\sigma^2\boldsymbol{I}$ | Streaming-friendly; handles missing data |
| Factor Analysis | $\mathcal{O}(T(NDM + M^3))$ | $\mathcal{O}(NM + DM)$ | Cont. / Diagonal $\boldsymbol{\Psi}$ | Scale-invariant; no closed-form MLE |
| $K$-Means | $\mathcal{O}(T N K D)$ | $\mathcal{O}(NK + KD)$ | Hard discrete / $\epsilon{\to}0$ | Fast spherical clusters; no density |
| GMM (Full EM) | $\mathcal{O}(T N K D^2)$ | $\mathcal{O}(NK + KD^2)$ | Soft discrete / Full $\boldsymbol{\Sigma}_k$ | Multimodal density; singularity risk |

> [!info] Cross-Book Synthesis: Deterministic Subspaces vs. Generative Latent Models
> Deisenroth et al. [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 10–11) and Bishop & Bishop [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 15–16) emphasize that moving from deterministic PCA/$K$-means to PPCA/GMMs replaces hard geometric objectives with proper likelihoods $p(\boldsymbol{x} \mid \boldsymbol{\theta})$. Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 20–21)[[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 28) unifies both under a single shallow latent-factor taxonomy ($p(\boldsymbol{z}) \to p(\boldsymbol{x}\mid\boldsymbol{z})$), where $K$-means, GMMs, PCA, PPCA, FA, and ICA differ solely by whether $\boldsymbol{z}$ is discrete or continuous and whether $p(\boldsymbol{z})$ and $p(\boldsymbol{x}\mid\boldsymbol{z})$ are Gaussian, isotropic, or non-Gaussian.

## Visual Diagnostics & Geometric Structure

Figure `fig:em:pgm` contrasts the directed graphical models of PPCA/Factor Analysis and Gaussian Mixture Models in plate notation. Figure `fig:em:elbo_tikz` illustrates the two-stage variational EM geometry in parameter and distribution space, while Figure `fig:em:pgfplots` plots the tangential lower-bound climb alongside a PCA variance-retained scree diagnostic.

> [!example] Visual Diagram (LaTeX / PDF)
> *Directed graphical models in plate notation: PPCA/Factor Analysis (left, continuous Gaussian latent $\boldsymbol{z}_n \in \mathbb{R}^M$) versus GMM (right, discrete categorical latent $z_n \in \{1,\dots,K\}$).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Two-stage coordinate-ascent cycle of the EM algorithm: the E-step collapses the KL gap to achieve tangential contact, and the M-step raises the surrogate $Q(\boldsymbol{\theta};\boldsymbol{\theta}^{(t)})$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Left: Tangential EM lower-bound climb—each E-step matches $\log p(\boldsymbol{X}\mid\theta)$ at the current iterate, and each M-step maximizes the concave surrogate. Right: Scree eigenvalue decay and cumulative variance retained (90\% threshold dashed).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:em:gmm_oldfaithful` demonstrates GMM-EM on the Old Faithful dataset, showing soft responsibilities converging to bimodal elliptical contours, while Figure `fig:em:empca_rod` illustrates the zero-noise EM-PCA alternating orthogonal projection and subspace re-estimation.

> [!note] Source Figure
> *Figure:* EM algorithm applied to a two-component GMM on the rescaled Old Faithful dataset across iterations $L=1,2,5,20$, showing soft responsibility coloring and 1-std covariance ellipses [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 15.2, Fig. 15.7, p. 488). (`deep_learning_foundations_concepts_p488_fig1.png`)

> [!note] Source Figure
> *Figure:* Geometric progression of EM-PCA ($\sigma^2 \to 0$ limit): alternating orthogonal projections of data points onto the current principal subspace (E-step) and least-squares subspace updates (M-step) [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 16.3.2, Fig. 16.10, p. 535). (`deep_learning_foundations_concepts_p535_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] GMM MLE Covariance Collapse Singularity ($\det\boldsymbol{\Sigma}_k \to 0$)
> In unconstrained maximum likelihood estimation for a GMM with $K \ge 2$, if component $k$ centers on a single data point ($\boldsymbol{\mu}_k = \boldsymbol{x}_n$) with variance $\sigma_k^2 \to 0$, its density scales as $(2\pi \sigma_k^2)^{-D/2} \to +\infty$ while remaining components assign finite density to the remaining $N-1$ points [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 15.2.2, p. 485) (Figure `fig:em:gmm_singularity`). **Remedy:** Add diagonal regularization $\boldsymbol{\Sigma}_k \leftarrow \boldsymbol{\Sigma}_k + \epsilon\boldsymbol{I}_D$ ($\epsilon \approx 10^{-6}$) in every M-step, place a conjugate Normal-Inverse-Wishart prior (MAP-EM), or re-initialize collapsing components whose $N_k$ or $\det(\boldsymbol{\Sigma}_k)$ drops below threshold.

> [!note] Source Figure
> *Figure:* Pathological MLE singularity in a two-component GMM: one Gaussian collapses onto a single observation ($\sigma_k \to 0$), driving the log-likelihood to $+\infty$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 15.2.2, Fig. 15.6, p. 485). (`deep_learning_foundations_concepts_p485_fig1.png`)

> [!warning] Non-Convexity, Local Optima, and $k$-Means++ Initialization
> Both $K$-means and GMM-EM optimize non-convex objectives and converge only to local optima or saddle points. Uniformly random initialization frequently traps EM in poor partitions. **Remedy:** Initialize centers sequentially via $k$-means++ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 21.3.4, PDF p. 755), sampling $\boldsymbol{\mu}_t = \boldsymbol{x}_n$ with probability proportional to $D_{t-1}(\boldsymbol{x}_n) = \min_{j < t}\lVert \boldsymbol{x}_n - \boldsymbol{\mu}_j \rVert_2^2$, which guarantees an $\mathcal{O}(\log K)$-competitive distortion bound, and select the highest log-likelihood across multiple restarts.

> [!warning] Identifiability: Label Switching ($K!$) and Rotational Ambiguity ($\boldsymbol{W}\boldsymbol{R}$)
> - **Label Switching:** Any $K$-component mixture has $K!$ parameter permutations yielding identical marginal densities $p(\boldsymbol{x}\mid\boldsymbol{\theta})$, complicating Bayesian posterior averaging [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 15.2.2, p. 485).
> - **Rotational Ambiguity:** In PPCA and Factor Analysis, $\boldsymbol{W}' = \boldsymbol{W}\boldsymbol{R}$ for any orthogonal $\boldsymbol{R} \in \mathrm{O}(M)$ produces the exact same marginal covariance $\boldsymbol{W}\boldsymbol{R}\boldsymbol{R}^{\top}\boldsymbol{W}^{\top} + \boldsymbol{\Psi} = \boldsymbol{W}\boldsymbol{W}^{\top} + \boldsymbol{\Psi}$, leaving $M(M-1)/2$ unidentifiable degrees of freedom [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 20.2.4, PDF p. 702)[[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 28.3.1, PDF p. 978). **Remedy:** Fix the gauge via lower-triangular $\boldsymbol{W}$ with $w_{kk} > 0$, sparsity-inducing priors, varimax rotations, or non-Gaussian latent priors (ICA).

> [!warning] Scale Sensitivity and Rank Deficiency ($N < D$) in PCA
> PCA on the raw covariance matrix $\boldsymbol{S}$ is not scale-invariant: a feature measured in millimeters rather than meters dominates $\lambda_1$ purely due to unit scaling [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 20.1.3.1, PDF p. 692). Standardize features to zero mean and unit variance (correlation PCA) whenever units are incommensurate. Furthermore, when $N < D$, $\mathrm{rank}(\boldsymbol{S}) \le N - 1$ and at least $D - N + 1$ eigenvalues vanish; use the $\mathcal{O}(N^3)$ dual Gram matrix or truncated SVD instead of $\mathcal{O}(D^3)$ covariance eigendecomposition [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 16.1.5, p. 518).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[EM Algorithm]], [[Gaussian Mixture Models (GMM)]], [[Principal Component Analysis (PCA)]]
- **Prerequisites (Directed $\leftarrow$):** [[Eigendecomposition & Spectral Theorem]], [[Gaussian Mixture Models (GMM)]], [[Multivariate Gaussian & Schur Complement]], [[Singular Value Decomposition (SVD)]], [[Variational Inference & ELBO]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[EM Algorithm]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Autoencoders]], [[EM Algorithm]], [[Eigendecomposition & Spectral Theorem]], [[Gaussian Mixture Models (GMM)]], [[Low-Rank Approximation (Eckart-Young)]], [[Multivariate Gaussian & Schur Complement]], [[Singular Value Decomposition (SVD)]], [[Variational Inference & ELBO]]
- **Synthesized Sources:** [[Source - Foundations]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
