---
type: monograph
domain: "D04"
parent_topic: "[[PT 04 - Autoencoders, VAEs, Normalizing Flows, EBMs & GANs]]"
sub_topic: "[[ST 04 - Autoencoders, Variational Autoencoders & Normalizing Flows]]"
cheatsheet: "[[Cheatsheet - Autoencoders, Variational Autoencoders (VAEs), Reparameterization & VQ-VAE]]"
part_of: 
  - "[[ST 04 - Autoencoders, Variational Autoencoders & Normalizing Flows]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
concepts: 
  - "[[Autoencoders]]"
  - "[[Variational Autoencoder (VAE)]]"
prerequisites: 
  - "[[Reparameterization Trick]]"
  - "[[Variational Inference & ELBO]]"
enables: 
  - "[[AutoRec]]"
related: 
  - "[[Autoencoders]]"
  - "[[DDPM]]"
  - "[[Information Bottleneck]]"
  - "[[Principal Component Analysis (PCA)]]"
  - "[[Reparameterization Trick]]"
  - "[[Variational Autoencoder (VAE)]]"
  - "[[Variational Inference & ELBO]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
  - "[[Source - Tutorial on Diffusion Models for Imaging and Vision]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-04
---

# Autoencoders, Variational Autoencoders (VAEs), Reparameterization & VQ-VAE

> [!abstract] Monograph Navigation
> - **Domain:** [[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion|Deep Generative Modeling: VAEs, Flows, EBMs, GANs & Diffusion]]
> - **Parent / Sub-Topic:** [[PT 04 - Autoencoders, VAEs, Normalizing Flows, EBMs & GANs|Autoencoders, VAEs, Normalizing Flows, EBMs & GANs]] → [[ST 04 - Autoencoders, Variational Autoencoders & Normalizing Flows|Autoencoders, Variational Autoencoders & Normalizing Flows]]
> - **Visual Cheatsheet:** [[Cheatsheet - Autoencoders, Variational Autoencoders (VAEs), Reparameterization & VQ-VAE]]
> - **Description:** Synthesizes undercomplete/denoising/contractive autoencoders, amortized variational inference via the VAE ELBO, pathwise reparameterization trick, posterior collapse mitigations (beta-VAE/free bits), and discrete VQ-VAE codebooks.

## Problem Formulation & Unified Notation

Representation learning and deep latent-variable generative modeling seek a low-dimensional latent code $\boldsymbol{z} \in \mathcal{Z}$ that captures the salient factors of variation underlying high-dimensional observations $\boldsymbol{x} \in \mathbb{R}^D$ drawn from an unknown data distribution $p_{\mathrm{data}}(\boldsymbol{x})$. Deterministic autoencoders learn a direct bottleneck mapping $\boldsymbol{x} \mapsto \boldsymbol{z} = e_\phi(\boldsymbol{x}) \mapsto \hat{\boldsymbol{x}} = d_\theta(\boldsymbol{z})$ by minimizing reconstruction distortion [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 19, p. 574); see also [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 14, p. 517). Because deterministic autoencoders impose no probabilistic prior over $\mathcal{Z}$, arbitrary samples $\boldsymbol{z} \sim p(\boldsymbol{z})$ do not decode to valid data points.

Variational Autoencoders (VAEs) resolve this by formulating a deep latent-variable model (DLVM) with joint density

$$
p_\theta(\boldsymbol{x}, \boldsymbol{z}) = p_\theta(\boldsymbol{z}) p_\theta(\boldsymbol{x} \mid \boldsymbol{z}),
$$

where $p_\theta(\boldsymbol{z})$ is a tractable prior (canonically $\mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_M)$) and $p_\theta(\boldsymbol{x} \mid \boldsymbol{z})$ is an exponential-family decoder parameterized by a neural network $d_\theta(\boldsymbol{z})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, p. 825). Maximum likelihood estimation requires evaluating the marginal likelihood

$$
p_\theta(\boldsymbol{x}) = \int_{\mathcal{Z}} p_\theta(\boldsymbol{x} \mid \boldsymbol{z}) p_\theta(\boldsymbol{z}) \mathrm{d}\boldsymbol{z},
$$

which is analytically intractable for nonlinear decoders, as is the exact posterior $p_\theta(\boldsymbol{z} \mid \boldsymbol{x}) = p_\theta(\boldsymbol{x}, \boldsymbol{z})/p_\theta(\boldsymbol{x})$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 17, p. 344) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 2.1, p. 39). Rather than performing per-datum iterative variational optimization, VAEs introduce an *amortized* inference (recognition) network $q_\phi(\boldsymbol{z} \mid \boldsymbol{x})$ that maps each observation $\boldsymbol{x}$ to variational posterior parameters $(\boldsymbol{\mu}_\phi(\boldsymbol{x}), \boldsymbol{\sigma}_\phi(\boldsymbol{x}))$, enabling joint optimization of $(\theta, \phi)$ via the Evidence Lower Bound (ELBO). Vector-Quantized VAEs (VQ-VAEs) replace the continuous Gaussian bottleneck with a discrete codebook $\mathcal{E} = \{\boldsymbol{e}_1, \dots, \boldsymbol{e}_K\} \subset \mathbb{R}^d$ and nearest-neighbor quantization [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, p. 849).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Source textbooks adopt swapped parameter symbols $(\theta, \phi)$ and different latent variable letters ($\boldsymbol{z}$ vs. $\boldsymbol{h}$). Throughout this monograph, we standardize on $\theta$ for generative/decoder parameters and $\phi$ for amortized encoder parameters (matching Murphy and Chan).
>
> | **Concept** | **Unified** | **Murphy / Chan** | **Bishop / Prince / Princ.** | **Goodfellow et al.** |
> | --- | --- | --- | --- | --- |
> | Input / Latent | $\boldsymbol{x} \in \mathbb{R}^D,  \boldsymbol{z} \in \mathbb{R}^M$ | $\boldsymbol{x},  \boldsymbol{z}$ | $\boldsymbol{x},  \boldsymbol{z}$ | $\boldsymbol{x},  \boldsymbol{h}$ (code) |
> | Encoder / Params | $q_\phi(\boldsymbol{z}\mid\boldsymbol{x}),  \phi$ | $q_\phi(\boldsymbol{z}\mid\boldsymbol{x}),  \phi$ | $q(\boldsymbol{z}\mid\boldsymbol{x},\phi)$ / $q_\theta(\boldsymbol{z}\mid\boldsymbol{x})$ | $f(\boldsymbol{x})$ / $p_{\mathrm{enc}}(\boldsymbol{h}\mid\boldsymbol{x})$ |
> | Decoder / Params | $p_\theta(\boldsymbol{x}\mid\boldsymbol{z}),  \theta$ | $p_\theta(\boldsymbol{x}\mid\boldsymbol{z}),  \theta$ | $p(\boldsymbol{x}\mid\boldsymbol{z},\boldsymbol{w})$ / $p_\phi(\boldsymbol{x}\mid\boldsymbol{z})$ | $g(\boldsymbol{h})$ / $p_{\mathrm{dec}}(\boldsymbol{x}\mid\boldsymbol{h})$ |
> | Reconstruction | $\hat{\boldsymbol{x}} = d_\theta(\boldsymbol{z})$ | $\hat{\boldsymbol{x}} = d_\theta(\boldsymbol{z})$ | $\boldsymbol{y}(\boldsymbol{z})$ / $f[\boldsymbol{z},\phi]$ | $\boldsymbol{r} = g(\boldsymbol{h})$ |
> | Pedagogical Focus | Unified synthesis | $\beta$-VAE, collapse, VQ-VAE | Linear AE–PCA, ELBO geom. | DAE score, CAE Jacobian |

## Core Mathematical Derivations & Theorems

### Linear Autoencoders, PCA Equivalence, and Regularized Score Interpretations

Before introducing stochastic latent variables, we establish the geometric baseline of deterministic undercomplete and regularized autoencoders.

> [!important] Linear Autoencoder–PCA Equivalence & Regularized Autoencoders
> **1. Linear Autoencoder–PCA Equivalence** [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 19, pp. 575–576): Consider a two-layer linear autoencoder with encoder $\boldsymbol{z}_n = \boldsymbol{W}_1\boldsymbol{x}_n \in \mathbb{R}^M$ and decoder $\hat{\boldsymbol{x}}_n = \boldsymbol{W}_2\boldsymbol{z}_n \in \mathbb{R}^D$ ($M < D$) minimizing the sum-of-squares loss
>
> $$
> \mathcal{E}(\boldsymbol{W}_1, \boldsymbol{W}_2) = \frac{1}{2}\sum_{n=1}^N \lVert \boldsymbol{W}_2 \boldsymbol{W}_1 \boldsymbol{x}_n - \boldsymbol{x}_n \rVert_2^2.
> $$
>
> At the unique global minimum (up to an invertible $M \times M$ basis change), the column space of $\boldsymbol{W}_2$ spans the exact $M$-dimensional principal subspace of the empirical covariance $\boldsymbol{S} = \frac{1}{N}\sum_{n=1}^N \tilde{\boldsymbol{x}}_n \tilde{\boldsymbol{x}}_n^{\top}$ (for centered data $\tilde{\boldsymbol{x}}_n$). Even with a single nonlinear hidden layer, if the decoder remains linear and squared error is used, the optimal projection still spans the linear PCA subspace [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 19, p. 576) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 14, p. 518).
>
> **2. Denoising Autoencoders (DAE) as Score Estimators** [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 14, pp. 525–529): Let $C(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}) = \mathcal{N}(\tilde{\boldsymbol{x}}; \boldsymbol{x}, \sigma^2 \boldsymbol{I}_D)$ corrupt clean data $\boldsymbol{x} \sim p_{\mathrm{data}}(\boldsymbol{x})$. Training $r_\psi(\tilde{\boldsymbol{x}}) = d_\theta(e_\phi(\tilde{\boldsymbol{x}}))$ under squared error
>
> $$
> \mathcal{L}_{\mathrm{DAE}}(\psi) = \mathbb{E}_{\boldsymbol{x} \sim p_{\mathrm{data}}} \mathbb{E}_{\tilde{\boldsymbol{x}} \sim C(\cdot\mid\boldsymbol{x})}\!\left[ \lVert r_\psi(\tilde{\boldsymbol{x}}) - \boldsymbol{x} \rVert_2^2 \right]
> $$
>
> yields the Bayes-optimal denoiser $r^*(\tilde{\boldsymbol{x}}) = \mathbb{E}[\boldsymbol{x} \mid \tilde{\boldsymbol{x}}]$. By Tweedie's formula for isotropic Gaussian corruption, the residual displacement estimates the Stein score of the smoothed data density $p_\sigma(\tilde{\boldsymbol{x}}) = \int p_{\mathrm{data}}(\boldsymbol{x})\mathcal{N}(\tilde{\boldsymbol{x}};\boldsymbol{x},\sigma^2\boldsymbol{I}) \mathrm{d}\boldsymbol{x}$:
>
> $$
> r^*(\tilde{\boldsymbol{x}}) - \tilde{\boldsymbol{x}} = \sigma^2 \nabla_{\tilde{\boldsymbol{x}}} \log p_\sigma(\tilde{\boldsymbol{x}}) + o(\sigma^2).
> $$
>
> **3. Contractive Autoencoders (CAE):** As shown in [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 14, pp. 536–538), a CAE penalizes the Frobenius norm of the encoder Jacobian $\boldsymbol{J}_{e}(\boldsymbol{x}) = \partial e_\phi(\boldsymbol{x}) / \partial \boldsymbol{x}$:
>
> $$
> \mathcal{L}_{\mathrm{CAE}}(\theta, \phi) = \mathcal{L}_{\mathrm{rec}}(\boldsymbol{x}, d_\theta(e_\phi(\boldsymbol{x}))) + \lambda \lVert \boldsymbol{J}_e(\boldsymbol{x}) \rVert_F^2.
> $$
>

> [!tip] Manifold Tangent Geometry vs. Orthogonal Projection
> In a DAE, the vector field $r^*(\tilde{\boldsymbol{x}}) - \tilde{\boldsymbol{x}}$ points orthogonally back toward the high-density data manifold, directly prefiguring score-based diffusion models [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 14, p. 529). Conversely, the CAE penalty $\lVert \boldsymbol{J}_e(\boldsymbol{x}) \rVert_F^2$ forces the encoder to be locally invariant along directions orthogonal to the data manifold while preserving sensitivity along the leading singular vectors of $\boldsymbol{J}_e(\boldsymbol{x})$, which span the local tangent plane [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 14, p. 537).

### The Variational ELBO, KL Decomposition, and Amortized Inference Gap

> [!important] Exact ELBO Derivation and Augmented-Space Divergence
> For any observation $\boldsymbol{x}$ and any variational distribution $q_\phi(\boldsymbol{z} \mid \boldsymbol{x})$ whose support covers $p_\theta(\boldsymbol{z} \mid \boldsymbol{x})$, we derive the ELBO via Jensen's inequality [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 17, pp. 345–346) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 2.1.2, p. 40):
>
> $$
> \begin{aligned}
> \log p_\theta(\boldsymbol{x})
>   &= \log \int q_\phi(\boldsymbol{z} \mid \boldsymbol{x}) \frac{p_\theta(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})} \mathrm{d}\boldsymbol{z}
>   \ge \mathbb{E}_{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\!\left[\log \frac{p_\theta(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\right]  \\
>   &= \underbrace{\mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\!\left[\log p_\theta(\boldsymbol{x}\mid\boldsymbol{z})\right]}_{\text{Reconstruction (Likelihood)}}
>   - \underbrace{D_{\mathrm{KL}}(q_\phi(\boldsymbol{z}\mid\boldsymbol{x}) \parallel p_\theta(\boldsymbol{z}))}_{\text{Prior Matching (Rate)}}
>   =: \mathcal{L}_{\mathrm{ELBO}}(\theta, \phi; \boldsymbol{x}).
> \end{aligned}
> $$
>
> Equivalently, multiplying and dividing inside the expectation by $p_\theta(\boldsymbol{z} \mid \boldsymbol{x})$ yields the exact identity [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 19, p. 584) [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 1.2, p. 5):
>
> $$
> \log p_\theta(\boldsymbol{x}) = \mathcal{L}_{\mathrm{ELBO}}(\theta, \phi; \boldsymbol{x}) + D_{\mathrm{KL}}(q_\phi(\boldsymbol{z}\mid\boldsymbol{x}) \parallel p_\theta(\boldsymbol{z}\mid\boldsymbol{x})).
> $$
>
> Averaging over the empirical data distribution $q_{\mathcal{D}}(\boldsymbol{x})$ reveals that maximizing the ELBO jointly over $(\theta,\phi)$ minimizes the joint KL divergence in the augmented $(\boldsymbol{x},\boldsymbol{z})$ space [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, p. 828, p. 831):
>
> $$
> \mathbb{E}_{q_{\mathcal{D}}(\boldsymbol{x})}\!\left[\log q_{\mathcal{D}}(\boldsymbol{x}) - \mathcal{L}_{\mathrm{ELBO}}(\theta, \phi; \boldsymbol{x})\right]
>   = D_{\mathrm{KL}}(q_{\mathcal{D},\phi}(\boldsymbol{x}, \boldsymbol{z}) \parallel p_\theta(\boldsymbol{x}, \boldsymbol{z}))
>   \ge D_{\mathrm{KL}}(q_{\mathcal{D}}(\boldsymbol{x}) \parallel p_\theta(\boldsymbol{x})),
> $$
>
> where $q_{\mathcal{D},\phi}(\boldsymbol{x}, \boldsymbol{z}) = q_{\mathcal{D}}(\boldsymbol{x}) q_\phi(\boldsymbol{z} \mid \boldsymbol{x})$.
>
> **Closed-Form Gaussian KL** [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 19, p. 585) [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 1.4, p. 9): When $p_\theta(\boldsymbol{z}) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_M)$ and $q_\phi(\boldsymbol{z} \mid \boldsymbol{x}) = \mathcal{N}(\boldsymbol{z}; \boldsymbol{\mu}_\phi(\boldsymbol{x}), \mathrm{diag}(\boldsymbol{\sigma}_\phi^2(\boldsymbol{x})))$, the KL term integrates analytically:
>
> $$
> D_{\mathrm{KL}}(q_\phi(\boldsymbol{z}\mid\boldsymbol{x}) \parallel p_\theta(\boldsymbol{z}))
>   = \frac{1}{2}\sum_{j=1}^{M}\!\left( \mu_{\phi,j}^2(\boldsymbol{x}) + \sigma_{\phi,j}^2(\boldsymbol{x}) - 1 - \log \sigma_{\phi,j}^2(\boldsymbol{x}) \right).
> $$
>

### The Pathwise Reparameterization Trick & Variance Reduction

To optimize $\mathcal{L}_{\mathrm{ELBO}}(\theta,\phi;\boldsymbol{x})$ via stochastic gradient ascent, we require gradients with respect to $\phi$, which appears inside the expectation measure $q_\phi(\boldsymbol{z}\mid\boldsymbol{x})$.

> [!important] Pathwise Reparameterization vs. Score-Function (REINFORCE) Estimators
> If we draw $\boldsymbol{z} \sim q_\phi(\boldsymbol{z}\mid\boldsymbol{x})$ directly, the sampling node blocks backpropagation to $\phi$. The *score-function* (REINFORCE) identity
>
> $$
> \nabla_\phi \mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}[f(\boldsymbol{z})]
>   = \mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\!\left[f(\boldsymbol{z}) \nabla_\phi \log q_\phi(\boldsymbol{z}\mid\boldsymbol{x})\right]
> $$
>
> is unbiased for any $f(\boldsymbol{z}) = \log p_\theta(\boldsymbol{x}\mid\boldsymbol{z})$, but suffers from prohibitively high Monte Carlo variance [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 19, p. 586).
>
> For continuous location-scale families, the *reparameterization trick* expresses $\boldsymbol{z} \sim q_\phi(\boldsymbol{z}\mid\boldsymbol{x})$ as a deterministic differentiable transformation $g_\phi(\boldsymbol{\epsilon}, \boldsymbol{x})$ of an auxiliary parameter-free noise variable $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon}) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_M)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, p. 827) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 17, p. 352) [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 1.3, p. 8):
>
> $$
> \boldsymbol{z} = g_\phi(\boldsymbol{\epsilon}, \boldsymbol{x}) = \boldsymbol{\mu}_\phi(\boldsymbol{x}) + \boldsymbol{\sigma}_\phi(\boldsymbol{x}) \odot \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_M).
> $$
>
> Applying the law of the unconscious statistician transfers the $\phi$-dependence from the measure into the integrand, enabling low-variance pathwise differentiation:
>
> $$
> \nabla_\phi \mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}[f(\boldsymbol{z})]
>   = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_M)}\!\left[ \left(\nabla_{\boldsymbol{z}} f(\boldsymbol{z})\big|_{\boldsymbol{z}=g_\phi(\boldsymbol{\epsilon},\boldsymbol{x})}\right)^{\top} \nabla_\phi g_\phi(\boldsymbol{\epsilon}, \boldsymbol{x}) \right]
>   \approx \frac{1}{L}\sum_{l=1}^L \nabla_\phi f\!\left(g_\phi(\boldsymbol{\epsilon}^{(l)}, \boldsymbol{x})\right),
> $$
>
> where $L=1$ per datum in a mini-batch typically suffices for stochastic gradient optimization [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 19, p. 587).

### Rate–Distortion Trade-Off ($\beta$-VAE), VQ-VAE, and Hierarchical Diffusion Bridge

> [!important] $\beta$-VAE Rate–Distortion Objective and VQ-VAE Quantization
> **1. $\beta$-VAE and Total Correlation** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, pp. 831–832): Weighting the KL term by $\beta \ge 0$ yields the $\beta$-VAE loss:
>
> $$
> \mathcal{L}_\beta(\theta, \phi; \boldsymbol{x})
>   = \underbrace{-\mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}[\log p_\theta(\boldsymbol{x}\mid\boldsymbol{z})]}_{\text{Distortion } D}
>   + \beta \underbrace{D_{\mathrm{KL}}(q_\phi(\boldsymbol{z}\mid\boldsymbol{x}) \parallel p_\theta(\boldsymbol{z}))}_{\text{Rate } R}.
> $$
>
> Setting $\beta=1$ recovers the negative ELBO; $\beta=0$ recovers a noisy autoencoder. Setting $\beta > 1$ penalizes the latent channel capacity $R$, which upper-bounds the mutual information $I(\boldsymbol{x}; \boldsymbol{z})$ and suppresses Total Correlation $\mathrm{TC}(\boldsymbol{z}) = D_{\mathrm{KL}}(q_\phi(\boldsymbol{z}) \parallel \prod_j q_\phi(z_j))$, encouraging disentangled factors (subject to inductive bias, as nonlinear latent models are fundamentally unidentifiable without structural priors) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, p. 832) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 17, p. 355).
>
> **2. Vector-Quantized VAE (VQ-VAE)** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, pp. 849–851): To avoid posterior collapse and learn discrete representations, VQ-VAE maintains a codebook $\mathcal{E} = \{\boldsymbol{e}_1, \dots, \boldsymbol{e}_K\} \subset \mathbb{R}^d$. Given continuous encoder output $\boldsymbol{z}_e(\boldsymbol{x})$, nearest-neighbor lookup assigns
>
> $$
> k^* = \arg\min_{k \in \{1,\dots,K\}} \lVert \boldsymbol{z}_e(\boldsymbol{x}) - \boldsymbol{e}_k \rVert_2, \qquad \boldsymbol{z}_q(\boldsymbol{x}) = \boldsymbol{e}_{k^*}.
> $$
>
> Because $\arg\min$ has zero gradient almost everywhere, the backward pass uses the *straight-through estimator* (STE), copying gradients directly from $\boldsymbol{z}_q$ to $\boldsymbol{z}_e$: $\nabla_{\boldsymbol{z}_e} \mathcal{L}_{\mathrm{rec}} \approx \nabla_{\boldsymbol{z}_q} \mathcal{L}_{\mathrm{rec}}$. Codebook vectors and encoder outputs are aligned via stop-gradient ($\operatorname{sg}[\cdot]$) MSE terms:
>
> $$
> \mathcal{L}_{\mathrm{VQ}}(\theta, \phi, \mathcal{E}; \boldsymbol{x})
>   = -\log p_\theta(\boldsymbol{x} \mid \boldsymbol{z}_q(\boldsymbol{x}))
>   + \lVert \operatorname{sg}[\boldsymbol{z}_e(\boldsymbol{x})] - \boldsymbol{e}_{k^*} \rVert_2^2
>   + \beta_{\mathrm{com}} \lVert \boldsymbol{z}_e(\boldsymbol{x}) - \operatorname{sg}[\boldsymbol{e}_{k^*}] \rVert_2^2,
> $$
>
> where the middle term updates the codebook (or is replaced by an exponential moving average with $\gamma=0.9$), and the final commitment term ($\beta_{\mathrm{com}} \approx 0.25$) prevents $\boldsymbol{z}_e(\boldsymbol{x})$ from drifting unboundedly [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, p. 851).

> [!info] Cross-Book Synthesis: From 1-Step VAE to Hierarchical Diffusion ELBO
> Both [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 2.1.5, pp. 44–45) and [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 1–2, pp. 9–10) emphasize that a flat 1-step VAE is the $T=1$ special case of a Hierarchical VAE (HVAE) with latent chain $\boldsymbol{z}_{1:T}$ (or $\boldsymbol{x}_{1:T}$). In a top-down HVAE, $p_\theta(\boldsymbol{x}, \boldsymbol{z}_{1:L}) = p_\theta(\boldsymbol{z}_L)\prod_{l=2}^L p_\theta(\boldsymbol{z}_{l-1}\mid \boldsymbol{z}_l) p_\theta(\boldsymbol{x}\mid \boldsymbol{z}_1)$, and the ELBO decomposes into a reconstruction term plus $L$ step-wise KL divergences between generative conditionals and variational posteriors [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, p. 843) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 2.1.5, p. 45). Denoising Diffusion Probabilistic Models (DDPMs) specialize this HVAE by fixing the encoder $q(\boldsymbol{x}_{1:T}\mid \boldsymbol{x}_0)$ to a non-learned Gaussian Markov noising chain and keeping latent dimensionality equal to $\mathbb{R}^D$, thereby eliminating the amortization gap and enabling high-fidelity progressive synthesis [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 17, p. 355).

## Algorithmic Implementation & Comparative Synthesis

Algorithm `alg:vae:sgvb_vq` unifies Stochastic Gradient Variational Bayes (SGVB) training for continuous Gaussian VAEs and discrete VQ-VAE training with EMA codebook updates.

> [!example] Algorithm: Amortized SGVB (Continuous $\beta$-VAE) and VQ-VAE Training
> **Input:** Dataset $\mathcal{D}$, encoder $e_\phi$, decoder $d_\theta$, mode $\in \{\text{VAE}, \text{VQ-VAE}\}$, hyperparameters $\beta$ (KL weight) or $(\beta_{\mathrm{com}}, \gamma)$ (commitment, EMA decay), learning rate $\eta$.
> **Output:** Trained parameters $(\theta, \phi)$ (and codebook $\mathcal{E}$ for VQ-VAE).
> Initialize $\theta, \phi$ (and $\mathcal{E}=\{\boldsymbol{e}_k\}_{k=1}^K$, counts $N_k^{(0)} \leftarrow 1$, sums $\boldsymbol{m}_k^{(0)} \leftarrow \boldsymbol{e}_k$ if VQ-VAE)
> **while** not converged:
>
>   Sample mini-batch $\{\boldsymbol{x}^{(b)}\}_{b=1}^B \sim \mathcal{D}$
>   **if** mode is VAE:
>
>     $(\boldsymbol{\mu}^{(b)}, \log \boldsymbol{\sigma}^{2(b)}) \leftarrow e_\phi(\boldsymbol{x}^{(b)})$  *// Clamp $\log \boldsymbol{\sigma}^2 \in [-20, 10]$*
>     Sample $\boldsymbol{\epsilon}^{(b)} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_M)$ and set $\boldsymbol{z}^{(b)} \leftarrow \boldsymbol{\mu}^{(b)} + \exp(\tfrac{1}{2}\log \boldsymbol{\sigma}^{2(b)}) \odot \boldsymbol{\epsilon}^{(b)}$
>     Compute $\mathcal{L}_{\mathrm{ELBO}}^{(b)} \leftarrow -\log p_\theta(\boldsymbol{x}^{(b)} \mid d_\theta(\boldsymbol{z}^{(b)})) + \beta D_{\mathrm{KL}}(q_\phi(\boldsymbol{z}\mid\boldsymbol{x}^{(b)}) \parallel p(\boldsymbol{z}))$ via `(eq:vae:gaussian_kl)`
>
> **else**:
>
>     Compute $\boldsymbol{z}_e^{(b)} \leftarrow e_\phi(\boldsymbol{x}^{(b)})$ and assign $k^*(b) \leftarrow \arg\min_k \lVert \boldsymbol{z}_e^{(b)} - \boldsymbol{e}_k \rVert_2$
>     Set $\boldsymbol{z}_q^{(b)} \leftarrow \boldsymbol{e}_{k^*(b)}$; in backward pass apply STE: $\boldsymbol{z}_q^{(b)} \leftarrow \boldsymbol{z}_e^{(b)} + \operatorname{sg}[\boldsymbol{e}_{k^*(b)} - \boldsymbol{z}_e^{(b)}]$
>     Compute $\mathcal{L}_{\mathrm{VQ}}^{(b)} \leftarrow -\log p_\theta(\boldsymbol{x}^{(b)} \mid d_\theta(\boldsymbol{z}_q^{(b)})) + \beta_{\mathrm{com}}\lVert \boldsymbol{z}_e^{(b)} - \operatorname{sg}[\boldsymbol{e}_{k^*(b)}] \rVert_2^2$
>     Update EMA codebook: $N_k \leftarrow \gamma N_k + (1-\gamma) n_k$, $\boldsymbol{m}_k \leftarrow \gamma \boldsymbol{m}_k + (1-\gamma)\sum_{b:k^*(b)=k}\boldsymbol{z}_e^{(b)}$, $\boldsymbol{e}_k \leftarrow \boldsymbol{m}_k / N_k$
>
>   Update $(\theta, \phi) \leftarrow (\theta, \phi) - \eta \nabla_{\theta,\phi}\!\left(\frac{1}{B}\sum_{b=1}^B \mathcal{L}^{(b)}\right)$

Table `tab:vae:comparison` synthesizes the structural, probabilistic, and computational trade-offs across the autoencoder family.

**Table:** Comparative synthesis of deterministic, regularized, variational, and quantized autoencoders.

| **Model** | **Latent Space** | **Objective** | **Generative?** | **Primary Limitation / Inductive Bias** |
| --- | --- | --- | --- | --- |
| Linear AE | $\mathbb{R}^M$ ($M < D$) | MSE `(eq:vae:linear_ae)` | No | Spans linear PCA subspace only [[Source - Foundations]] (Ch. 19, p. 575) |
| DAE / CAE | $\mathbb{R}^M$ (overcomp.) | DAE `(eq:vae:dae_loss)` / CAE `(eq:vae:cae_loss)` | Implicit (MCMC) | Estimates local score/tangents; no $p(\boldsymbol{z})$  [[Source - Deep Learning]] (Ch. 14, p. 529) |
| Standard VAE | $\mathbb{R}^M$, $\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$ | ELBO `(eq:vae:elbo)` ($\beta=1$) | Yes ($\boldsymbol{z}\sim p(\boldsymbol{z})$) | Blurry samples; amortization gap [[Source - The Principles of Diffusion Models]] (Sec. 2.1.4, p. 43) |
| $\beta$-VAE | $\mathbb{R}^M$, factorized | Rate–dist. `(eq:vae:beta_vae)` | Yes | Sharpness vs. disentanglement trade-off  [[Source - Probabilistic Machine Learning]] (Ch. 21, p. 832) |
| VQ-VAE | $\{1,\dots,K\}^{H\times W}$ | STE + commit. `(eq:vae:vq_loss)` | With prior | Needs post-hoc prior (PixelCNN/Transf.)  [[Source - Probabilistic Machine Learning]] (Ch. 21, p. 851) |
| HVAE | $\boldsymbol{z}_{1:L}$ top-down | Multi-layer ELBO | Yes (progr.) | Needs residual/skip paths for stability  [[Source - Probabilistic Machine Learning]] (Ch. 21, p. 844) |

## Visual Diagnostics & Geometric Structure

Figure `fig:vae:reparam_tikz` contrasts the continuous reparameterization computational graph against the discrete VQ-VAE straight-through quantization bottleneck. Figure `fig:vae:rate_distortion` plots the theoretical rate–distortion frontier governed by $\beta$ in `(eq:vae:beta_vae)`. Figures `fig:vae:prince_graph`, `fig:vae:bishop_em`, and `fig:vae:murphy_vqvae` present the canonical source visualizations of the VAE computational graph, EM-vs-ELBO optimization geometry, and VQ-VAE codebook architecture.

> [!example] Visual Diagram (LaTeX / PDF)
> *Comparison of continuous pathwise reparameterization (top) versus discrete VQ-VAE nearest-neighbor quantization with the Straight-Through Estimator (bottom).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Rate–distortion frontier $D(R)$ for the $\beta$-VAE objective `(eq:vae:beta_vae)`. At optimality, operating points satisfy $\partial D / \partial R = -\beta$: large $\beta$ drives the model toward posterior collapse ($R \to 0$), whereas $\beta \to 0$ approaches a deterministic autoencoder with minimal distortion [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, p. 832).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!note] Source Figure
> *Figure:* Amortized VAE encoder–decoder computational graph illustrating how the ELBO balances prior matching $D_{\mathrm{KL}}(q(\boldsymbol{z}\mid\boldsymbol{x},\theta) \parallel p(\boldsymbol{z}))$ against data likelihood $\log p(\boldsymbol{x}\mid\boldsymbol{z}^*,\phi)$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 17, p. 351). (`understandingdl_simonprince_p351_fig1.png`)

> [!note] Source Figure
> *Figure:* Geometric comparison between (a) classical EM (exact E-step closes the gap to $\log p(\boldsymbol{x}\mid\boldsymbol{w})$) and (b) VAE joint gradient ascent on the ELBO, where a residual amortization gap remains [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 19, p. 585). (`deep_learning_foundations_concepts_p585_fig1.png`)

> [!note] Source Figure
> *Figure:* VQ-VAE architecture mapping encoder outputs $\boldsymbol{z}_e(\boldsymbol{x})$ to discrete codebook embeddings $\boldsymbol{e}_k$ via nearest-neighbor lookup, with red straight-through gradient flow $\nabla_{\boldsymbol{z}}\mathcal{L}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, p. 850). (`probabilisticml_kevinmurphy_advancedtopics_p850_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Posterior Collapse (Variational Over-Pruning) in Expressive Decoders
> **Failure Mode:** When the decoder $p_\theta(\boldsymbol{x} \mid \boldsymbol{z})$ is highly expressive (e.g., an autoregressive PixelCNN or Transformer), it can model $p_\theta(\boldsymbol{x} \mid \boldsymbol{z}) \approx p_{\mathrm{data}}(\boldsymbol{x})$ independently of $\boldsymbol{z}$. The encoder then collapses to the prior, $q_\phi(\boldsymbol{z} \mid \boldsymbol{x}) = p(\boldsymbol{z})$, driving $D_{\mathrm{KL}}(q_\phi(\boldsymbol{z}\mid\boldsymbol{x}) \parallel p(\boldsymbol{z})) \to 0$ and rendering the latent code uninformative [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, pp. 840–841).
>
> **Remedies:** (1) *Cyclical KL annealing*: ramp $\beta$ from $0 \to 1$ over training cycles; (2) *Free bits*: replace the per-dimension KL with $\sum_{j=1}^M \max(\lambda, D_{\mathrm{KL}}(q_\phi(z_j\mid\boldsymbol{x}) \parallel p(z_j)))$ to guarantee a minimum rate $\lambda$ per dimension; (3) *Architectural*: add skip connections from $\boldsymbol{z}$ to all decoder layers (Skip-VAE) or weaken the autoregressive receptive field [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, pp. 841–843).

> [!warning] Blurry Generations, Prior–Aggregated Posterior Mismatch & Codebook Collapse
> **Failure Mode:** Under a fixed-variance Gaussian decoder $p_\theta(\boldsymbol{x}\mid\boldsymbol{z}) = \mathcal{N}(\boldsymbol{x}; d_\theta(\boldsymbol{z}), \sigma_{\mathrm{dec}}^2\boldsymbol{I})$, the optimal decoder predicts the conditional mean $d^*(\boldsymbol{z}) = \mathbb{E}[\boldsymbol{x} \mid \boldsymbol{z}]$ over all inputs whose posteriors overlap at $\boldsymbol{z}$, causing characteristic blurriness [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 2.1.4, p. 43) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, p. 831). Furthermore, sampling $\boldsymbol{z} \sim p(\boldsymbol{z})$ mismatches the aggregated posterior $q_\phi(\boldsymbol{z}) = \frac{1}{N}\sum_{n=1}^N q_\phi(\boldsymbol{z}\mid\boldsymbol{x}_n)$, while in VQ-VAE unused codebook vectors can suffer dead-code collapse.
>
> **Remedies:** Sample from a learned prior matching $q_\phi(\boldsymbol{z})$ (e.g., PixelCNN/Transformer over VQ-VAE tokens, or VampPrior), adopt hierarchical latents (HVAE/Diffusion), and reset inactive VQ-VAE codebook entries to randomly perturbed encoder outputs $\boldsymbol{z}_e(\boldsymbol{x})$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 17, p. 355) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 21, p. 851).

> [!warning] Log-Variance Explosion and Numerical Underflow in Gaussian KL
> **Failure Mode:** Parameterizing the encoder variance directly via $\exp(\boldsymbol{\ell}_\phi(\boldsymbol{x}))$ where $\boldsymbol{\ell}_\phi = \log \boldsymbol{\sigma}_\phi^2$ can overflow $\exp(\cdot)$ in float16/bfloat16 or underflow $\log \sigma_{\phi,j}^2$ in `(eq:vae:gaussian_kl)` during early training.
>
> **Remedy:** Clamp the predicted log-variance $\boldsymbol{\ell}_\phi(\boldsymbol{x}) \in [-20, 10]$ (or parameterize $\boldsymbol{\sigma}_\phi = \operatorname{softplus}(\boldsymbol{s}_\phi) + 10^{-6}$) before computing $\exp(\tfrac{1}{2}\boldsymbol{\ell}_\phi)$ and evaluate the KL term in float32 precision.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[Autoencoders]], [[Variational Autoencoder (VAE)]]
- **Prerequisites (Directed $\leftarrow$):** [[Reparameterization Trick]], [[Variational Inference & ELBO]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[AutoRec]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Autoencoders]], [[DDPM]], [[Information Bottleneck]], [[Principal Component Analysis (PCA)]], [[Reparameterization Trick]], [[Variational Autoencoder (VAE)]], [[Variational Inference & ELBO]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]], [[Source - Understanding Deep Learning]]
