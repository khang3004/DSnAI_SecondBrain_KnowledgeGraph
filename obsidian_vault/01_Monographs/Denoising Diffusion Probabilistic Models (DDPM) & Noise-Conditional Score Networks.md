---
type: monograph
domain: "D04"
parent_topic: "[[PT 04 - Diffusion Models, Score SDEs, Flow Matching & Fast Samplers]]"
sub_topic: "[[ST 04 - DDPM, NCSN, Score SDEs & Flow Matching]]"
cheatsheet: "[[Cheatsheet - Denoising Diffusion Probabilistic Models (DDPM) & Noise-Conditional Score Networks]]"
part_of: 
  - "[[ST 04 - DDPM, NCSN, Score SDEs & Flow Matching]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
concepts: 
  - "[[DDPM]]"
prerequisites: 
  - "[[Score Matching (DSM & SSM)]]"
  - "[[Variational Inference & ELBO]]"
enables: 
  - "[[Classifier-Free Guidance (CFG)]]"
  - "[[Consistency Models]]"
  - "[[Score-Based SDE]]"
related: 
  - "[[Classifier-Free Guidance (CFG)]]"
  - "[[Consistency Models]]"
  - "[[Score Matching (DSM & SSM)]]"
  - "[[Score-Based SDE]]"
  - "[[Semantic & Instance Segmentation (FCN, U-Net & SAM)]]"
  - "[[Variational Autoencoder (VAE)]]"
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
  - "[[Source - Tutorial on Diffusion Models for Imaging and Vision]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-04
---

# Denoising Diffusion Probabilistic Models (DDPM) & Noise-Conditional Score Networks

> [!abstract] Monograph Navigation
> - **Domain:** [[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion|Deep Generative Modeling: VAEs, Flows, EBMs, GANs & Diffusion]]
> - **Parent / Sub-Topic:** [[PT 04 - Diffusion Models, Score SDEs, Flow Matching & Fast Samplers|Diffusion Models, Score SDEs, Flow Matching & Fast Samplers]] → [[ST 04 - DDPM, NCSN, Score SDEs & Flow Matching|DDPM, NCSN, Score SDEs & Flow Matching]]
> - **Visual Cheatsheet:** [[Cheatsheet - Denoising Diffusion Probabilistic Models (DDPM) & Noise-Conditional Score Networks]]
> - **Description:** Derives the forward Gaussian Markov chain, closed-form marginal q(x_t|x_0), reverse posterior q(x_{t-1}|x_t,x_0), variational ELBO decomposition, epsilon-prediction vs. x_0-prediction vs. v-prediction parameterization, and Tweedie equivalence with multi-scale NCSN.

## Problem Formulation & Unified Notation

Generative modeling seeks to sample from an unknown, potentially high-dimensional data distribution $p_{\mathrm{data}}(\boldsymbol{x}_0)$ on $\mathbb{R}^D$ given an i.i.d. training set $\{\boldsymbol{x}_0^{(n)}\}_{n=1}^N$. Unlike Variational Autoencoders (VAEs) that compress data into a low-dimensional latent bottleneck or Energy-Based Models (EBMs) that suffer from an intractable normalizing constant $Z_\theta = \int \exp(-E_\theta(\boldsymbol{x})) \mathrm{d}\boldsymbol{x}$, diffusion and score-based generative models construct a gradual, reversible bridge between $p_{\mathrm{data}}(\boldsymbol{x}_0)$ and a tractable isotropic Gaussian prior $\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.1, p. 867).

Two foundational discrete-time paradigms independently converged on this principle:
{
- **Denoising Diffusion Probabilistic Models (DDPMs)** [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 2, p. 10): Formulated as a hierarchical VAE of fixed latent dimension $D$ with a non-learned Markovian Gaussian encoder
  
$$
q(\boldsymbol{x}_{1:T} \mid \boldsymbol{x}_0) = \prod_{t=1}^T q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}), \qquad q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}) = \mathcal{N}\!\left(\boldsymbol{x}_t;  \sqrt{1 - \beta_t} \boldsymbol{x}_{t-1},  \beta_t \boldsymbol{I}\right),
$$

  where $\{\beta_t \in (0, 1)\}_{t=1}^T$ is a variance schedule chosen so that $q(\boldsymbol{x}_T \mid \boldsymbol{x}_0) \approx \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$. Generation inverts this chain via a learned Gaussian Markov decoder:
  
$$
p_\theta(\boldsymbol{x}_{0:T}) = p(\boldsymbol{x}_T)\prod_{t=1}^T p_\theta(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t), \qquad p_\theta(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t) = \mathcal{N}\!\left(\boldsymbol{x}_{t-1};  \boldsymbol{\mu}_\theta(\boldsymbol{x}_t, t),  \Sigma_\theta(\boldsymbol{x}_t, t)\right),
$$

  with $p(\boldsymbol{x}_T) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$. Detailed discrete-time derivations appear in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.2, p. 867), [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 18.3, p. 355), and [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 20.2, p. 585).
- **Noise-Conditional Score Networks (NCSNs):** Rather than specifying a normalized density, NCSNs estimate the Stein score $\nabla_{\boldsymbol{x}} \log q_{\sigma_i}(\boldsymbol{x})$ of data perturbed by a geometric sequence of noise scales $\sigma_1 > \sigma_2 > \dots > \sigma_L > 0$,
  
$$
q_{\sigma_i}(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}_0) = \mathcal{N}(\tilde{\boldsymbol{x}};  \boldsymbol{x}_0, \sigma_i^2 \boldsymbol{I}), \qquad q_{\sigma_i}(\tilde{\boldsymbol{x}}) = \int p_{\mathrm{data}}(\boldsymbol{x}_0)  q_{\sigma_i}(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}_0) \mathrm{d}\boldsymbol{x}_0,
$$

  using a single noise-conditioned network $\boldsymbol{s}_\theta(\tilde{\boldsymbol{x}}, \sigma_i) \approx \nabla_{\tilde{\boldsymbol{x}}} \log q_{\sigma_i}(\tilde{\boldsymbol{x}})$ trained via multi-scale denoising score matching and sampled via Annealed Langevin Dynamics (ALD) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 3.4, p. 80) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.3.2, p. 876) [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 20.3.3, p. 598).

}

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> | **Concept** | **Chan / Murphy Adv.** | **Prince / Bishop** | **Diffusion Principles** |
> | — | — | — | — |
> | Clean & latent states | $\boldsymbol{x}_0 \in \mathbb{R}^D$ (data), $\boldsymbol{x}_1,\dots,\boldsymbol{x}_T$ (latents) | $\boldsymbol{x}$ (data), $\boldsymbol{z}_1,\dots,\boldsymbol{z}_T$ (latents) | $\boldsymbol{x}_0$ (data), $\boldsymbol{x}_1,\dots,\boldsymbol{x}_L$ (Sec. 2.2); $\boldsymbol{x}_t$ (Sec. 6.2) |
> | [3pt] Step-wise & cumulative signal | $\alpha_t = 1-\beta_t$, $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ | $\beta_t$, cumulative $\alpha_t = \prod_{s=1}^t (1-\beta_s)$ | Sec. 2.2: $\sqrt{\alpha_i}=\sqrt{1-\beta_i^2}$, $\bar{\alpha}_i=\prod_{s=1}^i \sqrt{\alpha_s}$; Sec. 6.2: $\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_0+\sigma_t\boldsymbol{\epsilon}$ |
> | [3pt] Decoder parameters | $\theta$ in $p_\theta(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t)$ | $\phi_t$ (Prince), $\boldsymbol{w}$ (Bishop) | $\phi$ in $p_\phi(\boldsymbol{x}_{i-1}\mid\boldsymbol{x}_i)$ |
> | [3pt] Noise / score predictor | $\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)$, $\boldsymbol{s}_\theta(\tilde{\boldsymbol{x}}, \sigma_t)$ | $\boldsymbol{g}_t[\boldsymbol{z}_t, \phi_t]$ (Prince), $\boldsymbol{s}(\boldsymbol{z}, \boldsymbol{w}, \sigma_i^2)$ (Bishop) | $\boldsymbol{\epsilon}_\phi(\boldsymbol{x}_t, t)$, $\boldsymbol{x}_\phi(\boldsymbol{x}_t, t)$, $\boldsymbol{s}_\phi(\boldsymbol{x}_t, t)$, $\boldsymbol{v}_\phi(\boldsymbol{x}_t, t)$ |
>
> 
> *Unified Convention Adopted Here:* We adopt the standard DDPM notation $\boldsymbol{x}_0, \boldsymbol{x}_1, \dots, \boldsymbol{x}_T$, $\alpha_t := 1 - \beta_t$, and $\bar{\alpha}_t := \prod_{s=1}^t \alpha_s$ [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 2.2, p. 14) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.2.1, p. 869). For unified continuous-time parameterizations, we write the general affine perturbation $\boldsymbol{x}_t = \alpha_t^{\mathrm{aff}} \boldsymbol{x}_0 + \sigma_t \boldsymbol{\epsilon}$ (where VP-DDPM corresponds to $\alpha_t^{\mathrm{aff}} = \sqrt{\bar{\alpha}_t}$ and $\sigma_t = \sqrt{1 - \bar{\alpha}_t}$) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 6.2.1, p. 170).

## Core Mathematical Derivations & Theorems

> [!important] Closed-Form Forward Marginal & Tractable Reverse Posterior
> Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. Under the forward Markov chain `(eq:ddpm:forward_chain)`:
> - **Closed-Form Forward Marginal:** For any $t \in \{1,\dots,T\}$,
>
>
> $$
> q(\boldsymbol{x}_t \mid \boldsymbol{x}_0) = \mathcal{N}\!\left(\boldsymbol{x}_t;  \sqrt{\bar{\alpha}_t} \boldsymbol{x}_0,  (1 - \bar{\alpha}_t)\boldsymbol{I}\right), \qquad \boldsymbol{x}_t = \sqrt{\bar{\alpha}_t} \boldsymbol{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}).
>
> $$
>
>   *Proof:* By induction. For $t=1$, $q(\boldsymbol{x}_1 \mid \boldsymbol{x}_0) = \mathcal{N}(\sqrt{\alpha_1}\boldsymbol{x}_0, (1-\alpha_1)\boldsymbol{I})$ holds by definition. Assuming `(eq:ddpm:marginal_q)` holds at $t-1$, we write $\boldsymbol{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\boldsymbol{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{t-1}$ and $\boldsymbol{x}_t = \sqrt{\alpha_t}\boldsymbol{x}_{t-1} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_t$ with independent $\boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_t \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$. Substituting yields
>
>
> $$
>
>     \boldsymbol{x}_t = \sqrt{\alpha_t \bar{\alpha}_{t-1}} \boldsymbol{x}_0 + \sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})} \boldsymbol{\epsilon}_{t-1} + \sqrt{1-\alpha_t} \boldsymbol{\epsilon}_t.
>
>
> $$
>
>   Since the sum of independent zero-mean Gaussians is Gaussian with variance $\alpha_t(1-\bar{\alpha}_{t-1}) + (1-\alpha_t) = 1 - \alpha_t\bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t$, `(eq:ddpm:marginal_q)` follows [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 2.3, p. 14) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 18.2, p. 353).
> - **Tractable Reverse Conditional Posterior:** Conditioning on $\boldsymbol{x}_0$, the single-step reverse posterior is Gaussian:
>
>
> $$
> q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0) = \mathcal{N}\!\left(\boldsymbol{x}_{t-1};  \tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0),  \tilde{\beta}_t \boldsymbol{I}\right),
>
> $$
>
>   where
>
>
> $$
> \tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}\boldsymbol{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\boldsymbol{x}_t, \qquad \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t.
>
> $$
>
>   *Proof:* By Bayes' rule and the Markov property $q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}, \boldsymbol{x}_0) = q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1})$,
>
>
> $$
> \begin{aligned}
> q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0) &= \frac{q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}) q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0)}{q(\boldsymbol{x}_t \mid \boldsymbol{x}_0)} \\
>     &\propto \exp\!\left(-\frac{1}{2}\left[\frac{\lVert \boldsymbol{x}_t - \sqrt{\alpha_t}\boldsymbol{x}_{t-1} \rVert_2^2}{\beta_t} + \frac{\lVert \boldsymbol{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\boldsymbol{x}_0 \rVert_2^2}{1 - \bar{\alpha}_{t-1}}\right]\right).
> \end{aligned}
>
> $$
>
>   Expanding the quadratic form in $\boldsymbol{x}_{t-1}$ gives precision $\tilde{\beta}_t^{-1} = \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}} = \frac{\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1-\bar{\alpha}_{t-1})} = \frac{1 - \bar{\alpha}_t}{\beta_t(1-\bar{\alpha}_{t-1})}$, and linear term $\frac{\sqrt{\alpha_t}}{\beta_t}\boldsymbol{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\boldsymbol{x}_0$. Multiplying the linear term by $\tilde{\beta}_t$ recovers $\tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0)$ in `(eq:ddpm:posterior_params)` [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 2.6, p. 22) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Eq. (18.15), p. 355).

> [!tip] Why Conditioning on $\boldsymbol{x}_0$ Unlocks Tractability
> While the marginal reverse transition $q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t) = \int q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0) q(\boldsymbol{x}_0 \mid \boldsymbol{x}_t) \mathrm{d}\boldsymbol{x}_0$ is a complex multimodal mixture that depends on the unknown data distribution $p_{\mathrm{data}}(\boldsymbol{x}_0)$, conditioning on $\boldsymbol{x}_0$ pins down both endpoints of the Brownian bridge, making $q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0)$ a closed-form unimodal Gaussian [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 18.2, p. 355). Crucially, as formalized in Theorem 2.2.1 of [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (p. 47), minimizing the expected conditional KL divergence $\mathbb{E}_{p_{\mathrm{data}}(\boldsymbol{x}_0)q(\boldsymbol{x}_t \mid \boldsymbol{x}_0)}[D_{\mathrm{KL}}(q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t,\boldsymbol{x}_0) \parallel p_\theta(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t))]$ shares the exact same gradient with respect to $\theta$ and the exact same unique minimizer $p_\theta^*(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t) = q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)$ as the intractable marginal KL divergence.

> [!important] Variational ELBO Decomposition & Simplified $\boldsymbol{\epsilon}$-Objective
> Maximizing the marginal log-likelihood $\log p_\theta(\boldsymbol{x}_0)$ is lower-bounded via Jensen's inequality by the Evidence Lower Bound (ELBO) $\mathcal{L}_{\mathrm{ELBO}}(\boldsymbol{x}_0; \theta)$ [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 20.2.2, p. 588) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.2.3, p. 870):
>
> $$
> \begin{aligned}
> -\log p_\theta(\boldsymbol{x}_0) &\le \mathbb{E}_{q(\boldsymbol{x}_{1:T} \mid \boldsymbol{x}_0)}\!\left[-\log \frac{p_\theta(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T} \mid \boldsymbol{x}_0)}\right]  \\
>   &= \underbrace{D_{\mathrm{KL}}(q(\boldsymbol{x}_T \mid \boldsymbol{x}_0) \parallel p(\boldsymbol{x}_T))}_{\mathcal{L}_T \text{ (prior matching)}} - \underbrace{\mathbb{E}_{q(\boldsymbol{x}_1 \mid \boldsymbol{x}_0)}\!\left[\log p_\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_1)\right]}_{\mathcal{L}_0 \text{ (reconstruction)}}  \\
>   &\quad + \sum_{t=2}^T \underbrace{\mathbb{E}_{q(\boldsymbol{x}_t \mid \boldsymbol{x}_0)}\!\left[D_{\mathrm{KL}}(q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0) \parallel p_\theta(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t))\right]}_{\mathcal{L}_{t-1} \text{ (step-wise consistency)}}.
> \end{aligned}
>
> $$
>
> *Proof:* Factoring $q(\boldsymbol{x}_{1:T} \mid \boldsymbol{x}_0) = q(\boldsymbol{x}_1 \mid \boldsymbol{x}_0)\prod_{t=2}^T q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}, \boldsymbol{x}_0)$ and substituting Bayes' rule $q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}, \boldsymbol{x}_0) = \frac{q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0) q(\boldsymbol{x}_t \mid \boldsymbol{x}_0)}{q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0)}$ yields a telescoping product $\prod_{t=2}^T \frac{q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0)}{q(\boldsymbol{x}_t \mid \boldsymbol{x}_0)} = \frac{q(\boldsymbol{x}_1 \mid \boldsymbol{x}_0)}{q(\boldsymbol{x}_T \mid \boldsymbol{x}_0)}$. Canceling $q(\boldsymbol{x}_1 \mid \boldsymbol{x}_0)$ leaves $\log \frac{q(\boldsymbol{x}_T \mid \boldsymbol{x}_0)}{p(\boldsymbol{x}_T)}$, which integrates to $\mathcal{L}_T$, while the remaining terms yield $\sum_{t=2}^T \mathcal{L}_{t-1} + \mathcal{L}_0$ [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 2.5, pp. 19–20).
>
> **Noise Reparameterization:** Fixing $\Sigma_\theta(\boldsymbol{x}_t, t) = \sigma_t^2 \boldsymbol{I}$ (typically $\sigma_t^2 = \beta_t$ or $\tilde{\beta}_t$), each Gaussian KL term $\mathcal{L}_{t-1}$ reduces to
>
> $$
> \mathcal{L}_{t-1} = \mathbb{E}_{q(\boldsymbol{x}_t \mid \boldsymbol{x}_0)}\!\left[\frac{1}{2\sigma_t^2}\lVert \tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0) - \boldsymbol{\mu}_\theta(\boldsymbol{x}_t, t) \rVert_2^2\right] + C_t.
>
> $$
>
> Inverting `(eq:ddpm:marginal_q)` as $\boldsymbol{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\boldsymbol{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon})$ and substituting into `(eq:ddpm:posterior_params)` gives
>
> $$
> \tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0) = \frac{1}{\sqrt{\alpha_t}}\!\left(\boldsymbol{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}\right).
>
> $$
>
> Parameterizing $\boldsymbol{\mu}_\theta(\boldsymbol{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\boldsymbol{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)\right)$ transforms $\mathcal{L}_{t-1}$ into a weighted noise-prediction loss, and dropping the time-dependent prefactor yields the unweighted Ho et al. objective [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. (25.24), p. 871) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Eq. (18.39), p. 362):
>
> $$
> \mathcal{L}_{\mathrm{simple}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}\{1,\dots,T\},  \boldsymbol{x}_0 \sim p_{\mathrm{data}},  \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})}\!\left[\lVert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\!\left(\sqrt{\bar{\alpha}_t}\boldsymbol{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon},  t\right) \rVert_2^2\right].
>
> $$

> [!important] Denoising Score Matching (Vincent's Theorem) & Tweedie's Formula
> To estimate the score $\nabla_{\tilde{\boldsymbol{x}}} \log q_\sigma(\tilde{\boldsymbol{x}})$ without knowing $p_{\mathrm{data}}$, Explicit Score Matching minimizes $J_{\mathrm{ESM}}(\theta) = \frac{1}{2}\mathbb{E}_{q_\sigma(\tilde{\boldsymbol{x}})}[\lVert \boldsymbol{s}_\theta(\tilde{\boldsymbol{x}}, \sigma) - \nabla_{\tilde{\boldsymbol{x}}} \log q_\sigma(\tilde{\boldsymbol{x}}) \rVert_2^2]$, which is intractable. Denoising Score Matching (DSM) replaces the marginal score with the tractable conditional score $\nabla_{\tilde{\boldsymbol{x}}} \log q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}_0) = -\frac{\tilde{\boldsymbol{x}} - \boldsymbol{x}_0}{\sigma^2} = -\frac{\boldsymbol{\epsilon}}{\sigma}$:
>
> $$
> J_{\mathrm{DSM}}(\theta; \sigma) = \frac{1}{2}\mathbb{E}_{p_{\mathrm{data}}(\boldsymbol{x}_0)  q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}_0)}\!\left[\lVert \boldsymbol{s}_\theta(\tilde{\boldsymbol{x}}, \sigma) - \nabla_{\tilde{\boldsymbol{x}}} \log q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}_0) \rVert_2^2\right] = J_{\mathrm{ESM}}(\theta; \sigma) + C,
>
> $$
>
> where $C$ is independent of $\theta$ [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Eq. (87), p. 37).
>
> *Proof:* Expanding $J_{\mathrm{ESM}}(\theta)$ leaves a cross-term $-\mathbb{E}_{q_\sigma(\tilde{\boldsymbol{x}})}[\boldsymbol{s}_\theta(\tilde{\boldsymbol{x}})^\top \nabla_{\tilde{\boldsymbol{x}}} \log q_\sigma(\tilde{\boldsymbol{x}})]$. Using $q_\sigma(\tilde{\boldsymbol{x}})\nabla_{\tilde{\boldsymbol{x}}}\log q_\sigma(\tilde{\boldsymbol{x}}) = \nabla_{\tilde{\boldsymbol{x}}} q_\sigma(\tilde{\boldsymbol{x}}) = \int p_{\mathrm{data}}(\boldsymbol{x}_0) \nabla_{\tilde{\boldsymbol{x}}} q_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x}_0) \mathrm{d}\boldsymbol{x}_0 = \int p_{\mathrm{data}}(\boldsymbol{x}_0) q_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x}_0) \nabla_{\tilde{\boldsymbol{x}}}\log q_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x}_0) \mathrm{d}\boldsymbol{x}_0$, this cross-term equals $-\mathbb{E}_{p_{\mathrm{data}}(\boldsymbol{x}_0)q_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x}_0)}[\boldsymbol{s}_\theta(\tilde{\boldsymbol{x}})^\top \nabla_{\tilde{\boldsymbol{x}}}\log q_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x}_0)]$, matching $J_{\mathrm{DSM}}(\theta)$ up to $\theta$-independent variance constants [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 3.3, pp. 37–38).
>
> **Tweedie's Formula & the DDPM–NCSN Bridge:** Under the forward kernel $\boldsymbol{x}_t = \sqrt{\bar{\alpha}_t}\boldsymbol{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ with $\sigma_t = \sqrt{1-\bar{\alpha}_t}$, the Bayes-optimal MMSE estimators $\boldsymbol{\epsilon}^*(\boldsymbol{x}_t, t) = \mathbb{E}[\boldsymbol{\epsilon} \mid \boldsymbol{x}_t]$ and $\boldsymbol{x}_0^*(\boldsymbol{x}_t, t) = \mathbb{E}[\boldsymbol{x}_0 \mid \boldsymbol{x}_t]$ satisfy Tweedie's identity [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 6.3, p. 179) [[Source - Foundations|deep_learning_foundations_concepts]] (Eq. (20.53), p. 597):
>
> $$
> \nabla_{\boldsymbol{x}_t} \log q_t(\boldsymbol{x}_t) = -\frac{1}{\sqrt{1 - \bar{\alpha}_t}} \mathbb{E}[\boldsymbol{\epsilon} \mid \boldsymbol{x}_t] = \frac{\sqrt{\bar{\alpha}_t} \mathbb{E}[\boldsymbol{x}_0 \mid \boldsymbol{x}_t] - \boldsymbol{x}_t}{1 - \bar{\alpha}_t}.
>
> $$
>
> Thus, training an $\boldsymbol{\epsilon}$-prediction DDPM is mathematically equivalent to multi-scale DSM with $\boldsymbol{s}_\theta(\boldsymbol{x}_t, t) \equiv -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.3.3, p. 876).

> [!important] Four-Way Parameterization Equivalence ($\boldsymbol{x}_0, \boldsymbol{\epsilon}, \boldsymbol{s}, \boldsymbol{v}$)
> Consider any differentiable affine perturbation $\boldsymbol{x}_t = \alpha_t^{\mathrm{aff}} \boldsymbol{x}_0 + \sigma_t \boldsymbol{\epsilon}$ with $\sigma_t > 0$ and non-degenerate Jacobian $(\alpha_t^{\mathrm{aff}})'\sigma_t - \alpha_t^{\mathrm{aff}}\sigma_t' \neq 0$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 6.3.3, pp. 179–180). Define the four oracle targets $\boldsymbol{x}_0^* = \mathbb{E}[\boldsymbol{x}_0 \mid \boldsymbol{x}_t]$, $\boldsymbol{\epsilon}^* = \mathbb{E}[\boldsymbol{\epsilon} \mid \boldsymbol{x}_t]$, $\boldsymbol{s}^* = \nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$, and velocity $\boldsymbol{v}^* = \mathbb{E}[(\alpha_t^{\mathrm{aff}})'\boldsymbol{x}_0 + \sigma_t'\boldsymbol{\epsilon} \mid \boldsymbol{x}_t]$. Under the algebraic head conversions
>
> $$
> \boldsymbol{s}_\theta \equiv -\frac{1}{\sigma_t}\boldsymbol{\epsilon}_\theta \equiv \frac{\alpha_t^{\mathrm{aff}}\boldsymbol{x}_\theta - \boldsymbol{x}_t}{\sigma_t^2} \equiv -\frac{\alpha_t^{\mathrm{aff}} \boldsymbol{v}_\theta - (\alpha_t^{\mathrm{aff}})' \boldsymbol{x}_t}{\sigma_t\!\left((\alpha_t^{\mathrm{aff}})'\sigma_t - \alpha_t^{\mathrm{aff}}\sigma_t'\right)},
>
> $$
>
> all four per-sample squared regression losses are identical up to known time-dependent weights:
>
> $$
> \lVert \boldsymbol{s}_\theta - \boldsymbol{s}^* \rVert_2^2 = \frac{1}{\sigma_t^2}\lVert \boldsymbol{\epsilon}_\theta - \boldsymbol{\epsilon}^* \rVert_2^2 = \frac{(\alpha_t^{\mathrm{aff}})^2}{\sigma_t^4}\lVert \boldsymbol{x}_\theta - \boldsymbol{x}_0^* \rVert_2^2 = \frac{(\alpha_t^{\mathrm{aff}})^2}{\sigma_t^2\left((\alpha_t^{\mathrm{aff}})'\sigma_t - \alpha_t^{\mathrm{aff}}\sigma_t'\right)^2}\lVert \boldsymbol{v}_\theta - \boldsymbol{v}^* \rVert_2^2.
>
> $$

## Algorithmic Implementation & Comparative Synthesis

Both DDPM and NCSN parameterize the denoiser with a single time- or noise-conditioned U-Net augmented by sinusoidal positional embeddings [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 18.6.1, p. 363) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.2.5, p. 873). Algorithm `alg:ddpm:train_sample` details DDPM training and ancestral sampling [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Algs. 25.1–25.2, pp. 871–872), whereas Algorithm `alg:ddpm:ald` details NCSN sampling via Annealed Langevin Dynamics [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 3.4, p. 82) [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 20.3.3, p. 598).

> [!example] Algorithm: DDPM Training ($\mathcal{L}_{\mathrm{simple}}$) and Ancestral Sampling
> **Input:** Dataset $p_{\mathrm{data}}(\boldsymbol{x}_0)$, schedule $\{\beta_t\}_{t=1}^T$ ($\alpha_t = 1-\beta_t$, $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$), reverse variance $\sigma_t^2$
> **Output:** Trained denoiser $\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)$ and generated sample $\boldsymbol{x}_0$
>  *// Phase 1: Training via Unweighted Denoising Objective*
> **while** not converged:
>
>   Sample $\boldsymbol{x}_0 \sim p_{\mathrm{data}}(\boldsymbol{x}_0)$, $t \sim \mathcal{U}\{1,\dots,T\}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$
>   Construct $\boldsymbol{x}_t \leftarrow \sqrt{\bar{\alpha}_t} \boldsymbol{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$
>   Take gradient descent step on $\nabla_\theta \lVert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t) \rVert_2^2$
>
>  *// Phase 2: Ancestral Reverse Sampling*
> Sample $\boldsymbol{x}_T \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$
> **for** $t = T, T-1, \dots, 1$:
>
>   Draw $\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ if $t > 1$, else $\boldsymbol{z} \leftarrow \boldsymbol{0}$
>   $\boldsymbol{x}_{t-1} \leftarrow \frac{1}{\sqrt{\alpha_t}}\left(\boldsymbol{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)\right) + \sigma_t \boldsymbol{z}$
>
> **return** $\boldsymbol{x}_0$

> [!example] Algorithm: NCSN Sampling via Annealed Langevin Dynamics (ALD)
> **Input:** Trained score network $\boldsymbol{s}_\theta(\boldsymbol{x}, \sigma_i)$, noise levels $\sigma_1 > \sigma_2 > \dots > \sigma_L$, step scale $\epsilon_0$, inner steps $K$
> **Output:** Generated sample $\tilde{\boldsymbol{x}}_0$
> Initialize $\tilde{\boldsymbol{x}}_0 \sim \mathcal{U}([-1,1]^D)$ or $\mathcal{N}(\boldsymbol{0}, \sigma_1^2 \boldsymbol{I})$
> **for** $i = 1, 2, \dots, L$:
>
>   Set step size $\tau_i \leftarrow \epsilon_0 \cdot (\sigma_i / \sigma_L)^2$
>   **for** $k = 1, 2, \dots, K$:
>
>     Draw $\boldsymbol{z}_k \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$
>     $\tilde{\boldsymbol{x}}_k \leftarrow \tilde{\boldsymbol{x}}_{k-1} + \frac{\tau_i}{2} \boldsymbol{s}_\theta(\tilde{\boldsymbol{x}}_{k-1}, \sigma_i) + \sqrt{\tau_i} \boldsymbol{z}_k$
>
>   $\tilde{\boldsymbol{x}}_0 \leftarrow \tilde{\boldsymbol{x}}_K$
>
> **return** $\tilde{\boldsymbol{x}}_0$

**Table:** Architectural, mathematical, and computational comparison of DDPM and NCSN [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Table 3.1, p. 84) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.3.3, p. 876).

| **Dimension** | **DDPM (Variational / VP)** | **NCSN (Score-Based / VE)** |
| — | — | — |
| Forward transition | $\boldsymbol{x}_t = \sqrt{1-\beta_t} \boldsymbol{x}_{t-1} + \sqrt{\beta_t} \boldsymbol{\epsilon}$ (variance-preserving) | $\tilde{\boldsymbol{x}}_i = \boldsymbol{x}_0 + \sigma_i\boldsymbol{\epsilon}$ (variance-exploding across scales) |
| Prior distribution | $p(\boldsymbol{x}_T) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ | $p_{\mathrm{prior}}(\boldsymbol{x}_1) = \mathcal{N}(\boldsymbol{0}, \sigma_1^2 \boldsymbol{I})$ |
| Training target | $\mathbb{E}_{t,\boldsymbol{x}_0,\boldsymbol{\epsilon}}[\lVert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t) \rVert_2^2]$ | $\sum_{i=1}^L \lambda_i \mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{\epsilon}}[\lVert \boldsymbol{s}_\theta(\tilde{\boldsymbol{x}}, \sigma_i) + \boldsymbol{\epsilon}/\sigma_i \rVert_2^2]$ |
| Sampling dynamics | Ancestral Markov chain $p_\theta(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)$ | Annealed Langevin MCMC across $\sigma_1 \to \sigma_L$ |
| Sequential NFE cost | $\mathcal{O}(T)$ network passes ($T \approx 1000$) | $\mathcal{O}(LK)$ network passes ($L \approx 10$–$1000$, $K \ge 1$) |

> [!info] Cross-Book Synthesis: Variational Diffusion vs. Score-Based Langevin Dynamics
> All five texts highlight a remarkable structural convergence: [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 2–3) and [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 18) frame DDPM as a fixed-encoder hierarchical VAE whose ELBO collapses to denoising least squares, whereas [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Ch. 3) and [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 20.3) derive NCSNs from Energy-Based Models to bypass intractable partition functions. [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.3.3) and [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 3.5) unify both as discrete approximations to continuous-time stochastic differential equations (VP-SDE vs. VE-SDE), sharing the same U-Net denoiser and the same sequential $\mathcal{O}(T)$ / $\mathcal{O}(LK)$ sampling bottleneck.

## Visual Diagnostics & Geometric Structure

Figure `fig:ddpm:forward_reverse_chain` synthesizes the Markovian forward/reverse graphical structure alongside the four-way Tweedie–score equivalence, while Figure `fig:ddpm:noise_schedules` contrasts the linear $\beta_t$ and cosine noise schedules in terms of signal retention $\bar{\alpha}_t$ and log-SNR.

> [!example] Visual Diagram (LaTeX / PDF)
> *Unified DDPM/NCSN computational graph: the forward Markov chain admits a closed-form Gaussian jump $q(\boldsymbol{x}_t \mid \boldsymbol{x}_0)$, while the learned reverse denoiser $p_\theta(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)$ can be equivalently parameterized via $\boldsymbol{x}_0$, $\boldsymbol{\epsilon}$, score $\boldsymbol{s}_\theta$, or velocity $\boldsymbol{v}_\theta$ through Tweedie's identity `(eq:ddpm:tweedie)` and `(eq:ddpm:head_conversions)`.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Comparison of the Ho et al. linear schedule ($\beta_1=10^{-4}, \beta_T=0.02, T=1000$, approximated continuously by $\bar{\alpha}(t)=\exp(-0.1t-9.95t^2)$) against the Nichol–Dhariwal cosine schedule ($s=0.008$). The linear schedule destroys signal prematurely ($t > 0.6$), whereas the cosine schedule maintains an approximately linear log-SNR decay.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:ddpm:1d_diffusion` illustrates how the forward Markov chain smooths a multimodal 1D density into $\mathcal{N}(0, 1)$, while Figure `fig:ddpm:posterior_tractable` visualizes the exact conditional reverse posterior $q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0)$ against the forward kernel. Figure `fig:ddpm:swiss_roll` demonstrates the learned score vector field and Langevin sampling trajectories on the 2D Swiss roll manifold.

> [!note] Source Figure
> *Figure:* Forward Gaussian convolution gradually transforms a multimodal 1D data distribution $q(\boldsymbol{x}_0)$ into an isotropic Gaussian $q(\boldsymbol{x}_T)$, erasing high-frequency details before coarse modes [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 25.2, p. 868). (`probabilisticml_kevinmurphy_advancedtopics_p902_fig2.png`)

> [!note] Source Figure
> *Figure:* Diffusion kernel $q(\boldsymbol{z}_t \mid \boldsymbol{x}^*)$ (left) and tractable reverse conditional Gaussian $q(\boldsymbol{z}_{t-1} \mid \boldsymbol{z}_t^*, \boldsymbol{x}^*)$ (right, teal) compared with $q(\boldsymbol{z}_t^* \mid \boldsymbol{z}_{t-1})$ (orange) and $q(\boldsymbol{z}_{t-1} \mid \boldsymbol{x}^*)$ (gray) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Fig. 18.6, p. 354). (`understandingdl_simonprince_p368_fig2.png`)

> [!note] Source Figure
> *Figure:* Score-based generative modeling on the 2D Swiss roll: (a) empirical data, (b) learned score vector field $\boldsymbol{s}_\theta(\boldsymbol{x})$, (c) superposition on data density, and (d) Langevin sampling trajectories [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 25.5, p. 875). (`probabilisticml_kevinmurphy_advancedtopics_p909_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Manifold Hypothesis, Low-Density Score Blindness & Mode Mixing
> **Failure Mode:** Training a single-scale score network $\boldsymbol{s}_\theta(\boldsymbol{x})$ on unperturbed data fails for three fundamental reasons [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 20.3.3, pp. 597–598) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.3.2, p. 874): (i) under the manifold hypothesis, $p_{\mathrm{data}}$ is supported on a low-dimensional manifold so $\nabla_{\boldsymbol{x}}\log p_{\mathrm{data}}(\boldsymbol{x})$ is undefined off-manifold; (ii) in low-density regions, DSM has vanishing sample weight and yields erratic score estimates; (iii) for disjoint modes $p_{\mathrm{data}} = \pi p_0 + (1-\pi)p_1$, $\nabla_{\boldsymbol{x}} \log p_{\mathrm{data}}(\boldsymbol{x})$ is independent of $\pi$, preventing Langevin dynamics from recovering true mixture weights.
> **Remedy:** Perturb data across a geometric ladder of noise scales $\sigma_1 > \dots > \sigma_L$ where $\sigma_1$ is large enough to bridge disjoint modes across $\mathbb{R}^D$, and anneal Langevin step sizes $\tau_i \propto (\sigma_i / \sigma_L)^2$ down to $\sigma_L \approx 0.01$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 3.4, p. 80).

> [!warning] Linear Schedule SNR Collapse & Endpoint Instability
> **Failure Mode:** Under a linear $\beta_t$ schedule, $\bar{\alpha}_t$ decays super-exponentially for $t > 0.6T$ (Figure `fig:ddpm:noise_schedules`), wasting forward steps in near-pure noise. Moreover, predicting $\boldsymbol{\epsilon}_\theta$ or $\boldsymbol{s}_\theta$ near $t \to 0$ ($\sigma_t \to 0$) amplifies score variance as $1/\sigma_t$, whereas predicting $\boldsymbol{x}_0$ near $t \to T$ ($\alpha_t^{\mathrm{aff}} \to 0$) suffers from signal collapse [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 6.3.3, p. 182).
> **Remedy:** Adopt a cosine or log-SNR-linear schedule [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.2.4, p. 872), use low-discrepancy stratified sampling $t_i = \bmod(u_0 + i/k, 1)$ for minibatch variance reduction, or adopt $\boldsymbol{v}$-prediction ($\boldsymbol{v}_\theta = \alpha_t'\boldsymbol{x}_0 + \sigma_t'\boldsymbol{\epsilon}$) whose target norm remains bounded at both endpoints [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Eq. (6.3.7), p. 181).

> [!warning] Reverse Gaussianity Breakdown & Sequential NFE Bottleneck
> **Failure Mode:** The reverse conditional $p_\theta(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)$ is only accurately approximated by an isotropic Gaussian when step sizes $\beta_t \to 0$ and $T \to \infty$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 18.3, p. 356). Truncating DDPM to small $T$ without changing the sampler induces severe discretization error, while large $T \approx 1000$ or $\mathcal{O}(LK)$ ALD steps makes inference prohibitively slow [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 3.5, p. 84).
> **Remedy:** Formulate the forward/reverse chains as continuous-time Score SDEs and Probability Flow ODEs, unlocking deterministic DDIM and high-order ODE solvers that achieve comparable sample quality in $10$–$50$ NFEs.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (1):** [[DDPM]]
- **Prerequisites (Directed $\leftarrow$):** [[Score Matching (DSM & SSM)]], [[Variational Inference & ELBO]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Classifier-Free Guidance (CFG)]], [[Consistency Models]], [[Score-Based SDE]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Classifier-Free Guidance (CFG)]], [[Consistency Models]], [[Score Matching (DSM & SSM)]], [[Score-Based SDE]], [[Semantic & Instance Segmentation (FCN, U-Net & SAM)]], [[Variational Autoencoder (VAE)]]
- **Synthesized Sources:** [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]], [[Source - Understanding Deep Learning]]
