---
type: monograph
domain: "D04"
parent_topic: "[[PT 04 - Autoencoders, VAEs, Normalizing Flows, EBMs & GANs]]"
sub_topic: "[[ST 04 - Energy-Based Models, Score Matching & Adversarial Networks]]"
cheatsheet: "[[Cheatsheet - Energy-Based Models, Partition Functions, Contrastive Divergence & Score Matching]]"
part_of: 
  - "[[ST 04 - Energy-Based Models, Score Matching & Adversarial Networks]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
concepts: 
  - "[[Energy-Based Models (EBM)]]"
  - "[[Score Matching (DSM & SSM)]]"
prerequisites: 
  - "[[Automatic Differentiation (AD)]]"
  - "[[Energy-Based Models (EBM)]]"
enables: 
  - "[[DDPM]]"
  - "[[Score Matching (DSM & SSM)]]"
  - "[[Score-Based SDE]]"
related: 
  - "[[Automatic Differentiation (AD)]]"
  - "[[DDPM]]"
  - "[[Energy-Based Models (EBM)]]"
  - "[[Graphical Models (BN & MRF)]]"
  - "[[MCMC & HMC Sampling]]"
  - "[[Score Matching (DSM & SSM)]]"
  - "[[Score-Based SDE]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
  - "[[Source - Tutorial on Diffusion Models for Imaging and Vision]]"
tags:
  - monograph
  - domain-04
---

# Energy-Based Models, Partition Functions, Contrastive Divergence & Score Matching

> [!abstract] Monograph Navigation
> - **Domain:** [[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion|Deep Generative Modeling: VAEs, Flows, EBMs, GANs & Diffusion]]
> - **Parent / Sub-Topic:** [[PT 04 - Autoencoders, VAEs, Normalizing Flows, EBMs & GANs|Autoencoders, VAEs, Normalizing Flows, EBMs & GANs]] → [[ST 04 - Energy-Based Models, Score Matching & Adversarial Networks|Energy-Based Models, Score Matching & Adversarial Networks]]
> - **Visual Cheatsheet:** [[Cheatsheet - Energy-Based Models, Partition Functions, Contrastive Divergence & Score Matching]]
> - **Description:** Addresses intractable normalizing constants Z(theta) via MLE positive/negative phases, Contrastive Divergence (CD-k), Noise Contrastive Estimation (NCE), Hyvärinen score matching, sliced score matching, and Vincent's denoising score matching equivalence.

## Problem Formulation & Unified Notation

Directed generative models (variational autoencoders, autoregressive models, and normalizing flows) construct normalized probability densities by composing locally normalized conditionals or tractable Jacobians. By contrast, *Energy-Based Models* (EBMs) specify an unnormalized Gibbs–Boltzmann density directly over $\boldsymbol{x} \in \mathbb{R}^D$ via a scalar energy function $E_\theta : \mathbb{R}^D \to \mathbb{R}$ parameterized by $\theta$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 24, p. 849 (PDF p. 883)):

$$
p_\theta(\boldsymbol{x}) = \frac{\exp(-E_\theta(\boldsymbol{x}))}{Z_\theta}, \qquad Z_\theta = \int_{\mathbb{R}^D} \exp(-E_\theta(\boldsymbol{x}))   \mathrm{d}\boldsymbol{x},
$$

where $Z_\theta$ is the *partition function* ensuring $\int_{\mathbb{R}^D} p_\theta(\boldsymbol{x}) \mathrm{d}\boldsymbol{x} = 1$, or a discrete sum $\sum_{\boldsymbol{x}} \exp(-E_\theta(\boldsymbol{x}))$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 18, p. 605 (PDF p. 620)). Lower energy $E_\theta(\boldsymbol{x})$ corresponds to higher probability density. Because $E_\theta$ is not constrained to integrate to unity, any smooth scalar neural network architecture is admissible, and independent constraints combine additively via Products of Experts (PoE), $E_{12}(\boldsymbol{x}) = E_1(\boldsymbol{x}) + E_2(\boldsymbol{x})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 24.1.1, p. 850 (PDF p. 884)).

However, this architectural freedom comes at a steep computational price: in high dimensions $D$, evaluating $Z_\theta$ requires an intractable $D$-dimensional integral, rendering exact likelihood evaluation $p_\theta(\boldsymbol{x})$ and direct ancestral sampling impossible. Addressing this intractability motivates two broad families of methods:
- **Partition-Gradient Approximation (MLE / MCMC / CD / PCD / NCE):** Estimating $\nabla_\theta \log Z_\theta$ via Markov Chain Monte Carlo (MCMC) or bypassing it via binary classification against a tractable noise distribution.
- **Partition-Free Score Estimation (SM / SSM / DSM):** Eliminating $Z_\theta$ entirely by differentiating $\log p_\theta(\boldsymbol{x})$ with respect to the data $\boldsymbol{x}$ rather than $\theta$, yielding the *Stein score* $\nabla_{\boldsymbol{x}} \log p_\theta(\boldsymbol{x}) = -\nabla_{\boldsymbol{x}} E_\theta(\boldsymbol{x})$.

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> | **Concept** | **Unified Monograph** | **Murphy Adv. / Goodfellow** | **Diff. Principles / Chan** |
> | — | — | — | — |
> | Energy / Potential | $E_\theta(\boldsymbol{x}) \in \mathbb{R}$ | $E_\theta(\boldsymbol{x})$ (Murphy); $\tilde{p}(\boldsymbol{x};\theta) = e^{-E(\boldsymbol{x};\theta)}$ (Goodfellow) | $E_\phi(\boldsymbol{x})$ (Diff. Princ.); $U(\boldsymbol{x})$ (Chan) |
> | Model Parameters | $\theta$ | $\theta$ (both) | $\phi$ (Diff. Princ.); $\theta$ (Chan) |
> | Stein Score ($\boldsymbol{x}$-grad) | $\boldsymbol{s}_\theta(\boldsymbol{x}) = \nabla_{\boldsymbol{x}} \log p_\theta(\boldsymbol{x})$ | $\boldsymbol{s}_\theta(\boldsymbol{x})$ (Hyv\"arinen score, Murphy Ch. 24) | $\boldsymbol{s}_\phi(\boldsymbol{x})$ (Diff. Princ.); $\boldsymbol{s}_\theta(\boldsymbol{x})$ vs. $\boldsymbol{s}_{\boldsymbol{x}}(\theta)$ (Chan) |
> | Data / Noisy Density | $p_{\mathcal{D}}(\boldsymbol{x})$, $q_\sigma(\tilde{\boldsymbol{x}})$ | $p_D(\boldsymbol{x})$, $q(\tilde{\boldsymbol{x}})$ (Murphy); $p_{\mathrm{data}}, p_{\mathrm{smoothed}}$ (Goodfellow) | $p_{\mathrm{data}}(\boldsymbol{x})$, $p_\sigma(\tilde{\boldsymbol{x}})$ (Diff. Princ.); $q(\boldsymbol{x})$ (Chan) |
> | Score Matching Loss | $D_F(p_{\mathcal{D}} \parallel  p_\theta)$ | $D_F(p_D \parallel  p_\theta)$ (Fisher divergence) | $L_{\mathrm{SM}}(\phi)$ (Diff. Princ.); $J_{\mathrm{ESM}}(\theta)$ (Chan) |
>
> 
> *Note:* [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 3.2, p. 33) explicitly warns against conflating *Stein's score function* $\boldsymbol{s}_\theta(\boldsymbol{x}) = \nabla_{\boldsymbol{x}} \log p_\theta(\boldsymbol{x})$ (used in Langevin dynamics and score matching) with the *ordinary Fisher score* $\boldsymbol{s}_{\boldsymbol{x}}(\theta) = \nabla_\theta \log p_\theta(\boldsymbol{x})$ (used in maximum likelihood estimation).

## Core Mathematical Derivations & Theorems

### Maximum Likelihood Gradient: Positive & Negative Phases

Fitting $p_\theta(\boldsymbol{x})$ to an empirical data distribution $p_{\mathcal{D}}(\boldsymbol{x})$ by maximizing $\ell(\theta) = \mathbb{E}_{\boldsymbol{x} \sim p_{\mathcal{D}}}[\log p_\theta(\boldsymbol{x})]$ is equivalent to minimizing $D_{\mathrm{KL}}(p_{\mathcal{D}} \parallel p_\theta)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 24.2, p. 851 (PDF p. 885)). Differentiating $\ell(\theta)$ with respect to $\theta$ exposes the partition function bottleneck.

> [!important] Theorem: EBM Log-Likelihood Gradient Decomposition
> Assume $\tilde{p}_\theta(\boldsymbol{x}) = \exp(-E_\theta(\boldsymbol{x}))$ is Lebesgue-integrable and satisfies Leibniz regularity conditions for differentiating under the integral sign [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 18.1, pp. 606–607 (PDF pp. 621–622)). Then the gradient of the log-partition function is
>
> $$
> \nabla_\theta \log Z_\theta = \mathbb{E}_{\boldsymbol{x} \sim p_\theta}[-\nabla_\theta E_\theta(\boldsymbol{x})],
> $$
>
> and consequently the expected log-likelihood gradient decomposes into a *positive phase* and a *negative phase*:
>
> $$
> \nabla_\theta \ell(\theta) = \underbrace{\mathbb{E}_{\boldsymbol{x} \sim p_{\mathcal{D}}}[-\nabla_\theta E_\theta(\boldsymbol{x})]}_{\text{Positive phase (push energy down)}} - \underbrace{\mathbb{E}_{\tilde{\boldsymbol{x}} \sim p_\theta}[-\nabla_\theta E_\theta(\tilde{\boldsymbol{x}})]}_{\text{Negative phase (push energy up)}}.
> $$
>
> *Proof.* Expanding $\nabla_\theta \log Z_\theta$ directly via the chain rule and Leibniz's rule yields:
>
> $$
> \begin{aligned}
> \nabla_\theta \log Z_\theta
>   &= \frac{1}{Z_\theta} \nabla_\theta \int_{\mathbb{R}^D} \exp(-E_\theta(\boldsymbol{x}))   \mathrm{d}\boldsymbol{x}
>    = \frac{1}{Z_\theta} \int_{\mathbb{R}^D} \nabla_\theta \exp(-E_\theta(\boldsymbol{x}))   \mathrm{d}\boldsymbol{x} \\
>   &= \int_{\mathbb{R}^D} \frac{\exp(-E_\theta(\boldsymbol{x}))}{Z_\theta} \bigl(-\nabla_\theta E_\theta(\boldsymbol{x})\bigr)   \mathrm{d}\boldsymbol{x}
>    = \mathbb{E}_{\boldsymbol{x} \sim p_\theta}\bigl[-\nabla_\theta E_\theta(\boldsymbol{x})\bigr].
> \end{aligned}
> $$
>
> Substituting this identity into $\nabla_\theta \log p_\theta(\boldsymbol{x}) = -\nabla_\theta E_\theta(\boldsymbol{x}) - \nabla_\theta \log Z_\theta$ and taking expectations under $p_{\mathcal{D}}$ gives `(eq:ebm:mle_grad)` [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. 24.7–24.15, p. 851 (PDF p. 885)).  $\square$

> [!tip] Wake–Sleep Dynamics of the Positive and Negative Phases
> As detailed in [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 18.1, pp. 608–609 (PDF pp. 623–624)), equation `(eq:ebm:mle_grad)` balances two opposing forces:
> - **Positive (clamped / Hebbian) phase:** Evaluates $-\nabla_\theta E_\theta(\boldsymbol{x})$ on real data $\boldsymbol{x} \sim p_{\mathcal{D}}$, lowering energy (raising unnormalized density) on the data manifold.
> - **Negative (unclamped / anti-Hebbian) phase:** Evaluates $+\nabla_\theta E_\theta(\tilde{\boldsymbol{x}})$ on model-generated samples ("fantasy particles") $\tilde{\boldsymbol{x}} \sim p_\theta$, raising energy in regions where the model places spurious probability mass.
>
> At equilibrium ($p_\theta = p_{\mathcal{D}}$), the two expectations cancel identically and $\nabla_\theta \ell(\theta) = \boldsymbol{0}$.

> [!note] Source Figure
> *Figure:* Positive and negative phases of EBM maximum likelihood training. Left: the positive phase pushes up $p_{\mathrm{model}}(x)$ on real data samples. Right: the negative phase pushes down $p_{\mathrm{model}}(x)$ on model samples, preventing trivial energy collapse [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 18, Fig. 18.1, p. 609 (PDF p. 624)). (`deep_learning_goodfellow_bengio_courville_2017_p624_fig1.png`)

### Contrastive Divergence & Langevin MCMC

Drawing exact independent samples $\tilde{\boldsymbol{x}} \sim p_\theta$ at every gradient step is computationally prohibitive. *Overdamped Langevin MCMC* exploits the fact that the Stein score is independent of $Z_\theta$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 3.1.4, p. 61 (PDF p. 66)):

$$
\boldsymbol{x}_{k+1} = \boldsymbol{x}_k - \frac{\epsilon^2}{2}\nabla_{\boldsymbol{x}} E_\theta(\boldsymbol{x}_k) + \epsilon \boldsymbol{z}_k, \qquad \boldsymbol{z}_k \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}),
$$

where step size $\epsilon > 0$ (alternatively written with $\eta = \epsilon^2/2$ as $\boldsymbol{x}_{k+1} = \boldsymbol{x}_k + \eta \boldsymbol{s}_\theta(\boldsymbol{x}_k) + \sqrt{2\eta} \boldsymbol{z}_k$ [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Eq. 75, p. 33)). In the limit $\epsilon \to 0$, `(eq:ebm:langevin)` is the Euler–Maruyama discretization of the Langevin SDE $\mathrm{d}\boldsymbol{x}(t) = -\nabla_{\boldsymbol{x}} E_\theta(\boldsymbol{x}(t)) \mathrm{d}t + \sqrt{2} \mathrm{d}\boldsymbol{w}(t)$, whose unique stationary distribution is $p_\theta(\boldsymbol{x})$.

To avoid burning in MCMC chains from random noise at each iteration, *$k$-step Contrastive Divergence* ($\mathrm{CD}_k$) initializes the chain at a data point $\boldsymbol{x}_0 \sim p_{\mathcal{D}}$ and runs only $k$ MCMC steps (e.g., $k=1$ block Gibbs or Langevin steps) to obtain $\boldsymbol{x}_k \sim p_k$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 24.2.2, p. 852 (PDF p. 886)). Rather than minimizing $D_{\mathrm{KL}}(p_{\mathcal{D}} \parallel p_\theta)$, $\mathrm{CD}_k$ approximately minimizes the difference of KL divergences:

$$
\mathrm{CD}_k = D_{\mathrm{KL}}(p_0 \parallel p_\infty) - D_{\mathrm{KL}}(p_k \parallel p_\infty),
$$

where $p_0 = p_{\mathcal{D}}$ and $p_\infty = p_\theta$. Alternatively, *Persistent Contrastive Divergence* (PCD / Stochastic Maximum Likelihood, SML) maintains a persistent buffer of Markov chains across updates, assuming $\theta$ evolves slowly enough that chains track $p_\theta$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 18.2, pp. 610–612 (PDF pp. 625–627)).

> [!example] Visual Diagram (LaTeX / PDF)
> *Two paradigms for confronting the intractable partition function $Z_\theta$. Top: Maximum likelihood / Contrastive Divergence balances a positive phase on data against a negative phase on MCMC/CD samples. Bottom: Score-based training differentiates $\log p_\theta(\boldsymbol{x})$ w.r.t. $\boldsymbol{x}$ so $\nabla_{\boldsymbol{x}} \log Z_\theta = \boldsymbol{0}$, progressing from $O(D^2)$ Implicit Score Matching to $O(D)$ Sliced SM and Hessian-free Denoising Score Matching.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

### Hyv\"arinen's Implicit Score Matching (ISM)

To bypass $Z_\theta$ and MCMC sampling during training, *Score Matching* minimizes the *Fisher divergence* between the data score $\nabla_{\boldsymbol{x}} \log p_{\mathcal{D}}(\boldsymbol{x})$ and the model score $\boldsymbol{s}_\theta(\boldsymbol{x}) = \nabla_{\boldsymbol{x}} \log p_\theta(\boldsymbol{x}) = -\nabla_{\boldsymbol{x}} E_\theta(\boldsymbol{x})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 24.3, p. 856 (PDF p. 890)):

$$
D_F(p_{\mathcal{D}} \| p_\theta) = \frac{1}{2} \mathbb{E}_{\boldsymbol{x} \sim p_{\mathcal{D}}}\!\left[ \lVert \boldsymbol{s}_\theta(\boldsymbol{x}) - \nabla_{\boldsymbol{x}} \log p_{\mathcal{D}}(\boldsymbol{x}) \rVert_2^2 \right].
$$

Although $\nabla_{\boldsymbol{x}} \log p_{\mathcal{D}}(\boldsymbol{x})$ is unknown, Hyv\"arinen (2005) showed that integration by parts eliminates it completely.

> [!important] Theorem: Hyv\"arinen's Implicit Score Matching Identity
> Suppose $p_{\mathcal{D}}(\boldsymbol{x})$ and $\boldsymbol{s}_\theta(\boldsymbol{x})$ are continuously differentiable on $\mathbb{R}^D$, $\mathbb{E}_{p_{\mathcal{D}}}[\lVert \boldsymbol{s}_\theta(\boldsymbol{x}) \rVert_2^2] < \infty$, $\mathbb{E}_{p_{\mathcal{D}}}[\lVert \nabla_{\boldsymbol{x}}\log p_{\mathcal{D}}(\boldsymbol{x}) \rVert_2^2] < \infty$, and the boundary condition $\lim_{|x_i| \to \infty} p_{\mathcal{D}}(\boldsymbol{x}) s_{\theta,i}(\boldsymbol{x}) = 0$ holds for all $i \in \{1,\dots,D\}$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Prop. 3.2.1, p. 65 (PDF p. 70)). Then:
>
> $$
> D_F(p_{\mathcal{D}} \| p_\theta) = \mathbb{E}_{\boldsymbol{x} \sim p_{\mathcal{D}}}\!\left[ \frac{1}{2}\lVert \boldsymbol{s}_\theta(\boldsymbol{x}) \rVert_2^2 + \mathrm{Tr}\bigl(\nabla_{\boldsymbol{x}} \boldsymbol{s}_\theta(\boldsymbol{x})\bigr) \right] + C,
> $$
>
> where $C = \frac{1}{2}\mathbb{E}_{p_{\mathcal{D}}}[\lVert \nabla_{\boldsymbol{x}}\log p_{\mathcal{D}}(\boldsymbol{x}) \rVert_2^2]$ is independent of $\theta$. For an EBM $\boldsymbol{s}_\theta = -\nabla_{\boldsymbol{x}} E_\theta$, $\mathrm{Tr}(\nabla_{\boldsymbol{x}}\boldsymbol{s}_\theta(\boldsymbol{x})) = -\mathrm{Tr}(\nabla_{\boldsymbol{x}}^2 E_\theta(\boldsymbol{x}))$.
>
> *Proof.* Expanding the squared norm in `(eq:ebm:fisher_div)` gives
> \[
>   D_F(p_{\mathcal{D}} \| p_\theta) = \frac{1}{2}\mathbb{E}_{p_{\mathcal{D}}}\!\left[\lVert \boldsymbol{s}_\theta(\boldsymbol{x}) \rVert_2^2\right] - \mathbb{E}_{p_{\mathcal{D}}}\!\left[\boldsymbol{s}_\theta(\boldsymbol{x})^{\top} \nabla_{\boldsymbol{x}} \log p_{\mathcal{D}}(\boldsymbol{x})\right] + C.
> \]
> Using $\nabla_{\boldsymbol{x}} \log p_{\mathcal{D}}(\boldsymbol{x}) = \frac{\nabla_{\boldsymbol{x}} p_{\mathcal{D}}(\boldsymbol{x})}{p_{\mathcal{D}}(\boldsymbol{x})}$, the cross-term integrates by parts coordinatewise:
>
> $$
> \begin{aligned}
> -\mathbb{E}_{p_{\mathcal{D}}}\!\left[\boldsymbol{s}_\theta(\boldsymbol{x})^{\top} \nabla_{\boldsymbol{x}} \log p_{\mathcal{D}}(\boldsymbol{x})\right]
>   &= -\sum_{i=1}^D \int_{\mathbb{R}^D} s_{\theta,i}(\boldsymbol{x}) \frac{\partial p_{\mathcal{D}}(\boldsymbol{x})}{\partial x_i}   \mathrm{d}\boldsymbol{x} \\
>   &= -\sum_{i=1}^D \underbrace{\Bigl[ s_{\theta,i}(\boldsymbol{x}) p_{\mathcal{D}}(\boldsymbol{x}) \Bigr]_{x_i=-\infty}^{x_i=+\infty}}_{=0} + \sum_{i=1}^D \int_{\mathbb{R}^D} p_{\mathcal{D}}(\boldsymbol{x}) \frac{\partial s_{\theta,i}(\boldsymbol{x})}{\partial x_i}   \mathrm{d}\boldsymbol{x} \\
>   &= \mathbb{E}_{p_{\mathcal{D}}}\!\left[\mathrm{Tr}\bigl(\nabla_{\boldsymbol{x}} \boldsymbol{s}_\theta(\boldsymbol{x})\bigr)\right].
> \end{aligned}
> $$
>
> Substituting this back yields `(eq:ebm:ism)`.  $\square$

> [!tip] Geometric Meaning of Stationarity and Negative Divergence
> As highlighted in [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 3.2.1, pp. 65–66 (PDF pp. 70–71)), the two terms of `(eq:ebm:ism)` shape the score field around data modes:
> - The magnitude term $\frac{1}{2}\lVert \boldsymbol{s}_\theta(\boldsymbol{x}) \rVert_2^2$ drives $\boldsymbol{s}_\theta(\boldsymbol{x}) \to \boldsymbol{0}$ on high-density regions, making data modes stationary points.
> - The divergence term $\mathrm{Tr}(\nabla_{\boldsymbol{x}}\boldsymbol{s}_\theta(\boldsymbol{x})) = \nabla_{\boldsymbol{x}} \cdot \boldsymbol{s}_\theta(\boldsymbol{x})$ penalizes positive divergence, encouraging nearby score vectors to converge inward so stationary points act as attractive sinks.

### Sliced Score Matching & Denoising Score Matching

Computing $\mathrm{Tr}(\nabla_{\boldsymbol{x}}\boldsymbol{s}_\theta(\boldsymbol{x}))$ in `(eq:ebm:ism)` requires $D$ backward passes (an $O(D^2)$ Hessian-diagonal cost), which is prohibitive for high-dimensional images. Two scalable alternatives resolve this bottleneck:

**1. Sliced Score Matching (SSM)..** 
Song et al. (2019) project both scores onto random isotropic probe vectors $\boldsymbol{v} \sim p(\boldsymbol{v})$ satisfying $\mathbb{E}[\boldsymbol{v}\boldsymbol{v}^{\top}] = \boldsymbol{I}$ (e.g., $\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$ or Rademacher), as reviewed in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 24.3.3, pp. 858–859 (PDF pp. 892–893)). By Hutchinson's trace identity $\mathrm{Tr}(\boldsymbol{A}) = \mathbb{E}_{\boldsymbol{v}}[\boldsymbol{v}^{\top} \boldsymbol{A} \boldsymbol{v}]$, the sliced Fisher divergence becomes

$$
D_{\mathrm{SF}}(p_{\mathcal{D}} \| p_\theta) = \mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x})} \mathbb{E}_{p(\boldsymbol{v})}\!\left[\frac{1}{2}\bigl(\boldsymbol{v}^{\top} \boldsymbol{s}_\theta(\boldsymbol{x})\bigr)^2 + \boldsymbol{v}^{\top} \bigl(\nabla_{\boldsymbol{x}}\boldsymbol{s}_\theta(\boldsymbol{x})\bigr)\boldsymbol{v}\right] + C.
$$

Because $\boldsymbol{v}^{\top} (\nabla_{\boldsymbol{x}}\boldsymbol{s}_\theta(\boldsymbol{x})) \boldsymbol{v} = \boldsymbol{v}^{\top} \nabla_{\boldsymbol{x}}(\boldsymbol{v}^{\top} \boldsymbol{s}_\theta(\boldsymbol{x}))$, it is evaluated in $O(D)$ time via a single Jacobian-vector or vector-Jacobian product (JVP/VJP) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 3.3.1, p. 68 (PDF p. 73)).

**2. Denoising Score Matching (DSM) & Tweedie's Formula..** 
Instead of matching scores on raw data (where $p_{\mathcal{D}}$ may lie on a low-dimensional manifold and lack a smooth density), Vincent (2011) perturbs data $\boldsymbol{x} \sim p_{\mathcal{D}}$ with a known kernel $q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x})$, forming the smoothed marginal $q_\sigma(\tilde{\boldsymbol{x}}) = \int q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}) p_{\mathcal{D}}(\boldsymbol{x}) \mathrm{d}\boldsymbol{x}$ [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 3.3, pp. 36–38).

> [!important] Theorem: Vincent's DSM Equivalence & Tweedie's Formula
> For any fixed noise kernel $q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x})$, the explicit score matching objective on the smoothed marginal $q_\sigma(\tilde{\boldsymbol{x}})$ is equivalent, up to a $\theta$-independent constant $C$, to regressing $\boldsymbol{s}_\theta(\tilde{\boldsymbol{x}})$ onto the conditional score $\nabla_{\tilde{\boldsymbol{x}}} \log q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x})$ [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Eq. 87, pp. 37–38):
>
> $$
> \underbrace{\frac{1}{2}\mathbb{E}_{q_\sigma(\tilde{\boldsymbol{x}})}\!\left[\lVert \boldsymbol{s}_\theta(\tilde{\boldsymbol{x}}) - \nabla_{\tilde{\boldsymbol{x}}}\log q_\sigma(\tilde{\boldsymbol{x}}) \rVert_2^2\right]}_{J_{\mathrm{ESM}}(\theta;\sigma)}
>   = \underbrace{\frac{1}{2}\mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x}) q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x})}\!\left[\lVert \boldsymbol{s}_\theta(\tilde{\boldsymbol{x}}) - \nabla_{\tilde{\boldsymbol{x}}}\log q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}) \rVert_2^2\right]}_{J_{\mathrm{DSM}}(\theta;\sigma)} + C.
> $$
>
> *Proof.* Expanding $J_{\mathrm{ESM}}(\theta;\sigma)$ leaves $\frac{1}{2}\mathbb{E}_{q_\sigma}[\lVert \boldsymbol{s}_\theta(\tilde{\boldsymbol{x}}) \rVert_2^2] - \mathbb{E}_{q_\sigma}[\boldsymbol{s}_\theta(\tilde{\boldsymbol{x}})^{\top} \nabla_{\tilde{\boldsymbol{x}}}\log q_\sigma(\tilde{\boldsymbol{x}})] + C_1$. Rewriting the cross-term via $q_\sigma(\tilde{\boldsymbol{x}}) = \int p_{\mathcal{D}}(\boldsymbol{x}) q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}) \mathrm{d}\boldsymbol{x}$:
>
> $$
> \begin{aligned}
> \mathbb{E}_{q_\sigma(\tilde{\boldsymbol{x}})}\!\left[\boldsymbol{s}_\theta(\tilde{\boldsymbol{x}})^{\top} \nabla_{\tilde{\boldsymbol{x}}}\log q_\sigma(\tilde{\boldsymbol{x}})\right]
>   &= \int_{\mathbb{R}^D} \boldsymbol{s}_\theta(\tilde{\boldsymbol{x}})^{\top} \nabla_{\tilde{\boldsymbol{x}}} \left(\int_{\mathbb{R}^D} p_{\mathcal{D}}(\boldsymbol{x}) q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}) \mathrm{d}\boldsymbol{x}\right) \mathrm{d}\tilde{\boldsymbol{x}} \\
>   &= \iint \boldsymbol{s}_\theta(\tilde{\boldsymbol{x}})^{\top} p_{\mathcal{D}}(\boldsymbol{x}) q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}) \nabla_{\tilde{\boldsymbol{x}}}\log q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}) \mathrm{d}\boldsymbol{x} \mathrm{d}\tilde{\boldsymbol{x}} \\
>   &= \mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x}) q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x})}\!\left[\boldsymbol{s}_\theta(\tilde{\boldsymbol{x}})^{\top} \nabla_{\tilde{\boldsymbol{x}}}\log q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x})\right],
> \end{aligned}
> $$
>
> which matches the cross-term of $J_{\mathrm{DSM}}(\theta;\sigma)$, leaving only $\theta$-independent constants $C = C_1 - C_2$.  $\square$
>
> **Gaussian Specialization & Tweedie's Formula:** For $q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}) = \mathcal{N}(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}, \sigma^2 \boldsymbol{I})$ with $\tilde{\boldsymbol{x}} = \boldsymbol{x} + \sigma \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$, the conditional score is $\nabla_{\tilde{\boldsymbol{x}}}\log q_\sigma(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}) = -\frac{\tilde{\boldsymbol{x}} - \boldsymbol{x}}{\sigma^2} = -\frac{\boldsymbol{\epsilon}}{\sigma}$, giving
>
> $$
> J_{\mathrm{DSM}}(\theta;\sigma) = \frac{1}{2}\mathbb{E}_{\boldsymbol{x} \sim p_{\mathcal{D}},  \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})}\!\left[\lVert \boldsymbol{s}_\theta(\boldsymbol{x} + \sigma\boldsymbol{\epsilon}) + \frac{\boldsymbol{\epsilon}}{\sigma} \rVert_2^2\right].
> $$
>
> Furthermore, by *Tweedie's formula* [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Lemma 3.3.2, p. 73 (PDF p. 78)), the posterior mean denoiser satisfies
>
> $$
> \mathbb{E}[\boldsymbol{x} \mid \tilde{\boldsymbol{x}}] = \tilde{\boldsymbol{x}} + \sigma^2 \nabla_{\tilde{\boldsymbol{x}}} \log q_\sigma(\tilde{\boldsymbol{x}}),
> $$
>
> establishing that the optimal score $\boldsymbol{s}_{\theta^*}(\tilde{\boldsymbol{x}})$ points directly from the noisy sample toward the minimum-MSE denoised reconstruction.

**Unifying CD-1 and Score Matching via de Bruijn's Identity..** 
Although Contrastive Divergence (KL-based) and Score Matching (Fisher-divergence-based) appear distinct, Hyv\"arinen (2007) proved—as summarized in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 24.45, p. 860 (PDF p. 894))—that 1-step Langevin MCMC from $\boldsymbol{x} \sim p_{\mathcal{D}}$ with step size $\epsilon$ yields a $\mathrm{CD}_1$ gradient whose leading-order expansion is $\frac{\epsilon^2}{2}\nabla_\theta D_F(p_{\mathcal{D}} \| p_\theta) + o(\epsilon^2)$. This reflects *de Bruijn's identity*, linking KL and Fisher divergences under Gaussian smoothing $\tilde{\boldsymbol{x}}_t \sim \mathcal{N}(\boldsymbol{x}, t\boldsymbol{I})$:

$$
\frac{\mathrm{d}}{\mathrm{d}t} D_{\mathrm{KL}}(q_t \parallel p_{\theta,t}) = -\frac{1}{2} D_F(q_t \| p_{\theta,t}).
$$

## Algorithmic Implementation & Comparative Synthesis

We present the two canonical training algorithms: Persistent Contrastive Divergence (PCD) with Langevin MCMC (Algorithm `alg:ebm:pcd`) and Denoising Score Matching (DSM) with Langevin sampling (Algorithm `alg:ebm:dsm`).

> [!example] Algorithm: Persistent Contrastive Divergence (PCD / SML) with Langevin MCMC
> **Input:** Dataset $\mathcal{D}$, initial $\theta$, replay buffer $\mathcal{B} = \{\tilde{\boldsymbol{x}}^{(j)}\}_{j=1}^M$, learning rate $\alpha$, Langevin step $\epsilon$, MCMC steps $K$.
>   **Output:** Trained energy parameters $\theta$.
>   Initialize buffer particles $\tilde{\boldsymbol{x}}^{(j)} \sim p_0$ (e.g., Uniform or $\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$) for $j=1,\dots,M$
>   **while** not converged:
>
>     Sample minibatch $\{\boldsymbol{x}^{(i)}\}_{i=1}^B \sim \mathcal{D}$ and persistent states $\{\tilde{\boldsymbol{x}}^{(i)}_0\}_{i=1}^B \sim \mathcal{B}$
>     **for** $k = 0$ to $K-1$:
>
>       Sample $\boldsymbol{z}_k^{(i)} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ for $i=1,\dots,B$
>       $\tilde{\boldsymbol{x}}_{k+1}^{(i)} \leftarrow \tilde{\boldsymbol{x}}_k^{(i)} - \frac{\epsilon^2}{2} \nabla_{\boldsymbol{x}} E_\theta(\tilde{\boldsymbol{x}}_k^{(i)}) + \epsilon \boldsymbol{z}_k^{(i)}$  *// Langevin negative-phase update*
>
>     Replace sampled buffer entries in $\mathcal{B}$ with $\{\tilde{\boldsymbol{x}}_K^{(i)}\}_{i=1}^B$
>     $\boldsymbol{g} \leftarrow \frac{1}{B}\sum_{i=1}^B \nabla_\theta E_\theta(\boldsymbol{x}^{(i)}) - \frac{1}{B}\sum_{i=1}^B \nabla_\theta E_\theta(\tilde{\boldsymbol{x}}_K^{(i)})$  *// Negative log-likelihood grad*
>     $\theta \leftarrow \theta - \alpha \boldsymbol{g}$

> [!example] Algorithm: Denoising Score Matching (DSM) Training & Langevin Sampling
> **Input:** Dataset $\mathcal{D}$, noise scale $\sigma > 0$, score network $\boldsymbol{s}_\theta$, learning rate $\alpha$, sampling step $\eta$, steps $T$.
>   **Output:** Trained score model $\boldsymbol{s}_\theta$ and generated sample $\boldsymbol{x}_T$.
>   **while** training not converged:
>
>     Sample minibatch $\{\boldsymbol{x}^{(i)}\}_{i=1}^B \sim \mathcal{D}$ and $\boldsymbol{\epsilon}^{(i)} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$
>     Construct perturbed inputs $\tilde{\boldsymbol{x}}^{(i)} \leftarrow \boldsymbol{x}^{(i)} + \sigma \boldsymbol{\epsilon}^{(i)}$
>     $\mathcal{L}_{\mathrm{DSM}}(\theta) \leftarrow \frac{1}{2B}\sum_{i=1}^B \lVert \boldsymbol{s}_\theta(\tilde{\boldsymbol{x}}^{(i)}) + \boldsymbol{\epsilon}^{(i)}/\sigma \rVert_2^2$
>     $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}_{\mathrm{DSM}}(\theta)$
>
>   Initialize $\boldsymbol{x}_0 \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$
>   **for** $t = 0$ to $T-1$:
>
>     Sample $\boldsymbol{z}_t \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$
>     $\boldsymbol{x}_{t+1} \leftarrow \boldsymbol{x}_t + \eta \boldsymbol{s}_\theta(\boldsymbol{x}_t) + \sqrt{2\eta} \boldsymbol{z}_t$  *// Euler–Maruyama Langevin dynamics*

**Table:** Comparative synthesis of EBM and score-based training objectives [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 24, pp. 851–863) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 18, pp. 606–626).

| **Method** | **Partition $Z_\theta$** | **Per-Step Cost** | **Consistency** | **Primary Bottleneck** |
| — | — | — | — | — |
| Exact MCMC MLE | Estimates $\nabla_\theta \log Z_\theta$ | $O(K_{\mathrm{burn}})$ MCMC | Consistent | Prohibitive burn-in per step |
| $\mathrm{CD}_k$ | Biased $\nabla_\theta \log Z_\theta$ | $O(k)$ MCMC ($k \approx 1$) | Biased | Spurious modes far from data |
| PCD / SML | Persistent $\nabla_\theta \log Z_\theta$ | $O(K)$ persistent | Asymptotic | Chain mixing vs. $\theta$ drift |
| Hyv\"arinen ISM | Eliminated | $O(D^2)$ Hessian trace | Consistent | Quadratic scaling in $D$ |
| Sliced SM (SSM) | Eliminated | $O(D)$ JVP/VJP | Consistent | Probe variance; manifold fragility |
| Vincent DSM | Eliminated | $O(1)$ forward/back | Matches $q_\sigma$ | Bias vs. $\sigma\!\to\!0$ variance |
| NCE | Learns $c \approx -\log Z_\theta$ | $O(1)$ binary loss | Consistent | Designing high-$D$ noise $p_n$ |

> [!info] Cross-Book Synthesis: Noise Contrastive Estimation (NCE) & Partition Estimation (AIS)
> Both [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 18.6–18.7, pp. 620–626 (PDF pp. 635–641)) and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 24.4, pp. 860–862 (PDF pp. 894–896)) highlight two complementary approaches when $Z_\theta$ itself is required:
> - **Noise Contrastive Estimation (NCE):** Treats $c = -\log Z_\theta$ as a learnable scalar parameter in $\log p_\theta(\boldsymbol{x}) = -E_\theta(\boldsymbol{x}) + c$ and trains a binary logistic classifier distinguishing $\boldsymbol{x} \sim p_{\mathcal{D}}$ ($y=1$) from a tractable noise density $p_n(\boldsymbol{x})$ ($y=0$) via posterior $p_{n,\theta}(y=1\mid\boldsymbol{x}) = \sigma(\log p_\theta(\boldsymbol{x}) - \log p_n(\boldsymbol{x}))$. At the optimum, $p_{\theta^*} \equiv p_{\mathcal{D}}$ and $c^* = -\log Z_{\theta^*}$ (yielding a self-normalized EBM). Moreover, setting $p_n(\boldsymbol{x}) = p_{\mathcal{D}}(\boldsymbol{x}-\boldsymbol{v})$ and Taylor-expanding as $\lVert \boldsymbol{v} \rVert_2 \to 0$ recovers Sliced Score Matching [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 24.58, p. 862 (PDF p. 896)).
> - **Annealed Importance Sampling (AIS):** For evaluating test log-likelihoods after training, simple importance sampling from a tractable proposal $p_0$ suffers catastrophic weight variance when $D_{\mathrm{KL}}(p_0 \parallel p_1)$ is large. AIS bridges $p_0$ and $p_1 = p_\theta$ through intermediate annealed distributions $p_{\eta_0}, \dots, p_{\eta_n}$ ($0 = \eta_0 < \dots < \eta_n = 1$) with MCMC transitions, estimating the ratio $Z_1 / Z_0$ reliably [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 18.7.1, pp. 625–626 (PDF pp. 640–641)).

## Visual Diagnostics & Geometric Structure

Figure `fig:ebm:score_vector_field` illustrates the exact 1D correspondence between an asymmetric double-well energy landscape $E_\theta(x)$ and its Stein score $s_\theta(x) = -\partial_x E_\theta(x)$. Zero-crossings of $s_\theta(x)$ with negative slope ($\partial_x s_\theta(x) < 0$) correspond to stable probability modes (energy valleys), whereas the zero-crossing with positive slope ($\partial_x s_\theta(x) > 0$) marks the unstable energy barrier (saddle). In 2D (Figure `fig:ebm:score_field_2d`), the score vector field $\nabla_{\boldsymbol{x}}\log p(\boldsymbol{x})$ points orthogonally across density contours directly toward high-density modes.

> [!example] Visual Diagram (LaTeX / PDF)
> *Top: Asymmetric bimodal energy landscape $E_\theta(x)$ with stable minima $x_1^\star, x_2^\star$ and an energy barrier $x_s$. Bottom: Corresponding Stein score $s_\theta(x) = -\partial_x E_\theta(x)$. Zero-crossings with negative slope act as attractive sinks for Langevin dynamics, while $x_s$ acts as an unstable repeller.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!note] Source Figure
> *Figure:* 2D bimodal data distribution and its Stein score vector field $\nabla_{\boldsymbol{x}}\log p(\boldsymbol{x})$, illustrating how score vectors point toward high-density modes across the ambient space [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 3.1.2, Fig. 3.2, p. 59 (PDF p. 64)). (`theprinciple_diffusion_models_p64_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Spurious Modes in Short-Run Contrastive Divergence ($\mathrm{CD}_k$)
> When $k$ is small (e.g., $\mathrm{CD}_1$) and chains are initialized at data points, MCMC samples rarely travel far enough to explore modes of $p_\theta(\boldsymbol{x})$ separated by high energy barriers [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 18.2, Fig. 18.2, p. 611 (PDF p. 626)). Consequently, the negative phase fails to push down spurious modes far from the training data (Figure `fig:ebm:cd_spurious`). **Remedy:** Use Persistent CD (PCD) with a replay buffer, or regularize energy magnitudes (e.g., $L_2$ penalty on $E_\theta(\boldsymbol{x})$).

> [!note] Source Figure
> *Figure:* Failure of short-run Contrastive Divergence ($\mathrm{CD}_k$) to suppress spurious modes of $p_{\mathrm{model}}(x)$ located far from $p_{\mathrm{data}}(x)$, because $k$-step chains initialized at data cannot cross low-density valleys [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 18, Fig. 18.2, p. 611 (PDF p. 626)). (`deep_learning_goodfellow_bengio_courville_2017_p626_fig1.png`)

> [!warning] Manifold Hypothesis & Ill-Defined Scores in Low-Density Regions
> Natural data (e.g., images) often concentrate on a low-dimensional manifold embedded in $\mathbb{R}^D$. Away from the manifold, $p_{\mathcal{D}}(\boldsymbol{x}) \approx 0$, so $\nabla_{\boldsymbol{x}}\log p_{\mathcal{D}}(\boldsymbol{x})$ is undefined or uninformative, and score matching receives no training signal in low-density regions—causing Langevin chains initialized from noise to stall [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 3.3.3, p. 72 (PDF p. 77)). **Remedy:** Perturb data with multi-scale Gaussian noise $\{\sigma_i\}_{i=1}^L$ as in Noise Conditional Score Networks (NCSN) and sample via annealed Langevin dynamics.

> [!warning] Variance Explosion of DSM as $\sigma \to 0$
> While DSM is biased toward the smoothed density $q_\sigma(\tilde{\boldsymbol{x}})$ rather than $p_{\mathcal{D}}(\boldsymbol{x})$, shrinking $\sigma \to 0$ to reduce bias triggers severe Monte Carlo variance explosion [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 24.3.2.1, Eq. 24.36, p. 858 (PDF p. 892)). Expanding `(eq:ebm:dsm_gauss)` around $\sigma \approx 0$ yields cross-terms scaling as $\boldsymbol{\epsilon}^{\top} \boldsymbol{s}_\theta(\boldsymbol{x}) / \sigma$ and $\lVert \boldsymbol{\epsilon} \rVert_2^2 / \sigma^2$, whose variance grows as $O(1/\sigma^2)$. **Remedy:** Weight multi-scale DSM losses by $\lambda(\sigma) = \sigma^2$ (as in NCSN/DDPM) to normalize target variance across noise scales.

> [!warning] Non-Conservative Score Fields & Langevin Discretization Bias
> Directly parameterizing a vector-valued network $\boldsymbol{s}_\theta : \mathbb{R}^D \to \mathbb{R}^D$ rather than $-\nabla_{\boldsymbol{x}} E_\theta(\boldsymbol{x})$ does not guarantee a conservative vector field (i.e., $\nabla_{\boldsymbol{x}} \boldsymbol{s}_\theta$ need not be symmetric), so $\boldsymbol{s}_\theta$ may not correspond to the gradient of any scalar log-density [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 24.3.5, p. 860 (PDF p. 894)). Moreover, finite step sizes $\epsilon > 0$ in `(eq:ebm:langevin)` introduce discretization bias unless corrected via a Metropolis–Hastings accept/reject step (MALA) or annealed step-size decay.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[Energy-Based Models (EBM)]], [[Score Matching (DSM & SSM)]]
- **Prerequisites (Directed $\leftarrow$):** [[Automatic Differentiation (AD)]], [[Energy-Based Models (EBM)]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[DDPM]], [[Score Matching (DSM & SSM)]], [[Score-Based SDE]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Automatic Differentiation (AD)]], [[DDPM]], [[Energy-Based Models (EBM)]], [[Graphical Models (BN & MRF)]], [[MCMC & HMC Sampling]], [[Score Matching (DSM & SSM)]], [[Score-Based SDE]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]]
