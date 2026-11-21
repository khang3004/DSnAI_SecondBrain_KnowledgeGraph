---
type: monograph
domain: "D02"
parent_topic: "[[PT 02 - Latent Variables, Graphical Models, VI & MCMC Sampling]]"
sub_topic: "[[ST 02 - Graphical Models, Gaussian Processes, Variational Inference & MCMC]]"
cheatsheet: "[[Cheatsheet - Variational Inference, Evidence Lower Bound (ELBO) & Expectation Propagation]]"
part_of: 
  - "[[ST 02 - Graphical Models, Gaussian Processes, Variational Inference & MCMC]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
concepts: 
  - "[[Reparameterization Trick]]"
  - "[[Variational Inference & ELBO]]"
prerequisites: 
  - "[[Automatic Differentiation (AD)]]"
  - "[[KL Divergence]]"
  - "[[Variational Inference & ELBO]]"
enables: 
  - "[[DDPM]]"
  - "[[EM Algorithm]]"
  - "[[Reparameterization Trick]]"
  - "[[Variational Autoencoder (VAE)]]"
related: 
  - "[[EM Algorithm]]"
  - "[[Exponential Families]]"
  - "[[Gaussian Mixture Models (GMM)]]"
  - "[[KL Divergence]]"
  - "[[PPO (Proximal Policy Optimization)]]"
  - "[[Reparameterization Trick]]"
  - "[[Variational Autoencoder (VAE)]]"
  - "[[Variational Inference & ELBO]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-02
---

# Variational Inference, Evidence Lower Bound (ELBO) & Expectation Propagation

> [!abstract] Monograph Navigation
> - **Domain:** [[D02 - Classical, Kernel & Probabilistic Machine Learning|Classical, Kernel & Probabilistic Machine Learning]]
> - **Parent / Sub-Topic:** [[PT 02 - Latent Variables, Graphical Models, VI & MCMC Sampling|Latent Variables, Graphical Models, VI & MCMC Sampling]] → [[ST 02 - Graphical Models, Gaussian Processes, Variational Inference & MCMC|Graphical Models, Gaussian Processes, Variational Inference & MCMC]]
> - **Visual Cheatsheet:** [[Cheatsheet - Variational Inference, Evidence Lower Bound (ELBO) & Expectation Propagation]]
> - **Description:** Derives log p(x) = ELBO(q) + KL(q || p), mean-field coordinate ascent (CAVI), black-box score-function (REINFORCE) vs. pathwise reparameterized gradients, tighter IWAE bounds, and Expectation Propagation.

## Problem Formulation & Unified Notation

Consider a generative probabilistic model with observed data $\boldsymbol{x} \in \mathcal{X}$, latent variables $\boldsymbol{z} \in \mathcal{Z}$, and model parameters $\boldsymbol{\theta} \in \Theta$. The joint density $p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z}) = p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z}) p_{\boldsymbol{\theta}}(\boldsymbol{z})$ induces the exact posterior $p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x}) = p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z}) / p_{\boldsymbol{\theta}}(\boldsymbol{x})$, where the marginal likelihood (or model evidence)

$$
p_{\boldsymbol{\theta}}(\boldsymbol{x}) = \int_{\mathcal{Z}} p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z}) \mathrm{d}\boldsymbol{z}
$$

is typically intractable due to high-dimensional integration or non-conjugate couplings across $\boldsymbol{z}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 10, PDF p. 473). Variational inference (VI) recasts posterior approximation as a deterministic optimization problem over a tractable variational family $\mathcal{Q} = \{ q_{\boldsymbol{\psi}}(\boldsymbol{z}) : \boldsymbol{\psi} \in \Psi \}$ by minimizing the reverse Kullback–Leibler (KL) divergence:

$$
\boldsymbol{\psi}^* = \arg\min_{\boldsymbol{\psi} \in \Psi} D_{\mathrm{KL}}(q_{\boldsymbol{\psi}}(\boldsymbol{z}) \parallel p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})).
$$

By contrast, Expectation Propagation (EP) approximates a factorized target posterior $p(\boldsymbol{z} \mid \boldsymbol{x}) \propto f_0(\boldsymbol{z}) \prod_{i=1}^K f_i(\boldsymbol{z})$ within an exponential family $\mathcal{Q}$ through iterative local forward-KL projections $D_{\mathrm{KL}}(q_i^{\mathrm{tilted}} \parallel \tilde{q})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.7, PDF pp. 513–514).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> The four source texts adopt distinct symbol conventions for the variational distribution, parameters, and ELBO objective. Table `tab:vi:notation` establishes the unified notation used throughout this monograph.
>
> | **Symbol** | **Meaning** | **Murphy** | **Bishop** | **Prince / Goodfellow** |
> | --- | --- | --- | --- | --- |
> | $\boldsymbol{x}, \boldsymbol{z}$ | Observed & latent vars | $\boldsymbol{x}, \boldsymbol{z}$ | $\boldsymbol{X}, \boldsymbol{Z}$ | $\boldsymbol{x}, \boldsymbol{z}$ / $\boldsymbol{v}, \boldsymbol{h}$ |
> | $\boldsymbol{\theta}$ | Generative parameters | $\boldsymbol{\theta}$ | $\boldsymbol{\theta}$ ($\boldsymbol{w}$) | $\boldsymbol{\phi}$ / $\boldsymbol{\theta}$ |
> | $q_{\boldsymbol{\psi}},  q_{\boldsymbol{\phi}}$ | Local / amortized family | $q_{\boldsymbol{\psi}}(\boldsymbol{z}), q_{\boldsymbol{\phi}}$ | $q(\boldsymbol{Z})$ | $q(\boldsymbol{z}\mid\boldsymbol{x},\boldsymbol{\theta})$ / $q(\boldsymbol{h}\mid\boldsymbol{v})$ |
> | $\mathrm{ELBO}$ | Evidence Lower Bound | $\text{\L}(\boldsymbol{\theta},\boldsymbol{\psi}\mid\boldsymbol{x})$ | $\mathcal{L}(q, \boldsymbol{\theta})$ | $\mathrm{ELBO}$ / $\mathcal{L}(\boldsymbol{v},\boldsymbol{\theta},q)$ |
> | $\mathcal{F}$ | Variational Free Energy | $\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\psi}\mid\boldsymbol{x})$ | — | — |
> | $D_{\mathrm{KL}}(q \parallel p)$ | Reverse KL divergence | $D_{\mathrm{KL}}(q \parallel p)$ | $\mathrm{KL}(q \parallel p)$ | $D_{\mathrm{KL}}[q \parallel \Pr]$ |

## Core Mathematical Derivations & Theorems

### The Fundamental ELBO Decomposition & Equivalent Forms

> [!important] Theorem: ELBO Identity, Free Energy, and Reconstruction–KL Forms
> For any distribution $q(\boldsymbol{z})$ whose support satisfies $\mathrm{supp}(q) \subseteq \mathrm{supp}(p_{\boldsymbol{\theta}}(\cdot \mid \boldsymbol{x}))$, the log marginal likelihood decomposes exactly as [[Source - Foundations|deep_learning_foundations_concepts]] (Eq. 15.52, PDF p. 500) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Eq. 17.17, PDF p. 347):
>
> $$
> \log p_{\boldsymbol{\theta}}(\boldsymbol{x}) = \mathrm{ELBO}(q; \boldsymbol{\theta}) + D_{\mathrm{KL}}(q(\boldsymbol{z}) \parallel p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})),
> $$
>
> where the Evidence Lower Bound admits three algebraically equivalent representations:
>
> $$
> \begin{aligned}
> \mathrm{ELBO}(q; \boldsymbol{\theta}) &= \mathbb{E}_{q(\boldsymbol{z})}\!\left[\log \frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})}{q(\boldsymbol{z})}\right]  \\
>   &= \mathbb{E}_{q(\boldsymbol{z})}\!\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})\right] + \mathrm{H}(q) = -\mathcal{F}(q; \boldsymbol{\theta}) \quad\text{(Neg. Free Energy)}  \\
>   &= \mathbb{E}_{q(\boldsymbol{z})}\!\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})\right] - D_{\mathrm{KL}}(q(\boldsymbol{z}) \parallel p_{\boldsymbol{\theta}}(\boldsymbol{z})) \quad\text{(Recon. $-$ Prior KL)}.
> \end{aligned}
> $$
>
> Since $D_{\mathrm{KL}}(q \parallel p) \ge 0$ with equality if and only if $q(\boldsymbol{z}) = p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})$ a.e., we have $\mathrm{ELBO}(q; \boldsymbol{\theta}) \le \log p_{\boldsymbol{\theta}}(\boldsymbol{x})$, and maximizing $\mathrm{ELBO}(q; \boldsymbol{\theta})$ with respect to $q$ is strictly equivalent to minimizing $D_{\mathrm{KL}}(q(\boldsymbol{z}) \parallel p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x}))$.

**Proof.** 
Using the product rule $\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z}) = \log p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x}) + \log p_{\boldsymbol{\theta}}(\boldsymbol{x})$ inside `(eq:vi:elbo-ratio)` yields:

$$
\begin{aligned}
\mathrm{ELBO}(q; \boldsymbol{\theta}) &= \int q(\boldsymbol{z}) \log \frac{p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x}) p_{\boldsymbol{\theta}}(\boldsymbol{x})}{q(\boldsymbol{z})} \mathrm{d}\boldsymbol{z} \\
  &= \left(\int q(\boldsymbol{z}) \mathrm{d}\boldsymbol{z}\right) \log p_{\boldsymbol{\theta}}(\boldsymbol{x}) \\
  &\quad - \int q(\boldsymbol{z}) \log \frac{q(\boldsymbol{z})}{p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})} \mathrm{d}\boldsymbol{z} \\
  &= \log p_{\boldsymbol{\theta}}(\boldsymbol{x}) - D_{\mathrm{KL}}(q(\boldsymbol{z}) \parallel p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})).
\end{aligned}
$$

Splitting $\log \frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})}{q(\boldsymbol{z})} = \log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z}) - \log q(\boldsymbol{z})$ gives `(eq:vi:elbo-energy)` via $\mathrm{H}(q) = -\mathbb{E}_q[\log q(\boldsymbol{z})]$. Factoring $p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z}) = p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z}) p_{\boldsymbol{\theta}}(\boldsymbol{z})$ gives `(eq:vi:elbo-recon)` [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.1.1, PDF pp. 473–475).
 $\blacksquare$

> [!note] Source Figure
> *Figure:* Geometric decomposition of the log marginal likelihood $\ln p(\mathbf{X}|\boldsymbol{\theta}) = \mathcal{L}(q,\boldsymbol{\theta}) + \mathrm{KL}(q\|p)$ from [[Source - Foundations|deep_learning_foundations_concepts]] (Fig. 15.13, PDF p. 501). Because $\mathrm{KL}(q\|p) \ge 0$, maximizing the ELBO $\mathcal{L}(q,\boldsymbol{\theta})$ closes the gap to the evidence $\ln p(\mathbf{X}|\boldsymbol{\theta})$. (`deep_learning_foundations_concepts_p501_fig1.png`)

> [!tip] Statistical Physics & Rate–Distortion Interpretation
> In statistical physics [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.1.1.1, PDF p. 474), $E_{\boldsymbol{\theta}}(\boldsymbol{z}) = -\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})$ is the energy and $\mathcal{F}(q; \boldsymbol{\theta}) = \mathbb{E}_q[E_{\boldsymbol{\theta}}(\boldsymbol{z})] - \mathrm{H}(q)$ is the Variational Free Energy (VFE), which upper-bounds the Helmholtz free energy $-\log p_{\boldsymbol{\theta}}(\boldsymbol{x})$. In `(eq:vi:elbo-recon)`, the first term measures expected log-likelihood (reconstruction fidelity), while $-D_{\mathrm{KL}}(q(\boldsymbol{z}) \parallel p_{\boldsymbol{\theta}}(\boldsymbol{z}))$ acts as an information-theoretic regularizer penalizing posterior divergence from the prior [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 17.4.2, PDF p. 348). For a diagonal Gaussian $q(\boldsymbol{z}) = \mathcal{N}(\boldsymbol{\mu}, \mathrm{diag}(\boldsymbol{\sigma}^2))$ and standard normal prior $p(\boldsymbol{z}) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_D)$, the prior KL evaluates in closed form to $\frac{1}{2}\sum_{d=1}^D (\sigma_d^2 + \mu_d^2 - 1 - \log \sigma_d^2)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 10.50, PDF p. 482) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Eq. 17.24, PDF p. 350).

Drawing $S > 1$ importance samples $\boldsymbol{z}^{(1)}, \dots, \boldsymbol{z}^{(S)} \overset{\mathrm{i.i.d.}}{\sim} q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})$ and applying Jensen's inequality [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.5.1, Eq. 10.167, PDF p. 507) yields the Importance Weighted Autoencoder (IWAE) multi-sample bound:

$$
\mathrm{ELBO}(q_{\boldsymbol{\phi}}) = \mathcal{L}_1 \le \mathcal{L}_S \triangleq \mathbb{E}_{\boldsymbol{z}^{(1:S)} \sim q_{\boldsymbol{\phi}}}\!\left[\log \frac{1}{S}\sum_{s=1}^S \frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z}^{(s)})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}^{(s)} \mid \boldsymbol{x})}\right] \le \log p_{\boldsymbol{\theta}}(\boldsymbol{x}),
$$

which converges monotonically to $\log p_{\boldsymbol{\theta}}(\boldsymbol{x})$ as $S \to \infty$, though at the cost of a vanishing signal-to-noise ratio in the encoder gradient $\nabla_{\boldsymbol{\phi}} \mathcal{L}_S$ unless doubly reparameterized estimators are used [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.5.1.1, PDF p. 508).

### Mean-Field Coordinate Ascent VI (CAVI) & Natural Gradients

To make the expectation in `(eq:vi:elbo-energy)` tractable without fixing a parametric family a priori (free-form VI), the mean-field approximation restricts $q$ to factorize completely across $J$ latent blocks:

$$
q(\boldsymbol{z}) = \prod_{j=1}^J q_j(\boldsymbol{z}_j).
$$

> [!important] Derivation: Optimal Mean-Field Complete-Conditional Update
> Under `(eq:vi:mean-field)`, holding $\{q_k\}_{k \neq j}$ fixed, the ELBO as a functional of $q_j$ reduces to [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.3.1, PDF pp. 490–491) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 19.4.3, PDF p. 663):
>
> $$
> \mathrm{ELBO}_j(q_j) = \int q_j(\boldsymbol{z}_j) \mathbb{E}_{-j}\!\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})\right]\mathrm{d}\boldsymbol{z}_j - \int q_j(\boldsymbol{z}_j) \log q_j(\boldsymbol{z}_j) \mathrm{d}\boldsymbol{z}_j + \mathrm{const},
> $$
>
> where $\mathbb{E}_{-j}[\cdot]$ integrates over $\prod_{k \neq j} q_k(\boldsymbol{z}_k)$. Form the Lagrangian with normalization multiplier $\lambda$ and set the functional derivative $\frac{\delta}{\delta q_j(\boldsymbol{z}_j)}$ to zero [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 19.4.2, PDF pp. 661–662):
>
> $$
> \begin{aligned}
> \frac{\delta}{\delta q_j(\boldsymbol{z}_j)} &\left\{ \mathrm{ELBO}_j(q_j) + \lambda \left(\int q_j(\boldsymbol{z}_j) \mathrm{d}\boldsymbol{z}_j - 1\right) \right\}  \\
>   &\qquad = \mathbb{E}_{-j}\!\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})\right] - \log q_j(\boldsymbol{z}_j) - 1 + \lambda = 0.
> \end{aligned}
> $$
>
> Solving for $q_j(\boldsymbol{z}_j)$ and absorbing $1 - \lambda$ into the normalizing constant yields the closed-form CAVI fixed-point update:
>
> $$
> \log q_j^*(\boldsymbol{z}_j) = \mathbb{E}_{-j}\!\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})\right] + \mathrm{const} = \mathbb{E}_{\boldsymbol{z}_{\mathrm{mb}(j)}}\!\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{z}_j, \boldsymbol{z}_{\mathrm{mb}(j)}, \boldsymbol{x})\right] + \mathrm{const},
> $$
>
> where $\boldsymbol{z}_{\mathrm{mb}(j)}$ denotes the Markov blanket of $\boldsymbol{z}_j$. Equivalently, $\mathrm{ELBO}_j(q_j) = -D_{\mathrm{KL}}(q_j \parallel f_j) + \mathrm{const}$ with $f_j(\boldsymbol{z}_j) \propto \exp(\mathbb{E}_{-j}[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})])$, proving concavity in each $q_j$ individually [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 10.83, PDF p. 491).

In conditionally conjugate exponential-family models with global parameters $\boldsymbol{\beta}$ (natural variational parameter $\boldsymbol{\lambda}$) and local latents $\boldsymbol{z}_{1:N}$, the complete conditional takes the form
$p(\boldsymbol{\beta} \mid \boldsymbol{x}, \boldsymbol{z}) = h(\boldsymbol{\beta})\exp(\boldsymbol{\eta}(\boldsymbol{x}, \boldsymbol{z})^{\top} \boldsymbol{t}(\boldsymbol{\beta}) - A(\boldsymbol{\eta}))$.
Multiplying the Euclidean gradient $\nabla_{\boldsymbol{\lambda}}\mathrm{ELBO}$ by the inverse Fisher information $\boldsymbol{F}(\boldsymbol{\lambda})^{-1} = (\nabla^2_{\boldsymbol{\lambda}} A(\boldsymbol{\lambda}))^{-1}$ cancels the log-partition Hessian and yields the natural gradient [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.1.4, PDF p. 478):

$$
\tilde{\nabla}_{\boldsymbol{\lambda}} \mathrm{ELBO}(\boldsymbol{\lambda}) = \mathbb{E}_q[\boldsymbol{\eta}(\boldsymbol{x}, \boldsymbol{z})] - \boldsymbol{\lambda},
$$

so Stochastic Variational Inference (SVI) updates $\boldsymbol{\lambda}_{t+1} = (1 - \rho_t)\boldsymbol{\lambda}_t + \rho_t \hat{\boldsymbol{\eta}}_t$ using a minibatch estimate $\hat{\boldsymbol{\eta}}_t$ with Robbins–Monro step sizes $\sum_t \rho_t = \infty$, $\sum_t \rho_t^2 < \infty$.

### Black-Box VI (Score-Function) vs. Pathwise Reparameterization

For non-conjugate models, we optimize fixed-form variational parameters $\boldsymbol{\phi}$ via stochastic gradients of $\mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z})}[\tilde{\mathcal{L}}(\boldsymbol{\phi}, \boldsymbol{z})]$, where $\tilde{\mathcal{L}}(\boldsymbol{\phi}, \boldsymbol{z}) = \log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z}) - \log q_{\boldsymbol{\phi}}(\boldsymbol{z})$. Because the expectation depends on $\boldsymbol{\phi}$, $\nabla_{\boldsymbol{\phi}}$ cannot pass directly inside $\mathbb{E}_{q_{\boldsymbol{\phi}}}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 10.34, PDF p. 480).

> [!important] Score-Function (REINFORCE) vs. Pathwise Reparameterized Gradient
> **1. Score-Function (BBVI) Estimator:** Using $\nabla_{\boldsymbol{\phi}} q_{\boldsymbol{\phi}}(\boldsymbol{z}) = q_{\boldsymbol{\phi}}(\boldsymbol{z}) \nabla_{\boldsymbol{\phi}} \log q_{\boldsymbol{\phi}}(\boldsymbol{z})$ and $\mathbb{E}_{q_{\boldsymbol{\phi}}}[\nabla_{\boldsymbol{\phi}} \log q_{\boldsymbol{\phi}}(\boldsymbol{z})] = \boldsymbol{0}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.2.3, Eq. 10.66–10.74, PDF pp. 488–489):
>
> $$
> \nabla_{\boldsymbol{\phi}} \mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z})}\!\left[\left(\tilde{\mathcal{L}}(\boldsymbol{\phi}, \boldsymbol{z}) - c\right) \nabla_{\boldsymbol{\phi}} \log q_{\boldsymbol{\phi}}(\boldsymbol{z})\right],
> $$
>
> where $c$ is a control-variate baseline with optimal variance-minimizing value $c_i^* = \frac{\mathrm{Cov}(g_i \tilde{\mathcal{L}},  g_i)}{\mathrm{Var}[g_i]}$ for $g_i = \partial_{\phi_i} \log q_{\boldsymbol{\phi}}(\boldsymbol{z})$. Rao–Blackwellization further reduces variance by replacing $\tilde{\mathcal{L}}$ with only the Markov-blanket terms $\tilde{\mathcal{L}}_j(\boldsymbol{z}_j, \boldsymbol{z}_{\mathrm{mb}(j)})$ depending on $\boldsymbol{z}_j$.
>
> **2. Pathwise Reparameterization Estimator:** For continuous $\boldsymbol{z}$, express $\boldsymbol{z} = g_{\boldsymbol{\phi}}(\boldsymbol{\epsilon}, \boldsymbol{x})$ as a differentiable, invertible map of parameter-free noise $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$ (e.g., $\boldsymbol{z} = \boldsymbol{\mu} + \boldsymbol{L}\boldsymbol{\epsilon}$ with $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.2.1, Eq. 10.36–10.42, PDF pp. 481–482):
>
> $$
> \nabla_{\boldsymbol{\phi}} \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z})}[f(\boldsymbol{z})] = \mathbb{E}_{p(\boldsymbol{\epsilon})}\!\left[\nabla_{\boldsymbol{z}} f(\boldsymbol{z})^{\top} \nabla_{\boldsymbol{\phi}} g_{\boldsymbol{\phi}}(\boldsymbol{\epsilon}, \boldsymbol{x})\right], \qquad \log q_{\boldsymbol{\phi}}(\boldsymbol{z}) = \log p(\boldsymbol{\epsilon}) - \log \lvert \det \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{\epsilon}} \rvert.
> $$
>

### Amortized VI & Expectation Propagation (EP) vs. Laplace

Amortized VI replaces per-datapoint variational parameters $\boldsymbol{\psi}_n$ with a shared inference network $q_{\boldsymbol{\phi}}(\boldsymbol{z}_n \mid \boldsymbol{x}_n)$ parameterized by $\boldsymbol{\phi}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.1.5, PDF p. 478) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 17.5–17.6, PDF pp. 349–351). This amortizes inference cost to $O(1)$ forward passes at test time, at the expense of the *amortization gap* $\mathcal{G}_{\mathrm{amort}}(\boldsymbol{x}_n) = \mathrm{ELBO}(\boldsymbol{\psi}_n^*; \boldsymbol{x}_n) - \mathrm{ELBO}(q_{\boldsymbol{\phi}^*}(\cdot \mid \boldsymbol{x}_n); \boldsymbol{x}_n) \ge 0$, which semi-amortized VI closes via a few local gradient steps initialized from $\boldsymbol{\phi}(\boldsymbol{x}_n)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.1.6, PDF p. 479) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 19.5.2, PDF p. 667).

Expectation Propagation (EP) takes a fundamentally different route by approximating a factorized posterior $p(\boldsymbol{\theta} \mid \mathcal{D}) \propto f_0(\boldsymbol{\theta}) \prod_{i=1}^K f_i(\boldsymbol{\theta})$ with $q(\boldsymbol{\theta}) \propto f_0(\boldsymbol{\theta}) \prod_{i=1}^K \tilde{f}_i(\boldsymbol{\theta})$ in an exponential family $\mathcal{Q}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.7.1, Eq. 10.192–10.198, PDF pp. 513–514):

$$
\begin{aligned}
q_{\setminus i}(\boldsymbol{\theta}) &\propto \frac{q(\boldsymbol{\theta})}{\tilde{f}_i(\boldsymbol{\theta})} \quad\text{(Cavity)}, \qquad
  q_i^{\mathrm{tilted}}(\boldsymbol{\theta}) = \frac{1}{Z_i} f_i(\boldsymbol{\theta}) q_{\setminus i}(\boldsymbol{\theta}) \quad\text{(Tilted)},  \\
  q_i^{\mathrm{proj}} &= \arg\min_{\tilde{q} \in \mathcal{Q}} D_{\mathrm{KL}}(q_i^{\mathrm{tilted}} \parallel \tilde{q}) \quad\text{(Moment Match)},  \\
  \tilde{f}_i^{\mathrm{new}}(\boldsymbol{\theta}) &\propto \left(\frac{q_i^{\mathrm{proj}}(\boldsymbol{\theta})}{q_{\setminus i}(\boldsymbol{\theta})}\right)^{\delta} \tilde{f}_i(\boldsymbol{\theta})^{1-\delta},
\end{aligned}
$$

where $\delta \in (0, 1]$ is a damping factor. Whereas the Laplace approximation fits a single local Gaussian $\mathcal{N}(\hat{\boldsymbol{\theta}}, \boldsymbol{H}^{-1})$ using the Hessian $\boldsymbol{H} = -\nabla^2_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta}, \mathcal{D})|_{\hat{\boldsymbol{\theta}}}$ at the MAP mode $\hat{\boldsymbol{\theta}}$, EP matches global moments of each tilted distribution in the context of the cavity prior $q_{\setminus i}$.

## Algorithmic Implementation & Comparative Synthesis

> [!example] Visual Diagram (LaTeX / PDF)
> *TikZ / PGFPlots Diagram* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

**Table:** Complexity, variance, and applicability comparison across approximate inference paradigms [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 10, PDF pp. 473–515).

| **Method** | **Divergence** | **Latents** | **Memory** | **Variance & Convergence Profile** |
| --- | --- | --- | --- | --- |
| CAVI (Alg. `alg:vi:cavi`) | $D_{\mathrm{KL}}(q \parallel p)$ | Conjugate | $O(N J)$ | Zero gradient variance; monotonic local convergence |
| BBVI (REINFORCE) | $D_{\mathrm{KL}}(q \parallel p)$ | Discrete/Cont. | $O(B D)$ | High variance; needs CV baselines & Rao–Blackwell |
| Pathwise SVI (VAE) | $D_{\mathrm{KL}}(q \parallel p)$ | Continuous | $O(B D)$ | Low variance via $\nabla_{\boldsymbol{z}} f$; $O(1)$ test-time inference |
| Expectation Prop. | Local $D_{\mathrm{KL}}(p \parallel q)$ | Exp. family | $O(K D^2)$ | Mode-covering; damped updates; no convergence guarantee |
| Laplace Approx. | Local $2^{\text{nd}}$-order | Continuous $C^2$ | $O(D^2)$ | Zero-cost post-MAP; fails on skewed/multimodal targets |

> [!info] Cross-Book Synthesis: Divergent Perspectives on Variational Inference
> [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 15.4, 16.3) introduces the ELBO primarily as a vehicle for deriving exact and generalized EM in latent-variable models (GMMs, PPCA, Factor Analysis). [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 19.4–19.5) emphasizes the calculus-of-variations derivation of free-form mean-field fixed points and the biological/algorithmic wake-sleep motivation for learned inference networks. [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 17.4–17.6) prioritizes geometric intuition for amortized VAEs, highlighting the closed-form Gaussian prior KL and single-sample Monte Carlo ELBO. Finally, [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 10) unifies all four threads—CAVI, SVI natural gradients, REINFORCE vs. pathwise reparameterization, IWAE tighter bounds, and Expectation Propagation—within a single optimization taxonomy.

## Visual Diagnostics & Geometric Structure

> [!example] Visual Diagram (LaTeX / PDF)
> *Architectural taxonomy of approximate inference pathways: Reverse-KL variational inference (left, splitting into conjugate CAVI/SVI, score-function BBVI, and pathwise reparameterization) versus local Forward-KL Expectation Propagation (right, matching moments of tilted distributions).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Unimodal Gaussian approximations to a bimodal target $p(z) = \frac{1}{2}\mathcal{N}(-1.8, 0.55^2) + \frac{1}{2}\mathcal{N}(1.8, 0.55^2)$. Reverse-KL VI is zero-avoiding and collapses onto a single mode; Forward-KL / EP is zero-forcing and spans both modes via moment matching ($\mu=0, \sigma=\sqrt{1.8^2+0.55^2}\approx 1.88$); Laplace captures only local curvature at one MAP mode.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!note] Source Figure
> *Figure:* Mean-field (diagonal covariance, green dotted) versus low-rank-plus-diagonal (red dashed) and full-rank (blue dash-dot) Gaussian VI approximations to a correlated 2D Gaussian posterior (black solid) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 10.2, PDF p. 476). Mean-field VI underestimates marginal variances along correlated directions. (`probabilisticml_kevinmurphy_advancedtopics_p476_fig1.png`)

> [!note] Source Figure
> *Figure:* Expectation Propagation step combining a non-Gaussian logistic likelihood factor $f_i = p(y_i \mid \boldsymbol{\theta})$ (blue) with the Gaussian cavity prior $g_{-i}(\boldsymbol{\theta})$ (red) to form the tilted distribution $q_i^{\mathrm{tilted}}(\boldsymbol{\theta}) \propto f_i(\boldsymbol{\theta}) g_{-i}(\boldsymbol{\theta})$ (green), which is then projected back to $\mathcal{Q}$ by moment matching [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 10.15, PDF p. 514). (`probabilisticml_kevinmurphy_advancedtopics_p514_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Mean-Field Covariance Underestimation & Zero-Avoidance
> Minimizing $D_{\mathrm{KL}}(q \parallel p)$ penalizes $q(\boldsymbol{z}) > 0$ wherever $p(\boldsymbol{z}\mid\boldsymbol{x}) \approx 0$. Consequently, a factorized Gaussian $q(\boldsymbol{z}) = \prod_j \mathcal{N}(z_j \mid \mu_j, \sigma_j^2)$ matched to a correlated posterior $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ sets $\sigma_j^2 = (\boldsymbol{\Sigma}^{-1})_{jj}^{-1} \le \Sigma_{jj}$ (matching conditional rather than marginal precisions), severely underestimating uncertainty (Figure `fig:vi:meanfield-2d`). *Remedy:* Use structured factorizations, low-rank-plus-diagonal covariances $\boldsymbol{\Sigma} = \mathrm{diag}(\boldsymbol{d}) + \boldsymbol{U}\boldsymbol{U}^{\top}$, or normalizing flows [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.1.2, 10.4, PDF pp. 476, 505–506).

> [!warning] High Variance of Unbaselined Score-Function Gradients
> The naive REINFORCE estimator $\tilde{\mathcal{L}}(\boldsymbol{\phi}, \boldsymbol{z})\nabla_{\boldsymbol{\phi}} \log q_{\boldsymbol{\phi}}(\boldsymbol{z})$ exhibits variance scaling quadratically with $\lvert \tilde{\mathcal{L}} \rvert$ and linearly with latent dimension $D$, causing training divergence. *Remedy:* Always subtract optimal control variates $c_i^*$ `(eq:vi:reinforce)`, apply Rao–Blackwellization over Markov blankets, or use continuous relaxations (Gumbel-Softmax / Concrete) to enable pathwise gradients [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.2.3.2, PDF p. 489).

> [!warning] Amortized Inference Gap & Posterior Collapse
> When the encoder $q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})$ has limited capacity, the amortization gap degrades the ELBO; conversely, in VAEs with powerful autoregressive decoders $p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z})$, optimizing the ELBO can drive $q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x}) \to p(\boldsymbol{z})$ so that $D_{\mathrm{KL}}(q_{\boldsymbol{\phi}} \parallel p) \to 0$ and latent codes are ignored (posterior collapse). *Remedy:* Apply semi-amortized local refinement, KL annealing / $\beta$-VAE scheduling, or free-bits thresholding [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.1.6, PDF p. 479).

> [!warning] EP Negative Precision Matrices & Non-Guaranteed Convergence
> Subtracting natural parameters in the cavity step $q_{\setminus i} \propto q / \tilde{f}_i$ or site update $\tilde{f}_i^{\mathrm{new}} \propto q_i^{\mathrm{proj}} / q_{\setminus i}$ can produce non-positive-definite precision matrices for non-log-concave likelihoods, and vanilla EP has no global Lyapunov function guaranteeing convergence. *Remedy:* Damp site updates with $\delta \in (0, 1]$ as in `(eq:vi:ep-update)` (e.g., $\delta = 0.5$ decaying to $1/K$) and reject updates that violate positive-definiteness [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 10.7.4, Eq. 10.199, PDF p. 515).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[Reparameterization Trick]], [[Variational Inference & ELBO]]
- **Prerequisites (Directed $\leftarrow$):** [[Automatic Differentiation (AD)]], [[KL Divergence]], [[Variational Inference & ELBO]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[DDPM]], [[EM Algorithm]], [[Reparameterization Trick]], [[Variational Autoencoder (VAE)]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[EM Algorithm]], [[Exponential Families]], [[Gaussian Mixture Models (GMM)]], [[KL Divergence]], [[PPO (Proximal Policy Optimization)]], [[Reparameterization Trick]], [[Variational Autoencoder (VAE)]], [[Variational Inference & ELBO]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
