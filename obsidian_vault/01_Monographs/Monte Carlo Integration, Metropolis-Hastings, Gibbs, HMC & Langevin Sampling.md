---
type: monograph
domain: "D02"
parent_topic: "[[PT 02 - Latent Variables, Graphical Models, VI & MCMC Sampling]]"
sub_topic: "[[ST 02 - Graphical Models, Gaussian Processes, Variational Inference & MCMC]]"
cheatsheet: "[[Cheatsheet - Monte Carlo Integration, Metropolis-Hastings, Gibbs, HMC & Langevin Sampling]]"
part_of: 
  - "[[ST 02 - Graphical Models, Gaussian Processes, Variational Inference & MCMC]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
concepts: 
  - "[[MCMC & HMC Sampling]]"
prerequisites: 
  - "[[Graphical Models (BN & MRF)]]"
enables: []
related: 
  - "[[Energy-Based Models (EBM)]]"
  - "[[Score Matching (DSM & SSM)]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Tutorial on Diffusion Models for Imaging and Vision]]"
tags:
  - monograph
  - domain-02
---

# Monte Carlo Integration, Metropolis-Hastings, Gibbs, HMC & Langevin Sampling

> [!abstract] Monograph Navigation
> - **Domain:** [[D02 - Classical, Kernel & Probabilistic Machine Learning|Classical, Kernel & Probabilistic Machine Learning]]
> - **Parent / Sub-Topic:** [[PT 02 - Latent Variables, Graphical Models, VI & MCMC Sampling|Latent Variables, Graphical Models, VI & MCMC Sampling]] → [[ST 02 - Graphical Models, Gaussian Processes, Variational Inference & MCMC|Graphical Models, Gaussian Processes, Variational Inference & MCMC]]
> - **Visual Cheatsheet:** [[Cheatsheet - Monte Carlo Integration, Metropolis-Hastings, Gibbs, HMC & Langevin Sampling]]
> - **Description:** Synthesizes importance sampling and variance reduction (control variates), ergodic Markov chains and detailed balance, Metropolis-Hastings, Gibbs sampling, Hamiltonian Monte Carlo (leapfrog symplectic dynamics), and ULA/SGLD Langevin sampling.

## Problem Formulation & Unified Notation

In Bayesian inference, statistical physics, and energy-based generative modeling, computing posterior expectations or marginal likelihoods reduces to evaluating high-dimensional integrals over a state space $\Theta \subseteq \mathbb{R}^d$:

$$
\mu = \mathbb{E}_{\pi}[f(\boldsymbol{\theta})] = \int_{\Theta} f(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) \mathrm{d}\boldsymbol{\theta},
  \qquad
  \pi(\boldsymbol{\theta}) = \frac{\tilde{\pi}(\boldsymbol{\theta})}{Z},
  \qquad
  Z = \int_{\Theta} \tilde{\pi}(\boldsymbol{\theta}) \mathrm{d}\boldsymbol{\theta},
$$

where $f \colon \Theta \to \mathbb{R}$ is an integrable test function, $\tilde{\pi}(\boldsymbol{\theta}) \ge 0$ is an unnormalized target density (such as the joint $p(\boldsymbol{\theta}, \mathcal{D})$ or Boltzmann factor $\exp(-E(\boldsymbol{\theta}))$), and $Z$ is the typically intractable normalizing partition function [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 11.1–11.5, pp. 517–525) [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 14.1, pp. 445–452). Deterministic quadrature scales exponentially as $\mathcal{O}(K^d)$ and fails for $d \gg 1$. Monte Carlo methods bypass the curse of dimensionality in integration error by replacing grid summation with stochastic simulation, though designing proposals or Markov kernels that explore $\pi(\boldsymbol{\theta})$ efficiently in high dimensions remains the central algorithmic challenge.

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> The four synthesized sources adopt distinct conventions for state variables, unnormalized densities, energy functions, and step sizes. Table `tab:mcmc:notation` establishes a unified notation used throughout this leaf monograph.
>
> | **Concept** | **Unified Symbol** | **Source Variants** | **Reconciliation & Role** |
> | --- | --- | --- | --- |
> | State / Target | $\boldsymbol{\theta} \in \mathbb{R}^d$, $\pi(\boldsymbol{\theta})$ | $\boldsymbol{x}, \pi(\boldsymbol{x})$ [[Source - Probabilistic Machine Learning]] (p. 524);  $\boldsymbol{z}, p(\boldsymbol{z})$ [[Source - Foundations]] (p. 452);  $\boldsymbol{x}, p(\boldsymbol{x})$ [[Source - Tutorial on Diffusion Models for Imaging and Vision]] (p. 30) | $\boldsymbol{\theta}$ denotes continuous parameters or latent states; $\pi = \tilde{\pi}/Z$ is the normalized target. |
> | Potential Energy | $E(\boldsymbol{\theta}) = -\log\tilde{\pi}(\boldsymbol{\theta})$ | $E(\boldsymbol{\theta})$ [[Source - Probabilistic Machine Learning]] (p. 551);  $E(\boldsymbol{x}, \boldsymbol{w})$ [[Source - Foundations]] (p. 468);  $U(\boldsymbol{x})$ [[Source - Tutorial on Diffusion Models for Imaging and Vision]] (p. 32) | Gradient $-\nabla E(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}}\log\pi(\boldsymbol{\theta})$ equals the Stein score, independent of $Z$. |
> | HMC Momentum | $\boldsymbol{r} \in \mathbb{R}^d$, $\boldsymbol{M} \succ \boldsymbol{0}$ | $\boldsymbol{v}, \boldsymbol{\Sigma}$ [[Source - Probabilistic Machine Learning]] (p. 551);  $(\boldsymbol{q}, \boldsymbol{p})$ in physics | $\boldsymbol{r} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{M})$ with kinetic energy $K(\boldsymbol{r}) = \frac{1}{2}\boldsymbol{r}^{\top} \boldsymbol{M}^{-1} \boldsymbol{r}$ ($\boldsymbol{M}^{-1} = \boldsymbol{\Sigma}$). |
> | Step Size | $\epsilon > 0$ | $\eta$ [[Source - Probabilistic Machine Learning]] (p. 552);  $\eta$ [[Source - Foundations]] (p. 470);  $\tau$ [[Source - Tutorial on Diffusion Models for Imaging and Vision]] (p. 30) | $\epsilon$ denotes discrete integrator / SDE step size; $L$ denotes leapfrog steps per HMC proposal. |
> | Sample Count | $S$ (chains $M$) | $N_s$ [[Source - Probabilistic Machine Learning]] (p. 525);  $L$ [[Source - Foundations]] (p. 452);  $n$ [[Source - Deep Learning]] (p. 606) | $S$ avoids collision with $L$ (leapfrog steps) and $N$ (dataset size). |

## Core Mathematical Derivations & Theorems

### Standard Monte Carlo, Convergence Diagnostics ($\hat{R$), and Effective Sample Size}

> [!important] Unbiased Monte Carlo Variance & Correlated Chain ESS
> Given i.i.d. draws $\boldsymbol{\theta}^{(1)}, \dots, \boldsymbol{\theta}^{(S)} \overset{\mathrm{i.i.d.}}{\sim} \pi$, the empirical estimator $\hat{\mu}_S = \frac{1}{S}\sum_{s=1}^S f(\boldsymbol{\theta}^{(s)})$ is unbiased ($\mathbb{E}[\hat{\mu}_S] = \mu$) and, provided $\sigma^2 = \mathrm{Var}_{\pi}[f(\boldsymbol{\theta})] < \infty$, satisfies
>
> $$
> \mathrm{Var}[\hat{\mu}_S] = \frac{\sigma^2}{S},
>   \qquad
>   \sqrt{S}(\hat{\mu}_S - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2) \quad (S \to \infty).
> $$
>
> Crucially, the $\mathcal{O}(S^{-1/2})$ standard error is independent of ambient dimension $d$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 17.1.2, pp. 606–607). However, when samples form a stationary Markov chain with lag-$\ell$ autocorrelation $\rho(\ell) = \mathrm{Corr}[f(\boldsymbol{\theta}^{(0)}), f(\boldsymbol{\theta}^{(\ell)})]$, the estimator variance inflates by the integrated autocorrelation time $\tau_{\mathrm{int}}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 12.6.3, pp. 563–565):
>
> $$
> \begin{aligned}
> \mathrm{Var}[\hat{\mu}_S]
>   &= \frac{\sigma^2}{S^2} \sum_{i=1}^S \sum_{j=1}^S \mathrm{Corr}\bigl[f(\boldsymbol{\theta}^{(i)}), f(\boldsymbol{\theta}^{(j)})\bigr]
>    \xrightarrow{S \to \infty}  \frac{\sigma^2}{S} \underbrace{\left(1 + 2\sum_{\ell=1}^{\infty} \rho(\ell)\right)}_{\tau_{\mathrm{int}}}, \\
>   \mathrm{ESS} &\triangleq \frac{S}{1 + 2\sum_{\ell=1}^{\infty} \rho(\ell)}.
> \end{aligned}
> $$
>
> To diagnose non-stationarity across $M$ parallel chains of length $S$ (each split into halves to detect within-chain drift), let $\bar{f}_{\cdot m}$ be chain $m$'s mean, $\bar{f}_{\cdot\cdot}$ the grand mean, $W = \frac{1}{M}\sum_{m=1}^M s_m^2$ the mean within-chain variance, and $B = \frac{S}{M-1}\sum_{m=1}^M (\bar{f}_{\cdot m} - \bar{f}_{\cdot\cdot})^2$ the scaled between-chain variance. The Gelman–Rubin potential scale reduction factor is [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 12.6.2.2, pp. 562–563)
>
> $$
> \hat{V}^+ = \frac{S - 1}{S} W + \frac{1}{S} B,
>   \qquad
>   \hat{R} = \sqrt{\frac{\hat{V}^+}{W}},
> $$
>
> with $\hat{R} < 1.01$ required before declaring convergence.

### Importance Sampling & High-Dimensional $\chi^2$-Divergence Degeneracy

When direct sampling from $\pi$ is infeasible, importance sampling (IS) draws $\boldsymbol{\theta}^{(s)} \overset{\mathrm{i.i.d.}}{\sim} q$ from an easy-to-sample proposal satisfying $\mathrm{supp}(\pi) \subseteq \mathrm{supp}(q)$ [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 14.1.5, pp. 452–454) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 11.5, pp. 524–526). For unnormalized $\tilde{\pi} = Z\pi$, self-normalized importance sampling (SNIS) forms normalized weights $W_s = \tilde{w}_s / \sum_{j=1}^S \tilde{w}_j$ with $\tilde{w}_s = \tilde{\pi}(\boldsymbol{\theta}^{(s)}) / q(\boldsymbol{\theta}^{(s)})$, yielding the consistent (though finite-sample biased) ratio estimator $\hat{\mu}_{\mathrm{SNIS}} = \sum_{s=1}^S W_s f(\boldsymbol{\theta}^{(s)})$.

> [!important] Optimal Proposal & $\chi^2$-Divergence Weight Collapse
> For direct IS with known $Z$, minimizing $\mathrm{Var}_q[\frac{\pi(\boldsymbol{\theta})}{q(\boldsymbol{\theta})} f(\boldsymbol{\theta})]$ over proposals $q$ via Cauchy–Schwarz yields the zero-variance (for $f \ge 0$) optimal proposal $q^*(\boldsymbol{\theta}) \propto |f(\boldsymbol{\theta})|\pi(\boldsymbol{\theta})$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 17.2, pp. 608–609). For general $f$, the variance of the normalized weights $w(\boldsymbol{\theta}) = \pi(\boldsymbol{\theta})/q(\boldsymbol{\theta})$ is governed by the Pearson $\chi^2$-divergence:
>
> $$
> \mathrm{Var}_q[w(\boldsymbol{\theta})] = \mathbb{E}_q\!\left[\left(\frac{\pi(\boldsymbol{\theta})}{q(\boldsymbol{\theta})}\right)^2\right] - 1 = \chi^2(\pi  \|  q),
>   \qquad
>   \mathrm{ESS}_{\mathrm{IS}} \approx \frac{S}{1 + \chi^2(\pi  \|  q)} = \frac{\bigl(\sum_{s=1}^S \tilde{w}_s\bigr)^2}{\sum_{s=1}^S \tilde{w}_s^2}.
> $$
>
> Consider two isotropic $d$-dimensional Gaussians $\pi = \mathcal{N}(\boldsymbol{0}, \sigma_\pi^2 \boldsymbol{I}_d)$ and $q = \mathcal{N}(\boldsymbol{0}, \sigma_q^2 \boldsymbol{I}_d)$ with $\sigma_q^2 > \sigma_\pi^2 / 2$. Direct integration gives
>
> $$
> 1 + \chi^2(\pi  \|  q) = \left(\frac{\sigma_q^2}{\sigma_\pi \sqrt{2\sigma_q^2 - \sigma_\pi^2}}\right)^d.
> $$
>
> Because the base strictly exceeds $1$ whenever $\sigma_q \neq \sigma_\pi$, the required sample size $S$ to maintain fixed $\mathrm{ESS}_{\mathrm{IS}}$ grows exponentially in $d$, causing a single particle to capture $W_s \to 1$ while all others vanish.

### Markov Chain Ergodicity, Detailed Balance & Metropolis–Hastings

To avoid exponential importance-weight collapse, Markov chain Monte Carlo (MCMC) constructs an ergodic (irreducible and aperiodic) Markov transition kernel $T(\boldsymbol{\theta}' \mid \boldsymbol{\theta})$ admitting $\pi$ as its unique stationary distribution: $\pi(\boldsymbol{\theta}') = \int T(\boldsymbol{\theta}' \mid \boldsymbol{\theta})\pi(\boldsymbol{\theta}) \mathrm{d}\boldsymbol{\theta}$ [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 14.2.2, pp. 457–460). A sufficient condition for stationarity is *detailed balance* (reversibility):

$$
T(\boldsymbol{\theta}' \mid \boldsymbol{\theta}) \pi(\boldsymbol{\theta}) = T(\boldsymbol{\theta} \mid \boldsymbol{\theta}') \pi(\boldsymbol{\theta}'),
$$

since integrating over $\boldsymbol{\theta}$ immediately recovers $\int T(\boldsymbol{\theta}' \mid \boldsymbol{\theta})\pi(\boldsymbol{\theta}) \mathrm{d}\boldsymbol{\theta} = \pi(\boldsymbol{\theta}') \int T(\boldsymbol{\theta} \mid \boldsymbol{\theta}') \mathrm{d}\boldsymbol{\theta} = \pi(\boldsymbol{\theta}')$.

> [!important] Metropolis–Hastings Detailed Balance & Random-Walk $\mathcal{O}(d^2)$ Scaling
> Given a proposal $q(\boldsymbol{\theta}' \mid \boldsymbol{\theta})$, the Metropolis–Hastings (MH) kernel proposes $\boldsymbol{\theta}' \sim q(\cdot \mid \boldsymbol{\theta})$ and accepts with probability [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 12.2.1–12.2.2, pp. 534–536)
>
> $$
> \alpha(\boldsymbol{\theta}, \boldsymbol{\theta}') = \min\!\left(1,  \frac{\tilde{\pi}(\boldsymbol{\theta}') q(\boldsymbol{\theta} \mid \boldsymbol{\theta}')}{\tilde{\pi}(\boldsymbol{\theta}) q(\boldsymbol{\theta}' \mid \boldsymbol{\theta})}\right),
> $$
>
> where the unknown $Z$ cancels. For $\boldsymbol{\theta}' \neq \boldsymbol{\theta}$, the transition density is $T(\boldsymbol{\theta}' \mid \boldsymbol{\theta}) = q(\boldsymbol{\theta}' \mid \boldsymbol{\theta})\alpha(\boldsymbol{\theta}, \boldsymbol{\theta}')$. Multiplying by $\pi(\boldsymbol{\theta}) = \tilde{\pi}(\boldsymbol{\theta})/Z$ gives
>
> $$
> T(\boldsymbol{\theta}' \mid \boldsymbol{\theta}) \pi(\boldsymbol{\theta})
>   = \frac{1}{Z}\min\!\bigl(\tilde{\pi}(\boldsymbol{\theta})q(\boldsymbol{\theta}' \mid \boldsymbol{\theta}),  \tilde{\pi}(\boldsymbol{\theta}')q(\boldsymbol{\theta} \mid \boldsymbol{\theta}')\bigr)
>   = T(\boldsymbol{\theta} \mid \boldsymbol{\theta}') \pi(\boldsymbol{\theta}'),
> $$
>
> which is symmetric in $(\boldsymbol{\theta}, \boldsymbol{\theta}')$, proving detailed balance. However, for isotropic random-walk proposals $q(\boldsymbol{\theta}' \mid \boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta}, \sigma^2 \boldsymbol{I}_d)$, maintaining $\mathcal{O}(1)$ acceptance in $d$ dimensions forces $\sigma = \mathcal{O}(d^{-1/2})$ (optimal asymptotic acceptance $\approx 0.234$); because diffusive random walks traverse distance $\propto \sqrt{t} \sigma$ in $t$ steps [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 14.2.1, p. 457), exploring an $\mathcal{O}(1)$ target width requires $t = \mathcal{O}(d)$ steps per proposal scale, or $\mathcal{O}(d^2)$ gradient-free density evaluations for an independent sample (and $\mathcal{O}((L/\ell)^2)$ in anisotropic valleys of aspect ratio $L/\ell$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 12.3.7, p. 545)).

### Gibbs Sampling & Rao–Blackwellized (Collapsed) Gibbs

 Gibbs sampling updates one coordinate (or block) $\theta_j$ at a time from its exact full conditional $\pi(\theta_j \mid \boldsymbol{\theta}_{-j})$ [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 14.2.4, pp. 461–465) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 12.3, pp. 539–546). Viewing a single-site update as an MH proposal $q(\boldsymbol{\theta}' \mid \boldsymbol{\theta}) = \pi(\theta'_j \mid \boldsymbol{\theta}_{-j})\mathbb{I}(\boldsymbol{\theta}'_{-j} = \boldsymbol{\theta}_{-j})$ and factoring $\pi(\boldsymbol{\theta}) = \pi(\theta_j \mid \boldsymbol{\theta}_{-j})\pi(\boldsymbol{\theta}_{-j})$, the MH acceptance ratio simplifies identically to unity:

$$
\frac{\pi(\boldsymbol{\theta}') q(\boldsymbol{\theta} \mid \boldsymbol{\theta}')}{\pi(\boldsymbol{\theta}) q(\boldsymbol{\theta}' \mid \boldsymbol{\theta})}
  = \frac{\pi(\theta'_j \mid \boldsymbol{\theta}_{-j})\pi(\boldsymbol{\theta}_{-j}) \pi(\theta_j \mid \boldsymbol{\theta}_{-j})}{\pi(\theta_j \mid \boldsymbol{\theta}_{-j})\pi(\boldsymbol{\theta}_{-j}) \pi(\theta'_j \mid \boldsymbol{\theta}_{-j})} = 1.
$$

When conjugate priors permit analytic marginalization of parameters $\boldsymbol{\phi}$ (e.g., mixture weights/means in a Dirichlet-multinomial Gaussian mixture), *collapsed* or *Rao–Blackwellized* Gibbs sampling runs the Markov chain solely on the remaining latent variables $\boldsymbol{z}$ and estimates expectations via $\mathbb{E}[f(\boldsymbol{\phi}, \boldsymbol{z})] = \mathbb{E}_{\boldsymbol{z}}[\mathbb{E}[f(\boldsymbol{\phi}, \boldsymbol{z}) \mid \boldsymbol{z}]]$. By the Law of Total Variance, $\mathrm{Var}[\mathbb{E}[f \mid \boldsymbol{z}]] = \mathrm{Var}[f] - \mathbb{E}[\mathrm{Var}[f \mid \boldsymbol{z}]] \le \mathrm{Var}[f]$, strictly reducing Monte Carlo variance while eliminating coupling between $\boldsymbol{\phi}$ and $\boldsymbol{z}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 11.6.2 & 12.3.8, pp. 528, 546).

### Hamiltonian Monte Carlo (HMC), Symplectic Leapfrog & NUTS

To suppress random-walk diffusion, Hamiltonian Monte Carlo (HMC) augments $\boldsymbol{\theta} \in \mathbb{R}^d$ with an auxiliary momentum $\boldsymbol{r} \in \mathbb{R}^d$ and defines the Hamiltonian $H(\boldsymbol{\theta}, \boldsymbol{r}) = E(\boldsymbol{\theta}) + K(\boldsymbol{r})$, where $E(\boldsymbol{\theta}) = -\log\tilde{\pi}(\boldsymbol{\theta})$ and $K(\boldsymbol{r}) = \frac{1}{2}\boldsymbol{r}^{\top} \boldsymbol{M}^{-1} \boldsymbol{r}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 12.5.1, p. 551). The augmented target $\pi(\boldsymbol{\theta}, \boldsymbol{r}) \propto \exp(-H(\boldsymbol{\theta}, \boldsymbol{r})) = \tilde{\pi}(\boldsymbol{\theta})\exp(-K(\boldsymbol{r}))$ factorizes, preserving $\pi(\boldsymbol{\theta})$ as its exact marginal.

> [!important] Symplectic Leapfrog Volume Preservation, Reversibility & $\mathcal{O}(d^{1/4})$ Scaling
> Hamilton's continuous-time equations $\dot{\boldsymbol{\theta}} = \nabla_{\boldsymbol{r}} H = \boldsymbol{M}^{-1}\boldsymbol{r}$ and $\dot{\boldsymbol{r}} = -\nabla_{\boldsymbol{\theta}} H = -\nabla E(\boldsymbol{\theta})$ conserve total energy ($\frac{\mathrm{d}H}{\mathrm{d}t} = \nabla_{\boldsymbol{\theta}}H^{\top}\dot{\boldsymbol{\theta}} + \nabla_{\boldsymbol{r}}H^{\top}\dot{\boldsymbol{r}} = 0$) and preserve phase-space volume (Liouville's theorem: $\nabla \cdot (\dot{\boldsymbol{\theta}}, \dot{\boldsymbol{r}}) = 0$). Discretizing with step size $\epsilon$ via the $L$-step *leapfrog integrator* executes, for each step:
>
> $$
> \boldsymbol{r}_{t+1/2} = \boldsymbol{r}_t - \frac{\epsilon}{2}\nabla E(\boldsymbol{\theta}_t), \qquad
>   \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \epsilon \boldsymbol{M}^{-1} \boldsymbol{r}_{t+1/2}, \qquad
>   \boldsymbol{r}_{t+1} = \boldsymbol{r}_{t+1/2} - \frac{\epsilon}{2}\nabla E(\boldsymbol{\theta}_{t+1}).
> $$
>
> Each sub-step is a shear transformation whose Jacobian is block-triangular with unit diagonal, so $|\det \boldsymbol{J}_{\mathrm{LF}}| = 1$ (exact volume preservation, eliminating Jacobian factors in MH). Composing $L$ steps with a momentum negation $\Phi_{\mathrm{flip}}(\boldsymbol{\theta}, \boldsymbol{r}) = (\boldsymbol{\theta}, -\boldsymbol{r})$ yields an involution $(\Phi_{\mathrm{flip}} \circ \Psi_{\mathrm{LF}}^L)^2 = \mathrm{Id}$, guaranteeing exact time-reversibility. Because the leapfrog integrator is second-order symplectic, the global Hamiltonian error remains bounded at $\Delta H = H(\boldsymbol{\theta}^*, \boldsymbol{r}^*) - H(\boldsymbol{\theta}_0, \boldsymbol{r}_0) = \mathcal{O}(\epsilon^2)$ over arbitrarily long trajectories, and MH acceptance $\alpha = \min(1, \exp(-\Delta H))$ requires $\epsilon = \mathcal{O}(d^{-1/4})$ to keep $\Delta H = \mathcal{O}(1)$, yielding $\mathcal{O}(d^{1/4})$ leapfrog steps per nearly independent sample [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 12.5.2–12.5.3, pp. 552–554).

> [!tip] Typical Set Geometry & NUTS U-Turn Criterion
> In high dimensions, probability mass concentrates not at the mode $\arg\max \pi(\boldsymbol{\theta})$, where volume is negligible, nor in the tails, where density vanishes, but on a thin shell called the *typical set* where density and volume balance. HMC glides coherently along level sets of $H$ inside this typical set. To prevent trajectories from wasting computation by doubling back, the No-U-Turn Sampler (NUTS) builds a recursive binary tree forward and backward in time and halts when the span $(\boldsymbol{\theta}^+ - \boldsymbol{\theta}^-)$ ceases to increase:
>
> $$
> (\boldsymbol{\theta}^+ - \boldsymbol{\theta}^-)^{\top} \boldsymbol{M}^{-1} \boldsymbol{r}^+ < 0
>   \quad\text{or}\quad
>   (\boldsymbol{\theta}^+ - \boldsymbol{\theta}^-)^{\top} \boldsymbol{M}^{-1} \boldsymbol{r}^- < 0,
> $$
>
> thus adaptively selecting the trajectory length $L$ without manual tuning [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 12.5.4.1, p. 554).

### Langevin Dynamics (ULA, MALA, SGLD) & Sequential Monte Carlo

 The overdamped Langevin stochastic differential equation (SDE) evolves $\boldsymbol{\theta}_t \in \mathbb{R}^d$ via drift along the Stein score $\nabla_{\boldsymbol{\theta}}\log\pi(\boldsymbol{\theta}_t) = -\nabla E(\boldsymbol{\theta}_t)$ plus isotropic Brownian motion $\boldsymbol{W}_t$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 12.5.6, p. 556) [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 14.3.3, p. 470) [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 3.1, p. 30):

$$
\mathrm{d}\boldsymbol{\theta}_t = \nabla_{\boldsymbol{\theta}}\log\pi(\boldsymbol{\theta}_t) \mathrm{d}t + \sqrt{2} \mathrm{d}\boldsymbol{W}_t.
$$

By the Fokker–Planck equation, $\pi(\boldsymbol{\theta})$ is the unique stationary density of `(eq:mcmc:langevin_sde)`. Euler–Maruyama discretization with step size $\epsilon$ gives the *Unadjusted Langevin Algorithm (ULA)*:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \epsilon \nabla_{\boldsymbol{\theta}}\log\pi(\boldsymbol{\theta}_t) + \sqrt{2\epsilon} \boldsymbol{\xi}_t, \qquad \boldsymbol{\xi}_t \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_d).
$$

Because finite $\epsilon > 0$ introduces $\mathcal{O}(\epsilon)$ asymptotic bias, the *Metropolis-Adjusted Langevin Algorithm (MALA)* treats `(eq:mcmc:ula)` as an asymmetric Gaussian proposal $q(\boldsymbol{\theta}' \mid \boldsymbol{\theta}_t) = \mathcal{N}(\boldsymbol{\theta}_t + \epsilon\nabla\log\pi(\boldsymbol{\theta}_t),  2\epsilon\boldsymbol{I}_d)$ and applies an MH accept/reject step—which is algebraically equivalent to $L=1$ leapfrog HMC with step size $\sqrt{2\epsilon}$ and scales as $\mathcal{O}(d^{1/3})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 12.5.6, pp. 555–556). Replacing $\nabla\log\pi$ with minibatch gradients yields Stochastic Gradient Langevin Dynamics (SGLD) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 12.7.1, p. 568), while replacing it with a learned score network $\boldsymbol{s}_{\boldsymbol{\phi}}(\boldsymbol{x}, \sigma)$ yields Score-Matching Langevin Dynamics (SMLD) for diffusion generative models [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 3.1, pp. 30–32).

For sequential state-space posteriors $\pi_t(\boldsymbol{z}_{1:t}) = \tilde{\gamma}_t(\boldsymbol{z}_{1:t})/Z_t$, *Sequential Monte Carlo (SMC)* propagates $N_s$ particles via $q_t(\boldsymbol{z}_t \mid \boldsymbol{z}_{1:t-1})$ with recursive incremental weights $\alpha_t^{(i)} = \frac{\tilde{\gamma}_t(\boldsymbol{z}_{1:t}^{(i)})}{\tilde{\gamma}_{t-1}(\boldsymbol{z}_{1:t-1}^{(i)}) q_t(\boldsymbol{z}_t^{(i)} \mid \boldsymbol{z}_{1:t-1}^{(i)})}$ and triggers low-variance resampling (stratified or systematic) whenever $\mathrm{ESS}_{\mathrm{IS}} < N_s/2$ to prevent weight degeneracy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 13.2, pp. 581–587).

## Algorithmic Implementation & Comparative Synthesis

Algorithms `alg:mcmc:hmc` and `alg:mcmc:mala` formalize HMC and MALA in unified notation; Table `tab:mcmc:comparison` summarizes asymptotic mixing and computational trade-offs across all sampler families.

> [!example] Algorithm: Hamiltonian Monte Carlo (HMC) with Symplectic Leapfrog Integrator
> **Input:** Unnormalized log-density $\log\tilde{\pi}(\boldsymbol{\theta}) = -E(\boldsymbol{\theta})$, mass matrix $\boldsymbol{M}$, step size $\epsilon$, leapfrog steps $L$, iterations $S$, initial $\boldsymbol{\theta}^{(0)}$
> **Output:** Markov chain $\{\boldsymbol{\theta}^{(s)}\}_{s=1}^S$
> **for** $s = 1, \dots, S$:
>
>   Sample auxiliary momentum $\boldsymbol{r}_0 \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{M})$
>   Initialize $(\boldsymbol{\theta}'_0, \boldsymbol{r}'_0) \leftarrow (\boldsymbol{\theta}^{(s-1)}, \boldsymbol{r}_0)$
>   $\boldsymbol{r}'_{1/2} \leftarrow \boldsymbol{r}'_0 - \frac{\epsilon}{2}\nabla E(\boldsymbol{\theta}'_0)$  *// Initial half-kick*
>   **for** $\ell = 1, \dots, L-1$:
>
>     $\boldsymbol{\theta}'_{\ell} \leftarrow \boldsymbol{\theta}'_{\ell-1} + \epsilon \boldsymbol{M}^{-1} \boldsymbol{r}'_{\ell-1/2}$  *// Full position drift*
>     $\boldsymbol{r}'_{\ell+1/2} \leftarrow \boldsymbol{r}'_{\ell-1/2} - \epsilon \nabla E(\boldsymbol{\theta}'_{\ell})$  *// Full momentum kick*
>
>   $\boldsymbol{\theta}'_L \leftarrow \boldsymbol{\theta}'_{L-1} + \epsilon \boldsymbol{M}^{-1} \boldsymbol{r}'_{L-1/2}$
>   $\boldsymbol{r}'_L \leftarrow \boldsymbol{r}'_{L-1/2} - \frac{\epsilon}{2}\nabla E(\boldsymbol{\theta}'_L)$  *// Final half-kick*
>   Compute $\Delta H = E(\boldsymbol{\theta}'_L) + \frac{1}{2}(\boldsymbol{r}'_L)^{\top} \boldsymbol{M}^{-1} \boldsymbol{r}'_L - E(\boldsymbol{\theta}^{(s-1)}) - \frac{1}{2}\boldsymbol{r}_0^{\top} \boldsymbol{M}^{-1} \boldsymbol{r}_0$
>   Sample $u \sim \mathcal{U}(0, 1)$; **if** $u \le \min(1, \exp(-\Delta H))$ **then** $\boldsymbol{\theta}^{(s)} \leftarrow \boldsymbol{\theta}'_L$ **else** $\boldsymbol{\theta}^{(s)} \leftarrow \boldsymbol{\theta}^{(s-1)}$

> [!example] Algorithm: Metropolis-Adjusted Langevin Algorithm (MALA) / ULA
> **Input:** Score $\nabla\log\tilde{\pi}(\boldsymbol{\theta}) = -\nabla E(\boldsymbol{\theta})$, step size $\epsilon$, iterations $S$, initial $\boldsymbol{\theta}^{(0)}$, boolean `adjust`
> **Output:** Samples $\{\boldsymbol{\theta}^{(s)}\}_{s=1}^S$
> **for** $s = 1, \dots, S$:
>
>   Sample $\boldsymbol{\xi}_s \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_d)$ and propose $\boldsymbol{\theta}' \leftarrow \boldsymbol{\theta}^{(s-1)} + \epsilon \nabla\log\tilde{\pi}(\boldsymbol{\theta}^{(s-1)}) + \sqrt{2\epsilon} \boldsymbol{\xi}_s$
>   **if** `adjust` is `true` (MALA):
>
>     Evaluate $\log q(\boldsymbol{\theta} \mid \boldsymbol{\theta}') = -\frac{1}{4\epsilon}\lVert \boldsymbol{\theta} - \boldsymbol{\theta}' - \epsilon\nabla\log\tilde{\pi}(\boldsymbol{\theta}') \rVert_2^2 + C$
>     $\alpha \leftarrow \min\!\left(1,  \frac{\tilde{\pi}(\boldsymbol{\theta}') q(\boldsymbol{\theta}^{(s-1)} \mid \boldsymbol{\theta}')}{\tilde{\pi}(\boldsymbol{\theta}^{(s-1)}) q(\boldsymbol{\theta}' \mid \boldsymbol{\theta}^{(s-1)})}\right)$
>     Sample $u \sim \mathcal{U}(0, 1)$; **if** $u \le \alpha$ **then** $\boldsymbol{\theta}^{(s)} \leftarrow \boldsymbol{\theta}'$ **else** $\boldsymbol{\theta}^{(s)} \leftarrow \boldsymbol{\theta}^{(s-1)}$
>
> **else**:
>
>     $\boldsymbol{\theta}^{(s)} \leftarrow \boldsymbol{\theta}'$  *// Unadjusted Langevin (ULA): $\mathcal{O}(\epsilon)$ bias*

**Table:** Comparative synthesis of Monte Carlo and MCMC samplers in $d$ dimensions.

| **Method** | **Gradients** | **Dim. Scaling** | **Asymptotic Bias** | **Primary Strength / Bottleneck** |
| --- | --- | --- | --- | --- |
| IS / SNIS | None | $\mathcal{O}(\exp(d))$ | $0$ ($S\to\infty$) | Parallelizable; collapses under $\chi^2(\pi\parallel q)$ mismatch |
| RW-MH | None | $\mathcal{O}(d^2)$ | $0$ | Universal; diffusive random walk in high $d$ |
| Gibbs (Blocked) | None | $\mathcal{O}((L/\ell)^2)$ | $0$ | Acceptance $1$; stalls under strong correlation |
| ULA / SGLD | $\nabla\log\tilde{\pi}$ | $\mathcal{O}(d^{1/3})$ | $\mathcal{O}(\epsilon)$ | Scales to $N\gg 1$ minibatches; step-size bias |
| MALA ($L=1$ HMC) | $\nabla\log\tilde{\pi}$ | $\mathcal{O}(d^{1/3})$ | $0$ | Exact stationary $\pi$; local-only steps |
| HMC / NUTS | $\nabla\log\tilde{\pi}$ | $\mathcal{O}(d^{1/4})$ | $0$ | Coherent typical-set orbits; fails on sharp funnels |
| SMC (SISR) | Optional | Particle $N_s$ | $0$ ($N_s\to\infty$) | Handles multimodality via tempering; path degeneracy |

> [!info] Cross-Book Synthesis: Energy-Based Models, Mode Mixing & Annealed Bridges
> [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 17.5, pp. 614–617) and [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 14.3, pp. 466–470) highlight a fundamental tension in energy-based models (EBMs) $p_{\boldsymbol{w}}(\boldsymbol{x}) = \exp(-E(\boldsymbol{x}, \boldsymbol{w}))/Z(\boldsymbol{w})$: maximum-likelihood gradients require sampling from $p_{\boldsymbol{w}}$ via $\nabla_{\boldsymbol{w}}\mathbb{E}_{p_{\mathcal{D}}}[\log p_{\boldsymbol{w}}(\boldsymbol{x})] = -\mathbb{E}_{p_{\mathcal{D}}}[\nabla_{\boldsymbol{w}}E] + \mathbb{E}_{p_{\boldsymbol{w}}}[\nabla_{\boldsymbol{w}}E]$, yet sharp learned modes separated by high-energy barriers trap single-chain Gibbs or HMC samplers. [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 3.1, pp. 30–32) resolves this in generative modeling by running score-driven Langevin dynamics across annealed noise scales, directly paralleling Annealed Importance Sampling (AIS) and tempered SMC in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 11.5.4 & 13.5, pp. 526, 600).

## Visual Diagnostics & Geometric Structure

Figure `fig:mcmc:architecture` contrasts the diffusive proposal–reject loop of Random-Walk MH with the symplectic phase-space $(\boldsymbol{\theta}, \boldsymbol{r})$ trajectory of HMC, while Figure `fig:mcmc:scaling` quantifies both the $\mathcal{O}(d^{1/4})$ vs. $\mathcal{O}(d^2)$ dimensional scaling and symplectic energy conservation.

> [!example] Visual Diagram (LaTeX / PDF)
> *Architectural contrast between Random-Walk Metropolis–Hastings (top, diffusive $\mathcal{O}(d^2)$ mixing) and Hamiltonian Monte Carlo (bottom, coherent volume-preserving leapfrog orbits in phase space $(\boldsymbol{\theta}, \boldsymbol{r})$ with $\mathcal{O}(d^{1/4})$ mixing).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Left: Asymptotic cost per independent sample as a function of dimension $d$. Right: Standard Euler integration exhibits secular energy drift $|\Delta H|$ (collapsing MH acceptance), whereas the symplectic leapfrog integrator preserves a shadow Hamiltonian with bounded $\mathcal{O}(\epsilon^2)$ oscillation.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Empirical diagnostics from three synthesized textbooks further illuminate these geometric behaviors. Figure `fig:mcmc:gibbs_paths` illustrates axis-aligned Gibbs sampling on independent, correlated, and separated bimodal Gaussians; Figure `fig:mcmc:neals_funnel` demonstrates HMC failure in Neal's funnel and its resolution via non-centered parameterization; and Figure `fig:mcmc:langevin_traj` visualizes Langevin SDE trajectories converging to a bimodal Gaussian mixture.

> [!note] Source Figure
> *Figure:* Gibbs sampling trajectories on three 2D distributions [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Fig. 17.1, p. 616): (left) rapid mixing under independence; (center) slow diagonal crawling under strong correlation; (right) complete mode trapping when modes are separated by low-density barriers. (`deep_learning_goodfellow_bengio_courville_2017_p616_fig1.png`)

> [!note] Source Figure
> *Figure:* Neal's funnel geometry ($v \sim \mathcal{N}(0, 9)$, $x \mid v \sim \mathcal{N}(0, e^v)$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 12.18, p. 566): (a) true joint density; (b) centered HMC fails to enter the narrow neck ($v < -3$) due to extreme curvature; (c) non-centered reparameterization $x = e^{v/2}\tilde{x}$ flattens the geometry and restores uniform exploration. (`probabilisticml_kevinmurphy_advancedtopics_p566_fig1.png`)

> [!note] Source Figure
> *Figure:* Score-driven Langevin dynamics trajectories initialized across $[-10, 10]$ converging to the two modes of a Gaussian mixture under $\boldsymbol{x}_{t+1} = \boldsymbol{x}_t + \epsilon\nabla\log p(\boldsymbol{x}_t) + \sqrt{2\epsilon} \boldsymbol{\xi}_t$ [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Fig. 17, p. 31). (`diffusionmodels_stanleychan_p31_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Importance-Weight Collapse in High Dimensions
> Even when $\mathrm{supp}(\pi) \subseteq \mathrm{supp}(q)$, any mismatch in scale or location between $q$ and $\pi$ inflates $1 + \chi^2(\pi \| q)$ exponentially in $d$ (Eq. `(eq:mcmc:is_curse)`). **Remedy:** Monitor $\mathrm{ESS}_{\mathrm{IS}} = (\sum_s \tilde{w}_s)^2 / \sum_s \tilde{w}_s^2$ in log-space via log-sum-exp; for sequential or multimodal targets, bridge via Annealed Importance Sampling or SMC with adaptive resampling when $\mathrm{ESS}_{\mathrm{IS}} < N_s/2$.

> [!warning] Multimodal Trapping & Funnel Curvature in HMC
> In hierarchical models with varying curvature (such as Neal's funnel, Fig. `fig:mcmc:neals_funnel`), a global step size $\epsilon$ tuned for the wide mouth diverges in the narrow neck, causing leapfrog trajectories to reject repeatedly and under-explore the neck. Moreover, single-chain HMC cannot cross wide zero-density barriers between separated modes. **Remedy:** Apply non-centered reparameterizations ($\boldsymbol{\theta} = \boldsymbol{\mu} + \sigma\tilde{\boldsymbol{\theta}}$ with $\tilde{\boldsymbol{\theta}} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$), diagonal/dense mass-matrix adaptation $\boldsymbol{M}$ during warmup, and tempered SMC or parallel tempering for multimodal targets.

> [!warning] Non-Symplectic Euler Drift & Unadjusted Langevin Bias
> Replacing leapfrog with standard or modified Euler integration violates volume preservation ($|\det \boldsymbol{J}| \neq 1$) and induces secular energy drift $\Delta H \to \infty$, driving HMC acceptance to zero. Similarly, running ULA/SGLD with fixed $\epsilon > 0$ omits the MH correction and samples from a biased stationary distribution at $\mathcal{O}(\epsilon)$ Wasserstein/KL error. **Remedy:** Always use a symplectic reversible integrator (leapfrog) for HMC, or apply MALA / decaying step-size schedules $\sum_t \epsilon_t = \infty, \sum_t \epsilon_t^2 < \infty$ in SGLD.

> [!warning] Insufficient Burn-In, Correlated $\mathrm{ESS} \ll S$, and Single-Chain Blindness
> Reporting raw iteration counts $S$ when lag-$\ell$ autocorrelation decays slowly overestimates precision by $\tau_{\mathrm{int}} = S/\mathrm{ESS}$, while computing $\hat{R}$ on a single mode-trapped chain can falsely yield $\hat{R} \approx 1.00$. **Remedy:** Initialize $M \ge 4$ overdispersed chains, discard warmup, verify split-$\hat{R} < 1.01$ and bulk/tail $\mathrm{ESS} \ge 400$, and inspect rank-normalized trace plots (trankplots) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 12.6.2, pp. 561–563).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (1):** [[MCMC & HMC Sampling]]
- **Prerequisites (Directed $\leftarrow$):** [[Graphical Models (BN & MRF)]]
- **Enables / Unlocks (Directed $\rightarrow$):** None
- **Related Concepts (Undirected $\leftrightarrow$):** [[Energy-Based Models (EBM)]], [[Score Matching (DSM & SSM)]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]]
