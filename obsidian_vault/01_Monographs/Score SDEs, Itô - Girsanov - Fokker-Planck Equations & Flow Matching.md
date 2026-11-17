---
type: monograph
domain: "D04"
parent_topic: "[[PT 04 - Diffusion Models, Score SDEs, Flow Matching & Fast Samplers]]"
sub_topic: "[[ST 04 - DDPM, NCSN, Score SDEs & Flow Matching]]"
cheatsheet: "[[Cheatsheet - Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]"
part_of: 
  - "[[ST 04 - DDPM, NCSN, Score SDEs & Flow Matching]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
concepts: 
  - "[[Flow Matching & Rectified Flow]]"
  - "[[Fokker-Planck Equation]]"
  - "[[Probability Flow ODE]]"
  - "[[Score-Based SDE]]"
prerequisites: 
  - "[[DDPM]]"
  - "[[Fokker-Planck Equation]]"
  - "[[Probability Flow ODE]]"
  - "[[Score Matching (DSM & SSM)]]"
  - "[[Score-Based SDE]]"
  - "[[Wasserstein Distance]]"
enables: 
  - "[[Consistency Models]]"
  - "[[Flow Matching & Rectified Flow]]"
  - "[[Fokker-Planck Equation]]"
  - "[[Probability Flow ODE]]"
related: 
  - "[[Consistency Models]]"
  - "[[DDPM]]"
  - "[[Flow Matching & Rectified Flow]]"
  - "[[Fokker-Planck Equation]]"
  - "[[Normalizing Flows]]"
  - "[[Probability Flow ODE]]"
  - "[[Score Matching (DSM & SSM)]]"
  - "[[Score-Based SDE]]"
  - "[[Wasserstein Distance]]"
sources: 
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
  - "[[Source - Tutorial on Diffusion Models for Imaging and Vision]]"
tags:
  - monograph
  - domain-04
---

# Score SDEs, Itô/Girsanov/Fokker-Planck Equations & Flow Matching

> [!abstract] Monograph Navigation
> - **Domain:** [[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion|Deep Generative Modeling: VAEs, Flows, EBMs, GANs & Diffusion]]
> - **Parent / Sub-Topic:** [[PT 04 - Diffusion Models, Score SDEs, Flow Matching & Fast Samplers|Diffusion Models, Score SDEs, Flow Matching & Fast Samplers]] → [[ST 04 - DDPM, NCSN, Score SDEs & Flow Matching|DDPM, NCSN, Score SDEs & Flow Matching]]
> - **Visual Cheatsheet:** [[Cheatsheet - Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]
> - **Description:** Unifies continuous-time VE/VP/sub-VP SDEs, Anderson's reverse-time SDE, deterministic Probability Flow ODE via Fokker-Planck, Girsanov likelihood weighting, conditional Flow Matching / Rectified Flow velocity fields, and dynamic Benamou-Brenier OT connections.

## Problem Formulation & Unified Notation

Discrete-time Denoising Diffusion Probabilistic Models (DDPMs) and Noise-Conditional Score Networks (NCSNs / SMLD) both corrupt data $\boldsymbol{x}_0 \sim p_{\mathrm{data}}$ through a finite sequence of noise perturbations and learn to invert the process. Passing to the continuum limit ($\Delta t \to 0$, $N \to \infty$) unifies both paradigms under a single mathematical object: an It\^o stochastic differential equation (SDE) over $t \in [0, T]$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 4.1.1, pp. 87–89) [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 4.1, pp. 39–40) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.4.1, p. 877):

$$
\mathrm{d}\boldsymbol{x}_t = \boldsymbol{f}(\boldsymbol{x}_t, t) \mathrm{d}t + g(t) \mathrm{d}\boldsymbol{w}_t, \qquad \boldsymbol{x}_0 \sim p_0 = p_{\mathrm{data}},
$$

where $\boldsymbol{f} : \mathbb{R}^D \times [0, T] \to \mathbb{R}^D$ is the *drift coefficient*, $g : [0, T] \to \mathbb{R}_{\ge 0}$ is the scalar *diffusion coefficient*, and $\boldsymbol{w}_t \in \mathbb{R}^D$ is a standard $D$-dimensional Wiener process (Brownian motion) whose infinitesimal increments satisfy $\mathrm{d}\boldsymbol{w}_t \sim \mathcal{N}(\boldsymbol{0}, \mathrm{d}t \boldsymbol{I})$. The forward SDE induces a continuous family of marginal densities $\{p_t(\boldsymbol{x})\}_{t \in [0, T]}$ evolving from $p_0 = p_{\mathrm{data}}$ to a tractable terminal prior $p_T \approx p_{\mathrm{prior}}$.

Flow Matching (FM) and Rectified Flow (RF) generalize this transport perspective beyond Gaussian diffusion [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 5.2, p. 136). Instead of prescribing a stochastic forward noising SDE, one directly parameterizes a time-dependent velocity field $\boldsymbol{v}_\phi(\boldsymbol{x}_t, t)$ governing an ordinary differential equation (ODE) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.4.7, p. 881),

$$
\frac{\mathrm{d}\boldsymbol{x}_t}{\mathrm{d}t} = \boldsymbol{v}_\phi(\boldsymbol{x}_t, t), \qquad \boldsymbol{x}_0 \sim p_{\mathrm{src}},
$$

whose flow map $\Psi_{0 \to t}$ pushes an arbitrary source distribution $p_{\mathrm{src}}$ to a target distribution $p_{\mathrm{tgt}}$ along a prescribed probability path $\{p_t\}_{t \in [0, 1]}$.

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> | **Concept** | **Diffusion Principles (Ch. 4–5, App. B–C)** | **Chan (Sec. 4)** | **Murphy Adv. (Sec. 25.4)** |
> | — | — | — | — |
> | **Time Direction** | $t=0$ (data) $\to T$ (noise) in Score SDE; $t=0$ ($p_{\mathrm{src}}$) $\to 1$ ($p_{\mathrm{tgt}}$) in FM. | $t=0$ (data) $\to t=1$ (noise) with $\Delta t = 1/N$. | $t=0$ (data) $\to t=1$ (noise), $t \in [0, 1]$. |
> | **Marginal Density** | $p_t(\boldsymbol{x}_t)$ (with $p_{\mathrm{prior}} \approx p_T$); conditional $p_t(\boldsymbol{x}_t \mid \boldsymbol{x}_0)$. | $p_t(\boldsymbol{x})$ in continuous limit; $p_i(\boldsymbol{x}_i)$ at step $i$. | $q_t(\boldsymbol{x}_t)$ for forward marginals; $p_t(\boldsymbol{x})$ in PF-ODE. |
> | **Score Network** | $\boldsymbol{s}_\phi(\boldsymbol{x}, t) \approx \nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$. | $\boldsymbol{s}_\theta(\boldsymbol{x}_i, i)$ or $\boldsymbol{\epsilon}_\theta(\boldsymbol{x}(t), t)$. | $\boldsymbol{s}_\theta(\boldsymbol{x}_t, t) = -\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)/\sigma_t$. |
> | **Reverse Time** | $\bar{\boldsymbol{x}}(t)$ with reverse Wiener $\bar{\boldsymbol{w}}(t) := \boldsymbol{w}(T-t)-\boldsymbol{w}(T)$. | Uses $\mathrm{d}\boldsymbol{x} = \boldsymbol{x}(t) - \boldsymbol{x}(t-\Delta t)$ and $\mathrm{d}\boldsymbol{w} = -\boldsymbol{z}(t)\sqrt{\Delta t}$. | $\mathrm{d}\boldsymbol{x}_t = [\boldsymbol{f} - g^2 \nabla \log q_t] \mathrm{d}t + g \mathrm{d}\boldsymbol{w}_t$ with $\mathrm{d}t < 0$. |
> | **FM Velocity** | $\boldsymbol{v}_t(\boldsymbol{x}_t)$ (Eulerian) or $\partial_t \Psi_{0 \to t} \circ \Psi_{0 \to t}^{-1}$ (Lagrangian). | Focuses on semi-linear ODE $\dot{\boldsymbol{x}} = a(t)\boldsymbol{x} + b(t)$ for fast solvers. | Notes FM as simulation-free CNF without Gaussian noise. |

## Core Mathematical Derivations & Theorems

### Continuous-Time Limits of DDPM (VP-SDE) and SMLD (VE-SDE)

> [!important] Derivation of VP-SDE and VE-SDE Perturbation Kernels
> **1. DDPM $\to$ Variance Preserving (VP) SDE.** Let $\Delta t = 1/N$, $t = i/N \in [0, 1]$, and $\beta_i = \beta(t)\Delta t$ for a continuous schedule $\beta : [0, 1] \to \mathbb{R}_{\ge 0}$ [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 4.3, p. 43) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.4.1, p. 877). Expanding the DDPM step $\boldsymbol{x}_{t+\Delta t} = \sqrt{1 - \beta(t)\Delta t} \boldsymbol{x}_t + \sqrt{\beta(t)\Delta t} \boldsymbol{\epsilon}_t$ via $\sqrt{1 - u} = 1 - \frac{1}{2}u + \mathcal{O}(u^2)$ yields
>
> $$
> \boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t = -\frac{1}{2}\beta(t)\boldsymbol{x}_t \Delta t + \sqrt{\beta(t)} \sqrt{\Delta t} \boldsymbol{\epsilon}_t + \mathcal{O}(\Delta t^2)
>    \xrightarrow{\Delta t \to 0} 
>   \mathrm{d}\boldsymbol{x}_t = -\frac{1}{2}\beta(t)\boldsymbol{x}_t \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}\boldsymbol{w}_t.
> $$
>
>
> **2. SMLD / NCSN $\to$ Variance Exploding (VE) SDE.** Discretizing a strictly increasing noise scale $\sigma(t)$ gives the Markov chain $\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \sqrt{\sigma^2(t+\Delta t) - \sigma^2(t)} \boldsymbol{\epsilon}_t$ [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 4.4, p. 45) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 4.3.1, p. 110). Letting $\Delta t \to 0$ produces zero drift and diffusion $g(t) = \sqrt{\frac{\mathrm{d}[\sigma^2(t)]}{\mathrm{d}t}}$:
>
> $$
> \mathrm{d}\boldsymbol{x}_t = \sqrt{\frac{\mathrm{d}[\sigma^2(t)]}{\mathrm{d}t}} \mathrm{d}\boldsymbol{w}_t.
> $$
>
>
> **3. Closed-Form Transition Kernels for Linear SDEs.** Whenever $\boldsymbol{f}(\boldsymbol{x}, t) = f(t)\boldsymbol{x}$ is linear in $\boldsymbol{x}$, the conditional distribution $p_t(\boldsymbol{x}_t \mid \boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_t;  \boldsymbol{m}(t), P(t)\boldsymbol{I})$ remains Gaussian even for non-Gaussian $p_{\mathrm{data}}$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 4.3.3, pp. 112–114). Its moments satisfy the linear ODEs
>
> $$
> \frac{\mathrm{d}\boldsymbol{m}(t)}{\mathrm{d}t} = f(t)\boldsymbol{m}(t), \qquad \frac{\mathrm{d}P(t)}{\mathrm{d}t} = 2f(t)P(t) + g^2(t),
> $$
>
> with $\boldsymbol{m}(0) = \boldsymbol{x}_0$ and $P(0) = 0$. Solving via the integrating factor $E(s \to t) = \exp\!\left(\int_s^t f(u) \mathrm{d}u\right)$ gives:
>
> $$
> \begin{aligned}
> \text{VP-SDE:}\quad & p_t(\boldsymbol{x}_t \mid \boldsymbol{x}_0) = \mathcal{N}\!\left(\boldsymbol{x}_t;  e^{-\frac{1}{2}B(t)}\boldsymbol{x}_0,  \bigl(1 - e^{-B(t)}\bigr)\boldsymbol{I}\right), \qquad B(t) := \int_0^t \beta(s) \mathrm{d}s,  \\
>   \text{VE-SDE:}\quad & p_t(\boldsymbol{x}_t \mid \boldsymbol{x}_0) = \mathcal{N}\!\left(\boldsymbol{x}_t;  \boldsymbol{x}_0,  \bigl(\sigma^2(t) - \sigma^2(0)\bigr)\boldsymbol{I}\right).
> \end{aligned}
> $$

### It\^o's Lemma & Derivation of the Fokker-Planck Equation

Because sample paths of a Wiener process are almost surely continuous yet nowhere differentiable, classical calculus fails. Specifically, Brownian increments scale as $\mathrm{d}\boldsymbol{w}_t = \mathcal{O}(\sqrt{\mathrm{d}t})$ and obey the quadratic variation rule $(\mathrm{d}w_{t,i})(\mathrm{d}w_{t,j}) = \delta_{ij} \mathrm{d}t$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. C.1.1–C.1.2, pp. 413–415). Retaining $\mathcal{O}(\mathrm{d}t)$ terms in a second-order stochastic Taylor expansion yields It\^o's Lemma.

> [!important] Multidimensional It\^o's Lemma & Fokker-Planck Derivation
> Let $\boldsymbol{x}_t \in \mathbb{R}^D$ solve $\mathrm{d}\boldsymbol{x}_t = \boldsymbol{f}(\boldsymbol{x}_t, t) \mathrm{d}t + g(t) \mathrm{d}\boldsymbol{w}_t$, and let $\varphi : \mathbb{R}^D \times [0, T] \to \mathbb{R}$ be $C^{2,1}$-smooth. Expanding $\varphi(\boldsymbol{x}_{t+\mathrm{d}t}, t+\mathrm{d}t) - \varphi(\boldsymbol{x}_t, t)$ and substituting $\mathrm{d}\boldsymbol{x}_t \mathrm{d}\boldsymbol{x}_t^\top = g^2(t) \mathrm{d}t \boldsymbol{I} + \mathcal{O}(\mathrm{d}t^{3/2})$ yields **It\^o's Formula** [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. C.1.3, Eq. (C.1.1), p. 416):
>
> $$
> \mathrm{d}\varphi(\boldsymbol{x}_t, t) = \left( \frac{\partial \varphi}{\partial t} + (\nabla_{\boldsymbol{x}}\varphi)^\top \boldsymbol{f}(\boldsymbol{x}_t, t) + \frac{1}{2}g^2(t) \mathrm{Tr}\!\left(\nabla_{\boldsymbol{x}}^2 \varphi\right) \right)\mathrm{d}t + g(t)(\nabla_{\boldsymbol{x}}\varphi)^\top \mathrm{d}\boldsymbol{w}_t.
> $$
>
> **Derivation of the Fokker-Planck Equation** [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. C.1.4, pp. 417–418): Take expectations under $\boldsymbol{x}_t \sim p_t$ for a compactly supported test function $\varphi(\boldsymbol{x})$ independent of $t$. Since $\mathbb{E}[(\nabla_{\boldsymbol{x}}\varphi)^\top \mathrm{d}\boldsymbol{w}_t] = 0$,
>
> $$
> \frac{\mathrm{d}}{\mathrm{d}t}\mathbb{E}_{p_t}[\varphi(\boldsymbol{x}_t)] = \int_{\mathbb{R}^D} \varphi(\boldsymbol{x}) \frac{\partial p_t(\boldsymbol{x})}{\partial t} \mathrm{d}\boldsymbol{x} = \int_{\mathbb{R}^D} \left( (\nabla_{\boldsymbol{x}}\varphi)^\top \boldsymbol{f}(\boldsymbol{x}, t) + \frac{1}{2}g^2(t) \Delta \varphi(\boldsymbol{x}) \right) p_t(\boldsymbol{x}) \mathrm{d}\boldsymbol{x},
> $$
>
> where $\Delta = \mathrm{Tr}(\nabla_{\boldsymbol{x}}^2)$ is the Laplacian. Applying the divergence theorem (integration by parts) with vanishing boundary terms gives
>
> $$
> \int_{\mathbb{R}^D} \varphi(\boldsymbol{x}) \left[ \frac{\partial p_t(\boldsymbol{x})}{\partial t} + \nabla_{\boldsymbol{x}} \cdot \bigl(\boldsymbol{f}(\boldsymbol{x}, t) p_t(\boldsymbol{x})\bigr) - \frac{1}{2}g^2(t) \Delta p_t(\boldsymbol{x}) \right] \mathrm{d}\boldsymbol{x} = 0.
> $$
>
> Since $\varphi$ is arbitrary, the integrand vanishes pointwise, yielding the **Fokker-Planck Equation**:
>
> $$
> \frac{\partial p_t(\boldsymbol{x})}{\partial t} = -\nabla_{\boldsymbol{x}} \cdot \bigl(\boldsymbol{f}(\boldsymbol{x}, t) p_t(\boldsymbol{x})\bigr) + \frac{1}{2}g^2(t) \Delta p_t(\boldsymbol{x}).
> $$

### Anderson's Reverse-Time SDE & the Probability Flow ODE

To construct a deterministic ODE or a reverse-time SDE that shares the exact marginal densities $\{p_t\}_{t \in [0, T]}$ of the forward SDE `(eq:sde:forward_sde)`, we rewrite the diffusive Laplacian in `(eq:sde:fokker_planck)` as a transport term using the log-derivative identity $\Delta p_t = \nabla_{\boldsymbol{x}} \cdot (p_t \nabla_{\boldsymbol{x}}\log p_t)$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. B.1.3, p. 407):

$$
\frac{\partial p_t(\boldsymbol{x})}{\partial t} = -\nabla_{\boldsymbol{x}} \cdot \Bigl( \underbrace{\bigl[\boldsymbol{f}(\boldsymbol{x}, t) - \tfrac{1}{2}g^2(t)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\bigr]}_{:= \tilde{\boldsymbol{v}}_t(\boldsymbol{x})} p_t(\boldsymbol{x}) \Bigr).
$$

Comparing `(eq:sde:fp_as_continuity)` with the deterministic continuity equation $\partial_t p_t + \nabla_{\boldsymbol{x}}\cdot(p_t \boldsymbol{v}_t) = 0$ immediately reveals two canonical generative dynamics [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Thm. 4.1.1, pp. 98–99) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.4.2–25.4.4, pp. 878–880):

$$
\begin{aligned}
**Probability Flow ODE (PF-ODE):**\quad & \mathrm{d}\tilde{\boldsymbol{x}}_t = \left[\boldsymbol{f}(\tilde{\boldsymbol{x}}_t, t) - \frac{1}{2}g^2(t)\nabla_{\boldsymbol{x}}\log p_t(\tilde{\boldsymbol{x}}_t)\right]\mathrm{d}t,  \\
  **Anderson Reverse-Time SDE:**\quad & \mathrm{d}\bar{\boldsymbol{x}}_t = \left[\boldsymbol{f}(\bar{\boldsymbol{x}}_t, t) - g^2(t)\nabla_{\boldsymbol{x}}\log p_t(\bar{\boldsymbol{x}}_t)\right]\mathrm{d}t + g(t) \mathrm{d}\bar{\boldsymbol{w}}_t,
\end{aligned}
$$

where $\bar{\boldsymbol{w}}_t$ is a standard Wiener process in reverse time ($t$ running from $T$ down to $0$).

> [!tip] Why $\frac{1}{2}g^2$ in PF-ODE vs. $g^2$ in Reverse SDE?
> Rewriting the reverse SDE `(eq:sde:reverse_sde)` as
> \[
>   \mathrm{d}\bar{\boldsymbol{x}}_t = \underbrace{\left[\boldsymbol{f}(\bar{\boldsymbol{x}}_t, t) - \tfrac{1}{2}g^2(t)\nabla_{\boldsymbol{x}}\log p_t(\bar{\boldsymbol{x}}_t)\right]\mathrm{d}t}_{\text{Deterministic PF-ODE}}  -  \underbrace{\tfrac{1}{2}g^2(t)\nabla_{\boldsymbol{x}}\log p_t(\bar{\boldsymbol{x}}_t) \mathrm{d}t + g(t) \mathrm{d}\bar{\boldsymbol{w}}_t}_{\text{Time-Varying Langevin Corrector}}
> \]
> shows that the extra $-\frac{1}{2}g^2 \nabla \log p_t$ drift exactly counterbalances the diffusive spreading of $g(t) \mathrm{d}\bar{\boldsymbol{w}}_t$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.4.5, Eq. (25.57), p. 881) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 4.1.3, p. 94). In practice, the continuous noise injection in the SDE acts as an error-correcting Langevin step that compensates for discretization errors and imperfect score estimates, whereas the PF-ODE is deterministic, invertible, uniquely identifiable across architectures, and enables exact log-likelihood evaluation via the instantaneous change-of-variables formula [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.4.4, p. 880):
>
> $$
> \log p_0(\boldsymbol{x}_0) = \log p_T(\boldsymbol{x}_T) + \int_0^T \nabla_{\boldsymbol{x}} \cdot \tilde{\boldsymbol{v}}_t(\boldsymbol{x}_t) \mathrm{d}t, \qquad \tilde{\boldsymbol{v}}_t(\boldsymbol{x}) = \boldsymbol{f}(\boldsymbol{x}, t) - \frac{1}{2}g^2(t)\boldsymbol{s}_\phi(\boldsymbol{x}, t).
> $$

### Girsanov's Theorem & Path-Measure Likelihood

While the Fokker-Planck equation guarantees *marginal* alignment $q_t(\boldsymbol{x}_t) = p_t(\boldsymbol{x}_t)$ at every fixed $t$, it does not constrain the joint trajectory distribution $p(\boldsymbol{x}_{0:T})$. Girsanov's theorem provides the Radon-Nikodym derivative (change-of-measure) between path measures $\mathbb{P}$ and $\mathbb{P}_\phi$ induced by two SDEs sharing the same diffusion $g(t)$ and differing only in their drifts $\boldsymbol{f}$ and $\tilde{\boldsymbol{f}}$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. C.2, pp. 422–425).

> [!important] Girsanov Path-Measure KL & Implicit Likelihood Equivalence
> Let $\boldsymbol{\delta}_t := \boldsymbol{f}(\boldsymbol{x}_t, t) - \tilde{\boldsymbol{f}}(\boldsymbol{x}_t, t)$ denote the drift discrepancy along a path $\boldsymbol{x}_{0:T}$. Under Novikov's regularity condition, Girsanov's theorem gives the path likelihood ratio [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. C.2, p. 423):
>
> $$
> \frac{\mathrm{d}\mathbb{P}_{\boldsymbol{f}}}{\mathrm{d}\mathbb{P}_{\tilde{\boldsymbol{f}}}}(\boldsymbol{x}_{0:T}) = \exp\!\left( \int_0^T \frac{\boldsymbol{\delta}_t^\top}{g(t)} \mathrm{d}\boldsymbol{w}_t - \frac{1}{2}\int_0^T \frac{\|\boldsymbol{\delta}_t\|_2^2}{g^2(t)} \mathrm{d}t \right).
> $$
>
> Taking $\mathbb{P}$ as the true reverse-time SDE (drift $\boldsymbol{f} - g^2\nabla_{\boldsymbol{x}}\log p_t$) and $\mathbb{P}_\phi$ as the learned reverse SDE (drift $\boldsymbol{f} - g^2\boldsymbol{s}_\phi$), we have $\boldsymbol{\delta}_t = g^2(t)(\boldsymbol{s}_\phi(\boldsymbol{x}_t, t) - \nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}_t))$. Since the It\^o integral has zero expectation under $\mathbb{P}$, the KL divergence between the full path measures simplifies to [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. C.2.1, Eq. (C.2.1), p. 425):
>
> $$
> D_{\mathrm{KL}}(\mathbb{P}  \|  \mathbb{P}_\phi) = \frac{1}{2} \mathbb{E}_{\mathbb{P}}\!\left[ \int_0^T g^2(t) \bigl\|\boldsymbol{s}_\phi(\boldsymbol{x}_t, t) - \nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}_t)\bigr\|_2^2 \mathrm{d}t \right] + \mathrm{Const.}
> $$
>
> By the data-processing inequality, $D_{\mathrm{KL}}(p_{\mathrm{data}}  \|  p_\phi(\boldsymbol{x}_0)) \le D_{\mathrm{KL}}(\mathbb{P}  \|  \mathbb{P}_\phi)$. Consequently, weighting score matching by $\lambda(t) = g^2(t)$ (the likelihood-weighting schedule) directly minimizes an upper bound on the negative data log-likelihood $-\mathbb{E}_{p_{\mathrm{data}}}[\log p_\phi(\boldsymbol{x}_0)]$.

### Flow Matching, Conditional Flow Matching & Rectified Flow

Flow Matching (FM) bypasses stochastic forward SDEs by learning a deterministic velocity field $\boldsymbol{v}_\phi(\boldsymbol{x}_t, t)$ directly from a tractable conditional probability path [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 5.2–5.3, pp. 136–150). Let $\boldsymbol{z}$ be a conditioning variable (e.g., $\boldsymbol{z} = \boldsymbol{x}_1$ or $(\boldsymbol{x}_0, \boldsymbol{x}_1) \sim \pi$) generating a conditional path $p_t(\boldsymbol{x} \mid \boldsymbol{z})$ and velocity $\boldsymbol{v}_t(\boldsymbol{x} \mid \boldsymbol{z})$ that obey $\partial_t p_t(\boldsymbol{x} \mid \boldsymbol{z}) + \nabla_{\boldsymbol{x}} \cdot (p_t(\boldsymbol{x} \mid \boldsymbol{z}) \boldsymbol{v}_t(\boldsymbol{x} \mid \boldsymbol{z})) = 0$.

> [!important] Conditional Flow Matching (CFM) Equivalence & Affine Rectified Flow
> Define the marginal density $p_t(\boldsymbol{x}) = \int p_t(\boldsymbol{x} \mid \boldsymbol{z}) \pi(\boldsymbol{z}) \mathrm{d}\boldsymbol{z}$ and the marginal velocity field
>
> $$
> \boldsymbol{v}_t(\boldsymbol{x}) := \mathbb{E}_{\boldsymbol{z} \sim p_t(\boldsymbol{z} \mid \boldsymbol{x})}\!\left[\boldsymbol{v}_t(\boldsymbol{x} \mid \boldsymbol{z})\right] = \int \boldsymbol{v}_t(\boldsymbol{x} \mid \boldsymbol{z}) \frac{p_t(\boldsymbol{x} \mid \boldsymbol{z}) \pi(\boldsymbol{z})}{p_t(\boldsymbol{x})} \mathrm{d}\boldsymbol{z}.
> $$
>
> Then $(p_t, \boldsymbol{v}_t)$ satisfies the marginal continuity equation $\partial_t p_t + \nabla_{\boldsymbol{x}}\cdot(p_t \boldsymbol{v}_t) = 0$. Moreover, the intractable Marginal FM objective $\mathcal{L}_{\mathrm{FM}}(\phi) = \mathbb{E}_{t, \boldsymbol{x}_t \sim p_t}[\|\boldsymbol{v}_\phi(\boldsymbol{x}_t, t) - \boldsymbol{v}_t(\boldsymbol{x}_t)\|_2^2]$ and the simulation-free **Conditional Flow Matching (CFM)** objective
>
> $$
> \mathcal{L}_{\mathrm{CFM}}(\phi) := \mathbb{E}_{t \sim \mathcal{U}[0,1],  \boldsymbol{z} \sim \pi,  \boldsymbol{x}_t \sim p_t(\cdot \mid \boldsymbol{z})}\!\left[ \bigl\|\boldsymbol{v}_\phi(\boldsymbol{x}_t, t) - \boldsymbol{v}_t(\boldsymbol{x}_t \mid \boldsymbol{z})\bigr\|_2^2 \right]
> $$
>
> have identical gradients: $\nabla_\phi \mathcal{L}_{\mathrm{FM}}(\phi) = \nabla_\phi \mathcal{L}_{\mathrm{CFM}}(\phi)$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 5.2.2, pp. 141–147).
>
> **Gaussian Bridge & Canonical Linear Interpolant.** For a Gaussian path $p_t(\boldsymbol{x} \mid \boldsymbol{z}) = \mathcal{N}(\boldsymbol{x};  \boldsymbol{\mu}_t(\boldsymbol{z}), \sigma_t^2(\boldsymbol{z})\boldsymbol{I})$ induced by $\Psi_{0 \to t}(\boldsymbol{x}) = \boldsymbol{\mu}_t + \sigma_t \frac{\boldsymbol{x}-\boldsymbol{\mu}_0}{\sigma_0}$, differentiating along trajectories gives the unique velocity [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Prop. 5.3.1, p. 150):
>
> $$
> \boldsymbol{v}_t(\boldsymbol{x} \mid \boldsymbol{z}) = \frac{\sigma_t'(\boldsymbol{z})}{\sigma_t(\boldsymbol{z})}\bigl(\boldsymbol{x} - \boldsymbol{\mu}_t(\boldsymbol{z})\bigr) + \boldsymbol{\mu}_t'(\boldsymbol{z}).
> $$
>
> Setting $\boldsymbol{z} = (\boldsymbol{x}_0, \boldsymbol{x}_1) \sim p_{\mathrm{src}} \times p_{\mathrm{tgt}}$ with the canonical linear interpolant $\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1$ ($\sigma_t \to 0$) reduces `(eq:sde:cfm_loss)` to the **Rectified Flow** objective [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 5.4, p. 159):
>
> $$
> \mathcal{L}_{\mathrm{RF}}(\phi) = \mathbb{E}_{t \sim \mathcal{U}[0,1],  (\boldsymbol{x}_0, \boldsymbol{x}_1) \sim \pi}\!\left[ \bigl\|\boldsymbol{v}_\phi(\boldsymbol{x}_t, t) - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\bigr\|_2^2 \right].
> $$

Iteratively replacing the independent coupling $\pi^{(0)} = p_{\mathrm{src}} \otimes p_{\mathrm{tgt}}$ with the deterministic flow coupling $\pi^{(k+1)}(\boldsymbol{x}_0, \boldsymbol{x}_1) = p_{\mathrm{src}}(\boldsymbol{x}_0)\delta(\boldsymbol{x}_1 - \Phi_{0 \to 1}^{(k)}(\boldsymbol{x}_0))$ (**Reflow**) preserves marginals, never increases convex transport cost $\mathbb{E}[c(\hat{\boldsymbol{z}}_1 - \boldsymbol{z}_0)] \le \mathbb{E}[c(\boldsymbol{x}_1 - \boldsymbol{x}_0)]$, and drives path curvature toward zero: $\min_{k \le K}\mathcal{S}(\boldsymbol{Z}^{(k)}) \le \frac{1}{K}\mathbb{E}[\|\boldsymbol{x}_1 - \boldsymbol{x}_0\|_2^2]$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 5.4.2–5.4.3, pp. 161–163).

## Algorithmic Implementation & Comparative Synthesis

> [!example] Algorithm: Simulation-Free CFM Training & Euler-Maruyama / PF-ODE Sampling
> **Input:** Dataset $p_{\mathrm{data}}$ (or coupling $\pi$), prior $p_{\mathrm{prior}} = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$, steps $N$, mode $\in \{\text{Reverse-SDE}, \text{PF-ODE}, \text{CFM}\}$.
> **Output:** Trained network ($\boldsymbol{s}_\phi$ or $\boldsymbol{v}_\phi$) and generated sample $\boldsymbol{x}_{\mathrm{gen}}$.
>  *// 1. Simulation-Free Training (Score Matching or CFM)*
> **while** not converged:
>
>   Sample $t \sim \mathcal{U}[0, 1]$, $\boldsymbol{x}_0 \sim p_{\mathrm{src}}$, $\boldsymbol{x}_1 \sim p_{\mathrm{tgt}}$
>   **if** mode is CFM:
>
>     $\boldsymbol{x}_t \leftarrow (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1$
>     Update $\phi$ with $\nabla_\phi \|\boldsymbol{v}_\phi(\boldsymbol{x}_t, t) - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\|_2^2$
>
> **else**:
>
>     Sample $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$;\quad $\boldsymbol{x}_t \leftarrow \alpha_t \boldsymbol{x}_0 + \sigma_t \boldsymbol{\epsilon}$
>     Update $\phi$ with $\nabla_\phi \lambda(t)\|\boldsymbol{s}_\phi(\boldsymbol{x}_t, t) + \boldsymbol{\epsilon}/\sigma_t\|_2^2$
>
>
>  *// 2. Generation ($N$ Euler / Euler-Maruyama Steps)*
> Draw $\boldsymbol{x} \sim p_{\mathrm{prior}}$;\quad $\Delta t \leftarrow 1/N$
> **for** $k = N, N-1, \dots, 1$:
>
>   $t \leftarrow k \Delta t$
>   **if** mode is CFM:
> $\boldsymbol{x} \leftarrow \boldsymbol{x} + \boldsymbol{v}_\phi(\boldsymbol{x}, 1 - t) \Delta t$
>   **else if** mode is PF-ODE:
> $\boldsymbol{x} \leftarrow \boldsymbol{x} - \bigl[\boldsymbol{f}(\boldsymbol{x}, t) - \frac{1}{2}g^2(t)\boldsymbol{s}_\phi(\boldsymbol{x}, t)\bigr]\Delta t$
>   **else**:
>
>     Sample $\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$
>     $\boldsymbol{x} \leftarrow \boldsymbol{x} - \bigl[\boldsymbol{f}(\boldsymbol{x}, t) - g^2(t)\boldsymbol{s}_\phi(\boldsymbol{x}, t)\bigr]\Delta t + g(t)\sqrt{\Delta t} \boldsymbol{z}$
>
>
> **return** $\boldsymbol{x}_{\mathrm{gen}} \leftarrow \boldsymbol{x}$

**Table:** Comparative synthesis of continuous-time generative differential equation paradigms.

| **Paradigm** | **Dynamics** | **Perturbation Kernel** | **Training Target** | **Sampling / Likelihood** |
| — | — | — | — | — |
| **VE-SDE** (SMLD) | $\mathrm{d}\boldsymbol{x}_t = \sqrt{\frac{\mathrm{d}\sigma^2}{\mathrm{d}t}} \mathrm{d}\boldsymbol{w}_t$ | $\mathcal{N}(\boldsymbol{x}_0, (\sigma_t^2 - \sigma_0^2)\boldsymbol{I})$ | $\boldsymbol{s}_\phi \approx -\boldsymbol{\epsilon}/\sqrt{\sigma_t^2-\sigma_0^2}$ | Stochastic SDE / Predictor-Corrector; ELBO via Girsanov. |
| **VP-SDE** (DDPM) | $\mathrm{d}\boldsymbol{x}_t = -\frac{1}{2}\beta_t\boldsymbol{x}_t \mathrm{d}t + \sqrt{\beta_t} \mathrm{d}\boldsymbol{w}_t$ | $\mathcal{N}(e^{-\frac{1}{2}B_t}\boldsymbol{x}_0, (1-e^{-B_t})\boldsymbol{I})$ | $\boldsymbol{s}_\phi \approx -\boldsymbol{\epsilon}/\sqrt{1-e^{-B_t}}$ | Stochastic SDE or semi-linear ODE; ELBO via Girsanov. |
| **PF-ODE** | $\mathrm{d}\boldsymbol{x}_t = [\boldsymbol{f} - \frac{1}{2}g^2 \boldsymbol{s}_\phi] \mathrm{d}t$ | Shares $p_t(\boldsymbol{x}_t)$ with forward SDE | Trained via DSM ($\boldsymbol{s}_\phi$) | Deterministic RK4/Heun; exact $\log p_0$ via trace. |
| **CFM / Rectified Flow** | $\mathrm{d}\boldsymbol{x}_t = \boldsymbol{v}_\phi(\boldsymbol{x}_t, t) \mathrm{d}t$ | $\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1$ (arbitrary $p_{\mathrm{src}}, p_{\mathrm{tgt}}$) | $\boldsymbol{v}_\phi \approx \boldsymbol{x}_1 - \boldsymbol{x}_0$ | Straightened ODE (few Euler steps); exact $\log p_1$ via trace. |

> [!info] Cross-Book Synthesis: SDEs, Semi-Linear ODEs, and Flow Matching
> [[Source - Tutorial on Diffusion Models for Imaging and Vision|diffusionmodels_stanleychan]] (Sec. 4.5, pp. 47–48) emphasizes that the VP-SDE Probability Flow ODE is *semi-linear* ($\dot{\boldsymbol{x}}_t = a(t)\boldsymbol{x}_t + \boldsymbol{b}(t)$), allowing the linear drift $a(t)\boldsymbol{x}_t$ to be integrated analytically via the Variation of Constants formula (the foundation of exponential integrators like DPM-Solver) while reserving numerical quadrature solely for the nonlinear score term. [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.4.4–25.4.5, pp. 880–881) highlights two complementary operational trade-offs: deterministic PF-ODEs yield *identifiable* latent codes that support semantic interpolation, whereas alternating deterministic ODE steps with small stochastic Langevin noise injections (*churn*) corrects integration drift. Finally, [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 5.3, pp. 148–150) contrasts the *Eulerian* (density-first) view of Score SDEs—where $p_t$ is fixed and infinitely many zero-flux vector fields $\boldsymbol{v}_t + \boldsymbol{w}_t$ ($\nabla\cdot(p_t\boldsymbol{w}_t)=0$) generate the same marginals—with the *Lagrangian* (flow-first) view of Flow Matching, which directly prescribes straight particle trajectories $\Psi_{0 \to t}$ to minimize numerical discretization error.

## Visual Diagnostics & Geometric Structure

> [!note] Source Figure
> *Figure:* Continuous-time forward diffusion SDE: green stochastic sample paths evolve from the multimodal data distribution $q(\boldsymbol{x}_0)$ toward the Gaussian prior $q(\boldsymbol{x}_T)$, while the background heatmap depicts the marginal density $p_t(\boldsymbol{x})$ governed by the Fokker-Planck equation [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.4.1, Fig. 25.6, p. 878). (`probabilisticml_kevinmurphy_advancedtopics_p912_fig1.png`)

> [!note] Source Figure
> *Figure:* Comparison of reverse-time generative dynamics sharing the same marginal densities $q(\boldsymbol{x}_t)$: (a) Anderson's reverse-time SDE injects Brownian noise balanced by $-g^2\nabla\log p_t$; (b) the deterministic Probability Flow ODE follows smooth, non-intersecting characteristic streamlines with $-\frac{1}{2}g^2\nabla\log p_t$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.4.3–25.4.4, Fig. 25.7, p. 879). (`probabilisticml_kevinmurphy_advancedtopics_p913_fig1.png`)

> [!example] Visual Diagram (LaTeX / PDF)
> *Unified mathematical bridge connecting discrete diffusion, Forward It\^o SDEs, Anderson's Reverse SDE, the Fokker-Planck Probability Flow ODE, and Conditional Flow Matching / Rectified Flow.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Geometric contrast between curved VP-SDE Probability Flow ODE trajectories (requiring many small solver steps) and straightened Rectified Flow / Optimal Transport interpolations $\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1$ (allowing large or single Euler steps).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!note] Source Figure
> *Figure:* Iterative Reflow straightening on a 2D six-mode transport task: (a) 1-rectified flow exhibits crossing/curved paths from independent pairings; (b)–(c) 2- and 3-rectified flows progressively straighten trajectories without increasing transport cost [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 5.4.2, Fig. 5.7, p. 163). (`theprinciple_diffusion_models_p168_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Omitting the It\^o Correction or Misplacing the $\frac{1}{2}$ Factor in PF-ODE
> Applying the classical chain rule to $\varphi(\boldsymbol{x}_t, t)$ omits the second-order quadratic variation term $\frac{1}{2}g^2(t)\mathrm{Tr}(\nabla_{\boldsymbol{x}}^2\varphi) \mathrm{d}t$ in `(eq:sde:ito_lemma)`, which breaks conservation of probability mass. Similarly, dropping $g(t) \mathrm{d}\bar{\boldsymbol{w}}_t$ from Anderson's reverse SDE `(eq:sde:reverse_sde)` without halving the score coefficient from $g^2(t)$ to $\frac{1}{2}g^2(t)$ yields an ODE whose marginals do *not* match $p_t(\boldsymbol{x})$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 4.1.4, p. 95).

> [!warning] Score Singularity at $t \to 0$ and VE-SDE Variance Explosion
> As $t \to 0$, $\sigma_t \to 0$ causes the target score $-\boldsymbol{\epsilon}/\sigma_t$ to diverge when $p_{\mathrm{data}}$ lies on a lower-dimensional manifold. Practical implementations truncate integration at $t = \varepsilon \in [10^{-5}, 10^{-3}]$ and train with noise-normalized $\boldsymbol{\epsilon}_\theta$ or $\boldsymbol{v}_\phi$ parameterizations. For VE-SDEs, $\sigma(T) \gg \sigma(0)$ must hold so $p_T(\boldsymbol{x}_T) \approx \mathcal{N}(\boldsymbol{0}, \sigma^2(T)\boldsymbol{I})$ is independent of $p_{\mathrm{data}}$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 4.3.1, p. 110).

> [!warning] Straight-Line Couplings Are Necessary but Not Sufficient for Optimal Transport
> Although $L_2$-optimal transport induces straight-line paths $\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + tT(\boldsymbol{x}_0)$, not every straight-line coupling is optimal. For example, mapping $p_{\mathrm{src}} = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ to $p_{\mathrm{tgt}} = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ via a non-identity rotation $\boldsymbol{x}_1 = \boldsymbol{A}\boldsymbol{x}_0$ ($\boldsymbol{A}^\top\boldsymbol{A} = \boldsymbol{I}$, $\det\boldsymbol{A} = 1$) produces straight trajectories $\boldsymbol{x}_t = ((1-t)\boldsymbol{I} + t\boldsymbol{A})\boldsymbol{x}_0$ with strictly positive transport cost, whereas the $c$-optimal coupling is the identity $\boldsymbol{x}_1 = \boldsymbol{x}_0$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 5.4.3, pp. 163–164).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (4):** [[Flow Matching & Rectified Flow]], [[Fokker-Planck Equation]], [[Probability Flow ODE]], [[Score-Based SDE]]
- **Prerequisites (Directed $\leftarrow$):** [[DDPM]], [[Fokker-Planck Equation]], [[Probability Flow ODE]], [[Score Matching (DSM & SSM)]], [[Score-Based SDE]], [[Wasserstein Distance]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Consistency Models]], [[Flow Matching & Rectified Flow]], [[Fokker-Planck Equation]], [[Probability Flow ODE]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Consistency Models]], [[DDPM]], [[Flow Matching & Rectified Flow]], [[Fokker-Planck Equation]], [[Normalizing Flows]], [[Probability Flow ODE]], [[Score Matching (DSM & SSM)]], [[Score-Based SDE]], [[Wasserstein Distance]]
- **Synthesized Sources:** [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]]
