---
type: monograph
domain: "D04"
parent_topic: "[[PT 04 - Diffusion Models, Score SDEs, Flow Matching & Fast Samplers]]"
sub_topic: "[[ST 04 - Guidance, Fast ODE Solvers & Consistency Distillation]]"
cheatsheet: "[[Cheatsheet - Classifier - Classifier-Free Guidance, DDIM - DPM-Solver ODEs & Consistency Models]]"
part_of: 
  - "[[ST 04 - Guidance, Fast ODE Solvers & Consistency Distillation]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
concepts: 
  - "[[Classifier-Free Guidance (CFG)]]"
  - "[[Consistency Models]]"
prerequisites: 
  - "[[DDPM]]"
  - "[[Probability Flow ODE]]"
enables: []
related: 
  - "[[CLIP & Vision-Language Alignment]]"
  - "[[DDPM]]"
  - "[[Flow Matching & Rectified Flow]]"
  - "[[Probability Flow ODE]]"
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
tags:
  - monograph
  - domain-04
---

# Classifier/Classifier-Free Guidance, DDIM/DPM-Solver ODEs & Consistency Models

> [!abstract] Monograph Navigation
> - **Domain:** [[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion|Deep Generative Modeling: VAEs, Flows, EBMs, GANs & Diffusion]]
> - **Parent / Sub-Topic:** [[PT 04 - Diffusion Models, Score SDEs, Flow Matching & Fast Samplers|Diffusion Models, Score SDEs, Flow Matching & Fast Samplers]] → [[ST 04 - Guidance, Fast ODE Solvers & Consistency Distillation|Guidance, Fast ODE Solvers & Consistency Distillation]]
> - **Visual Cheatsheet:** [[Cheatsheet - Classifier - Classifier-Free Guidance, DDIM - DPM-Solver ODEs & Consistency Models]]
> - **Description:** Covers Bayes-rule classifier guidance vs. Classifier-Free Guidance (CFG), non-Markovian DDIM, exponential integrators (DEIS, DPM-Solver/DPM-Solver++/v3), progressive distillation, Consistency Models (CD/CTM), Mean Flow, and RL/DPO diffusion alignment.

## Problem Formulation & Unified Notation

Standard Denoising Diffusion Probabilistic Models (DDPMs) suffer from two operational bottlenecks at inference time: (i) *uncontrolled generation*, where sampling from the unconditional marginal $p_0(\boldsymbol{x})$ cannot enforce semantic constraints $\boldsymbol{c}$ without external steering, and (ii) *prohibitive sampling latency*, where discretizing the reverse-time Markov chain requires $N \approx 500$–$1000$ neural function evaluations (NFEs). Both challenges are unified by the continuous-time Probability Flow Ordinary Differential Equation (PF-ODE) associated with a linear–Gaussian forward perturbation kernel $q(\boldsymbol{x}_t \mid \boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_t; \alpha_t \boldsymbol{x}_0, \sigma_t^2 \boldsymbol{I})$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Ch. 8–9, pp. 232, 268):

$$
\frac{\mathrm{d}\boldsymbol{x}_t}{\mathrm{d}t} = f(t)\boldsymbol{x}_t - \frac{1}{2} g^2(t) \nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t),
  \qquad f(t) = \frac{\alpha'_t}{\alpha_t}, \quad g^2(t) = 2\sigma_t\sigma'_t - 2\frac{\alpha'_t}{\alpha_t}\sigma_t^2,
$$

where $t \in [0, T]$ runs from clean data ($t=0$) to the Gaussian prior ($t=T$), and the marginal densities $p_t(\boldsymbol{x}_t)$ of `(eq:dsolver:pf_ode)` match those of the reverse-time SDE [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 20.3.4, p. 609). Controllable generation replaces the unconditional score $\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$ in `(eq:dsolver:pf_ode)` with a guided score $\boldsymbol{s}(\boldsymbol{x}_t, t, \boldsymbol{c}; \omega)$, while fast sampling replaces naive Euler steps with exponential integrators (DDIM, DEIS, DPM-Solver++) or learned flow maps (Progressive Distillation, Consistency Models, CTM, and Mean Flow).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Source texts adopt slightly different conventions for forward noise schedules, guidance weights, and latent state symbols. Table `tab:dsolver:notation` establishes a unified notation used throughout this monograph.
>
> 
> {
>
> | **Concept** | **Unified** | **Diffusion Princ.** | **Murphy Adv.** | **Bishop** |
> | — | — | — | — | — |
> | Noisy state | $\boldsymbol{x}_t$ | $\boldsymbol{x}_t$ (Ch. 8–11) | $\boldsymbol{x}_t$ (Sec. 25.5) | $\boldsymbol{z}$ |
> | Forward kernel | $\alpha_t \boldsymbol{x}_0 + \sigma_t \boldsymbol{\epsilon}$ | $\alpha_t \boldsymbol{x}_0 + \sigma_t \boldsymbol{\epsilon}$ | $\sqrt{\bar{\alpha}_t}\boldsymbol{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ | $q(\boldsymbol{z}_t \mid \boldsymbol{x})$ |
> | Guidance scale | $\omega \ge 0$ | $\omega$ (Sec. 8.2–8.3) | $w = \omega - 1$ (Sec. 25.6.3) | $\lambda$ |
> | Half-log SNR | $\lambda_t = \log(\alpha_t/\sigma_t)$ | $\lambda_t$ (Eq. 9.4.2) | — | — |
> | Flow / consistency | $\Psi_{s\to t}(\boldsymbol{x}_s)$ | $\Psi_{s\to t}$, $f_\theta(\boldsymbol{x}_s, s)$ | $f(\boldsymbol{z}_s; \theta)$ (Fig. 25.12) | — |
> }

## Guidance, Controllable Generation & Preference Alignment

### Classifier Guidance vs. Classifier-Free Guidance (CFG)

Given a conditioning variable $\boldsymbol{c}$ (class label, text prompt, or reference signal), the goal is to sample from $p_0(\boldsymbol{x}\mid\boldsymbol{c})$ by steering the reverse dynamics with the conditional score $\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t\mid\boldsymbol{c})$.

> [!important] Bayes Decomposition and Classifier Guidance
> Applying Bayes' theorem to $p_t(\boldsymbol{x}_t \mid \boldsymbol{c}) = p_t(\boldsymbol{x}_t) p_t(\boldsymbol{c}\mid\boldsymbol{x}_t) / p(\boldsymbol{c})$ and noting that $\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{c}) = \boldsymbol{0}$ yields the exact conditional score decomposition [[Source - Foundations|deep_learning_foundations_concepts]] (Eq. 20.58, p. 610):
>
> $$
> \nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t \mid \boldsymbol{c}) = \nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t) + \nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{c} \mid \boldsymbol{x}_t).
>
> $$
>
> To trade off sample diversity against conditional fidelity, Classifier Guidance (CG) introduces a tempered tilted density $p_t(\boldsymbol{x}_t \mid \boldsymbol{c}, \omega) \propto p_t(\boldsymbol{x}_t)  p_t(\boldsymbol{c}\mid\boldsymbol{x}_t)^{\omega}$ with guidance scale $\omega \ge 0$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Eq. 8.2.2–8.2.3, p. 237):
>
> $$
> \boldsymbol{s}_{\mathrm{CG}}(\boldsymbol{x}_t, t, \boldsymbol{c}; \omega)
>   := \boldsymbol{s}_\phi(\boldsymbol{x}_t, t) + \omega \nabla_{\boldsymbol{x}_t} \log p_\psi(\boldsymbol{c} \mid \boldsymbol{x}_t, t),
>
> $$
>
> where $p_\psi(\boldsymbol{c}\mid\boldsymbol{x}_t, t)$ is a separately trained time-dependent classifier on noisy inputs $\boldsymbol{x}_t$. Setting $\omega = 0$ recovers unconditional generation, $\omega = 1$ yields the true conditional score, and $\omega > 1$ sharpens mode concentration at the expense of diversity [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.6.2, p. 920).

Training a noise-robust classifier $p_\psi(\boldsymbol{c}\mid\boldsymbol{x}_t, t)$ across all $t \in [0, T]$ is expensive and brittle for open-vocabulary conditioning. Classifier-Free Guidance (CFG) eliminates the external classifier by inverting `(eq:dsolver:bayes_score)`.

> [!important] Classifier-Free Guidance via Implicit Classifier
> From `(eq:dsolver:bayes_score)`, the implicit classifier gradient equals the difference between conditional and unconditional scores:
>
> $$
> \nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{c} \mid \boldsymbol{x}_t)
>   = \nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t \mid \boldsymbol{c}) - \nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t).
>
> $$
>
> Substituting this identity into `(eq:dsolver:cg_score)` yields the Classifier-Free Guidance score [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Eq. 8.3.3, p. 242):
>
> $$
> \boldsymbol{s}_{\phi}^{\mathrm{CFG}}(\boldsymbol{x}_t, t, \boldsymbol{c}; \omega)
>   := \omega \boldsymbol{s}_\phi(\boldsymbol{x}_t, t, \boldsymbol{c}) + (1 - \omega) \boldsymbol{s}_\phi(\boldsymbol{x}_t, t, \emptyset),
>
> $$
>
> where a single neural network $\boldsymbol{s}_\phi$ is trained jointly on paired $(\boldsymbol{x}_0, \boldsymbol{c})$ and unconditional $(\boldsymbol{x}_0, \emptyset)$ by randomly dropping $\boldsymbol{c} \gets \emptyset$ with probability $10\%$–$20\%$ [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 20.4.2, p. 611). Note that [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.6.3, p. 920) writes `(eq:dsolver:cfg_score)` as $(1+w)\boldsymbol{s}_\phi(\boldsymbol{x}_t, t, \boldsymbol{c}) - w\boldsymbol{s}_\phi(\boldsymbol{x}_t, t, \emptyset)$ with $w = \omega - 1$.

> [!tip] Why CFG Outperforms Classifier Guidance
> As highlighted by [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 20.4.2, p. 611), a discriminative classifier $p_\psi(\boldsymbol{c}\mid\boldsymbol{x}_t)$ can ignore most coordinates of $\boldsymbol{x}_t$ as long as it extracts features sufficient to predict $\boldsymbol{c}$, causing adversarial or off-manifold gradients under large $\omega$. By contrast, CFG relies on the conditional generative density $p_t(\boldsymbol{x}_t\mid\boldsymbol{c})$, which must assign high likelihood to all structural aspects of $\boldsymbol{x}_t$. Figure `fig:dsolver:glide_cfg` illustrates this dramatic semantic sharpening on GLIDE.

> [!note] Source Figure
> *Figure:* Illustration of Classifier-Free Guidance on GLIDE for the prompt "A stained glass window of a panda eating bamboo" [[Source - Foundations|deep_learning_foundations_concepts]] (Fig. 20.7, p. 612). Left: unguided conditional model ($\omega=1$, i.e. $\lambda=0$ in Bishop's notation). Right: guided sampling ($\lambda=3$), demonstrating sharper semantic alignment with reduced sample diversity. (`deep_learning_foundations_concepts_p612_fig1.png`)

### Training-Free Guidance & Diffusion-DPO Alignment

When an auxiliary differentiable loss $\ell(\cdot, \boldsymbol{c}) : \mathbb{R}^D \to \mathbb{R}_{\ge 0}$ (e.g., CLIP similarity or an inverse-problem measurement residual) is given at test time, training-free guidance injects corrections into the DDIM step either in *data space* ($\hat{\boldsymbol{x}}_0(\boldsymbol{x}_t) + \eta^{\mathrm{data}}_t G_0$, with $G_0 = -\nabla_{\boldsymbol{x}_0}\ell(\boldsymbol{x}_0, \boldsymbol{c})$) or in *noise space* via a surrogate likelihood $\tilde{p}_t(\boldsymbol{c}\mid\boldsymbol{x}_t) \propto \exp(-\eta \ell(\hat{\boldsymbol{x}}_0(\boldsymbol{x}_t), \boldsymbol{c}))$, requiring backpropagation through the denoiser Jacobian $\nabla_{\boldsymbol{x}_t}\hat{\boldsymbol{x}}_0(\boldsymbol{x}_t)^{\top}$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Ch. 8.4, pp. 243–244).

For human-preference alignment without online RL rollouts or a separate reward model, Diffusion-DPO extends Direct Preference Optimization (DPO) to diffusion models. Inverting the closed-form KL-regularized RLHF optimum $\pi^*(\boldsymbol{x}\mid\boldsymbol{c}) = Z(\boldsymbol{c})^{-1}\pi_{\mathrm{ref}}(\boldsymbol{x}\mid\boldsymbol{c})\exp(r(\boldsymbol{c},\boldsymbol{x})/\beta)$ defines the implicit reward [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Ch. 8.5, pp. 248–252):

$$
r_\pi(\boldsymbol{c}, \boldsymbol{x}) = \beta\log\frac{\pi(\boldsymbol{x}\mid\boldsymbol{c})}{\pi_{\mathrm{ref}}(\boldsymbol{x}\mid\boldsymbol{c})} + \beta\log Z(\boldsymbol{c}).
$$

Substituting $r_\pi$ into the Bradley–Terry logistic loss on winner–loser pairs $(\boldsymbol{x}^w, \boldsymbol{x}^l, \boldsymbol{c})$ cancels the intractable partition function $Z(\boldsymbol{c})$ exactly.

## Fast ODE Solvers: DDIM, Exponential Integrators & DPM-Solver++

### DDIM as Exponential Euler on the Semilinear PF-ODE

Writing the PF-ODE `(eq:dsolver:pf_ode)` in $\boldsymbol{\epsilon}$-prediction form ($\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) \approx -\boldsymbol{\epsilon}_\phi(\boldsymbol{x}_t, t)/\sigma_t$) reveals a *semilinear* structure comprising a stiff linear drift $f(t)\boldsymbol{x}_t$ and a nonlinear residual:

$$
\frac{\mathrm{d}\boldsymbol{x}_t}{\mathrm{d}t} = \frac{\alpha'_t}{\alpha_t}\boldsymbol{x}_t + \frac{g^2(t)}{2\sigma_t}\boldsymbol{\epsilon}_\phi(\boldsymbol{x}_t, t).
$$

Integrating `(eq:dsolver:semilinear_ode)` from $s$ to $t < s$ using the integrating factor $E(s \to t) = \exp\!\left(\int_s^t f(u) du\right) = \alpha_t/\alpha_s$ eliminates discretization error on the linear term.

> [!important] DDIM = Exponential Euler (Proposition 9.2.1)
> Fixing the integrand $\boldsymbol{\epsilon}_\phi(\boldsymbol{x}_\tau, \tau) \approx \boldsymbol{\epsilon}_\phi(\tilde{\boldsymbol{x}}_s, s)$ on $\tau \in [t, s]$ in the variation-of-constants formula gives [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Prop. 9.2.1, pp. 268–270):
>
> $$
> \begin{aligned}
> \tilde{\boldsymbol{x}}_t
>   &= \frac{\alpha_t}{\alpha_s}\tilde{\boldsymbol{x}}_s + \left(\int_s^t \frac{\alpha_t}{\alpha_\tau} \frac{g^2(\tau)}{2\sigma_\tau} d\tau\right)\boldsymbol{\epsilon}_\phi(\tilde{\boldsymbol{x}}_s, s)  \\
>   &= \frac{\alpha_t}{\alpha_s}\tilde{\boldsymbol{x}}_s - \alpha_t\!\left(\frac{\sigma_s}{\alpha_s} - \frac{\sigma_t}{\alpha_t}\right)\boldsymbol{\epsilon}_\phi(\tilde{\boldsymbol{x}}_s, s)
>   = \alpha_t \hat{\boldsymbol{x}}_0(\tilde{\boldsymbol{x}}_s, s) + \sigma_t \boldsymbol{\epsilon}_\phi(\tilde{\boldsymbol{x}}_s, s),
> \end{aligned}
>
> $$
>
> since $\frac{g^2(\tau)}{2\sigma_\tau \alpha_\tau} = \frac{\mathrm{d}}{\mathrm{d}\tau}\!\left(\frac{\sigma_\tau}{\alpha_\tau}\right)$. Equivalently, [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.5.1, pp. 916–917) derives `(eq:dsolver:ddim)` from a non-Markovian forward process $q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0)$ with variance $\tilde{\sigma}_t^2 = 0$, preserving training marginals while allowing arbitrary step skipping. Crucially, under $\boldsymbol{v}$-prediction, plain Euler equals DDIM; under $\boldsymbol{\epsilon}$-, $\boldsymbol{x}$-, or $\boldsymbol{s}$-prediction, *exponential Euler* equals DDIM $\ne$ plain Euler [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Obs. 9.2.1, p. 272).

### DPM-Solver, Log-SNR Reparameterization & DPM-Solver++

Because DDIM is first-order (global error $O(h)$), quality degrades below $\sim 25$ steps. DPM-Solver raises the convergence order by changing variables from physical time $t$ to the strictly decreasing half-log signal-to-noise ratio $\lambda_t := \log(\alpha_t/\sigma_t)$ with inverse $t_\lambda(\cdot)$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 9.4.1, pp. 287–289).

> [!important] DPM-Solver: Exponentially Weighted Integral in Log-SNR
> Under $\hat{\boldsymbol{x}}_\lambda := \boldsymbol{x}_{t_\lambda(\lambda)}$ and $\hat{\boldsymbol{\epsilon}}_\phi(\hat{\boldsymbol{x}}_\lambda, \lambda) := \boldsymbol{\epsilon}_\phi(\boldsymbol{x}_{t_\lambda(\lambda)}, t_\lambda(\lambda))$, the exact PF-ODE solution $\tilde{\Psi}_{s\to t}(\boldsymbol{x}_s)$ simplifies to [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Prop. 9.4.1, p. 288):
>
> $$
> \tilde{\Psi}_{s\to t}(\boldsymbol{x}_s) = \frac{\alpha_t}{\alpha_s}\boldsymbol{x}_s - \alpha_t \int_{\lambda_s}^{\lambda_t} e^{-\lambda} \hat{\boldsymbol{\epsilon}}_\phi(\hat{\boldsymbol{x}}_\lambda, \lambda) d\lambda.
>
> $$
>
> Expanding $\hat{\boldsymbol{\epsilon}}_\phi$ in an $(n-1)$-th order Taylor series around $\lambda_s$ with step $h = \lambda_t - \lambda_s > 0$ yields DPM-Solver-$n$ with local error $O(h^{n+1})$ and global error $O(h^n)$:
>
> $$
> \tilde{\boldsymbol{x}}_t = \frac{\alpha_t}{\alpha_s}\tilde{\boldsymbol{x}}_s - \alpha_t \sum_{k=0}^{n-1} \hat{\boldsymbol{\epsilon}}_\phi^{(k)}(\hat{\boldsymbol{x}}_{\lambda_s}, \lambda_s)  C_k + O(h^{n+1}),
>   \qquad C_k := \int_{\lambda_s}^{\lambda_t} e^{-\lambda}\frac{(\lambda - \lambda_s)^k}{k!} d\lambda.
>
> $$
>
> For $n=1$, $C_0 = e^{-\lambda_s}(1 - e^{-h}) = \frac{\sigma_s}{\alpha_s} - \frac{\sigma_t}{\alpha_t}$, recovering DDIM `(eq:dsolver:ddim)` [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Eq. 9.4.8, p. 291).

Under large CFG scales ($\omega \gg 1$), $\boldsymbol{\epsilon}$-prediction and its higher-order $\lambda$-derivatives amplify, destabilizing DPM-Solver-$n$. DPM-Solver++ resolves this by switching to $\boldsymbol{x}$-prediction ($\hat{\boldsymbol{x}}_\phi$) with dynamic thresholding [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 9.5.2, p. 300):

$$
\tilde{\Psi}_{s\to t}(\boldsymbol{x}_s) = \frac{\sigma_t}{\sigma_s}\boldsymbol{x}_s + \sigma_t \int_{\lambda_s}^{\lambda_t} e^{\lambda} \hat{\boldsymbol{x}}_\phi(\hat{\boldsymbol{x}}_\lambda, \lambda) d\lambda.
$$

Whereas DEIS fits a multistep Lagrange polynomial over past anchors, DPM-Solver++ provides both a second-order single-step Runge–Kutta midpoint variant (2S, Algorithm `alg:dsolver:dpmpp2s`) and a second-order multistep Adams–Bashforth variant (2M) using Newton backward divided differences at 1 NFE per step [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 9.6, pp. 306–308).

> [!example] Algorithm: DPM-Solver++(2S) Midpoint Sampling [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Alg. 6, p. 302)
> **Input:** Initial noise $\boldsymbol{x}_T \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$, schedule $\{t_i\}_{i=0}^M$ with $t_0=T, t_M=0$, data predictor $\hat{\boldsymbol{x}}_\phi$
>   **Output:** Generated clean sample $\tilde{\boldsymbol{x}}_{t_M}$
>   Initialize $\tilde{\boldsymbol{x}}_{t_0} \gets \boldsymbol{x}_T$, compute $\lambda_{t_i} \gets \log(\alpha_{t_i}/\sigma_{t_i})$ for $i=0,\dots,M$
>   **for** $i \gets 1$ to $M$:
>
>     $h_i \gets \lambda_{t_i} - \lambda_{t_{i-1}}$; \quad $s_{\mathrm{mid}} \gets t_\lambda\!\left(\tfrac{1}{2}(\lambda_{t_{i-1}} + \lambda_{t_i})\right)$
>     $\hat{\boldsymbol{x}}_0 \gets \hat{\boldsymbol{x}}_\phi(\tilde{\boldsymbol{x}}_{t_{i-1}}, t_{i-1})$  *// Optionally apply dynamic thresholding*
>     $\boldsymbol{u}_i \gets \frac{\sigma_{s_{\mathrm{mid}}}}{\sigma_{t_{i-1}}}\tilde{\boldsymbol{x}}_{t_{i-1}} - \alpha_{s_{\mathrm{mid}}}\!\left(e^{-h_i/2} - 1\right)\hat{\boldsymbol{x}}_0$
>     $\tilde{\boldsymbol{x}}_{t_i} \gets \frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}\tilde{\boldsymbol{x}}_{t_{i-1}} - \alpha_{t_i}\!\left(e^{-h_i} - 1\right)\hat{\boldsymbol{x}}_\phi(\boldsymbol{u}_i, s_{\mathrm{mid}})$

**Table:** Comparison of PF-ODE / SDE samplers and their classical numerical analogues [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Table 9.1, p. 306).

| **Solver** | **Numerical Analogue** | **Param.** | **Order** | **Typical NFE** | **CFG Stability** |
| — | — | — | — | — | — |
| DDPM | Euler–Maruyama (SDE) | $\boldsymbol{\epsilon}$ | $O(h^{1/2})$ | $500$–$1000$ | High (small steps) |
| DDIM | Exp. Euler (ODE) | $\boldsymbol{\epsilon}, \boldsymbol{x}, \boldsymbol{v}$ | $O(h)$ | $25$–$100$ | High |
| DEIS | Exp. Adams–Bashforth | $\boldsymbol{\epsilon}$ | $O(h^r)$ | $10$–$20$ | Moderate |
| DPM-Solver-$n$ | Exp. Runge–Kutta ($\lambda$) | $\boldsymbol{\epsilon}$ | $O(h^n)$ | $10$–$20$ | Degrades at large $\omega$ |
| DPM-Solver++ (2S/2M) | Exp. RK / AB ($\lambda$) | $\boldsymbol{x}$ + thresh. | $O(h^2)$ | $10$–$20$ | High at large $\omega$ |

## Distillation, Consistency Models & General Flow Maps

Training-free ODE solvers still require $10$–$25$ NFEs because large steps induce curved trajectory errors and multimodal posteriors [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.5.2, p. 917). Reaching $1$–$4$ NFEs requires learning a direct state-to-state or state-to-origin flow map.

### Progressive Distillation (PD)

Progressive Distillation iteratively halves the required step count by training a student $\boldsymbol{x}_\theta(\boldsymbol{x}_s, s)$ to match two consecutive DDIM steps of a frozen teacher $\boldsymbol{x}_\phi$ ($s \to u \to t$) in a single step $s \to t$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 10.3, pp. 339–341).\\
[[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.5.3, pp. 917–918) details how each round halves the sampling budget.

> [!important] Two-Steps-in-One Target in Progressive Distillation (Lemma 10.3.1)
> Let $\tilde{\boldsymbol{x}}_u = \mathrm{Solver}_{s\to u}(\boldsymbol{x}_s; \boldsymbol{x}_\phi)$ and $\tilde{\boldsymbol{x}}_t = \mathrm{Solver}_{u\to t}(\tilde{\boldsymbol{x}}_u; \boldsymbol{x}_\phi)$ denote two teacher DDIM steps. Equating $\tilde{\boldsymbol{x}}_t$ to a single student DDIM step $\frac{\sigma_t}{\sigma_s}\boldsymbol{x}_s + \alpha_s\!\left(\frac{\alpha_t}{\alpha_s} - \frac{\sigma_t}{\sigma_s}\right)\tilde{\boldsymbol{x}}$ yields the closed-form pseudo-clean target [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Lemma 10.3.1, p. 341):
>
> $$
> \tilde{\boldsymbol{x}} = \frac{\sigma_s \tilde{\boldsymbol{x}}_t - \sigma_t \boldsymbol{x}_s}{\alpha_t \sigma_s - \alpha_s \sigma_t}.
>
> $$
>
> Minimizing $\mathbb{E}_{s, \boldsymbol{x}_s}[w(\lambda_s)\lVert \boldsymbol{x}_\theta(\boldsymbol{x}_s, s) - \tilde{\boldsymbol{x}} \rVert_2^2]$ halves $N \to N/2 \to N/4 \to \dots$, and extends to CFG by first distilling $\omega \sim \mathcal{U}[\omega_{\min}, \omega_{\max}]$ into an $\omega$-conditioned teacher [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 10.3.4, p. 344).

> [!note] Source Figure
> *Figure:* Progressive Distillation [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 25.12, p. 918): each distillation round compresses two deterministic teacher DDIM steps into a single student step, halving the total NFE budget until $1$–$4$ step generation is achieved. (`probabilisticml_kevinmurphy_advancedtopics_p918_fig1.png`)

### Consistency Models (CM), CTM & Mean Flow

Consistency Models (CMs) and their generalizations exploit the fundamental *semigroup property* of deterministic ODE flow maps $\Psi_{s\to t}$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Eq. 11.2.1, p. 353):

$$
\Psi_{u\to t} \circ \Psi_{s\to u} = \Psi_{s\to t}, \qquad \Psi_{s\to s} = \mathrm{Id}, \qquad \forall  s, u, t \in [0, T].
$$

> [!note] Source Figure
> *Figure:* Flow map semigroup property along a PF-ODE oracle trajectory [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Fig. 11.2, p. 353): transitioning from $s \to u$ and then $u \to t$ equals the direct map $\Psi_{s\to t}$. (`theprinciple_diffusion_models_p353_fig1.png`)

> [!important] Consistency Function, Discrete Equivalence & Continuous Limit
> Fixing $t=0$ defines the consistency function $f^*(\boldsymbol{x}_s, s) := \Psi_{s\to 0}(\boldsymbol{x}_s)$, which satisfies self-consistency $f^*(\boldsymbol{x}(s), s) = f^*(\boldsymbol{x}(u), u)$ and boundary condition $f^*(\boldsymbol{x}_0, 0) = \boldsymbol{x}_0$. On a grid $0 = s_1 < \dots < s_N = T$ with stop-gradient target $\theta^-$, *Consistency Distillation (CD)* regresses $f_\theta(\boldsymbol{x}_{s_{i+1}}, s_{i+1})$ toward $f_{\theta^-}(\tilde{\boldsymbol{x}}^\phi_{s_i}, s_i)$ using a one-step teacher ODE update, whereas *Consistency Training (CT)* uses a one-point unbiased score estimator $\tilde{\boldsymbol{x}}_{s_i} = \alpha_{s_i}\boldsymbol{x}_0 + \sigma_{s_i}\boldsymbol{\epsilon}$ sharing the same $(\boldsymbol{x}_0, \boldsymbol{\epsilon})$ pair. By Taylor expansion around $\mathbb{E}[\boldsymbol{x}_{s'} \mid \boldsymbol{x}_s] = \boldsymbol{x}_{s'}^{\mathrm{DDIM}}$, [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Thm. 11.2.1, p. 358) proves:
>
> $$
> \mathcal{L}_{\mathrm{CM}}(\theta, \theta^-) = \mathcal{L}_{\mathrm{CT}}(\theta, \theta^-) + O(\Delta s^2).
>
> $$
>
> Taking $\Delta s \to 0$ yields the continuous-time objective [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Prop. 11.3.1, p. 362):
>
> $$
> \mathcal{L}_{\mathrm{CM}}^\infty(\theta, \theta^-) = \mathbb{E}_{s, \boldsymbol{x}_0, \boldsymbol{\epsilon}}\!\left[\omega(s)  f_\theta(\boldsymbol{x}_s, s)^{\top} \frac{\mathrm{d}}{\mathrm{d}s}f_{\theta^-}(\boldsymbol{x}_s, s)\right],
>   \quad \frac{\mathrm{d}}{\mathrm{d}s}f_{\theta^-} = \partial_s f_{\theta^-} + \partial_{\boldsymbol{x}} f_{\theta^-} \boldsymbol{v}^*,
>
> $$
>
> stabilized in sCM via the trigonometric schedule $\boldsymbol{x}_s = \cos(s)\boldsymbol{x}_0 + \sin(s)\boldsymbol{z}$ ($\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, \sigma_d^2\boldsymbol{I})$) and $f_\theta(\boldsymbol{x}, s) = \cos(s)\boldsymbol{x} - \sin(s)\sigma_d F_\theta(\boldsymbol{x}/\sigma_d, c_{\mathrm{noise}}(s))$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Eq. 11.3.8, p. 365).

> [!example] Algorithm: Consistency Training (CT) & Multi-Step Consistency Sampling [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Ch. 11.2)
> **Input:** Dataset $p_{\mathrm{data}}$, model $f_\theta$, EMA target $\theta^-$, discretization $\{s_i\}_{i=1}^N$, sampling steps $\{\tau_1, \dots, \tau_K\}$
>   **Output:** Trained consistency model $f_\theta$ and $K$-step sample $\hat{\boldsymbol{x}}_0$
>    *// Training Step*
>   Sample $\boldsymbol{x}_0 \sim p_{\mathrm{data}}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$, and index $i \sim \mathcal{U}\{1, \dots, N-1\}$
>   $\boldsymbol{x}_{s_{i+1}} \gets \alpha_{s_{i+1}}\boldsymbol{x}_0 + \sigma_{s_{i+1}}\boldsymbol{\epsilon}$; \quad $\boldsymbol{x}_{s_i} \gets \alpha_{s_i}\boldsymbol{x}_0 + \sigma_{s_i}\boldsymbol{\epsilon}$
>   Compute $\mathcal{L} \gets \omega(s_i)  d\!\left(f_\theta(\boldsymbol{x}_{s_{i+1}}, s_{i+1}),  f_{\theta^-}(\boldsymbol{x}_{s_i}, s_i)\right)$ and update $\theta \gets \theta - \eta\nabla_\theta\mathcal{L}$
>   Update EMA target $\theta^- \gets \mu\theta^- + (1-\mu)\theta$
>    *// Multi-Step Inference ($K \ge 1$)*
>   Sample $\hat{\boldsymbol{x}}_T \sim \mathcal{N}(\boldsymbol{0}, \sigma_T^2\boldsymbol{I})$ and set $\hat{\boldsymbol{x}}_0 \gets f_\theta(\hat{\boldsymbol{x}}_T, T)$
>   **for** $k \gets 1$ to $K-1$:
>
>     Sample $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$; \quad $\hat{\boldsymbol{x}}_{\tau_k} \gets \alpha_{\tau_k}\hat{\boldsymbol{x}}_0 + \sigma_{\tau_k}\boldsymbol{\epsilon}$; \quad $\hat{\boldsymbol{x}}_0 \gets f_\theta(\hat{\boldsymbol{x}}_{\tau_k}, \tau_k)$

> [!info] Cross-Book Synthesis: General Flow Maps (CTM, Mean Flow) & Discrete Diffusion
> While CMs learn only the origin map $\Psi_{s\to 0}$, **Consistency Trajectory Models (CTM)** and **Mean Flow (MF)** parameterize the general two-time flow map $\Psi_{s\to t}(\boldsymbol{x}_s)$ for arbitrary $s \ge t \ge 0$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 11.4–11.5, pp. 370–383):
>
> $$
> \Psi_{s\to t}(\boldsymbol{x}_s)
>   \approx \underbrace{\frac{t}{s}\boldsymbol{x}_s + \frac{s-t}{s} g_\theta(\boldsymbol{x}_s, s, t)}_{\text{CTM}}
>   = \underbrace{\boldsymbol{x}_s + (t-s) h_\theta(\boldsymbol{x}_s, s, t)}_{\text{Mean Flow}},
>
> $$
>
> where $g_\theta(\boldsymbol{x}_s, s, t) = \boldsymbol{x}_s - s  h_\theta(\boldsymbol{x}_s, s, t)$ and both losses are equivalent up to a $1/s^2$ time weighting. Setting $s=t$ recovers the instantaneous velocity $\boldsymbol{v}^*(\boldsymbol{x}_s, s)$, unifying diffusion/flow-matching training with $1$-step generation. For categorical data (text, segmentation), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 25.7, pp. 921–924) extends diffusion to discrete state spaces (D3PM) via row-stochastic transition matrices $\boldsymbol{Q}_t$ with tractable marginals $q(\boldsymbol{x}_t\mid\boldsymbol{x}_0) = \mathrm{Cat}(\boldsymbol{x}_t \mid \boldsymbol{x}_0 \bar{\boldsymbol{Q}}_t)$.

## Visual Diagnostics & Acceleration Trade-Offs

Figure `fig:dsolver:acceleration_taxonomy` synthesizes the four acceleration regimes alongside CFG score extrapolation, and Figure `fig:dsolver:nfe_vs_quality` quantifies the resulting NFE–FID and CFG fidelity–diversity trade-offs.

> [!example] Visual Diagram (LaTeX / PDF)
> *Taxonomy of diffusion acceleration paradigms (from $1000$-NFE Markovian SDEs to $1$-NFE semigroup flow maps) paired with Classifier-Free Guidance score extrapolation.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Illustrative performance trade-offs synthesized from [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Ch. 8–11). Left: FID vs. NFE comparing SDE, exponential-integrator ODE, and distilled/consistency samplers. Right: Precision–Recall Pareto frontier as CFG guidance scale $\omega$ increases.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Plain Euler vs. Exponential Euler Instability in $\boldsymbol{\epsilon}$-Prediction
> Applying a naive Euler discretization directly to the $\boldsymbol{\epsilon}$-parameterized PF-ODE `(eq:dsolver:semilinear_ode)` approximates the linear drift $f(t)\boldsymbol{x}_t$ numerically rather than analytically, causing severe trajectory drift at large step sizes. **Remedy:** Always use exponential-integrator updates (DDIM or DPM-Solver++) for $\boldsymbol{\epsilon}$-, $\boldsymbol{x}$-, or $\boldsymbol{s}$-prediction, or parameterize the network in $\boldsymbol{v}$-prediction where plain Euler coincides with DDIM [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Obs. 9.2.1, p. 272).

> [!warning] High-Order Solver Blow-Up Under Large CFG Scales ($\omega \gg 1$)
> Under strong Classifier-Free Guidance ($\omega \ge 7.5$), $\boldsymbol{\epsilon}$-predicted outputs and their $\lambda$-derivatives amplify, causing higher-order Taylor/multistep solvers (DPM-Solver-2/3, DEIS) to overshoot data bounds and underperform DDIM. **Remedy:** Switch to DPM-Solver++ in $\boldsymbol{x}$-prediction with dynamic thresholding (clipping $\hat{\boldsymbol{x}}_\phi$ to training-data bounds) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 9.5.1–9.5.2, p. 300).

> [!warning] Consistency Training Collapse & Unbounded Tangent Derivatives
> Training a Consistency Model without a stop-gradient EMA target $\theta^-$ collapses immediately to the trivial constant solution $f_\theta(\boldsymbol{x}_s, s) \equiv \boldsymbol{c}$ ($\mathcal{L}=0$). Moreover, continuous-time consistency training `(eq:dsolver:cm_cont)` suffers exploding gradients when $\frac{\mathrm{d}}{\mathrm{d}s}f_{\theta^-}$ is unbounded. **Remedy:** Enforce stop-gradient EMA updates $\theta^- \gets \mu\theta^- + (1-\mu)\theta$ and adopt the trigonometric sCM parameterization `(eq:dsolver:cm_cont)` with bounded $c_{\mathrm{noise}}(s) = s$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 11.3.3, pp. 364–365).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[Classifier-Free Guidance (CFG)]], [[Consistency Models]]
- **Prerequisites (Directed $\leftarrow$):** [[DDPM]], [[Probability Flow ODE]]
- **Enables / Unlocks (Directed $\rightarrow$):** None
- **Related Concepts (Undirected $\leftrightarrow$):** [[CLIP & Vision-Language Alignment]], [[DDPM]], [[Flow Matching & Rectified Flow]], [[Probability Flow ODE]]
- **Synthesized Sources:** [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]]
