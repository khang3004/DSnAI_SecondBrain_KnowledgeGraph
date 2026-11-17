---
type: monograph
domain: "D01"
parent_topic: "[[PT 01 - Convex, Smooth & Stochastic Optimization]]"
sub_topic: "[[ST 01 - Stochastic, Adaptive, Proximal & Second-Order Optimizers]]"
cheatsheet: "[[Cheatsheet - Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]"
part_of: 
  - "[[ST 01 - Stochastic, Adaptive, Proximal & Second-Order Optimizers]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
concepts: 
  - "[[AdamW]]"
  - "[[Momentum & Nesterov NAG]]"
  - "[[Natural Gradient & K-FAC]]"
prerequisites: 
  - "[[Condition Number]]"
  - "[[Gradient Descent]]"
  - "[[Maximum Likelihood & Fisher Info]]"
  - "[[Momentum & Nesterov NAG]]"
enables: 
  - "[[AdamW]]"
  - "[[TRPO]]"
related: 
  - "[[AdamW]]"
  - "[[Condition Number]]"
  - "[[Gradient Descent]]"
  - "[[Maximum Likelihood & Fisher Info]]"
  - "[[Momentum & Nesterov NAG]]"
  - "[[TRPO]]"
  - "[[Weight Decay & Dropout]]"
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Deep Learning]]"
  - "[[Source - Handbook of Convergence Theorems]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-01
---

# Momentum, Nesterov, Adam/AdamW, LR Schedules & Second-Order / Natural Gradient

> [!abstract] Monograph Navigation
> - **Domain:** [[D01 - Mathematical Foundations for Machine Intelligence|Mathematical Foundations for Machine Intelligence]]
> - **Parent / Sub-Topic:** [[PT 01 - Convex, Smooth & Stochastic Optimization|Convex, Smooth & Stochastic Optimization]] → [[ST 01 - Stochastic, Adaptive, Proximal & Second-Order Optimizers|Stochastic, Adaptive, Proximal & Second-Order Optimizers]]
> - **Visual Cheatsheet:** [[Cheatsheet - Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]
> - **Description:** Reconciles Polyak heavy-ball vs. Nesterov accelerated gradient, derives AdaGrad/RMSProp/Adam/AdamW bias correction and decoupled weight decay, cosine warmup schedules, and Fisher/natural-gradient & L-BFGS curvature methods.

## Problem Formulation & Unified Notation

First-order steepest descent $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \boldsymbol{g}_t$ suffers a fundamental geometric handicap on modern statistical models: its step direction $-\boldsymbol{g}_t$ is tied to the canonical Euclidean inner product $\lVert \boldsymbol{\delta} \rVert_2^2$, so on any anisotropic landscape with Hessian condition number $\kappa = L/\mu \gg 1$, progress along shallow ravine axes is throttled by the step-size stability limit $\eta_t < 2/L$ imposed by steep orthogonal walls [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.2.1, p. 282) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 6.4, p. 88). In stochastic mini-batch regimes $\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{z}}[\ell(\boldsymbol{\theta};\boldsymbol{z})]$ or finite-sum empirical risk minimization $\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N}\sum_{n=1}^N \ell_n(\boldsymbol{\theta})$, sampling variance $\sigma_*^2$ further forces decaying step sizes unless gradient directions are smoothed across iterations [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.3, pp. 271–273) [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ch. 6–7, pp. 34–39).

Following the general framework of [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. (6.48), p. 273) and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. (8.68), p. 298), every optimizer synthesized in this monograph—from Polyak heavy-ball momentum and Nesterov acceleration to adaptive moment methods (AdaGrad, RMSProp, Adam, AdamW), quasi-Newton methods (BFGS, L-BFGS), Gauss–Newton, and Natural Gradient Descent (NGD/K-FAC)—specializes a single *preconditioned, regularized update template*:

$$
\boldsymbol{\theta}_{t} = (1 - \eta_t \lambda)\boldsymbol{\theta}_{t-1} - \eta_t \boldsymbol{P}_t^{-1} \boldsymbol{d}_t,
$$

where $\eta_t > 0$ is the scheduled global learning rate, $\lambda \ge 0$ is the decoupled weight-decay coefficient, $\boldsymbol{d}_t \in \mathbb{R}^P$ is an instantaneous, lookahead, or exponentially smoothed gradient surrogate, and $\boldsymbol{P}_t \in \mathbb{S}_{++}^P$ is a symmetric positive-definite preconditioning metric:
- **Isotropic first-order:** $\boldsymbol{P}_t = \boldsymbol{I}_P$, varying $\boldsymbol{d}_t$ via momentum memory or lookahead gradient evaluation [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ch. 7, p. 36) [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 12.4–12.5, pp. 148–152).
- **Diagonal adaptive preconditioning:** $\boldsymbol{P}_t = \mathrm{diag}(\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon\boldsymbol{1})$, scaling each coordinate inversely with its smoothed gradient root-mean-square magnitude across iterations, as developed in [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.5, pp. 306–311) and [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 6.4, pp. 88–90).
- **Structured curvature & information geometry:** $\boldsymbol{P}_t$ approximates the Hessian $\boldsymbol{H}_t = \nabla^2\mathcal{L}(\boldsymbol{\theta}_{t-1})$ (Newton, Levenberg–Marquardt, BFGS/L-BFGS), the Generalized Gauss–Newton matrix $\boldsymbol{G}_t = \boldsymbol{J}_t^{\top} \boldsymbol{\Lambda}_t \boldsymbol{J}_t$, or the Fisher Information Matrix $\boldsymbol{F}_t$ (NGD, K-FAC) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.4 & Sec. 17.3.2, pp. 279–286, 652–653).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Because optimization theory spans machine learning, numerical PDEs, and information geometry, our six source textbooks collide heavily on symbols for learning rates, momentum weights, velocity sign conventions, and second-moment accumulators. Table `tab:optalg:notation` establishes the canonical notation used throughout this monograph. In particular, note two algebraic conventions that alter effective step sizes if overlooked:
> - **Convex-combination vs. unnormalized momentum:** Adam and Prince formulate velocity as a normalized EMA $\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + (1 - \beta_1)\boldsymbol{g}_t$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Eq. (6.11), p. 86), keeping $\boldsymbol{m}_t$ on the same scale as $\boldsymbol{g}_t$ ($\boldsymbol{m}_\infty = \boldsymbol{g}$ for constant $\boldsymbol{g}$). By contrast, [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Eq. (8.15), p. 296), [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Eq. (58), p. 36), and [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Eq. (12.7), p. 149) use unnormalized accumulation $\boldsymbol{m}_t = \beta \boldsymbol{m}_{t-1} + \boldsymbol{g}_t$, amplifying steady-state gradients by $\frac{1}{1-\beta}$ (a factor of $10\times$ at $\beta = 0.9$). Both rules yield identical iterates under the learning-rate rescaling $\eta_{\mathrm{raw}} = (1-\beta)\eta_{\mathrm{ema}}$.
> - **Bias-correction indexing:** $1$-indexed loops ($t = 1,2,\dots$ starting from $\boldsymbol{m}_0 = \boldsymbol{v}_0 = \boldsymbol{0}$) divide by $1 - \beta_1^t$ and $1 - \beta_2^t$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Alg. 8.7, p. 311) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. (8.80), p. 300), whereas $0$-indexed updates computing $\boldsymbol{m}_{t+1}$ at step $t=0$ divide by $1 - \beta^{t+1}$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Eq. (6.16), p. 90). Both represent the same $k$-th update step.

**Table:** Unified Notation and Cross-Book Convention Reconciliation across Optimization Sources.

| **Concept** | **Unified Symbol** | **Source Textbook Conventions & Reconciliation** |
| — | — | — |
| Parameter vector | $\boldsymbol{\theta}_t \in \mathbb{R}^P$ | $\boldsymbol{\theta}$ in [[Source - Deep Learning]] (Sec. 8.3), [[Source - Probabilistic Machine Learning]] (Sec. 6.1), and [[Source - Machine Learning co ban]] (Ch. 12); $\boldsymbol{\phi}_t$ in [[Source - Understanding Deep Learning]] (Sec. 6.3); $\boldsymbol{x}_t \in \mathbb{R}^d$ in [[Source - Handbook of Convergence Theorems]] (Ch. 7). |
| Learning rate | $\eta_t > 0$ | $\eta_t$ in [[Source - Probabilistic Machine Learning]] (Sec. 6.3) and [[Source - Machine Learning co ban]] (Eq. (12.2)); $\epsilon$ in [[Source - Deep Learning]] (Alg. 8.1–8.7); $\alpha$ in [[Source - Understanding Deep Learning]] (Sec. 6.3); $\gamma_t$ in [[Source - Handbook of Convergence Theorems]] (Alg. 7.1). |
| Stochastic gradient | $\boldsymbol{g}_t = \nabla\mathcal{L}_{B_t}(\boldsymbol{\theta}_{t-1})$ | $\hat{\boldsymbol{g}}$ or $\boldsymbol{g}$ in [[Source - Deep Learning]] (Alg. 8.1); $\sum_{i\in B_t}\frac{\partial \ell_i}{\partial \phi}$ in [[Source - Understanding Deep Learning]] (Eq. (6.11)); $\nabla f_{i_t}(\boldsymbol{x}_t)$ in [[Source - Handbook of Convergence Theorems]] (Eq. (58)). |
| 1st-moment state | $\boldsymbol{m}_t,  \beta_1 \in [0,1)$ | $(\boldsymbol{v}, \alpha)$ with negative-gradient sign in [[Source - Deep Learning]] (Eq. (8.15)) and $(\boldsymbol{s}, \rho_1)$ in Alg. 8.7; $(\boldsymbol{m}_t, \beta)$ in [[Source - Understanding Deep Learning]] (Eq. (6.15)); $(\boldsymbol{v}_t, \gamma)$ in [[Source - Machine Learning co ban]] (Eq. (12.7)). |
| 2nd-moment state | $\boldsymbol{v}_t,  \beta_2 \in [0,1)$ | $(\boldsymbol{r}, \rho)$ in [[Source - Deep Learning]] (Alg. 8.5) and $(\boldsymbol{r}, \rho_2)$ in Alg. 8.7; $(\boldsymbol{v}_t, \gamma)$ in [[Source - Understanding Deep Learning]] (Eq. (6.15)); $(\boldsymbol{s}_t, \beta_2)$ in [[Source - Probabilistic Machine Learning]] (Eq. (8.78)). |
| Numerical stabilizer | $\epsilon > 0$ | $\delta$ ($10^{-7}$–$10^{-8}$) in [[Source - Deep Learning]] (Alg. 8.4–8.7); $\epsilon$ in [[Source - Understanding Deep Learning]] (Eq. (6.14)) and [[Source - Probabilistic Machine Learning]] (Eq. (8.79), $10^{-6}$). |
| Weight decay rate | $\lambda \ge 0$ | Decoupled `weight\_decay` in [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]] (p. 372) and [[Source - Understanding Deep Learning]] (p. 156, $\lambda'$); distinct from $L_2$ coefficient $\lambda$ in Eq. (9.5). |
| Fisher / Hessian | $\boldsymbol{F}(\boldsymbol{\theta}),  \boldsymbol{H}(\boldsymbol{\theta})$ | $\boldsymbol{F}(\boldsymbol{\theta})$ in [[Source - Probabilistic Machine Learning]] (Eq. (6.84)); $\boldsymbol{H} = \nabla^2 J$ in [[Source - Deep Learning]] (Eq. (8.26)) and [[Source - Probabilistic Machine Learning]] (Eq. (8.40)). |

## Core Mathematical Derivations & Theorems

### Polyak Heavy-Ball Momentum: Equivalences & Spectral Acceleration

In optimization literature, momentum appears in three superficially distinct algebraic formats: the ML velocity update, Polyak's two-step Heavy-Ball recursion, and the Iterate-Moving-Average (IMA) form. [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Sec. 7.1, Lemmas 7.2–7.3, pp. 36–37) establishes their exact trajectory equivalence.

> [!important] Proposition: Three-Way Equivalence & Stochastic Momentum Rate
> Consider the stochastic momentum sequence initialized at $\boldsymbol{\theta}_0 = \boldsymbol{\theta}_{-1} \in \mathbb{R}^P$, $\boldsymbol{m}_0 = \boldsymbol{0}$, with step sizes $\{\gamma_t\}_{t \ge 1}$ and momentum weights $\{\beta_t\}_{t \ge 1} \subset [0, 1)$:
>
> $$
> \text{(ML Momentum)}\qquad \boldsymbol{m}_t = \beta_t \boldsymbol{m}_{t-1} + \boldsymbol{g}_t, \qquad \boldsymbol{\theta}_{t} = \boldsymbol{\theta}_{t-1} - \gamma_t \boldsymbol{m}_t.
>
> $$
>
> Then:
> - **Heavy-Ball Equivalence:** For $t \ge 1$, the iterates $\{\boldsymbol{\theta}_t\}$ obey Polyak's Heavy-Ball form
>
>
> $$
> \boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \hat{\gamma}_t \boldsymbol{g}_t + \hat{\beta}_t(\boldsymbol{\theta}_{t-1} - \boldsymbol{\theta}_{t-2}),
>
> $$
>
>   with $\hat{\gamma}_t = \gamma_t$, $\hat{\beta}_1 = 0$, and $\hat{\beta}_t = \frac{\gamma_t}{\gamma_{t-1}}\beta_t$ for $t \ge 2$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lemma 7.2, p. 36). When $\gamma_t \equiv \gamma$ is constant, $\hat{\beta}_t = \beta_t$.
> - **Iterate-Moving-Average (IMA) Equivalence:** Choosing $(\eta_t, \lambda_t)$ satisfying $\beta_t \lambda_{t+1} = \frac{\gamma_{t-1}}{\gamma_t}\lambda_t - \beta_t$ and $\eta_t = (1 + \lambda_{t+1})\gamma_t$ (with $\boldsymbol{z}_0 = \boldsymbol{\theta}_0$ and $\boldsymbol{z}_{t-1} = \boldsymbol{\theta}_{t-1} + \lambda_t(\boldsymbol{\theta}_{t-1} - \boldsymbol{\theta}_{t-2})$), `(eq:optalg:ml_mom)` is identical to
>
>
> $$
> \boldsymbol{z}_t = \boldsymbol{z}_{t-1} - \eta_t \boldsymbol{g}_t, \qquad \boldsymbol{\theta}_t = \frac{\lambda_{t+1}}{\lambda_{t+1}+1}\boldsymbol{\theta}_{t-1} + \frac{1}{\lambda_{t+1}+1}\boldsymbol{z}_t.
>
> $$
> - **Stochastic Convex Convergence:** If each component $f_i$ is convex and $L_{\max}$-smooth with gradient noise $\sigma_f^* = \inf_{\boldsymbol{\theta}^*\in\arg\min f}\mathrm{Var}[\nabla f_i(\boldsymbol{\theta}^*)]$ at optima, then choosing $\gamma_t = \frac{2\eta}{t+2}$ and $\beta_t = \frac{t-1}{t+1}$ ($t \ge 1$, corresponding to constant $\eta_t \equiv \eta \le \frac{1}{4L_{\max}}$ and $\lambda_t = \frac{t-1}{2}$) yields last-iterate convergence [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Thm. 7.4 & Cor. 7.5, pp. 38–39):
>
>
> $$
> \mathbb{E}\bigl[f(\boldsymbol{\theta}_T) - \inf f\bigr] \le \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{\eta T} + 2\eta\sigma_f^*,
>
> $$
>
>   giving $O(1/\varepsilon^2)$ complexity for $\eta = \frac{1}{4L_{\max}\sqrt{T}}$ (without iterate averaging).
>
>
> **Proof of (1)–(2).** From `(eq:optalg:ml_mom)` at step $t-1$ ($t \ge 2$), $\boldsymbol{m}_{t-1} = \frac{1}{\gamma_{t-1}}(\boldsymbol{\theta}_{t-2} - \boldsymbol{\theta}_{t-1})$. Substituting into $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \gamma_t(\beta_t\boldsymbol{m}_{t-1} + \boldsymbol{g}_t)$ gives $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \gamma_t\boldsymbol{g}_t + \frac{\gamma_t\beta_t}{\gamma_{t-1}}(\boldsymbol{\theta}_{t-1} - \boldsymbol{\theta}_{t-2})$, proving `(eq:optalg:hb_form)`. For (2), isolating $\boldsymbol{z}_t = (1 + \lambda_{t+1})\boldsymbol{\theta}_t - \lambda_{t+1}\boldsymbol{\theta}_{t-1}$ and inserting `(eq:optalg:hb_form)` yields
>
> $$
>
>   \boldsymbol{z}_t = \boldsymbol{\theta}_{t-1} + (1 + \lambda_{t+1})\hat{\beta}_t(\boldsymbol{\theta}_{t-1} - \boldsymbol{\theta}_{t-2}) - (1 + \lambda_{t+1})\gamma_t \boldsymbol{g}_t = \boldsymbol{z}_{t-1} - \eta_t \boldsymbol{g}_t,
>
> $$
>
> because the stated recurrence ensures $(1+\lambda_{t+1})\hat{\beta}_t = (1+\lambda_{t+1})\frac{\gamma_t}{\gamma_{t-1}}\beta_t = \lambda_t$.

While Theorem `thm:optalg:momentum_equiv` guarantees last-iterate stability in stochastic convex optimization, Polyak Heavy-Ball reveals its signature *square-root condition-number speedup* on deterministic strongly convex quadratics.

> [!important] Theorem: Polyak Heavy-Ball $\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$ Spectral Acceleration
> Let $f(\boldsymbol{\theta}) = \frac{1}{2}\boldsymbol{\theta}^{\top} \boldsymbol{A}\boldsymbol{\theta} - \boldsymbol{b}^{\top}\boldsymbol{\theta}$ with $\boldsymbol{A} \in \mathbb{S}_{++}^P$ having eigenvalues $0 < \mu = \lambda_1 \le \dots \le \lambda_P = L$ and condition number $\kappa = L/\mu$. Under constant Heavy-Ball updates $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \nabla f(\boldsymbol{\theta}_t) + \beta(\boldsymbol{\theta}_t - \boldsymbol{\theta}_{t-1})$ with optimal hyperparameters
>
> $$
> \alpha^* = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}, \qquad \beta^* = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^2,
>
> $$
>
> the augmented error state $\boldsymbol{\xi}_t = [(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*)^{\top},  (\boldsymbol{\theta}_{t-1} - \boldsymbol{\theta}^*)^{\top}]^{\top}$ contracts asymptotically with spectral radius
>
> $$
> \rho_{\mathrm{HB}}^* = \sqrt{\beta^*} = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} = 1 - \frac{2}{\sqrt{\kappa} + 1},
>
> $$
>
> achieving $\lVert \boldsymbol{\theta}_T - \boldsymbol{\theta}^* \rVert_2 \le C_\delta (\rho_{\mathrm{HB}}^* + \delta)^T \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2$ for any $\delta > 0$, improving iteration complexity from $O(\kappa \log(1/\varepsilon))$ for optimal gradient descent ($\rho_{\mathrm{GD}}^* = \frac{\kappa - 1}{\kappa + 1}$) to $O(\sqrt{\kappa}\log(1/\varepsilon))$.
>
> **Proof.** Write $\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\top}$ via the spectral theorem and transform the error $\boldsymbol{e}_t = \boldsymbol{Q}^{\top}(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*)$, where $\nabla f(\boldsymbol{\theta}_t) = \boldsymbol{A}(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*)$. Each eigenmode $i \in \{1,\dots,P\}$ decouples into a $2\times 2$ linear dynamical system:
>
> $$
>
>   \begin{bmatrix} e_{t+1,i} \\ e_{t,i} \end{bmatrix}
>   = \underbrace{\begin{bmatrix} 1 + \beta - \alpha \lambda_i & -\beta \\ 1 & 0 \end{bmatrix}}_{\boldsymbol{T}(\lambda_i)}
>   \begin{bmatrix} e_{t,i} \\ e_{t-1,i} \end{bmatrix}.
>
> $$
>
> The eigenvalues $r \in \mathbb{C}$ of $\boldsymbol{T}(\lambda_i)$ satisfy the characteristic quadratic $r^2 - (1 + \beta - \alpha\lambda_i)r + \beta = 0$. Whenever the discriminant is non-positive,
>
> $$
>
>   \Delta(\lambda_i) = (1 + \beta - \alpha\lambda_i)^2 - 4\beta \le 0 \iff (1 - \sqrt{\beta})^2 \le \alpha \lambda_i \le (1 + \sqrt{\beta})^2,
>
> $$
>
> both roots form a complex-conjugate pair $r_{1,2} = \frac{1}{2}(1 + \beta - \alpha\lambda_i \pm \mathrm{i}\sqrt{-\Delta(\lambda_i)})$ lying strictly on the circle of radius $|r_1| = |r_2| = \sqrt{\beta}$ in $\mathbb{C}$. Enforcing $\Delta(\lambda_i) \le 0$ uniformly across the entire spectrum $[\mu, L]$ requires $(1-\sqrt{\beta})^2 \le \alpha \mu$ and $\alpha L \le (1+\sqrt{\beta})^2$. Minimizing the common spectral radius $\sqrt{\beta}$ makes both endpoint constraints tight:
>
> $$
>
>   \frac{L}{\mu} = \kappa = \left(\frac{1 + \sqrt{\beta^*}}{1 - \sqrt{\beta^*}}\right)^2 \implies \sqrt{\kappa}(1 - \sqrt{\beta^*}) = 1 + \sqrt{\beta^*} \implies \sqrt{\beta^*} = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}.
>
> $$
>
> Substituting $\sqrt{\beta^*}$ back into $\alpha^* \mu = (1 - \sqrt{\beta^*})^2 = \frac{4}{(\sqrt{\kappa}+1)^2}$ yields $\alpha^* = \frac{4}{\mu(\sqrt{\kappa}+1)^2} = \frac{4}{(\sqrt{L}+\sqrt{\mu})^2}$. By Gelfand's spectral radius formula $\lim_{T\to\infty}\lVert \boldsymbol{T}(\lambda_i)^T \rVert_2^{1/T} = \rho(\boldsymbol{T}(\lambda_i)) = \sqrt{\beta^*}$, establishing `(eq:optalg:hb_spectral)`.

> [!tip] Newtonian Particle with Viscous Drag & Non-Normal Overshoot
> [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.3.2, pp. 298–300) interprets momentum as Euler discretization of a unit-mass particle sliding down potential $J(\boldsymbol{\theta})$ under viscous drag: $\ddot{\boldsymbol{\theta}}(t) = -\nabla J(\boldsymbol{\theta}(t)) - \gamma\dot{\boldsymbol{\theta}}(t)$. Unlike turbulent drag ($\propto -\lVert \dot{\boldsymbol{\theta}} \rVert^2$, too weak to stop oscillation) or dry Coulomb friction (constant magnitude, halting short of minima), linear viscous drag dissipates oscillations across steep canyon walls while allowing sustained acceleration along the flat valley floor ([[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 12.4, pp. 148–150) notes that finite inertia can also carry iterates over shallow local-minimum barriers). Note, however, that because $\boldsymbol{T}(\lambda_i)$ in Theorem `thm:optalg:hb_rate` is non-normal, $\lVert \boldsymbol{T}(\lambda_i)^t \rVert_2$ exhibits a transient early-step hump before exponential decay takes hold (see Figure `fig:optalg:dynamics`a), and on general non-quadratic strongly convex objectives Heavy-Ball with fixed $(\alpha^*,\beta^*)$ lacks a global convergence guarantee without lookahead correction.

### Nesterov Accelerated Gradient (NAG): Lookahead Correction & $O(1/T^2)$ Rate

To cure Heavy-Ball's late-stage valley overshoot, Nesterov Accelerated Gradient (NAG) evaluates the gradient not at the current iterate $\boldsymbol{\theta}_{t-1}$, but at the momentum-extrapolated *lookahead* point [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Eq. (6.12), p. 88) [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Eq. (12.10), p. 152) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Alg. 8.3, p. 300):

$$
\begin{aligned}
\text{(Sutskever Lookahead)}\quad & \tilde{\boldsymbol{\theta}}_t = \boldsymbol{\theta}_{t-1} + \beta_t \boldsymbol{v}_{t-1},\quad \boldsymbol{v}_t = \beta_t \boldsymbol{v}_{t-1} - \eta_t \nabla f(\tilde{\boldsymbol{\theta}}_t),\quad \boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} + \boldsymbol{v}_t,  \\
\text{(Nesterov Extrapolation)}\quad & \boldsymbol{y}_t = \boldsymbol{x}_t + \beta_t(\boldsymbol{x}_t - \boldsymbol{x}_{t-1}),\quad \boldsymbol{x}_{t+1} = \boldsymbol{y}_t - \eta_t \nabla f(\boldsymbol{y}_t).
\end{aligned}
$$

Setting $\boldsymbol{v}_t = \boldsymbol{x}_{t+1} - \boldsymbol{x}_t$ and $\boldsymbol{y}_t = \boldsymbol{x}_t + \beta_t \boldsymbol{v}_{t-1}$ shows exact equivalence when $\beta_t$ is constant ($\boldsymbol{x}_{t+1} - \boldsymbol{x}_t = \beta(\boldsymbol{x}_t - \boldsymbol{x}_{t-1}) - \eta\nabla f(\boldsymbol{y}_t)$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eqs. (8.35)–(8.38), p. 288); for time-varying $\beta_t$, `(eq:optalg:nag_nesterov)` is the canonical convex acceleration scheme whose $O(1/T^2)$ suboptimality matches the information-theoretic first-order oracle lower bound.

> [!important] Theorem: Optimal $O(1/T^2)$ Convergence of NAG via Lyapunov Potential
> Let $f : \mathbb{R}^P \to \mathbb{R}$ be convex and $L$-smooth with minimizer $\boldsymbol{x}^* \in \arg\min f$. Starting from $\boldsymbol{x}_0 = \boldsymbol{x}_{-1} \in \mathbb{R}^P$, define $\lambda_0 = 0$, $\lambda_1 = 1$, and
>
> $$
> \lambda_{t+1} = \frac{1 + \sqrt{1 + 4\lambda_t^2}}{2}, \qquad \beta_t = \frac{\lambda_t - 1}{\lambda_{t+1}} \qquad (t \ge 1).
>
> $$
>
> Running `(eq:optalg:nag_nesterov)` with $\eta_t \equiv 1/L$ guarantees, for every $T \ge 1$:
>
> $$
> f(\boldsymbol{x}_T) - f(\boldsymbol{x}^*) \le \frac{2L\lVert \boldsymbol{x}_0 - \boldsymbol{x}^* \rVert_2^2}{(T + 1)^2} = O\!\left(\frac{1}{T^2}\right),
>
> $$
>
> improving quadratically upon deterministic gradient descent's $O(1/T)$ bound $\frac{L\lVert \boldsymbol{x}_0-\boldsymbol{x}^* \rVert_2^2}{2T}$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Thm. 3.4, p. 14) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.3.3, p. 300). For $\mu$-strongly convex $L$-smooth objectives, constant $\eta_t \equiv 1/L$ and $\beta_t \equiv \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$ achieve global linear convergence $f(\boldsymbol{x}_T) - f(\boldsymbol{x}^*) \le \left(1 - \frac{1}{\sqrt{\kappa}}\right)^T \bigl(f(\boldsymbol{x}_0) - f(\boldsymbol{x}^*) + \frac{\mu}{2}\lVert \boldsymbol{x}_0 - \boldsymbol{x}^* \rVert_2^2\bigr)$.
>
> **Proof.** Let $A_t \triangleq \lambda_t^2$ and $a_{t+1} \triangleq \lambda_{t+1}$ for $t \ge 0$ (so $A_0 = 0$, $A_1 = a_1 = 1$). By `(eq:optalg:nag_seq)`, $\lambda_{t+1}^2 - \lambda_{t+1} = \lambda_t^2$, meaning $A_{t+1} = A_t + a_{t+1}$ and $a_{t+1}^2 = A_{t+1}$. Define the auxiliary mirror sequence $\boldsymbol{z}_0 = \boldsymbol{x}_0$ and
>
> $$
>
>   \boldsymbol{z}_{t+1} = \boldsymbol{z}_t - \frac{a_{t+1}}{L}\nabla f(\boldsymbol{y}_t), \qquad \boldsymbol{y}_t = \frac{A_t}{A_{t+1}}\boldsymbol{x}_t + \frac{a_{t+1}}{A_{t+1}}\boldsymbol{z}_t.
>
> $$
>
> (Unrolling $a_t\boldsymbol{z}_t = A_t\boldsymbol{x}_t - A_{t-1}\boldsymbol{x}_{t-1}$ gives $\boldsymbol{z}_t = \boldsymbol{x}_t + (a_t - 1)(\boldsymbol{x}_t - \boldsymbol{x}_{t-1})$, which upon substitution recovers $\boldsymbol{y}_t = \boldsymbol{x}_t + \frac{a_t - 1}{a_{t+1}}(\boldsymbol{x}_t - \boldsymbol{x}_{t-1}) = \boldsymbol{x}_t + \beta_t(\boldsymbol{x}_t - \boldsymbol{x}_{t-1})$.) Introduce the Lyapunov potential
>
> $$
>
>   \mathcal{E}_t \triangleq A_t\bigl(f(\boldsymbol{x}_t) - f(\boldsymbol{x}^*)\bigr) + \frac{L}{2}\lVert \boldsymbol{z}_t - \boldsymbol{x}^* \rVert_2^2.
>
> $$
>
> By $L$-smoothness at $\boldsymbol{y}_t$ and $\boldsymbol{x}_{t+1} = \boldsymbol{y}_t - \frac{1}{L}\nabla f(\boldsymbol{y}_t)$, descent yields $f(\boldsymbol{x}_{t+1}) \le f(\boldsymbol{y}_t) - \frac{1}{2L}\lVert \nabla f(\boldsymbol{y}_t) \rVert_2^2$. Meanwhile, convexity of $f$ at $\boldsymbol{y}_t$ gives $f(\boldsymbol{x}_t) \ge f(\boldsymbol{y}_t) + \langle \nabla f(\boldsymbol{y}_t), \boldsymbol{x}_t - \boldsymbol{y}_t \rangle$ and $f(\boldsymbol{x}^*) \ge f(\boldsymbol{y}_t) + \langle \nabla f(\boldsymbol{y}_t), \boldsymbol{x}^* - \boldsymbol{y}_t \rangle$. Weighting these two inequalities by $A_t$ and $a_{t+1}$ ($A_{t+1} = A_t + a_{t+1}$) and using $A_t(\boldsymbol{x}_t - \boldsymbol{y}_t) = a_{t+1}(\boldsymbol{y}_t - \boldsymbol{z}_t)$ yields
>
> $$
>
>   A_{t+1} f(\boldsymbol{y}_t) \le A_t f(\boldsymbol{x}_t) + a_{t+1} f(\boldsymbol{x}^*) + a_{t+1}\langle \nabla f(\boldsymbol{y}_t), \boldsymbol{z}_t - \boldsymbol{x}^* \rangle.
>
> $$
>
> Combining with $A_{t+1}f(\boldsymbol{x}_{t+1}) \le A_{t+1}f(\boldsymbol{y}_t) - \frac{A_{t+1}}{2L}\lVert \nabla f(\boldsymbol{y}_t) \rVert_2^2$ and subtracting $A_{t+1}f(\boldsymbol{x}^*)$ produces
>
> $$
> A_{t+1}\bigl(f(\boldsymbol{x}_{t+1}) - f(\boldsymbol{x}^*)\bigr) - A_t\bigl(f(\boldsymbol{x}_t) - f(\boldsymbol{x}^*)\bigr)
>   \le a_{t+1}\langle \nabla f(\boldsymbol{y}_t), \boldsymbol{z}_t - \boldsymbol{x}^* \rangle - \frac{A_{t+1}}{2L}\lVert \nabla f(\boldsymbol{y}_t) \rVert_2^2.
>
> $$
>
> Expanding $\frac{L}{2}\lVert \boldsymbol{z}_{t+1} - \boldsymbol{x}^* \rVert_2^2$ via $\boldsymbol{z}_{t+1} = \boldsymbol{z}_t - \frac{a_{t+1}}{L}\nabla f(\boldsymbol{y}_t)$ yields
>
> $$
> \frac{L}{2}\lVert \boldsymbol{z}_{t+1} - \boldsymbol{x}^* \rVert_2^2 - \frac{L}{2}\lVert \boldsymbol{z}_t - \boldsymbol{x}^* \rVert_2^2
>   = -a_{t+1}\langle \nabla f(\boldsymbol{y}_t), \boldsymbol{z}_t - \boldsymbol{x}^* \rangle + \frac{a_{t+1}^2}{2L}\lVert \nabla f(\boldsymbol{y}_t) \rVert_2^2.
>
> $$
>
> Adding `(eq:optalg:nag_step1)` and `(eq:optalg:nag_step2)` cancels the cross-term identically and leaves $\mathcal{E}_{t+1} - \mathcal{E}_t \le \frac{a_{t+1}^2 - A_{t+1}}{2L}\lVert \nabla f(\boldsymbol{y}_t) \rVert_2^2 = 0$. Finally, induction on `(eq:optalg:nag_seq)` shows $\lambda_t \ge \frac{t+1}{2}$ for all $t \ge 1$ (since $\lambda_1 = 1$ and $\lambda_{t+1} > \frac{1 + 2\lambda_t}{2} = \lambda_t + \frac{1}{2}$), whence $A_T = \lambda_T^2 \ge \frac{(T+1)^2}{4}$. Since $\mathcal{E}_T \le \mathcal{E}_0 = \frac{L}{2}\lVert \boldsymbol{x}_0 - \boldsymbol{x}^* \rVert_2^2$, dividing by $A_T$ yields `(eq:optalg:nag_bound)`.

> [!tip] Why Lookahead Acts as Curvature-Aware Damping
> Writing $\nabla f(\boldsymbol{y}_t) \approx \nabla f(\boldsymbol{x}_t) + \beta_t \boldsymbol{H}(\boldsymbol{x}_t)(\boldsymbol{x}_t - \boldsymbol{x}_{t-1})$, NAG implicitly injects a finite-difference Hessian-velocity braking term $-\eta_t \beta_t \boldsymbol{H}(\boldsymbol{x}_t)(\boldsymbol{x}_t - \boldsymbol{x}_{t-1})$ without forming $\boldsymbol{H}$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Fig. 12.11, p. 151) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Fig. 6.8, p. 87). Along high-curvature directions where momentum threatens to overshoot across the canyon floor, $\nabla f(\boldsymbol{y}_t)$ reverses sign ahead of impact and brakes velocity immediately; along flat valley directions where curvature is tiny, $\nabla f(\boldsymbol{y}_t) \approx \nabla f(\boldsymbol{x}_t)$ preserves full momentum acceleration. In noisy mini-batch regimes, however, [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.3.3, p. 300) cautions that stochastic variance $O(\sigma/\sqrt{T})$ dominates deterministic acceleration unless mini-batch variance is simultaneously controlled.

### Adaptive Moments: AdaGrad $\to$ RMSProp $\to$ Adam Bias Correction $\to$ AdamW

Even with momentum, isotropic methods apply a single scalar step size $\eta_t$ across all $P$ coordinates, failing when parameter sensitivities span orders of magnitude across layers or sparse features [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 6.4, p. 88) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.6, pp. 298–300). Diagonal adaptive optimizers construct coordinate-wise preconditioners $\boldsymbol{P}_t = \mathrm{diag}(\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon\boldsymbol{1})$:
- **AdaGrad:** Accumulates the unattenuated sum of squared gradients $\boldsymbol{v}_t = \boldsymbol{v}_{t-1} + \boldsymbol{g}_t \odot \boldsymbol{g}_t$, updating $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \frac{\eta}{\sqrt{\boldsymbol{v}_t} + \epsilon} \odot \boldsymbol{g}_t$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Alg. 8.4, p. 308) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eqs. (8.69)–(8.71), p. 299). While optimal for sparse convex problems, monotonic growth of $\boldsymbol{v}_t$ prematurely freezes learning rates before reaching local minima in deep non-convex trajectories [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.5.1, p. 307) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (p. 93).
- **RMSProp:** Replaces cumulative summation with an exponentially weighted moving average (EWMA) $\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + (1 - \beta_2)\boldsymbol{g}_t^{\odot 2}$, forgetting ancient curvature history so the optimizer adapts locally inside each non-convex basin [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Alg. 8.5, p. 309) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eqs. (8.72)–(8.74), p. 299).
- **Adam:** Couples EWMA first-moment smoothing $\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + (1-\beta_1)\boldsymbol{g}_t$ with RMSProp's second-moment accumulator $\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + (1-\beta_2)\boldsymbol{g}_t^{\odot 2}$ and exact zero-initialization bias correction [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Alg. 8.7, p. 311) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Eqs. (6.15)–(6.17), p. 90).

> [!important] Derivation: Adam Bias Correction & Adaptive Step-Size Normalization
> Initializing $\boldsymbol{m}_0 = \boldsymbol{v}_0 = \boldsymbol{0}$ with decay rates $\beta_1, \beta_2 \in [0, 1)$, unrolling the Adam second-moment recurrence for $t \ge 1$ gives
>
> $$
> \boldsymbol{m}_t = (1 - \beta_1)\sum_{i=1}^t \beta_1^{t-i}\boldsymbol{g}_i, \qquad
>   \boldsymbol{v}_t = (1 - \beta_2)\sum_{i=1}^t \beta_2^{t-i}\boldsymbol{g}_i^{\odot 2}.
>
> $$
>
> Assuming stationary gradient moments $\mathbb{E}[\boldsymbol{g}_i] = \mathbb{E}[\boldsymbol{g}]$ and $\mathbb{E}[\boldsymbol{g}_i^{\odot 2}] = \mathbb{E}[\boldsymbol{g}^{\odot 2}]$ (or shifting weights onto the geometric sum $(1 - \beta_2)\sum_{i=1}^t \beta_2^{t-i} = 1 - \beta_2^t$ up to a transient non-stationarity remainder $\boldsymbol{\zeta}_t$),
>
> $$
>
>   \mathbb{E}[\boldsymbol{m}_t] = (1 - \beta_1^t)\mathbb{E}[\boldsymbol{g}] + \boldsymbol{\zeta}_t^{(1)}, \qquad
>   \mathbb{E}[\boldsymbol{v}_t] = (1 - \beta_2^t)\mathbb{E}[\boldsymbol{g}^{\odot 2}] + \boldsymbol{\zeta}_t^{(2)}.
>
> $$
>
> Dividing out the geometric normalization shortfall yields the bias-corrected moments [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Alg. 8.7, p. 311) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eqs. (8.80)–(8.81), p. 300):
>
> $$
> \hat{\boldsymbol{m}}_t = \frac{\boldsymbol{m}_t}{1 - \beta_1^t}, \qquad
>   \hat{\boldsymbol{v}}_t = \frac{\boldsymbol{v}_t}{1 - \beta_2^t}, \qquad
>   \Delta\boldsymbol{\theta}_t = -\eta_t \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon}.
>
> $$
>
> **Scale Invariance & Early-Step Stabilization:** At $t=1$, without bias correction ($\beta_1=0.9, \beta_2=0.999$), $\boldsymbol{m}_1/\sqrt{\boldsymbol{v}_1} = \frac{1-\beta_1}{\sqrt{1-\beta_2}}\mathrm{sign}(\boldsymbol{g}_1) \approx 3.162 \mathrm{sign}(\boldsymbol{g}_1)$, inflating the first step by over $3\times$ relative to $\eta_1$ (whereas RMSProp without correction exhibits severe early variance bias [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.5.3, p. 309)). With bias correction, $\hat{\boldsymbol{m}}_1/\sqrt{\hat{\boldsymbol{v}}_1} = \mathrm{sign}(\boldsymbol{g}_1)$, capping $|\Delta\theta_{1,j}| \le \eta_1$. Moreover, in the stationary regime ($\epsilon \to 0$, $t \gg (1-\beta_2)^{-1}$), Jensen's inequality on the ratio gives $|\hat{m}_{t,j}|/\sqrt{\hat{v}_{t,j}} \le 1$ whenever gradients remain aligned ($\boldsymbol{g}_i \approx \boldsymbol{g}$), bounding per-coordinate displacement to an approximate trust region $|\Delta\theta_{t,j}| \lesssim \eta_t$ invariant to positive gradient rescaling $\boldsymbol{g}_i \mapsto c \boldsymbol{g}_i$ ($c > 0$).

Although Kingma & Ba (2015) proved $O(1/\sqrt{T})$ regret for Adam in convex optimization, Reddi et al. (2018) constructed a stochastic convex counterexample where exponential forgetting of large rare gradients in $\boldsymbol{v}_t$ allows $\eta_t / \sqrt{\hat{\boldsymbol{v}}_t}$ to increase and diverge [[Source - Understanding Deep Learning|understandingdl_simonprince]] (p. 93) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.6.4, p. 301). Remedies include **AMSGrad** ($\hat{v}_{t,\max} = \max(\hat{v}_{t-1,\max}, \hat{v}_t)$ enforcing non-increasing effective step sizes), **Yogi** ($\boldsymbol{v}_t = \boldsymbol{v}_{t-1} + (1-\beta_2)\mathrm{sign}(\boldsymbol{g}_t^{\odot 2} - \boldsymbol{v}_{t-1}) \odot \boldsymbol{g}_t^{\odot 2}$ preventing excessive growth), and **ADOPT** (preconditioning $\boldsymbol{g}_t$ by $\sqrt{\boldsymbol{v}_{t-1}}+\epsilon$ *before* momentum accumulation) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eqs. (8.82)–(8.83), p. 301).

Equally critical in modern deep learning and LLM pretraining is how $L_2$ regularization interacts with adaptive scaling [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 6.6 & Sec. 9.1, pp. 94, 156) [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 5, pp. 371–372).

> [!important] Theorem: Coupled $L_2$ Regularization vs. Decoupled Weight Decay (AdamW)
> Consider minimizing the $L_2$-penalized objective $\tilde{\mathcal{L}}(\boldsymbol{\theta}) = \mathcal{L}(\boldsymbol{\theta}) + \frac{\lambda}{2}\lVert \boldsymbol{\theta} \rVert_2^2$, whose stochastic gradient is $\tilde{\boldsymbol{g}}_t = \boldsymbol{g}_t + \lambda\boldsymbol{\theta}_{t-1}$.
> - **Exact Equivalence under Vanilla SGD:** With no adaptive preconditioning ($\boldsymbol{P}_t = \boldsymbol{I}_P$), substituting $\tilde{\boldsymbol{g}}_t$ into SGD gives
>
>
> $$
>
>     \boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t(\boldsymbol{g}_t + \lambda\boldsymbol{\theta}_{t-1}) = (1 - \eta_t\lambda)\boldsymbol{\theta}_{t-1} - \eta_t \boldsymbol{g}_t,
>
>
> $$
>
>   so $L_2$ regularization with strength $\lambda$ is algebraically identical to multiplicative weight decay with factor $\lambda_{\mathrm{WD}} = \eta_t\lambda$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (p. 156).
> - **Coupled Distortion under Adam ($L_2$):** Injecting $\tilde{\boldsymbol{g}}_t = \boldsymbol{g}_t + \lambda\boldsymbol{\theta}_{t-1}$ into `(eq:optalg:adam_unroll)` mixes $\lambda\boldsymbol{\theta}_{t-1}$ into *both* $\boldsymbol{m}_t$ and $\boldsymbol{v}_t$, yielding (for $\beta_1=0$ for clarity, or exponentially smoothed for $\beta_1>0$):
>
>
> $$
>
>     \boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t \frac{\hat{\boldsymbol{m}}_t^{\mathcal{L}}}{\sqrt{\hat{\boldsymbol{v}}_t^{\mathrm{coupled}}} + \epsilon}
>     - \eta_t \frac{\lambda_{\mathrm{eff},t}}{\sqrt{\hat{\boldsymbol{v}}_t^{\mathrm{coupled}}} + \epsilon} \odot \boldsymbol{\theta}_{t-1}.
>
>
> $$
>
>   Consequently, parameters with historically large loss gradients (large $\sqrt{\hat{v}_{t,j}}$) receive strongly attenuated decay $\propto \lambda / \sqrt{\hat{v}_{t,j}}$, while flat directions undergo excessive shrinkage, coupling the optimal regularization hyperparameter $\lambda$ tightly to $\eta_t$ and gradient scale [[Source - Understanding Deep Learning|understandingdl_simonprince]] (p. 156).
> - **AdamW Decoupling:** Loshchilov & Hutter (2019) decouple weight decay from gradient moments using pure loss gradients $\boldsymbol{g}_t = \nabla\mathcal{L}_{B_t}(\boldsymbol{\theta}_{t-1})$:
>
>
> $$
> \boldsymbol{\theta}_t = (1 - \eta_t \lambda)\boldsymbol{\theta}_{t-1} - \eta_t \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon},
>
> $$
>
>   restoring uniform exponential decay $(1 - \eta_t\lambda)$ across all weights independently of gradient curvature [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (p. 371) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (p. 94).

### Learning Rate Schedules: Warmup, Cosine Annealing & Gradient Clipping

Because stochastic gradients retain non-zero variance $\sigma_f^*$ at optima, stochastic optimization requires step-size scheduling. Classical Robbins–Monro theory guarantees convergence under $\sum_{t=1}^\infty \eta_t = \infty$ and $\sum_{t=1}^\infty \eta_t^2 < \infty$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Eqs. (8.12)–(8.13), pp. 294–295) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. (6.44), p. 272), instantiated via step decay, polynomial decay $\eta_t = \eta_0(\beta t + 1)^{-\alpha}$, or linear decay [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. (6.45)–(6.47), p. 272).

For deep Transformer and LLM pretraining, [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (App. D, pp. 758–775) and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.3, p. 296) prescribe a three-part stabilization schedule over total steps $T$ with warmup horizon $T_w \in [0.001 T,  0.20 T]$:

$$
\eta_t =
  \begin{cases}
    \eta_{\mathrm{init}} + \dfrac{t}{T_w}(\eta_{\max} - \eta_{\mathrm{init}}), & 0 \le t < T_w \quad\text{(Linear Warmup)}, \\[8pt]
    \eta_{\min} + \dfrac{1}{2}(\eta_{\max} - \eta_{\min})\!\left[1 + \cos\!\left(\pi \dfrac{t - T_w}{T - T_w}\right)\right], & T_w \le t \le T \quad\text{(Cosine Annealing)},
  \end{cases}
$$

paired post-warmup with global $L_2$-norm gradient clipping over the concatenated parameter gradient $\boldsymbol{g}_t \in \mathbb{R}^P$ [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Sec. D.3, pp. 766–770) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.2.4, p. 289):

$$
\boldsymbol{g}_t \leftarrow \min\!\left(1,  \frac{\tau_{\mathrm{clip}}}{\lVert \boldsymbol{g}_t \rVert_2}\right)\boldsymbol{g}_t, \qquad \lVert \boldsymbol{g}_t \rVert_2 = \left(\sum_{k=1}^{L_{\mathrm{layers}}} \lVert \boldsymbol{g}_t^{(k)} \rVert_F^2\right)^{1/2}.
$$

> [!tip] Why Adaptive Moments Need Warmup & Global Clipping
> Two complementary mechanisms explain `(eq:optalg:warmup_cosine)`–`(eq:optalg:grad_clip)`: (i) During the first $t \ll (1-\beta_2)^{-1}$ iterations, Adam's second-moment accumulator $\hat{\boldsymbol{v}}_t$ is estimated from only a handful of mini-batches, producing high-variance step ratios $\hat{\boldsymbol{m}}_t/\sqrt{\hat{\boldsymbol{v}}_t}$ unless $\eta_t$ ramps up smoothly (or Rectified Adam delays adaptive variance scaling) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (p. 93) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (p. 296); (ii) Deep architectures and sequence models exhibit sharp "cliff" nonlinearities where $\lVert \boldsymbol{g}_t \rVert_2$ spikes abruptly, catapulting unclipped iterates out of valid trust regions [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Fig. 8.3, p. 289) [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Sec. D.3, p. 766). Notice that global clipping `(eq:optalg:grad_clip)` preserves the exact descent direction $-\boldsymbol{g}_t/\lVert \boldsymbol{g}_t \rVert_2$ across all parameter tensors while enforcing $\lVert \boldsymbol{g}_t \rVert_2 \le \tau_{\mathrm{clip}}$ (typically $\tau_{\mathrm{clip}} = 1.0$).

### Second-Order, Quasi-Newton & Information-Geometric Methods

Diagonal preconditioners cannot decouple rotated, off-diagonal parameter correlations [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.6.5, p. 301). Full curvature methods replace $\boldsymbol{P}_t = \boldsymbol{I}$ with structured $P \times P$ curvature matrices.

**Newton's Method & Tikhonov Trust-Region Damping..** 
Minimizing the second-order Taylor model $M_t(\boldsymbol{\delta}) = \mathcal{L}(\boldsymbol{\theta}_t) + \boldsymbol{g}_t^{\top}\boldsymbol{\delta} + \frac{1}{2}\boldsymbol{\delta}^{\top} \boldsymbol{H}_t\boldsymbol{\delta}$ over $\boldsymbol{\delta} = \boldsymbol{\theta} - \boldsymbol{\theta}_t$ yields Newton's step $\boldsymbol{\delta}_t = -\boldsymbol{H}_t^{-1}\boldsymbol{g}_t$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Eqs. (8.26)–(8.27), p. 311) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eqs. (8.39)–(8.42), p. 289). On strictly positive-definite quadratics, Newton reaches $\boldsymbol{\theta}^*$ in a single step; near strict non-degenerate local minima, it achieves local quadratic convergence $\lVert \boldsymbol{\theta}_{t+1}-\boldsymbol{\theta}^* \rVert_2 \le C\lVert \boldsymbol{\theta}_t-\boldsymbol{\theta}^* \rVert_2^2$. Near saddle points where $\boldsymbol{H}_t$ is indefinite, constraining $\lVert \boldsymbol{\delta} \rVert_2 \le r_t$ via a quadratic trust region produces the Levenberg–Marquardt / Tikhonov-damped step $\boldsymbol{\delta}_t = -(\boldsymbol{H}_t + \lambda_{\mathrm{LM}}\boldsymbol{I}_P)^{-1} \boldsymbol{g}_t$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Eq. (8.28), p. 312) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eqs. (8.51)–(8.52), p. 292). Because forming and inverting $\boldsymbol{H}_t \in \mathbb{R}^{P \times P}$ costs $O(P^2)$ memory and $O(P^3)$ time, exact Newton is intractable for large $P$.

> [!important] Derivation: BFGS Secant Update, Inverse Hessian & Two-Loop L-BFGS
> Quasi-Newton methods maintain a symmetric positive-definite Hessian approximation $\boldsymbol{B}_t \approx \boldsymbol{H}_t$ (or inverse $\boldsymbol{C}_t = \boldsymbol{B}_t^{-1} \approx \boldsymbol{H}_t^{-1}$) using only first-order differences $\boldsymbol{s}_t = \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t$ and $\boldsymbol{y}_t = \boldsymbol{g}_{t+1} - \boldsymbol{g}_t$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.3.2, pp. 290–291) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.6.3, pp. 316–317). By the fundamental theorem of calculus, $\boldsymbol{y}_t = \left(\int_0^1 \nabla^2\mathcal{L}(\boldsymbol{\theta}_t + \tau\boldsymbol{s}_t) \mathrm{d}\tau\right)\boldsymbol{s}_t$, motivating the *secant equations* $\boldsymbol{B}_{t+1}\boldsymbol{s}_t = \boldsymbol{y}_t$ and $\boldsymbol{C}_{t+1}\boldsymbol{y}_t = \boldsymbol{s}_t$.
>
> Finding the closest $\boldsymbol{C}_{t+1} \in \mathbb{S}_{++}^P$ to $\boldsymbol{C}_t$ in weighted Frobenius norm subject to $\boldsymbol{C}_{t+1}\boldsymbol{y}_t = \boldsymbol{s}_t$ yields the rank-two **BFGS inverse Hessian update** ($\rho_t \triangleq (\boldsymbol{y}_t^{\top} \boldsymbol{s}_t)^{-1}$):
>
> $$
> \boldsymbol{C}_{t+1} = \boldsymbol{V}_t^{\top} \boldsymbol{C}_t \boldsymbol{V}_t + \rho_t \boldsymbol{s}_t \boldsymbol{s}_t^{\top}, \qquad \boldsymbol{V}_t \triangleq \boldsymbol{I}_P - \rho_t \boldsymbol{y}_t \boldsymbol{s}_t^{\top},
>
> $$
>
> while Sherman–Morrison–Woodbury inversion gives the primal Hessian update [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eqs. (8.44), (8.48), pp. 290–291):
>
> $$
> \boldsymbol{B}_{t+1} = \boldsymbol{B}_t + \frac{\boldsymbol{y}_t \boldsymbol{y}_t^{\top}}{\boldsymbol{y}_t^{\top} \boldsymbol{s}_t} - \frac{(\boldsymbol{B}_t\boldsymbol{s}_t)(\boldsymbol{B}_t\boldsymbol{s}_t)^{\top}}{\boldsymbol{s}_t^{\top} \boldsymbol{B}_t \boldsymbol{s}_t}.
>
> $$
>
> **Positive-Definiteness Preservation:** If $\boldsymbol{C}_t \succ \boldsymbol{0}$ and a Wolfe line search guarantees the curvature condition $\boldsymbol{y}_t^{\top} \boldsymbol{s}_t > 0$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. (8.47), p. 291), then for any nonzero $\boldsymbol{u} \in \mathbb{R}^P$,
>
> $$
>
>   \boldsymbol{u}^{\top} \boldsymbol{C}_{t+1}\boldsymbol{u} = (\boldsymbol{V}_t\boldsymbol{u})^{\top} \boldsymbol{C}_t (\boldsymbol{V}_t\boldsymbol{u}) + \rho_t (\boldsymbol{s}_t^{\top} \boldsymbol{u})^2 > 0,
>
> $$
>
> since $\boldsymbol{V}_t\boldsymbol{u} = \boldsymbol{0} \implies \boldsymbol{u} = \rho_t(\boldsymbol{s}_t^{\top}\boldsymbol{u})\boldsymbol{y}_t \neq \boldsymbol{0} \implies \boldsymbol{s}_t^{\top}\boldsymbol{u} \neq 0$, so both terms cannot vanish simultaneously. Hence $\boldsymbol{C}_{t+1} \succ \boldsymbol{0}$ inductively!
>
> **Two-Loop L-BFGS Compression:** Rather than storing $\boldsymbol{C}_t \in \mathbb{R}^{P \times P}$ ($O(P^2)$), L-BFGS retains only the most recent $M$ curvature pairs $\{(\boldsymbol{s}_i, \boldsymbol{y}_i)\}_{i=t-M}^{t-1}$ ($M \approx 5$–$20$) and unrolls `(eq:optalg:bfgs_inv)` from a scaled diagonal seed $\boldsymbol{C}_t^0 = \gamma_t \boldsymbol{I}_P$, $\gamma_t = \frac{\boldsymbol{s}_{t-1}^{\top}\boldsymbol{y}_{t-1}}{\boldsymbol{y}_{t-1}^{\top}\boldsymbol{y}_{t-1}}$:
>
> $$
> \boldsymbol{C}_t = (\boldsymbol{V}_{t-1}^{\top} \cdots \boldsymbol{V}_{t-M}^{\top})\boldsymbol{C}_t^0(\boldsymbol{V}_{t-M}\cdots\boldsymbol{V}_{t-1})
>   + \sum_{k=t-M}^{t-1} \rho_k (\boldsymbol{V}_{t-1}^{\top}\cdots\boldsymbol{V}_{k+1}^{\top})\boldsymbol{s}_k\boldsymbol{s}_k^{\top}(\boldsymbol{V}_{k+1}\cdots\boldsymbol{V}_{t-1}).
>
> $$
>
> Applying $\boldsymbol{C}_t$ to $\boldsymbol{g}_t$ via a backward pass (computing $\boldsymbol{q} \leftarrow \boldsymbol{V}_{t-M}\cdots\boldsymbol{V}_{t-1}\boldsymbol{g}_t$ and scalars $\alpha_i = \rho_i \boldsymbol{s}_i^{\top}\boldsymbol{q}$) followed by $\boldsymbol{r} \leftarrow \boldsymbol{C}_t^0\boldsymbol{q}$ and a forward pass (accumulating $\boldsymbol{V}_i^{\top}$ products and $(\alpha_i - \rho_i\boldsymbol{y}_i^{\top}\boldsymbol{r})\boldsymbol{s}_i$) yields the exact quasi-Newton direction $\boldsymbol{d}_t = -\boldsymbol{C}_t\boldsymbol{g}_t$ in $O(MP)$ memory and $4MP + O(P)$ FLOPs [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.3.2, p. 291) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.6.3, p. 317).

**Generalized Gauss–Newton (GGN) & Natural Gradient Descent (NGD/K-FAC)..** 
For probabilistic prediction models $p_{\boldsymbol{\theta}}(y\mid\boldsymbol{x}) = p(y\mid f(\boldsymbol{x};\boldsymbol{\theta}))$ minimizing negative log-likelihood $\mathcal{L}(\boldsymbol{\theta}) = -\mathbb{E}_{\hat{p}}[\log p_{\boldsymbol{\theta}}(y\mid\boldsymbol{x})]$, second-order curvature admits two closely allied positive-semidefinite factorizations [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 3.3.4, 5.1.9, 6.4, 17.3.2).

> [!important] Theorem: Natural Gradient Trust Region, Invariance, GGN & K-FAC
> Let $p_{\boldsymbol{\theta}}(y\mid\boldsymbol{x})$ be a conditional model with score $\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x},y) = \nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(y\mid\boldsymbol{x})$ and averaged **Fisher Information Matrix (FIM)** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. (3.40), (6.83)–(6.84), pp. 76, 280):
>
> $$
> \boldsymbol{F}(\boldsymbol{\theta}) \triangleq \mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x})}\mathbb{E}_{p_{\boldsymbol{\theta}}(y\mid\boldsymbol{x})}\!\left[\boldsymbol{s}_{\boldsymbol{\theta}}\boldsymbol{s}_{\boldsymbol{\theta}}^{\top}\right]
>   = -\mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x})}\mathbb{E}_{p_{\boldsymbol{\theta}}(y\mid\boldsymbol{x})}\!\left[\nabla^2_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(y\mid\boldsymbol{x})\right] \succeq \boldsymbol{0}.
>
> $$
> - **KL Trust-Region Derivation:** Second-order Taylor expansion of the expected KL divergence (using $\mathbb{E}_{p_{\boldsymbol{\theta}}}[\boldsymbol{s}_{\boldsymbol{\theta}}] = \boldsymbol{0}$) yields $\mathbb{E}_{p_{\mathcal{D}}}[D_{\mathrm{KL}}(p_{\boldsymbol{\theta}} \parallel p_{\boldsymbol{\theta}+\boldsymbol{\delta}})] = \frac{1}{2}\boldsymbol{\delta}^{\top} \boldsymbol{F}(\boldsymbol{\theta})\boldsymbol{\delta} + O(\lVert \boldsymbol{\delta} \rVert_2^3)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. (5.81), p. 233; Eq. (6.82), p. 280). Minimizing the linearized objective subject to a KL penalty $\min_{\boldsymbol{\delta}} \boldsymbol{g}_t^{\top}\boldsymbol{\delta} + \frac{1}{2\eta_t}\boldsymbol{\delta}^{\top}\boldsymbol{F}(\boldsymbol{\theta}_t)\boldsymbol{\delta}$ gives the **Natural Gradient** update [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. (6.85)–(6.90), p. 281):
>
>
> $$
> \tilde{\nabla}\mathcal{L}(\boldsymbol{\theta}_t) \triangleq \boldsymbol{F}(\boldsymbol{\theta}_t)^{-1} \nabla\mathcal{L}(\boldsymbol{\theta}_t), \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \tilde{\nabla}\mathcal{L}(\boldsymbol{\theta}_t).
>
> $$
> - **Reparameterization Invariance:** Let $\boldsymbol{\theta} = \psi(\boldsymbol{\xi})$ be any smooth diffeomorphism with Jacobian $\boldsymbol{J} = \frac{\partial \boldsymbol{\theta}}{\partial \boldsymbol{\xi}}$. By the chain rule, $\nabla_{\boldsymbol{\xi}}\log p = \boldsymbol{J}^{\top}\nabla_{\boldsymbol{\theta}}\log p$ and $\nabla_{\boldsymbol{\xi}}\mathcal{L} = \boldsymbol{J}^{\top}\nabla_{\boldsymbol{\theta}}\mathcal{L}$, so $\boldsymbol{F}_{\boldsymbol{\xi}} = \boldsymbol{J}^{\top}\boldsymbol{F}_{\boldsymbol{\theta}}\boldsymbol{J}$. Consequently, for small $\eta_t$, the induced step in $\boldsymbol{\theta}$-space is invariant to parameterization:
>
>
> $$
>
>     \Delta\boldsymbol{\theta} \approx \boldsymbol{J}\Delta\boldsymbol{\xi}
>     = -\eta_t \boldsymbol{J}(\boldsymbol{J}^{\top} \boldsymbol{F}_{\boldsymbol{\theta}}\boldsymbol{J})^{-1} (\boldsymbol{J}^{\top} \nabla_{\boldsymbol{\theta}}\mathcal{L})
>     = -\eta_t \boldsymbol{F}_{\boldsymbol{\theta}}^{-1} \nabla_{\boldsymbol{\theta}}\mathcal{L}.
>
>
> $$
>
>   In exponential families with natural parameters $\boldsymbol{\lambda}$, cumulant $A(\boldsymbol{\lambda})$, and moment parameters $\boldsymbol{\mu} = \nabla_{\boldsymbol{\lambda}}A(\boldsymbol{\lambda})$, $\boldsymbol{F}(\boldsymbol{\lambda}) = \nabla^2_{\boldsymbol{\lambda}} A(\boldsymbol{\lambda}) = \frac{\partial \boldsymbol{\mu}}{\partial \boldsymbol{\lambda}}$, giving the remarkable duality $\tilde{\nabla}_{\boldsymbol{\lambda}}L(\boldsymbol{\lambda}) = \boldsymbol{F}(\boldsymbol{\lambda})^{-1}\nabla_{\boldsymbol{\lambda}}L(\boldsymbol{\lambda}) = \nabla_{\boldsymbol{\mu}}L(\boldsymbol{\mu})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. (6.99)–(6.101), p. 284).
> - **Generalized Gauss–Newton (GGN) & Equivalence:** Let $f(\boldsymbol{x};\boldsymbol{\theta}) \in \mathbb{R}^C$ denote network logits, $\boldsymbol{J}_{\boldsymbol{\theta}}(\boldsymbol{x}) = \frac{\partial f}{\partial \boldsymbol{\theta}} \in \mathbb{R}^{C \times P}$ the network Jacobian, $\boldsymbol{r} = \nabla_f \log p(y\mid f)$, and $\boldsymbol{\Lambda} = -\nabla^2_f \log p(y\mid f) \succeq \boldsymbol{0}$. Differentiating `(eq:optalg:fim)` gives the exact NLL Hessian [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. (17.14)–(17.15), pp. 652–653):
>
>
> $$
> \nabla^2_{\boldsymbol{\theta}}\bigl(-\log p(y\mid f)\bigr) = \boldsymbol{J}_{\boldsymbol{\theta}}^{\top} \boldsymbol{\Lambda} \boldsymbol{J}_{\boldsymbol{\theta}} - \sum_{c=1}^C r_c \nabla^2_{\boldsymbol{\theta}} f_c
>     \quad \xRightarrow{\text{drop }\nabla^2 f_c}\quad
>     \boldsymbol{G}(\boldsymbol{\theta}) = \mathbb{E}_{\hat{p}}\!\left[\boldsymbol{J}_{\boldsymbol{\theta}}^{\top} \boldsymbol{\Lambda} \boldsymbol{J}_{\boldsymbol{\theta}}\right] \succeq \boldsymbol{0}.
>
> $$
>
>   Because $\boldsymbol{\Lambda}$ is the output Fisher of an exponential family loss (e.g., $\boldsymbol{I}_C$ for Gaussian MSE, $\mathrm{diag}(\boldsymbol{\pi})-\boldsymbol{\pi}\boldsymbol{\pi}^{\top}$ for softmax cross-entropy), $\boldsymbol{G}(\boldsymbol{\theta})$ equals $\boldsymbol{F}(\boldsymbol{\theta})$ identically and stays guaranteed positive semidefinite even at non-convex saddles [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.4.2.2, p. 281; Sec. 17.3.2, p. 653).
> - **K-FAC Kronecker Factorization:** For a linear layer $\boldsymbol{s}_\ell = \boldsymbol{W}_\ell \boldsymbol{a}_{\ell-1}$ with vectorized weight gradient $\mathrm{vec}(\nabla_{\boldsymbol{W}_\ell}\log p) = \boldsymbol{g}_{\boldsymbol{s}_\ell} \otimes \boldsymbol{a}_{\ell-1}$, assuming approximate independence between forward activations $\boldsymbol{a}_{\ell-1} \in \mathbb{R}^{d_{\mathrm{in}}}$ and backpropagated pre-activation scores $\boldsymbol{g}_{\boldsymbol{s}_\ell} \in \mathbb{R}^{d_{\mathrm{out}}}$ yields the block-diagonal Kronecker approximation [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.4.4, p. 283) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 17.3.2, p. 653):
>
>
> $$
> \begin{aligned}
>     \boldsymbol{F}_\ell &\approx \mathbb{E}[\boldsymbol{a}_{\ell-1}\boldsymbol{a}_{\ell-1}^{\top}] \otimes \mathbb{E}[\boldsymbol{g}_{\boldsymbol{s}_\ell}\boldsymbol{g}_{\boldsymbol{s}_\ell}^{\top}] = \boldsymbol{A}_{\ell-1} \otimes \boldsymbol{G}_\ell, \\
>     \boldsymbol{F}_\ell^{-1} \mathrm{vec}(\nabla_{\boldsymbol{W}_\ell}\mathcal{L}) &\approx \mathrm{vec}\!\left(\boldsymbol{G}_\ell^{-1} (\nabla_{\boldsymbol{W}_\ell}\mathcal{L}) \boldsymbol{A}_{\ell-1}^{-1}\right),
>   \end{aligned}
>
> $$
>
>   reducing per-layer inversion complexity from $O((d_{\mathrm{in}}d_{\mathrm{out}})^3)$ to $O(d_{\mathrm{in}}^3 + d_{\mathrm{out}}^3)$.

> [!tip] Riemannian Whitening in Distribution Space
> Figure `fig:optalg:murphy_ngd` illustrates why NGD outperforms steepest descent on curved statistical manifolds [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.4.2–6.4.3, pp. 281–283): Euclidean balls in $\boldsymbol{\theta}$-space correspond to highly eccentric KL ellipsoids whenever parameters interact (such as $\mu$ and $\sigma$ in a Gaussian, where shifting $\mu$ by $\delta$ matters vastly more when $\sigma$ is small [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 6.5, p. 280)). Multiplying by $\boldsymbol{F}^{-1}$ is algebraically equivalent to performing ordinary gradient descent in the locally whitened coordinates $\boldsymbol{\phi} = \boldsymbol{F}^{1/2}\boldsymbol{\theta}$, restoring spherical gradient fields and escaping flat plateaus where Euclidean gradients vanish.

## Algorithmic Implementation & Comparative Synthesis

Modern large-scale training pipelines pair AdamW with linear warmup, cosine decay, and post-warmup global $L_2$ norm clipping in Algorithm `alg:optalg:adamw`, synthesizing the schedules of [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Sec. D.1–D.3, pp. 760–770), the bias-corrected moments of [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Alg. 8.7, p. 311), and the decoupled decay rule of [[Source - Understanding Deep Learning|understandingdl_simonprince]] (p. 156); by contrast, convex linear probes, physics-informed nets, and small-batch scientific fitting routinely deploy Two-Loop L-BFGS via Algorithm `alg:optalg:lbfgs` [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.3.2, pp. 290–291).

> [!example] Algorithm: Decoupled AdamW with Linear Warmup, Cosine Decay & Global Norm Clipping
> **Input:** Initial weights $\boldsymbol{\theta}_0 \in \mathbb{R}^P$; schedule $(\eta_{\mathrm{init}}, \eta_{\max}, \eta_{\min}, T_w, T)$; decay $\lambda$; $(\beta_1, \beta_2, \epsilon)$; clip $\tau_{\mathrm{clip}}$.
> **Output:** Trained parameters $\boldsymbol{\theta}_T$.
> Initialize $\boldsymbol{m}_0 \leftarrow \boldsymbol{0}$, $\boldsymbol{v}_0 \leftarrow \boldsymbol{0}$
> **for** $t = 1$ to $T$:
>
>   Sample mini-batch $B_t$ and compute pure loss gradient $\boldsymbol{g}_t \leftarrow \frac{1}{|B_t|}\sum_{i \in B_t} \nabla_{\boldsymbol{\theta}}\ell_i(\boldsymbol{\theta}_{t-1})$
>   **if** $t \le T_w$:
>
>     $\eta_t \leftarrow \eta_{\mathrm{init}} + \frac{t - 1}{T_w}(\eta_{\max} - \eta_{\mathrm{init}})$  *// Linear warmup*
>
> **else**:
>
>     $\eta_t \leftarrow \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\bigl(1 + \cos(\pi\frac{t - 1 - T_w}{T - T_w})\bigr)$  *// Cosine annealing*
>     **if** $\lVert \boldsymbol{g}_t \rVert_2 > \tau_{\mathrm{clip}}$:
>
>       $\boldsymbol{g}_t \leftarrow \frac{\tau_{\mathrm{clip}}}{\lVert \boldsymbol{g}_t \rVert_2}\boldsymbol{g}_t$  *// Global $L_2$ norm clip*
>
>
>   $\boldsymbol{m}_t \leftarrow \beta_1 \boldsymbol{m}_{t-1} + (1 - \beta_1)\boldsymbol{g}_t$; \quad $\boldsymbol{v}_t \leftarrow \beta_2 \boldsymbol{v}_{t-1} + (1 - \beta_2)(\boldsymbol{g}_t \odot \boldsymbol{g}_t)$
>   $\hat{\boldsymbol{m}}_t \leftarrow \boldsymbol{m}_t / (1 - \beta_1^t)$; \quad $\hat{\boldsymbol{v}}_t \leftarrow \boldsymbol{v}_t / (1 - \beta_2^t)$  *// Bias correction*
>   $\boldsymbol{\theta}_t \leftarrow (1 - \eta_t \lambda)\boldsymbol{\theta}_{t-1} - \eta_t \hat{\boldsymbol{m}}_t / \bigl(\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon\bigr)$  *// Decoupled weight decay*
>
> **return** $\boldsymbol{\theta}_T$

> [!example] Algorithm: Two-Loop L-BFGS Direction Computation with Curvature Guard
> **Input:** Current gradient $\boldsymbol{g}_t \in \mathbb{R}^P$; history buffer $\mathcal{H}_t = \{(\boldsymbol{s}_i, \boldsymbol{y}_i)\}_{i=\max(0, t-M)}^{t-1}$ with $\rho_i = (\boldsymbol{y}_i^{\top}\boldsymbol{s}_i)^{-1} > 0$.
> **Output:** Quasi-Newton descent direction $\boldsymbol{d}_t = -\boldsymbol{C}_t \boldsymbol{g}_t$.
> $\boldsymbol{q} \leftarrow \boldsymbol{g}_t$
> **for** $i = t-1$ to $\max(0, t-M)$:
>
>   $\alpha_i \leftarrow \rho_i (\boldsymbol{s}_i^{\top} \boldsymbol{q})$; \quad $\boldsymbol{q} \leftarrow \boldsymbol{q} - \alpha_i \boldsymbol{y}_i$  *// Backward loop applying $\boldsymbol{V}_i$*
>
> **if** $t > 0$:
>
>   $\gamma_t \leftarrow \frac{\boldsymbol{s}_{t-1}^{\top} \boldsymbol{y}_{t-1}}{\boldsymbol{y}_{t-1}^{\top} \boldsymbol{y}_{t-1}}$; \quad $\boldsymbol{r} \leftarrow \gamma_t \boldsymbol{q}$  *// Initial Hessian scale $\boldsymbol{C}_t^0 = \gamma_t \boldsymbol{I}_P$*
>
> **else**:
>
>   $\boldsymbol{r} \leftarrow \boldsymbol{q}$
>
> **for** $i = \max(0, t-M)$ to $t-1$:
>
>   $\beta_i \leftarrow \rho_i (\boldsymbol{y}_i^{\top} \boldsymbol{r})$; \quad $\boldsymbol{r} \leftarrow \boldsymbol{r} + (\alpha_i - \beta_i)\boldsymbol{s}_i$  *// Forward loop reconstructing $-\boldsymbol{d}_t$*
>
> **return** $\boldsymbol{d}_t \leftarrow -\boldsymbol{r}$  *// Use Wolfe line search; skip pair if $\boldsymbol{y}_t^{\top}\boldsymbol{s}_t \le \varepsilon_{\mathrm{curv}}\lVert \boldsymbol{s}_t \rVert\lVert \boldsymbol{y}_t \rVert$*

Table `tab:optalg:complexity` synthesizes the computational cost, optimizer state overhead, and convergence geometry across all optimizer classes.

**Table:** Computational Complexity, Memory Footprint ($P$ Parameters), and Convergence Comparison.

| **Optimizer** | **Preconditioner $\boldsymbol{P}_t$** | **Step Time** | **State Mem.** | **Convex Rate / Key Trade-Off** |
| — | — | — | — | — |
| SGD (Fixed/Decay) | $\boldsymbol{I}_P$ | $O(P)$ | $0P$ (stateless) | $O(1/T)$ smooth, $(1-\mu/L)^T$ strongly convex [[Source - Handbook of Convergence Theorems]] (Thms. 3.4–3.6); flat minima generalization. |
| Heavy-Ball / NAG | $\boldsymbol{I}_P$ (history / lookahead) | $O(P)$ | $1P$ ($\boldsymbol{m}_t$) | $O(1/T^2)$ convex (NAG); $(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1})^{2T}$ quadratic (HB) [[Source - Deep Learning]] (Sec. 8.3). |
| AdaGrad | $\mathrm{diag}(\sqrt{\sum \boldsymbol{g}_i^2}+\epsilon)$ | $O(P)$ | $1P$ ($\boldsymbol{v}_t$) | $O(1/\sqrt{T})$ convex regret; stalls prematurely in non-convex deep nets [[Source - Deep Learning]] (Sec. 8.5.1). |
| RMSProp | $\mathrm{diag}(\sqrt{\boldsymbol{v}_t}+\epsilon)$ | $O(P)$ | $1P$ ($\boldsymbol{v}_t$) | Adapts to local basin curvature; lacks bias correction and momentum [[Source - Deep Learning]] (Sec. 8.5.2). |
| Adam / AdamW | $\mathrm{diag}(\sqrt{\hat{\boldsymbol{v}}_t}+\epsilon)$ | $O(P)$ | $2P$ ($\boldsymbol{m}_t,\boldsymbol{v}_t$);  $12P$ B FP32 | Robust across Transformer depths; AdamW decouples $\lambda\boldsymbol{\theta}$ [[Source - Understanding Deep Learning]] (Sec. 6.4, 9.1) [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]] (p. 371). |
| L-BFGS ($M$ pairs) | Secant low-rank $\boldsymbol{C}_t$ | $O(MP)$ | $2MP$ ($\boldsymbol{s}_i,\boldsymbol{y}_i$) | Superlinear (batch); sensitive to mini-batch noise, needs line search [[Source - Probabilistic Machine Learning]] (Sec. 8.3.2). |
| Newton / LM | $\boldsymbol{H}_t + \lambda_{\mathrm{LM}}\boldsymbol{I}_P$ | $O(P^3)$ | $O(P^2)$ | Quadratic local rate; intractable for $P \gg 10^4$, needs damping [[Source - Deep Learning]] (Sec. 8.6.1). |
| K-FAC (Block NGD) | $\bigoplus_\ell (\boldsymbol{A}_{\ell-1}\otimes \boldsymbol{G}_\ell)$ | $O(\sum_\ell d_{\mathrm{in}}^3$  $+ d_{\mathrm{out}}^3)$ | $O(\sum_\ell d_{\mathrm{in}}^2$  $+ d_{\mathrm{out}}^2)$ | KL-invariant, escapes plateaus; requires amortized block inversions [[Source - Probabilistic Machine Learning]] (Sec. 6.4.4). |

> [!info] Cross-Book Synthesis: Generalization, Robustness & Curvature Scaling
> Synthesizing all six textbooks resolves a long-standing pedagogical tension between SGD+Momentum and Adam/AdamW:
> - **Ravine Robustness vs. Generalization:** [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 6.6, p. 94) summarizes Wilson et al. (2017)'s observation that SGD+Momentum can achieve lower test error than vanilla Adam with $L_2$ coupling, alongside Choi et al. (2019) showing properly tuned Adam/AdamW matches SGD generalization while converging faster, and Keskar & Socher (2017)'s SWATS switching from Adam to SGD. In deep Transformers, [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (pp. 371–372, 758–775) adopts AdamW + linear warmup + cosine decay + gradient clipping because layer-wise gradient magnitudes vary sharply across attention, LayerNorm, and embedding matrices [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 6.4, p. 91).
> - **True Curvature vs. Diagonal Scaling:** Textbook framings separate optimization along two complementary axes: Goodfellow et al. treat AdaGrad, RMSProp, and Adam as axis-aligned heuristics alongside batch BFGS and conjugate-gradient methods [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.5–8.6), whereas Murphy unifies adaptive and second-order optimizers via preconditioning [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.4): Adam acts as a diagonal smoothed square-root empirical Fisher, while K-FAC and Shampoo ([[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.6.5)) capture off-diagonal layer-wise Kronecker couplings at tractable cost.

## Visual Diagnostics & Geometric Structure

Figure `fig:optalg:taxonomy` maps the mathematical preconditioning hierarchy linking isotropic first-order, diagonal adaptive, and structured curvature/natural-gradient methods, alongside deep-network stabilizers.

> [!example] Visual Diagram (LaTeX / PDF)
> *Taxonomy and preconditioning hierarchy uniting first-order momentum, diagonal adaptive moments, and structured second-order / natural-gradient optimizers.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:optalg:dynamics` quantifies two pivotal empirical dynamics derived in Section `sec:optalg:theory`: (a) on an ill-conditioned quadratic ($\kappa = 100$), optimal Polyak Heavy-Ball exhibits a $13.6\times$ transient overshoot from non-normal companion dynamics before locking onto the $(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1})^{2t} = (9/11)^{2t}$ slope, whereas Nesterov lookahead damps the steep mode on step 1; (b) under standard Adam ($\beta_1=0.9, \beta_2=0.999$), the second-moment bias factor $1 - \beta_2^t$ stays far below unity throughout a $10\%$ warmup window ($0.095$ at $t=100$) and even at $t=1000$ ($0.632$), underscoring why bias correction and warmup must act together.

> [!example] Visual Diagram (LaTeX / PDF)
> *Quantitative optimization dynamics. **(a)** Exact suboptimality trajectories on $f(\boldsymbol{\theta}) = \frac{1}{2}(\theta_1^2 + 100\theta_2^2)$ ($\kappa=100$, $\boldsymbol{\theta}_0=[1,1]^{\top}$): Heavy-Ball suffers transient non-normal overshoot before achieving the $(9/11)^{2t}$ slope, whereas Nesterov lookahead damps the steep direction immediately. **(b)** Normalized linear-warmup + cosine-annealing schedule `(eq:optalg:warmup_cosine)` ($\eta_{\mathrm{init}}/\eta_{\max}=0.05$, $\eta_{\min}/\eta_{\max}=0.1$) juxtaposed with Adam's geometric bias factors $1-\beta_1^t$ and $1-\beta_2^t$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

The three source textbook visualizations below corroborate these geometric regimes directly on stochastic multi-valley, anisotropic ravine, and probabilistic manifold landscapes. Figure `fig:optalg:prince_momentum` contrasts noisy zig-zagging in vanilla SGD against momentum-smoothed traversal across narrow basins [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Fig. 6.7, p. 87). Figure `fig:optalg:prince_adam` isolates why isotropic step sizes fail on ill-conditioned ravines ($\alpha=0.05$ stalls horizontally; $\alpha=1.0$ oscillates vertically), why unsmoothed sign normalization chatters at the valley floor, and how Adam's dual EWMA moments smoothly track the minimum [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Fig. 6.9, p. 89). Finally, Figure `fig:optalg:murphy_ngd` demonstrates Fisher whitening in Natural Gradient Descent: transforming raw coupled coordinates $\boldsymbol{\theta}$ into $\boldsymbol{\phi} = \boldsymbol{F}^{1/2}\boldsymbol{\theta}$ turns skewed trajectories into direct radial descent and collapses KL divergence orders of magnitude faster [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 6.6, p. 283).

> [!note] Source Figure
> *Figure:* Stochastic gradient descent trajectories on a multi-valley non-convex loss surface [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Fig. 6.7, p. 87 (PDF p. 101)). **(a)** Unsmoothed mini-batch SGD oscillates noisily across valley walls; **(b)** Momentum smoothing averages out orthogonal batch noise and accelerates along consistent downhill directions. (`understandingdl_simonprince_p101_fig1.png`)

> [!note] Source Figure
> *Figure:* Ravine geometry under adaptive preconditioning [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Fig. 6.9, p. 89 (PDF p. 103)). **(a)** Small fixed step ($\alpha=0.05$) crawls along the shallow horizontal axis; **(b)** Large step ($\alpha=1.0$) diverges/oscillates across the steep vertical wall; **(c)** Unsmoothed normalized (sign) gradients advance diagonally but chatter around the minimum; **(d)** Adam ($\alpha=0.05,\beta_1=0.9,\beta_2=0.99$) smooths both moments to converge smoothly. (`understandingdl_simonprince_p103_fig1.png`)

> [!note] Source Figure
> *Figure:* Natural Gradient Descent versus Euclidean steepest descent on a coupled 2D Gaussian cross-entropy objective [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 6.6, p. 283 (PDF p. 317)). **(a)** Parameter-space trajectories from $(1,-1)$ to $(0,0)$; **(b)** KL divergence vs. update steps showing rapid collapse under NGD; **(c)** Skewed Euclidean gradient field in $\boldsymbol{\theta}$; **(d)** Isotropic whitened gradient field in $\boldsymbol{\phi} = \boldsymbol{F}^{1/2}\boldsymbol{\theta}$. (`probabilisticml_kevinmurphy_advancedtopics_p317_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Pitfall 1: $L_2$ Regularization Coupling Suppresses Weight Decay in Adam
> **Failure Mode:** Passing `weight\_decay` through the loss penalty $\frac{\lambda}{2}\lVert \boldsymbol{\theta} \rVert_2^2$ inside standard Adam multiplies the decay force on coordinate $j$ by $1/(\sqrt{\hat{v}_{t,j}} + \epsilon)$. Parameters with large gradient variance (e.g., dense attention projections or early layers) experience severely weakened shrinkage, inflating weight norms and degrading generalization [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 6.6 & Sec. 9.1, pp. 94, 156).\\
> **Remedy:** Always use decoupled weight decay (AdamW, `(eq:optalg:adamw_update)`), and exclude 1D scale/bias parameters (LayerNorm/RMSNorm $\boldsymbol{\gamma},\boldsymbol{\beta}$) from $\lambda > 0$ so normalization invariances are not artificially shrunk [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (pp. 371–372).

> [!warning] Pitfall 2: $\epsilon$-Denominator Masking & Underflow in FP16/BF16 Mixed Precision
> **Failure Mode:** Standard references specify different default Adam stabilizers: $\delta = 10^{-8}$ in [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Alg. 8.7, p. 311) versus $\epsilon = 10^{-6}$ in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.6.3, p. 300). In IEEE FP16 (smallest positive normal number $2^{-14} \approx 6.1\times 10^{-5}$, subnormal underflow below $5.96\times 10^{-8}$), small gradients $|g_{t,j}| < 10^{-4}$ cause $g_{t,j}^2$ to flush to zero in half precision. Conversely, raising $\epsilon$ aggressively (e.g., $\epsilon = 10^{-4}$–$10^{-2}$) on coordinates where $\sqrt{\hat{v}_{t,j}} \ll \epsilon$ converts Adam's update into un-normalized momentum $-\frac{\eta_t}{\epsilon}\hat{\boldsymbol{m}}_t$, destroying adaptive scale invariance on small-gradient embeddings.\\
> **Remedy:** Maintain optimizer accumulators $(\boldsymbol{m}_t,\boldsymbol{v}_t)$ and master parameters strictly in FP32 even when forward/backward passes run in BF16/FP16, and tune $\epsilon \in [10^{-8}, 10^{-6}]$ only after verifying typical $\sqrt{\hat{v}_{t,j}}$ magnitudes across layers.

> [!warning] Pitfall 3: Indefinite Hessian Saddle Attraction & Empirical Fisher Collapse
> **Failure Mode:** Because exact Newton solves $\nabla\mathcal{L} + \boldsymbol{H}\boldsymbol{\delta} = \boldsymbol{0}$ for *any* stationary point, negative eigenvalues $\lambda_i(\boldsymbol{H}_t) < 0$ turn $-\boldsymbol{H}_t^{-1}\boldsymbol{g}_t$ into an *ascent* direction along concave ridges, pulling iterates toward high-cost saddles or maxima [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.2.3 & Sec. 8.6.1, pp. 288, 312) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 8.14b, p. 290). Separately, approximating the true FIM `(eq:optalg:fim)` with the empirical Fisher $\hat{\boldsymbol{F}} = \frac{1}{|B|}\sum_{n\in B}\nabla\log p(y_n\mid\boldsymbol{x}_n,\boldsymbol{\theta})\nabla\log p(y_n\mid\boldsymbol{x}_n,\boldsymbol{\theta})^{\top}$ fails on flat plateaus: as training loss gradients vanish, $\hat{\boldsymbol{F}} \to \boldsymbol{0}$ becomes singular and stalls progress [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.4.4, Eqs. (6.95)–(6.97), p. 284).\\
> **Remedy:** Regularize Newton via Levenberg–Marquardt damping $(\boldsymbol{H}_t + \lambda_{\mathrm{LM}}\boldsymbol{I}_P)^{-1}$ (monitoring that $\lambda_{\mathrm{LM}}$ does not grow so large that it collapses curvature into tiny GD steps), or substitute guaranteed-PSD curvature matrices: GGN $\boldsymbol{G} = \boldsymbol{J}^{\top}\boldsymbol{\Lambda}\boldsymbol{J}$ or true Fisher $\boldsymbol{F}$ sampled over model predictions $y \sim p_{\boldsymbol{\theta}}(y\mid\boldsymbol{x})$.

> [!warning] Pitfall 4: Optimizer State Memory Footprint in Large Language Models
> **Failure Mode:** While AdamW computes updates in $O(P)$ FLOPs, its auxiliary state dominates GPU memory in mixed-precision LLM pretraining [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 5 & App. D). Storing BF16 weights ($2P$ bytes), BF16 gradients ($2P$ bytes), FP32 master weights ($4P$ bytes), FP32 first moments $\boldsymbol{m}_t$ ($4P$ bytes), and FP32 second moments $\boldsymbol{v}_t$ ($4P$ bytes) consumes $16P$ bytes statically—$12P$ bytes for master/optimizer states alone ($6\times$ the BF16 model footprint). For a $P = 7\times 10^9$ model, static optimizer states occupy $\approx 112\text{ GB}$ before any activations or KV cache.\\
> **Remedy:** Shard optimizer states across $N_{\mathrm{GPU}}$ devices via ZeRO-1/FSDP (reducing per-GPU optimizer state to $12P / N_{\mathrm{GPU}}$ bytes), compress $(\boldsymbol{m}_t,\boldsymbol{v}_t)$ using blockwise 8-bit quantization or factored second-moment estimators (Adafactor), and avoid L-BFGS unless $2MP$ history fits comfortably inside memory.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[AdamW]], [[Momentum & Nesterov NAG]], [[Natural Gradient & K-FAC]]
- **Prerequisites (Directed $\leftarrow$):** [[Condition Number]], [[Gradient Descent]], [[Maximum Likelihood & Fisher Info]], [[Momentum & Nesterov NAG]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[AdamW]], [[TRPO]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[AdamW]], [[Condition Number]], [[Gradient Descent]], [[Maximum Likelihood & Fisher Info]], [[Momentum & Nesterov NAG]], [[TRPO]], [[Weight Decay & Dropout]]
- **Synthesized Sources:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Deep Learning]], [[Source - Handbook of Convergence Theorems]], [[Source - Machine Learning co ban]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
