---
type: monograph
domain: "D01"
parent_topic: "[[PT 01 - Convex, Smooth & Stochastic Optimization]]"
sub_topic: "[[ST 01 - Convex Duality & Deterministic Gradient Descent Theory]]"
cheatsheet: "[[Cheatsheet - Gradient Descent Dynamics - Strong Convexity, Smoothness, PL Condition & Saddle Escape]]"
part_of: 
  - "[[ST 01 - Convex Duality & Deterministic Gradient Descent Theory]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
concepts: 
  - "[[Gradient Descent]]"
  - "[[Polyak-Lojasiewicz (PL) Condition]]"
prerequisites: 
  - "[[Condition Number]]"
  - "[[Convexity & Subgradients]]"
  - "[[Gradient Descent]]"
enables: 
  - "[[Gradient Boosted Decision Trees (GBDT)]]"
  - "[[Momentum & Nesterov NAG]]"
  - "[[Natural Gradient & K-FAC]]"
  - "[[Polyak-Lojasiewicz (PL) Condition]]"
  - "[[Stochastic Gradient Descent (SGD)]]"
related: 
  - "[[Condition Number]]"
  - "[[Convexity & Subgradients]]"
  - "[[Gradient Descent]]"
  - "[[Momentum & Nesterov NAG]]"
  - "[[Polyak-Lojasiewicz (PL) Condition]]"
  - "[[Stochastic Gradient Descent (SGD)]]"
sources: 
  - "[[Source - Fall 2022]]"
  - "[[Source - Foundations]]"
  - "[[Source - Handbook of Convergence Theorems]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - monograph
  - domain-01
---

# Gradient Descent Dynamics: Strong Convexity, Smoothness, PL Condition & Saddle Escape

> [!abstract] Monograph Navigation
> - **Domain:** [[D01 - Mathematical Foundations for Machine Intelligence|Mathematical Foundations for Machine Intelligence]]
> - **Parent / Sub-Topic:** [[PT 01 - Convex, Smooth & Stochastic Optimization|Convex, Smooth & Stochastic Optimization]] → [[ST 01 - Convex Duality & Deterministic Gradient Descent Theory|Convex Duality & Deterministic Gradient Descent Theory]]
> - **Visual Cheatsheet:** [[Cheatsheet - Gradient Descent Dynamics - Strong Convexity, Smoothness, PL Condition & Saddle Escape]]
> - **Description:** Derives tight convergence rates O(1/T) for L-smooth convex functions, linear rate (1 - mu/L)^T under strong convexity or the Polyak-Lojasiewicz (PL) condition, local strong convexity, and perturbed saddle-point escape.

## Problem Formulation & Unified Notation

We consider unconstrained continuous optimization of a continuously differentiable objective $f : \mathbb{R}^d \to \mathbb{R}$ bounded below by $f^* \triangleq \inf_{\boldsymbol{x} \in \mathbb{R}^d} f(\boldsymbol{x}) > -\infty$:

$$
\mathrm{minimize}_{\boldsymbol{x} \in \mathbb{R}^d}   f(\boldsymbol{x}),
$$

assuming a nonempty minimizer set $\mathcal{X}^* \triangleq \arg\min_{\boldsymbol{x} \in \mathbb{R}^d} f(\boldsymbol{x})$. Throughout, we follow the standard unconstrained setup ([[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Prob. 3.1, p. 13); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.1, p. 275)). Closed-form roots of $\nabla f(\boldsymbol{x}) = \boldsymbol{0}$ rarely exist beyond linear least squares ([[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 7.1, p. 227); [[Source - Fall 2022|gradient_descent_unconstrained]] (Slide 5)). Moreover, zeroth-order methods require $O(d^2)$ queries on quadratics versus $O(d)$ for gradients ([[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 7.2.1, p. 214)), so first-order methods iterate via

$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \eta_t \nabla f(\boldsymbol{x}_t), \qquad t = 0, 1, 2, \dots,
$$

starting from $\boldsymbol{x}_0 \in \mathbb{R}^d$ with step size (learning rate) $\eta_t > 0$. By Cauchy–Schwarz, the unit-ball steepest-descent direction satisfies $\arg\min_{\lVert \boldsymbol{d} \rVert_2 \le 1} \langle \nabla f(\boldsymbol{x}_t), \boldsymbol{d} \rangle = -\nabla f(\boldsymbol{x}_t)/\lVert \nabla f(\boldsymbol{x}_t) \rVert_2$ whenever $\nabla f(\boldsymbol{x}_t) \neq \boldsymbol{0}$ [[Source - Fall 2022|gradient_descent_unconstrained]] (Slide 11), moving orthogonally to the level-set contours $\{\boldsymbol{x} : f(\boldsymbol{x}) = c\}$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 7.1, p. 228).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> The five primary texts adopt incompatible symbol assignments—most critically overloading $\mu$ to denote strong convexity, the linear convergence factor, and momentum simultaneously, and using opposite vector shapes for $\nabla f$. Table `tab:gdconv:notation` enforces a single unambiguous convention throughout this monograph: $\boldsymbol{x}_t \in \mathbb{R}^d$ is the iterate, $\nabla f(\boldsymbol{x}) \in \mathbb{R}^{d \times 1}$ is a column gradient, $\eta_t > 0$ is the step size, $\mu > 0$ is the strong-convexity or Polyak–Łojasiewicz (PL) constant, $L > 0$ is the gradient Lipschitz constant, and $\kappa \triangleq L/\mu$ is the condition number.

**Table:** Unified Notation and Cross-Book Convention Reconciliation.

| **Concept** | **Unified Symbol** | **Source Textbook Conventions & Reconciliation** |
| — | — | — |
| Parameter vector | $\boldsymbol{x}_t \in \mathbb{R}^d$, $\boldsymbol{x}^*$ | $\boldsymbol{x}_t \in \mathbb{R}^d$ in [[Source - Handbook of Convergence Theorems]] (Sec. 3) and [[Source - Mathematics for Machine Learning]] (Sec. 7.1); $\boldsymbol{\theta}_t \in \mathbb{R}^d$ in [[Source - Fall 2022]] (Slide 10) and [[Source - Probabilistic Machine Learning]] (Sec. 8.1); $\boldsymbol{w}^{(\tau)} \in \mathbb{R}^W$ in [[Source - Foundations]] (Sec. 7.2). |
| Objective function | $f(\boldsymbol{x})$, $f^* = \inf f$ | $f(\boldsymbol{x})$ in [[Source - Handbook of Convergence Theorems]] (Sec. 2) and [[Source - Mathematics for Machine Learning]] (Eq. (7.4)); $f(\boldsymbol{\theta})$ or $\mathcal{L}_n(\boldsymbol{\theta})$ in [[Source - Fall 2022]] (Slides 5, 9); $\mathcal{L}(\boldsymbol{\theta})$ in [[Source - Probabilistic Machine Learning]] (Eq. (8.1)); $E(\boldsymbol{w})$ in [[Source - Foundations]] (Sec. 7.1). |
| Gradient vector | $\nabla f(\boldsymbol{x}) \in \mathbb{R}^{d \times 1}$ (column) | Column vector in [[Source - Handbook of Convergence Theorems]] (Rem. 2.2), [[Source - Fall 2022]] (Slide 10), [[Source - Probabilistic Machine Learning]] (Sec. 8.1.1.1), and [[Source - Foundations]] (Eq. (7.1)); row vector $(\nabla f)(\boldsymbol{x}) \in \mathbb{R}^{1\times d}$ requiring explicit transpose in [[Source - Mathematics for Machine Learning]] (Eq. (7.5)). |
| Step size / LR | $\eta_t > 0$ (fixed $\eta$) | $\eta_t, \eta$ in [[Source - Fall 2022]] (Slide 10), [[Source - Probabilistic Machine Learning]] (Eq. (8.15)), and [[Source - Foundations]] (Eq. (7.16)); $\gamma$ or $\gamma_i$ in [[Source - Handbook of Convergence Theorems]] (Alg. 3.2) and [[Source - Mathematics for Machine Learning]] (Eq. (7.6)). |
| Strong convexity | $\mu > 0$ | $\mu$ in [[Source - Handbook of Convergence Theorems]] (Def. 2.11) and [[Source - Fall 2022]] (Slide 12); $m > 0$ in [[Source - Probabilistic Machine Learning]] (Eq. (8.9)). (Murphy reserves $\mu$ for linear rate factor; Bishop reserves $\mu$ for momentum.) |
| Smoothness constant | $L > 0$ | $L$ ($\lVert \nabla f(\boldsymbol{x}) - \nabla f(\boldsymbol{y}) \rVert_2 \le L\lVert \boldsymbol{x}-\boldsymbol{y} \rVert_2$) in [[Source - Handbook of Convergence Theorems]] (Def. 2.24), [[Source - Fall 2022]] (Slide 14), and [[Source - Probabilistic Machine Learning]] (Sec. 8.2.2.1); $\lambda_{\max}(\nabla^2 E)$ in [[Source - Foundations]] (Sec. 7.3). |
| Condition number | $\kappa = L/\mu$ ($\lambda_{\max}/\lambda_{\min}$) | Hessian condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ in [[Source - Probabilistic Machine Learning]] (Sec. 8.2.3) and [[Source - Foundations]] (p. 220); linear-system $\kappa(\boldsymbol{A}) = \sigma_{\max}(\boldsymbol{A})/\sigma_{\min}(\boldsymbol{A})$ in [[Source - Mathematics for Machine Learning]] (p. 230) yields $\kappa(\boldsymbol{A}^{\top}\boldsymbol{A}) = \kappa(\boldsymbol{A})^2$. |

## Core Mathematical Derivations & Theorems

### $L$-Smoothness, Descent Lemma, Co-Coercivity & Non-Convex Stationarity

A differentiable function $f : \mathbb{R}^d \to \mathbb{R}$ is *$L$-smooth* ($L > 0$) if its gradient is $L$-Lipschitz continuous in Euclidean norm:

$$
\lVert \nabla f(\boldsymbol{y}) - \nabla f(\boldsymbol{x}) \rVert_2 \le L \lVert \boldsymbol{y} - \boldsymbol{x} \rVert_2, \qquad \forall \boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^d.
$$

When $f \in C^2(\mathbb{R}^d)$, the Mean-Value Inequality applied to $\nabla f$ implies that `(eq:gdconv:lsmooth)` is equivalent to a uniform spectral-norm bound $\lVert \nabla^2 f(\boldsymbol{x}) \rVert_2 \le L$, i.e., every eigenvalue $\lambda$ of the symmetric Hessian $\nabla^2 f(\boldsymbol{x})$ satisfies $\lvert \lambda \rvert \le L$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lem. 2.6 & Lem. 2.26, pp. 5, 11) [[Source - Fall 2022|gradient_descent_unconstrained]] (Slide 14). Smoothness furnishes a global quadratic majorant whose minimization at each iterate yields guaranteed descent.

> [!important] Lemma: $L$-Smoothness Descent Lemma & Monotonic Progress
> Let $f : \mathbb{R}^d \to \mathbb{R}$ be $L$-smooth. Then for all $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^d$:
>
> $$
> f(\boldsymbol{y}) \le f(\boldsymbol{x}) + \langle \nabla f(\boldsymbol{x}), \boldsymbol{y} - \boldsymbol{x} \rangle + \frac{L}{2}\lVert \boldsymbol{y} - \boldsymbol{x} \rVert_2^2.
> $$
>
> Consequently, a gradient step $\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \eta \nabla f(\boldsymbol{x}_t)$ with $\eta > 0$ satisfies
>
> $$
> f(\boldsymbol{x}_{t+1}) - f(\boldsymbol{x}_t) \le -\eta\left(1 - \frac{\eta L}{2}\right)\lVert \nabla f(\boldsymbol{x}_t) \rVert_2^2.
> $$
>
> In particular, for $0 < \eta < \frac{2}{L}$, the objective decreases strictly whenever $\nabla f(\boldsymbol{x}_t) \neq \boldsymbol{0}$; minimizing the coefficient $-\eta(1 - \eta L/2)$ gives the optimal greedy step $\eta = \frac{1}{L}$, yielding $f(\boldsymbol{x}_{t+1}) - f(\boldsymbol{x}_t) \le -\frac{1}{2L}\lVert \nabla f(\boldsymbol{x}_t) \rVert_2^2$ and the global gradient bound
>
> $$
> \frac{1}{2L}\lVert \nabla f(\boldsymbol{x}) \rVert_2^2 \le f(\boldsymbol{x}) - f^*, \qquad \forall \boldsymbol{x} \in \mathbb{R}^d.
> $$

**Proof.**
Fix $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^d$ and define $\phi(s) \triangleq f(\boldsymbol{x} + s(\boldsymbol{y}-\boldsymbol{x}))$ for $s \in [0, 1]$. By the Fundamental Theorem of Calculus and Cauchy–Schwarz [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lem. 2.25, p. 11),

$$
\begin{aligned}
f(\boldsymbol{y}) - f(\boldsymbol{x}) - \langle \nabla f(\boldsymbol{x}), \boldsymbol{y}-\boldsymbol{x} \rangle
  &= \int_0^1 \langle \nabla f(\boldsymbol{x} + s(\boldsymbol{y}-\boldsymbol{x})) - \nabla f(\boldsymbol{x}), \boldsymbol{y}-\boldsymbol{x} \rangle  \mathrm{d}s \\
  &\le \int_0^1 \lVert \nabla f(\boldsymbol{x} + s(\boldsymbol{y}-\boldsymbol{x})) - \nabla f(\boldsymbol{x}) \rVert_2 \lVert \boldsymbol{y}-\boldsymbol{x} \rVert_2  \mathrm{d}s \\
  &\le \int_0^1 L s \lVert \boldsymbol{y}-\boldsymbol{x} \rVert_2^2  \mathrm{d}s = \frac{L}{2}\lVert \boldsymbol{y}-\boldsymbol{x} \rVert_2^2,
\end{aligned}
$$

proving `(eq:gdconv:quad_upper)`. Setting $\boldsymbol{y} = \boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \eta\nabla f(\boldsymbol{x}_t)$ directly yields
\[
  f(\boldsymbol{x}_{t+1}) - f(\boldsymbol{x}_t) \le -\eta\lVert \nabla f(\boldsymbol{x}_t) \rVert_2^2 + \frac{L\eta^2}{2}\lVert \nabla f(\boldsymbol{x}_t) \rVert_2^2 = -\eta\left(1 - \frac{\eta L}{2}\right)\lVert \nabla f(\boldsymbol{x}_t) \rVert_2^2.
\]
Finally, evaluating `(eq:gdconv:sufficient_decrease)` at any $\boldsymbol{x}_t = \boldsymbol{x} \in \mathbb{R}^d$ with $\eta = 1/L$ and invoking $f^* \le f(\boldsymbol{x} - \frac{1}{L}\nabla f(\boldsymbol{x}))$ yields `(eq:gdconv:grad_upper)` ([[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lem. 2.28, p. 12)). 

General non-convex minimization is NP-hard ([[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.1.1, p. 275)), so for $L$-smooth objectives first-order guarantees target $\epsilon$-first-order stationary points ($\epsilon$-FOSP) satisfying $\lVert \nabla f(\boldsymbol{x}) \rVert_2 \le \epsilon$. Telescoping `(eq:gdconv:sufficient_decrease)` for $0 < \eta \le 1/L$ (or generally $0 < \eta < 2/L$ with constant $c(\eta) = \eta(1 - \eta L/2) > 0$) gives
\[
  \sum_{t=0}^{T-1} \lVert \nabla f(\boldsymbol{x}_t) \rVert_2^2 \le \frac{f(\boldsymbol{x}_0) - f(\boldsymbol{x}_T)}{\eta(1 - \eta L/2)} \le \frac{f(\boldsymbol{x}_0) - f^*}{\eta(1 - \eta L/2)}.
\]
Thus, at $\eta = 1/L$, $\min_{0 \le t < T} \lVert \nabla f(\boldsymbol{x}_t) \rVert_2 \le \sqrt{\frac{2L(f(\boldsymbol{x}_0) - f^*)}{T}}$, guaranteeing an $\epsilon$-FOSP within $T \ge \frac{2L(f(\boldsymbol{x}_0)-f^*)}{\epsilon^2} = O(\epsilon^{-2})$ iterations [[Source - Fall 2022|gradient_descent_unconstrained]] (Thm. 6, Slides 82–84).

When $f$ is simultaneously convex and $L$-smooth, smoothness combines with the first-order convexity inequality $f(\boldsymbol{y}) \ge f(\boldsymbol{x}) + \langle \nabla f(\boldsymbol{x}), \boldsymbol{y}-\boldsymbol{x} \rangle$ to produce *co-coercivity* (the Baillon–Haddad theorem), a condition strictly stronger than monotonicity and $L$-Lipschitzness taken separately [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lem. 2.29, pp. 12–13).

> [!important] Lemma: Co-Coercivity (Baillon–Haddad) for Convex $L$-Smooth Functions
> If $f : \mathbb{R}^d \to \mathbb{R}$ is convex and $L$-smooth, then for all $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^d$:
>
> $$
> \begin{aligned}
> \frac{1}{2L}\lVert \nabla f(\boldsymbol{y}) - \nabla f(\boldsymbol{x}) \rVert_2^2 &\le f(\boldsymbol{y}) - f(\boldsymbol{x}) - \langle \nabla f(\boldsymbol{x}), \boldsymbol{y}-\boldsymbol{x} \rangle,  \\
>   \frac{1}{L}\lVert \nabla f(\boldsymbol{y}) - \nabla f(\boldsymbol{x}) \rVert_2^2 &\le \langle \nabla f(\boldsymbol{y}) - \nabla f(\boldsymbol{x}), \boldsymbol{y}-\boldsymbol{x} \rangle.
> \end{aligned}
> $$

**Proof.**
Fix $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^d$. For any auxiliary point $\boldsymbol{z} \in \mathbb{R}^d$, combine convexity between $(\boldsymbol{x}, \boldsymbol{z})$ with `(eq:gdconv:quad_upper)` between $(\boldsymbol{y}, \boldsymbol{z})$:
\[
  f(\boldsymbol{x}) - f(\boldsymbol{y}) = \big(f(\boldsymbol{x}) - f(\boldsymbol{z})\big) + \big(f(\boldsymbol{z}) - f(\boldsymbol{y})\big)
  \le \langle \nabla f(\boldsymbol{x}), \boldsymbol{x}-\boldsymbol{z} \rangle + \langle \nabla f(\boldsymbol{y}), \boldsymbol{z}-\boldsymbol{y} \rangle + \frac{L}{2}\lVert \boldsymbol{z}-\boldsymbol{y} \rVert_2^2.
\]
Minimizing the right-hand side over $\boldsymbol{z}$ yields $\boldsymbol{z} = \boldsymbol{y} - \frac{1}{L}(\nabla f(\boldsymbol{y}) - \nabla f(\boldsymbol{x}))$. Substituting this optimal $\boldsymbol{z}$ gives $f(\boldsymbol{x}) - f(\boldsymbol{y}) \le \langle \nabla f(\boldsymbol{x}), \boldsymbol{x}-\boldsymbol{y} \rangle - \frac{1}{2L}\lVert \nabla f(\boldsymbol{y})-\nabla f(\boldsymbol{x}) \rVert_2^2$, which rearranges to `(eq:gdconv:bregman_lb)`. Swapping $\boldsymbol{x}$ and $\boldsymbol{y}$ in `(eq:gdconv:bregman_lb)` and adding the two inequalities eliminates the function-value terms and yields `(eq:gdconv:cocoercive)`. 

### General Convex $L$-Smooth Convergence ($O(1/T)$) & Sublinear Tightness

Without positive curvature lower bounds, convex $L$-smooth objectives can contain arbitrarily flat basins (e.g., $f(\boldsymbol{x}) = \lVert \boldsymbol{x} \rVert_2^4$ where $\nabla^2 f(\boldsymbol{0}) = \boldsymbol{0}$; [[Source - Fall 2022|gradient_descent_unconstrained]] (Slide 60)). Nevertheless, gradient descent remains Fej\'er-monotone with respect to every minimizer $\boldsymbol{x}^* \in \mathcal{X}^*$ and achieves a tight worst-case $O(1/T)$ sublinear rate in objective gap and $O(1/T)$ gradient-norm convergence.

> [!important] Theorem: $O(1/T)$ Objective & $O(1/T)$ Gradient-Norm Convergence for Convex $L$-Smooth GD
> Let $f : \mathbb{R}^d \to \mathbb{R}$ be convex and $L$-smooth with nonempty minimizer set $\mathcal{X}^*$. For any $\boldsymbol{x}^* \in \mathcal{X}^*$ and constant step size $0 < \eta \le \frac{1}{L}$, gradient descent `(eq:gdconv:gd_update)` satisfies for all $T \ge 1$:
> - **Fej\'er Monotonicity:** $\lVert \boldsymbol{x}_{t+1} - \boldsymbol{x}^* \rVert_2 \le \lVert \boldsymbol{x}_t - \boldsymbol{x}^* \rVert_2 \le \lVert \boldsymbol{x}_0 - \boldsymbol{x}^* \rVert_2$ for all $t \ge 0$.
> - **Objective Suboptimality ($O(1/T)$):**
>
> $$
> f(\boldsymbol{x}_T) - f^* \le \frac{\lVert \boldsymbol{x}_0 - \boldsymbol{x}^* \rVert_2^2}{2\eta T}.
> $$
> - **Improved Stationarity ($O(1/\epsilon)$):** For $\eta = \frac{1}{L}$, $\min_{0 \le t < T}\lVert \nabla f(\boldsymbol{x}_t) \rVert_2 \le \frac{2L\lVert \boldsymbol{x}_0 - \boldsymbol{x}^* \rVert_2}{T}$ (so $T = O(L\lVert \boldsymbol{x}_0-\boldsymbol{x}^* \rVert_2/\epsilon)$ suffices for $\lVert \nabla f(\boldsymbol{x}_t) \rVert_2 \le \epsilon$).

**Proof.**
Following [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Thm. 3.4, pp. 13–15), apply co-coercivity `(eq:gdconv:cocoercive)` at $(\boldsymbol{x}_t, \boldsymbol{x}^*)$ with $\nabla f(\boldsymbol{x}^*) = \boldsymbol{0}$ to obtain
\[
  \lVert \boldsymbol{x}_{t+1} - \boldsymbol{x}^* \rVert_2^2 = \lVert \boldsymbol{x}_t - \boldsymbol{x}^* \rVert_2^2 - 2\eta\langle \nabla f(\boldsymbol{x}_t), \boldsymbol{x}_t-\boldsymbol{x}^* \rangle + \eta^2\lVert \nabla f(\boldsymbol{x}_t) \rVert_2^2 \le \lVert \boldsymbol{x}_t - \boldsymbol{x}^* \rVert_2^2 - \eta\left(\frac{2}{L} - \eta\right)\lVert \nabla f(\boldsymbol{x}_t) \rVert_2^2,
\]
proving Fej\'er monotonicity for $0 < \eta \le \frac{2}{L}$ [[Source - Fall 2022|gradient_descent_unconstrained]] (Lem. 2, Slides 63–64). Next, writing $-\nabla f(\boldsymbol{x}_t) = \frac{1}{\eta}(\boldsymbol{x}_{t+1}-\boldsymbol{x}_t)$ and expanding $\lVert \boldsymbol{x}_{t+1}-\boldsymbol{x}^* \rVert_2^2 - \lVert \boldsymbol{x}_t-\boldsymbol{x}^* \rVert_2^2$:

$$
\begin{aligned}
\frac{1}{2\eta}\left(\lVert \boldsymbol{x}_{t+1}-\boldsymbol{x}^* \rVert_2^2 - \lVert \boldsymbol{x}_t-\boldsymbol{x}^* \rVert_2^2\right)
  &= -\frac{1}{2\eta}\lVert \boldsymbol{x}_{t+1}-\boldsymbol{x}_t \rVert_2^2 - \langle \nabla f(\boldsymbol{x}_t), \boldsymbol{x}_{t+1}-\boldsymbol{x}_t \rangle + \langle \nabla f(\boldsymbol{x}_t), \boldsymbol{x}^*-\boldsymbol{x}_t \rangle \\
  &\le \left(\frac{L}{2} - \frac{1}{2\eta}\right)\lVert \boldsymbol{x}_{t+1}-\boldsymbol{x}_t \rVert_2^2 + \big(f(\boldsymbol{x}_t) - f(\boldsymbol{x}_{t+1})\big) + \big(f^* - f(\boldsymbol{x}_t)\big) \\
  &\le -\big(f(\boldsymbol{x}_{t+1}) - f^*\big),
\end{aligned}
$$

where the first inequality combines `(eq:gdconv:quad_upper)` at $(\boldsymbol{x}_t, \boldsymbol{x}_{t+1})$ with convexity $f^* - f(\boldsymbol{x}_t) \ge \langle \nabla f(\boldsymbol{x}_t), \boldsymbol{x}^*-\boldsymbol{x}_t \rangle$, and the second uses $\eta \le 1/L$. Now define the discrete Lyapunov energy $E_t \triangleq \frac{1}{2\eta}\lVert \boldsymbol{x}_t-\boldsymbol{x}^* \rVert_2^2 + t(f(\boldsymbol{x}_t)-f^*)$. Using monotonicity $f(\boldsymbol{x}_{t+1}) \le f(\boldsymbol{x}_t)$ from `(eq:gdconv:sufficient_decrease)`,
\[
  E_{t+1} - E_t = \big(f(\boldsymbol{x}_{t+1}) - f^*\big) + t\big(f(\boldsymbol{x}_{t+1}) - f(\boldsymbol{x}_t)\big) + \frac{1}{2\eta}\left(\lVert \boldsymbol{x}_{t+1}-\boldsymbol{x}^* \rVert_2^2 - \lVert \boldsymbol{x}_t-\boldsymbol{x}^* \rVert_2^2\right) \le 0.
\]
Hence $T(f(\boldsymbol{x}_T)-f^*) \le E_T \le E_0 = \frac{1}{2\eta}\lVert \boldsymbol{x}_0-\boldsymbol{x}^* \rVert_2^2$, giving `(eq:gdconv:convex_rate)`. For claim (3) with $\eta = 1/L$ and even $T=2m$ (analogous for odd $T$), telescoping `(eq:gdconv:sufficient_decrease)` over the second half $t \in \{m, \dots, 2m-1\}$ and applying `(eq:gdconv:convex_rate)` at $t=m$ yields
\[
  \frac{m}{2L}\min_{m \le t < 2m}\lVert \nabla f(\boldsymbol{x}_t) \rVert_2^2 \le \frac{1}{2L}\sum_{t=m}^{2m-1}\lVert \nabla f(\boldsymbol{x}_t) \rVert_2^2 \le f(\boldsymbol{x}_m) - f^* \le \frac{L\lVert \boldsymbol{x}_0-\boldsymbol{x}^* \rVert_2^2}{2m},
\]
so $\min_{0 \le t < T}\lVert \nabla f(\boldsymbol{x}_t) \rVert_2 \le \frac{L\lVert \boldsymbol{x}_0-\boldsymbol{x}^* \rVert_2}{m} = \frac{2L\lVert \boldsymbol{x}_0-\boldsymbol{x}^* \rVert_2}{T}$ [[Source - Fall 2022|gradient_descent_unconstrained]] (Thm. 7, Slides 85–87). 

Because Fej\'er monotonicity traps the entire trajectory inside the ball $\mathbb{B}(\boldsymbol{x}^*, \lVert \boldsymbol{x}_0-\boldsymbol{x}^* \rVert_2)$, Theorem `thm:gdconv:convex` immediately localizes: if $f$ is only convex and $L$-smooth on $\mathbb{B}(\boldsymbol{x}^*, r)$ with initialization $\lVert \boldsymbol{x}_0-\boldsymbol{x}^* \rVert_2 \le r$, all bounds remain valid [[Source - Fall 2022|gradient_descent_unconstrained]] (Prop. 1, Slide 67). However, [[Source - Fall 2022|gradient_descent_unconstrained]] (Claim 1, Slides 68–72) emphasizes that `(eq:gdconv:convex_rate)` is a worst-case minimax guarantee: on the population singular phase-retrieval quartic $\mathcal{L}(\theta) = 3\theta^4 + 1$ ($\theta^*=0$, $L=36$ on $\mathbb{B}(0,1)$), $\theta_t = \theta_{t-1}(1 - 12\eta\theta_{t-1}^2)$ decays as $\lvert \theta_t \rvert = \Theta(t^{-1/2})$, so $\mathcal{L}(\theta_t) - \mathcal{L}(0) = \Theta(t^{-2})$ rather than $O(t^{-1})$.

### $\mu$-Strong Convexity, Linear Contraction & Optimal Step Size $\eta^* = \frac{2 {\mu+L}$}

A function $f : \mathbb{R}^d \to \mathbb{R}$ is *$\mu$-strongly convex* ($\mu > 0$) if $g(\boldsymbol{x}) \triangleq f(\boldsymbol{x}) - \frac{\mu}{2}\lVert \boldsymbol{x} \rVert_2^2$ is convex. This definition ([[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lem. 2.12, p. 7)) corresponds to the quadratic formulation of [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. (8.10), p. 281); for differentiable $f$, it is equivalent to a uniform quadratic lower bound at every point, and for $f \in C^2(\mathbb{R}^d)$ to a uniform lower spectral bound on the Hessian:

$$
\begin{aligned}
f(\boldsymbol{y}) &\ge f(\boldsymbol{x}) + \langle \nabla f(\boldsymbol{x}), \boldsymbol{y}-\boldsymbol{x} \rangle + \frac{\mu}{2}\lVert \boldsymbol{y}-\boldsymbol{x} \rVert_2^2, && \forall \boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^d,  \\
  \nabla^2 f(\boldsymbol{x}) &\succeq \mu \boldsymbol{I}_d \iff \lambda_{\min}\big(\nabla^2 f(\boldsymbol{x})\big) \ge \mu > 0, && \forall \boldsymbol{x} \in \mathbb{R}^d
\end{aligned}
$$

These three formulations are mutually equivalent ([[Source - Fall 2022|gradient_descent_unconstrained]] (Slide 12); for formal proofs, see [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lem. 2.14–2.15, p. 8)). As shown in [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lem. 2.13, p. 8), strong convexity guarantees a unique global minimizer $\boldsymbol{x}^*$ and boosts convergence from sublinear $O(1/T)$ to geometric (linear) contraction. While elementary proofs often restrict $\eta \le 1/L$ to obtain contraction $(1 - \eta\mu)^T$, exploiting the residual convex smooth function $g(\boldsymbol{x}) = f(\boldsymbol{x}) - \frac{\mu}{2}\lVert \boldsymbol{x} \rVert_2^2$ extends the exact convergence certificate across the entire stable interval $0 < \eta < \frac{2}{L}$ and identifies the optimal step size $\eta^* = \frac{2}{\mu+L}$.

> [!important] Theorem: Linear Contraction Under $\mu$-Strong Convexity, $L$-Smoothness & Optimal Step Size
> Let $f : \mathbb{R}^d \to \mathbb{R}$ be $\mu$-strongly convex and $L$-smooth ($0 < \mu \le L$) with unique minimizer $\boldsymbol{x}^*$ and condition number $\kappa \triangleq L/\mu \ge 1$. Then:
> - **Refined Co-Coercivity (Nesterov's Inequality):** For all $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^d$,
>
> $$
> \langle \nabla f(\boldsymbol{x}) - \nabla f(\boldsymbol{y}), \boldsymbol{x}-\boldsymbol{y} \rangle \ge \frac{\mu L}{\mu + L}\lVert \boldsymbol{x}-\boldsymbol{y} \rVert_2^2 + \frac{1}{\mu + L}\lVert \nabla f(\boldsymbol{x}) - \nabla f(\boldsymbol{y}) \rVert_2^2.
> $$
> - **Standard Regime ($0 < \eta \le \frac{1}{L}$):** $\lVert \boldsymbol{x}_t - \boldsymbol{x}^* \rVert_2^2 \le (1 - \eta\mu)^t \lVert \boldsymbol{x}_0 - \boldsymbol{x}^* \rVert_2^2$; at $\eta = \frac{1}{L}$, $\lVert \boldsymbol{x}_t - \boldsymbol{x}^* \rVert_2^2 \le (1 - \kappa^{-1})^t \lVert \boldsymbol{x}_0 - \boldsymbol{x}^* \rVert_2^2$ and $f(\boldsymbol{x}_t) - f^* \le \frac{L}{2}(1 - \kappa^{-1})^t \lVert \boldsymbol{x}_0-\boldsymbol{x}^* \rVert_2^2$.
> - **Tight Optimal Rate ($0 < \eta < \frac{2}{L}$):** For all $t \ge 0$,
>
> $$
> \lVert \boldsymbol{x}_t - \boldsymbol{x}^* \rVert_2 \le \rho(\eta)^t \lVert \boldsymbol{x}_0 - \boldsymbol{x}^* \rVert_2, \qquad \rho(\eta) \triangleq \max\{\lvert 1 - \eta\mu \rvert,  \lvert 1 - \eta L \rvert\} < 1.
> $$
>
>   At the optimal fixed step size $\eta^* = \frac{2}{\mu + L}$, the contraction factor is minimized to
>
> $$
> \rho(\eta^*) = \frac{L - \mu}{L + \mu} = \frac{\kappa - 1}{\kappa + 1}, \qquad \lVert \boldsymbol{x}_t - \boldsymbol{x}^* \rVert_2^2 \le \left(\frac{\kappa - 1}{\kappa + 1}\right)^{2t}\lVert \boldsymbol{x}_0 - \boldsymbol{x}^* \rVert_2^2,
> $$
>
>   requiring $T \ge \frac{\kappa + 1}{2}\log(1/\epsilon)$ iterations (vs. $\kappa \log(1/\epsilon)$ at $\eta = 1/L$) to achieve $\lVert \boldsymbol{x}_T - \boldsymbol{x}^* \rVert_2 \le \epsilon\lVert \boldsymbol{x}_0 - \boldsymbol{x}^* \rVert_2$.

**Proof.**
*Step 1 (Refined co-coercivity):* When $L = \mu$, `(eq:gdconv:sc_quad)` and `(eq:gdconv:quad_upper)` force $f$ to take the isotropic form $f(\boldsymbol{x}) = f^* + (\mu/2)\lVert \boldsymbol{x}-\boldsymbol{x}^* \rVert_2^2$, where `(eq:gdconv:nesterov_ineq)` holds with equality. Assume $L > \mu$. By [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lem. 2.12, p. 7), $g(\boldsymbol{x}) = f(\boldsymbol{x}) - \frac{\mu}{2}\lVert \boldsymbol{x} \rVert_2^2$ is convex and $(L-\mu)$-smooth. Applying co-coercivity `(eq:gdconv:cocoercive)` to $g$ with smoothness constant $L - \mu$ yields
\[
  \langle \nabla f(\boldsymbol{x})-\nabla f(\boldsymbol{y}) - \mu(\boldsymbol{x}-\boldsymbol{y}), \boldsymbol{x}-\boldsymbol{y} \rangle \ge \frac{1}{L-\mu}\lVert \nabla f(\boldsymbol{x})-\nabla f(\boldsymbol{y}) - \mu(\boldsymbol{x}-\boldsymbol{y}) \rVert_2^2.
\]
Multiplying through by $(L - \mu)$, expanding the squared norm, and collecting the inner-product terms on the left immediately produces `(eq:gdconv:nesterov_ineq)`.

*Step 2 (Contraction for $0 < \eta \le \frac{2}{\mu+L}$):* Setting $\boldsymbol{y} = \boldsymbol{x}^*$ ($\nabla f(\boldsymbol{x}^*) = \boldsymbol{0}$) in `(eq:gdconv:nesterov_ineq)` and substituting into $\lVert \boldsymbol{x}_{t+1}-\boldsymbol{x}^* \rVert_2^2 = \lVert \boldsymbol{x}_t-\boldsymbol{x}^* \rVert_2^2 - 2\eta\langle \nabla f(\boldsymbol{x}_t), \boldsymbol{x}_t-\boldsymbol{x}^* \rangle + \eta^2\lVert \nabla f(\boldsymbol{x}_t) \rVert_2^2$ yields

$$
\lVert \boldsymbol{x}_{t+1}-\boldsymbol{x}^* \rVert_2^2 \le \left(1 - \frac{2\eta\mu L}{\mu + L}\right)\lVert \boldsymbol{x}_t - \boldsymbol{x}^* \rVert_2^2 + \eta\left(\eta - \frac{2}{\mu + L}\right)\lVert \nabla f(\boldsymbol{x}_t) \rVert_2^2.
$$

For $0 < \eta \le \frac{2}{\mu + L}$, the last coefficient is non-positive, and since $\lVert \nabla f(\boldsymbol{x}_t) \rVert_2 \ge \mu\lVert \boldsymbol{x}_t-\boldsymbol{x}^* \rVert_2$ (by Cauchy–Schwarz on `(eq:gdconv:sc_quad)`), we obtain
\[
  \lVert \boldsymbol{x}_{t+1}-\boldsymbol{x}^* \rVert_2^2 \le \left(1 - \frac{2\eta\mu L}{\mu+L} + \eta\mu^2\left(\eta - \frac{2}{\mu+L}\right)\right)\lVert \boldsymbol{x}_t-\boldsymbol{x}^* \rVert_2^2 = (1 - \eta\mu)^2\lVert \boldsymbol{x}_t-\boldsymbol{x}^* \rVert_2^2.
\]
For $0 < \eta \le \frac{1}{L}$, $(1-\eta\mu)^2 \le 1 - \eta\mu$ recovers the bound of [[Source - Fall 2022|gradient_descent_unconstrained]] (Thm. 1, Slides 15–18) and [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Thm. 3.6, p. 16). Applying `(eq:gdconv:quad_upper)` at $\boldsymbol{x}^*$ then converts this distance contraction into the objective gap bound
$f(\boldsymbol{x}_t) - f^* \le \frac{L}{2}\lVert \boldsymbol{x}_t-\boldsymbol{x}^* \rVert_2^2$ [[Source - Fall 2022|gradient_descent_unconstrained]] (Cor. 1, Slide 19).

*Step 3 (Full interval $0 < \eta < \frac{2}{L}$ via Fixed-Point Operator):* Define the GD map $T(\boldsymbol{x}) \triangleq \boldsymbol{x} - \eta\nabla f(\boldsymbol{x})$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (pp. 16–17). When $f \in C^2$, its Jacobian is $DT(\boldsymbol{x}) = \boldsymbol{I}_d - \eta\nabla^2 f(\boldsymbol{x})$, whose eigenvalues lie in $[1 - \eta L,  1 - \eta\mu]$ by `(eq:gdconv:sc_hess)` and `(eq:gdconv:lsmooth)`. Hence $\lVert DT(\boldsymbol{x}) \rVert_2 \le \max\{\lvert 1 - \eta\mu \rvert, \lvert 1 - \eta L \rvert\} = \rho(\eta)$, and by the Mean-Value Inequality around the fixed point $T(\boldsymbol{x}^*) = \boldsymbol{x}^*$, $\lVert \boldsymbol{x}_{t+1}-\boldsymbol{x}^* \rVert_2 = \lVert T(\boldsymbol{x}_t) - T(\boldsymbol{x}^*) \rVert_2 \le \rho(\eta)\lVert \boldsymbol{x}_t - \boldsymbol{x}^* \rVert_2$ (which likewise holds for $C^1$ functions via `(eq:gdconv:nesterov_ineq)` with $\boldsymbol{y}=\boldsymbol{x}^*$ and $\nabla f(\boldsymbol{x}_t)$ bounded by $L\lVert \boldsymbol{x}_t-\boldsymbol{x}^* \rVert_2$). Minimizing $\rho(\eta) = \max\{1 - \eta\mu,  \eta L - 1\}$ equates the two branches: $1 - \eta^*\mu = \eta^* L - 1 \implies \eta^* = \frac{2}{\mu + L}$, giving $\rho(\eta^*) = \frac{L - \mu}{L + \mu} = \frac{\kappa - 1}{\kappa + 1}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. (8.29), p. 286) [[Source - Fall 2022|gradient_descent_unconstrained]] (Slide 15). 

> [!tip] Decoupled Eigenmodes & Spectral Rate Bottleneck
> Expanding a local quadratic approximation $f(\boldsymbol{x}) \approx f^* + \frac{1}{2}(\boldsymbol{x}-\boldsymbol{x}^*)^{\top} \boldsymbol{H} (\boldsymbol{x}-\boldsymbol{x}^*)$ in the orthonormal eigenbasis $\{\boldsymbol{u}_i\}_{i=1}^d$ of $\boldsymbol{H} = \nabla^2 f(\boldsymbol{x}^*)$ with $\boldsymbol{x}_t - \boldsymbol{x}^* = \sum_{i=1}^d \alpha_i^{(t)} \boldsymbol{u}_i$ decouples `(eq:gdconv:gd_update)` into $d$ independent scalar recursions [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 7.3, Eqs. (7.24)–(7.30), pp. 218–220):
> \[
>   \alpha_i^{(t)} = (1 - \eta \lambda_i)^t \alpha_i^{(0)}, \qquad i = 1, \dots, d.
> \]
> Stability requires $\lvert 1 - \eta\lambda_{\max} \rvert < 1 \iff \eta < 2/\lambda_{\max}$. When $\eta \approx 2/\lambda_{\max}$, the steepest mode ($\lambda_{\max}$) oscillates with multiplier $\approx -1$ while the flattest valley mode ($\lambda_{\min}$) contracts sluggishly by $1 - 2\lambda_{\min}/\lambda_{\max} = 1 - 2/\kappa$. At $\eta^* = 2/(\lambda_{\min}+\lambda_{\max})$, the two extreme modes balance symmetrically at $\pm\frac{\kappa - 1}{\kappa + 1}$, cutting iteration complexity roughly in half relative to $\eta = 1/L$ without changing the $O(\kappa \log(1/\epsilon))$ scaling.

As in the convex setting, [[Source - Fall 2022|gradient_descent_unconstrained]] (Thm. 2, Slides 37–40) shows that if $f$ is only *locally* $\mu$-strongly convex and $L$-smooth on $\mathbb{B}(\boldsymbol{x}^*, r)$ with $\lVert \boldsymbol{x}_0 - \boldsymbol{x}^* \rVert_2 \le r$ (as occurs in logistic regression, where $\lambda_{\min}(\nabla^2 \mathcal{T}_n(\boldsymbol{\theta})) \to 0$ as $\lVert \boldsymbol{\theta} \rVert_2 \to \infty$, and non-zero phase retrieval; [[Source - Fall 2022|gradient_descent_unconstrained]] (Slides 29–36)), linear contraction keeps all iterates inside $\mathbb{B}(\boldsymbol{x}^*, r)$ and yields the identical rate $(1-\eta\mu)^{t/2}$ in parameter distance. Moreover, decomposing total estimation error into optimization error plus statistical error, $\lVert \boldsymbol{x}_t^n - \boldsymbol{x}^* \rVert_2 \le (1-\eta\mu)^{t/2}\lVert \boldsymbol{x}_0 - \hat{\boldsymbol{x}}_n \rVert_2 + \lVert \hat{\boldsymbol{x}}_n - \boldsymbol{x}^* \rVert_2$, reveals that only $T = O(\log n)$ GD iterations are needed to match the $O(n^{-1/2})$ statistical error floor of empirical risk minimization [[Source - Fall 2022|gradient_descent_unconstrained]] (Slides 24–26, 78).

### Polyak–Łojasiewicz (PL), Quadratic Growth & KL Landscapes

Local strong convexity still requires warm-start initialization inside $\mathbb{B}(\boldsymbol{x}^*, r)$. Remarkably, global linear convergence survives even on non-convex landscapes and non-isolated solution manifolds under the *Polyak–Łojasiewicz (PL)* gradient-dominance inequality ([[Source - Fall 2022|gradient_descent_unconstrained]] (Def. 1, Slide 42)). Specifically, as formalized in [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Def. 2.17, p. 8), a differentiable function satisfies $\mu$-PL ($\mu > 0$) if

$$
\frac{1}{2}\lVert \nabla f(\boldsymbol{x}) \rVert_2^2 \ge \mu \big(f(\boldsymbol{x}) - f^*\big), \qquad \forall \boldsymbol{x} \in \mathbb{R}^d,
$$

for some constant $\mu > 0$. The PL condition does *not* imply convexity or uniqueness of minimizers, yet it forbids spurious local minima and saddle points: $\nabla f(\boldsymbol{x}) = \boldsymbol{0} \iff f(\boldsymbol{x}) = f^*$, so every stationary point is globally optimal [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lem. 2.22, p. 10) [[Source - Fall 2022|gradient_descent_unconstrained]] (Slide 43).

> [!important] Theorem: Global Linear Convergence Under PL & Landscape Implications
> Let $f : \mathbb{R}^d \to \mathbb{R}$ be $L$-smooth and satisfy the $\mu$-PL condition `(eq:gdconv:pl_def)`. Then:
> - **Hierarchy:** Every $\mu$-strongly convex function is $\mu$-PL. Conversely, every $C^1$ $\mu$-PL function satisfies the *Quadratic Growth (QG)* condition $f(\boldsymbol{x}) - f^* \ge \frac{\mu}{2}\mathrm{dist}(\boldsymbol{x}, \mathcal{X}^*)^2$ whenever $\mathcal{X}^* \neq \emptyset$ (and when $f$ is $L$-smooth, $\mu \le L$ via `(eq:gdconv:grad_upper)`).
> - **Global Linear Convergence:** For any step size $0 < \eta < \frac{2}{L}$ and arbitrary initialization $\boldsymbol{x}_0 \in \mathbb{R}^d$, gradient descent `(eq:gdconv:gd_update)` satisfies
>
> $$
> f(\boldsymbol{x}_T) - f^* \le \big(1 - 2\eta\mu(1 - \tfrac{\eta L}{2})\big)^T \big(f(\boldsymbol{x}_0) - f^*\big).
> $$
>
>   In particular, for $0 < \eta \le \frac{1}{L}$, $f(\boldsymbol{x}_T) - f^* \le (1 - \eta\mu)^T (f(\boldsymbol{x}_0) - f^*)$; at the optimal step $\eta = \frac{1}{L}$, $f(\boldsymbol{x}_T) - f^* \le (1 - \frac{\mu}{L})^T(f(\boldsymbol{x}_0) - f^*)$, attaining $f(\boldsymbol{x}_T) - f^* \le \epsilon(f(\boldsymbol{x}_0) - f^*)$ in $T \ge \frac{L}{\mu}\log(1/\epsilon)$ iterations.

**Proof.**
*Strong Convexity $\implies$ PL:* Following [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lem. 2.18, p. 9) and [[Source - Fall 2022|gradient_descent_unconstrained]] (Lem. 1, Slide 16), evaluate `(eq:gdconv:sc_quad)` at $\boldsymbol{y} = \boldsymbol{x}^*$ and complete the square:

$$
\begin{aligned}
f(\boldsymbol{x}) - f^* &\le \langle \nabla f(\boldsymbol{x}), \boldsymbol{x} - \boldsymbol{x}^* \rangle - \frac{\mu}{2}\lVert \boldsymbol{x}-\boldsymbol{x}^* \rVert_2^2 \\
  &= \frac{1}{2\mu}\lVert \nabla f(\boldsymbol{x}) \rVert_2^2 - \frac{1}{2}\left\|\sqrt{\mu}(\boldsymbol{x}-\boldsymbol{x}^*) - \frac{1}{\sqrt{\mu}}\nabla f(\boldsymbol{x})\right\|_2^2 \le \frac{1}{2\mu}\lVert \nabla f(\boldsymbol{x}) \rVert_2^2.
\end{aligned}
$$

*PL $\implies$ Quadratic Growth:* Define $h(\boldsymbol{x}) \triangleq \sqrt{f(\boldsymbol{x}) - f^*}$. For any $\boldsymbol{x} \notin \mathcal{X}^*$, $h$ is differentiable with $\nabla h(\boldsymbol{x}) = \frac{\nabla f(\boldsymbol{x})}{2\sqrt{f(\boldsymbol{x})-f^*}}$, so `(eq:gdconv:pl_def)` implies $\lVert \nabla h(\boldsymbol{x}) \rVert_2 \ge \sqrt{\mu/2}$. Following the normalized steepest-descent trajectory $\dot{\boldsymbol{z}}(s) = -\nabla h(\boldsymbol{z}(s))/\lVert \nabla h(\boldsymbol{z}(s)) \rVert_2$ from $\boldsymbol{z}(0) = \boldsymbol{x}$ (with unit speed $\lVert \dot{\boldsymbol{z}}(s) \rVert_2 = 1$), $\frac{\mathrm{d}}{\mathrm{d}s} h(\boldsymbol{z}(s)) = -\lVert \nabla h(\boldsymbol{z}(s)) \rVert_2 \le -\sqrt{\mu/2}$, so the path must enter $\mathcal{X}^*$ at some finite arc length $S \le \frac{h(\boldsymbol{x})}{\sqrt{\mu/2}}$. By the triangle inequality, $\mathrm{dist}(\boldsymbol{x}, \mathcal{X}^*) \le \lVert \boldsymbol{x} - \boldsymbol{z}(S) \rVert_2 \le S \le \sqrt{\frac{2}{\mu}(f(\boldsymbol{x}) - f^*)}$, giving $f(\boldsymbol{x}) - f^* \ge \frac{\mu}{2}\mathrm{dist}(\boldsymbol{x},\mathcal{X}^*)^2$.

*Global Linear Rate:* Following [[Source - Fall 2022|gradient_descent_unconstrained]] (Thm. 3, Slides 44–46) (and [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Thm. 3.9, pp. 17–18) for the general step-size proof), substituting `(eq:gdconv:pl_def)` into `(eq:gdconv:sufficient_decrease)` yields:
\[
  f(\boldsymbol{x}_{t+1}) - f^* \le f(\boldsymbol{x}_t) - f^* - \eta\left(1 - \frac{\eta L}{2}\right)\lVert \nabla f(\boldsymbol{x}_t) \rVert_2^2
  \le \Big(1 - 2\eta\mu\big(1 - \tfrac{\eta L}{2}\big)\Big)\big(f(\boldsymbol{x}_t) - f^*\big).
\]
Notice that for $0 < \eta \le \frac{1}{L}$, $1 - \frac{\eta L}{2} \ge \frac{1}{2}$ gives the factor $1 - \eta\mu$, and maximizing $2\eta\mu(1 - \eta L/2)$ over $\eta$ uniquely selects $\eta = \frac{1}{L}$ with factor $1 - \mu/L$. Unrolling to $T$ yields `(eq:gdconv:pl_contraction)`, and applying $\log(1/\rho) \ge 1 - \rho$ for $\rho = 1 - \mu/L$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lem. A.1, p. 66) gives the complexity $T \ge \frac{L}{\mu}\log(1/\epsilon)$. 

> [!tip] Why Overparameterized Models Satisfy PL Without Convexity
> Three canonical machine-learning landscapes illustrate PL without strong convexity:
> - **Underdetermined Least Squares ($n < d$):** For $\mathcal{L}_n(\boldsymbol{x}) = \frac{1}{n}\lVert \boldsymbol{X}\boldsymbol{x} - \boldsymbol{y} \rVert_2^2$ with $\boldsymbol{X} \in \mathbb{R}^{n \times d}$ of rank $n < d$, the Hessian $\frac{2}{n}\boldsymbol{X}^{\top}\boldsymbol{X}$ has $d-n$ zero eigenvalues ($\mu_{\mathrm{SC}} = 0$) and $\mathcal{L}_n^* = 0$. Yet $\lVert \nabla \mathcal{L}_n(\boldsymbol{x}) \rVert_2^2 = \frac{4}{n^2}(\boldsymbol{X}\boldsymbol{x}-\boldsymbol{y})^{\top} \boldsymbol{X}\boldsymbol{X}^{\top} (\boldsymbol{X}\boldsymbol{x}-\boldsymbol{y}) \ge \frac{4\lambda_{\min}(\boldsymbol{X}\boldsymbol{X}^{\top})}{n}(\mathcal{L}_n(\boldsymbol{x}) - \mathcal{L}_n^*)$, establishing global PL with $\mu = \frac{4}{n}\lambda_{\min}(\boldsymbol{X}\boldsymbol{X}^{\top})$ and convergence to the nearest minimizer [[Source - Fall 2022|gradient_descent_unconstrained]] (Slides 47–50) [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ex. 2.19, p. 9).
> - **Overparameterized Nonlinear Models:** For $f(\boldsymbol{x}) = \frac{1}{2}\lVert \Phi(\boldsymbol{x}) - \boldsymbol{y} \rVert_2^2$ with smooth $\Phi : \mathbb{R}^d \to \mathbb{R}^n$ ($d \ge n$) whose Jacobian satisfies $\lambda_{\min}(D\Phi(\boldsymbol{x})D\Phi(\boldsymbol{x})^{\top}) \ge \mu > 0$ (e.g., wide neural networks in the NTK regime), $\lVert \nabla f(\boldsymbol{x}) \rVert_2^2 = \lVert D\Phi(\boldsymbol{x})^{\top}(\Phi(\boldsymbol{x})-\boldsymbol{y}) \rVert_2^2 \ge 2\mu f(\boldsymbol{x}) \ge 2\mu(f(\boldsymbol{x})-f^*)$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ex. 2.21, pp. 9–10).
> - **Non-Convex Analytical Objectives:** $f(t) = t^2 + 3\sin^2(t)$ fails convexity ($f"(\pi/2) = -4 < 0$) yet satisfies $\frac{1}{40}$-PL [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ex. 2.20 & Lem. A.3, pp. 9, 67–68), and $f(\boldsymbol{x}) = \mathrm{dist}(\boldsymbol{x}; \Omega)^2$ for closed sets $\Omega$ (Fig. `fig:gdconv:pl_surface`) can exhibit curved, non-convex minimizer manifolds.

**Kurdyka–Łojasiewicz (KL) Generalization:** When PL fails globally or with quadratic exponent, objectives arising in practice (such as logistic losses on bounded sets or semi-algebraic functions including piecewise-linear ReLU networks; [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Rem. 2.23, Eq. (7), p. 10)) satisfy a local *Kurdyka–Łojasiewicz ($p$-Łojasiewicz)* inequality around any critical point $\boldsymbol{x}^*$:

$$
f(\boldsymbol{x}) - f(\boldsymbol{x}^*) \le \frac{1}{q\mu}\lVert \nabla f(\boldsymbol{x}) \rVert_2^q \quad \Big(\text{equivalently, } \lVert \nabla f(\boldsymbol{x}) \rVert_2 \ge c\big(f(\boldsymbol{x}) - f(\boldsymbol{x}^*)\big)^{\theta}\Big),
$$

on a neighborhood $\Omega \ni \boldsymbol{x}^*$, where $\frac{1}{p} + \frac{1}{q} = 1$ ($p \ge 1$) and $\theta = \frac{1}{q} \in [0, 1)$ is the KL exponent. Setting $\theta = 1/2$ ($p=q=2$) recovers the $\mu$-PL linear rate; $\theta \in (1/2, 1)$ yields polynomial sublinear decay $f(\boldsymbol{x}_t)-f(\boldsymbol{x}^*) = O(t^{-\frac{1}{2\theta-1}})$ (e.g., $\theta = 3/4$ on $f(x)=x^4$ gives $O(t^{-2})$), whereas $\theta \in [0, 1/2)$ yields finite-time or superlinear convergence. Similarly, when PL degrades near boundaries, non-uniform PL bounds $\lVert \nabla f(\boldsymbol{x}) \rVert_2^2 \ge 2\mu(\boldsymbol{x})(f(\boldsymbol{x})-f^*)$ motivate normalized gradient updates [[Source - Fall 2022|gradient_descent_unconstrained]] (Slide 50).

### Strict-Saddle Avoidance, Stable Manifolds & Perturbed Gradient Descent

In general non-convex landscapes, $\epsilon$-FOSP convergence guarantees only $\lVert \nabla f(\boldsymbol{x}) \rVert_2 \le \epsilon$, which can trap iterates near saddle points rather than local minima ([[Source - Fall 2022|gradient_descent_unconstrained]] (Slide 88)). For $f \in C^2(\mathbb{R}^d)$, stationary points $\nabla f(\bar{\boldsymbol{x}}) = \boldsymbol{0}$ fall into three spectral regimes of $\nabla^2 f(\bar{\boldsymbol{x}})$, as detailed in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.1.1.1, pp. 276–277) and [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 7.1.1, pp. 212–213):
- **Strict local minimum:** $\lambda_{\min}(\nabla^2 f(\bar{\boldsymbol{x}})) > 0$ ($\nabla^2 f(\bar{\boldsymbol{x}}) \succ \boldsymbol{0}$).
- **Strict saddle point:** $\lambda_{\min}(\nabla^2 f(\bar{\boldsymbol{x}})) < 0$ (at least one strictly negative curvature escape direction).
- **Non-strict (degenerate) saddle / flat plateau:** $\lambda_{\min}(\nabla^2 f(\bar{\boldsymbol{x}})) = 0$ with higher-order descent (e.g., $f(x_1, x_2) = x_1^3 - 3x_1 x_2^2$) or zero-curvature directions.

> [!important] Theorem: Almost-Sure Strict-Saddle Avoidance via the Stable Manifold Theorem
> Let $f \in C^2(\mathbb{R}^d)$ be $L$-smooth, and let $\mathcal{S}_{\mathrm{strict}} \triangleq \{\bar{\boldsymbol{x}} \in \mathbb{R}^d : \nabla f(\bar{\boldsymbol{x}}) = \boldsymbol{0},  \lambda_{\min}(\nabla^2 f(\bar{\boldsymbol{x}})) < 0\}$ be the set of strict saddle points. If $0 < \eta < \frac{1}{L}$, then the gradient-descent map $T(\boldsymbol{x}) = \boldsymbol{x} - \eta\nabla f(\boldsymbol{x})$ is a global diffeomorphism of $\mathbb{R}^d$, and the set of initializations $\{\boldsymbol{x}_0 \in \mathbb{R}^d : \lim_{t\to\infty} T^t(\boldsymbol{x}_0) \in \mathcal{S}_{\mathrm{strict}}\}$ has Lebesgue measure zero. Consequently, for any initialization distribution $\mu_0$ absolutely continuous with respect to Lebesgue measure (e.g., $\mathcal{N}(\boldsymbol{0}, \sigma^2\boldsymbol{I}_d)$),
>
> $$
> \mathbb{P}_{\boldsymbol{x}_0 \sim \mu_0}\!\left(\lim_{t\to\infty} \boldsymbol{x}_t = \bar{\boldsymbol{x}}\right) = 0 \qquad \forall \bar{\boldsymbol{x}} \in \mathcal{S}_{\mathrm{strict}}
> $$
>
> [[Source - Fall 2022|gradient_descent_unconstrained]] (Thm. 8, Slides 90–92).

**Proof Sketch.**
Because $f$ is $L$-smooth and $0 < \eta < 1/L$, the Jacobian $DT(\boldsymbol{x}) = \boldsymbol{I}_d - \eta\nabla^2 f(\boldsymbol{x})$ satisfies $\lambda_{\min}(DT(\boldsymbol{x})) \ge 1 - \eta L > 0$ for every $\boldsymbol{x} \in \mathbb{R}^d$, so $DT(\boldsymbol{x})$ is everywhere invertible (and $T$ is a global diffeomorphism). At any strict saddle $\bar{\boldsymbol{x}} \in \mathcal{S}_{\mathrm{strict}}$, $T(\bar{\boldsymbol{x}}) = \bar{\boldsymbol{x}}$ and $DT(\bar{\boldsymbol{x}})$ possesses at least one eigenvalue $1 - \eta\lambda_{\min}(\nabla^2 f(\bar{\boldsymbol{x}})) > 1$. By the Center–Stable Manifold Theorem from dynamical systems, the local stable set of trajectories converging to $\bar{\boldsymbol{x}}$ is contained in a smooth submanifold of dimension at most $d - 1$, which has $d$-dimensional Lebesgue measure zero. Since $T^{-1}$ is a diffeomorphism (preserving null sets), the countable union $\bigcup_{t=0}^\infty T^{-t}(\mathcal{W}_{\mathrm{loc}}^s(\bar{\boldsymbol{x}}))$ also has Lebesgue measure zero. Concrete illustration [[Source - Fall 2022|gradient_descent_unconstrained]] (Slides 91–92): on $f(x_1, x_2) = \frac{1}{2}x_1^2 + \frac{1}{4}x_2^4 - \frac{1}{2}x_2^2$, $\nabla f(\boldsymbol{x}) = (x_1, x_2^3 - x_2)^{\top}$ vanishes at the strict saddle $\bar{\boldsymbol{x}}=(0,0)$ ($\lambda_{\min} = -1$) and global minima $(0, \pm 1)$; only initializations on the $1$D line $\{(z,0) : z \in \mathbb{R}\}$ converge to $(0,0)$. 

**From Asymptotic Avoidance to Polynomial-Time Escape (Perturbed GD):** Although Theorem `thm:gdconv:strict_saddle` guarantees asymptotic escape with probability $1$, vanilla GD initialized within $\delta$ of the stable manifold requires $\Omega(\log(1/\delta))$ steps per saddle and can suffer exponential $O(\exp(d))$ slowdown across sequences of saddles. As noted in [[Source - Fall 2022|gradient_descent_unconstrained]] (Slide 89), injecting controlled isotropic perturbations converts asymptotic avoidance into polynomial-time convergence to an $(\epsilon, \sqrt{\rho\epsilon})$-*second-order stationary point* ($\epsilon$-SOSP), defined for $\rho$-Hessian-Lipschitz $f$ ($\lVert \nabla^2 f(\boldsymbol{x}) - \nabla^2 f(\boldsymbol{y}) \rVert_2 \le \rho\lVert \boldsymbol{x}-\boldsymbol{y} \rVert_2$) by

$$
\lVert \nabla f(\boldsymbol{x}) \rVert_2 \le \epsilon \qquad \text{and} \qquad \lambda_{\min}\big(\nabla^2 f(\boldsymbol{x})\big) \ge -\sqrt{\rho\epsilon}.
$$

Specifically, whenever $\lVert \nabla f(\boldsymbol{x}_t) \rVert_2 \le g_{\mathrm{thresh}}$, Perturbed Gradient Descent (Algorithm `alg:gdconv:pgd`) draws a single ball perturbation $\boldsymbol{\xi}_t \sim \mathrm{Unif}(\mathbb{B}_0(r))$ (or Gaussian noise) and runs standard GD for $t_{\mathrm{thresh}}$ steps. Geometrically, the set of points near $\boldsymbol{x}_t$ that remain trapped in the saddle neighborhood for $t_{\mathrm{thresh}}$ steps forms a thin "pancake" of thickness $O(r / \sqrt{d})$ along the negative-curvature eigenvector $\boldsymbol{e}_{\min}$; a uniform ball kick of radius $r$ misses this thin slab with high probability, guaranteeing an objective drop of $\Omega(\sqrt{\epsilon^3/\rho})$ within $\tilde{O}(1)$ steps and locating an $\epsilon$-SOSP in $\tilde{O}(L(f(\boldsymbol{x}_0)-f^*)/\epsilon^2)$ iterations (only polylogarithmic in $d$).

## Algorithmic Implementation & Comparative Synthesis

In practice, neither the global Lipschitz constant $L$ nor the strong-convexity modulus $\mu$ is known a priori; moreover, non-convex polynomial or deep-network losses violate global smoothness ($\lVert \nabla^2 f(\boldsymbol{x}) \rVert \to \infty$ as $\lVert \boldsymbol{x} \rVert \to \infty$; [[Source - Fall 2022|gradient_descent_unconstrained]] (Slide 31)). Two adaptive deterministic mechanisms eliminate manual tuning of $L$: **Armijo–Goldstein backtracking line search** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.2.2.2, Eq. (8.27), p. 286) [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 7.1.1, pp. 229–230) and the **Barzilai–Borwein (BB) two-point spectral step size**.

Using the parameter displacement $\boldsymbol{s}_{t-1} \triangleq \boldsymbol{x}_t - \boldsymbol{x}_{t-1}$ and gradient difference $\boldsymbol{y}_{t-1} \triangleq \nabla f(\boldsymbol{x}_t) - \nabla f(\boldsymbol{x}_{t-1})$ from quasi-Newton secant theory [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eqs. (8.45)–(8.46), p. 290), approximating the Hessian by a scalar multiple $\alpha_t \boldsymbol{I}_d$ in the secant equation $\alpha_t \boldsymbol{s}_{t-1} \approx \boldsymbol{y}_{t-1}$ in least squares ($\min_{\alpha} \lVert \alpha\boldsymbol{s}_{t-1} - \boldsymbol{y}_{t-1} \rVert_2^2$ or $\min_{\alpha} \lVert \boldsymbol{s}_{t-1} - \alpha^{-1} \boldsymbol{y}_{t-1} \rVert_2^2$) produces the long and short Barzilai–Borwein step sizes:

$$
\eta_t^{\mathrm{BB1}} = \frac{\lVert \boldsymbol{s}_{t-1} \rVert_2^2}{\langle \boldsymbol{s}_{t-1}, \boldsymbol{y}_{t-1} \rangle}, \qquad
  \eta_t^{\mathrm{BB2}} = \frac{\langle \boldsymbol{s}_{t-1}, \boldsymbol{y}_{t-1} \rangle}{\lVert \boldsymbol{y}_{t-1} \rVert_2^2},
$$

whenever $\langle \boldsymbol{s}_{t-1}, \boldsymbol{y}_{t-1} \rangle > 0$. By Cauchy–Schwarz, $0 < \eta_t^{\mathrm{BB2}} \le \eta_t^{\mathrm{BB1}}$, and on $\mu$-strongly convex $L$-smooth functions both lie inside $[1/L, 1/\mu]$ as inverse Rayleigh quotients of the path-averaged Hessian $\overline{\boldsymbol{H}}_{t-1} = \int_0^1 \nabla^2 f(\boldsymbol{x}_{t-1} + \tau\boldsymbol{s}_{t-1}) \mathrm{d}\tau$. Because BB step sizes deliberately exceed $1/L$ along curved directions to break zig-zagging, pairing BB initialization with Armijo backtracking (or a non-monotone reference window $f_{\mathrm{ref}} = \max_{0 \le j < M} f(\boldsymbol{x}_{t-j})$) preserves global convergence without sacrificing spectral acceleration.

Furthermore, for any $L$-smooth function and Armijo parameter $c_1 \in (0, 1)$ (typically $c_1 = 10^{-4}$; [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (p. 286)), the Descent Lemma `(eq:gdconv:sufficient_decrease)` guarantees that the Armijo sufficient-decrease test

$$
f(\boldsymbol{x}_t - \eta\nabla f(\boldsymbol{x}_t)) \le f(\boldsymbol{x}_t) - c_1 \eta \lVert \nabla f(\boldsymbol{x}_t) \rVert_2^2
$$

holds automatically whenever $1 - \frac{\eta L}{2} \ge c_1 \iff \eta \le \frac{2(1 - c_1)}{L}$. With shrink factor $\beta \in (0, 1)$, backtracking from $\eta_{\mathrm{init}}$ terminates in at most $\lceil \log(\frac{2(1-c_1)}{L\eta_{\mathrm{init}}}) / \log \beta \rceil_+$ function evaluations and guarantees $\eta_t \ge \min\{\eta_{\mathrm{init}}, \frac{2\beta(1-c_1)}{L}\}$, immediately restoring $O(1/T)$ convex and $(1 - c_{\mathrm{arm}}\mu/L)^T$ strongly-convex/PL convergence without knowing $L$.

> [!example] Algorithm: Adaptive Deterministic Gradient Descent (Armijo Backtracking + Optional BB)
> **Input:** Differentiable $f : \mathbb{R}^d \to \mathbb{R}$; initial point $\boldsymbol{x}_0 \in \mathbb{R}^d$; tolerance $\epsilon > 0$; Armijo constant $c_1 \in (0, 1/2)$ (e.g., $10^{-4}$); shrink factor $\beta \in (0, 1)$ (e.g., $0.5$); safeguarded step bounds $0 < \eta_{\min} \le \eta_0 \le \eta_{\max}$; mode $UseBB \in \{\text{true}, \text{false}\}$.
> **Output:** Approximate stationary/optimal point $\boldsymbol{x}_T$.
> $\boldsymbol{g}_0 \gets \nabla f(\boldsymbol{x}_0)$
> **for** $t = 0, 1, 2, \dots, T_{\max}-1$:
>
>   **if** $\lVert \boldsymbol{g}_t \rVert_2 \le \epsilon$:
>
>     **return** $\boldsymbol{x}_t$
>
>   **if** $UseBB$ **and** $t \ge 1$ **and** $\langle \boldsymbol{s}_{t-1}, \boldsymbol{y}_{t-1} \rangle > 0$:
>
>     $\eta \gets \mathrm{clip}\!\left(\frac{\langle \boldsymbol{s}_{t-1}, \boldsymbol{s}_{t-1} \rangle}{\langle \boldsymbol{s}_{t-1}, \boldsymbol{y}_{t-1} \rangle},  \eta_{\min},  \eta_{\max}\right)$  *// BB1 spectral trial step*
>
>   **else**:
>
>     $\eta \gets \eta_0$  *// Fixed upper trial step*
>
>   **while** $f(\boldsymbol{x}_t - \eta \boldsymbol{g}_t) > f(\boldsymbol{x}_t) - c_1 \eta \lVert \boldsymbol{g}_t \rVert_2^2$:
>
>     $\eta \gets \beta \eta$  *// Armijo sufficient-decrease backtracking*
>
>   $\boldsymbol{x}_{t+1} \gets \boldsymbol{x}_t - \eta \boldsymbol{g}_t$
>   $\boldsymbol{g}_{t+1} \gets \nabla f(\boldsymbol{x}_{t+1})$
>   $\boldsymbol{s}_t \gets \boldsymbol{x}_{t+1} - \boldsymbol{x}_t$; \quad $\boldsymbol{y}_t \gets \boldsymbol{g}_{t+1} - \boldsymbol{g}_t$
>
> **return** $\boldsymbol{x}_{T_{\max}}$

> [!example] Algorithm: Perturbed Gradient Descent (PGD) for Strict-Saddle Escape
> **Input:** $L$-smooth, $\rho$-Hessian-Lipschitz $f : \mathbb{R}^d \to \mathbb{R}$; initial $\boldsymbol{x}_0 \in \mathbb{R}^d$; step size $\eta \le 1/L$; threshold $g_{\mathrm{thresh}} = \epsilon$; ball radius $r = \tilde{\Theta}(\epsilon / L)$; escape window $t_{\mathrm{thresh}} = \tilde{\Theta}(L / \sqrt{\rho\epsilon})$; drop threshold $f_{\mathrm{thresh}}$.
> **Output:** $\epsilon$-Second-Order Stationary Point ($\epsilon$-SOSP) $\hat{\boldsymbol{x}}$.
> $t_{\mathrm{noise}} \gets -t_{\mathrm{thresh}} - 1$
> **for** $t = 0, 1, 2, \dots, T_{\max}-1$:
>
>   $\boldsymbol{g}_t \gets \nabla f(\boldsymbol{x}_t)$
>   **if** $\lVert \boldsymbol{g}_t \rVert_2 \le g_{\mathrm{thresh}}$ **and** $t - t_{\mathrm{noise}} > t_{\mathrm{thresh}}$:
>
>     $\tilde{\boldsymbol{x}}_t \gets \boldsymbol{x}_t$; \quad $t_{\mathrm{noise}} \gets t$
>     Sample $\boldsymbol{\xi}_t \sim \mathrm{Unif}(\mathbb{B}_0(r))$ and set $\boldsymbol{x}_t \gets \tilde{\boldsymbol{x}}_t + \boldsymbol{\xi}_t$  *// Perturb off stable manifold*
>
>   **if** $t - t_{\mathrm{noise}} = t_{\mathrm{thresh}}$ **and** $f(\boldsymbol{x}_t) - f(\tilde{\boldsymbol{x}}_{t_{\mathrm{noise}}}) > -f_{\mathrm{thresh}}$:
>
>     **return** $\tilde{\boldsymbol{x}}_{t_{\mathrm{noise}}}$  *// Confirmed $\epsilon$-SOSP (no escape direction found)*
>
>   $\boldsymbol{x}_{t+1} \gets \boldsymbol{x}_t - \eta \nabla f(\boldsymbol{x}_t)$
>
> **return** $\boldsymbol{x}_{T_{\max}}$

> [!info] Cross-Book Synthesis: First-Order Convergence, Conditioning & Second-Order Acceleration
> Synthesizing the optimization chapters across our five sources exposes a sharp division between *non-asymptotic worst-case guarantees* and *geometric preconditioning*:
> - **Theoretical Oracle Hierarchy:** [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Table 1, p. 4) and [[Source - Fall 2022|gradient_descent_unconstrained]] (Slides 15, 44, 61, 82) categorize deterministic GD cleanly by landscape assumption: $O(1/\epsilon^2)$ to $\epsilon$-FOSP in smooth non-convex problems, $O(1/\epsilon)$ under global/local convexity, and $O(\kappa \log(1/\epsilon))$ under $\mu$-strong convexity or $\mu$-PL.
> - **Curvature Skew vs. Preconditioning:** Why does fixed-metric GD stall in elongated valleys? As analyzed in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.2.3 & 8.3.1, pp. 286–289), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 7.1.1, p. 230), and [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 7.3–7.4, pp. 218–229), $\nabla f$ points normal to elliptical contours rather than toward $\boldsymbol{x}^*$. Three remedies appear across the literature: (i) *data/activation normalization* ([[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 7.4, pp. 226–229)), (ii) *momentum / Nesterov acceleration* reducing $\kappa$ to $\sqrt{\kappa}$ ([[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.2.4); [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 7.1.2)), and (iii) *second-order / quasi-Newton preconditioning* ($\boldsymbol{P}^{-1}$ or $\boldsymbol{H}_t^{-1}$ in Newton and L-BFGS) transforming skewed ellipses into isotropic circles at $O(d^3)$ or $O(Md)$ per-step cost ([[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.3, pp. 289–291)).

**Table:** Deterministic First-Order Convergence Rates, Step-Size Regimes, and Iteration Complexity.

| **Function Class** | **Step Size** | **Convergence Metric** | **Complexity $T(\epsilon)$** | **Primary Source & Notes** |
| — | — | — | — | — |
| $L$-Smooth Non-Convex | $0 < \eta \le \frac{1}{L}$ | $\min_{t < T}\lVert \nabla f(\boldsymbol{x}_t) \rVert_2 \le \epsilon$ | $O\!\left(\frac{L\Delta_0}{\epsilon^2}\right)$ | [[Source - Fall 2022]] (Thm. 6); $\Delta_0 = f(\boldsymbol{x}_0)-f^*$; may stall at saddles. |
| $L$-Smooth + $\rho$-Hessian-Lip. (PGD) | $\eta \le \frac{1}{L}$, $\boldsymbol{\xi}_t \in \mathbb{B}_0(r)$ | $(\epsilon,\sqrt{\rho\epsilon})$-SOSP | $\tilde{O}\!\left(\frac{L\Delta_0}{\epsilon^2}\right)$ | [[Source - Fall 2022]] (Slide 89); escapes strict saddles in polylog$(d)$. |
| Convex + $L$-Smooth | $0 < \eta \le \frac{1}{L}$ | $f(\boldsymbol{x}_T) - f^* \le \epsilon$ | $O\!\left(\frac{L R_0^2}{\epsilon}\right)$ | [[Source - Handbook of Convergence Theorems]] (Thm. 3.4); $R_0 = \lVert \boldsymbol{x}_0-\boldsymbol{x}^* \rVert_2$; $O(\frac{LR_0}{\epsilon})$ for $\lVert \nabla f \rVert_2 \le \epsilon$. |
| Local / Global KL ($\theta \in (\frac{1}{2}, 1)$) | $0 < \eta \le \frac{1}{L}$ | $f(\boldsymbol{x}_T) - f^* \le \epsilon$ | $O\!\left(\epsilon^{-(2\theta-1)}\right)$ | [[Source - Handbook of Convergence Theorems]] (Rem. 2.23); semi-algebraic / quartics ($\theta=3/4 \implies O(1/\sqrt{\epsilon})$). |
| $\mu$-PL + $L$-Smooth (Non-Convex OK) | $\eta = \frac{1}{L}$ | $f(\boldsymbol{x}_T) - f^* \le \epsilon \Delta_0$ | $\kappa \log\!\left(\frac{1}{\epsilon}\right)$ | [[Source - Handbook of Convergence Theorems]] (Thm. 3.9); [[Source - Fall 2022]] (Thm. 3); global from any $\boldsymbol{x}_0$. |
| $\mu$-SC + $L$-Smooth (Standard) | $\eta = \frac{1}{L}$ | $\lVert \boldsymbol{x}_T - \boldsymbol{x}^* \rVert_2^2 \le \epsilon R_0^2$ | $\kappa \log\!\left(\frac{1}{\epsilon}\right)$ | [[Source - Handbook of Convergence Theorems]] (Thm. 3.6); [[Source - Fall 2022]] (Thm. 1); rate $(1 - 1/\kappa)^T$. |
| $\mu$-SC + $L$-Smooth (Optimal $\eta^*$) | $\eta^* = \frac{2}{\mu + L}$ | $\lVert \boldsymbol{x}_T - \boldsymbol{x}^* \rVert_2 \le \epsilon R_0$ | $\frac{\kappa + 1}{2}\log\!\left(\frac{1}{\epsilon}\right)$ | [[Source - Probabilistic Machine Learning]] (Eq. (8.29)); exact factor $\big(\frac{\kappa-1}{\kappa+1}\big)^T$. |

## Visual Diagnostics & Geometric Structure

Figure `fig:gdconv:hierarchy` synthesizes the structural containment and implication hierarchy across $L$-smooth optimization classes, pairing each function family with its sharpest non-asymptotic deterministic GD rate. Notice that convexity is neither necessary nor sufficient for linear convergence: $\mu$-PL lies strictly outside the convex cone while guaranteeing $(1-\mu/L)^T$ decay, whereas convex functions without quadratic growth ($f(\boldsymbol{x}) = \lVert \boldsymbol{x} \rVert_2^4$) degrade to sublinear rates.

> [!example] Visual Diagram (LaTeX / PDF)
> *Function-class inclusion and implication hierarchy inside $L$-smooth optimization. $\mu$-strong convexity implies $\mu$-PL (Theorem `thm:gdconv:pl_rate`), which in turn implies Quadratic Growth (QG), but $\mu$-PL encompasses non-convex objectives with non-isolated global minimizers while retaining the global linear contraction factor $(1-\mu/L)^T$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:gdconv:rates` compares the exact convergence trajectories across landscape classes (Panel a) and illustrates step-size stability and eigenmode damping on an anisotropic $\kappa = 10$ quadratic (Panel b). Notice in Panel (b) that $\eta = 1/L$ annihilates the high-curvature mode in one step yet lags behind $\eta^* = 2/(\mu+L)$ on the flat valley floor, while $\eta = 1.95/L$ stalls due to violent cross-valley oscillations and $\eta = 2.08/L > 2/L$ diverges exponentially.

> [!example] Visual Diagram (LaTeX / PDF)
> *Analytic convergence dynamics of deterministic gradient descent. **(a)** Sublinear $O(1/T)$ convex and $O(t^{-2})$ KL decay vs. linear contraction under $\mu$-strong convexity / $\mu$-PL, highlighting the acceleration from $\eta=1/L$ to $\eta^* = 2/(\mu+L)$ and the severe slowdown when $\kappa = L/\mu$ increases from $4$ to $25$. **(b)** Exact trajectory ratio on $f(\boldsymbol{x}) = \frac{1}{2}(0.1x_1^2 + x_2^2)$ initialized at $\boldsymbol{x}_0 = (1,1)^{\top}$: $\eta = 0.2/L$ creeps slowly, $\eta = 1/L$ immediately kills the steep $x_2$ mode, $\eta^* = 2/(\mu+L)$ achieves minimax optimal decay, $\eta = 1.95/L$ rings across the valley walls, and $\eta = 2.08/L > 2/L$ diverges exponentially.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Three 300 DPI benchmark figures from the synthesized texts confirm these geometric behaviors directly on canonical 2D and 3D landscapes. First, Figure `fig:gdconv:stepsize_osc` contrasts safe monotone progress ($\eta = 0.100$) against sustained limit-cycle oscillation across valley walls ($\eta = 0.600$) on a convex Rosenbrock-like valley [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.2.2.1, Fig. 8.11, p. 285). Second, Figure `fig:gdconv:condition_zigzag` demonstrates the exact-line-search contraction factor $(\frac{\kappa-1}{\kappa+1})^2$: when $\kappa(\boldsymbol{A}) = 30.234$, steepest-descent steps zigzag almost orthogonally to the valley axis, whereas at $\kappa(\boldsymbol{A}) = 1.854$ two steps nearly reach the minimizer [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.2.3, Fig. 8.12, p. 287). Third, Figure `fig:gdconv:pl_surface` visualizes the non-convex squared-distance function $f(\boldsymbol{x}) = \mathrm{dist}(\boldsymbol{x}; \Omega)^2$ satisfying the global $\mu$-PL condition, whose minimizer set $\Omega$ forms a curved non-convex valley free of spurious local minima [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ex. 2.20, Fig. 1, p. 9).

> [!note] Source Figure
> *Figure:* Effect of fixed step size on a curved convex valley [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 8.11, p. 285 (PDF p. 315)): **(a)** $\eta = 0.100$ produces slow, stable monotone progress; **(b)** $\eta = 0.600$ violates local curvature stability ($\eta > 2/L_{\mathrm{loc}}$) and bounces persistently across the canyon walls without reaching $(1,1)^{\top}$. (`probabilisticml_kevinmurphy_introduction_p315_fig1.png`)

> [!note] Source Figure
> *Figure:* Steepest descent with exact line search under contrasting Hessian condition numbers [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 8.12, p. 287 (PDF p. 317)): **(a)** $\kappa(\boldsymbol{A}) = 30.234$ yields $(\frac{\kappa-1}{\kappa+1})^2 \approx 0.876$, producing pronounced orthogonal zig-zagging; **(b)** $\kappa(\boldsymbol{A}) = 1.854$ yields $(\frac{\kappa-1}{\kappa+1})^2 \approx 0.090$, converging almost directly to the minimum. (`probabilisticml_kevinmurphy_introduction_p317_fig1.png`)

> [!note] Source Figure
> *Figure:* Surface of the non-convex $\mu$-PL function $f(\boldsymbol{x}) = \mathrm{dist}(\boldsymbol{x}; \Omega)^2$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Fig. 1, p. 9). Although the global minimizer manifold $\Omega$ (white curve) is non-convex and non-isolated, every stationary point is globally optimal, enabling global linear convergence from any initialization. (`stochastic_gradient_decent_algorithm_explain_p9_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

Even when formal convergence theorems hold, naive deterministic GD frequently stalls due to scaling pathologies, unbounded curvature, or degenerate saddles.

> [!warning] Pitfall 1: Zig-Zagging Under Extreme Conditioning ($\kappa \gg 1$) & Unnormalized Features
> **Symptom:** Gradient descent makes rapid initial objective decrease for $2$–$5$ steps, then stalls with tiny zig-zagging steps while $\lVert \boldsymbol{x}_t - \boldsymbol{x}^* \rVert_2$ barely decays. \\
> **Root Cause:** In linear least squares $f(\boldsymbol{x}) = \frac{1}{n}\lVert \boldsymbol{X}\boldsymbol{x} - \boldsymbol{y} \rVert_2^2$, the Hessian condition number is $\kappa(\boldsymbol{X}^{\top}\boldsymbol{X}) = \big(\sigma_{\max}(\boldsymbol{X})/\sigma_{\min}(\boldsymbol{X})\big)^2$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ex. 7.2, p. 230). Unstandardized inputs spanning disparate scales (e.g., height in meters vs. platelet counts $\sim 3\times 10^5$) inflate $\kappa$ quadratically [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 7.4.1, p. 226), forcing $\eta < 2/L$ to crawl along $\lambda_{\min}$ at rate $1 - 1/\kappa$ [[Source - Foundations|deep_learning_foundations_concepts]] (Eq. (7.30), p. 219). \\
> **Remedy:** Standardize inputs to zero mean and unit variance, apply batch/layer normalization to smooth hidden Hessians [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 7.4.2–7.4.3, pp. 227–229), employ Barzilai–Borwein steps (Algorithm `alg:gdconv:armijo_bb`) or momentum [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.2.4), or precondition ([[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (p. 230); L-BFGS in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.3.2)).

> [!warning] Pitfall 2: Instability Beyond $\eta = 2/L$ & Breakdown of Global Smoothness
> **Symptom:** Loss spikes violently or overflows to `NaN` despite starting from a seemingly small learning rate $\eta$. \\
> **Root Cause:** First, any constant $\eta > 2/\lambda_{\max}(\nabla^2 f)$ induces $\lvert 1 - \eta\lambda_{\max} \rvert > 1$, causing exponential amplification along the top eigenvector [[Source - Foundations|deep_learning_foundations_concepts]] (p. 219) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. (8.20), p. 284). Second, many canonical ML objectives are *not globally $L$-smooth*: in quartic phase retrieval $\mathcal{L}_n(\theta) = \frac{1}{n}\sum_i (Y_i - X_i^2\theta^2)^2$, $\nabla^2 \mathcal{L}_n(\theta) = \frac{12}{n}(\sum X_i^4)\theta^2 - \frac{4}{n}\sum Y_i X_i^2$ grows unboundedly as $\lvert \theta \rvert \to \infty$, so any fixed $\eta > 0$ fails outside a bounded ball $\mathbb{B}(\hat{\theta}_n, r_0)$ [[Source - Fall 2022|gradient_descent_unconstrained]] (Slides 30–32). \\
> **Remedy:** Verify local initialization $\lVert \boldsymbol{x}_0 - \boldsymbol{x}^* \rVert_2 \le r_0$ when $L$ is only local, or replace fixed $\eta$ with Armijo backtracking (Algorithm `alg:gdconv:armijo_bb`), which adapts dynamically to local curvature $L_{\mathrm{loc}}$ without prior knowledge of $L$.

> [!warning] Pitfall 3: Non-Strict Saddles, Flat Plateaus & Deterministic Manifold Traps
> **Symptom:** Gradient norm drops below $\epsilon$, yet objective value remains far above $f^*$ and Hessian exhibits near-zero or negative eigenvalues. \\
> **Root Cause:** Theorem `thm:gdconv:strict_saddle` guarantees almost-sure avoidance *only* for strict saddles ($\lambda_{\min}(\nabla^2 f(\bar{\boldsymbol{x}})) < 0$). On non-strict saddles like the monkey saddle $f(x_1,x_2) = x_1^3 - 3x_1x_2^2$ ($\nabla f(\boldsymbol{0})=\boldsymbol{0}$, $\nabla^2 f(\boldsymbol{0})=\boldsymbol{0}$) or flat plateaus ($\nabla f \approx \boldsymbol{0}$, $\nabla^2 f \approx \boldsymbol{0}$), $DT(\bar{\boldsymbol{x}}) = \boldsymbol{I}_d$ lacks an unstable direction; GD and PGD can stall indefinitely. Moreover, symmetric initialization (e.g., equal hidden weights, or $(z, 0)$ in [[Source - Fall 2022|gradient_descent_unconstrained]] (Slide 92)) lands on the zero-measure stable manifold. \\
> **Remedy:** Following [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 7.2.5, p. 216), break symmetries via random initialization; inject ball perturbations when $\lVert \nabla f(\boldsymbol{x}_t) \rVert_2 \le \epsilon$ (Algorithm `alg:gdconv:pgd`) to escape strict saddles in polylog$(d)$ time, and use cubic-regularized Newton on degenerate saddles.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[Gradient Descent]], [[Polyak-Lojasiewicz (PL) Condition]]
- **Prerequisites (Directed $\leftarrow$):** [[Condition Number]], [[Convexity & Subgradients]], [[Gradient Descent]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Gradient Boosted Decision Trees (GBDT)]], [[Momentum & Nesterov NAG]], [[Natural Gradient & K-FAC]], [[Polyak-Lojasiewicz (PL) Condition]], [[Stochastic Gradient Descent (SGD)]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Condition Number]], [[Convexity & Subgradients]], [[Gradient Descent]], [[Momentum & Nesterov NAG]], [[Polyak-Lojasiewicz (PL) Condition]], [[Stochastic Gradient Descent (SGD)]]
- **Synthesized Sources:** [[Source - Fall 2022]], [[Source - Foundations]], [[Source - Handbook of Convergence Theorems]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
