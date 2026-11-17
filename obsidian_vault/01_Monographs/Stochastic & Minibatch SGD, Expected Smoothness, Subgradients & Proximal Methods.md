---
type: monograph
domain: "D01"
parent_topic: "[[PT 01 - Convex, Smooth & Stochastic Optimization]]"
sub_topic: "[[ST 01 - Stochastic, Adaptive, Proximal & Second-Order Optimizers]]"
cheatsheet: "[[Cheatsheet - Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]"
part_of: 
  - "[[ST 01 - Stochastic, Adaptive, Proximal & Second-Order Optimizers]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
concepts: 
  - "[[Proximal & Subgradient Methods]]"
  - "[[SVRG & Variance Reduction]]"
  - "[[Stochastic Gradient Descent (SGD)]]"
prerequisites: 
  - "[[Convexity & Subgradients]]"
  - "[[Gradient Descent]]"
  - "[[Stochastic Gradient Descent (SGD)]]"
enables: 
  - "[[Proximal & Subgradient Methods]]"
  - "[[SVRG & Variance Reduction]]"
related: 
  - "[[Convexity & Subgradients]]"
  - "[[Gradient Descent]]"
  - "[[Linear & Ridge Regression]]"
  - "[[Proximal & Subgradient Methods]]"
  - "[[SVRG & Variance Reduction]]"
  - "[[Stochastic Gradient Descent (SGD)]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Handbook of Convergence Theorems]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-01
---

# Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods

> [!abstract] Monograph Navigation
> - **Domain:** [[D01 - Mathematical Foundations for Machine Intelligence|Mathematical Foundations for Machine Intelligence]]
> - **Parent / Sub-Topic:** [[PT 01 - Convex, Smooth & Stochastic Optimization|Convex, Smooth & Stochastic Optimization]] → [[ST 01 - Stochastic, Adaptive, Proximal & Second-Order Optimizers|Stochastic, Adaptive, Proximal & Second-Order Optimizers]]
> - **Visual Cheatsheet:** [[Cheatsheet - Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]
> - **Description:** Derives finite-sum SGD convergence to an O(eta * sigma^2) noise ball, diminishing step-size schedules, expected smoothness for minibatching, subgradient methods, and proximal operator splitting (ISTA/LASSO).

Modern empirical risk minimization (ERM) routinely couples massive finite-sum objectives over millions of data points with nonsmooth sparsity-inducing penalties or hard parameter constraints. Deterministic full-batch gradient descent incurs $\mathcal{O}(n)$ gradient evaluations per update and cannot evaluate derivatives at kinks or domain boundaries, while naive stochastic approximation suffers persistent gradient variance that prevents convergence under constant step sizes. This monograph develops a unified non-asymptotic and geometric theory bridging expected smoothness and variance transfer for smooth finite sums, Robbins–Monro decay schedules and constant-step noise balls, minibatch scaling laws, control-variate variance reduction (SVRG/SAGA), subdifferential calculus for nonsmooth convex losses, and forward–backward proximal operator splitting (PGD, SPGD, and ISTA).

## Problem Formulation & Unified Notation

We study both unconstrained smooth finite-sum optimization and composite nonsmooth minimization over $\mathbb{R}^d$. In the smooth finite-sum setting [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ch. 4, p. 18) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.1, p. 293), we seek

$$
\mathrm{minimize}_{\boldsymbol{\theta} \in \mathbb{R}^d}  f(\boldsymbol{\theta}) \triangleq \frac{1}{n}\sum_{i=1}^n f_i(\boldsymbol{\theta}),
$$

where $f_i : \mathbb{R}^d \to \mathbb{R}$ is the $i$-th sample loss and $\arg\min_{\boldsymbol{\theta}} f \ne \emptyset$. In population risk minimization [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.1.1, pp. 275–276), one replaces `(eq:sgd:erm)` with the expectation $f(\boldsymbol{\theta}) = \mathbb{E}_{\xi \sim \mathcal{D}}[f_\xi(\boldsymbol{\theta})]$; see [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Remark 5.11, pp. 30–31), which recovers `(eq:sgd:erm)` under the empirical measure $\mathcal{D} = \hat{p}_{\mathcal{D}}$.

For composite objectives [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Chs. 10–11, pp. 52, 56), we also include a proper, lower semi-continuous (l.s.c.), convex, possibly nonsmooth regularizer $g : \mathbb{R}^d \to \mathbb{R} \cup \{+\infty\}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.6, p. 308):

$$
\mathrm{minimize}_{\boldsymbol{\theta} \in \mathbb{R}^d}  F(\boldsymbol{\theta}) \triangleq f(\boldsymbol{\theta}) + g(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n f_i(\boldsymbol{\theta}) + g(\boldsymbol{\theta}).
$$

At iteration $t$, stochastic first-order methods query a randomized gradient oracle $\boldsymbol{g}_t$ constructed by sampling an index $i_t \sim \mathrm{Unif}\{1,\dots,n\}$ or a minibatch $B_t \subset \{1,\dots,n\}$ uniformly among all $\binom{n}{b}$ subsets of cardinality $\lvert B_t \rvert = b$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ch. 6, pp. 31–32):

$$
\nabla f_{B_t}(\boldsymbol{\theta}_t) \triangleq \frac{1}{b}\sum_{i \in B_t} \nabla f_i(\boldsymbol{\theta}_t), \qquad
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \nabla f_{B_t}(\boldsymbol{\theta}_t).
$$

Conditioned on the current iterate $\boldsymbol{\theta}_t$, uniform sampling guarantees an *unbiased* gradient estimator $\mathbb{E}[\nabla f_{B_t}(\boldsymbol{\theta}_t) \mid \boldsymbol{\theta}_t] = \nabla f(\boldsymbol{\theta}_t)$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Remark 5.2, p. 24; Remark 6.2, p. 32), whose second moment decomposes cleanly into the squared full gradient plus the oracle variance [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Def. 4.5 & Lemma 4.6, p. 19):

$$
\begin{aligned}
\mathrm{Var}\bigl[\nabla f_{B_t}(\boldsymbol{\theta})\bigr]
&\triangleq \mathbb{E}\Bigl[\lVert \nabla f_{B_t}(\boldsymbol{\theta}) - \nabla f(\boldsymbol{\theta}) \rVert^2\Bigr]
= \mathbb{E}\Bigl[\lVert \nabla f_{B_t}(\boldsymbol{\theta}) \rVert^2\Bigr] - \lVert \nabla f(\boldsymbol{\theta}) \rVert^2  \\
&\le \mathbb{E}\Bigl[\lVert \nabla f_{B_t}(\boldsymbol{\theta}) - \boldsymbol{y} \rVert^2\Bigr] \qquad \forall  \boldsymbol{y} \in \mathbb{R}^d.
\end{aligned}
$$

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> The four synthesized texts diverge substantially in symbol choices and analytical emphasis: Garrigos & Gower develop tight non-asymptotic operator bounds, Murphy emphasizes practical schedules and composite regularization, Goodfellow et al. focus on empirical versus generalization risk, and Prince frames SGD geometrically via fluctuating batch landscapes. Table `tab:sgd:notation` reconciles all symbols into the unified notation used throughout this monograph.
>
>
>
>
>
>
>
>
>
> | **Concept** | **Garrigos & Gower** | **Murphy (PML1)** | **Goodfellow et al.** | **Prince (UDL)** | **Unified Symbol** |
> | — | — | — | — | — | — |
> | Parameter vector | $\boldsymbol{x} \in \mathbb{R}^d$ | $\boldsymbol{\theta}, \boldsymbol{w} \in \mathbb{R}^D$ | $\boldsymbol{\theta} \in \mathbb{R}^n$ | $\boldsymbol{\phi} \in \mathbb{R}^P$ | $\boldsymbol{\theta} \in \mathbb{R}^d$ |
> | Finite-sum loss | $f(\boldsymbol{x}) = \frac{1}{n}\sum f_i(\boldsymbol{x})$ | $L(\boldsymbol{\theta}) = \frac{1}{N}\sum L_n(\boldsymbol{\theta})$ | $J(\boldsymbol{\theta}) = \frac{1}{m}\sum L(\boldsymbol{x}^{(i)}, y^{(i)}, \boldsymbol{\theta})$ | $L[\boldsymbol{\phi}] = \sum_i \ell_i[\boldsymbol{\phi}]$ | $f(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n f_i(\boldsymbol{\theta})$ |
> | Step size (LR) | $\gamma_t > 0$ | $\eta_t > 0$ | $\epsilon_k > 0$ | $\alpha > 0$ | $\eta_t > 0$ |
> | Minibatch size | $b \in \{1,\dots,n\}$ | $\lvert B_t \rvert \ll N$ | $m' \ll m$ | $\lvert B \rvert$ | $b = \lvert B_t \rvert \in \{1,\dots,n\}$ |
> | Composite split | $F = f + g$ ($g$ l.s.c.) | $L = L_s + L_r$ | $J + \lambda\Omega(\boldsymbol{\theta})$ | $L[\boldsymbol{\phi}] + \lambda g[\boldsymbol{\phi}]$ | $F(\boldsymbol{\theta}) = f(\boldsymbol{\theta}) + g(\boldsymbol{\theta})$ |
> | Gradient noise | $\sigma_f^*$, $\sigma_b^*$, $\Delta_f^*$ | Implicit in $\eta_t$ | Std. err. $\sigma/\sqrt{m'}$ | Var. $\mathrm{Var}[\nabla L_b]$ | $\sigma_f^*, \sigma_b^*$ (grad.), $\Delta_f^*$ (func.) |

## Core Mathematical Derivations & Theorems

### Expected Smoothness, Interpolation & Variance Transfer

Classical stochastic approximation often assumes uniformly bounded second moments $\mathbb{E}[\lVert \nabla f_i(\boldsymbol{\theta}) \rVert^2] \le G^2$ across all $\boldsymbol{\theta} \in \mathbb{R}^d$. For unconstrained $\mu$-strongly convex functions, however, $\lVert \nabla f(\boldsymbol{\theta}) \rVert \ge \mu\lVert \boldsymbol{\theta} - \boldsymbol{\theta}^* \rVert$ grows unboundedly as $\lVert \boldsymbol{\theta} \rVert \to \infty$, ruling out any finite global $G$.

Instead, expected smoothness [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ch. 4, p. 20) avoids this artificial hypothesis by lifting componentwise smoothness and convexity directly into expectation.

> [!important] Expected Smoothness & Interpolation
> Following [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ch. 4, pp. 19–23), assume each component $f_i : \mathbb{R}^d \to \mathbb{R}$ is convex and $L_i$-smooth ($\lVert \nabla f_i(\boldsymbol{x}) - \nabla f_i(\boldsymbol{y}) \rVert \le L_i\lVert \boldsymbol{x} - \boldsymbol{y} \rVert$), with $L_{\max} \triangleq \max_{1\le i\le n} L_i$ and $L_{\mathrm{avg}} \triangleq \frac{1}{n}\sum_{i=1}^n L_i$. By the triangle inequality, $f$ is $L_f$-smooth with $L_f \le L_{\mathrm{avg}} \le L_{\max}$.
> - **Expected Smoothness (Cocoercivity in Expectation).** By convex $L_i$-smoothness, each $f_i$ satisfies $\frac{1}{2L_{\max}}\lVert \nabla f_i(\boldsymbol{y}) - \nabla f_i(\boldsymbol{x}) \rVert^2 \le \frac{1}{2L_i}\lVert \nabla f_i(\boldsymbol{y}) - \nabla f_i(\boldsymbol{x}) \rVert^2 \le f_i(\boldsymbol{y}) - f_i(\boldsymbol{x}) - \langle \nabla f_i(\boldsymbol{x}), \boldsymbol{y} - \boldsymbol{x} \rangle$. Averaging uniformly over $i \in \{1,\dots,n\}$ yields
>
>
> $$
> \frac{1}{2L_{\max}}\mathbb{E}\Bigl[\lVert \nabla f_i(\boldsymbol{y}) - \nabla f_i(\boldsymbol{x}) \rVert^2\Bigr]
>   \le f(\boldsymbol{y}) - f(\boldsymbol{x}) - \langle \nabla f(\boldsymbol{x}), \boldsymbol{y} - \boldsymbol{x} \rangle \triangleq D_f(\boldsymbol{y}; \boldsymbol{x}).
>
> $$
>
>   Setting $\boldsymbol{x} = \boldsymbol{\theta}^* \in \arg\min f$ (where $\nabla f(\boldsymbol{\theta}^*) = \boldsymbol{0}$) and $\boldsymbol{y} = \boldsymbol{\theta}$ gives
>
>
> $$
> \mathbb{E}\Bigl[\lVert \nabla f_i(\boldsymbol{\theta}) - \nabla f_i(\boldsymbol{\theta}^*) \rVert^2\Bigr] \le 2L_{\max}\bigl(f(\boldsymbol{\theta}) - \inf f\bigr).
>
> $$
> - **Interpolation Constants.** Define the *function noise* $\Delta_f^*$ and *gradient noise at optimum* $\sigma_f^*$ by
>
>
> $$
> \begin{aligned}
> \Delta_f^* &\triangleq \inf f - \frac{1}{n}\sum_{i=1}^n \inf f_i \ge 0,  \\
>   \sigma_f^* &\triangleq \inf_{\boldsymbol{\theta}^* \in \arg\min f} \mathrm{Var}\bigl[\nabla f_i(\boldsymbol{\theta}^*)\bigr]
>   = \inf_{\boldsymbol{\theta}^* \in \arg\min f} \mathbb{E}\Bigl[\lVert \nabla f_i(\boldsymbol{\theta}^*) \rVert^2\Bigr] \ge 0.
> \end{aligned}
>
> $$
>
>   Under convexity and $L_i$-smoothness, $\mathrm{Var}[\nabla f_i(\boldsymbol{\theta}^*)]$ is invariant across all minimizers $\boldsymbol{\theta}^* \in \arg\min f$ by `(eq:sgd:exp-smooth-opt)`. Moreover, *interpolation* ($\exists \boldsymbol{\theta}^*$ with $f_i(\boldsymbol{\theta}^*) = \inf f_i \forall i$, typical of overparameterized models with $d \gg n$) holds if and only if $\Delta_f^* = 0$, which under convexity is equivalent to $\sigma_f^* = 0$. In general, $L_{\max}$-smoothness gives $\sigma_f^* \le 2L_{\max}\Delta_f^*$, and $\mu$-strong convexity of each $f_i$ gives $2\mu\Delta_f^* \le \sigma_f^*$.

Combining expected smoothness with the interpolation constants bounds the stochastic gradient second moment strictly in terms of suboptimality and noise at the solution.

> [!important] Variance Transfer Lemmas
> Assume each component $f_i$ is $L_i$-smooth with $L_{\max} = \max_i L_i$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lemmas 4.19–4.20, p. 24).
> - **Smooth (Possibly Nonconvex) Variance Transfer:** Since $f_i$ is $L_{\max}$-smooth and bounded below, $\lVert \nabla f_i(\boldsymbol{\theta}) \rVert^2 \le 2L_{\max}(f_i(\boldsymbol{\theta}) - \inf f_i) = 2L_{\max}(f_i(\boldsymbol{\theta}) - f_i(\boldsymbol{\theta}^*)) + 2L_{\max}(f_i(\boldsymbol{\theta}^*) - \inf f_i)$. Taking expectations over $i$ yields
>
>
> $$
> \mathbb{E}\Bigl[\lVert \nabla f_i(\boldsymbol{\theta}) \rVert^2\Bigr] \le 2L_{\max}\bigl(f(\boldsymbol{\theta}) - \inf f\bigr) + 2L_{\max}\Delta_f^*.
>
> $$
> - **Smooth Convex Variance Transfer:** If each $f_i$ is also convex, using $\lVert \boldsymbol{a}+\boldsymbol{b} \rVert^2 \le 2\lVert \boldsymbol{a} \rVert^2 + 2\lVert \boldsymbol{b} \rVert^2$ together with `(eq:sgd:exp-smooth-opt)` gives
>
>
> $$
> \begin{aligned}
> \mathbb{E}\Bigl[\lVert \nabla f_i(\boldsymbol{\theta}) \rVert^2\Bigr]
>   &\le 2\mathbb{E}\Bigl[\lVert \nabla f_i(\boldsymbol{\theta}) - \nabla f_i(\boldsymbol{\theta}^*) \rVert^2\Bigr] + 2\mathbb{E}\Bigl[\lVert \nabla f_i(\boldsymbol{\theta}^*) \rVert^2\Bigr]  \\
>   &\le 4L_{\max}\bigl(f(\boldsymbol{\theta}) - \inf f\bigr) + 2\sigma_f^*.
> \end{aligned}
>
> $$

> [!tip] Why Expected Smoothness Replaces Bounded Gradients
> Variance transfer `(eq:sgd:vt-grad)` decomposes stochastic gradient energy into two geometrically distinct pieces: a *state-dependent curvature term* $4L_{\max}(f(\boldsymbol{\theta}) - \inf f)$ that vanishes automatically as $\boldsymbol{\theta} \to \boldsymbol{\theta}^*$, and an *irreducible conflict term* $2\sigma_f^*$ measuring disagreement among per-sample gradients at $\boldsymbol{\theta}^*$. Far from the optimum, SGD behaves almost like deterministic GD; only within an $O(\eta\sigma_f^*)$ neighborhood of $\boldsymbol{\theta}^*$ does gradient conflict stall progress. When a deep neural network or overparameterized linear system ($d > n$) interpolates the training set ($\sigma_f^* = 0$), the noise ball collapses completely to zero [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ex. 4.11–4.12, p. 21).

### Robbins–Monro Conditions, Noise Ball & Convergence Rates

Because $\sigma_f^* > 0$ in general non-interpolated problems, taking a constant step size $\eta$ causes SGD iterates to oscillate around $\boldsymbol{\theta}^*$ inside a stationary noise ball. As established in stochastic approximation [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.3, p. 295) and optimization theory [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.3.1, pp. 294–295), eliminating this floor requires decaying step sizes according to the **Robbins–Monro conditions**:

$$
\sum_{t=0}^{\infty} \eta_t = \infty \qquad \text{and} \qquad \sum_{t=0}^{\infty} \eta_t^2 < \infty \quad (\text{implying } \eta_t \to 0).
$$

Intuitively, $\sum \eta_t = \infty$ guarantees sufficient cumulative step length to reach $\boldsymbol{\theta}^*$ from any finite initialization $\boldsymbol{\theta}_0$, while $\sum \eta_t^2 < \infty$ ensures finite accumulated variance so fluctuations decay to zero.

> [!important] Convex & Smooth SGD: Constant vs. Decaying Step Sizes
> Suppose each $f_i$ is convex and $L_i$-smooth with $L_{\max} = \max_i L_i$, and let SGD run with step sizes $0 < \eta_t < \frac{1}{2L_{\max}}$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Thms. 5.3 & 5.5, pp. 25–27). Expanding the squared distance to $\boldsymbol{\theta}^* \in \arg\min f$ and taking conditional expectation $\mathbb{E}_t[\cdot] \triangleq \mathbb{E}[\cdot \mid \boldsymbol{\theta}_t]$ yields
>
> $$
> \begin{aligned}
> \mathbb{E}_t\bigl[\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert^2\bigr]
> &= \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2 - 2\eta_t\langle \nabla f(\boldsymbol{\theta}_t), \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rangle + \eta_t^2 \mathbb{E}_t\bigl[\lVert \nabla f_{i_t}(\boldsymbol{\theta}_t) \rVert^2\bigr]  \\
> &\le \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2 - 2\eta_t\bigl(f(\boldsymbol{\theta}_t) - f(\boldsymbol{\theta}^*)\bigr) + \eta_t^2\Bigl(4L_{\max}\bigl(f(\boldsymbol{\theta}_t) - f(\boldsymbol{\theta}^*)\bigr) + 2\sigma_f^*\Bigr)  \\
> &= \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2 - 2\eta_t(1 - 2\eta_t L_{\max})\bigl(f(\boldsymbol{\theta}_t) - f(\boldsymbol{\theta}^*)\bigr) + 2\eta_t^2 \sigma_f^*,
> \end{aligned}
>
> $$
>
> where the inequality uses convexity of $f$ and variance transfer `(eq:sgd:vt-grad)`. Taking full expectations, summing over $t = 0,\dots,T-1$ with telescopic cancellation, dropping $\mathbb{E}\lVert \boldsymbol{\theta}_T - \boldsymbol{\theta}^* \rVert^2 \ge 0$, and applying Jensen's inequality to the weighted average $\bar{\boldsymbol{\theta}}_T \triangleq \sum_{t=0}^{T-1} p_{T,t}\boldsymbol{\theta}_t$ with $p_{T,t} = \frac{\eta_t(1 - 2\eta_t L_{\max})}{\sum_{k=0}^{T-1}\eta_k(1 - 2\eta_k L_{\max})}$ yields
>
> $$
> \mathbb{E}\bigl[f(\bar{\boldsymbol{\theta}}_T) - f(\boldsymbol{\theta}^*)\bigr]
> \le \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2}{2\sum_{t=0}^{T-1}\eta_t(1 - 2\eta_t L_{\max})} + \frac{\sum_{t=0}^{T-1}\eta_t^2}{\sum_{t=0}^{T-1}\eta_t(1 - 2\eta_t L_{\max})} \sigma_f^*.
>
> $$
>
> Specializing `(eq:sgd:convex-master)` yields two canonical regimes:
> - **Constant step size** ($\eta_t \equiv \eta < \frac{1}{2L_{\max}}$, uniform average $\bar{\boldsymbol{\theta}}_T = \frac{1}{T}\sum_{t=0}^{T-1}\boldsymbol{\theta}_t$):
>
>
> $$
> \mathbb{E}\bigl[f(\bar{\boldsymbol{\theta}}_T) - f(\boldsymbol{\theta}^*)\bigr]
>   \le \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2}{2\eta(1 - 2\eta L_{\max})T} + \frac{\eta}{1 - 2\eta L_{\max}}\sigma_f^*.
>
> $$
>
>   If interpolation holds ($\sigma_f^* = 0$), constant-step SGD matches deterministic GD's $\mathcal{O}(1/T)$ rate. Otherwise, a fixed horizon choice $\eta = \frac{1}{2L_{\max}\sqrt{T}}$ (for $T \ge 4$) recovers $\mathcal{O}(1/\sqrt{T})$ rate and $\mathcal{O}(1/\varepsilon^2)$ complexity [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Cor. 5.6, p. 27).
> - **Anytime decaying schedule** ($\eta_t = \frac{\eta_0}{\sqrt{t+1}}$ with $\eta_0 \le \frac{1}{2L_{\max}}$): using integral bounds $\sum_{t=0}^{T-1} (t+1)^{-1} \le \log T$ and $\sum_{t=0}^{T-1}(t+1)^{-1/2} \ge 2(\sqrt{T} - \sqrt{2})$ (so $\sum_{t=0}^{T-1}\eta_t(1-2\eta_t L_{\max}) \ge \eta_0\sqrt{T}$ for $T \ge 49$) yields
>
>
> $$
> \mathbb{E}\bigl[f(\bar{\boldsymbol{\theta}}_T) - f(\boldsymbol{\theta}^*)\bigr]
>   \le \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2}{2\eta_0\sqrt{T}} + \frac{\eta_0 \log T}{\sqrt{T}}\sigma_f^*
>   = \mathcal{O}\!\left(\frac{\log T}{\sqrt{T}}\right).
>
> $$

Note that the optimal polynomial exponent in $\eta_t = \eta_0 (t+1)^{-\alpha}$ for minimizing the variance ratio $\frac{\sum \eta_t^2}{\sum \eta_t}$ over convex objectives is $\alpha = 1/2$, which balances $\mathcal{O}(T^{-(1-\alpha)})$ bias decay against $\mathcal{O}(T^{-\alpha})$ variance decay [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Remark 5.4, p. 26) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. (8.64), p. 295). When $f$ possesses quadratic curvature—either $\mu$-strong convexity or the nonconvex $\mu$-Polyak–Łojasiewicz ($\mu$-PL) inequality $\frac{1}{2}\lVert \nabla f(\boldsymbol{\theta}) \rVert^2 \ge \mu(f(\boldsymbol{\theta}) - \inf f)$—constant-step SGD converges *linearly* up to an explicit $\mathcal{O}(\eta \sigma_f^*/\mu)$ noise ball.

> [!important] Strongly Convex & PL Noise-Ball Bounds
> Following [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Thms. 5.7 & 5.9, pp. 28–30), let each component $f_i$ be $L_i$-smooth with $L_{\max} = \max_i L_i$.
> - **$\mu$-Strongly Convex Case:** Suppose each $f_i$ is convex and $f$ is $\mu$-strongly convex, so $\langle \nabla f(\boldsymbol{\theta}), \boldsymbol{\theta} - \boldsymbol{\theta}^* \rangle \ge f(\boldsymbol{\theta}) - f(\boldsymbol{\theta}^*) + \frac{\mu}{2}\lVert \boldsymbol{\theta} - \boldsymbol{\theta}^* \rVert^2$. Repeating `(eq:sgd:convex-step)` with constant step size $0 < \eta \le \frac{1}{2L_{\max}}$ gives
>
>
> $$
> \begin{aligned}
> \mathbb{E}\bigl[\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert^2\bigr]
>   &\le (1 - \eta\mu)\mathbb{E}\bigl[\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2\bigr] + 2\eta(2\eta L_{\max} - 1)\mathbb{E}\bigl[f(\boldsymbol{\theta}_t) - f(\boldsymbol{\theta}^*)\bigr] + 2\eta^2\sigma_f^*  \\
>   &\le (1 - \eta\mu)\mathbb{E}\bigl[\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2\bigr] + 2\eta^2\sigma_f^*.
> \end{aligned}
>
> $$
>
>   Unrolling `(eq:sgd:sc-recurrence)` and summing the geometric series $\sum_{k=0}^{t-1}(1-\eta\mu)^k \le \frac{1}{\eta\mu}$ yields
>
>
> $$
> \mathbb{E}\bigl[\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2\bigr]
>   \le (1 - \eta\mu)^t \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2 + \frac{2\eta \sigma_f^*}{\mu}.
>
> $$
>
>   Setting $\eta = \min\bigl\{\frac{\varepsilon\mu}{4\sigma_f^*},  \frac{1}{2L_{\max}}\bigr\}$ guarantees $\mathbb{E}\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2 \le \varepsilon$ after $t \ge \max\bigl\{\frac{4\sigma_f^*}{\varepsilon\mu^2},  \frac{2L_{\max}}{\mu}\bigr\}\log\frac{2\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2}{\varepsilon} = \tilde{\mathcal{O}}(1/\varepsilon)$ iterations [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Cor. 5.8, p. 29). Under Robbins–Monro decay $\eta_t = \frac{2}{\mu(t + t_0)}$ with $t_0 \ge \frac{4L_{\max}}{\mu}$, `(eq:sgd:sc-recurrence)` yields $\mathbb{E}\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2 = \mathcal{O}\bigl(\frac{\sigma_f^*}{\mu^2 t}\bigr)$ without tuning $\eta$ to $\varepsilon$.
> - **Nonconvex $\mu$-PL Case:** Even without convexity, if $f$ is $\mu$-PL and $0 < \eta \le \frac{\mu}{L_f L_{\max}}$, applying $L_f$-smoothness $f(\boldsymbol{\theta}_{t+1}) \le f(\boldsymbol{\theta}_t) - \eta\langle \nabla f(\boldsymbol{\theta}_t), \nabla f_{i_t}(\boldsymbol{\theta}_t) \rangle + \frac{L_f \eta^2}{2}\lVert \nabla f_{i_t}(\boldsymbol{\theta}_t) \rVert^2$ together with `(eq:sgd:vt-func)` and the $\mu$-PL inequality gives
>
>
> $$
> \mathbb{E}\bigl[f(\boldsymbol{\theta}_t) - \inf f\bigr]
>   \le (1 - \eta\mu)^t \bigl(f(\boldsymbol{\theta}_0) - \inf f\bigr) + \frac{\eta L_f L_{\max}}{\mu} \Delta_f^*.
>
> $$

### Minibatch Expected Smoothness & Linear-to-Saturation Scaling

Evaluating gradients on a minibatch $B_t \subset \{1,\dots,n\}$ of size $b$ without replacement within each step simultaneously utilizes SIMD/GPU parallelism and shrinks both the effective smoothness constant $L_b$ and the gradient noise $\sigma_b^*$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ch. 6, pp. 31–35) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.1.3, pp. 278–279).

> [!important] Exact Minibatch Expected Smoothness & Noise Scaling
> Assume each $f_i$ is convex and $L_i$-smooth, and let $B \subset \{1,\dots,n\}$ be sampled uniformly over subsets of size $b \in \{1,\dots,n\}$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lemma 6.5 & Thm. 6.11, pp. 32–35). Define $\sigma_b^* \triangleq \inf_{\boldsymbol{\theta}^*\in\arg\min f} \mathrm{Var}_B[\nabla f_B(\boldsymbol{\theta}^*)]$. Then $f$ is $L_b$-smooth in expectation:
>
> $$
> \mathbb{E}_B\Bigl[\lVert \nabla f_B(\boldsymbol{y}) - \nabla f_B(\boldsymbol{x}) \rVert^2\Bigr] \le 2L_b  D_f(\boldsymbol{y}; \boldsymbol{x}),
> \qquad
> L_b = \frac{n(b-1)}{b(n-1)} L_f + \frac{n-b}{b(n-1)} L_{\max},
>
> $$
>
> and the minibatch gradient noise at $\boldsymbol{\theta}^*$ obeys the finite-population correction law
>
> $$
> \sigma_b^* = \frac{n-b}{b(n-1)}\sigma_f^*.
>
> $$
>
> Consequently, all single-sample SGD guarantees hold verbatim upon substituting $(L_{\max}, \sigma_f^*) \mapsto (L_b, \sigma_b^*)$: for $\mu$-strongly convex $f$ and $0 < \eta \le \frac{1}{2L_b}$,
>
> $$
> \mathbb{E}\bigl[\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2\bigr]
> \le (1 - \eta\mu)^t\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2 + \frac{2\eta \sigma_b^*}{\mu},
> \qquad
> T_\varepsilon(b) = \max\!\left\{\frac{2L_b}{\mu},  \frac{4\sigma_b^*}{\varepsilon\mu^2}\right\}\log\frac{2\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2}{\varepsilon}.
>
> $$

Equation `(eq:sgd:minibatch-Lb)`–`(eq:sgd:minibatch-sc)` unifies the entire spectrum between single-sample SGD ($b=1 \implies L_1 = L_{\max},  \sigma_1^* = \sigma_f^*$) and deterministic full-batch GD ($b=n \implies L_n = L_f,  \sigma_n^* = 0$), and rigorously explains the empirical **linear-to-saturation batch scaling law**:
- **Linear scaling regime ($b \ll b_{\mathrm{crit}}$):** When the noise term dominates ($\frac{4\sigma_b^*}{\varepsilon\mu^2} > \frac{2L_b}{\mu}$), we may scale the step size $\eta(b) = \frac{\varepsilon\mu}{4\sigma_b^*} \propto b$ linearly with $b$ without inflating the noise ball $\frac{2\eta(b)\sigma_b^*}{\mu} = \frac{\varepsilon}{2}$; iteration count drops linearly as $T_\varepsilon(b) \propto 1/b$ while total gradient evaluations $b\cdot T_\varepsilon(b)$ remain nearly invariant.
- **Curvature saturation regime ($b \ge b_{\mathrm{crit}}$):** Solving $\frac{2L_b}{\mu} = \frac{4\sigma_b^*}{\varepsilon\mu^2}$ defines the *critical batch size*
  
$$
b_{\mathrm{crit}} = \frac{n\bigl(2\sigma_f^*/(\varepsilon\mu) - L_{\max} + L_f\bigr)}{(n-1)L_f + 2\sigma_f^*/(\varepsilon\mu) - L_{\max} + L_f}
  \approx \frac{2\sigma_f^*}{\varepsilon \mu L_f} \quad (\text{for } n \gg b_{\mathrm{crit}}).
$$

  Beyond $b_{\mathrm{crit}}$, the maximum stable step size saturates at $\eta \le \frac{1}{2L_b} \to \frac{1}{2L_f}$ and total sample complexity $b \cdot T_\varepsilon(b) \approx \frac{2 b L_f}{\mu}\log(2\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2/\varepsilon)$ grows linearly in $b$.

### Variance Reduction: SVRG & SAGA Control Variates

Standard SGD must shrink $\eta_t \to 0$ because $\mathrm{Var}[\nabla f_{i_t}(\boldsymbol{\theta}_t)] \to \sigma_f^* > 0$ even at $\boldsymbol{\theta}^*$. Variance-reduced finite-sum methods eliminate this floor by constructing an unbiased *control-variate* estimator whose variance vanishes as the iterates approach $\boldsymbol{\theta}^*$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.5, pp. 297–298).

> [!important] Control-Variate Variance Vanishing in SVRG & SAGA
> Given a snapshot parameter $\tilde{\boldsymbol{\theta}}$ with exact precomputed full gradient $\nabla f(\tilde{\boldsymbol{\theta}}) = \frac{1}{n}\sum_{i=1}^n \nabla f_i(\tilde{\boldsymbol{\theta}})$, **Stochastic Variance-Reduced Gradient (SVRG)** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eqs. (8.66)–(8.67), pp. 297–298) updates at step $t$ via $i_t \sim \mathrm{Unif}\{1,\dots,n\}$:
>
> $$
> \boldsymbol{g}_t^{\mathrm{SVRG}} \triangleq \nabla f_{i_t}(\boldsymbol{\theta}_t) - \nabla f_{i_t}(\tilde{\boldsymbol{\theta}}) + \nabla f(\tilde{\boldsymbol{\theta}}), \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \boldsymbol{g}_t^{\mathrm{SVRG}}.
>
> $$
>
> Because $\mathbb{E}[\nabla f_{i_t}(\tilde{\boldsymbol{\theta}})] = \nabla f(\tilde{\boldsymbol{\theta}})$, the estimator is strictly unbiased: $\mathbb{E}[\boldsymbol{g}_t^{\mathrm{SVRG}} \mid \boldsymbol{\theta}_t] = \nabla f(\boldsymbol{\theta}_t)$. Furthermore, applying variance bound `(eq:sgd:variance-decomp)` (centering around $\boldsymbol{y} = \nabla f(\tilde{\boldsymbol{\theta}}) - \nabla f(\boldsymbol{\theta}_t)$) and expected smoothness `(eq:sgd:exp-smooth-opt)` gives
>
> $$
> \begin{aligned}
> \mathrm{Var}\bigl[\boldsymbol{g}_t^{\mathrm{SVRG}} \mid \boldsymbol{\theta}_t\bigr]
> &\le \mathbb{E}\Bigl[\lVert \nabla f_{i_t}(\boldsymbol{\theta}_t) - \nabla f_{i_t}(\tilde{\boldsymbol{\theta}}) \rVert^2\Bigr]  \\
> &\le 2\mathbb{E}\Bigl[\lVert \nabla f_{i_t}(\boldsymbol{\theta}_t) - \nabla f_{i_t}(\boldsymbol{\theta}^*) \rVert^2\Bigr] + 2\mathbb{E}\Bigl[\lVert \nabla f_{i_t}(\tilde{\boldsymbol{\theta}}) - \nabla f_{i_t}(\boldsymbol{\theta}^*) \rVert^2\Bigr]  \\
> &\le 4L_{\max}\bigl(f(\boldsymbol{\theta}_t) - \inf f\bigr) + 4L_{\max}\bigl(f(\tilde{\boldsymbol{\theta}}) - \inf f\bigr).
> \end{aligned}
>
> $$
>
> Because the right-hand side of `(eq:sgd:svrg-var)` has zero $\sigma_f^*$ floor and shrinks to $0$ as $\boldsymbol{\theta}_t, \tilde{\boldsymbol{\theta}} \to \boldsymbol{\theta}^*$, SVRG converges *linearly* with a constant step size $\eta = \Theta(1/L_{\max})$, requiring only $\mathcal{O}\bigl((n + L_{\max}/\mu)\log(1/\varepsilon)\bigr)$ gradient evaluations for $\mu$-strongly convex finite sums while storing only $\mathcal{O}(d)$ parameters. By contrast, **SAGA** replaces the periodic full-batch snapshot $\tilde{\boldsymbol{\theta}}$ with a table of $n$ cached per-example gradients $\{\boldsymbol{g}_i^{\mathrm{loc}}\}_{i=1}^n$ and running average $\bar{\boldsymbol{g}} = \frac{1}{n}\sum_{i=1}^n \boldsymbol{g}_i^{\mathrm{loc}}$, updating $\boldsymbol{g}_t^{\mathrm{SAGA}} = \nabla f_{i_t}(\boldsymbol{\theta}_t) - \boldsymbol{g}_{i_t}^{\mathrm{loc}} + \bar{\boldsymbol{g}}$ and overwriting $\boldsymbol{g}_{i_t}^{\mathrm{loc}} \leftarrow \nabla f_{i_t}(\boldsymbol{\theta}_t)$ at $\mathcal{O}(nd)$ memory cost [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.5.2, p. 298).

### Nonsmooth Convexity, Subgradients & Stochastic Subgradient Descent

When losses contain kinks (e.g., hinge loss, ReLU networks, or $\ell_1$ penalties $\lVert \boldsymbol{\theta} \rVert_1$) or take $+\infty$ outside a constraint set $\mathcal{C}$, classical gradients fail to exist [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ch. 8, p. 39) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.1.4.1, p. 282). For proper l.s.c. convex functions $f : \mathbb{R}^d \to \mathbb{R} \cup \{+\infty\}$, the **subdifferential** $\partial f(\boldsymbol{\theta})$ generalizes the derivative as the set of supporting affine lower bounds:

$$
\partial f(\boldsymbol{\theta}) \triangleq \{\boldsymbol{\eta} \in \mathbb{R}^d  \middle|  f(\boldsymbol{y}) \ge f(\boldsymbol{\theta}) + \langle \boldsymbol{\eta}, \boldsymbol{y} - \boldsymbol{\theta} \rangle   \forall  \boldsymbol{y} \in \mathbb{R}^d\}.
$$

Whenever $f$ is convex and continuous on $\mathbb{R}^d$, $\partial f(\boldsymbol{\theta}) \ne \emptyset$; at differentiable points $\partial f(\boldsymbol{\theta}) = \{\nabla f(\boldsymbol{\theta})\}$; and Fermat's optimality condition becomes $\boldsymbol{0} \in \partial f(\boldsymbol{\theta}^*)$. Moreover, when $f$ is differentiable convex and $g$ is proper l.s.c. convex, the sum rule $\partial(f + g)(\boldsymbol{\theta}) = \{\nabla f(\boldsymbol{\theta})\} + \partial g(\boldsymbol{\theta})$ holds [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lemmas 8.5–8.9, pp. 41–42). For the scalar absolute value $f(\theta) = \lvert \theta \rvert$, $\partial\lvert \theta \rvert = \{-1\}$ for $\theta < 0$, $[-1, 1]$ for $\theta = 0$, and $\{+1\}$ for $\theta > 0$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. (8.14), p. 282).

> [!important] Stochastic Subgradient Descent (SSD) Complexity
> Following [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ch. 9, pp. 46–52), consider $f(\boldsymbol{\theta}) = \mathbb{E}_{\xi\sim\mathcal{D}}[f_\xi(\boldsymbol{\theta})]$ where each $f_\xi$ is convex and possesses a measurable subgradient selection $\boldsymbol{g}(\boldsymbol{\theta}, \xi) \in \partial f_\xi(\boldsymbol{\theta})$ with bounded second moment $\mathbb{E}_\xi[\lVert \boldsymbol{g}(\boldsymbol{\theta}, \xi) \rVert^2] \le G^2$. Then $\mathbb{E}_\xi[\boldsymbol{g}(\boldsymbol{\theta}, \xi)] \in \partial f(\boldsymbol{\theta})$ (Lemma 9.4), and iterating $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \boldsymbol{g}(\boldsymbol{\theta}_t, \xi_t)$ gives
>
> $$
> \mathbb{E}_t\bigl[\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert^2\bigr]
> \le \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2 - 2\eta_t\bigl(f(\boldsymbol{\theta}_t) - \inf f\bigr) + \eta_t^2 G^2.
>
> $$
>
> Summing from $t=0$ to $T-1$ and applying Jensen's inequality to the step-weighted average $\bar{\boldsymbol{\theta}}_T = \frac{\sum_{t=0}^{T-1}\eta_t\boldsymbol{\theta}_t}{\sum_{t=0}^{T-1}\eta_t}$ yields
>
> $$
> \mathbb{E}\bigl[f(\bar{\boldsymbol{\theta}}_T) - \inf f\bigr]
> \le \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2 + G^2\sum_{t=0}^{T-1}\eta_t^2}{2\sum_{t=0}^{T-1}\eta_t}.
>
> $$
> - **Fixed-horizon optimal constant step:** Setting $\eta_t \equiv \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert}{G\sqrt{T}}$ minimizes `(eq:sgd:ssd-master)`, giving $\mathbb{E}[f(\bar{\boldsymbol{\theta}}_T) - \inf f] \le \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert G}{\sqrt{T}}$, i.e., $T \ge \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2 G^2}{\varepsilon^2} = \mathcal{O}(1/\varepsilon^2)$ complexity [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Cor. 9.7, p. 49).
> - **Anytime decaying schedule:** Taking $\eta_t = \frac{\eta}{\sqrt{t+1}}$ yields $\mathcal{O}\bigl(\frac{\log T}{\sqrt{T}}\bigr)$ convergence, while adding projection onto a ball $B(\boldsymbol{0}, R)$ containing $\boldsymbol{\theta}^*$ (Projected SSD) eliminates the $\log T$ factor to achieve $\frac{1}{\sqrt{T}}\bigl(\frac{3R^2}{\eta} + \eta G^2\bigr)$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Thms. 9.6 & 9.11, pp. 48, 51).
> - **Strongly convex nonsmooth case:** If $f$ is $\mu$-strongly convex and $\eta \in (0, 1/\mu)$, constant-step SSD satisfies $\mathbb{E}\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2 \le (1-\eta\mu)^t\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2 + \frac{\eta G^2}{\mu}$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Thm. 9.8, p. 49).

### Composite Operator Splitting, Proximal Gradient & ISTA

Subgradient descent slows to $\mathcal{O}(1/\sqrt{T})$ even when $F = f + g$ combines a smooth $L$-Lipschitz-gradient loss $f$ with a simple nonsmooth penalty $g$ (such as $\lambda\lVert \boldsymbol{\theta} \rVert_1$ or an indicator $\delta_{\mathcal{C}}$). **Proximal gradient descent (PGD)** preserves smooth convergence rates by decoupling $f$ (forward explicit gradient step) from $g$ (backward implicit proximal step) [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ch. 8.4 & 10, pp. 42–44, 52–56) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.6, pp. 308–311).

> [!important] Proximal Operator, Soft-Thresholding (ISTA) & Composite Convergence
> For proper l.s.c. convex $g : \mathbb{R}^d \to \mathbb{R} \cup \{+\infty\}$ and $\eta > 0$, the **proximal operator** is uniquely defined (by $1/\eta$-strong convexity of the objective) as
>
> $$
> \operatorname{prox}_{\eta g}(\boldsymbol{v}) \triangleq \arg\min_{\boldsymbol{z} \in \mathbb{R}^d} \left\{ g(\boldsymbol{z}) + \frac{1}{2\eta}\lVert \boldsymbol{z} - \boldsymbol{v} \rVert^2 \right\}.
>
> $$
>
> **Fundamental Algebraic Properties** [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lemmas 8.15–8.17, pp. 43–44):
> - *Subdifferential characterization:* $\boldsymbol{p} = \operatorname{prox}_{\eta g}(\boldsymbol{v}) \iff \frac{\boldsymbol{v} - \boldsymbol{p}}{\eta} \in \partial g(\boldsymbol{p})$.
> - *Non-expansiveness ($1$-Lipschitz):* $\lVert \operatorname{prox}_{\eta g}(\boldsymbol{x}) - \operatorname{prox}_{\eta g}(\boldsymbol{y}) \rVert \le \lVert \boldsymbol{x} - \boldsymbol{y} \rVert$ for all $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^d$.
> - *Fixed-point optimality:* $\boldsymbol{\theta}^* \in \arg\min(f + g) \iff \boldsymbol{\theta}^* = \operatorname{prox}_{\eta g}(\boldsymbol{\theta}^* - \eta \nabla f(\boldsymbol{\theta}^*))$ for any $\eta > 0$.
>
>
> **Canonical Closed-Form Operators** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Secs. 8.6.1–8.6.3, 11.4.9.3, pp. 308–311, 398):
> - *Indicator $\delta_{\mathcal{C}}$ (Projected GD):* $\operatorname{prox}_{\eta\delta_{\mathcal{C}}}(\boldsymbol{v}) = \Pi_{\mathcal{C}}(\boldsymbol{v}) = \arg\min_{\boldsymbol{z}\in\mathcal{C}}\lVert \boldsymbol{z} - \boldsymbol{v} \rVert^2$; for box constraints $[\boldsymbol{l}, \boldsymbol{u}]$, $[\Pi_{[\boldsymbol{l},\boldsymbol{u}]}(\boldsymbol{v})]_j = \min(\max(v_j, l_j), u_j)$.
> - *$\ell_1$ norm $g(\boldsymbol{\theta}) = \lambda\lVert \boldsymbol{\theta} \rVert_1$ (ISTA Soft-Thresholding):* Setting $0 \in \eta\lambda\partial\lvert z_j \rvert + z_j - v_j$ coordinatewise gives
>
>
> $$
> \bigl[\operatorname{prox}_{\eta\lambda\lVert \cdot \rVert_1}(\boldsymbol{v})\bigr]_j
>   = \operatorname{SoftThreshold}(v_j, \eta\lambda)
>   \triangleq \operatorname{sign}(v_j) \max\bigl(\lvert v_j \rvert - \eta\lambda,  0\bigr),
>
> $$
>
>   yielding the Iterative Soft-Thresholding Algorithm $\boldsymbol{\theta}_{t+1} = \operatorname{SoftThreshold}(\boldsymbol{\theta}_t - \eta\nabla f(\boldsymbol{\theta}_t),  \eta\lambda)$.
>
>
> **Convergence Guarantees for Forward–Backward Splitting** ($\boldsymbol{\theta}_{t+1} = \operatorname{prox}_{\eta_t g}(\boldsymbol{\theta}_t - \eta_t \boldsymbol{g}_t)$):
> - *Deterministic PGD* ($\boldsymbol{g}_t = \nabla f(\boldsymbol{\theta}_t)$, $f$ convex & $L$-smooth, $\eta \in (0, 1/L]$) [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Thms. 10.3 & 10.5, pp. 53–56): Using the decreasing Lyapunov energy $E_t = \frac{1}{2\eta}\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2 + t(F(\boldsymbol{\theta}_t) - \inf F)$ establishes $F(\boldsymbol{\theta}_t) - \inf F \le \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2}{2\eta t} = \mathcal{O}(1/t)$ (avoiding the $\mathcal{O}(1/\sqrt{t})$ subgradient bottleneck). If $f$ is $\mu$-strongly convex, non-expansiveness + Lemma 8.17 gives linear contraction $\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert^2 \le (1 - \eta\mu)\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2$.
> - *Stochastic PGD (SPGD)* ($\boldsymbol{g}_t = \nabla f_{i_t}(\boldsymbol{\theta}_t)$, composite noise $\sigma_F^* \triangleq \inf_{\boldsymbol{\theta}^*\in\arg\min F}\mathrm{Var}[\nabla f_i(\boldsymbol{\theta}^*)]$) [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Def. 8.21, Thms. 11.5 & 11.9, pp. 45, 57–62): Notice that $\sigma_F^* \ne \sigma_f^*$ in general because $\nabla f(\boldsymbol{\theta}^*) \in -\partial g(\boldsymbol{\theta}^*)$ need not vanish at the composite minimizer; rather, $\sigma_F^* \le 4L_{\max}(f(\boldsymbol{\theta}^*) - \inf f) + 2\sigma_f^*$. For convex $f_i$ and nonincreasing $\eta_t$ with $\eta_0 < \frac{1}{4L_{\max}}$,
>
>
> $$
> \mathbb{E}\bigl[F(\bar{\boldsymbol{\theta}}_t) - \inf F\bigr]
>   \le \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2 + 2\eta_0(F(\boldsymbol{\theta}_0) - \inf F)}{2(1 - 4\eta_0 L_{\max})\sum_{s=0}^{t-1}\eta_s} + \frac{2\sigma_F^* \sum_{s=0}^{t-1}\eta_s^2}{(1 - 4\eta_0 L_{\max})\sum_{s=0}^{t-1}\eta_s},
>
> $$
>
>   giving $\mathcal{O}\bigl((\log t)/\sqrt{t}\bigr)$ under $\eta_t = \eta_0/\sqrt{t+1}$; if $f$ is $\mu$-strongly convex and $\eta \in (0, \frac{1}{2L_{\max}}]$, SPGD contracts to a composite noise ball:
>
>
> $$
> \mathbb{E}\bigl[\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2\bigr] \le (1 - \eta\mu)^t \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2 + \frac{2\eta \sigma_F^*}{\mu}.
>
> $$

> [!tip] Why Exact Sparsity Requires Proximal Splitting, Not Subgradients
> Subgradient updates on $F(\boldsymbol{\theta}) = f(\boldsymbol{\theta}) + \lambda\lVert \boldsymbol{\theta} \rVert_1$ evaluate $\nabla f(\boldsymbol{\theta}_t) + \lambda\operatorname{sign}(\boldsymbol{\theta}_t)$, so iterates almost surely overshoot zero rather than hitting $0$ exactly. By contrast, the proximal step $\operatorname{prox}_{\eta g}$ solves a regularized local quadratic model and maps the entire deadzone $[-\eta\lambda, +\eta\lambda]$ identically onto $0$, producing exact structural sparsity at every step while preserving $\mathcal{O}(1/t)$ (convex) and linear (strongly convex) smooth convergence speeds [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.6.2 & 11.4.3, pp. 310, 388).

## Algorithmic Implementation & Comparative Synthesis

Algorithms `alg:sgd:mini-spgd` and `alg:sgd:svrg` synthesize minibatch stochastic proximal gradient descent (covering vanilla SGD for $g \equiv 0$, projected SGD for $g = \delta_{\mathcal{C}}$, and stochastic ISTA for $g = \lambda\lVert \cdot \rVert_1$) alongside an epoch-based SVRG variance-reduction wrapper.

> [!example] Algorithm: Minibatch Stochastic Proximal Gradient Descent (Mini-SPGD / Stochastic ISTA)
> **Input:** Initial parameters $\boldsymbol{\theta}_0 \in \mathbb{R}^d$; component functions $\{f_i\}_{i=1}^n$; convex regularizer $g$; minibatch size $b \in \{1,\dots,n\}$; step-size schedule $\{\eta_t\}_{t=0}^{T-1}$ satisfying $\eta_t \le 1/(2L_b)$ (or $< 1/(4L_{\max})$ for convex SPGD)
> **Output:** Final iterate $\boldsymbol{\theta}_T$ or Polyak–Ruppert / step-weighted average $\bar{\boldsymbol{\theta}}_T$
> Initialize $\bar{\boldsymbol{\theta}}_0 \leftarrow \boldsymbol{\theta}_0$ and $w_0 \leftarrow 0$
> **for** $t = 0, 1, \dots, T-1$:
>
>   Sample minibatch $B_t \subset \{1,\dots,n\}$ of cardinality $b$ (uniform subset or shuffled epoch partition)
>   Compute unbiased minibatch gradient $\boldsymbol{g}_t \leftarrow \frac{1}{b}\sum_{i \in B_t} \nabla f_i(\boldsymbol{\theta}_t)$
>   Forward smooth step: $\boldsymbol{v}_t \leftarrow \boldsymbol{\theta}_t - \eta_t \boldsymbol{g}_t$
>   Backward proximal step: $\boldsymbol{\theta}_{t+1} \leftarrow \operatorname{prox}_{\eta_t g}(\boldsymbol{v}_t)$
>      *// e.g., $\operatorname{SoftThreshold}(\boldsymbol{v}_t, \eta_t\lambda)$ or $\Pi_{\mathcal{C}}(\boldsymbol{v}_t)$*
>   Update running average $\bar{\boldsymbol{\theta}}_{t+1} \leftarrow \frac{w_t \bar{\boldsymbol{\theta}}_t + \eta_t \boldsymbol{\theta}_{t+1}}{w_t + \eta_t}$, $w_{t+1} \leftarrow w_t + \eta_t$
>
> **return** $\boldsymbol{\theta}_T$ (strongly convex / PL) or $\bar{\boldsymbol{\theta}}_T$ (general convex)

> [!example] Algorithm: Stochastic Variance-Reduced Gradient (SVRG) with Optional Proximal Splitting
> **Input:** Initial snapshot $\tilde{\boldsymbol{\theta}}_0 \in \mathbb{R}^d$; outer epochs $S$; inner loop length $m$ (typically $m \approx 2n$); constant step size $\eta \le \frac{1}{10 L_{\max}}$
> **Output:** Final snapshot $\tilde{\boldsymbol{\theta}}_S$
> **for** $s = 0, 1, \dots, S-1$:
>
>   Compute exact full-batch snapshot gradient $\tilde{\boldsymbol{\mu}}_s \leftarrow \frac{1}{n}\sum_{i=1}^n \nabla f_i(\tilde{\boldsymbol{\theta}}_s)$  *// $\mathcal{O}(n)$ oracle calls*
>   Initialize inner state $\boldsymbol{\theta}_0^{(s)} \leftarrow \tilde{\boldsymbol{\theta}}_s$
>   **for** $k = 0, 1, \dots, m-1$:
>
>     Sample index $i_k \sim \mathrm{Unif}\{1,\dots,n\}$
>     Form control-variate estimator $\boldsymbol{v}_k \leftarrow \nabla f_{i_k}(\boldsymbol{\theta}_k^{(s)}) - \nabla f_{i_k}(\tilde{\boldsymbol{\theta}}_s) + \tilde{\boldsymbol{\mu}}_s$
>     Update $\boldsymbol{\theta}_{k+1}^{(s)} \leftarrow \operatorname{prox}_{\eta g}\bigl(\boldsymbol{\theta}_k^{(s)} - \eta \boldsymbol{v}_k\bigr)$  *// Reduces to $\boldsymbol{\theta}_k^{(s)} - \eta\boldsymbol{v}_k$ if $g \equiv 0$*
>
>   Set $\tilde{\boldsymbol{\theta}}_{s+1} \leftarrow \frac{1}{m}\sum_{k=0}^{m-1} \boldsymbol{\theta}_k^{(s)}$ (or uniform random iterate $\boldsymbol{\theta}_k^{(s)}$)
>
> **return** $\tilde{\boldsymbol{\theta}}_S$

Table `tab:sgd:complexity` compiles the exact iteration and gradient-oracle complexities across smooth, nonsmooth, minibatch, variance-reduced, and composite proximal methods.

**Table:** Non-asymptotic iteration complexities to achieve expected error $\le \varepsilon$ ($\mathbb{E}[F(\boldsymbol{\theta}_t) - \inf F] \le \varepsilon$ in convex/PL regimes; $\mathbb{E}\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2 \le \varepsilon$ in strongly convex regimes). Let $D_0 = \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert$ and $\delta_0 = f(\boldsymbol{\theta}_0) - \inf f$. Adapted from [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Table 1b, p. 4) and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.5, pp. 297–298).

| **Method** | **Oracle Cost/Step** | **Convex ($L$-Smooth / $G$-Lip.)** | **$\mu$-Strongly Convex** | **$\mu$-PL ($L$-Smooth)** |
| — | — | — | — | — |
| Full-Batch GD | $n$ | $\mathcal{O}\!\left(\frac{L_f D_0^2}{\varepsilon}\right)$ | $\mathcal{O}\!\left(\frac{L_f}{\mu}\log\frac{1}{\varepsilon}\right)$ | $\mathcal{O}\!\left(\frac{L_f}{\mu}\log\frac{1}{\varepsilon}\right)$ |
| Single SGD ($b=1$) | $1$ | $\mathcal{O}\!\left(\frac{L_{\max}D_0^2}{\varepsilon} + \frac{\sigma_f^* D_0^2}{\varepsilon^2}\right)$ | $\tilde{\mathcal{O}}\!\left(\max\!\left\{\frac{L_{\max}}{\mu},  \frac{\sigma_f^*}{\varepsilon\mu^2}\right\}\right)$ | $\tilde{\mathcal{O}}\!\left(\frac{L_f L_{\max}}{\mu}\max\!\left\{1, \frac{\Delta_f^*}{\varepsilon\mu}\right\}\right)$ |
| Minibatch SGD ($b$) | $b$ | $\mathcal{O}\!\left(\frac{L_b D_0^2}{\varepsilon} + \frac{\sigma_b^* D_0^2}{\varepsilon^2}\right)$ | $\tilde{\mathcal{O}}\!\left(\max\!\left\{\frac{L_b}{\mu},  \frac{\sigma_b^*}{\varepsilon\mu^2}\right\}\right)$ | $\tilde{\mathcal{O}}\!\left(\frac{L_f L_b}{\mu}\max\!\left\{1, \frac{\Delta_b^*}{\varepsilon\mu}\right\}\right)$ |
| Stoch. Subgrad. (SSD) | $1$ | $\mathcal{O}\!\left(\frac{G^2 D_0^2}{\varepsilon^2}\right)$ ($G$-Lipschitz) | $\tilde{\mathcal{O}}\!\left(\frac{G^2}{\varepsilon\mu^2}\right)$ ($G$-Lipschitz) | N/A (requires smoothness) |
| Proximal GD (ISTA) | $n$ | $\mathcal{O}\!\left(\frac{L_f D_0^2}{\varepsilon}\right)$ | $\mathcal{O}\!\left(\frac{L_f}{\mu}\log\frac{1}{\varepsilon}\right)$ | Linear under proximal-PL |
| Stoch. Prox. (SPGD) | $1$ | $\mathcal{O}\!\left(\frac{L_{\max}D_0^2 + \delta_F}{\varepsilon} + \frac{\sigma_F^*}{\varepsilon^2}\right)$ | $\tilde{\mathcal{O}}\!\left(\max\!\left\{\frac{L_{\max}}{\mu},  \frac{\sigma_F^*}{\varepsilon\mu^2}\right\}\right)$ | Linear to $\mathcal{O}(\eta \sigma_F^*/\mu)$ ball |
| SVRG / SAGA | $2$ (+$n$/epoch) / $1$ | $\mathcal{O}\!\left(n + \frac{L_{\max}D_0^2}{\varepsilon}\right)$ | $\mathcal{O}\!\left(\left(n + \frac{L_{\max}}{\mu}\right)\log\frac{1}{\varepsilon}\right)$ | $\mathcal{O}\!\left(\left(n + \frac{L_{\max}}{\mu}\right)\log\frac{1}{\varepsilon}\right)$ |

> [!info] Cross-Book Synthesis: Pure Optimization vs. Generalization-Driven SGD
> Comparing the four texts reveals a fundamental tension between minimizing finite-sum empirical risk $f(\boldsymbol{\theta})$ and minimizing true population risk $F_{\mathrm{pop}}(\boldsymbol{\theta})$:
> - **Cramér–Rao Generalization Ceiling vs. Linear Convergence:** Garrigos & Gower [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ch. 5–6, pp. 24–35) and Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.5, pp. 297–298) emphasize that variance reduction (SVRG/SAGA) recovers linear $\mathcal{O}(\log(1/\varepsilon))$ optimization on fixed empirical sums. Conversely, Goodfellow et al. [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.1.3 & 8.3.1, pp. 280–282, 295–296) note that in the first pass over data, SGD follows an unbiased gradient of the *true generalization risk*, and because the Cramér–Rao statistical estimation limit caps excess test error decay at $\Omega(1/k)$ after $k$ samples, driving empirical error below statistical noise with faster-than-$\mathcal{O}(1/k)$ solvers yields diminishing returns and risks overfitting.
> - **Implicit Gradient-Variance Regularization & Valley Escape:** Prince [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 6.2, pp. 83–86; Sec. 9.2.2, p. 144) demonstrates both visually and via modified-loss backward error analysis that SGD with finite step size $\eta$ and batch size $b$ minimizes an effective surrogate objective $\tilde{L}_{\mathrm{SGD}}(\boldsymbol{\theta}) = L(\boldsymbol{\theta}) + \frac{\eta}{4}\lVert \nabla L(\boldsymbol{\theta}) \rVert^2 + \frac{\eta}{4K}\sum_{k=1}^K \lVert \nabla L_{B_k}(\boldsymbol{\theta}) - \nabla L(\boldsymbol{\theta}) \rVert^2$, actively biasing iterates toward flat basins where per-batch gradients agree—a bias lost when $b \to n$ or when variance is artificially stripped.
> - **Practical Schedules & Averaging:** To reconcile fast early traversal with low asymptotic variance, Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.3–8.4.4, pp. 294–297) recommends linear warmup followed by cosine or polynomial $\eta_t \propto t^{-1/2}$ decay, paired with **Polyak–Ruppert iterate averaging** $\bar{\boldsymbol{\theta}}_t = \frac{1}{t}\sum_{k=1}^t \boldsymbol{\theta}_k$ (or Stochastic Weight Averaging, SWA) to match the asymptotic efficiency of second-order Hessian methods without inverting a $d\times d$ matrix.

## Visual Diagnostics & Geometric Structure

We examine the geometry of stochastic trajectories, composite operator splitting, and convergence trade-offs through three extracted textbook visualizations, a custom TikZ forward–backward splitting architecture, and a quantitative PGFPlots synthesis.

> [!note] Source Figure
> *Figure:* Illustration of single-sample SGD ($b=1$, the LMS / Widrow–Hoff rule) on linear least squares [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 8.16, Sec. 8.4.2, p. 294). (a) Iterates meander stochastically toward the global OLS solution (red cross) rather than following orthogonal contour normals. (b) Because individual steps follow noisy sample gradients $-\nabla f_{i_t}(\boldsymbol{\theta}_t)$ rather than $-\nabla f(\boldsymbol{\theta}_t)$, the full empirical objective decreases *non-monotonically* before settling into the constant-step noise floor. (`probabilisticml_kevinmurphy_introduction_p324_fig1.png`)

Figure `fig:sgd:lms-traj` exhibits the classic non-monotonic behavior of SGD under a fixed step size: individual stochastic updates can temporarily increase the full empirical risk $f(\boldsymbol{\theta}_t)$ whenever $-\nabla f_{i_t}(\boldsymbol{\theta}_t)$ has a positive inner product with $\nabla f(\boldsymbol{\theta}_t)$. In nonconvex neural landscapes, Prince shows in Figure `fig:sgd:prince-escape` that this exact stochasticity acts as a beneficial mechanism for escaping spurious basins and saddle plateaus.

> [!note] Source Figure
> *Figure:* Deterministic gradient descent vs. stochastic gradient descent on the nonconvex Gabor regression landscape [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Fig. 6.5, Sec. 6.2, p. 83). (a) Exact line-search GD starting at point 2 remains trapped inside its initial local valley. (b) Minibatch SGD introduces gradient variance $\mathrm{Var}[\nabla f_B(\boldsymbol{\theta})]$ that permits temporary uphill excursions with respect to $f$, enabling trajectory 2 to cross the barrier and settle near the global minimum. (`understandingdl_simonprince_p97_fig1.png`)

When objectives include nonsmooth structure $F = f + g$, proximal gradient descent alternates a forward smooth step with a backward proximal operator. Figure `fig:sgd:pgd-proj` illustrates the special case $g = \delta_{\mathcal{C}}$, where the proximal operator reduces to Euclidean projection onto a convex polytope $\mathcal{C}$, while Figure `fig:sgd:prox-tikz` diagrams the full forward–backward architecture bridging smooth gradient oracles (Mini-SGD, SVRG, SSD) with proximal geometry.

> [!note] Source Figure
> *Figure:* Geometric interpretation of projected gradient descent as proximal splitting with indicator $g = \delta_{\mathcal{C}}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 8.22, Sec. 8.6.1, p. 309). Starting from $\boldsymbol{w} \in \mathcal{C}$, an unconstrained forward step $\boldsymbol{w}' = \boldsymbol{w} - \eta \nabla L_s(\boldsymbol{w})$ exits the feasible set $\mathcal{C}$; the backward operator $\operatorname{prox}_{\eta \delta_{\mathcal{C}}}(\boldsymbol{w}') = \Pi_{\mathcal{C}}(\boldsymbol{w}')$ maps it to the closest feasible point in Euclidean distance, preserving non-expansiveness `(eq:sgd:prox-def)`. (`probabilisticml_kevinmurphy_introduction_p339_fig1.png`)

> [!example] Visual Diagram (LaTeX / PDF)
> *Unified Forward–Backward Operator Splitting architecture for composite stochastic optimization $\min_{\boldsymbol{\theta}} f(\boldsymbol{\theta}) + g(\boldsymbol{\theta})$. An explicit stochastic forward step on the smooth sum $f$ (via Minibatch SGD, SVRG, or a subgradient oracle) produces $\boldsymbol{v}_t$, which is stabilized by the non-expansive backward operator $\operatorname{prox}_{\eta_t g}$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Chs. 6, 10–11); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.6).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

To make the step-size and batch-size trade-offs quantitatively explicit, Figure `fig:sgd:rates-pgf` plots exact analytical upper bounds derived in `(eq:sgd:sc-noiseball)` and `(eq:sgd:minibatch-sc)` on a benchmark $\mu$-strongly convex, $L_{\max}$-smooth finite sum.

> [!example] Visual Diagram (LaTeX / PDF)
> *Quantitative convergence and batch-scaling behavior for $\mu$-strongly convex finite-sum minimization [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Thms. 5.7 & 6.11, pp. 28, 35). (a) For $\mu=0.4, L_{\max}=2.0, \sigma_f^*=1.0$, constant step sizes contract linearly into an irreducible $\frac{2\eta\sigma_f^*}{\mu}$ noise ball; Robbins–Monro decay ($\eta_t \propto 1/t$) pierces the floor at $\mathcal{O}(1/t)$, whereas SVRG (or interpolation $\sigma_f^*=0$) preserves global linear convergence. (b) Exact scaling from `(eq:sgd:minibatch-Lb)`–`(eq:sgd:minibatch-sc)` ($n=10^4, L_f=1, L_{\max}=10, \mu=0.1, \sigma_f^*=2, \varepsilon=0.1$): for $b < b_{\mathrm{crit}}$, iterations drop as $1/b$ at constant total gradient cost; past $b_{\mathrm{crit}}$, curvature $L_f/\mu$ caps the step size and extra samples inflate compute linearly.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

Applying stochastic or nonsmooth first-order methods outside their theoretical operating hypotheses induces systematic divergence, premature stalling, or degraded generalization.

> [!warning] Batch Size vs. Learning Rate Linear Scaling Breakdown
> **Failure Mode.** Practitioners frequently invoke the "linear scaling rule" ($\eta \propto b$) when scaling data-parallel training to large batch sizes. As proved in `(eq:sgd:minibatch-Lb)`–`(eq:sgd:bcrit)`, linear step-size scaling remains valid *only* while variance dominates curvature ($b < b_{\mathrm{crit}}$) [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Lemma 6.5 & Cor. 6.12, pp. 32, 35). Once $b > b_{\mathrm{crit}}$, $L_b \to L_f$ saturates and pushing $\eta > \frac{1}{2L_f}$ violates stability, causing explosive divergence; conversely, holding $\eta$ fixed wastes parallel compute with sublinear speedups [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.1.3, p. 278; Sec. 12.1.3, p. 447). Furthermore, oversized batches strip the implicit gradient-variance penalty $\frac{\eta}{4K}\sum_{k}\lVert \nabla L_{B_k} - \nabla L \rVert^2$, degrading test generalization [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 9.2.2, pp. 144–145).
>
> **Remedy.** Estimate $b_{\mathrm{crit}}$ via the gradient noise scale `(eq:sgd:bcrit)`; cap $\eta(b) \le \min\bigl\{b \eta(1),  \frac{1}{2L_f}\bigr\}$ using short linear warmup in early ill-conditioned regions [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.3, p. 296), and avoid training far beyond $b_{\mathrm{crit}}$ without explicit regularization or second-order preconditioning.

> [!warning] Constant Step-Size Noise Floor Oscillation & Premature Decay
> **Failure Mode.** Running constant-step SGD with $\sigma_f^* > 0$ traps iterates in a persistent oscillation ball of radius $\mathcal{O}(\eta \sigma_f^* / \mu)$ regardless of runtime [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Thm. 5.7, p. 28) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.3.1, p. 294). Conversely, aggressive exponential schedules $\eta_t = \eta_0 e^{-\lambda t}$ violate the Robbins–Monro divergence requirement $\sum_{t=0}^{\infty}\eta_t = \infty$ in `(eq:sgd:robbins-monro)` and freeze parameters prematurely far from $\boldsymbol{\theta}^*$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 8.18b, p. 295).
>
> **Remedy.** Verify $\sum \eta_t = \infty$ and $\sum \eta_t^2 < \infty$ (e.g., polynomial $\eta_t = \eta_0 (1 + \beta t)^{-\alpha}$ with $\alpha \in (1/2, 1]$ for strongly convex problems or $\alpha = 1/2$ for general convex problems), pair cosine/step decay with Polyak–Ruppert iterate averaging [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.4, p. 297), or switch to SVRG/SAGA on fixed convex finite sums.

> [!warning] Subgradient Non-Descent Property & Monotonicity Breakdown
> **Failure Mode.** Unlike negative gradients $-\nabla f(\boldsymbol{\theta})$ of smooth functions, a negative subgradient $-\boldsymbol{g} \in -\partial f(\boldsymbol{\theta})$ at a nonsmooth ridge or kink is *not guaranteed to be a descent direction*: moving along $-\boldsymbol{g}$ can increase $f(\boldsymbol{\theta})$ for every positive step size $\eta > 0$ [[Source - Handbook of Convergence Theorems|stochastic_gradient_decent_algorithm_explain]] (Ch. 9, pp. 46–48) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.1.4.1, p. 282). Similarly, single-sample SGD steps are not descent directions for $f$ even when each $f_i$ is smooth (Figure `fig:sgd:lms-traj`b). Consequently, Armijo backtracking line search fails or stalls at zero step length [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.3, p. 296).
>
> **Remedy.** Never require monotonic loss decrease per step in SSD or non-interpolated SGD, and never terminate based solely on single-step loss increases. Instead, track either the step-weighted iterate average $\bar{\boldsymbol{\theta}}_T$ (guaranteed by `(eq:sgd:ssd-master)`) or the best-loss checkpoint, and replace subgradients with proximal splitting `(eq:sgd:prox-def)` whenever the objective has composite structure $f + g$.

> [!warning] Non-i.i.d. Ordering, Without-Replacement Shuffling & SVRG State Drift
> **Failure Mode.** Drawing consecutive minibatches from unshuffled data sorted by class, patient, or time induces severe gradient bias ($\mathbb{E}[\nabla f_{B_t}] \ne \nabla f$) and stalls learning [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.1.3, p. 280). Even with epoch-level random reshuffling without replacement [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (p. 293, fn. 2) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 6.2.1, p. 85), consecutive minibatches within an epoch are negatively correlated rather than strictly i.i.d.; additionally, reusing a fixed training set across multiple epochs violates unbiasedness with respect to the true population risk $J^*(\boldsymbol{\theta})$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 8.1.3, pp. 280–281). Finally, applying SVRG or SAGA to deep networks trained with dropout, BatchNorm, or stochastic data augmentation breaks control-variate correlation because each $f_i$ acquires independent per-step randomness [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.4.5.3, p. 298).
>
> **Remedy.** Always randomly permute dataset indices before every epoch (random reshuffling) or maintain a streaming shuffle buffer; monitor held-out validation risk for early stopping after multiple passes; and restrict SVRG/SAGA to deterministic convex/ERM pipelines while using minibatch SGD with momentum or adaptive preconditioning for stochastic deep-learning layers.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[Proximal & Subgradient Methods]], [[SVRG & Variance Reduction]], [[Stochastic Gradient Descent (SGD)]]
- **Prerequisites (Directed $\leftarrow$):** [[Convexity & Subgradients]], [[Gradient Descent]], [[Stochastic Gradient Descent (SGD)]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Proximal & Subgradient Methods]], [[SVRG & Variance Reduction]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Convexity & Subgradients]], [[Gradient Descent]], [[Linear & Ridge Regression]], [[Proximal & Subgradient Methods]], [[SVRG & Variance Reduction]], [[Stochastic Gradient Descent (SGD)]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Handbook of Convergence Theorems]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
