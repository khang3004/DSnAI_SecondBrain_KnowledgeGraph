---
type: monograph
domain: "D02"
parent_topic: "[[PT 02 - Linear Models, Generalization, Kernels & Ensembles]]"
sub_topic: "[[ST 02 - Linear, Regularized & Generalized Linear Models]]"
cheatsheet: "[[Cheatsheet - Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]"
part_of: 
  - "[[ST 02 - Linear, Regularized & Generalized Linear Models]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
concepts: 
  - "[[Bias-Variance Trade-off]]"
  - "[[Double Descent]]"
  - "[[Rademacher Complexity & VC Dimension]]"
prerequisites: 
  - "[[Bias-Variance Trade-off]]"
  - "[[Cauchy-Schwarz Inequality]]"
  - "[[Concentration Inequalities]]"
  - "[[Moore-Penrose Pseudoinverse]]"
enables: 
  - "[[Double Descent]]"
  - "[[Gradient Boosted Decision Trees (GBDT)]]"
  - "[[Random Forests & Bagging]]"
related: 
  - "[[Bias-Variance Trade-off]]"
  - "[[Cauchy-Schwarz Inequality]]"
  - "[[Concentration Inequalities]]"
  - "[[Double Descent]]"
  - "[[Gradient Boosted Decision Trees (GBDT)]]"
  - "[[Linear & Ridge Regression]]"
  - "[[Moore-Penrose Pseudoinverse]]"
  - "[[Random Forests & Bagging]]"
  - "[[Weight Decay & Dropout]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-02
---

# Bias-Variance Decomposition, Overfitting, Generalization & Double Descent

> [!abstract] Monograph Navigation
> - **Domain:** [[D02 - Classical, Kernel & Probabilistic Machine Learning|Classical, Kernel & Probabilistic Machine Learning]]
> - **Parent / Sub-Topic:** [[PT 02 - Linear Models, Generalization, Kernels & Ensembles|Linear Models, Generalization, Kernels & Ensembles]] → [[ST 02 - Linear, Regularized & Generalized Linear Models|Linear, Regularized & Generalized Linear Models]]
> - **Visual Cheatsheet:** [[Cheatsheet - Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]
> - **Description:** Derives the exact squared-error bias-variance-noise decomposition, links effective model capacity and VC/Rademacher intuition to classical U-shaped curves, and explains modern overparameterized double descent and benign overfitting.

## Problem Formulation & Unified Notation

The central distinction between numerical optimization and statistical machine learning is *generalization*: minimizing error on previously unseen inputs drawn from the data-generating distribution rather than merely memorizing the training sample [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 5.2, p. 110). Consider a supervised regression problem with inputs $\boldsymbol{x} \in \mathcal{X} \subseteq \mathbb{R}^d$ and scalar targets $y \in \mathbb{R}$ governed by an additive-noise data-generating process:

$$
y = f(\boldsymbol{x}) + \epsilon,
  \qquad
  f(\boldsymbol{x}) = \mathbb{E}[y \mid \boldsymbol{x}],
  \qquad
  \mathbb{E}[\epsilon \mid \boldsymbol{x}] = 0,
  \qquad
  \mathrm{Var}(\epsilon \mid \boldsymbol{x}) = \sigma_\epsilon^2,
$$

where $f : \mathcal{X} \to \mathbb{R}$ is the Bayes-optimal regression function and $\epsilon$ is irreducible observation noise. A training dataset $\mathcal{D} = \{(\boldsymbol{x}_i, y_i)\}_{i=1}^N \overset{\mathrm{i.i.d.}}{\sim} p_{\mathrm{data}}(\boldsymbol{x}, y)$ of size $N$ is used by a learning algorithm $\mathcal{A}$ to produce a hypothesis $\hat{f}_{\mathcal{D}} \in \mathcal{F}$ with $P$ effective parameters (e.g., $\hat{f}_{\mathcal{D}}(\boldsymbol{x}) = \boldsymbol{\phi}(\boldsymbol{x})^{\top} \hat{\boldsymbol{w}}$ for a feature map $\boldsymbol{\phi} : \mathcal{X} \to \mathbb{R}^P$). Under squared-error loss $\ell(\hat{y}, y) = (\hat{y} - y)^2$, we contrast the *empirical training risk* and the *expected population (test) risk*:

$$
\hat{R}_N(\hat{f}_{\mathcal{D}}) = \frac{1}{N} \sum_{i=1}^N \big(\hat{f}_{\mathcal{D}}(\boldsymbol{x}_i) - y_i\big)^2,
  \qquad
  R(\hat{f}_{\mathcal{D}}) = \mathbb{E}_{(\boldsymbol{x}, y) \sim p_{\mathrm{data}}}\!\Big[\big(\hat{f}_{\mathcal{D}}(\boldsymbol{x}) - y\big)^2\Big].
$$

Averaging $R(\hat{f}_{\mathcal{D}})$ over random draws of the training set $\mathcal{D}$ of size $N$ yields the *expected generalization error* $\mathbb{E}_{\mathcal{D}}[R(\hat{f}_{\mathcal{D}})]$, which decomposes cleanly into bias, variance, and irreducible noise.

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Because statistical, deep learning, and probabilistic texts adopt different symbols for sample count, parameter count, targets, and predictors, the table below establishes a unified notation used throughout this monograph.
>
> 
>
>
> | Concept | Unified | Goodfellow et al. | Bishop / Chan | Prince / Tiep Vu |
> | — | — | — | — | — |
> | Training sample size | $N$ | $m$ | $N$ | $I$ (Prince), $N$ (Tiep Vu) |
> | Model capacity / params | $P$ | Capacity / $d_{\mathrm{VC}}$ | $M$ (Bishop), $d$ (Chan) | $D$ units / $P$ params |
> | Ground-truth function | $f(\boldsymbol{x})$ | True $p_{\mathrm{data}}(y\mid\boldsymbol{x})$ | $h(\boldsymbol{x})$ (Bishop), $f(\boldsymbol{x})$ (Chan) | True function $f(x)$ |
> | Learned predictor | $\hat{f}_{\mathcal{D}}(\boldsymbol{x})$ | $\hat{g}(\boldsymbol{x})$ / $\hat{\theta}_m$ | $f(\boldsymbol{x}; \mathcal{D})$ (Bishop), $g(\mathcal{D})$ (Chan) | $\mathrm{f}[x, \hat{\phi}]$ / $f(x)$ |
> | Irreducible noise | $\epsilon, \sigma_\epsilon^2$ | Bayes error | $t - h(\boldsymbol{x})$ (Bishop), $e, \sigma^2$ (Chan) | $\epsilon \sim \mathcal{N}(0, \sigma^2)$ |
> | Weight penalty | $\lambda \ge 0$ | Weight decay $\alpha$ | $\lambda$ (Bishop) | $\lambda$ (Tiep Vu) |

## Core Mathematical Derivations & Theorems

### Exact Squared-Error Bias–Variance–Noise Decomposition

> [!important] Theorem: Exact Bias–Variance–Noise Decomposition
> Let $y = f(\boldsymbol{x}) + \epsilon$ satisfy `(eq:biasvar:dgp)` with $\epsilon$ independent of the training set $\mathcal{D}$ at any test point $\boldsymbol{x}$, and define the ensemble-averaged predictor $\bar{f}(\boldsymbol{x}) \triangleq \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(\boldsymbol{x})]$. Then the pointwise expected squared test error decomposes exactly into three non-negative terms [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 4.3, pp. 124–125) [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Thm. 7.5, pp. 431–432):
>
> $$
> \mathbb{E}_{\mathcal{D},  \epsilon}\!\Big[\big(y - \hat{f}_{\mathcal{D}}(\boldsymbol{x})\big)^2  \Big|  \boldsymbol{x}\Big]
>   = \underbrace{\big(\bar{f}(\boldsymbol{x}) - f(\boldsymbol{x})\big)^2}_{\mathrm{Bias}^2(\hat{f}_{\mathcal{D}};  \boldsymbol{x})}
>    +  \underbrace{\mathbb{E}_{\mathcal{D}}\!\Big[\big(\hat{f}_{\mathcal{D}}(\boldsymbol{x}) - \bar{f}(\boldsymbol{x})\big)^2\Big]}_{\mathrm{Var}(\hat{f}_{\mathcal{D}};  \boldsymbol{x})}
>    +  \underbrace{\sigma_\epsilon^2}_{\text{Irreducible Noise}}.
>
> $$
>
> **Proof.** Fix $\boldsymbol{x}$ and add and subtract $f(\boldsymbol{x})$ inside the square:
>
> $$
> \begin{aligned}
> \mathbb{E}_{\mathcal{D},  \epsilon}\!\Big[\big(f(\boldsymbol{x}) + \epsilon - \hat{f}_{\mathcal{D}}(\boldsymbol{x})\big)^2\Big]
>   &= \mathbb{E}_{\mathcal{D}}\!\Big[\big(f(\boldsymbol{x}) - \hat{f}_{\mathcal{D}}(\boldsymbol{x})\big)^2\Big]
>    + 2 \mathbb{E}_{\mathcal{D},  \epsilon}\!\Big[\epsilon \big(f(\boldsymbol{x}) - \hat{f}_{\mathcal{D}}(\boldsymbol{x})\big)\Big]
>    + \mathbb{E}_{\epsilon}[\epsilon^2].
> \end{aligned}
>
> $$
>
> Because test noise $\epsilon$ is independent of $\mathcal{D}$ and $\mathbb{E}[\epsilon \mid \boldsymbol{x}] = 0$, the cross-term vanishes:
>
> $$
>
>   \mathbb{E}_{\mathcal{D},  \epsilon}\!\Big[\epsilon \big(f(\boldsymbol{x}) - \hat{f}_{\mathcal{D}}(\boldsymbol{x})\big)\Big]
>   = \mathbb{E}_{\epsilon}[\epsilon] \cdot \mathbb{E}_{\mathcal{D}}\!\big[f(\boldsymbol{x}) - \hat{f}_{\mathcal{D}}(\boldsymbol{x})\big] = 0,
>
> $$
>
> while $\mathbb{E}_{\epsilon}[\epsilon^2] = \sigma_\epsilon^2$. For the first term of `(eq:biasvar:step1)`, add and subtract $\bar{f}(\boldsymbol{x}) = \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(\boldsymbol{x})]$:
>
> $$
> \begin{aligned}
> \mathbb{E}_{\mathcal{D}}\!\Big[\big(\hat{f}_{\mathcal{D}}(\boldsymbol{x}) - f(\boldsymbol{x})\big)^2\Big]
>   &= \mathbb{E}_{\mathcal{D}}\!\Big[\big(\hat{f}_{\mathcal{D}}(\boldsymbol{x}) - \bar{f}(\boldsymbol{x}) + \bar{f}(\boldsymbol{x}) - f(\boldsymbol{x})\big)^2\Big] \\
>   &= \mathbb{E}_{\mathcal{D}}\!\Big[\big(\hat{f}_{\mathcal{D}}(\boldsymbol{x}) - \bar{f}(\boldsymbol{x})\big)^2\Big]
>    + \big(\bar{f}(\boldsymbol{x}) - f(\boldsymbol{x})\big)^2 \\
>   &\quad + 2\big(\bar{f}(\boldsymbol{x}) - f(\boldsymbol{x})\big)\underbrace{\mathbb{E}_{\mathcal{D}}\!\big[\hat{f}_{\mathcal{D}}(\boldsymbol{x}) - \bar{f}(\boldsymbol{x})\big]}_{= 0}.
> \end{aligned}
>
> $$
>
> Integrating `(eq:biasvar:pointwise_bv)` against $p_{\mathrm{data}}(\boldsymbol{x})$ yields the global expected test risk $\mathbb{E}_{\mathcal{D}}[R(\hat{f}_{\mathcal{D}})] = \int \mathrm{Bias}^2(\boldsymbol{x}) p_{\mathrm{data}}(\boldsymbol{x}) \mathrm{d}\boldsymbol{x} + \int \mathrm{Var}(\boldsymbol{x}) p_{\mathrm{data}}(\boldsymbol{x}) \mathrm{d}\boldsymbol{x} + \sigma_\epsilon^2$.  $\blacksquare$

Note that for parameter estimation of a vector $\boldsymbol{\theta} \in \mathbb{R}^P$ by $\hat{\boldsymbol{\theta}}_N$, the exact same identity holds as $\mathrm{MSE}(\hat{\boldsymbol{\theta}}_N) = \lVert \mathrm{Bias}(\hat{\boldsymbol{\theta}}_N) \rVert_2^2 + \mathrm{Tr}(\mathrm{Cov}(\hat{\boldsymbol{\theta}}_N))$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Eq. 5.53–5.54, p. 129).

> [!tip] Geometric & Physical Meaning of Bias, Variance, and Capacity
> Bias measures *systematic approximation error*: how far the average model $\bar{f}$ across infinitely many training sets lies from the true function $f$. Variance measures *estimation sensitivity*: how wildly individual fits $\hat{f}_{\mathcal{D}}$ fluctuate around $\bar{f}$ due to finite, noisy samples $\mathcal{D}$ [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Fig. 7.15, p. 432). In classical parametric models ($P < N$), increasing capacity $P$ (or decreasing regularization $\lambda$) shrinks $\mathrm{Bias}^2$ by enlarging $\mathcal{F}$, while inflating $\mathrm{Var}$ as the model chases sample noise—producing the classical U-shaped test-error curve [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Fig. 5.6, p. 130).

### Fixed-Design Linear Risk ($N > P$) vs. Minimum-Norm Interpolation ($P \ge N$)

To see both the classical trade-off and the modern interpolation singularity analytically, consider linear regression in a $P$-dimensional feature space with design matrix $\boldsymbol{\Phi} \in \mathbb{R}^{N \times P}$ and targets $\boldsymbol{y} = \boldsymbol{\Phi}\boldsymbol{w}^\star + \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma_\epsilon^2 \boldsymbol{I}_N)$.

> [!important] Theorem: Exact Linear Risk Across the $P = N$ Threshold
> **(1) Underparameterized Regime ($N > P$, $\mathrm{rank}(\boldsymbol{\Phi}) = P$).**
> The ordinary least squares (OLS) estimator $\hat{\boldsymbol{w}} = (\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^{\top}\boldsymbol{y}$ produces in-sample predictions $\hat{\boldsymbol{y}} = \boldsymbol{H}\boldsymbol{y}$, where $\boldsymbol{H} = \boldsymbol{\Phi}(\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^{\top}$ is the orthogonal projection (hat) matrix satisfying $\boldsymbol{H}^2 = \boldsymbol{H} = \boldsymbol{H}^{\top}$ and $\mathrm{Tr}(\boldsymbol{H}) = P$. Evaluating on the same design $\boldsymbol{\Phi}$ with fresh test noise $\boldsymbol{\epsilon}' \sim \mathcal{N}(\boldsymbol{0}, \sigma_\epsilon^2 \boldsymbol{I}_N)$ independent of $\boldsymbol{\epsilon}$ [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Thm. 7.4, pp. 424–425):
>
> $$
> \begin{aligned}
> \mathbb{E}_{\boldsymbol{\epsilon}}[\hat{R}_N] &= \frac{1}{N}\mathbb{E}\big[\lVert (\boldsymbol{H} - \boldsymbol{I}_N)\boldsymbol{\epsilon} \rVert_2^2\big] = \sigma_\epsilon^2\!\left(1 - \frac{P}{N}\right),  \\
>   \mathbb{E}_{\boldsymbol{\epsilon}, \boldsymbol{\epsilon}'}[R_{\mathrm{fixed}}] &= \frac{1}{N}\mathbb{E}\big[\lVert \boldsymbol{H}\boldsymbol{\epsilon} - \boldsymbol{\epsilon}' \rVert_2^2\big] = \sigma_\epsilon^2\!\left(1 + \frac{P}{N}\right).
> \end{aligned}
>
> $$
>
> Here the optimism gap $R_{\mathrm{fixed}} - \hat{R}_N = \frac{2P}{N}\sigma_\epsilon^2$ grows linearly with $P/N$ until $\hat{R}_N \to 0$ at $P = N$.
>
> **(2) Overparameterized Regime ($P > N$, $\mathrm{rank}(\boldsymbol{\Phi}) = N$).**
> When $P > N$, $\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}$ is singular and infinitely many interpolators satisfy $\boldsymbol{\Phi}\boldsymbol{w} = \boldsymbol{y}$. Running gradient descent from $\boldsymbol{w}_0 = \boldsymbol{0}$ or taking the ridgeless limit $\lambda \to 0^+$ of Ridge regression $\hat{\boldsymbol{w}}_\lambda = (\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi} + \lambda\boldsymbol{I}_P)^{-1}\boldsymbol{\Phi}^{\top}\boldsymbol{y}$ selects the unique *minimum-$\ell_2$-norm interpolator*:
>
> $$
> \hat{\boldsymbol{w}}_{\mathrm{min}} = \arg\min_{\boldsymbol{w} : \boldsymbol{\Phi}\boldsymbol{w} = \boldsymbol{y}} \lVert \boldsymbol{w} \rVert_2 = \boldsymbol{\Phi}^{+} \boldsymbol{y} = \boldsymbol{\Phi}^{\top}(\boldsymbol{\Phi}\boldsymbol{\Phi}^{\top})^{-1} \boldsymbol{y}.
>
> $$
>
> Decomposing $\boldsymbol{y} = \boldsymbol{\Phi}\boldsymbol{w}^\star + \boldsymbol{\epsilon}$ and letting $\boldsymbol{P}_{\parallel} = \boldsymbol{\Phi}^{+}\boldsymbol{\Phi}$ be the orthogonal projector onto $\mathrm{row}(\boldsymbol{\Phi})$, the parameter bias and variance for an out-of-sample test covariance $\boldsymbol{\Sigma} = \mathbb{E}[\boldsymbol{\phi}(\boldsymbol{x})\boldsymbol{\phi}(\boldsymbol{x})^{\top}]$ are:
>
> $$
> \mathrm{Bias}^2(\hat{\boldsymbol{w}}_{\mathrm{min}}) = \lVert \boldsymbol{\Sigma}^{1/2}(\boldsymbol{I}_P - \boldsymbol{P}_{\parallel})\boldsymbol{w}^\star \rVert_2^2,
>   \qquad
>   \mathrm{Var}(\hat{\boldsymbol{w}}_{\mathrm{min}}) = \sigma_\epsilon^2 \mathrm{Tr}\!\big(\boldsymbol{\Sigma} (\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi})^{+}\big) = \sigma_\epsilon^2 \sum_{j=1}^N \frac{\boldsymbol{v}_j^{\top} \boldsymbol{\Sigma}\boldsymbol{v}_j}{s_j^2(\boldsymbol{\Phi})},
>
> $$
>
> where $\{s_j(\boldsymbol{\Phi}), \boldsymbol{v}_j\}_{j=1}^N$ are the nonzero singular values and right singular vectors of $\boldsymbol{\Phi}$.

Equation `(eq:biasvar:overparam_bv)` explains the **double descent cusp** rigorously: for isotropic random features ($\boldsymbol{\Sigma} = \boldsymbol{I}_P$), the Marchenko–Pastur law gives $s_{\min}^2(\boldsymbol{\Phi}) \propto (\sqrt{P} - \sqrt{N})^2$. As $P \to N$, $s_{\min}(\boldsymbol{\Phi}) \to 0$ and the variance $\sigma_\epsilon^2 \sum_{j=1}^N s_j^{-2} \approx \frac{\sigma_\epsilon^2}{|1 - P/N|}$ diverges, forcing the model to contort its weights to fit noise [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 8.4, pp. 129–131). Once $P \gg N$, additional orthogonal directions absorb the noise energy across $P - N$ extra degrees of freedom, $s_{\min}(\boldsymbol{\Phi})$ grows again, and $\mathrm{Var}(\hat{\boldsymbol{w}}_{\mathrm{min}}) \approx \sigma_\epsilon^2 \frac{N}{P - N} \to 0$ (**benign overfitting**), while Ridge regularization ($\lambda > 0$) replaces $s_j^{-2}$ with $\frac{s_j^2}{(s_j^2 + \lambda)^2}$, completely smoothing out the $P = N$ singularity.

### Capacity Bounds: VC-Dimension and Rademacher Complexity

Statistical learning theory upper-bounds the generalization gap $R(f) - \hat{R}_N(f)$ uniformly over $\mathcal{F}$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 5.2, p. 114):

> [!important] Uniform Convergence Capacity Bounds
> **(1) VC-Dimension Bound (Binary Classification).**
> If $\mathcal{F}$ has Vapnik–Chervonenkis dimension $d_{\mathrm{VC}}$ (the largest $m$ such that $\mathcal{F}$ can shatter $m$ points), then with probability at least $1 - \delta$ over $\mathcal{D}$:
>
> $$
> R(f) \le \hat{R}_N(f) + \sqrt{\frac{d_{\mathrm{VC}}\big(\ln(2N/d_{\mathrm{VC}}) + 1\big) + \ln(4/\delta)}{N}},
>   \qquad \forall f \in \mathcal{F}.
>
> $$
>
> **(2) Rademacher Complexity Bound (Norm-Constrained Predictors).**
> For bounded losses $\ell \in [0, 1]$ and empirical Rademacher complexity $\hat{\mathfrak{R}}_N(\mathcal{F}) = \mathbb{E}_{\boldsymbol{\sigma} \in \{\pm 1\}^N}\!\big[\sup_{f \in \mathcal{F}} \frac{1}{N}\sum_{i=1}^N \sigma_i f(\boldsymbol{x}_i)\big]$, with probability at least $1 - \delta$:
>
> $$
> R(f) \le \hat{R}_N(f) + 2\hat{\mathfrak{R}}_N(\mathcal{F}) + 3\sqrt{\frac{\ln(2/\delta)}{2N}}.
>
> $$
>
> For linear predictors $\lVert \boldsymbol{w} \rVert_2 \le B$ with $\lVert \boldsymbol{\phi}(\boldsymbol{x}_i) \rVert_2 \le R_\phi$, $\hat{\mathfrak{R}}_N(\mathcal{F}) \le \frac{B R_\phi}{\sqrt{N}}$ *independently of $P$*.

While `(eq:biasvar:vc_bound)` becomes vacuous when $P \ge N$ (since $d_{\mathrm{VC}} \propto P$), the norm-based Rademacher bound `(eq:biasvar:rademacher)` shows why overparameterized models that find small-norm interpolators ($\lVert \hat{\boldsymbol{w}}_{\mathrm{min}} \rVert_2$) can generalize well even in infinite-dimensional feature spaces.

> [!tip] Epoch-Wise Double Descent, Implicit Bias & the Goldilocks Zone
> Double descent occurs not only as parameter count $P$ increases (*model-wise*), but also as training time $t$ increases (*epoch-wise*) and sample count $N$ increases (*sample-wise*) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 8.4, pp. 127–130). During early training, gradient descent learns low-frequency, large-singular-value components first; near the interpolation epoch, it fits label noise and test error peaks; with continued training or wider networks, implicit regularization drives weights toward minimum-norm / flat-minima solutions inside a "Goldilocks zone" of $\ell_2$ weight norm where the model interpolates smoothly between sparse high-dimensional training points [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 20.4.4–20.4.5, pp. 412–413).

## Algorithmic Implementation & Comparative Synthesis

To empirically locate optimal capacity or regularization $\lambda$ without leaking test data—and to diagnose whether a model sits in the underparameterized, critical, or overparameterized regime—we pair nested $K$-fold cross-validation [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 5.3.1, p. 122) [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 8.2, pp. 93–95) with a systematic capacity/regularization sweep (Algorithm `alg:biasvar:nested_cv`).

> [!example] Algorithm: Nested $K$-Fold Cross-Validation & Empirical Double-Descent Sweep
> **Input:** Dataset $\mathcal{D} = \{(\boldsymbol{x}_i, y_i)\}_{i=1}^N$; capacity grid $\mathcal{P} = \{P_1, \dots, P_m\}$; ridge grid $\Lambda = \{\lambda_1, \dots, \lambda_r\}$; outer folds $K_{\mathrm{out}}$, inner folds $K_{\mathrm{in}}$.
> **Output:** Unbiased outer test risk curve $\hat{R}_{\mathrm{test}}(P)$ and optimal $(\hat{P}^\star, \hat{\lambda}^\star)$.
> Partition $\mathcal{D}$ into $K_{\mathrm{out}}$ disjoint outer folds $\{\mathcal{D}_1, \dots, \mathcal{D}_{K_{\mathrm{out}}}\}$
> **for** $P \in \mathcal{P}$:
>
>   **for** $k = 1, \dots, K_{\mathrm{out}}$:
>
>     $\mathcal{D}_{\mathrm{train}}^{(k)} \leftarrow \mathcal{D} \setminus \mathcal{D}_k$, \quad $\mathcal{D}_{\mathrm{test}}^{(k)} \leftarrow \mathcal{D}_k$
>     Partition $\mathcal{D}_{\mathrm{train}}^{(k)}$ into $K_{\mathrm{in}}$ inner folds $\{\mathcal{I}_1, \dots, \mathcal{I}_{K_{\mathrm{in}}}\}$
>     **for** $\lambda \in \Lambda$:
>
>       $\mathrm{CV}_{\mathrm{in}}(\lambda) \leftarrow \frac{1}{K_{\mathrm{in}}} \sum_{j=1}^{K_{\mathrm{in}}} \frac{1}{|\mathcal{I}_j|} \sum_{(\boldsymbol{x}, y) \in \mathcal{I}_j} \big(\hat{f}_{P,\lambda}(\boldsymbol{x}; \mathcal{D}_{\mathrm{train}}^{(k)} \setminus \mathcal{I}_j) - y\big)^2$
>
>     $\lambda_k^\star \leftarrow \arg\min_{\lambda \in \Lambda} \mathrm{CV}_{\mathrm{in}}(\lambda)$
>     Fit $\hat{f}_{P, \lambda_k^\star}$ on full $\mathcal{D}_{\mathrm{train}}^{(k)}$ (via min-norm $\boldsymbol{\Phi}^{+}\boldsymbol{y}$ if $\lambda_k^\star = 0$ and $P > |\mathcal{D}_{\mathrm{train}}^{(k)}|$)
>     $e_k(P) \leftarrow \frac{1}{|\mathcal{D}_{\mathrm{test}}^{(k)}|} \sum_{(\boldsymbol{x}, y) \in \mathcal{D}_{\mathrm{test}}^{(k)}} \big(\hat{f}_{P, \lambda_k^\star}(\boldsymbol{x}) - y\big)^2$
>
>   $\hat{R}_{\mathrm{test}}(P) \leftarrow \frac{1}{K_{\mathrm{out}}} \sum_{k=1}^{K_{\mathrm{out}}} e_k(P)$
>
> **return** $\hat{R}_{\mathrm{test}}(\cdot)$ and $(\hat{P}^\star, \hat{\lambda}^\star) = \arg\min_{P \in \mathcal{P}} \hat{R}_{\mathrm{test}}(P)$

Table `tab:biasvar:regimes` summarizes the statistical and computational trade-offs across the four generalization regimes.

**Table:** Statistical Behavior and Computational Complexity Across Generalization Regimes

| Regime | Ratio $P/N$ | $\mathrm{Bias}^2$ | $\mathrm{Var}$ | Complexity & Dominant Mechanism |
| — | — | — | — | — |
| Underparameterized | $P \ll N$ | High | Low ($\propto \frac{P}{N}\sigma_\epsilon^2$) | $\mathcal{O}(NP^2 + P^3)$; model lacks capacity to represent $f$. |
| Classical Sweet Spot | $P < N$ | Moderate | Moderate | $\mathcal{O}(NP^2 + P^3)$; optimal classical bias–variance balance. |
| Critical Interpolation | $P \approx N$ | Low | Divergent ($\propto \frac{\sigma_\epsilon^2}{\mid 1 - P/N\mid }$) | $\kappa(\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}) \to \infty$; weights contort to fit $\epsilon$ exactly. |
| Overparameterized Min-Norm | $P \gg N$ | Low | Decaying ($\propto \frac{N}{P-N}\sigma_\epsilon^2$) | $\mathcal{O}(PN^2 + N^3)$ dual solve; implicit min-$\ell_2$-norm smoothness. |
| Ridge-Regularized ($\lambda > 0$) | Any $P/N$ | Controlled | Bounded ($\le \frac{\sigma_\epsilon^2}{4\lambda}$) | $\mathcal{O}(\min(NP^2, PN^2))$; damps small singular values $s_j$. |

> [!info] Cross-Book Synthesis: Classical Trade-Off vs. Modern Double Descent
> Classical treatments [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 5.2–5.4) [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 4.3) [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Sec. 7.2–7.3) [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 8) focus on the $P \le N$ regime, where increasing capacity monotonically trades bias for variance and produces a U-shaped test error curve mitigated by validation sets, early stopping, and $\ell_1/\ell_2$ regularization. Prince [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 8 & 20) extends this classical picture beyond the interpolation threshold $P = N$: when $\mathcal{F}$ is overparameterized and optimization favors minimum-norm or flat-minima solutions, the classical U-curve forms only the left half of a *double descent* curve, with the second descent driven by smoother high-dimensional interpolation.

## Visual Diagnostics & Geometric Structure

Figure `fig:biasvar:tikz_regimes` maps the four capacity regimes geometrically, while Figure `fig:biasvar:pgf_double_descent` plots the exact Bias$^2$, Variance, and Total Excess Risk across $P/N$ alongside the smoothing effect of Ridge regularization.

> [!example] Visual Diagram (LaTeX / PDF)
> *Conceptual regime map across the capacity-to-sample ratio $P/N$. Classical learning theory governs Regimes 1–3; implicit minimum-$\ell_2$-norm bias governs Regime 4.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Analytical Bias$^2$, Variance, and Total Excess Test Risk across $P/N$. In the ridgeless limit ($\lambda \to 0^+$), variance diverges at $P/N = 1$ before descending again in the overparameterized regime ($P > N$). Positive Ridge regularization ($\lambda > 0$) eliminates the interpolation cusp.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:biasvar:bishop_fits` illustrates the classical frequentist ensemble experiment on sinusoidal data, showing how decreasing regularization $\lambda$ shifts error from bias to variance. Figures `fig:biasvar:prince_dd` and `fig:biasvar:prince_smooth` demonstrate empirical double descent across MNIST-1D, MNIST, and CIFAR-100, together with the geometric smoothing mechanism in overparameterized networks.

> [!note] Source Figure
> *Figure:* Classical bias–variance trade-off on $L=100$ datasets ($N=25$, $M=25$ Gaussian basis functions) as $\ln\lambda$ decreases from $3$ to $-3$: left column shows $20$ individual fits $\hat{f}_{\mathcal{D}}$ (variance); right column compares the ensemble mean $\bar{f}$ (red) against $f(x)=\sin(2\pi x)$ (green, bias) [[Source - Foundations|deep_learning_foundations_concepts]] (Fig. 4.7, p. 127). (`deep_learning_foundations_concepts_p146_fig1.png`)

> [!note] Source Figure
> *Figure:* Empirical double descent across (a–b) MNIST-1D without and with $15\%$ label noise, (c) shallow networks on MNIST, and (d) ResNet-18 on CIFAR-100. Test error peaks near the interpolation threshold (dashed vertical line) and descends below the classical minimum in the overparameterized regime [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Fig. 8.10, p. 130). (`understandingdl_simonprince_p144_fig1.png`)

> [!note] Source Figure
> *Figure:* Geometric mechanism of the second descent: (a) sparse training points in high-dimensional space; (b) a model with just enough capacity ($6$ units) contorts wildly between points; (c–f) adding hidden units enables progressively smoother minimum-norm interpolation [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Fig. 8.11, p. 131). (`understandingdl_simonprince_p145_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Validation-Set Leakage and Test-Set Staleness
> Selecting hyperparameters ($\lambda$, polynomial degree, early-stopping epoch) on the test set—or fitting preprocessing statistics (mean, variance, PCA) on $\mathcal{D}_{\mathrm{train}} \cup \mathcal{D}_{\mathrm{val}}$ prior to splitting—leaks information and produces optimistically biased risk estimates [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 5.3, p. 121) [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 8.2, p. 94). Always compute all data-dependent transforms strictly inside each training fold (Algorithm `alg:biasvar:nested_cv`).

> [!warning] Ill-Conditioning and Variance Catastrophe at $P \approx N$
> Solving unregularized normal equations $\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}\boldsymbol{w} = \boldsymbol{\Phi}^{\top}\boldsymbol{y}$ near the interpolation threshold $P \approx N$ is numerically catastrophic: the smallest singular value $s_{\min}(\boldsymbol{\Phi}) \to 0$, inflating the condition number $\kappa(\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}) = (s_{\max}/s_{\min})^2$ and amplifying both label noise $\sigma_\epsilon^2$ and floating-point roundoff. Remedy: add explicit Ridge damping $\lambda > 0$, use SVD/QR solvers rather than forming $\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}$, or move cleanly into the overparameterized regime with weight decay [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 8.4, p. 129).

> [!warning] Misinterpreting Zero Training Loss as Generalization (Confidence Blow-Up & Grokking)
> In classification with cross-entropy loss, test accuracy can plateau while test negative log-likelihood *increases* because softmax logits diverge to drive training probabilities toward $1$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 8.1, p. 120). Conversely, in *grokking*, training loss reaches zero long before generalization suddenly improves as weight decay shrinks $\lVert \boldsymbol{w} \rVert_2$ into the Goldilocks zone [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 20.4.4, pp. 412–413). Never terminate model diagnostics solely on $\hat{R}_N \approx 0$.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[Bias-Variance Trade-off]], [[Double Descent]], [[Rademacher Complexity & VC Dimension]]
- **Prerequisites (Directed $\leftarrow$):** [[Bias-Variance Trade-off]], [[Cauchy-Schwarz Inequality]], [[Concentration Inequalities]], [[Moore-Penrose Pseudoinverse]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Double Descent]], [[Gradient Boosted Decision Trees (GBDT)]], [[Random Forests & Bagging]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Bias-Variance Trade-off]], [[Cauchy-Schwarz Inequality]], [[Concentration Inequalities]], [[Double Descent]], [[Gradient Boosted Decision Trees (GBDT)]], [[Linear & Ridge Regression]], [[Moore-Penrose Pseudoinverse]], [[Random Forests & Bagging]], [[Weight Decay & Dropout]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Machine Learning co ban]], [[Source - Understanding Deep Learning]]
