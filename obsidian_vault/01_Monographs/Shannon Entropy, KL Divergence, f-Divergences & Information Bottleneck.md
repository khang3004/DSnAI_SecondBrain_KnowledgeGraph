---
type: monograph
domain: "D01"
parent_topic: "[[PT 01 - Probability Theory, Information Theory & Optimal Transport]]"
sub_topic: "[[ST 01 - Information Theory, Divergences & Optimal Transport]]"
cheatsheet: "[[Cheatsheet - Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]"
part_of: 
  - "[[ST 01 - Information Theory, Divergences & Optimal Transport]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
concepts: 
  - "[[Information Bottleneck]]"
  - "[[KL Divergence]]"
  - "[[Mutual Information & DPI]]"
  - "[[Shannon Entropy]]"
  - "[[f-Divergence]]"
prerequisites: 
  - "[[KL Divergence]]"
  - "[[Mutual Information & DPI]]"
  - "[[Shannon Entropy]]"
enables: 
  - "[[DPO (Direct Preference Optimization)]]"
  - "[[Information Bottleneck]]"
  - "[[KL Divergence]]"
  - "[[Maximum Likelihood & Fisher Info]]"
  - "[[Mutual Information & DPI]]"
  - "[[RLHF (KL-Regularized)]]"
  - "[[Sinkhorn Algorithm]]"
  - "[[TRPO]]"
  - "[[Variational Inference & ELBO]]"
related: 
  - "[[DPO (Direct Preference Optimization)]]"
  - "[[Information Bottleneck]]"
  - "[[KL Divergence]]"
  - "[[Maximum Likelihood & Fisher Info]]"
  - "[[Mutual Information & DPI]]"
  - "[[RLHF (KL-Regularized)]]"
  - "[[Shannon Entropy]]"
  - "[[TRPO]]"
  - "[[Variational Autoencoder (VAE)]]"
  - "[[Variational Inference & ELBO]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-01
---

# Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck

> [!abstract] Monograph Navigation
> - **Domain:** [[D01 - Mathematical Foundations for Machine Intelligence|Mathematical Foundations for Machine Intelligence]]
> - **Parent / Sub-Topic:** [[PT 01 - Probability Theory, Information Theory & Optimal Transport|Probability Theory, Information Theory & Optimal Transport]] → [[ST 01 - Information Theory, Divergences & Optimal Transport|Information Theory, Divergences & Optimal Transport]]
> - **Visual Cheatsheet:** [[Cheatsheet - Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]
> - **Description:** Derives entropy, cross-entropy, forward vs. reverse KL asymmetry (mode-covering vs. mode-seeking), mutual information, data-processing inequality, Jensen-Shannon divergence, and the Information Bottleneck.

## Problem Formulation & Unified Notation

Information theory furnishes the variational geometry underlying statistical estimation, data compression, representation learning, and generative modeling. Let $(\mathcal{X}, \mathcal{F}, \mu)$ be a $\sigma$-finite reference measure space (counting measure on a finite alphabet $\lvert \mathcal{X} \rvert = K$ in the discrete case, or Lebesgue measure on $\mathbb{R}^d$ in the continuous case). We denote probability distributions $P, Q$ by their Radon–Nikodym densities (or probability mass functions) $p = \mathrm{d}P / \mathrm{d}\mu$ and $q = \mathrm{d}Q / \mathrm{d}\mu$ on $\mathcal{X}$. Whenever comparing $P$ against $Q$ via likelihood ratios $u(x) = p(x)/q(x)$, we require absolute continuity $P \ll Q$, meaning $q(x) = 0 \implies p(x) = 0$ $\mu$-almost everywhere [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.1.2, p. 222) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 3.12, p. 71).

Throughout this monograph, we adopt the natural logarithm $\ln$ (units of *nats*) as the default analytical base for derivatives, convex conjugates, and variational bounds, explicitly indicating base-$2$ logarithms $\log_2$ (units of *bits*) when interpreting binary coding lengths ($1\text{ nat} = (\ln 2)^{-1} \approx 1.4427\text{ bits}$; $1\text{ bit} = \ln 2 \approx 0.6931\text{ nats}$); see [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.1.3.1, p. 224). We enforce the information-theoretic continuity convention $0 \ln 0 := 0$ and $0 \ln(0/q) := 0$, justified by $\lim_{\epsilon \to 0^+} \epsilon \ln \epsilon = 0$ [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 2.5.1, p. 46) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 3.13, p. 75).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> The five source texts exhibit deliberate pedagogical divergences in logarithmic base, symbol overloading, and parametric conventions that must be reconciled before chaining derivations across domains:
> 
>
>
>
> | **Concept** | **Unified Monograph** | **Source Variants** | **Reconciliation** |
> | — | — | — | — |
> | Logarithm & units | $\ln$ (nats) default; $\log_2$ (bits) for coding | $\log_2$ in Murphy Intro; $\ln$ in Murphy Adv., Goodfellow, Prince; mixed in Bishop | Multiply nats by $(\ln 2)^{-1}$ to convert to bits. |
> | Self-information | $\iota(x) \triangleq -\ln p(x)$ | $h(x)$ in Bishop; $I(x)$ in Goodfellow | Avoids collision with differential entropy $h(X)$ and MI $\mathrm{I}(X;Y)$. |
> | Discrete vs. diff. entropy | $\mathrm{H}(X)$ discrete; $h(X)$ continuous | $\mathrm{H}(X)$ vs. $h(X)$ in Murphy; $\mathrm{H}[x]$ for both in Bishop | Distinguishes non-negative $\mathrm{H}(X)\ge 0$ from coordinate-dependent $h(X)\in\mathbb{R}$. |
> | Cross-entropy | $H_{\mathrm{ce}}(p, q) \triangleq -\mathbb{E}_p[\ln q(X)]$ | $H_{\mathrm{ce}}(p,q)$ in Murphy; $H(P,Q)$ in Goodfellow | Disambiguates joint entropy $\mathrm{H}(X,Y)$ from cross-entropy $H_{\mathrm{ce}}(p,q)$. |
> | KL divergence | $D_{\mathrm{KL}}(p \parallel q)$ | $D_{\mathrm{KL}}(p\parallel q)$ in Murphy, Goodfellow, Prince; $\mathrm{KL}(p\parallel q)$ in Bishop | Ordered argument convention: $p$ is the expectation/reference measure. |
> | $\alpha$-divergence | Minka $\alpha \in \mathbb{R}\setminus\{0,1\}$ | Amari $\alpha' \in \mathbb{R}\setminus\{\pm 1\}$ vs. Minka $\alpha$ | $\alpha' = 2\alpha - 1$; $\alpha \to 0$ gives reverse KL, $\alpha \to 1$ gives forward KL. |
> | Jensen–Shannon scaling | $D_{\mathrm{JS}}(p \parallel q) = \frac{1}{2}D_{\mathrm{KL}}(p \parallel m) + \frac{1}{2}D_{\mathrm{KL}}(q \parallel m)$ | $D_{\mathrm{JS}}$ with $\frac{1}{2}$ in Prince/Murphy; $2D_{\mathrm{JS}}$ in some $f$-GAN tables | Bounded in $[0, \ln 2]$ nats ($[0,1]$ bits) with $m = \frac{p+q}{2}$. |
> | IB latent variable | $T$ (with $T \equiv Z$ in Murphy Adv.) | $Z$ in Murphy Adv. Sec. 5.6; $T$ in Tishby IB | Markov chain $Y \leftrightarrow X \to T$ for data; $X \to T \to \hat{Y}$ for inference. |
>
>
> 
> *Synthesized Sources:* [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 6.1–6.3, pp. 207–225), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 2.7.1, 5.1–5.6, 26.2, pp. 55–57, 219–256, 897), [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 2.5, pp. 46–53), [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 3.13, pp. 73–76), and [[Source - Understanding Deep Learning|understandingdl_simonprince]] (App. C.5, pp. 459–461).

## Core Mathematical Derivations & Theorems

### Shannon Entropy, Multiplicity, and Maximum Entropy

Information theory formalizes uncertainty through three axioms on the self-information (surprisal) $\iota(x)$ of observing an event of probability $p(x)$: (i) certainty carries zero information ($\iota(1) = 0$), (ii) strictly decreasing monotonicity in $p(x)$, and (iii) additivity across independent events $p(x, y) = p(x)p(y) \implies \iota(x, y) = \iota(x) + \iota(y)$ [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 2.5.1, p. 46) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 3.13, p. 73). The unique continuous solution up to a positive scale factor (log base) is $\iota(x) = -\ln p(x)$.

> [!important] Shannon Entropy, Statistical Multiplicity & Differential Entropy
> For a discrete random variable $X \in \mathcal{X}$ with PMF $p$, the **Shannon entropy** is the expected self-information [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. 6.1, p. 207):
>
> $$
> \mathrm{H}(X) = \mathrm{H}(p) \triangleq -\sum_{x \in \mathcal{X}} p(x) \ln p(x) = \mathbb{E}_{X \sim p}[-\ln p(X)].
> $$
>
> **1. Statistical-Mechanics Derivation (Multiplicity).** Following [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 2.5.2, pp. 47–48), allocate $N$ distinguishable objects across $K$ bins with counts $\{n_k\}_{k=1}^K$ ($\sum_k n_k = N$). The number of microstates per macrostate is the multinomial multiplicity $W = N! / \prod_{k=1}^K n_k!$. Taking $\frac{1}{N}\ln W$ with $p_k = \lim_{N\to\infty} n_k / N$ fixed and applying Stirling's asymptotic expansion $\ln N! = N\ln N - N + O(\ln N)$ yields
>
> $$
> \lim_{N\to\infty} \frac{1}{N} \ln W
>   = \lim_{N\to\infty} \frac{1}{N}\left(N\ln N - N - \sum_{k=1}^K (n_k\ln n_k - n_k)\right)
>   = -\sum_{k=1}^K p_k \ln p_k = \mathrm{H}(p).
> $$
>
>
> **2. Continuum Limit & Differential Entropy.** Discretizing a continuous density $p(x)$ on $\mathbb{R}^d$ into hypercubes of volume $\Delta^d$ with cell probabilities $p(x_i)\Delta^d$ (by the mean value theorem) produces a discrete entropy [[Source - Foundations|deep_learning_foundations_concepts]] (Eq. 2.90, p. 50):
>
> $$
> \mathrm{H}_\Delta = -\sum_i p(x_i)\Delta^d \ln\bigl(p(x_i)\Delta^d\bigr)
>   = -\sum_i p(x_i)\Delta^d \ln p(x_i) - d\ln\Delta.
> $$
>
> Dropping the divergent quantization offset $-d\ln\Delta$ and taking $\Delta \to 0^+$ defines the **differential entropy** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 5.95, p. 235):
>
> $$
> h(X) \triangleq -\int_{\mathcal{X}} p(x) \ln p(x)  \mathrm{d}x.
> $$
>
>
> **3. Maximum Entropy (MaxEnt) Principle.**
> - *Discrete ($K$ states):* Maximizing $\mathrm{H}(p) + \lambda\bigl(\sum_{k=1}^K p_k - 1\bigr)$ gives $\partial_{p_k}\mathcal{L} = -\ln p_k - 1 + \lambda = 0 \implies p_k = 1/K$, with strictly negative Hessian $\nabla^2_{p}\mathcal{L} = -\mathrm{diag}(p_1^{-1}, \dots, p_K^{-1}) \prec \boldsymbol{0}$. Hence $\mathrm{H}(X) \le \ln K$ [[Source - Foundations|deep_learning_foundations_concepts]] (Eq. 2.87–2.88, pp. 48–49).
> - *Continuous (fixed mean $\boldsymbol{\mu} \in \mathbb{R}^d$ and covariance $\boldsymbol{\Sigma} \succ \boldsymbol{0}$):* Applying Euler–Lagrange calculus of variations to $-\int p \ln p \mathrm{d}x$ subject to normalization, $\mathbb{E}_p[\boldsymbol{x}] = \boldsymbol{\mu}$, and $\mathbb{E}_p[(\boldsymbol{x}-\boldsymbol{\mu})(\boldsymbol{x}-\boldsymbol{\mu})^{\top}] = \boldsymbol{\Sigma}$ forces $p^\star(\boldsymbol{x}) = \exp\bigl(\lambda_0 + \boldsymbol{\lambda}_1^{\top} \boldsymbol{x} + \boldsymbol{x}^{\top} \boldsymbol{\Lambda}_2 \boldsymbol{x}\bigr) = \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})$, yielding [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. 6.26, p. 212) [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 2.5.4, pp. 50–51):
>
> $$
> h\bigl(\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})\bigr)
>     = \frac{1}{2}\ln\det(2\pi e \boldsymbol{\Sigma})
>     = \frac{d}{2}\bigl(1 + \ln(2\pi)\bigr) + \frac{1}{2}\ln\det(\boldsymbol{\Sigma}).
> $$

> [!tip] Typical Sets, Source Coding & Quantized Differential Entropy
> Shannon's noiseless source coding theorem establishes that any uniquely decodable prefix code for symbols drawn i.i.d. from $p$ requires expected length at least $\mathrm{H}_2(p) = \mathrm{H}(p)/\ln 2$ bits per symbol [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 2.5.1, p. 47) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.4.1, p. 247). Equivalently, by the Asymptotic Equipartition Property (AEP), almost all probability mass of $N$ i.i.d. draws concentrates on the $\epsilon$-typical set
>
> $$
> A_\epsilon^{(N)} \triangleq \left\{ \boldsymbol{x}_{1:N} \in \mathcal{X}^N : \mathrm{H}(p) - \epsilon \le -\frac{1}{N}\ln p(\boldsymbol{x}_{1:N}) \le \mathrm{H}(p) + \epsilon \right\},
> $$
>
> which contains approximately $e^{N \mathrm{H}(p)}$ nearly equiprobable sequences, each occurring with asymptotic empirical probability $\approx e^{-N \mathrm{H}(p)}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.2.3, pp. 236–237). For continuous $X$ quantized to $n$-bit precision ($\Delta = 2^{-n}$), Eq. `(eq:infotheory:quantized_entropy)` implies $\mathrm{H}_{2^{-n}} \approx h_2(X) + n$ bits; thus $h_2(X) < 0$ simply indicates that $X$ is localized within a sub-unit interval where leading fractional bits are deterministic (e.g., $X \sim \mathcal{U}(0, 1/8)$ saves $3$ bits, giving $h_2(X) = -3$ bits) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 6.1.6, p. 212).

### Kullback–Leibler Divergence, Non-Negativity, and M- vs. I-Projections

Following [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.1.1–5.1.2, pp. 219–223), the **Kullback–Leibler (KL) divergence** (relative entropy) is the unique divergence (up to a positive constant) satisfying continuity on $P \ll Q$, non-negativity, permutation invariance, monotonicity on uniform refinements, and a probability-weighted chain rule across conditional decompositions:

$$
D_{\mathrm{KL}}(p \parallel q) \triangleq \int_{\mathcal{X}} p(x) \ln \frac{p(x)}{q(x)} \mathrm{d}\mu(x) = \mathbb{E}_{X \sim p}\left[\ln \frac{p(X)}{q(X)}\right].
$$

When encoding data from true distribution $p$ using a code optimized for model $q$, the expected code length is the **cross-entropy** $H_{\mathrm{ce}}(p, q) \triangleq -\mathbb{E}_p[\ln q(X)] = \mathrm{H}(p) + D_{\mathrm{KL}}(p \parallel q)$, so $D_{\mathrm{KL}}(p \parallel q)$ measures the exact excess coding penalty [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. 6.31, p. 214) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Eq. 3.51, p. 75).

> [!important] Gibbs' Information Inequality, Gaussian KL & Projection Duality
> **1. Proof of Non-Negativity (Information Inequality).** Let $A = \{x \in \mathcal{X} : p(x) > 0\}$ be the support of $p$. Using the strict concavity of $\ln u$ (equivalently strict convexity of $-\ln u$) and Jensen's inequality [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Thm. 6.2.1, p. 215) [[Source - Foundations|deep_learning_foundations_concepts]] (Eq. 2.105, p. 52):
>
> $$
> \begin{aligned}
>   -D_{\mathrm{KL}}(p \parallel q)
>   &= \int_A p(x)\ln\frac{q(x)}{p(x)} \mathrm{d}\mu(x)
>    \le \ln\left(\int_A p(x)\frac{q(x)}{p(x)} \mathrm{d}\mu(x)\right) \\
>   &= \ln\left(\int_A q(x) \mathrm{d}\mu(x)\right)
>    \le \ln 1 = 0.
> \end{aligned}
> $$
>
> (Alternatively, via the tangent bound $-\ln u \ge 1 - u$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Eq. C.29, p. 460), $D_{\mathrm{KL}}(p \parallel q) \ge \int_A p(1 - q/p) \mathrm{d}\mu = 1 - \int_A q \mathrm{d}\mu \ge 0$.) Equality requires both $q(x)/p(x) = c$ $\mu$-a.e. on $A$ and $\int_A q \mathrm{d}\mu = 1$, forcing $c = 1$ and $p = q$ $\mu$-a.e.
>
> **2. Chain Rule & Reparameterization Invariance.** For any joint distributions $p(x, y) = p(x)p(y|x)$ and $q(x, y) = q(x)q(y|x)$,
>
> $$
> D_{\mathrm{KL}}(p(x, y) \parallel q(x, y)) = D_{\mathrm{KL}}(p(x) \parallel q(x)) + \mathbb{E}_{p(x)}\bigl[D_{\mathrm{KL}}(p(y|x) \parallel q(y|x))\bigr].
> $$
>
> Under any smooth bijection $y = g(x)$, Jacobians cancel in the density ratio $\frac{p_Y(y)}{q_Y(y)} = \frac{p_X(x)\lvert \det \boldsymbol{J}_g(x) \rvert^{-1}}{q_X(x)\lvert \det \boldsymbol{J}_g(x) \rvert^{-1}} = \frac{p_X(x)}{q_X(x)}$, making $D_{\mathrm{KL}}(p \parallel q)$ strictly invariant to invertible coordinate changes [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 5.16, p. 223).
>
> **3. Closed-Form Multivariate Gaussian KL.** For $p = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ and $q = \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$ on $\mathbb{R}^d$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. 6.32, p. 214) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Eq. C.32, p. 461):
>
> $$
> D_{\mathrm{KL}}(p \parallel q) = \frac{1}{2}\left[
>     \mathrm{Tr}(\boldsymbol{\Sigma}_2^{-1} \boldsymbol{\Sigma}_1)
>     + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^{\top} \boldsymbol{\Sigma}_2^{-1} (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)
>     - d + \ln\frac{\det \boldsymbol{\Sigma}_2}{\det \boldsymbol{\Sigma}_1}
>   \right].
> $$
>
>
> **4. Forward KL (M-Projection) vs. Reverse KL (I-Projection).** Approximating a target $p$ within a parametric family $\mathcal{Q}$ reveals fundamental asymmetry [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.1.4, pp. 225–227):
> - *Forward (Inclusive) KL / Moment Projection:* $q_M^\star = \arg\min_{q \in \mathcal{Q}} D_{\mathrm{KL}}(p \parallel q)$. When $p = p_{\mathcal{D}} = \frac{1}{N}\sum_{n=1}^N \delta_{x_n}$, $D_{\mathrm{KL}}(p_{\mathcal{D}} \parallel q_\theta) = -\frac{1}{N}\sum_{n=1}^N \ln q_\theta(x_n) + \text{const}$, recovering Maximum Likelihood Estimation (MLE) [[Source - Foundations|deep_learning_foundations_concepts]] (Eq. 2.106, p. 53) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 3.13, p. 75). For an exponential family $q_\eta(x) = h(x)\exp(\eta^{\top} \boldsymbol{\phi}(x) - A(\eta))$, first-order optimality yields
>
> $$
> \nabla_\eta D_{\mathrm{KL}}(p \parallel q_\eta) = \nabla_\eta A(\eta) - \mathbb{E}_p[\boldsymbol{\phi}(X)] = \mathbb{E}_{q_\eta}[\boldsymbol{\phi}(X)] - \mathbb{E}_p[\boldsymbol{\phi}(X)] = \boldsymbol{0},
> $$
>
>   enforcing *moment matching*. Because $p(x) > 0, q(x) = 0 \implies D_{\mathrm{KL}}(p \parallel q) = +\infty$, $q_M^\star$ is **zero-avoiding** (mode-covering).
> - *Reverse (Exclusive) KL / Information Projection:* $q_I^\star = \arg\min_{q \in \mathcal{Q}} D_{\mathrm{KL}}(q \parallel p)$. Expanding $D_{\mathrm{KL}}(q \parallel p) = -\mathrm{H}(q) - \mathbb{E}_q[\ln \tilde{p}(x)] + \ln Z_p$ avoids the intractable normalizer $Z_p$ and yields the variational inference / ELBO objective. Because $q(x) > 0, p(x) = 0 \implies D_{\mathrm{KL}}(q \parallel p) = +\infty$, $q_I^\star$ is **zero-forcing** (mode-seeking).

> [!tip] Gaussian Covariance Dualism Under Factorized Approximations
> Approximating a correlated bivariate Gaussian $p(\boldsymbol{x}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1})$ by a factorized Gaussian $q(\boldsymbol{x}) = \prod_{i=1}^2 \mathcal{N}(x_i \mid m_i, v_i)$ makes the M- vs. I-projection contrast algebraically exact [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.1.4.2–5.1.4.3, p. 227):
>
> $$
> \text{M-projection: } m_i^\star = \mu_i,   v_i^\star = \Sigma_{ii},
>   \qquad
>   \text{I-projection: } m_i^\star = \mu_i,   v_i^\star = (\Lambda_{ii})^{-1} \le \Sigma_{ii}.
> $$
>
> By the Schur complement identity $(\Lambda_{11})^{-1} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} \le \Sigma_{11}$, forward KL matches marginal variances (covering support, underconfident) whereas reverse KL matches conditional precisions (hugging the high-density ridge, overconfident). On multimodal $p$, M-projection bridges across modes while I-projection locks onto a single high-mass mode (Figure `fig:infotheory:fwd_rev_kl`).

> [!note] Source Figure
> *Figure:* Forward KL vs. reverse KL when fitting a unimodal Gaussian $q$ (red contours) to a bimodal target mixture $p$ (blue contours): **(a)** minimizing forward KL $D_{\mathrm{KL}}(p \parallel q)$ (M-projection) exhibits zero-avoiding, mode-covering behavior spanning both modes; **(b–c)** minimizing reverse KL $D_{\mathrm{KL}}(q \parallel p)$ (I-projection) exhibits zero-forcing, mode-seeking behavior locking onto one mode. Reproduced from [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 6, Fig. 6.3, p. 217). (`probabilisticml_kevinmurphy_introduction_p247_fig1.png`)

### Mutual Information, Data Processing Inequality, and Fano's Bound

Mutual information quantifies statistical dependence by measuring how far a joint distribution $p(x, y)$ departs from independence $p(x)p(y)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. 6.47, p. 217) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 5.109, p. 238):

$$
\mathrm{I}(X; Y) \triangleq D_{\mathrm{KL}}(p(x, y) \parallel p(x)p(y))
  = \iint p(x, y)\ln\frac{p(x, y)}{p(x)p(y)} \mathrm{d}\mu(x) \mathrm{d}\mu(y) \ge 0,
$$

with equality iff $X \perp\!\!\!\perp Y$. Via the product rule $p(x, y) = p(x)p(y|x)$, $\mathrm{I}(X; Y)$ admits the equivalent entropy decompositions $\mathrm{I}(X; Y) = \mathrm{H}(X) - \mathrm{H}(X|Y) = \mathrm{H}(Y) - \mathrm{H}(Y|X) = \mathrm{H}(X) + \mathrm{H}(Y) - \mathrm{H}(X, Y)$ (for discrete variables; replace $\mathrm{H}$ with $h$ for continuous densities, where differential entropies cancel coordinate-Jacobians so $\mathrm{I}(X; Y)$ remains reparameterization invariant). For jointly Gaussian $(X, Y)$ with correlation $\rho \in (-1, 1)$, Eq. `(eq:infotheory:gaussian_entropy)` yields $\mathrm{I}(X; Y) = -\frac{1}{2}\ln(1 - \rho^2)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. 6.70, p. 220).

> [!important] Data Processing Inequality (MI & $f$-Divergence Forms) and Fano's Inequality
> **1. Data Processing Inequality for Mutual Information.** Suppose $X \to Y \to Z$ forms a Markov chain ($X \perp\!\!\!\perp Z \mid Y$, so $\mathrm{I}(X; Z \mid Y) = 0$). Expanding $\mathrm{I}(X; Y, Z)$ in two orders via the chain rule $\mathrm{I}(X; A, B) = \mathrm{I}(X; A) + \mathrm{I}(X; B \mid A)$ gives [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Thm. 6.3.1, p. 223) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Thm. 5.3.1, p. 239):
>
> $$
> \mathrm{I}(X; Y) + \underbrace{\mathrm{I}(X; Z \mid Y)}_{= 0}
>   = \mathrm{I}(X; Y, Z)
>   = \mathrm{I}(X; Z) + \underbrace{\mathrm{I}(X; Y \mid Z)}_{\ge 0}
>   \implies \mathrm{I}(X; Y) \ge \mathrm{I}(X; Z).
> $$
>
> Equality holds iff $\mathrm{I}(X; Y \mid Z) = 0$, i.e., $X \to Z \to Y$ is also a Markov chain ($Z$ is a *sufficient statistic* for $X$ given $Y$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.3.4, p. 240).
>
> **2. Channel Contraction (DPI for KL & $f$-Divergences).** Let $p(y) = \int t(y|x)p(x) \mathrm{d}x$ and $q(y) = \int t(y|x)q(x) \mathrm{d}x$ be outputs of a Markov kernel $t(y|x)$, so $p(x, y) = p(x)t(y|x)$ and $q(x, y) = q(x)t(y|x)$ share conditional $t(y|x)$. For any convex generator $f$, Jensen's inequality applied under $q(x|y)$ yields [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Thm. 5.1.3, p. 229):
>
> $$
> \begin{aligned}
> D_f\bigl(p(x)  \|  q(x)\bigr)
>   &= \iint q(x, y)  f\!\left(\frac{p(x)t(y|x)}{q(x)t(y|x)}\right) \mathrm{d}x \mathrm{d}y  \\
>   &= \int q(y) \left[ \int q(x|y)  f\!\left(\frac{p(x, y)}{q(x, y)}\right) \mathrm{d}x \right] \mathrm{d}y
>   \ge \int q(y)  f\!\left(\int q(x|y)\frac{p(x, y)}{q(x, y)} \mathrm{d}x\right) \mathrm{d}y  \\
>   &= \int q(y)  f\!\left(\frac{p(y)}{q(y)}\right) \mathrm{d}y
>   = D_f\bigl(p(y)  \|  q(y)\bigr).
> \end{aligned}
> $$
>
> Setting $f(u) = u\ln u$ gives $D_{\mathrm{KL}}(p(x) \parallel q(x)) \ge D_{\mathrm{KL}}(p(y) \parallel q(y))$; setting $t(y|x)$ to coordinate projection gives monotonicity $D_{\mathrm{KL}}(p(x,y) \parallel q(x,y)) \ge D_{\mathrm{KL}}(p(x) \parallel q(x))$.
>
> **3. Fano's Inequality.** For Markov chain $Y \to X \to \hat{Y} = g(X)$ over discrete labels $\mathcal{Y}$ with error indicator $E = \mathbb{I}(\hat{Y} \ne Y)$ and error probability $P_e = \mathbb{P}(Y \ne \hat{Y})$, expanding $\mathrm{H}(E, Y \mid \hat{Y})$ yields $\mathrm{H}(Y \mid \hat{Y}) + \mathrm{H}(E \mid Y, \hat{Y}) = \mathrm{H}(E \mid \hat{Y}) + \mathrm{H}(Y \mid E, \hat{Y})$. Because $\mathrm{H}(E \mid Y, \hat{Y}) = 0$, $\mathrm{H}(E \mid \hat{Y}) \le \mathrm{H}(E) = H_e(P_e)$, and $\mathrm{H}(Y \mid E, \hat{Y}) \le P_e \ln(\lvert \mathcal{Y} \rvert - 1)$ (or $P_e \ln\lvert \mathcal{Y} \rvert$ in weak form), combining with DPI $\mathrm{H}(Y \mid X) \le \mathrm{H}(Y \mid \hat{Y})$ establishes [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Thm. 6.3.2, p. 225):
>
> $$
> \mathrm{H}(Y \mid X) \le \mathrm{H}(E) + P_e \ln\lvert \mathcal{Y} \rvert
>   \implies
>   P_e \ge \frac{\mathrm{H}(Y \mid X) - \ln 2}{\ln\lvert \mathcal{Y} \rvert}
>   = \frac{\mathrm{H}(Y) - \mathrm{I}(X; Y) - \ln 2}{\ln\lvert \mathcal{Y} \rvert}.
> $$

### Csisz\'ar $f$-Divergences, Jensen–Shannon, and Pinsker's Inequality

Generalizing the density-ratio geometry of KL, a **Csisz\'ar $f$-divergence** compares $p$ and $q$ through any convex function $f : (0, \infty) \to \mathbb{R}$ satisfying $f(1) = 0$ (with boundary convention $0 f(0/0) := 0$ and $0 f(a/0) := a\lim_{u\to\infty} f(u)/u$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 2.7.1, p. 55):

$$
D_f(p  \|  q) \triangleq \int_{\mathcal{X}} q(x)  f\!\left(\frac{p(x)}{q(x)}\right) \mathrm{d}\mu(x) \ge f\!\left(\int_{\mathcal{X}} q(x)\frac{p(x)}{q(x)} \mathrm{d}\mu(x)\right) = f(1) = 0,
$$

where non-negativity follows directly from Jensen's inequality. Notice an affine gauge invariance: replacing $f(u)$ by $\tilde{f}(u) = f(u) + c(u - 1)$ leaves $D_{\tilde{f}}(p\|q) = D_f(p\|q)$ unchanged for normalized probability measures since $\mathbb{E}_q[p/q - 1] = 0$. Setting $c = -f'(1)$ (when differentiable at $1$) yields the **canonical non-negative generator** $\tilde{f}(1) = \tilde{f}'(1) = 0$ with $\tilde{f}(u) \ge 0$ everywhere.

> [!important] Pinsker's Inequality, Variational $f$-Divergence Bounds & Fisher–Rao Metric
> **1. Pinsker's Inequality ($D_{\mathrm{KL}}(p \parallel q) \ge 2 D_{\mathrm{TV}}(p, q)^2$).** Let $D_{\mathrm{TV}}(p, q) \triangleq \frac{1}{2}\int_{\mathcal{X}} \lvert p(x) - q(x) \rvert \mathrm{d}\mu(x)$ denote the total variation distance ($f_{\mathrm{TV}}(u) = \frac{1}{2}\lvert u-1 \rvert$), the unique divergence that is simultaneously an $f$-divergence and an integral probability metric (IPM) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 2.7.4, p. 61). Defining $A = \{x \in \mathcal{X} : p(x) \ge q(x)\}$, $\alpha = P(A)$, and $\beta = Q(A)$, we have $\alpha \ge \beta$ and $D_{\mathrm{TV}}(p, q) = \int_A (p - q) \mathrm{d}\mu = \alpha - \beta$. Passing both $P$ and $Q$ through the deterministic indicator map $T(x) = \mathbb{I}(x \in A) \in \{0, 1\}$, Eq. `(eq:infotheory:dpi_fdiv)` gives
>
> $$
> D_{\mathrm{KL}}(p \parallel q) \ge D_{\mathrm{KL}}(\mathrm{Bern}(\alpha) \parallel \mathrm{Bern}(\beta))
>   = \alpha \ln\frac{\alpha}{\beta} + (1 - \alpha)\ln\frac{1 - \alpha}{1 - \beta}.
> $$
>
> Fix $\alpha \in (0, 1)$ (boundary $\alpha\in\{0,1\}$ is immediate) and define $h(\beta) \triangleq \alpha\ln\frac{\alpha}{\beta} + (1-\alpha)\ln\frac{1-\alpha}{1-\beta} - 2(\alpha-\beta)^2$ on $\beta \in (0, \alpha]$. Clearly $h(\alpha) = 0$, and differentiating with respect to $\beta$ gives
>
> $$
> h'(\beta)
>   = -\frac{\alpha}{\beta} + \frac{1-\alpha}{1-\beta} + 4(\alpha - \beta)
>   = (\alpha - \beta)\left(4 - \frac{1}{\beta(1 - \beta)}\right) \le 0,
> $$
>
> since $\beta(1-\beta) \le \frac{1}{4}$ for all $\beta \in (0, 1)$ by AM–GM. Hence $h(\beta) \ge h(\alpha) = 0$ for all $\beta \le \alpha$, proving **Pinsker's inequality**:
>
> $$
> D_{\mathrm{TV}}(p, q) \le \sqrt{\frac{1}{2}D_{\mathrm{KL}}(p \parallel q)}
>   \qquad\Longleftrightarrow\qquad
>   D_{\mathrm{KL}}(p \parallel q) \ge 2 D_{\mathrm{TV}}(p, q)^2 = \frac{1}{2}\|p - q\|_1^2.
> $$
>
>
> **2. Fenchel–Legendre Variational Bound for $f$-Divergences.** Expressing convex $f$ via its biconjugate $f(u) = \sup_{t \in \mathrm{dom}(f^\dagger)} [ut - f^\dagger(t)]$, where $f^\dagger(t) \triangleq \sup_{u \in \mathbb{R}_+} [ut - f(u)]$, yields the lower bound used in $f$-GANs and neural divergence estimation [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. 26.22–26.25, p. 899):
>
> $$
> D_f(p  \|  q)
>   = \sup_{t:\mathcal{X} \to \mathrm{dom}(f^\dagger)} \Bigl(\mathbb{E}_{p}[t(X)] - \mathbb{E}_{q}[f^\dagger(t(X))]\Bigr)
>   \ge \sup_{t \in \mathcal{T}} \Bigl(\mathbb{E}_{p}[t(X)] - \mathbb{E}_{q}[f^\dagger(t(X))]\Bigr),
> $$
>
> achieved with equality at $t^\star(x) = f'(p(x)/q(x))$. Specializing to KL with a shift $\phi = t - 1$ gives the **Donsker–Varadhan variational representation** $D_{\mathrm{KL}}(p \parallel q) = \sup_{\phi} \mathbb{E}_p[\phi(X)] - \ln \mathbb{E}_q[e^{\phi(X)}]$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 5.45, p. 228).
>
> **3. Local Fisher–Rao Geometry & KL Hessian.** Consider a smooth parametric family $p_\theta(x)$ with score $\boldsymbol{s}(\theta) = \nabla_\theta \ln p_\theta(x)$ and Fisher Information Matrix (FIM) $F(\theta) = \mathbb{E}_{p_\theta}[\boldsymbol{s}(\theta)\boldsymbol{s}(\theta)^{\top}] = -\mathbb{E}_{p_\theta}[\nabla_\theta^2 \ln p_\theta(x)]$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 3.3.4, pp. 76–77). Because $\mathbb{E}_{p_\theta}[\boldsymbol{s}(\theta)] = \nabla_\theta \int p_\theta(x) \mathrm{d}x = \boldsymbol{0}$, second-order Taylor expansion of $D_{\mathrm{KL}}(p_\theta \parallel p_{\theta + \delta})$ around $\delta = \boldsymbol{0}$ yields [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 5.81, p. 233):
>
> $$
> \begin{aligned}
>   D_{\mathrm{KL}}(p_\theta \parallel p_{\theta + \delta})
>   &= -\delta^{\top} \underbrace{\mathbb{E}_{p_\theta}[\nabla_\theta \ln p_\theta(x)]}_{= \boldsymbol{0}}
>      - \frac{1}{2}\delta^{\top} \mathbb{E}_{p_\theta}\bigl[\nabla_\theta^2 \ln p_\theta(x)\bigr]\delta + O(\|\delta\|^3) \\
>   &= \frac{1}{2}\delta^{\top} F(\theta)\delta + O(\|\delta\|^3).
> \end{aligned}
> $$
>
> More generally, any twice-differentiable $f$-divergence satisfies $D_f(p_\theta \| p_{\theta+\delta}) = \frac{f"(1)}{2}\delta^{\top} F(\theta)\delta + O(\|\delta\|^3)$ (matching the $\chi^2$ local limit $D_{\mathrm{KL}}(p\|q) \approx \frac{1}{2}\chi^2(p, q)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ex. 6.2, p. 226)). Furthermore, for exponential family $p_\eta(x) = h(x)\exp(\eta^{\top} \boldsymbol{\phi}(x) - A(\eta))$, $D_{\mathrm{KL}}(p_{\eta_1} \parallel p_{\eta_2}) = A(\eta_2) - A(\eta_1) - (\eta_2 - \eta_1)^{\top} \nabla A(\eta_1) = B_A(\eta_2 \| \eta_1)$ equals the Bregman divergence generated by $A(\eta)$—and KL is the *unique* divergence that belongs simultaneously to both the $f$-divergence and Bregman families [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.1.10.1, p. 234).

### Information Bottleneck, Variational IB, and Conditional Entropy Bottleneck

In supervised representation learning, we seek a stochastic bottleneck representation $T$ (denoted $Z$ in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.6)) of input $X$ that preserves predictive information about target $Y$ while compressing irrelevant input nuisance. Under the data-encoder Markov condition $Y \leftrightarrow X \to T$ (so $T \perp\!\!\!\perp Y \mid X$ and $p(x, y, t) = p(x, y)e(t|x)$), DPI guarantees $\mathrm{I}(T; Y) \le \mathrm{I}(X; Y)$. The **Information Bottleneck (IB)** Lagrangian balances compression against predictive sufficiency via $\beta \ge 0$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 5.165, p. 252):

$$
\min_{e(t|x)}   \mathcal{L}_{\mathrm{IB}}[e] \triangleq \beta \mathrm{I}(X; T) - \mathrm{I}(T; Y).
$$

> [!important] Discrete IB Fixed-Point Equations, Variational IB (VIB) & CEB
> **1. Exact Blahut–Arimoto Fixed-Point Equations (Discrete $X, Y, T$).** Introducing Lagrange multipliers $\tilde{\lambda}(x)\bigl(\sum_t e(t|x) - 1\bigr)$ for encoder normalization and differentiating $\mathcal{L}_{\mathrm{IB}}[e]$ with respect to $e(t|x)$:
> - For the rate term $\mathrm{I}(X; T) = \sum_{x,t} p(x)e(t|x)\ln\frac{e(t|x)}{p(t)}$ with $p(t) = \sum_{x'} p(x')e(t|x')$:
>   \[
>     \frac{\partial \mathrm{I}(X; T)}{\partial e(t|x)} = p(x)\ln\frac{e(t|x)}{p(t)} + p(x) - \sum_{x'} p(x')e(t|x')\frac{p(x)}{p(t)} = p(x)\ln\frac{e(t|x)}{p(t)}.
>   \]
> - For the relevance term $\mathrm{I}(T; Y) = \sum_{t,y} p(t,y)\ln\frac{p(y|t)}{p(y)}$ with $p(t,y) = \sum_x p(x,y)e(t|x)$ and $p(y|t) = p(t,y)/p(t)$:
>   \[
>     \frac{\partial \mathrm{I}(T; Y)}{\partial e(t|x)} = p(x)\sum_{y} p(y|x)\ln\frac{p(y|t)}{p(y)} = -p(x)D_{\mathrm{KL}}(p(y|x) \parallel p(y|t)) + p(x)\sum_y p(y|x)\ln\frac{p(y|x)}{p(y)}.
>   \]
>
> Setting $\beta\partial\mathrm{I}(X;T)/\partial e(t|x) - \partial\mathrm{I}(T;Y)/\partial e(t|x) + \tilde{\lambda}(x) = 0$ and normalizing yields the coupled fixed-point system:
>
> $$
> \begin{aligned}
>   e(t|x) &= \frac{p(t)}{Z(x, \beta)}\exp\!\left(-\frac{1}{\beta}D_{\mathrm{KL}}(p(y|x) \parallel p(y|t))\right), \\
>   p(t) &= \sum_x p(x)e(t|x),\qquad
>   p(y|t) = \sum_x \frac{p(x)e(t|x)}{p(t)}p(y|x).
> \end{aligned}
> $$
>
>
> **2. Variational Information Bottleneck (VIB).** When $(X, T, Y)$ are high-dimensional or continuous, exact computation of $p(t)$ and $p(y|t)$ is intractable. Introducing a variational decoder $d_\phi(y|t) \approx p(y|t)$ and variational prior marginal $m(t) \approx p(t)$, non-negativity of KL ($D_{\mathrm{KL}}(p(y|t) \parallel d_\phi(y|t)) \ge 0$ and $D_{\mathrm{KL}}(p(t) \parallel m(t)) \ge 0$) furnishes a tractable lower bound on $\mathrm{I}(T;Y)$ and upper bound on $\mathrm{I}(X;T)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.6.2, Eqs. 5.168–5.181, pp. 253–254):
>
> $$
> \begin{aligned}
> \mathrm{I}(T; Y) &= \mathbb{E}_{p(t,y)}\!\left[\ln\frac{p(y|t)}{p(y)}\right] \ge \mathbb{E}_{p(x,y)e_\theta(t|x)}[\ln d_\phi(y|t)] + \mathrm{H}(Y), \\
>   \mathrm{I}(X; T) &= \mathbb{E}_{p(x,t)}\!\left[\ln\frac{e_\theta(t|x)}{p(t)}\right] \le \mathbb{E}_{p(x)}\!\left[D_{\mathrm{KL}}(e_\theta(t|x) \parallel m(t))\right].
> \end{aligned}
> $$
>
> Dropping the constant $\mathrm{H}(Y)$ yields the differentiable **VIB upper bound** optimized by reparameterized SGD:
>
> $$
> \mathcal{L}_{\mathrm{VIB}}(\theta, \phi) = -\mathbb{E}_{p_{\mathcal{D}}(x,y)e_\theta(t|x)}[\ln d_\phi(y|t)] + \beta \mathbb{E}_{p_{\mathcal{D}}(x)}\bigl[D_{\mathrm{KL}}(e_\theta(t|x) \parallel m(t))\bigr].
> $$
>
>
> **3. Conditional Entropy Bottleneck (CEB).** Observing that $\mathrm{I}(X; T)$ penalizes predictive information $\mathrm{I}(T; Y)$ inside $X$, CEB instead minimizes the residual nuisance $\mathrm{I}(X; T \mid Y) = \mathrm{I}(X; T) - \mathrm{I}(T; Y)$ via a conditional backward encoder $b_\psi(t|y) \approx p(t|y)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.6.3, Eqs. 5.183–5.190, pp. 255–256):
>
> $$
> \min  \beta\Bigl(\mathbb{E}_{p(x)e_\theta(t|x)}[\ln e_\theta(t|x)] - \mathbb{E}_{p(x,y)e_\theta(t|x)}[\ln b_\psi(t|y)]\Bigr) - \mathbb{E}_{p(x,y)e_\theta(t|x)}[\ln d_\phi(y|t)].
> $$

## Algorithmic Implementation & Comparative Synthesis

For finite discrete spaces $(\mathcal{X}, \mathcal{Y}, \mathcal{T})$, Algorithm `alg:infotheory:ba_ib` iterates the three coupled self-consistent equations `(eq:infotheory:ib_ba)` (a generalized Blahut–Arimoto alternating projection) to a local minimum of $\mathcal{L}_{\mathrm{IB}}$ for a fixed compression weight $\beta > 0$. By sweeping $\beta$ from $\infty$ (maximal compression $T \perp\!\!\!\perp X$, $\mathrm{I}(X;T)=\mathrm{I}(T;Y)=0$) down toward $0$ (maximal sufficiency), the solver traces the IB information curve in the $(\mathrm{I}(X;T), \mathrm{I}(T;Y))$ plane [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.6.1, p. 252).

> [!example] Algorithm: Blahut–Arimoto Iterative Information Bottleneck Solver
> **Input:** Joint table $p(x, y)$ on $\mathcal{X} \times \mathcal{Y}$, bottleneck cardinality $\lvert \mathcal{T} \rvert$, trade-off $\beta > 0$, tolerance $\varepsilon > 0$, max iterations $M$
> **Output:** Stochastic encoder $e(t|x)$, bottleneck marginal $p(t)$, decoder $p(y|t)$
> Initialize $e^{(0)}(t|x)$ randomly on the simplex $\Delta^{\lvert \mathcal{T} \rvert-1}$ for each $x \in \mathcal{X}$
> Compute marginals $p(x) \leftarrow \sum_{y} p(x, y)$ and $p(y|x) \leftarrow p(x, y)/p(x)$
> **for** $k \leftarrow 0, 1, \dots, M - 1$:
>
>   $p^{(k)}(t) \leftarrow \sum_{x \in \mathcal{X}} p(x)  e^{(k)}(t|x)$  *// Marginal update*
>   **for each** $t \in \mathcal{T}$ and $y \in \mathcal{Y}$:
>
>     $p^{(k)}(y|t) \leftarrow \frac{1}{p^{(k)}(t)} \sum_{x \in \mathcal{X}} p(x)  e^{(k)}(t|x)  p(y|x)$  *// Bayes decoder consistency*
>
>   **for each** $x \in \mathcal{X}$:
>
>     **for each** $t \in \mathcal{T}$:
>
>       $d_{x,t} \leftarrow \sum_{y \in \mathcal{Y}} p(y|x) \bigl(\ln p(y|x) - \ln \max(p^{(k)}(y|t), 10^{-15})\bigr)$
>       $\tilde{e}(t|x) \leftarrow p^{(k)}(t) \exp\!\left(-\frac{1}{\beta}  d_{x,t}\right)$  *// Log-sum-exp stabilized in impl.*
>
>     $Z(x) \leftarrow \sum_{t' \in \mathcal{T}} \tilde{e}(t'|x)$
>     $e^{(k+1)}(t|x) \leftarrow \tilde{e}(t|x) / Z(x)$ for all $t \in \mathcal{T}$
>
>   **if** $\max_{x,t}\lvert e^{(k+1)}(t|x) - e^{(k)}(t|x) \rvert < \varepsilon$:
>
>     **break**
>
>
> **return** $e^{(k+1)}(t|x), p^{(k+1)}(t), p^{(k+1)}(y|t)$

Each iteration of Algorithm `alg:infotheory:ba_ib` requires $O(\lvert \mathcal{X} \rvert\lvert \mathcal{T} \rvert\lvert \mathcal{Y} \rvert)$ floating-point operations and $O(\lvert \mathcal{X} \rvert\lvert \mathcal{T} \rvert + \lvert \mathcal{T} \rvert\lvert \mathcal{Y} \rvert)$ memory. When $\lvert \mathcal{X} \rvert$ is exponential or continuous, VIB replaces exact enumeration with minibatch stochastic gradient descent on Eq. `(eq:infotheory:vib_loss)` at $O(B \cdot d_T)$ cost per step (batch size $B$, bottleneck dimension $d_T$). Table `tab:infotheory:divergences` compares the major $f$-divergences, their Fenchel conjugates for neural critic estimation, and their support-mismatch geometry.

**Table:** Canonical $f$-divergences $D_f(p  \|  q)$, Fenchel conjugates $f^\dagger(t)$ for variational lower bounds (Eq. `(eq:infotheory:fenchel_fdiv)`), and structural properties [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 2.7.1 & Table 26.2, pp. 55–57, 899) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (App. C.5, p. 460).

| **Divergence** | **Canonical $\tilde{f}(u)$** | **Conjugate $f^\dagger(t)$ & Optimal $t^\star$** | **Sym.** | **Metric** | **Range** | **Support Behavior** |
| — | — | — | — | — | — | — |
| Forward KL $D_{\mathrm{KL}}(p \parallel q)$ | $u\ln u - u + 1$ | $e^t - 1$  $t^\star = \ln(p/q)$ | No | No | $[0,\infty]$ | Zero-avoiding |
| Reverse KL $D_{\mathrm{KL}}(q \parallel p)$ | $-\ln u + u - 1$ | $-\ln(1-t)$ ($t<1$)  $t^\star = 1 - q/p$ | No | No | $[0,\infty]$ | Zero-forcing |
| Jensen–Shannon $D_{\mathrm{JS}}(p \parallel q)$ | $\frac{1}{2}[u\ln u - (u+1)\ln\frac{u+1}{2}]$ | $-\frac{1}{2}\ln(2 - e^{2t})$  $t^\star = \frac{1}{2}\ln\frac{2p}{p+q}$ | Yes | $\sqrt{D_{\mathrm{JS}}}$ | $[0,\ln 2]$ | Smooth under disjoint supports |
| Squared Hellinger $D_H^2(p,q)$ | $\frac{1}{2}(\sqrt{u} - 1)^2$ | $\frac{t}{1 - 2t}$ ($t < \frac{1}{2}$)  $t^\star = \frac{1}{2}(1 - \sqrt{q/p})$ | Yes | $D_H$ | $[0,1]$ | Bounded gradient |
| Pearson $\chi^2(p,q)$ | $(u - 1)^2$ | $\frac{1}{4}t^2 + t$  $t^\star = 2(p/q - 1)$ | No | No | $[0,\infty]$ | Heavy tail sensitivity |
| Total Variation $D_{\mathrm{TV}}(p,q)$ | $\frac{1}{2}\lvert u - 1 \rvert$ | $t$ ($t \in [-\frac{1}{2}, \frac{1}{2}]$)  $t^\star = \frac{1}{2}\mathrm{sgn}(p-q)$ | Yes | Yes | $[0,1]$ | Also an IPM ($\parallel f\parallel _\infty \le \frac{1}{2}$) |
| Minka $\alpha$-div. $D_\alpha^M(p\parallel q)$ | $\frac{u^\alpha - \alpha u + \alpha - 1}{\alpha(\alpha - 1)}$ | Variational power-EP bound | No | No | Bounded if $\alpha\in(0,1)$ | $\alpha\to 1$: fwd;  $\alpha\to 0$: rev |

> [!info] Cross-Book Synthesis: Variational Bounds, Rate–Distortion & Implicit Models
> Synthesizing across the five monographs reveals that four seemingly distinct deep learning objectives are specializations of the same information-theoretic variational calculus:
> - **Maximum Likelihood & Cross-Entropy:** Minimizing $D_{\mathrm{KL}}(p_{\mathcal{D}} \parallel q_\theta)$ directly optimizes cross-entropy $H_{\mathrm{ce}}(p_{\mathcal{D}}, q_\theta)$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (p. 75) [[Source - Foundations|deep_learning_foundations_concepts]] (p. 53), prioritizing mode coverage at the risk of overfitting empirical Dirac spikes [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 6.2.5, p. 216).
> - **Rate–Distortion & $\beta$-VAE:** In unsupervised encoding $e(z|x)$ with decoder $d(x|z)$ and prior $m(z)$, distortion $D = -\mathbb{E}_{p(x)e(z|x)}[\ln d(x|z)]$ and rate $R = \mathbb{E}_{p(x)}[D_{\mathrm{KL}}(e(z|x) \parallel m(z))]$ sandwich mutual information via $h(X) - D \le \mathrm{I}(X; Z) \le R$; minimizing $D + \beta R$ recovers the ELBO at $\beta = 1$ and $\beta$-VAE for $\beta \ne 1$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.4.2, Eqs. 5.158–5.161, pp. 248–250).
> - **GANs, $f$-GANs & Density-Ratio Estimation:** Training a binary classifier on $P$ vs. $Q$ under log-loss recovers $D^\star(x) = \frac{p(x)}{p(x)+q(x)}$ and optimizes $2D_{\mathrm{JS}}(p \parallel q) - 2\ln 2$; swapping $(\tilde{f}, f^\dagger)$ in Eq. `(eq:infotheory:fenchel_fdiv)` yields $f$-GANs, while replacing density ratios with $1$-Lipschitz witness functions yields Wasserstein IPMs [[Source - Understanding Deep Learning|understandingdl_simonprince]] (App. C.5.2, p. 460) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 2.7.5 & 26.2.2–26.2.3, pp. 61–62, 897–899).
> - **Contrastive Representation Bounds:** Replacing normalized $d_\phi(y|t)$ with unnormalized energy $e^{f(x,y)}$ turns Barber–Agakov's lower bound into the Donsker–Varadhan / NWJ bound, and multi-sample normalization yields $\mathrm{I}_{\mathrm{NCE}}(X; Y) \le \ln K$, directly linking VIB to self-supervised contrastive learning [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.3.6, pp. 244–246).

## Visual Diagnostics & Geometric Structure

We combine quantitative plots, structural diagrams, and benchmark empirical diagnostics to expose the exact geometry of entropy, $f$-divergences, information projections, and bottleneck compression.

> [!example] Visual Diagram (LaTeX / PDF)
> *Analytical behavior of entropy and canonical convex generators. **(a)** Concave binary entropy peaking at the uniform distribution $p=0.5$ ($1\text{ bit} = \ln 2\text{ nats}$). **(b)** Canonical generators shifted so $\tilde{f}(1) = \tilde{f}'(1) = 0$: forward KL penalizes $u = p/q \gg 1$ ($q \to 0$ on support of $p$), whereas reverse KL diverges steeply as $u \to 0^+$ ($q > 0$ outside support of $p$); Jensen–Shannon and squared Hellinger stay bounded as $u \to 0^+$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:infotheory:pgf_entropy_fdiv`(b) makes the algebraic origin of KL asymmetry immediately transparent: because $\tilde{f}_{\mathrm{KL}}(u)$ grows superlinearly ($u\ln u$) as $u = p/q \to \infty$, minimizing forward KL forces $q > 0$ everywhere $p > 0$. Conversely, $\tilde{f}_{\mathrm{rKL}}(u) \sim -\ln u$ explodes as $u \to 0^+$ ($q > 0$ where $p \approx 0$), forcing $q$ to collapse onto high-density regions of $p$. Jensen–Shannon and squared Hellinger remain finite at $u = 0$ ($\tilde{f}_{\mathrm{JS}}(0) = \frac{1}{2}\ln 2$, $\tilde{f}_{H^2}(0) = \frac{1}{2}$), explaining their stability under disjoint support.

> [!example] Visual Diagram (LaTeX / PDF)
> *Graphical and information-diagram geometry of the Information Bottleneck. **(a)** Because $T$ depends on $Y$ only through $X$ ($Y \leftrightarrow X \to T$), DPI limits predictive information to $\mathrm{I}(T;Y) \le \mathrm{I}(X;Y)$, while decoder $d_\phi(\hat{y}|t)$ forms the inference chain $X \to T \to \hat{Y}$. **(b)** In the signed information diagram [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 5.6 & 5.12, pp. 240, 253), optimal compression squeezes residual nuisance $\mathrm{I}(X;T|Y) \to 0$ inside $\mathrm{H}(T)$ while retaining $\mathrm{I}(T;Y) \approx \mathrm{I}(X;Y)$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Information-geometric and metric comparison of divergences. **(a)** Minimizing forward KL projects along mixture ($m$-) geodesics (moment matching), whereas reverse KL projects along exponential ($e$-) geodesics (I-projection) with local Riemannian curvature governed by the Fisher Information Matrix $F(\theta)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.1.9–5.1.10, pp. 233–234). **(b)** Pinsker's bound $D_{\mathrm{KL}}(p \parallel q) \ge 2 D_{\mathrm{TV}}(p,q)^2$ is exact to second order near $p=q$, as shown against the exact symmetric Bernoulli divergence $\tau\ln\frac{1+\tau}{1-\tau} = 2\tau^2 + O(\tau^4)$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Complementing the theoretical geometry, Figure `fig:infotheory:vib_mnist` illustrates the empirical behavior of the Variational Information Bottleneck (Eq. `(eq:infotheory:vib_loss)`) on MNIST digits in a 2D latent space: while a deterministic MLP maps instances to sharp trajectories that encode instance-specific nuisance, VIB learns compact Gaussian posteriors $e_\theta(t|x) = \mathcal{N}(\boldsymbol{\mu}_\theta(x), \mathrm{diag}(\boldsymbol{\sigma}^2_\theta(x)))$ whose covariances expand across intra-class variations, discarding irrelevant pixel detail while preserving inter-class separability.

> [!note] Source Figure
> *Figure:* 2D MNIST embeddings from a classifier bottleneck: **(a)** deterministic encoder creates sharp instance-memorizing rays; **(b)** VIB encoder means $\boldsymbol{\mu}_\theta(x)$ separate digit classes cleanly; **(c)** VIB posterior covariance ellipses expand within each class cluster to discard label-irrelevant input bits under the $\beta D_{\mathrm{KL}}(e_\theta(t|x) \parallel m(t))$ rate penalty. Reproduced from [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 5, Fig. 5.13, p. 255). (`probabilisticml_kevinmurphy_advancedtopics_p289_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Disjoint-Support KL Blowup ($+\infty$) & Singular Manifolds
> If there exists a measurable set $S$ with $p(S) > 0$ and $q(S) = 0$, then $D_{\mathrm{KL}}(p \parallel q) = +\infty$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.1.2.1, p. 222) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (App. C.5.1, p. 460). This failure occurs routinely in high-dimensional generative modeling when real data $p^*$ and a neural generator $q_\theta$ concentrate on low-dimensional manifolds with measure-zero intersection: both $D_{\mathrm{KL}}(p^* \parallel q_\theta)$ and $D_{\mathrm{KL}}(q_\theta \parallel p^*)$ diverge to $+\infty$, while $D_{\mathrm{JS}}(p^* \parallel q_\theta)$ saturates flatly at $\ln 2$ with zero gradient everywhere.
> **Remedy:** Add controlled Gaussian jitter ($p_\sigma * p^*$, $p_\sigma * q_\theta$) to restore overlapping support, smooth empirical targets via label smoothing or data augmentation [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 6.2.5, p. 216), or replace density-ratio $f$-divergences with $1$-Lipschitz Integral Probability Metrics such as Wasserstein-$1$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 26.2.4, p. 900).

> [!warning] Differential Entropy Negativity, Divergence & Unit Dependence
> Unlike discrete entropy $\mathrm{H}(X) \in [0, \ln K]$, differential entropy $h(X)$ can be negative (for $X \sim \mathcal{N}(\mu, \sigma^2)$, $h(X) < 0$ whenever $\sigma^2 < (2\pi e)^{-1}$ [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 2.5.4, p. 51)), diverges as $-\ln\Delta \to +\infty$ under infinite quantization precision (Eq. `(eq:infotheory:quantized_entropy)`), and is *not* coordinate-invariant: under an invertible smooth map $Y = g(X)$ with Jacobian $\boldsymbol{J}_g(x)$,
>
> $$
> h(Y) = h(X) + \mathbb{E}_{p_X}\!\bigl[\ln\lvert \det \boldsymbol{J}_g(X) \rvert\bigr],
> $$
>
> so merely rescaling adult human heights from feet to centimeters adds $\log_2 30.48 \approx 4.93$ bits [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.2.2 & Fig. 5.5, pp. 236–237).
> **Remedy:** Never interpret raw $h(X)$ as an absolute information measure or minimize $h(T)$ in continuous networks without fixing a reference scale; always optimize relative quantities ($D_{\mathrm{KL}}(p \parallel q)$ or $\mathrm{I}(X;Y)$) where $\ln\lvert \det \boldsymbol{J}_g(X) \rvert$ cancels identically.

> [!warning] Deterministic Bottleneck Collapse & Log-Domain Numerical Instability
> Two severe implementation hazards arise in neural pipelines:
> - **Deterministic IB Singularity:** For a deterministic continuous map $T = g_\theta(X)$ with continuous $X$, $\mathrm{I}(X; T) = +\infty$ identically, rendering $\min_\theta \beta\mathrm{I}(X; T) - \mathrm{I}(T; Y)$ ill-posed unless stochastic encoder noise (as in VIB, $e_\theta(t|x) = \mathcal{N}(\boldsymbol{\mu}_\theta(x), \mathrm{diag}(\boldsymbol{\sigma}^2_\theta(x)))$) or fixed finite binning is enforced [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.6.2, p. 254).
> - **$0\ln 0$ & Softmax Underflow:** Evaluating $-p_k \ln(\mathrm{softmax}(\boldsymbol{z})_k)$ directly yields $0 \cdot (-\infty) = \mathrm{NaN}$ or catastrophic cancellation when probabilities underflow IEEE-754 float32 ($\approx 10^{-38}$).
>
> **Remedy:** Compute cross-entropy and KL terms strictly in log-space via log-sum-exp ($\ln\mathrm{softmax}(\boldsymbol{z})_k = z_k - \text{LogSumExp}(\boldsymbol{z})$), mask zero-probability support entries ($p_k = 0 \mapsto 0$) before multiplying by $\ln q_k$, and use non-parametric $k$-NN (KSG) or variational InfoNCE estimators (noting $\mathrm{I}_{\mathrm{NCE}} \le \ln K$ requires sufficiently large negative batch size $K$) rather than naive histogram binning for continuous MI [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 6.3.5, p. 220) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 5.3.6.4, p. 246).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (5):** [[Information Bottleneck]], [[KL Divergence]], [[Mutual Information & DPI]], [[Shannon Entropy]], [[f-Divergence]]
- **Prerequisites (Directed $\leftarrow$):** [[KL Divergence]], [[Mutual Information & DPI]], [[Shannon Entropy]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[DPO (Direct Preference Optimization)]], [[Information Bottleneck]], [[KL Divergence]], [[Maximum Likelihood & Fisher Info]], [[Mutual Information & DPI]], [[RLHF (KL-Regularized)]], [[Sinkhorn Algorithm]], [[TRPO]], [[Variational Inference & ELBO]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[DPO (Direct Preference Optimization)]], [[Information Bottleneck]], [[KL Divergence]], [[Maximum Likelihood & Fisher Info]], [[Mutual Information & DPI]], [[RLHF (KL-Regularized)]], [[Shannon Entropy]], [[TRPO]], [[Variational Autoencoder (VAE)]], [[Variational Inference & ELBO]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
