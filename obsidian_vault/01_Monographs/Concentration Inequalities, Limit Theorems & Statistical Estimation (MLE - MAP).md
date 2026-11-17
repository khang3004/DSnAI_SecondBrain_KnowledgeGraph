---
type: monograph
domain: "D01"
parent_topic: "[[PT 01 - Probability Theory, Information Theory & Optimal Transport]]"
sub_topic: "[[ST 01 - Probability Spaces, Gaussians & Statistical Estimation]]"
cheatsheet: "[[Cheatsheet - Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]"
part_of: 
  - "[[ST 01 - Probability Spaces, Gaussians & Statistical Estimation]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
concepts: 
  - "[[Concentration Inequalities]]"
  - "[[Law of Large Numbers & CLT]]"
  - "[[Maximum Likelihood & Fisher Info]]"
prerequisites: 
  - "[[KL Divergence]]"
  - "[[Law of Large Numbers & CLT]]"
enables: 
  - "[[Concentration Inequalities]]"
  - "[[Logistic Regression]]"
  - "[[Maximum Likelihood & Fisher Info]]"
  - "[[Multi-Armed Bandits & UCB]]"
  - "[[Natural Gradient & K-FAC]]"
  - "[[Rademacher Complexity & VC Dimension]]"
related: 
  - "[[Concentration Inequalities]]"
  - "[[KL Divergence]]"
  - "[[Law of Large Numbers & CLT]]"
  - "[[Logistic Regression]]"
  - "[[Maximum Likelihood & Fisher Info]]"
  - "[[Multi-Armed Bandits & UCB]]"
  - "[[Natural Gradient & K-FAC]]"
  - "[[Rademacher Complexity & VC Dimension]]"
sources: 
  - "[[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - monograph
  - domain-01
---

# Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE/MAP)

> [!abstract] Monograph Navigation
> - **Domain:** [[D01 - Mathematical Foundations for Machine Intelligence|Mathematical Foundations for Machine Intelligence]]
> - **Parent / Sub-Topic:** [[PT 01 - Probability Theory, Information Theory & Optimal Transport|Probability Theory, Information Theory & Optimal Transport]] → [[ST 01 - Probability Spaces, Gaussians & Statistical Estimation|Probability Spaces, Gaussians & Statistical Estimation]]
> - **Visual Cheatsheet:** [[Cheatsheet - Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]
> - **Description:** Synthesizes Markov/Chebyshev/Hoeffding/Chernoff concentration bounds, LLN and CLT asymptotics, Fisher information, Cramér-Rao lower bound, and MLE vs. MAP vs. Bayesian decision theory.

A central quantitative challenge in machine learning and statistical inference is bridging finite-sample empirical behavior with asymptotic population guarantees. Given $N$ observations drawn from an unknown data-generating process, how tightly does an empirical average concentrate around its expectation? When does a normalized estimator converge in law to a Gaussian, and where does that Gaussian approximation break down? Under what regularity conditions can any unbiased estimator beat the Cram\'er–Rao information floor, and how do prior assumptions transform point estimation from likelihood maximization into regularized risk minimization and full Bayesian decision theory? This monograph synthesizes non-asymptotic concentration bounds, classical limit theorems, likelihood curvature geometry, and frequentist versus Bayesian estimation and testing across four foundational texts.

## Problem Formulation & Unified Notation

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space supporting an i.i.d. sample $\mathcal{D} = \{x_1, \dots, x_N\}$ drawn from an unknown distribution $P^*$, or a supervised dataset $\mathcal{D} = \{(\boldsymbol{x}_n, y_n)\}_{n=1}^N \overset{\mathrm{i.i.d.}}{\sim} P^*(\boldsymbol{X}, Y)$, with empirical distribution $\hat{p}_{\mathcal{D}}(x) = \frac{1}{N}\sum_{n=1}^N \delta(x - x_n)$. In concentration and limit analysis, we track the sample mean $\bar{X}_N = \frac{1}{N}\sum_{n=1}^N X_n$ relative to the population mean $\mu = \mathbb{E}[X]$ and variance $\sigma^2 = \mathrm{Var}[X]$ [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 353–355).

In parametric statistical estimation, we posit a model family $\mathcal{M} = \{p(\cdot \mid \theta) : \theta \in \Theta \subseteq \mathbb{R}^D\}$ and assume in the well-specified regime that $P^* = p(\cdot \mid \theta^*)$. An estimator is a measurable map $\hat{\theta}_N = \delta(\mathcal{D}) \in \Theta$ evaluated through three complementary lenses:
- **Finite-sample concentration**: bounding sample-mean deviations $\mathbb{P}\bigl(\lvert \bar{X}_N - \mu \rvert \ge \varepsilon\bigr)$ [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 339–350) alongside uniform empirical risk deviations $\sup_{h \in \mathcal{H}} \lvert R(h) - R(h, \mathcal{D}) \rvert$ as explicit functions of $N$, $\varepsilon$, and $\lvert \mathcal{H} \rvert$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, pp. 196–197).
- **Asymptotic law & efficiency**: establishing estimator consistency ($\hat{\theta}_N \xrightarrow{p} \theta^*$ or $\xrightarrow{a.s.} \theta^*$), asymptotic normality ($\sqrt{N}(\hat{\theta}_N - \theta^*) \xrightarrow{d} \mathcal{N}(\boldsymbol{0}, \boldsymbol{F}(\theta^*)^{-1})$), and attainment of the Cram\'er–Rao covariance lower bound [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, pp. 153–160) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, pp. 75–79).
- **Decision-theoretic risk & posterior updating**: placing a prior $p(\theta)$ over $\Theta$ to form the posterior $p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) p(\theta)$, minimizing posterior expected loss $\rho(a \mid \mathcal{D}) = \mathbb{E}_{\theta \mid \mathcal{D}}[\ell(\theta, a)]$ or Bayes risk $R_{\pi_0}(\delta) = \mathbb{E}_{\theta \sim \pi_0}[R(\theta, \delta)]$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, pp. 169–191) [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 8, pp. 505, 532).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> The four source texts adopt distinct conventions reflecting probability engineering (Chan), probabilistic machine learning (Murphy PML1/PML2), and algorithmic optimization (Tiep Vu). Table `tab:asymp:notation` establishes a unified correspondence used throughout this monograph.
>
>
>
>
>
>
>
>
>
> | **Concept** | **Chan** | **Murphy (PML1/PML2)** | **Tiep Vu** | **Unified Notation** |
> | — | — | — | — | — |
> | Likelihood | $L(\theta \mid \boldsymbol{x}) = f_{\boldsymbol{X}}(\boldsymbol{x};\theta)$ (MLE) vs. $f_{\boldsymbol{X}\mid\Theta}(\boldsymbol{x}\mid\theta)$ (MAP) | $p(\mathcal{D} \mid \theta)$ with true generating value $\theta^*$ | $p(\boldsymbol{x}_1,\dots,\boldsymbol{x}_N \mid \theta)$ with parameter $\theta$ or $\lambda$ | $L(\theta) = p(\mathcal{D} \mid \theta)$, log-likelihood $\ell(\theta) = \log p(\mathcal{D}\mid\theta)$ |
> | Fisher info. matrix | Omitted (uses CRLB-free bias/MSE in Ch. 8) | $\boldsymbol{F}(\theta)$ per sample; $N\boldsymbol{F}(\theta)$ for $N$ i.i.d. points | Omitted (uses direct calculus & KKT conditions) | $\boldsymbol{F}(\theta) = \mathrm{Cov}_{p_\theta}[\nabla \log p(x\mid\theta)]$; total FIM $\boldsymbol{I}_N(\theta) = N\boldsymbol{F}(\theta)$ |
> | Variance MLE divisor | Contrasts $\frac{1}{N}$ (MLE, biased) vs. $\frac{1}{N-1}$ (unbiased $S^2$) | $\frac{1}{N}$ in MLE; notes bias vanishes as $N\to\infty$ | $\frac{1}{N}$ directly from $\nabla_{\sigma^2}\ell = 0$ | $\hat{\sigma}_{\mathrm{MLE}}^2 = \frac{1}{N}\sum_{n}(x_n - \bar{x})^2$, $S^2 = \frac{N}{N-1}\hat{\sigma}_{\mathrm{MLE}}^2$ |
> | Transform sign (CF) | $\Phi_X(j\omega) = \mathbb{E}[e^{-j\omega X}]$ (Fourier engineering sign) | Uses MGF/KL & exponential family $A(\eta)$ | Uses standard PDFs without transforms | MGF $M_X(s) = \mathbb{E}[e^{sX}]$; CF $\varphi_X(t) = \mathbb{E}[e^{itX}]$ |
> | Conjugate hyperparameters | $\mathrm{Beta}(\alpha,\beta)$, Gaussian $\mathcal{N}(\mu_0, \sigma_0^2)$ | Prior $\breve{m}, \breve{\tau}^2, \breve{\lambda}$ vs. posterior $\hat{m}, \hat{\tau}^2, \hat{\lambda}$ | $\mathrm{Beta}(\alpha,\beta)$, prior log-penalty $\lambda R(\theta)$ | Prior $\mathcal{N}(\mu_0, \sigma_0^2)$ or $\mathrm{Beta}(\alpha,\beta)$; posterior $(\alpha_N, \beta_N)$ |
> | Optimal point rule | Derives MMSE $\mathbb{E}[\Theta\mid\boldsymbol{X}]$ via orthogonality | Unifies $0\text{-}1 \to$ MAP, $\ell_2 \to$ mean, $\ell_1 \to$ median | Frames MAP as regularized MLE $\ell(\theta) + \log p(\theta)$ | $\hat{\theta}_{\mathrm{MLE}}$, $\hat{\theta}_{\mathrm{MAP}}$, and Bayes minimizer $\hat{\theta}_{\mathrm{MMSE}}$ |

## Core Mathematical Derivations & Theorems

### Non-Asymptotic Concentration Inequalities

Classical concentration inequalities trade progressively stronger moment assumptions for exponentially sharper bounds on deviation probabilities.

> [!important] Markov and Chebyshev Inequalities
> Let $X$ be a scalar random variable and $\varepsilon > 0$.
> - **Markov's inequality** [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 339–340): If $X \ge 0$ a.s. and $\mathbb{E}[X] < \infty$, then
>
> $$
> \mathbb{P}[X \ge \varepsilon] \le \frac{\mathbb{E}[X]}{\varepsilon}.
> $$
>
>   *Proof.* Since $\varepsilon \mathbf{1}_{\{X \ge \varepsilon\}} \le X$ pointwise whenever $X \ge 0$, taking expectations yields $\varepsilon \mathbb{P}[X \ge \varepsilon] \le \mathbb{E}[X]$.
> - **Chebyshev's inequality** [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 341–343): If $\mu = \mathbb{E}[X]$ and $\sigma^2 = \mathrm{Var}[X] < \infty$, applying `(eq:asymp:markov)` to the nonnegative variable $Y = (X - \mu)^2$ at threshold $\varepsilon^2$ gives
>
> $$
> \mathbb{P}\bigl[\lvert X - \mu \rvert \ge \varepsilon\bigr] = \mathbb{P}\bigl[(X-\mu)^2 \ge \varepsilon^2\bigr] \le \frac{\mathrm{Var}[X]}{\varepsilon^2}.
> $$
>
>   For $\bar{X}_N = \frac{1}{N}\sum_{n=1}^N X_n$ with uncorrelated i.i.d. summands of variance $\sigma^2$, $\mathrm{Var}[\bar{X}_N] = \sigma^2/N$, hence $\mathbb{P}[\lvert \bar{X}_N - \mu \rvert \ge \varepsilon] \le \frac{\sigma^2}{N\varepsilon^2}$.

Although Chebyshev establishes $O(N^{-1}\varepsilon^{-2})$ variance concentration, polynomial decay fails to control simultaneous events across exponentially large hypothesis spaces. Exponential concentration requires the moment-generating function (MGF) $M_X(s) \triangleq \mathbb{E}[e^{sX}]$ and cumulant-generating function (CGF) $\psi_X(s) \triangleq \log M_X(s)$, defined on the domain $\{s \in \mathbb{R} : M_X(s) < \infty\}$ [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 324–327).

> [!important] Chernoff Bound, Fenchel–Legendre Rate, and Sub-Gaussian Tails
> Assume $M_X(s) < \infty$ on $(0, s_{\max})$. For any $s > 0$, monotonicity of $x \mapsto e^{sx}$ and Markov's inequality `(eq:asymp:markov)` give [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 343–346):
>
> $$
> \mathbb{P}[X \ge \varepsilon] = \mathbb{P}\bigl[e^{sX} \ge e^{s\varepsilon}\bigr] \le \inf_{s > 0} e^{-s\varepsilon} M_X(s) = \exp\bigl(-\psi_X^*(\varepsilon)\bigr), \qquad \psi_X^*(\varepsilon) \triangleq \sup_{s > 0}\bigl(s\varepsilon - \psi_X(s)\bigr),
> $$
>
> where $\psi_X^*(\varepsilon)$ is the convex conjugate (Fenchel–Legendre dual) of the CGF $\psi_X$. For i.i.d. $X_1,\dots,X_N$, independence factorizes the MGF as $M_{\sum X_n}(s) = [M_X(s)]^N$, yielding the large-deviation rate $\mathbb{P}[\bar{X}_N \ge \varepsilon] \le \exp\bigl(-N \psi_X^*(\varepsilon)\bigr)$.
> - **Gaussian instantiation**: If $X_n \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)$, then $M_{X-\mu}(s) = \exp(\tfrac{1}{2}\sigma^2 s^2)$. Setting $\frac{d}{ds}[s\varepsilon - \frac{1}{2}\sigma^2 s^2] = 0$ gives $s^* = \varepsilon/\sigma^2$, hence $\psi_{X-\mu}^*(\varepsilon) = \frac{\varepsilon^2}{2\sigma^2}$ and the two-sided bound [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, p. 345):
>
> $$
> \mathbb{P}\bigl[\lvert \bar{X}_N - \mu \rvert \ge \varepsilon\bigr] \le 2\exp\!\left(-\frac{N\varepsilon^2}{2\sigma^2}\right).
> $$
> - **Sub-Gaussian class**: A random variable $X$ with mean $\mu$ is *$\sigma$-sub-Gaussian* if its centered MGF is dominated by a Gaussian MGF of variance proxy $\sigma^2$, i.e., $\psi_{X-\mu}(s) \le \frac{1}{2}\sigma^2 s^2$ for all $s \in \mathbb{R}$. Independent summation preserves sub-Gaussianity with additive variance proxy: if $X_n - \mu_n$ are independent $\sigma_n$-sub-Gaussian, then $\sum_{n=1}^N (X_n - \mu_n)$ is $\sqrt{\sum_{n=1}^N \sigma_n^2}$-sub-Gaussian, immediately inheriting `(eq:asymp:chernoff-gauss)` with $\sigma^2 = \frac{1}{N}\sum_{n=1}^N \sigma_n^2$.

When distributions are bounded rather than Gaussian, Hoeffding's lemma proves that every bounded random variable is automatically sub-Gaussian with variance proxy determined solely by its support width [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 348–350).

> [!important] Hoeffding's Lemma, Hoeffding's Inequality, and Uniform PAC Convergence
> - **Hoeffding's Lemma**: Suppose $X \in [a, b]$ a.s. with $\mathbb{E}[X] = \mu$. Define the tilted density $dQ_s(x) = \frac{e^{sx}}{\mathbb{E}[e^{sX}]} dP(x)$ and centered CGF $\psi(s) = \log \mathbb{E}[e^{s(X-\mu)}]$. Differentiating twice yields $\psi(0) = 0$, $\psi'(0) = 0$, and
>
> $$
> \psi"(s) = \frac{\mathbb{E}[X^2 e^{sX}]}{\mathbb{E}[e^{sX}]} - \left(\frac{\mathbb{E}[X e^{sX}]}{\mathbb{E}[e^{sX}]}\right)^2 = \mathrm{Var}_{Q_s}[X] \le \frac{(b-a)^2}{4},
> $$
>
>   where the variance bound holds for any distribution on $[a, b]$ since $\mathrm{Var}_{Q_s}[X] \le \mathbb{E}_{Q_s}[(X - \frac{a+b}{2})^2] \le (\frac{b-a}{2})^2$. By Taylor's theorem, $\psi(s) = \frac{1}{2}\psi"(\tilde{s}) s^2 \le \frac{1}{8}s^2(b-a)^2$ for all $s \in \mathbb{R}$, proving that $X$ is $\frac{b-a}{2}$-sub-Gaussian [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, p. 348).
> - **Hoeffding's Inequality**: For independent variables $X_n \in [a_n, b_n]$ with $\bar{X}_N = \frac{1}{N}\sum_{n=1}^N X_n$ and $\mu = \mathbb{E}[\bar{X}_N]$, Chernoff optimization with $s^* = \frac{4N\varepsilon}{\sum_{n=1}^N (b_n - a_n)^2}$ yields [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 349–350):
>
> $$
> \mathbb{P}\bigl[\lvert \bar{X}_N - \mu \rvert \ge \varepsilon\bigr] \le 2\exp\!\left(-\frac{2N^2\varepsilon^2}{\sum_{n=1}^N (b_n - a_n)^2}\right) \stackrel{X_n \in [0,1]}{=} 2\exp\bigl(-2N\varepsilon^2\bigr).
> $$
> - **Finite-Hypothesis Generalization Bound**: Let $\mathcal{H}$ be a finite hypothesis class with bounded loss $\ell(y, h(\boldsymbol{x})) \in [0, 1]$, population risk $R(h) = \mathbb{E}[\ell(Y, h(\boldsymbol{X}))]$, and empirical risk $R(h, \mathcal{D}) = \frac{1}{N}\sum_{n=1}^N \ell(y_n, h(\boldsymbol{x}_n))$. Combining the union bound [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, p. 333) with Hoeffding's inequality `(eq:asymp:hoeffding)` gives [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, Thm. 5.4.1, p. 197):
>
> $$
> \mathbb{P}\Bigl(\max_{h \in \mathcal{H}} \lvert R(h) - R(h,\mathcal{D}) \rvert > \varepsilon\Bigr) \le \sum_{h \in \mathcal{H}} \mathbb{P}\bigl(\lvert R(h) - R(h,\mathcal{D}) \rvert > \varepsilon\bigr) \le 2\lvert \mathcal{H} \rvert\exp\bigl(-2N\varepsilon^2\bigr).
> $$
>
>   Setting the RHS to $\delta$ guarantees with probability at least $1-\delta$ that the ERM excess risk satisfies $R(\hat{h}_{\mathrm{ERM}}) - \min_{h\in\mathcal{H}} R(h) \le \sqrt{\frac{2\log(2\lvert \mathcal{H} \rvert/\delta)}{N}}$.

> [!tip] Why Exponential Concentration Beats Polynomial Bounds in High Dimensions
> Chebyshev's bound only extracts information from the second moment $\mathrm{Var}[X]$, allowing heavy-tailed distributions with identical variance to place $O(\varepsilon^{-2})$ probability mass in the tail. By contrast, optimizing $e^{-s\varepsilon}\mathbb{E}[e^{sX}]$ simultaneously engages *every* higher moment $\mathbb{E}[X^k]$. Crucially, under the union bound over $M = \lvert \mathcal{H} \rvert$ hypotheses (or Bonferroni tests), polynomial $O(M/(N\varepsilon^2))$ bounds require $N = \Omega(M)$ samples, whereas exponential concentration $2M e^{-2N\varepsilon^2}$ requires only $N = O(\log M)$ samples.

### Asymptotic Convergence & Limit Theorems

Concentration bounds give explicit $N$-dependent probabilities; passing to $N \to \infty$ yields three distinct modes of stochastic convergence [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 357–372):
- **Convergence in probability** ($Y_N \xrightarrow{p} Y$): $\lim_{N\to\infty} \mathbb{P}\bigl[\lvert Y_N - Y \rvert > \varepsilon\bigr] = 0$ for every $\varepsilon > 0$.
- **Almost sure convergence** ($Y_N \xrightarrow{a.s.} Y$): $\mathbb{P}\bigl[\lim_{N\to\infty} Y_N = Y\bigr] = 1$, which implies $Y_N \xrightarrow{p} Y$ [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 362–363), whereas the converse fails when rare spikes drift in probability without vanishing almost surely.
- **Convergence in distribution** ($Y_N \xrightarrow{d} Y$): $\lim_{N\to\infty} F_{Y_N}(y) = F_Y(y)$ at every continuity point $y$ of $F_Y$; $Y_N \xrightarrow{p} Y \implies Y_N \xrightarrow{d} Y$, with the converse holding only when the limit $Y$ is a constant.

> [!important] Weak and Strong Laws of Large Numbers (WLLN & SLLN)
> Let $X_1, X_2, \dots \overset{\mathrm{i.i.d.}}{\sim} P$ with $\mathbb{E}[X_n] = \mu$ and sample mean $\bar{X}_N = \frac{1}{N}\sum_{n=1}^N X_n$.
> - **Weak Law of Large Numbers (WLLN)** [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 354–359): By Theorem 6.16, if $\mathrm{Var}[X_n] = \sigma^2 < \infty$, Chebyshev's inequality `(eq:asymp:chebyshev)` immediately implies $\mathbb{P}[\lvert \bar{X}_N - \mu \rvert \ge \varepsilon] \le \frac{\sigma^2}{N\varepsilon^2} \to 0$ as $N \to \infty$, so $\bar{X}_N \xrightarrow{p} \mu$. (Khinchin's theorem extends WLLN to any i.i.d. sequence with $\mathbb{E}[\lvert X_n \rvert] < \infty$; under sub-Gaussianity, Chernoff/Hoeffding strengthens the rate to $O(e^{-cN\varepsilon^2})$.)
> - **Strong Law of Large Numbers (SLLN)**: Assuming finite fourth moment $\mathbb{E}[(X_n - \mu)^4] = \mu_4 < \infty$ (Theorems 6.17–6.18) [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 361–366), let $Y_n = X_n - \mu$. Expanding $\mathbb{E}[(\sum_{n=1}^N Y_n)^4] = N\mu_4 + 3N(N-1)\sigma^4$ (since cross-terms containing an odd power vanish by zero mean and independence), Markov's inequality on $\bar{Y}_N^4$ gives
>
> $$
> \sum_{N=1}^\infty \mathbb{P}\bigl[\lvert \bar{X}_N - \mu \rvert \ge \varepsilon\bigr] \le \sum_{N=1}^\infty \frac{N\mu_4 + 3N(N-1)\sigma^4}{N^4 \varepsilon^4} < \infty.
> $$
>
>   By the first Borel–Cantelli lemma, $\mathbb{P}[\lvert \bar{X}_N - \mu \rvert \ge \varepsilon \text{ infinitely often}] = 0$ for every rational $\varepsilon > 0$, hence $\bar{X}_N \xrightarrow{a.s.} \mu$. (Kolmogorov's SLLN removes higher moments, requiring only $\mathbb{E}[\lvert X_n \rvert] < \infty$.)

While the laws of large numbers pin $\bar{X}_N$ to a point mass $\mu$, rescaling fluctuations by $\sqrt{N}$ uncovers universal Gaussian geometry.

> [!important] Central Limit Theorem & Berry–Esseen Quantitative Rate
> Let $X_1, X_2, \dots \overset{\mathrm{i.i.d.}}{\sim} P$ with mean $\mu$ and finite variance $\sigma^2 \in (0, \infty)$, and form the standardized average $Z_N = \frac{\sqrt{N}(\bar{X}_N - \mu)}{\sigma} = \frac{1}{\sqrt{N}}\sum_{n=1}^N \tilde{X}_n$, where $\tilde{X}_n = \frac{X_n - \mu}{\sigma}$ has $\mathbb{E}[\tilde{X}_n] = 0$, $\mathbb{E}[\tilde{X}_n^2] = 1$.
> - **Lindeberg–L\'evy CLT** [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, Thm. 6.19, pp. 372–373): The characteristic function $\varphi_{Z_N}(t) = \mathbb{E}[e^{it Z_N}]$ factorizes by independence and admits a second-order Taylor expansion around $t/\sqrt{N} = 0$:
>
> $$
> \varphi_{Z_N}(t) = \left[\varphi_{\tilde{X}}\!\left(\frac{t}{\sqrt{N}}\right)\right]^N = \left[1 + \frac{it}{\sqrt{N}}\mathbb{E}[\tilde{X}] - \frac{t^2}{2N}\mathbb{E}[\tilde{X}^2] + o\!\left(\frac{t^2}{N}\right)\right]^N \xrightarrow{N \to \infty} e^{-t^2/2}.
> $$
>
>   Because $e^{-t^2/2}$ is the characteristic function of $\mathcal{N}(0,1)$, L\'evy's continuity theorem implies $Z_N \xrightarrow{d} \mathcal{N}(0, 1)$, equivalently $\sqrt{N}(\bar{X}_N - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$.
> - **Berry–Esseen Error Bound** [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, p. 375): If the third absolute central moment $\rho = \mathbb{E}[\lvert X_1 - \mu \rvert^3] < \infty$, then uniform CDF approximation accuracy decays at $O(N^{-1/2})$:
>
> $$
> \sup_{z \in \mathbb{R}} \lvert F_{Z_N}(z) - \Phi(z) \rvert \le \frac{C \rho}{\sigma^3 \sqrt{N}}, \qquad (C < 0.4748).
> $$

> [!tip] Local Bulk Gaussianity vs. Tail Breakdown
> The Berry–Esseen bound `(eq:asymp:berry-esseen)` measures *additive* error in CDF probabilities, which is informative only near the bulk $\bar{X}_N = \mu + O(N^{-1/2})$ where $\Phi(z)$ is of order $1$. In the large-deviation tail $z = \Theta(\sqrt{N})$, $\Phi(-z)$ decays as $\exp(-z^2/2)$ and is vastly smaller than the $O(N^{-1/2})$ residual; skewed summands such as exponential lifetimes exhibit large relative errors in their upper tail even at $N = 1000$, as shown in Figure `fig:asymp:clt-tails`.

> [!note] Source Figure
> *Figure:* Failure of the Central Limit Theorem Gaussian approximation in the upper tail for normalized Erlang sums of i.i.d. $\mathrm{Exp}(\lambda=1)$ variables across $N \in \{1, 10, 100, 1000\}$. While density near $z=0$ matches $\mathcal{N}(0,1)$ closely once $N \ge 100$, severe skewness persists for $z \ge 2.5$, motivating exact exponential Chernoff bounds. Reproduced from [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, Fig. 6.22, p. 380). (`introduction_probability_data_science_p396_fig1.png`)

### Likelihood, Information Geometry, Efficiency & Asymptotics of MLE

We now pass from sample means to general parametric models $p(x \mid \theta)$ (or conditional $p(y \mid \boldsymbol{x}, \theta)$). Maximizing likelihood admits an exact information-theoretic and risk-minimization equivalence.

> [!important] MLE as Empirical KL Minimization, ERM, and Functional Invariance
> Given i.i.d. data $\mathcal{D} = \{x_1, \dots, x_N\}$ with empirical distribution $\hat{p}_{\mathcal{D}}$, the maximum likelihood estimator minimizes the negative log-likelihood $\mathrm{NLL}(\theta) = -\sum_{n=1}^N \log p(x_n \mid \theta)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, pp. 109–117); [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 4, pp. 53–58):
> - **KL-Divergence Equivalence**: Adding and subtracting the empirical entropy $H(\hat{p}_{\mathcal{D}})$ yields
>
> $$
> \frac{1}{N}\mathrm{NLL}(\theta) = -\mathbb{E}_{\hat{p}_{\mathcal{D}}}[\log p(X \mid \theta)] = H_{\mathrm{ce}}(\hat{p}_{\mathcal{D}}, p_\theta) = H(\hat{p}_{\mathcal{D}}) + D_{\mathrm{KL}}(\hat{p}_{\mathcal{D}} \parallel p_\theta).
> $$
>
>   Since $H(\hat{p}_{\mathcal{D}})$ is independent of $\theta$, $\hat{\theta}_{\mathrm{MLE}} = \arg\min_\theta D_{\mathrm{KL}}(\hat{p}_{\mathcal{D}} \parallel p_\theta)$. Moreover, by the WLLN, $-\frac{1}{N}\mathrm{NLL}(\theta) \xrightarrow{p} \mathbb{E}_{p^*}[\log p(X \mid \theta)] = -H(p^*) - D_{\mathrm{KL}}(p^* \parallel p_\theta)$, which is uniquely maximized at $\theta = \theta^*$ whenever the model is identifiable [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 8, pp. 499–500).
> - **ERM and Canonical Closed-Form Cases**: Under conditional log-loss $\ell(y, \theta; \boldsymbol{x}) = -\log p(y \mid \boldsymbol{x}, \theta)$, MLE coincides with empirical risk minimization $R(\theta, \mathcal{D}) = \frac{1}{N}\sum_{n=1}^N \ell(y_n, \theta; \boldsymbol{x}_n)$. Classical stationarity $\nabla_\theta \ell(\theta) = \boldsymbol{0}$ gives:
> - *Categorical* ($x_n \in \{1,\dots,K\}$, Lagrange constraint $\sum_k \theta_k = 1$): $\hat{\theta}_k = N_k / N$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 4, pp. 54–55);
> - *Multivariate Gaussian* ($\boldsymbol{x}_n \in \mathbb{R}^D$): $\hat{\boldsymbol{\mu}}_{\mathrm{MLE}} = \bar{\boldsymbol{x}} = \frac{1}{N}\sum_{n=1}^N \boldsymbol{x}_n$ and $\hat{\boldsymbol{\Sigma}}_{\mathrm{MLE}} = \frac{1}{N}\sum_{n=1}^N (\boldsymbol{x}_n - \bar{\boldsymbol{x}})(\boldsymbol{x}_n - \bar{\boldsymbol{x}})^\top$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, pp. 113–114); [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 4, pp. 57–58).
> - **Invariance Principle** [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 8, Thm. 8.1, pp. 500–502): For any (even non-invertible) mapping $\phi = h(\theta)$, defining the profile likelihood $\tilde{L}(\phi) \triangleq \sup_{\theta : h(\theta) = \phi} L(\theta)$, the supremum over $\phi$ equals $\sup_\theta L(\theta)$, so
>
> $$
> \widehat{h(\theta)}_{\mathrm{MLE}} = h\bigl(\hat{\theta}_{\mathrm{MLE}}\bigr).
> $$

To quantify the uncertainty and optimality of MLE, we analyze the local curvature of the log-likelihood via the Fisher information matrix (FIM) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, pp. 154–160); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, pp. 75–79).

> [!important] Fisher Information Identities, Exponential Families, and Cram\'er–Rao Bound
> Let $\boldsymbol{s}(\theta; x) \triangleq \nabla_\theta \log p(x \mid \theta) \in \mathbb{R}^D$ be the per-sample score vector, and assume sufficient smoothness to interchange differentiation and integration over $x$.
> - **Zero Expected Score & Bartlett Curvature Identity**: Differentiating $\int p(x\mid\theta) dx = 1$ gives
>
> $$
> \mathbb{E}_{p_\theta}[\boldsymbol{s}(\theta; X)] = \int \nabla_\theta p(x\mid\theta) dx = \nabla_\theta \int p(x\mid\theta) dx = \boldsymbol{0}.
> $$
>
>   Differentiating $\int \nabla_\theta \log p(x\mid\theta)  p(x\mid\theta) dx = \boldsymbol{0}^\top$ via the product rule yields
>
> $$
> \boldsymbol{F}(\theta) \triangleq \mathrm{Cov}_{p_\theta}[\boldsymbol{s}(\theta; X)] = \mathbb{E}_{p_\theta}\!\bigl[\boldsymbol{s}(\theta; X)\boldsymbol{s}(\theta; X)^\top\bigr] = -\mathbb{E}_{p_\theta}\!\bigl[\nabla^2_\theta \log p(X \mid \theta)\bigr].
> $$
>
>   In an exponential family $p(x\mid\eta) = h(x)\exp(\eta^\top T(x) - A(\eta))$, $\nabla^2_\eta \log p = -\nabla^2_\eta A(\eta)$ is independent of $x$, so $\boldsymbol{F}_\eta = \nabla^2_\eta A(\eta) = \mathrm{Cov}_{p_\eta}[T(X)]$ and in mean parameters $m = \nabla_\eta A(\eta)$, $\boldsymbol{F}_m = \boldsymbol{F}_\eta^{-1}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, pp. 78–79). Under smooth reparameterization $\beta = g(\alpha)$, the chain rule $\boldsymbol{s}_\alpha = \boldsymbol{J}_g(\alpha)^\top \boldsymbol{s}_\beta$ gives $\boldsymbol{F}_\alpha(\alpha) = \boldsymbol{J}_g(\alpha)^\top \boldsymbol{F}_\beta(g(\alpha)) \boldsymbol{J}_g(\alpha)$.
> - **Cram\'er–Rao Lower Bound (CRLB)**: Let $\hat{\theta}_N = \delta(\mathcal{D})$ be any unbiased estimator ($\mathbb{E}_\theta[\hat{\theta}_N] = \theta$) from $N$ i.i.d. samples [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, p. 160), and let $\boldsymbol{s}_N(\theta) = \sum_{n=1}^N \boldsymbol{s}(\theta; x_n)$ satisfy $\mathbb{E}_\theta[\boldsymbol{s}_N] = \boldsymbol{0}$ and $\mathrm{Cov}_\theta[\boldsymbol{s}_N] = N\boldsymbol{F}(\theta)$. Differentiating $\int \hat{\theta}_N(\mathcal{D}) p(\mathcal{D}\mid\theta) d\mathcal{D} = \theta$ yields $\mathrm{Cov}_\theta(\hat{\theta}_N, \boldsymbol{s}_N) = \mathbb{E}_\theta[(\hat{\theta}_N - \theta)\boldsymbol{s}_N^\top] = \boldsymbol{I}_D$. For any $\boldsymbol{u}, \boldsymbol{v} \in \mathbb{R}^D$, Cauchy–Schwarz on $\boldsymbol{u}^\top(\hat{\theta}_N - \theta)$ and $\boldsymbol{v}^\top \boldsymbol{s}_N$ gives $(\boldsymbol{u}^\top \boldsymbol{v})^2 \le (\boldsymbol{u}^\top \mathrm{Cov}_\theta[\hat{\theta}_N]\boldsymbol{u})(\boldsymbol{v}^\top N\boldsymbol{F}(\theta)\boldsymbol{v})$. Choosing $\boldsymbol{v} = (N\boldsymbol{F}(\theta))^{-1}\boldsymbol{u}$ proves the Loewner matrix bound:
>
> $$
> \mathrm{Cov}_\theta[\hat{\theta}_N] \succeq \bigl(N\boldsymbol{F}(\theta)\bigr)^{-1}.
> $$
>
>   More generally, for a biased scalar estimator with $\mathbb{E}_\theta[\hat{\psi}_N] = \psi(\theta)$, $\mathrm{Var}_\theta[\hat{\psi}_N] \ge \frac{[\psi'(\theta)]^2}{N F(\theta)}$.

MLE achieves the Cram\'er–Rao floor in the large-sample limit: under mild regularity conditions, its sampling distribution collapses to a Gaussian covariance matching $(N\boldsymbol{F}(\theta^*))^{-1}$.

> [!important] Consistency and Asymptotic Normality (Efficiency) of the MLE
> Suppose the model is well-specified ($\mathcal{D} \overset{\mathrm{i.i.d.}}{\sim} p(\cdot \mid \theta^*)$ with $\theta^* \in \mathrm{int}(\Theta)$), identifiable, and three-times continuously differentiable with integrable third derivatives [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, Thm. 4.7.1, p. 154); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, Thm. 3.3.1, p. 75). By consistency $\hat{\theta}_{\mathrm{MLE}} \xrightarrow{p} \theta^*$, expanding the score stationarity equation $\boldsymbol{0} = \boldsymbol{s}_N(\hat{\theta}_{\mathrm{MLE}})$ around $\theta^*$ gives
>
> $$
> \boldsymbol{0} = \frac{1}{N}\boldsymbol{s}_N(\theta^*) + \left(\frac{1}{N}\nabla^2_\theta \ell(\tilde{\theta}_N)\right)\!(\hat{\theta}_{\mathrm{MLE}} - \theta^*),
> $$
>
> for $\tilde{\theta}_N$ on the segment between $\hat{\theta}_{\mathrm{MLE}}$ and $\theta^*$. Multiplying by $\sqrt{N}$ and rearranging:
>
> $$
> \sqrt{N}(\hat{\theta}_{\mathrm{MLE}} - \theta^*) = \left(-\frac{1}{N}\nabla^2_\theta \ell(\tilde{\theta}_N)\right)^{-1} \left(\frac{1}{\sqrt{N}}\sum_{n=1}^N \boldsymbol{s}(\theta^*; X_n)\right).
> $$
>
> By the CLT on the i.i.d. zero-mean score vectors, $\frac{1}{\sqrt{N}}\boldsymbol{s}_N(\theta^*) \xrightarrow{d} \mathcal{N}(\boldsymbol{0}, \boldsymbol{F}(\theta^*))$. By the WLLN, $-\frac{1}{N}\nabla^2_\theta\ell(\tilde{\theta}_N) \xrightarrow{p} -\mathbb{E}_{\theta^*}[\nabla^2_\theta \log p(X\mid\theta^*)] = \boldsymbol{F}(\theta^*)$. Applying Slutsky's theorem yields
>
> $$
> \sqrt{N}(\hat{\theta}_{\mathrm{MLE}} - \theta^*) \xrightarrow{d} \mathcal{N}\bigl(\boldsymbol{0}, \boldsymbol{F}(\theta^*)^{-1}\bigr), \qquad \text{or} \qquad \hat{\theta}_{\mathrm{MLE}} \stackrel{\cdot}{\sim} \mathcal{N}\bigl(\theta^*, (N\boldsymbol{F}(\theta^*))^{-1}\bigr).
> $$
>
> Thus MLE is asymptotically unbiased and *asymptotically efficient*, saturating the CRLB `(eq:asymp:crlb)`.

> [!tip] Information Curvature and Local KL Geometry
> Expanding the population KL divergence around $\theta^*$ to second order gives $D_{\mathrm{KL}}(p_{\theta^*} \parallel p_{\theta^*+\delta}) = \frac{1}{2}\delta^\top \boldsymbol{F}(\theta^*)\delta + O(\|\delta\|^3)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, p. 76). Directions with high curvature (steep log-likelihood valleys) carry large Fisher information and admit tight parameter estimates, whereas nearly flat directions produce tiny eigenvalues in $\boldsymbol{F}(\theta^*)$ and huge estimation variance.

### Bayesian Estimation, Regularized ERM, Decision Theory & Testing

In finite samples, MLE can overfit severely when model capacity approaches or exceeds $N$. Introducing a prior $p(\theta)$ regularizes point estimates via MAP and propagates uncertainty via the full posterior $p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D}\mid\theta)p(\theta)}{p(\mathcal{D})}$.

> [!important] MAP Estimation, Regularized ERM, and Conjugate Posterior Shrinkage
> Maximizing the posterior $p(\theta \mid \mathcal{D})$ drops the parameter-independent marginal likelihood $p(\mathcal{D})$ [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 8, pp. 505–515); [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 4, pp. 58–62):
>
> $$
> \hat{\theta}_{\mathrm{MAP}} = \arg\max_{\theta \in \Theta} \bigl[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\bigr] = \arg\min_{\theta \in \Theta} \left[\frac{1}{N}\sum_{n=1}^N \ell_{\mathrm{NLL}}(\theta; x_n) - \frac{1}{N}\log p(\theta)\right].
> $$
>
> Thus a Gaussian prior $\mathcal{N}(\theta \mid \boldsymbol{0}, \tau^2 \boldsymbol{I})$ induces $\ell_2$ (ridge) regularization $\frac{1}{2N\tau^2}\|\theta\|_2^2$, while a Laplace prior induces $\ell_1$ (LASSO) regularization [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 8, pp. 518–519); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, pp. 121–124). Two canonical conjugate families yield closed-form shrinkage:
> - **Beta–Binomial**: For $N_1$ heads in $N = N_1 + N_0$ Bernoulli($\theta$) trials with prior $\mathrm{Beta}(\theta \mid \alpha, \beta)$ ($\alpha,\beta \ge 1$), the posterior is $\mathrm{Beta}(\theta \mid N_1 + \alpha, N_0 + \beta)$, giving [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 4, pp. 59–61); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, pp. 130–134):
>
> $$
> \hat{\theta}_{\mathrm{MAP}} = \frac{N_1 + \alpha - 1}{N + \alpha + \beta - 2}, \qquad p(y_{\mathrm{new}} = 1 \mid \mathcal{D}) = \mathbb{E}[\theta \mid \mathcal{D}] = \frac{N_1 + \alpha}{N + \alpha + \beta},
> $$
>
>   reducing at $\alpha = \beta = 1$ to $\hat{\theta}_{\mathrm{MAP}} = \hat{\theta}_{\mathrm{MLE}} = N_1/N$ while the posterior predictive yields Laplace's rule of succession $\frac{N_1 + 1}{N + 2}$ even under a uniform prior.
> - **Gaussian–Gaussian Shrinkage**: For $y_1,\dots,y_N \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)$ with known observation precision $\kappa = 1/\sigma^2$ and prior $\mathcal{N}(\mu \mid \breve{m}, \breve{\lambda}^{-1})$ ($\breve{\lambda} = 1/\breve{\tau}^2$), completing the square in the exponent gives posterior $\mathcal{N}(\mu \mid \hat{m}, \hat{\lambda}^{-1})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, pp. 141–143); [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 8, pp. 507–512):
>
> $$
> \hat{\lambda} = \breve{\lambda} + N\kappa, \qquad \hat{m} = \frac{N\kappa \bar{y} + \breve{\lambda}\breve{m}}{N\kappa + \breve{\lambda}} = (1 - w)\bar{y} + w\breve{m}, \qquad w = \frac{\breve{\lambda}}{\breve{\lambda} + N\kappa}.
> $$
>
>   Here the posterior mean (= MAP mode) smoothly interpolates between the prior mean $\breve{m}$ (dominant for small $N\kappa$) and $\hat{\theta}_{\mathrm{MLE}} = \bar{y}$ (dominant as $N \to \infty$), illustrated in Figure `fig:asymp:gaussian-shrinkage`.

> [!note] Source Figure
> *Figure:* Conjugate Gaussian prior, likelihood, and posterior shrinkage given observation $\bar{y} = 3$ (likelihood centered at MLE $3$). (a) Strong prior $\mathcal{N}(\mu\mid 0, 1)$ pulls the posterior mode/mean strongly toward $0$. (b) Weak prior $\mathcal{N}(\mu\mid 0, 5)$ assigns greater relative precision weight $(1-w)$ to the data. Reproduced from [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, Fig. 4.16, p. 142). (`probabilisticml_kevinmurphy_introduction_p172_fig1.png`)

Decision theory unifies MAP and MMSE [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, pp. 169–178): every canonical point estimator minimizes posterior expected loss under a chosen metric, extending Chan's orthogonality derivation [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 8, pp. 532–533) across $\ell_2$, $\ell_1$, and $0\text{-}1$ risks.

> [!important] Bayesian Decision Theory, MMSE/Median/MAP Optimality, and Bayes Risk
> Given a loss function $\ell(\theta, a)$, the optimal action minimizing posterior expected loss $\rho(a \mid \mathcal{D}) = \int \ell(\theta, a) p(\theta \mid \mathcal{D}) d\theta$ satisfies:
> - **Quadratic ($\ell_2$) loss** $\ell_2(\theta, a) = \|\theta - a\|_2^2$: Setting $\nabla_a \mathbb{E}_{\theta\mid\mathcal{D}}[\|\theta - a\|_2^2] = 2(a - \mathbb{E}[\theta \mid \mathcal{D}]) = \boldsymbol{0}$ gives the **MMSE estimator** $a^* = \mathbb{E}[\theta \mid \mathcal{D}]$, satisfying the orthogonality error principle $\mathbb{E}[(\theta - a^*)^\top \varphi(\mathcal{D})] = 0$ [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 8, pp. 532–533); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, p. 176).
> - **Absolute ($\ell_1$) loss** $\ell_1(\theta, a) = \lvert \theta - a \rvert$: Subgradient stationarity requires $\mathbb{P}(\theta < a^* \mid \mathcal{D}) = \mathbb{P}(\theta \ge a^* \mid \mathcal{D}) = \frac{1}{2}$, yielding the **posterior median** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, p. 177).
> - **$0\text{-}1$ classification loss** $\ell_{01}(\theta, a) = \mathbf{1}_{\{\theta \ne a\}}$: Minimizing $\rho(a\mid\mathcal{D}) = 1 - \mathbb{P}(\theta = a \mid \mathcal{D})$ yields the **MAP mode** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, p. 169).
>
> On the frequentist side, estimator quality at fixed $\theta^*$ decomposes as $\mathrm{MSE}(\hat{\theta} \mid \theta^*) = \mathbb{E}[(\hat{\theta} - \theta^*)^2] = \mathrm{Var}[\hat{\theta}] + \mathrm{bias}(\hat{\theta})^2$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, p. 161; Ch. 5, p. 190). Under shrinkage `(eq:asymp:gauss-shrink)`, $\mathrm{MSE}(\hat{m} \mid \theta^*) = \frac{N\sigma^2 + \breve{\lambda}^2 \sigma^4(\breve{m} - \theta^*)^2}{(N + \breve{\lambda}\sigma^2)^2}$, trading slight bias for lower variance near $\breve{m}$. Averaging frequentist risk over a prior $\pi_0(\theta)$ defines the integrated **Bayes risk** $R_{\pi_0}(\delta) = \int \pi_0(\theta) R(\theta, \delta) d\theta$, whose global minimizer is precisely the pointwise Bayes decision rule $\delta(\mathcal{D}) = \arg\min_a \rho(a \mid \mathcal{D})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, Eq. (5.74)–(5.75), p. 191).

Finally, classical hypothesis testing and Bayesian comparison diverge sharply in how they treat unobserved replications versus parameter uncertainty [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, pp. 198–204) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, pp. 131–138).

> [!important] Frequentist Testing (Neyman–Pearson, LRT, p-Values) vs. Bayesian Inference
> - **Neyman–Pearson Lemma & Likelihood Ratio Test (LRT)** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, pp. 198–200): For simple hypotheses $H_0: \theta = \theta_0$ vs. $H_1: \theta = \theta_1$ at fixed Type I error $\alpha = \mathbb{P}(\text{reject } H_0 \mid H_0)$, the rejection rule maximizing power $1 - \beta = \mathbb{P}(\text{reject } H_0 \mid H_1)$ thresholds the likelihood ratio $\Lambda(\mathcal{D}) = \frac{p(\mathcal{D}\mid\theta_1)}{p(\mathcal{D}\mid\theta_0)} > \eta_\alpha$. For composite nested nulls $\Theta_0 \subset \Theta$ with $\Delta D = \dim(\Theta) - \dim(\Theta_0)$, maximizing the ratio $\tilde{\Lambda} = \frac{\sup_{\theta\in\Theta} p(\mathcal{D}\mid\theta)}{\sup_{\theta\in\Theta_0} p(\mathcal{D}\mid\theta)}$ yields Wilks' statistic $2\log\tilde{\Lambda} \xrightarrow{d} \chi^2_{\Delta D}$ under $H_0$.
> - **NHST p-values & Posterior Error Rate**: Given test statistic $t(\mathcal{D})$, the p-value is $p = \mathbb{P}(t(\tilde{\mathcal{D}}) \ge t(\mathcal{D}) \mid H_0)$ for hypothetical replications $\tilde{\mathcal{D}}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, pp. 81, 131). Crucially, $p$ is *not* $\mathbb{P}(H_0 \mid \mathcal{D})$; by Bayes' rule, the false-discovery probability given significance ($p < \alpha$) is $\mathbb{P}(H_0 \mid \text{sig}) = \frac{\alpha p(H_0)}{\alpha p(H_0) + (1-\beta)p(H_1)}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, p. 201).
> - **Bayes Factors & ROPE** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, pp. 132–133): Model-comparison Bayes tests integrate out parameters via the marginal likelihood $p(\mathcal{D}\mid M_k) = \int p(\mathcal{D}\mid\theta_k, M_k)p(\theta_k \mid M_k) d\theta_k$ to form the Bayes factor $B_{1,0} = \frac{p(\mathcal{D}\mid M_1)}{p(\mathcal{D}\mid M_0)} = \frac{p(M_1\mid\mathcal{D})}{p(M_0\mid\mathcal{D})} / \frac{p(M_1)}{p(M_0)}$ (which penalizes parameter volume automatically, approximated for smooth models by BIC $= \log p(\mathcal{D}\mid\hat{\theta}_{\mathrm{MLE}}) - \frac{D}{2}\log N$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, p. 186)). Alternatively, estimation-based Bayesian testing declares practical equivalence if the $95\%$ Highest Posterior Density (HPD) interval falls inside a Region of Practical Equivalence (ROPE) $[\theta_0 - \epsilon, \theta_0 + \epsilon]$, or rejects $H_0$ if HPD and ROPE are disjoint [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, p. 133).

## Algorithmic Implementation & Comparative Synthesis

When $\ell(\theta) + \log p(\theta)$ lacks a closed-form solution (e.g., generalized linear models or non-conjugate hierarchical models), second-order Fisher scoring finds $\hat{\theta}$ by replacing the stochastic observed Hessian with the positive-definite expected Fisher information, and bootstrap resampling quantifies finite-sample sampling covariance without Gaussian asymptotics [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, pp. 155–157); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, pp. 73–75).

> [!example] Algorithm: Damped Fisher Scoring (MLE/MAP) with Bootstrap Uncertainty Quantification
> **Input:** Dataset $\mathcal{D} = \{x_n\}_{n=1}^N$; log-likelihood $\ell(\theta; \mathcal{D}) = \sum_{n=1}^N \log p(x_n \mid \theta)$; per-sample FIM $\boldsymbol{F}(\theta)$; log-prior $\log p(\theta)$ (zero for MLE); prior precision $\boldsymbol{P}_{\mathrm{prior}}(\theta) = -\nabla^2_\theta \log p(\theta)$; damping $\epsilon_d > 0$; tolerance $\tau > 0$; bootstrap replicates $B$.
> **Output:** Point estimate $\hat{\theta}$, Laplace/asymptotic covariance $\hat{\boldsymbol{\Sigma}}_{\mathrm{asy}}$, bootstrap covariance $\hat{\boldsymbol{\Sigma}}_{\mathrm{boot}}$, $95\%$ percentile intervals.
>
>  *// Phase 1: Damped Fisher Scoring for MLE or MAP*
> Initialize $\theta^{(0)} \in \mathrm{int}(\Theta)$, $t \leftarrow 0$, objective $\mathcal{J}(\theta) \triangleq \ell(\theta; \mathcal{D}) + \log p(\theta)$
> **repeat until** $\Delta\theta^\top \boldsymbol{G}^{(t)} \Delta\theta < \tau^2$:
>
>   Evaluate total score $\boldsymbol{g}^{(t)} \leftarrow \sum_{n=1}^N \nabla_\theta \log p(x_n \mid \theta^{(t)}) + \nabla_\theta \log p(\theta^{(t)})$
>   Evaluate Fisher curvature $\boldsymbol{G}^{(t)} \leftarrow N\boldsymbol{F}(\theta^{(t)}) + \boldsymbol{P}_{\mathrm{prior}}(\theta^{(t)}) + \epsilon_d \boldsymbol{I}_D$
>   Solve positive-definite linear system $\boldsymbol{G}^{(t)} \Delta\theta = \boldsymbol{g}^{(t)}$
>   Find step size $\eta \in (0, 1]$ by backtracking Armijo search on $\mathcal{J}(\theta^{(t)} + \eta \Delta\theta)$
>   Update $\theta^{(t+1)} \leftarrow \theta^{(t)} + \eta \Delta\theta$, $t \leftarrow t + 1$
>
> Set $\hat{\theta} \leftarrow \theta^{(t)}$ and $\hat{\boldsymbol{\Sigma}}_{\mathrm{asy}} \leftarrow \bigl(N\boldsymbol{F}(\hat{\theta}) + \boldsymbol{P}_{\mathrm{prior}}(\hat{\theta})\bigr)^{-1}$
>
>  *// Phase 2: Nonparametric (or Parametric) Bootstrap Resampling*
> **for** $b = 1, \dots, B$:
>
>   Draw $\mathcal{D}^{(b)} = \{x_{n}^{(b)}\}_{n=1}^N$ i.i.d. uniformly from $\mathcal{D}$ (nonparametric) or from $p(\cdot \mid \hat{\theta})$ (parametric)
>   Run Phase 1 on $\mathcal{D}^{(b)}$ warm-started at $\theta^{(0)} \leftarrow \hat{\theta}$ to obtain $\hat{\theta}^{(b)}$
>
> Compute $\bar{\theta}_{\mathrm{boot}} = \frac{1}{B}\sum_{b=1}^B \hat{\theta}^{(b)}$ and $\hat{\boldsymbol{\Sigma}}_{\mathrm{boot}} = \frac{1}{B-1}\sum_{b=1}^B (\hat{\theta}^{(b)} - \bar{\theta}_{\mathrm{boot}})(\hat{\theta}^{(b)} - \bar{\theta}_{\mathrm{boot}})^\top$
> Return $\hat{\theta}$, $\hat{\boldsymbol{\Sigma}}_{\mathrm{asy}}$, $\hat{\boldsymbol{\Sigma}}_{\mathrm{boot}}$, and coordinate empirical quantiles $[q_{0.025}, q_{0.975}]$ of $\{\hat{\theta}^{(b)}\}_{b=1}^B$

Notice two computational distinctions in Algorithm `alg:asymp:fisher-bootstrap`: (i) Fisher scoring guarantees a positive-definite descent metric $\boldsymbol{G}^{(t)} \succ \boldsymbol{0}$ even at parameter locations where the observed negative Hessian $-\nabla^2_\theta\ell(\theta)$ is indefinite; and (ii) each nonparametric bootstrap dataset covers only $1 - (1 - 1/N)^N \approx 1 - e^{-1} \approx 63.2\%$ of unique points in $\mathcal{D}$, acting as a frequentist sampling analog ("poor man's posterior") when closed-form Bayes is unavailable [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, p. 156). Table `tab:asymp:complexity` contrasts the major estimation paradigms.

**Table:** Computational complexity and statistical properties across estimation paradigms ($N$ samples, $D$ parameters, $T$ optimization steps, $B$ bootstrap draws, $S$ posterior samples).

| **Method** | **Time Complexity** | **Memory** | **Uncertainty Output** | **Small-$N$ & Invariance Behavior** |
| — | — | — | — | — |
| Conjugate Bayes | $O(ND)$ sufficient stats | $O(D)$ or $O(D^2)$ | Exact posterior & predictive | Exact shrinkage; requires conjugate family |
| ERM / First-order SGD | $O(T B_{\mathrm{mb}} D)$ ($B_{\mathrm{mb}}\!\ll\! N$) | $O(D)$ | None (point estimate only) | Overfits without explicit regularizer |
| MLE (Fisher Scoring) | $O(T(ND^2 + D^3))$ | $O(D^2)$ | Asymptotic $\mathcal{N}(\hat{\theta}, (N\boldsymbol{F})^{-1})$ | Bijective invariance; fails if $N < D$ |
| MAP + Laplace Approx. | $O(T(ND^2 + D^3))$ | $O(D^2)$ | Gaussian $\mathcal{N}(\hat{\theta}_{\mathrm{MAP}}, \boldsymbol{H}_{\mathrm{MAP}}^{-1})$ | Mitigates small-$N$; mode not invariant |
| Nonparam. Bootstrap | $O(B \cdot C_{\mathrm{fit}})$ ($B \ge 200$) | $O(BD + D^2)$ | Empirical sampling dist. | Handles non-Gaussian stats; fails on boundaries |
| Full Bayes (MCMC/VI) | $O(S \cdot C_{\mathrm{grad}})$ | $O(SD)$ | Full multimodal posterior | Coherent uncertainty; invariant under Jeffreys |

> [!info] Cross-Book Synthesis: Asymptotics, Shrinkage, and the Likelihood Principle
> Synthesizing across the four monographs reveals three complementary lenses on estimation:
> - **Chan (Signal Processing & Finite-Sample Bounds)** develops probability inequalities directly from MGFs, proving Hoeffding and Chernoff bounds [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 343–350) before showing how CLT tail breakdowns motivate exact finite-$N$ bounds [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 378–380). His Chapter 8 links frequentist MLE invariance with Bayesian MAP and orthogonality-derived MMSE [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 8, pp. 500–533).
> - **Murphy (PML1/PML2: Decision Theory)** embeds MLE in KL minimization and PAC union bounds [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Chs. 4–5, pp. 109–110, 197), proves Fisher score covariance, CRLB, and asymptotic normality [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, pp. 75–79), and critiques NHST p-values via Bayes' theorem and the Likelihood Principle—advocating instead Bayes factors, HPD intervals, and ROPE [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, pp. 200–204); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, pp. 132–135).
> - **Tiep Vu (ML Engineering & Optimization)** derives canonical MLEs via direct log-derivative calculus and Lagrange multipliers, then highlights MAP's operational identity $\log p(\mathcal{D}\mid\theta) + \log p(\theta)$ as anti-overfitting regularization in small-$N$ regimes [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 4, pp. 53–62).

## Visual Diagnostics & Geometric Structure

Two complementary geometric structures govern finite-sample and asymptotic statistics: (1) the exponential separation between moment-based and MGF-based concentration bounds, and (2) the Bayesian–frequentist estimation hierarchy trading computational tractability for posterior calibration.

> [!example] Visual Diagram (LaTeX / PDF)
> *Comparison of finite-sample concentration bounds for $N = 50$ i.i.d. samples $X_n \sim \mathrm{Uniform}[0, 1]$ ($\mu = 1/2$, $\sigma^2 = 1/12$). Raw Markov ignores both $N$ and dispersion, while Chebyshev decays polynomially as $O(N^{-1}\varepsilon^{-2})$. Note that for tiny deviations ($\varepsilon \lesssim 0.196$) at moderate $N=50$, Chebyshev using the exact variance $\sigma^2 = 1/12$ beats worst-case Hoeffding (which effectively assumes maximal bounded variance $1/4$); beyond $\varepsilon \approx 0.196$, Hoeffding's exponential rate $e^{-2N\varepsilon^2}$ overtakes Chebyshev, and the variance-adaptive sub-Gaussian bound $2e^{-N\varepsilon^2/(2\sigma^2)}$ dominates by orders of magnitude.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:asymp:tailbounds` demonstrates quantitatively why MGF-based bounds unlock finite-sample generalization: at $\varepsilon = 0.25$ ($N=50$), Chebyshev still permits a $2.67\%$ deviation probability, whereas Hoeffding reduces the upper bound to $3.86 \times 10^{-3}$ and exact variance-aware sub-Gaussian Chernoff drops below $1.5 \times 10^{-8}$. Figure `fig:asymp:hierarchy` maps the full hierarchy connecting these concentration principles with parametric estimation and Bayesian decision theory.

> [!example] Visual Diagram (LaTeX / PDF)
> *Unified architecture of statistical learning and estimation. Progressing downward replaces worst-case finite-sample bounds and plug-in point approximations ($\hat{\theta}_{\mathrm{MLE}}, \hat{\theta}_{\mathrm{MAP}}$) with curvature-informed uncertainty (CRLB, Laplace) and full posterior integration, culminating in Bayesian decision-theoretic loss minimization [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Chs. 4–5); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

Applying asymptotic theory or MAP point summaries blindly in finite-sample or misspecified regimes produces systematic mathematical pathologies.

> [!warning] Reparameterization Non-Invariance of MAP Modes vs. MLE & Jeffreys Invariance
> **Failure Mode.** Unlike MLE, where $\widehat{h(\theta)}_{\mathrm{MLE}} = h(\hat{\theta}_{\mathrm{MLE}})$ holds unconditionally [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 8, pp. 500–502), MAP estimation maximizes a *probability density*, which transforms via the Jacobian determinant under a smooth bijection $\phi = g(\theta)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, pp. 103–105):
>
> $$
> p_\phi(\phi \mid \mathcal{D}) = p_\theta\bigl(g^{-1}(\phi) \mid \mathcal{D}\bigr) \lvert \det \boldsymbol{J}_{g^{-1}}(\phi) \rvert.
> $$
>
> Because the Jacobian factor shifts stationary points whenever $g$ is nonlinear, $\hat{\phi}_{\mathrm{MAP}} \ne g(\hat{\theta}_{\mathrm{MAP}})$: an estimator that is a mode in rate coordinates $\theta \in (0, 1)$ ceases to be the mode in log-odds coordinates $\phi = \log(\theta/(1-\theta))$.
>
> **Remedy.** Summarize full posteriors via quantile-invariant posterior means, medians, or probability mass integrals (Bayesian inference is invariant under reparameterization because probabilities $p(\theta\in A\mid\mathcal{D})$ integrate density with $d\theta$). When an uninformative reference prior is required across coordinate systems, choose the **Jeffreys prior** $p_J(\theta) \propto \sqrt{\det \boldsymbol{F}(\theta)}$, whose Jacobian rule $\boldsymbol{F}_\phi = \boldsymbol{J}_{g^{-1}}^\top \boldsymbol{F}_\theta \boldsymbol{J}_{g^{-1}}$ yields $p_J(\phi) \propto \sqrt{\det \boldsymbol{F}_\theta(g^{-1}(\phi))} \lvert \det \boldsymbol{J}_{g^{-1}}(\phi) \rvert$, preserving posterior invariance [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, pp. 104–105).

> [!warning] Small-Sample Bias, Boundary Collapse & High-Dimensional Covariance Singularity
> **Failure Mode.** Although MLE is asymptotically unbiased, at finite $N$ it suffers severe pathologies: (i) the Gaussian variance MLE $\hat{\sigma}_{\mathrm{MLE}}^2 = \frac{1}{N}\sum_{n=1}^N (x_n - \bar{x})^2$ has expectation $\frac{N-1}{N}\sigma^2$ (collapsing to $\hat{\sigma}_{\mathrm{MLE}}^2 = 0$ at $N=1$ [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 8, pp. 492–494)); (ii) in $D$ dimensions with $N \le D$, the sample covariance $\hat{\boldsymbol{\Sigma}}_{\mathrm{MLE}}$ has rank at most $N-1 < D$ and is strictly singular, causing $\hat{\boldsymbol{\Sigma}}_{\mathrm{MLE}}^{-1}$ and log-determinants to diverge [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, p. 114); and (iii) discrete MLE predicts zero probability for any unobserved outcome (the "black swan" paradox, e.g., $N_1 = N$ heads gives $\hat{\theta}_{\mathrm{MLE}} = 1$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 4, p. 133); [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 4, p. 61)).
>
> **Remedy.** Apply Bessel correction $\frac{N}{N-1}\hat{\sigma}_{\mathrm{MLE}}^2$ for unbiased scalar variance estimation, add diagonal or conjugate inverse-Wishart shrinkage priors when $D \gtrsim N$, and replace plug-in MLE with MAP smoothing or conjugate posterior predictive averaging (e.g., Laplace's rule $\frac{N_1 + 1}{N + 2}$).

> [!warning] CLT Breakdown in Heavy Tails, Missing Moments, and Large Deviations
> **Failure Mode.** Limit theorems rely strictly on finite moments and local scaling: (i) for Cauchy variables $f(x) = \frac{1}{\pi(1+x^2)}$, the characteristic function is $\varphi_X(t) = e^{-\lvert t \rvert}$ and both $\mathbb{E}[X]$ and $M_X(s)$ fail to exist [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, pp. 331, 360); consequently $\varphi_{\bar{X}_N}(t) = [e^{-\lvert t \rvert/N}]^N = e^{-\lvert t \rvert}$, meaning $\bar{X}_N \stackrel{d}{=} X_1$ never concentrates as $N \to \infty$; (ii) heavy-tailed distributions (Pareto, log-normal) lack finite MGFs in $(0, \delta)$, invalidating Chernoff/Hoeffding bounds; and (iii) even when all moments exist, CLT Gaussian approximations misestimate tail probabilities $\mathbb{P}(\bar{X}_N - \mu \ge \varepsilon)$ at $\varepsilon = \Theta(1)$ by orders of magnitude (Figure `fig:asymp:clt-tails`).
>
> **Remedy.** Never extrapolate CLT quantiles beyond $O(N^{-1/2})$ from the mean; evaluate tail risks using exact Chernoff rate functions $\psi_X^*(\varepsilon)$ or Hoeffding bounds when bounded/sub-Gaussian, and use median-of-means or truncated M-estimators when $\mathbb{E}[e^{sX}] = \infty$.

> [!warning] p-Value Fallacy, Optional Stopping & Multiple-Testing Inflation
> **Failure Mode.** Standard Null Hypothesis Significance Testing (NHST) frequently triggers three inferential failures: (i) equating $p = \mathbb{P}(t(\tilde{\mathcal{D}}) \ge t(\mathcal{D}) \mid H_0)$ with $\mathbb{P}(H_0 \mid \mathcal{D})$ ignores low prior plausibility $p(H_0)$ and low statistical power $1-\beta$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, pp. 200–202); (ii) stopping data collection early upon observing $p < 0.05$ alters the hypothetical replication space $\tilde{\mathcal{D}}$ and inflates Type I error, violating the **Likelihood Principle** (inference should depend only on the observed likelihood $p(\mathcal{D}\mid\theta)$, not unobserved stopping intentions [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 5, pp. 202–204); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, p. 81)); and (iii) running $M$ independent tests at nominal level $\alpha$ inflates the family-wise error rate (FWER) via the union bound to $1 - (1-\alpha)^M \approx M\alpha$ [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 6, p. 333).
>
> **Remedy.** Pre-specify sample sizes or use sequential Bayesian updating/Bayes factors (which satisfy the Likelihood Principle), pair effect-size intervals (HPD) with practical ROPE thresholds [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 3, pp. 132–135), and control multiplicity via Bonferroni threshold $\alpha/M$, Benjamini–Hochberg FDR control, or Bayesian hierarchical partial pooling.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[Concentration Inequalities]], [[Law of Large Numbers & CLT]], [[Maximum Likelihood & Fisher Info]]
- **Prerequisites (Directed $\leftarrow$):** [[KL Divergence]], [[Law of Large Numbers & CLT]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Concentration Inequalities]], [[Logistic Regression]], [[Maximum Likelihood & Fisher Info]], [[Multi-Armed Bandits & UCB]], [[Natural Gradient & K-FAC]], [[Rademacher Complexity & VC Dimension]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Concentration Inequalities]], [[KL Divergence]], [[Law of Large Numbers & CLT]], [[Logistic Regression]], [[Maximum Likelihood & Fisher Info]], [[Multi-Armed Bandits & UCB]], [[Natural Gradient & K-FAC]], [[Rademacher Complexity & VC Dimension]]
- **Synthesized Sources:** [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Machine Learning co ban]], [[Source - Probabilistic Machine Learning]]
