---
type: monograph
domain: "D06"
parent_topic: "[[PT 06 - Bandits, Markov Decision Processes & Value-Based RL]]"
sub_topic: "[[ST 06 - Multi-Armed Bandits, MDPs & Dynamic Programming]]"
cheatsheet: "[[Cheatsheet - Multi-Armed & Contextual Bandits, UCB, Thompson Sampling & Regret Bounds]]"
part_of: 
  - "[[ST 06 - Multi-Armed Bandits, MDPs & Dynamic Programming]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
concepts: 
  - "[[Multi-Armed Bandits & UCB]]"
  - "[[Thompson Sampling]]"
prerequisites: 
  - "[[Concentration Inequalities]]"
enables: 
  - "[[MCTS & MuZero]]"
related: 
  - "[[Concentration Inequalities]]"
  - "[[MCTS & MuZero]]"
sources: 
  - "[[Source - Francis Bach]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Reinforcement Learning - An Overview]]"
tags:
  - monograph
  - domain-06
---

# Multi-Armed & Contextual Bandits, UCB, Thompson Sampling & Regret Bounds

> [!abstract] Monograph Navigation
> - **Domain:** [[D06 - Reinforcement Learning, Control & LLM Alignment|Reinforcement Learning, Control & LLM Alignment]]
> - **Parent / Sub-Topic:** [[PT 06 - Bandits, Markov Decision Processes & Value-Based RL|Bandits, Markov Decision Processes & Value-Based RL]] → [[ST 06 - Multi-Armed Bandits, MDPs & Dynamic Programming|Multi-Armed Bandits, MDPs & Dynamic Programming]]
> - **Visual Cheatsheet:** [[Cheatsheet - Multi-Armed & Contextual Bandits, UCB, Thompson Sampling & Regret Bounds]]
> - **Description:** Covers exploration-exploitation trade-offs, cumulative regret, epsilon-greedy and optimistic initialization, Hoeffding-derived UCB1, Bayesian Thompson Sampling, LinUCB contextual bandits, and gradient bandit REINFORCE updates.

## Problem Formulation & Unified Notation

Unlike supervised learning, which receives *instructive feedback* specifying the correct target independently of the learner's output, reinforcement learning relies on *evaluative feedback*, where the agent observes a scalar reward $r_t$ evaluating only the selected action $a_t$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 2, p. 25). A **bandit problem** isolates this exploration–exploitation dilemma in a single-step setting where actions do not alter future environment states.

> [!important] Unified Bandit & Contextual Bandit Formulation
> Let $\mathcal{A} = \{1, \dots, K\}$ be a finite action (arm) set. At each round $t \in \{1, \dots, T\}$:
> - In a **contextual bandit**, the environment draws a context (state) $\boldsymbol{s}_t \in \mathcal{S}$ from an exogenous process $p(\boldsymbol{s}_t \mid \boldsymbol{s}_{1:t-1})$ independent of the agent's past actions [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 1.2.5, p. 19); in a **context-free $K$-armed bandit**, $\mathcal{S} = \{\emptyset\}$ is trivial.
> - The agent selects an action $a_t \sim \pi_t(\cdot \mid \boldsymbol{s}_t, h_t)$ conditioned on the history $h_t = \{(\boldsymbol{s}_\tau, a_\tau, r_\tau)\}_{\tau=1}^{t-1}$.
> - The environment emits a stochastic reward $r_t \sim p_R(\cdot \mid \boldsymbol{s}_t, a_t)$ with conditional expectation $R(\boldsymbol{s}_t, a_t) \triangleq \mathbb{E}[r_t \mid \boldsymbol{s}_t, a_t]$ (or $q_*(a) = \mu_a = R(a)$ when context-free).
>
> The finite-horizon objective is to maximize expected cumulative reward $J = \sum_{t=1}^T \mathbb{E}[r_t]$ (or discounted $J = \sum_{t=1}^\infty \gamma^{t-1}\mathbb{E}[r_t]$ with $\gamma \in (0,1)$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.4.1, Eq. 34.48–34.49, pp. 1120–1121). Letting $\pi^*(\boldsymbol{s}) \in \arg\max_{a \in \mathcal{A}} R(\boldsymbol{s}, a)$ be the oracle policy, the **per-step regret** and **cumulative regret** are
>
> $$
> \ell_t \triangleq \mathbb{E}_{\boldsymbol{s}_t}\!\left[R(\boldsymbol{s}_t, \pi^*(\boldsymbol{s}_t))\right] - \mathbb{E}_{\boldsymbol{s}_t, a_t \sim \pi_t}\!\left[R(\boldsymbol{s}_t, a_t)\right],
>   \qquad
>   L_T \triangleq \mathbb{E}\!\left[\sum_{t=1}^T \ell_t\right] = \sum_{a \in \mathcal{A}} \mathbb{E}[N_{T+1}(a)] \Delta_a,
>
> $$
>
> where the second equality holds in the context-free case with suboptimality gap $\Delta_a \triangleq R^* - R(a)$ and pull count $N_t(a) \triangleq \sum_{\tau=1}^{t-1} \mathbb{I}(a_\tau = a)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.4.7, Eq. 34.70–34.72, pp. 1127–1128).

> [!tip] Cumulative Regret vs. Simple Regret
> Minimizing cumulative regret $L_T$ corresponds to "earning while learning": every exploratory pull of a suboptimal arm $a$ incurs immediate cost $\Delta_a$, so a consistent policy must achieve sublinear regret $L_T = o(T)$. By contrast, **pure exploration** (or *best-arm identification*) minimizes only the terminal *simple regret* $\ell_T$ after a fixed budget $T$, decoupling data collection from exploitation [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 1.2.7.1, p. 22).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> The three source texts adopt distinct symbols and framing conventions for the same underlying decision process. Table `tab:bandit:notation` establishes a unified correspondence.
>
>
>
> | **Concept** | **Sutton & Barto** | **Murphy RL** | **Murphy Adv.** | **Unified Monograph Notation** |
> | — | — | — | — | — |
> | Number of arms | $k$ | $K$ or $n$ | $K$ | $K$ discrete actions $a \in \{1,\dots,K\}$ |
> | Action & reward | $A_t, R_t$ | $a_t, r_t$ | $a_t, r_t$ | $a_t \in \mathcal{A}$, $r_t \in \mathbb{R}$ at step $t$ |
> | True arm value | $q_*(a)$ | $R(a), \mu(a)$ | $R(\boldsymbol{s},a), \mu_a$ | $R(\boldsymbol{s},a) = \mathbb{E}[r_t \mid \boldsymbol{s}, a]$; gap $\Delta_a$ |
> | Action estimate | $Q_t(a)$ | $\hat{R}_t(a), \hat{\mu}_t(a)$ | $\hat{R}_t(\boldsymbol{s},a), \hat{\mu}_t(a)$ | $\hat{\mu}_t(a)$ (mean), $\tilde{R}_t(\boldsymbol{s},a)$ (UCB) |
> | Pull count | $N_t(a)$ | $N_t(a)$ | $N_t(a)$ | $N_t(a) = \sum_{\tau=1}^{t-1}\mathbb{I}(a_\tau=a)$ |
> | Context / state | Associative cue | $o_t = w_t$ (POMDP) | $\boldsymbol{s}_t \in \mathcal{S}$ | $\boldsymbol{s}_t \in \mathcal{S}$ (feature vector $\boldsymbol{x}_{t,a} \in \mathbb{R}^d$) |
> | Total regret | (Empirical reward) | $\mathrm{Regret}_T(\pi)$ | $L_T$ | $L_T = \sum_{t=1}^T \ell_t$ |

## Core Mathematical Derivations & Theorems

### Incremental Action-Value Estimation & Nonstationary Tracking

Storing all historical rewards requires $O(T)$ memory; incremental updates achieve $O(1)$ per-step memory and compute [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 2.4, p. 31 (PDF p. 53)). Given $n-1$ observations of an arm with sample average $Q_n$, the $n$-th reward $r_n$ updates the estimate via

$$
Q_{n+1} = \frac{1}{n}\sum_{i=1}^n r_i = Q_n + \frac{1}{n}\bigl(r_n - Q_n\bigr).
$$

Replacing $1/n$ with a constant step size $\alpha \in (0,1]$ yields an exponential recency-weighted average for nonstationary environments:

$$
Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^n \alpha(1-\alpha)^{n-i} r_i.
$$

Because $(1-\alpha)^n + \sum_{i=1}^n \alpha(1-\alpha)^{n-i} = 1$, the constant step size tracks drifting reward distributions while retaining an initial bias $(1-\alpha)^n Q_1$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 2.5, p. 32). Setting $Q_1(a)$ optimistically high (e.g., $+5$ when $q_*(a)\sim\mathcal{N}(0,1)$) induces temporary exploration under a greedy policy [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 2.6, p. 34). To remove initial bias while tracking nonstationary rewards, the unbiased constant-step-size trick sets $\beta_n = \alpha/\bar{o}_n$ with trace $\bar{o}_n = \bar{o}_{n-1} + \alpha(1 - \bar{o}_{n-1})$, $\bar{o}_0 = 0$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ex. 2.7, p. 35).

### Optimism in the Face of Uncertainty: UCB1 & Lai–Robbins Bounds

Random $\epsilon$-greedy exploration wastes pulls on clearly inferior arms. The **Optimism in the Face of Uncertainty (OFU)** principle constructs an upper confidence bound $\tilde{R}_t(a) = \hat{\mu}_t(a) + \delta_t(a)$ such that $\tilde{R}_t(a) \ge R(a)$ with high probability, and selects $a_t = \arg\max_{a \in \mathcal{A}} \tilde{R}_t(a)$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 7.2.3, Eq. 7.23, p. 178).

> [!important] Derivation of Hoeffding UCB1 & Lai–Robbins Lower Bound
> Assume rewards for each arm $a$ are bounded in $[0,1]$ (or $1/2$-sub-Gaussian). By Hoeffding's inequality, for a fixed pull count $n = N_t(a) \ge 1$, the empirical mean $\hat{\mu}_t(a)$ satisfies
>
> $$
> \mathbb{P}\!\left(\mu_a - \hat{\mu}_t(a) \ge \delta\right) \le \exp\!\left(-2n\delta^2\right).
>
> $$
>
> Setting the failure probability to $t^{-4}$ (so that $\sum_{t=1}^\infty t^{-4} < \infty$ via a union bound over $t$) and solving $\exp(-2n\delta^2) = t^{-4}$ gives $\delta_t(a) = \sqrt{\frac{2\ln t}{N_t(a)}}$, yielding the **UCB1** rule (generalized with exploration scale $c > 0$ in [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Eq. 2.10, p. 35) and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 34.64, p. 1125)):
>
> $$
> a_t = \arg\max_{a \in \mathcal{A}} \left[ \hat{\mu}_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}} \right], \qquad \text{(pulling any arm with } N_t(a)=0 \text{ first)}.
>
> $$
>
> For any suboptimal arm ($\Delta_a > 0$), UCB1 can pull $a$ more than $\ell = \lceil 8 \ln T / \Delta_a^2 \rceil$ times only if either the optimal arm's UCB underestimates $\mu^*$ or arm $a$'s empirical mean overestimates $\mu_a + \Delta_a/2$, yielding logarithmic problem-dependent regret $L_T \le \sum_{a:\Delta_a>0} \frac{8 \ln T}{\Delta_a} + O(K)$ and minimax regret $O(\sqrt{KT\ln T})$.
>
> This matches the fundamental **Lai–Robbins asymptotic lower bound** for any uniformly good consistent policy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.4.7, Eq. 34.73, p. 1128 (PDF p. 1162)):
>
> $$
> \liminf_{T \to \infty} \frac{L_T}{\ln T} \ge \sum_{a: \Delta_a > 0} \frac{\Delta_a}{D_{\mathrm{KL}}(p_R(a) \parallel p_R(a^*))}.
>
> $$
>
> In the Bayesian UCB variant, one replaces the Hoeffding radius with $c\hat{\sigma}_t(a)$, where $\hat{\sigma}_t(a)$ is the posterior standard deviation (e.g., $\sqrt{\hat{\mu}_t(a)(1-\hat{\mu}_t(a))/N_t(a)}$ for Beta-Bernoulli) [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 7.2.3.3, Eq. 7.27–7.28, p. 179).

### Bayesian Optimal Gittins Indices & Thompson Sampling

In the Bayesian formulation, the agent maintains a belief state $b_t = p(\boldsymbol{\theta} \mid h_t)$ updated deterministically via Bayes' rule, $b_t = \mathrm{BayesRule}(b_{t-1}, a_t, r_t)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 34.55, p. 1123). For a context-free Beta-Bernoulli bandit with uniform prior $\mathrm{Beta}(1,1)$ and counts $(w_a, l_a)$, the finite-horizon Bayes-optimal policy satisfies the belief-state Bellman equation [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 7.2.1.1, Eq. 7.14–7.16, p. 176):

$$
\begin{aligned}
  V^*(b) = \max_{a \in \mathcal{A}} \Biggl[ &\frac{w_a+1}{w_a+l_a+2}\bigl(1 + V^*(\dots, w_a+1, l_a, \dots)\bigr) \\
  &+ \left(1 - \frac{w_a+1}{w_a+l_a+2}\right) V^*(\dots, w_a, l_a+1, \dots) \Biggr].
\end{aligned}
$$

While infinite-horizon discounted context-free bandits decouple via **Gittins indices**, the belief-state space scales as $O(h^{2K})$ and is provably intractable for general contextual bandits [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.4.4, p. 1124).

**Thompson Sampling (Probability Matching)** circumvents this intractability by choosing each action according to its posterior probability of optimality [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 7.2.2.1, p. 177):

$$
\pi_t(a \mid \boldsymbol{s}_t, h_t) = \int \mathbb{I}\!\left(a = \arg\max_{a' \in \mathcal{A}} R(\boldsymbol{s}_t, a'; \boldsymbol{\theta})\right) p(\boldsymbol{\theta} \mid h_t)  \mathrm{d}\boldsymbol{\theta}.
$$

Operationally, this integral is realized without numerical integration by drawing a *single* posterior sample $\tilde{\boldsymbol{\theta}}_t \sim p(\boldsymbol{\theta} \mid h_t)$ and acting greedily: $a_t = \arg\max_{a'} R(\boldsymbol{s}_t, a'; \tilde{\boldsymbol{\theta}}_t)$. Despite its simplicity, Thompson Sampling achieves $O(\log T)$ problem-dependent regret and $O(\sqrt{KT\log T})$ worst-case Bayesian regret [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.4.6–34.4.7, pp. 1126–1128).

> [!note] Source Figure
> *Figure:* Sequential Beta-Bernoulli belief-state tree for a two-armed bandit from [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 34.8, p. 1123 (PDF p. 1157)). Updating $(\alpha_a, \beta_a)$ after each pull transitions the agent deterministically through the belief-state MDP. (`probabilisticml_kevinmurphy_advancedtopics_p1157_fig1.png`)

### Contextual Bandits: LinUCB & Off-Policy Evaluation

When contexts $\boldsymbol{s}_t$ carry predictive structure across actions via feature maps $\boldsymbol{x}_{t,a} = \phi(\boldsymbol{s}_t, a) \in \mathbb{R}^d$, assuming a linear reward model $R(\boldsymbol{s}_t, a) = \boldsymbol{\theta}_a^{\top} \boldsymbol{x}_{t,a} + \varepsilon_t$ ($\varepsilon_t$ $\sigma$-sub-Gaussian) enables generalization across unseen contexts [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.4.5–34.4.6, pp. 1124–1126).

> [!important] Ridge Posterior & Disjoint LinUCB Decision Rule
> For each arm $a \in \mathcal{A}$ with ridge regularizer $\lambda > 0$, define the design Gram matrix and reward-weighted sufficient statistic:
>
> $$
> \boldsymbol{A}_a = \lambda \boldsymbol{I}_d + \sum_{\tau < t:  a_\tau = a} \boldsymbol{x}_{\tau,a}\boldsymbol{x}_{\tau,a}^{\top},
>   \qquad
>   \boldsymbol{b}_a = \sum_{\tau < t:  a_\tau = a} r_\tau \boldsymbol{x}_{\tau,a},
>   \qquad
>   \hat{\boldsymbol{\theta}}_a = \boldsymbol{A}_a^{-1} \boldsymbol{b}_a.
>
> $$
>
> Under a Gaussian prior $\boldsymbol{\theta}_a \sim \mathcal{N}(\boldsymbol{0}, \lambda^{-1} \boldsymbol{I}_d)$ and noise variance $\sigma^2$, the posterior covariance is $\sigma^2 \boldsymbol{A}_a^{-1}$, so the predictive standard deviation along $\boldsymbol{x}_{t,a}$ is proportional to $\sqrt{\boldsymbol{x}_{t,a}^{\top} \boldsymbol{A}_a^{-1} \boldsymbol{x}_{t,a}}$. **LinUCB** selects
>
> $$
> a_t = \arg\max_{a \in \mathcal{A}} \left[ \hat{\boldsymbol{\theta}}_a^{\top} \boldsymbol{x}_{t,a} + \alpha \sqrt{\boldsymbol{x}_{t,a}^{\top} \boldsymbol{A}_a^{-1} \boldsymbol{x}_{t,a}} \right],
>
> $$
>
> achieving regret $\tilde{O}(d\sqrt{T})$ independent of $K$ when parameters are shared across arms. Conversely, to evaluate a target policy $\pi$ offline from logged data $\{(\boldsymbol{s}_i, a_i, r_i)\}_{i=1}^n$ collected under a stochastic logging policy $\pi_0(a_i \mid \boldsymbol{s}_i) > 0$, one uses **Inverse Propensity Weighting (IPW)** or its variance-reduced **Doubly Robust (DR)** extension:
>
> $$
> \begin{aligned}
>   \hat{V}_{\mathrm{IPW}}(\pi) &= \frac{1}{n}\sum_{i=1}^n \frac{\mathbb{I}(\pi(\boldsymbol{s}_i) = a_i)}{\pi_0(a_i \mid \boldsymbol{s}_i)}  r_i, \\
>   \hat{V}_{\mathrm{DR}}(\pi) &= \frac{1}{n}\sum_{i=1}^n \left[ \hat{R}(\boldsymbol{s}_i, \pi(\boldsymbol{s}_i)) + \frac{\mathbb{I}(\pi(\boldsymbol{s}_i) = a_i)}{\pi_0(a_i \mid \boldsymbol{s}_i)}\bigl(r_i - \hat{R}(\boldsymbol{s}_i, a_i)\bigr) \right],
> \end{aligned}
>
> $$
>
> where $\hat{V}_{\mathrm{DR}}$ is unbiased if *either* the reward model $\hat{R}$ or the logging propensity $\pi_0$ is accurately specified.

### Gradient Bandits & Baseline Variance Reduction

Rather than estimating action values, **gradient bandit algorithms** parameterize a softmax policy over numerical preferences $H_t(a) \in \mathbb{R}$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 2.8, Eq. 2.11–2.13, pp. 37–40 (PDF pp. 59–62)):

$$
\pi_t(a) \triangleq \mathbb{P}\{a_t = a\} = \frac{e^{H_t(a)}}{\sum_{b=1}^K e^{H_t(b)}}.
$$

> [!important] Exact Stochastic Gradient Ascent & Baseline Invariance Proof
> Differentiating expected reward $\mathbb{E}[r_t] = \sum_{x \in \mathcal{A}} \pi_t(x) q_*(x)$ with respect to $H_t(a)$ and inserting an arbitrary action-independent scalar baseline $B_t$ (valid because $\sum_x \frac{\partial \pi_t(x)}{\partial H_t(a)} = \frac{\partial }{\partial H_t(a)} 1 = 0$) yields
>
> $$
> \frac{\partial \mathbb{E}[r_t]}{\partial H_t(a)}
>   = \sum_{x \in \mathcal{A}} \bigl(q_*(x) - B_t\bigr) \frac{\partial \pi_t(x)}{\partial H_t(a)}
>   = \sum_{x \in \mathcal{A}} \pi_t(x) \bigl(q_*(x) - B_t\bigr) \frac{\partial \pi_t(x)/\partial H_t(a)}{\pi_t(x)}.
>
> $$
>
> By the quotient rule on `(eq:bandit:softmax_policy)`, $\frac{\partial \pi_t(x)}{\partial H_t(a)} = \pi_t(x)\bigl(\mathbb{I}_{a=x} - \pi_t(a)\bigr)$. Substituting $B_t = \bar{r}_t$ (running average reward) and replacing $q_*(a_t)$ with $r_t$ (unbiased since $\mathbb{E}[r_t \mid a_t] = q_*(a_t)$) gives
>
> $$
> \frac{\partial \mathbb{E}[r_t]}{\partial H_t(a)} = \mathbb{E}\!\left[(r_t - \bar{r}_t)\bigl(\mathbb{I}_{a=a_t} - \pi_t(a)\bigr)\right],
>
> $$
>
> yielding the stochastic gradient ascent (REINFORCE) update $H_{t+1}(a) = H_t(a) + \alpha(r_t - \bar{r}_t)(\mathbb{I}_{a=a_t} - \pi_t(a))$ for all $a \in \mathcal{A}$.

## Algorithmic Implementation & Comparative Synthesis

Algorithm `alg:bandit:unified` unifies frequentist UCB1, Bayesian Beta-Bernoulli Thompson Sampling, and contextual LinUCB within a single online interaction loop.

> [!example] Algorithm: Unified Bandit Exploration: UCB1, Thompson Sampling & Contextual LinUCB
> **Input:** Horizon $T$, action set $\mathcal{A}=\{1,\dots,K\}$, mode $\in \{\mathrm{UCB1}, \mathrm{TS}, \mathrm{LinUCB}\}$, hyperparameters $(c, \alpha_0, \beta_0, \lambda, \alpha_{\mathrm{lin}})$
> **Output:** Action sequence $\{a_t\}_{t=1}^T$ and updated sufficient statistics
> Initialize $N(a) \leftarrow 0$, $\hat{\mu}(a) \leftarrow 0$; $(\alpha_a, \beta_a) \leftarrow (\alpha_0, \beta_0)$; $\boldsymbol{A}_a \leftarrow \lambda \boldsymbol{I}_d$, $\boldsymbol{b}_a \leftarrow \boldsymbol{0}$ for all $a \in \mathcal{A}$
> **for** $t = 1, 2, \dots, T$:
>
>   Observe context $\boldsymbol{s}_t$ (and arm features $\boldsymbol{x}_{t,a} \in \mathbb{R}^d$ if contextual)
>   **if** $\mathrm{mode} = \mathrm{UCB1}$:
>
>     **if** $\exists a : N(a) = 0$:
> $a_t \leftarrow a$
>     **else**:
> $a_t \leftarrow \arg\max_{a \in \mathcal{A}} \bigl[\hat{\mu}(a) + c\sqrt{\ln t / N(a)}\bigr]$
>
>   **else if** $\mathrm{mode} = \mathrm{TS}$ (Beta-Bernoulli):
>
>     Draw $\tilde{\theta}_{t,a} \sim \mathrm{Beta}(\alpha_a, \beta_a)$ independently for all $a \in \mathcal{A}$
>     $a_t \leftarrow \arg\max_{a \in \mathcal{A}} \tilde{\theta}_{t,a}$
>
>   **else**:
>
>     For all $a \in \mathcal{A}$, compute $\hat{\boldsymbol{\theta}}_a \leftarrow \boldsymbol{A}_a^{-1} \boldsymbol{b}_a$ and $u_{t,a} \leftarrow \hat{\boldsymbol{\theta}}_a^{\top} \boldsymbol{x}_{t,a} + \alpha_{\mathrm{lin}} \sqrt{\boldsymbol{x}_{t,a}^{\top} \boldsymbol{A}_a^{-1} \boldsymbol{x}_{t,a}}$
>     $a_t \leftarrow \arg\max_{a \in \mathcal{A}} u_{t,a}$
>
>   Execute $a_t$ and observe reward $r_t \sim p_R(\cdot \mid \boldsymbol{s}_t, a_t)$
>   $N(a_t) \leftarrow N(a_t) + 1$; \quad $\hat{\mu}(a_t) \leftarrow \hat{\mu}(a_t) + \frac{1}{N(a_t)}\bigl(r_t - \hat{\mu}(a_t)\bigr)$
>   $\alpha_{a_t} \leftarrow \alpha_{a_t} + r_t$, \quad $\beta_{a_t} \leftarrow \beta_{a_t} + (1 - r_t)$
>   $\boldsymbol{A}_{a_t} \leftarrow \boldsymbol{A}_{a_t} + \boldsymbol{x}_{t,a_t}\boldsymbol{x}_{t,a_t}^{\top}$, \quad $\boldsymbol{b}_{a_t} \leftarrow \boldsymbol{b}_{a_t} + r_t \boldsymbol{x}_{t,a_t}$

| **Algorithm** | **Per-Step Time** | **Memory** | **Regret Bound** | **Context?** | **Key Mechanism** |
| — | — | — | — | — | — |
| $\epsilon$-Greedy (const. $\epsilon$) | $O(K)$ | $O(K)$ | $\Omega(\epsilon T)$ (linear) | No | Uniform random exploration |
| Optimistic Init. | $O(K)$ | $O(K)$ | $O(1)$ transient | No | High $Q_1(a)$ forces early trials |
| UCB1 | $O(K)$ | $O(K)$ | $O\!\left(\sum \frac{\ln T}{\Delta_a}\right)$ | No | Frequentist Hoeffding bonus |
| Thompson Sampling | $O(K)$ | $O(K)$ | $O(\sqrt{KT\ln T})$ | Yes (GP/Lin) | Posterior probability matching |
| Gradient Bandit | $O(K)$ | $O(K)$ | Asymptotic opt. | Yes (Policy) | Softmax SGA with baseline $\bar{r}_t$ |
| LinUCB | $O(Kd^2)$ | $O(Kd^2)$ | $\tilde{O}(d\sqrt{T})$ | Yes (Linear) | Ridge ellipsoid confidence set |

> [!info] Cross-Book Synthesis: Bayes-Optimal vs. Regret-Minimizing Paradigms
> Sutton & Barto [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 2.10, p. 42) benchmark exploration heuristics via empirical 10-armed testbed parameter studies (averaged over 2000 tasks), showing that UCB and optimistic initialization outperform fixed $\epsilon$-greedy over finite horizons. Murphy RL [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 7.1.3, Table 7.1, p. 175) elevates this distinction to a foundational duality: **Bayes-optimal policies** (Gittins indices, BAMDPs) maximize expected return under a known prior $P_0$ via automatic posterior-driven exploration, whereas **regret-minimizing policies** (UCB, minimax algorithms) require no prior, manually inject optimism or randomization, and remain robust to adversarial or nonstationary shifts. Thompson Sampling bridges both worlds by using posterior samples to achieve minimax-optimal frequentist regret.

## Visual Diagnostics & Geometric Structure

Figure `fig:bandit:exploration_taxonomy` contrasts the four primary exploration–exploitation decision loops, highlighting how each method maps history $h_t$ (and context $\boldsymbol{s}_t$) to action selection.

> [!example] Visual Diagram (LaTeX / PDF)
> *Taxonomy of exploration–exploitation decision rules: undirected randomization ($\epsilon$-greedy), frequentist optimism (UCB1), Bayesian probability matching (Thompson Sampling), and contextual ridge-ellipsoid optimism (LinUCB).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:bandit:regret_curves` illustrates the asymptotic separation between linear regret under constant $\epsilon$-greedy ($\Omega(T)$), minimax worst-case regret $O(\sqrt{KT\ln T})$, and problem-dependent logarithmic regret $O(\ln T)$ achieved by UCB1 and Thompson Sampling near the Lai–Robbins lower bound.

> [!example] Visual Diagram (LaTeX / PDF)
> *Scaling of cumulative regret $L_T$ with horizon $T$. Fixed $\epsilon$-greedy suffers linear regret because it continues pulling suboptimal arms with probability $\epsilon(K-1)/K$, whereas UCB1 and Thompson Sampling achieve logarithmic problem-dependent regret.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Empirical validation from the source texts confirms these theoretical regimes. Figure `fig:bandit:sutton_ucb` reproduces the 10-armed testbed comparison from [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Fig. 2.4, p. 36 (PDF p. 58)), showing UCB ($c=2$) surpassing $\epsilon$-greedy ($\epsilon=0.1$) after the initial $K=10$ forced exploration steps. Figure `fig:bandit:murphy_ctx_ts` shows linear-Gaussian contextual Thompson Sampling from [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 34.10, p. 1127 (PDF p. 1161)) adapting smoothly as arm dominance reverses over time while tracking the $c\sqrt{t}$ regret envelope.

> [!note] Source Figure
> *Figure:* Average reward on the 10-armed testbed comparing UCB ($c=2$) against $\epsilon$-greedy ($\epsilon=0.1$) [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Fig. 2.4, p. 36 (PDF p. 58)). Notice the characteristic spike at step $K+1=11$ once every arm has been sampled once. (`reinforcement_learning_introduction_richard_s_su_p58_fig1.png`)

> [!note] Source Figure
> *Figure:* Linear-Gaussian contextual Thompson Sampling with context $\boldsymbol{s}_t = (1, t, t^2)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 34.10, p. 1127 (PDF p. 1161)) (also [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Fig. 7.1, p. 177)). (a) True nonstationary arm rewards; (b) cumulative reward showing adaptation from arm 2 to arm 0; (c) observed cumulative regret vs. $c\sqrt{t}$. (`probabilisticml_kevinmurphy_advancedtopics_p1161_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Negative Feedback Loops & Propensity Violations in Logged Bandits
> If a bandit policy exploits an inaccurate model too early, it starves alternative arms and locks into a self-reinforcing *negative feedback loop* [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 34.7, p. 1122 (PDF p. 1156)). Moreover, in offline evaluation via `(eq:bandit:ipw_dr)`, deterministic logging policies ($\pi_0(a \mid \boldsymbol{s}) \in \{0,1\}$) violate *positivity* (common support), making IPW undefined or causing extreme variance when $\pi_0(a_i \mid \boldsymbol{s}_i) \ll 1$. **Remedy:** Enforce stochastic logging floors $\pi_0(a \mid \boldsymbol{s}) \ge \epsilon_{\min} > 0$, clip importance weights, or use Doubly Robust estimation.

> [!warning] Nonstationarity Collapse in Sample Averages, Optimistic Init & UCB1
> Sample-average updates ($\alpha_n = 1/n$) and optimistic initial values ($Q_1(a)=+5$) treat $t=1$ as privileged; if true arm values drift, sample averages adapt arbitrarily slowly and optimistic initialization exhausts its exploration drive permanently [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 2.6, pp. 34–35 (PDF pp. 56–57)). Similarly, standard UCB1 accumulates $N_t(a)$ forever and fails to re-explore after regime shifts. **Remedy:** Use constant step sizes with the unbiased trace $\beta_n = \alpha/\bar{o}_n$, sliding-window / discounted UCB, or Thompson Sampling with posterior forgetting.

> [!warning] Missing Baseline in Gradient Bandits & Singular Gram Matrices in LinUCB
> Omitting the baseline $\bar{r}_t$ in `(eq:bandit:grad_bandit_sga)` does not bias the expected gradient, but when rewards are shifted (e.g., $q_*(a) \approx +4$), every update pushes preferences upward, inflating variance and severely degrading convergence [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Fig. 2.5, pp. 37–38 (PDF pp. 59–60)). In contextual LinUCB, the unregularized Gram matrix $\sum \boldsymbol{x}_{\tau,a}\boldsymbol{x}_{\tau,a}^{\top}$ is singular whenever $N_t(a) < d$. **Remedy:** Always maintain an adaptive baseline $B_t = \bar{r}_t$ in gradient bandits, and initialize $\boldsymbol{A}_a = \lambda \boldsymbol{I}_d$ ($\lambda > 0$) with incremental Cholesky or Sherman–Morrison updates in LinUCB.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[Multi-Armed Bandits & UCB]], [[Thompson Sampling]]
- **Prerequisites (Directed $\leftarrow$):** [[Concentration Inequalities]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[MCTS & MuZero]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Concentration Inequalities]], [[MCTS & MuZero]]
- **Synthesized Sources:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]]
