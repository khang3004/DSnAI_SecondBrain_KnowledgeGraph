---
type: monograph
domain: "D06"
parent_topic: "[[PT 06 - Bandits, Markov Decision Processes & Value-Based RL]]"
sub_topic: "[[ST 06 - Monte Carlo, Temporal-Difference, DQN & Function Approximation]]"
cheatsheet: "[[Cheatsheet - Monte Carlo, SARSA, Q-Learning, DQN, Eligibility Traces & the Deadly Triad]]"
part_of: 
  - "[[ST 06 - Monte Carlo, Temporal-Difference, DQN & Function Approximation]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
concepts: 
  - "[[Deadly Triad]]"
  - "[[MC & TD Learning]]"
  - "[[Q-Learning & DQN]]"
prerequisites: 
  - "[[Bellman Equations]]"
  - "[[MC & TD Learning]]"
  - "[[Q-Learning & DQN]]"
enables: 
  - "[[Deadly Triad]]"
  - "[[Generalized Advantage Estimation (GAE)]]"
  - "[[Offline RL (CQL)]]"
  - "[[Q-Learning & DQN]]"
related: 
  - "[[Bellman Equations]]"
  - "[[Deadly Triad]]"
  - "[[Generalized Advantage Estimation (GAE)]]"
  - "[[MC & TD Learning]]"
  - "[[Offline RL (CQL)]]"
  - "[[Q-Learning & DQN]]"
sources: 
  - "[[Source - Francis Bach]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Reinforcement Learning - An Overview]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-06
---

# Monte Carlo, SARSA, Q-Learning, DQN, Eligibility Traces & the Deadly Triad

> [!abstract] Monograph Navigation
> - **Domain:** [[D06 - Reinforcement Learning, Control & LLM Alignment|Reinforcement Learning, Control & LLM Alignment]]
> - **Parent / Sub-Topic:** [[PT 06 - Bandits, Markov Decision Processes & Value-Based RL|Bandits, Markov Decision Processes & Value-Based RL]] → [[ST 06 - Monte Carlo, Temporal-Difference, DQN & Function Approximation|Monte Carlo, Temporal-Difference, DQN & Function Approximation]]
> - **Visual Cheatsheet:** [[Cheatsheet - Monte Carlo, SARSA, Q-Learning, DQN, Eligibility Traces & the Deadly Triad]]
> - **Description:** Contrasts MC high-variance unbiased returns against biased low-variance TD(0) bootstrapping, n-step and TD(lambda) eligibility traces, on-policy SARSA vs. off-policy Q-learning & Double Q-learning maximization bias, DQN stabilization (replay + target nets), and Baird's Deadly Triad.

## Problem Formulation & Unified Notation

In model-free reinforcement learning (RL), an agent interacts with an unknown Markov Decision Process (MDP) $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ through sampled trajectories $(S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_T)$ without access to the transition kernel $\mathcal{P}(s' \mid s, a)$ or expected reward function [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 5, PDF p. 113) [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.3, p. 35). Given a policy $\pi(a \mid s)$, the agent seeks to estimate the state-value function $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$ or action-value function $q_\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$ (prediction), and ultimately discover an optimal policy $\pi^*$ maximizing the expected discounted return (control):

$$
G_t  \doteq  \sum_{k=0}^{T - t - 1} \gamma^k R_{t+k+1}, \qquad \gamma \in [0, 1],
$$

where $T = \infty$ for continuing tasks ($\gamma < 1$) and $T < \infty$ for episodic tasks. Model-free value estimation spans a fundamental bias–variance–memory spectrum between pure Monte Carlo (MC) rollouts, one-step Temporal-Difference ($\mathrm{TD}(0)$) bootstrapping, $n$-step truncated returns, and $\mathrm{TD}(\lambda)$ eligibility traces. When tabular lookup is replaced by parametric function approximation $\hat{v}(s, \boldsymbol{w})$ or $Q_{\boldsymbol{w}}(s, a)$ under off-policy sampling, the interaction of approximation, bootstrapping, and off-policy distribution shift forms the *Deadly Triad*, necessitating architectural and algorithmic stabilizers such as Gradient-TD, Experience Replay, Target Networks, and Double/Dueling DQN [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 11, PDF p. 286) [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.5.2, p. 42).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Source texts adopt slightly different indexing conventions for rewards, step sizes, and network parameters. We standardize on Sutton & Barto's $R_{t+1}$ transition indexing while mapping Murphy's and Prince's conventions explicitly in Table `tab:td:notation`.
> 
>
>
>
> | **Concept** | **Sutton & Barto** | **Murphy (RL / Adv.)** | **Prince** | **Unified Monograph Convention** |
> | — | — | — | — | — |
> | Step-size / LR | $\alpha$ | $\eta$ | $\alpha$ | $\alpha \in (0, 1]$ (TD/MC), $\eta$ for SGD |
> | Reward at $t \to t+1$ | $R_{t+1}$ | $r_t$ or $r_{t+1}$ | $r[s_t, a_t]$ | $R_{t+1}$ (emitted upon $S_t, A_t \to S_{t+1}$) |
> | Value parameters | $\boldsymbol{w} \in \mathbb{R}^d$ | $\boldsymbol{w}$ | $\boldsymbol{\phi}$ | $\boldsymbol{w}$ (linear/general), $\boldsymbol{\theta}$ (deep Q-net) |
> | Target net params | — | $\boldsymbol{w}^-$ | $\boldsymbol{\phi}^-$ | $\boldsymbol{\theta}^-$ (frozen / Polyak-averaged) |
> | Action-value fn. | $\hat{q}(s, a, \boldsymbol{w})$ | $Q_{\boldsymbol{w}}(s, a)$ | $q[s_t, a_t, \boldsymbol{\phi}]$ | $Q_{\boldsymbol{\theta}}(s, a)$ (or $q_\pi(s, a)$ tabular) |
> | State distribution | $\mu(s)$ (on/off-policy) | $d^\pi(s)$ / $\mathcal{D}$ | $\mathcal{D}$ (replay) | $\mu(s)$ for state weighting, $\mathcal{D}$ for buffer |

## Core Mathematical Derivations & Theorems

### Monte Carlo Prediction, Importance Sampling & TD(0) Bootstrapping

Monte Carlo methods estimate $v_\pi(s)$ from complete episodic returns $G_t$, avoiding any reliance on transition dynamics [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 5, PDF p. 114). In *first-visit MC*, only the first occurrence of state $s$ in an episode contributes to the sample average; in *every-visit MC*, all visits contribute. Both are unbiased in the limit (first-visit is unbiased per sample: $\mathbb{E}[G_t \mid S_t = s] = v_\pi(s)$), but suffer from high variance because $G_t$ accumulates stochasticity across the entire remaining trajectory [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.2.1, PDF p. 1184).

When evaluating a target policy $\pi$ using episodes generated by a distinct exploratory behavior policy $b$ satisfying the *coverage assumption* ($\pi(a \mid s) > 0 \implies b(a \mid s) > 0$), we weight each return by the trajectory importance-sampling (IS) ratio [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 5, PDF pp. 125–127):

$$
\rho_{t:T-1}  \doteq  \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}.
$$

*Ordinary IS* averages $\rho_{t:T-1} G_t$ over visits $\mathcal{J}(s)$: $V(s) = \frac{\sum_{t \in \mathcal{J}(s)} \rho_{t:T-1} G_t}{\lvert \mathcal{J}(s) \rvert}$, which is unbiased ($\mathbb{E}_b[\rho_{t:T-1} G_t \mid S_t = s] = v_\pi(s)$) but can have infinite variance when $\rho_{t:T-1}$ is unbounded [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 5, PDF pp. 128–129). *Weighted IS* normalizes by the sum of ratios:

$$
V(s)  \doteq  \frac{\sum_{t \in \mathcal{J}(s)} \rho_{t:T-1} G_t}{\sum_{t \in \mathcal{J}(s)} \rho_{t:T-1}},
$$

introducing finite-sample bias that vanishes asymptotically ($O(1/N)$) while drastically reducing variance.

In contrast, *Temporal-Difference* ($\mathrm{TD}(0)$) learning updates $V(S_t)$ immediately after a single transition $(S_t, A_t, R_{t+1}, S_{t+1})$ by bootstrapping from the current estimate $V(S_{t+1})$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 6, PDF p. 141) [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.3.2, p. 36):

$$
V(S_t)  \leftarrow  V(S_t) + \alpha \underbrace{\bigl[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \bigr]}_{\delta_t  \text{(TD error)}}.
$$

> [!note] Source Figure
> *Figure:* Backup diagrams contrasting sample-based full-depth Monte Carlo (left), sample-based one-step Temporal-Difference bootstrapping (center), and full-width one-step Dynamic Programming expectation (right) [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Fig. 2.3, p. 37). (`reinforement_learning_kevin_murphy_p37_fig1.png`)

> [!tip] Why TD(0) Outperforms MC Under Batch Training
> In finite-sample batch training, MC minimizes the empirical squared error on observed returns $\sum_t (G_t - V(S_t))^2$, whereas $\mathrm{TD}(0)$ converges to the *certainty-equivalence* estimate: the exact value function of the maximum-likelihood MDP inferred from the observed transitions [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 6, PDF pp. 148–150). By exploiting the Markov property rather than ignoring state-transition structure, $\mathrm{TD}(0)$ achieves strictly lower asymptotic estimation error on Markovian environments despite its transient bootstrapping bias.

### TD Control: SARSA, Q-Learning, Expected Sarsa & Double Learning

Extending $\mathrm{TD}(0)$ to action values $Q(s, a)$ yields three canonical tabular control algorithms, distinguished by how they construct the bootstrapping target [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 6, PDF pp. 151–156):
- **SARSA (On-Policy):** Samples $A_{t+1} \sim \pi(\cdot \mid S_{t+1})$ from the same $\epsilon$-greedy policy used for behavior:
  
$$
Q(S_t, A_t)  \leftarrow  Q(S_t, A_t) + \alpha \bigl[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \bigr].
$$
- **Q-Learning (Off-Policy):** Directly approximates $q_*$ independently of the exploratory behavior policy by bootstrapping from the greedy maximum:
  
$$
Q(S_t, A_t)  \leftarrow  Q(S_t, A_t) + \alpha \bigl[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \bigr].
$$
- **Expected Sarsa:** Replaces the single sampled next action $A_{t+1}$ with its expectation under target policy $\pi$, eliminating sampling variance over $A_{t+1}$:
  
$$
Q(S_t, A_t)  \leftarrow  Q(S_t, A_t) + \alpha \Bigl[ R_{t+1} + \gamma \sum_{a'} \pi(a' \mid S_{t+1}) Q(S_{t+1}, a') - Q(S_t, A_t) \Bigr].
$$

Notice that Q-learning is a special case of Expected Sarsa when $\pi$ is the deterministic greedy policy with respect to $Q$.

> [!important] Maximization Bias & Double Q-Learning
> In Q-learning, the target $\max_{a'} Q(S_{t+1}, a')$ uses the same noisy estimates both to select the maximizing action and to evaluate it. By Jensen's inequality for the convex $\max$ operator on random variables $\{X_a\}_{a \in \mathcal{A}}$ with $\mathbb{E}[X_a] = q(a)$:
>
> $$
> \mathbb{E}\bigl[\max_{a} X_a\bigr]  \ge  \max_{a} \mathbb{E}[X_a]  =  \max_{a} q(a),
>
> $$
>
> with strict inequality whenever multiple actions have noisy estimates near the maximum [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 6, PDF pp. 156–157) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 19.4.2, p. 401). This positive *maximization bias* causes Q-learning to overestimate risky/high-variance suboptimal actions.
>
> **Double Q-Learning** decouples action selection from evaluation by maintaining two independent estimators $Q_1, Q_2$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 6, PDF pp. 157–158) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.2.5, PDF p. 1188). At each step, with probability $0.5$ we update $Q_1$ using $Q_2$ to evaluate $Q_1$'s greedy action (and symmetrically for $Q_2$):
>
> $$
> Q_1(S_t, A_t)  \leftarrow  Q_1(S_t, A_t) + \alpha \Bigl[ R_{t+1} + \gamma Q_2\bigl(S_{t+1}, \arg\max_{a'} Q_1(S_{t+1}, a')\bigr) - Q_1(S_t, A_t) \Bigr].
>
> $$
>
> Since $Q_2$ is independent of the noise that selected $a^* = \arg\max_{a'} Q_1(S_{t+1}, a')$, $\mathbb{E}[Q_2(S_{t+1}, a^*)] = q(S_{t+1}, a^*)$ is unbiased for that action's true value.

> [!note] Source Figure
> *Figure:* Empirical demonstration of maximization bias on a two-state episodic MDP: standard Q-learning heavily over-selects the suboptimal `left` action entering state $B$ (whose actions have $\mathcal{N}(-0.1, 1)$ rewards), whereas Double Q-learning eliminates the bias [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Fig. 6.5, PDF p. 157). (`reinforcement_learning_introduction_richard_s_su_p157_fig1.png`)

### n-Step Bootstrapping & TD($\lambda$) Eligibility Traces

Between $\mathrm{TD}(0)$ ($1$-step) and MC ($\infty$-step) lies the $n$-step return [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 7, PDF p. 165):

$$
G_{t:t+n}  \doteq  R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n}), \qquad n \ge 1,
$$

with $G_{t:t+n} \doteq G_t$ if $t + n \ge T$. Rather than committing to a single horizon $n$, the *forward-view $\lambda$-return* forms an exponentially decaying convex combination of all $n$-step returns with trace-decay parameter $\lambda \in [0, 1]$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 12, PDF pp. 310–311) [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.3.3, p. 36):

$$
G_t^\lambda  \doteq  (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}  =  (1 - \lambda)\sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1} G_t.
$$

Setting $\lambda = 0$ recovers $G_t^0 = G_{t:t+1}$ ($\mathrm{TD}(0)$), while $\lambda = 1$ recovers $G_t^1 = G_t$ (MC).

> [!important] Forward–Backward Equivalence of $\mathrm{TD}(\lambda)$
> Computing $G_t^\lambda$ directly requires waiting until episode termination $T$ (non-causal). **Backward-view $\mathrm{TD}(\lambda)$** achieves causal, step-by-step updates using an *eligibility trace* vector $\boldsymbol{z}_t \in \mathbb{R}^d$ that tracks recently active feature gradients [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 12, PDF pp. 314–315):
>
> $$
> \boldsymbol{z}_{-1} = \boldsymbol{0}, \qquad \boldsymbol{z}_t  \doteq  \gamma \lambda \boldsymbol{z}_{t-1} + \nabla_{\boldsymbol{w}} \hat{v}(S_t, \boldsymbol{w}_t),
>
> $$
>
> paired with the semi-gradient update driven by the one-step TD error $\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \boldsymbol{w}_t) - \hat{v}(S_t, \boldsymbol{w}_t)$:
>
> $$
> \boldsymbol{w}_{t+1}  \doteq  \boldsymbol{w}_t + \alpha \delta_t \boldsymbol{z}_t.
>
> $$
>
> For linear function approximation $\hat{v}(s, \boldsymbol{w}) = \boldsymbol{w}^{\top} \boldsymbol{x}(s)$ under offline (end-of-episode) weight accumulation, the sum of backward-view updates equals the sum of forward-view $\lambda$-return updates:
>
> $$
> \sum_{t=0}^{T-1} \delta_t \boldsymbol{z}_t  =  \sum_{t=0}^{T-1} \bigl(G_t^\lambda - \boldsymbol{w}^{\top} \boldsymbol{x}(S_t)\bigr) \boldsymbol{x}(S_t),
>
> $$
>
> verified by telescoping the geometric series $G_t^\lambda - \hat{v}(S_t) = \sum_{k=t}^{T-1} (\gamma \lambda)^{k-t} \delta_k$ and exchanging summation indices [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 12, PDF pp. 315–316). For online step-by-step updates, *True Online $\mathrm{TD}(\lambda)$* eliminates the discrepancy via Dutch traces [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 12, PDF p. 321).

### Function Approximation, Semi-Gradient TD & The Deadly Triad

When $\lvert \mathcal{S} \rvert$ is large or continuous, we parameterize $\hat{v}(s, \boldsymbol{w})$ and minimize the weighted Value Error $\overline{\mathrm{VE}}(\boldsymbol{w}) \doteq \lVert \hat{v}(\cdot, \boldsymbol{w}) - v_\pi \rVert_\mu^2 = \sum_{s \in \mathcal{S}} \mu(s) [\hat{v}(s, \boldsymbol{w}) - v_\pi(s)]^2$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 9, PDF pp. 221–222). Because $v_\pi(S_t)$ is unknown, *semi-gradient $\mathrm{TD}(0)$* substitutes the bootstrapping target $U_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \boldsymbol{w}_t)$ and differentiates only the prediction $\hat{v}(S_t, \boldsymbol{w}_t)$, ignoring the dependence of $U_t$ on $\boldsymbol{w}_t$:

$$
\boldsymbol{w}_{t+1}  \doteq  \boldsymbol{w}_t + \alpha \bigl[ R_{t+1} + \gamma \hat{v}(S_{t+1}, \boldsymbol{w}_t) - \hat{v}(S_t, \boldsymbol{w}_t) \bigr] \nabla_{\boldsymbol{w}} \hat{v}(S_t, \boldsymbol{w}_t).
$$

Under *on-policy* sampling ($\mu = d^\pi$) and linear features $\hat{v}(s, \boldsymbol{w}) = \boldsymbol{w}^{\top} \boldsymbol{x}(s)$, semi-gradient TD(0) converges reliably to the TD fixed point $\boldsymbol{w}_{\mathrm{TD}}$ where the *Projected Bellman Error* vanishes: $\mathrm{PBE}(\boldsymbol{w}) \doteq \lVert \Pi \overline{\delta}_{\boldsymbol{w}} \rVert_\mu^2 = 0$, with $\Pi = \boldsymbol{X}(\boldsymbol{X}^{\top} \boldsymbol{D} \boldsymbol{X})^{-1} \boldsymbol{X}^{\top} \boldsymbol{D}$ and $\boldsymbol{D} = \mathrm{diag}(\mu)$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 11, PDF pp. 288–290).

However, when *off-policy* sampling ($\mu \ne d^\pi$) is combined with bootstrapping and function approximation, the semi-gradient operator $\Pi \mathcal{T}_\pi$ is no longer guaranteed to be a contraction in $\lVert \cdot \rVert_\mu$, causing weights to diverge to infinity—even in linear MDPs such as **Baird's 7-state counterexample** [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 11, PDF pp. 282–287) [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.5.2.5, p. 43).

> [!note] Source Figure
> *Figure:* **Baird's Counterexample:** (a) A 7-state MDP with target policy $\pi(\text{solid} \mid \cdot) = 1$, behavior policy $b(\text{dashed} \mid \cdot) = 6/7$, zero rewards, and $\gamma = 0.99$. Although $\boldsymbol{w} = \boldsymbol{0}$ represents the exact $v_\pi \equiv 0$, (b) off-policy semi-gradient $\mathrm{TD}(0)$ diverges exponentially [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Fig. 2.6, p. 43). (`reinforement_learning_kevin_murphy_p43_fig1.png`)

> [!important] Gradient-TD (GTD2 / TDC) Stabilization
> To obtain $O(d)$ true stochastic gradient descent on the Projected Bellman Error under arbitrary off-policy distributions, **Gradient-TD** rewrites $\nabla_{\boldsymbol{w}} \mathrm{PBE}(\boldsymbol{w})$ using expectations under behavior policy $b$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 11, PDF pp. 300–302):
>
> $$
> \nabla_{\boldsymbol{w}} \mathrm{PBE}(\boldsymbol{w})  =  2 \mathbb{E}\bigl[\rho_t (\gamma \boldsymbol{x}_{t+1} - \boldsymbol{x}_t)\boldsymbol{x}_t^{\top}\bigr] \bigl(\mathbb{E}[\boldsymbol{x}_t \boldsymbol{x}_t^{\top}]\bigr)^{-1} \mathbb{E}[\rho_t \delta_t \boldsymbol{x}_t].
>
> $$
>
> Because the first and third expectations both depend on $\boldsymbol{x}_{t+1}$, multiplying single-step samples is biased. GTD2/TDC circumvents this double-sampling issue by maintaining an auxiliary $d$-vector $\boldsymbol{v}_t \approx (\mathbb{E}[\boldsymbol{x}_t \boldsymbol{x}_t^{\top}])^{-1} \mathbb{E}[\rho_t \delta_t \boldsymbol{x}_t]$ via a fast-timescale LMS update ($\boldsymbol{v}_{t+1} = \boldsymbol{v}_t + \beta \rho_t (\delta_t - \boldsymbol{v}_t^{\top} \boldsymbol{x}_t) \boldsymbol{x}_t$) and updating $\boldsymbol{w}$ on a slower timescale:
>
> $$
> \boldsymbol{w}_{t+1}  =  \boldsymbol{w}_t + \alpha \rho_t \bigl(\delta_t \boldsymbol{x}_t - \gamma \boldsymbol{x}_{t+1} (\boldsymbol{x}_t^{\top} \boldsymbol{v}_t)\bigr) \qquad (\text{TDC}).
>
> $$

## Algorithmic Implementation & Comparative Synthesis

Scaling Q-learning to high-dimensional observations (e.g., Atari $84 \times 84 \times 4$ frame stacks) via deep neural networks $Q_{\boldsymbol{\theta}}(s, a)$—termed *Fitted Q-Iteration* or *Deep Q-Networks (DQN)*—triggers severe Deadly Triad oscillations because adjacent frames are strongly correlated and the regression target $R_{t+1} + \gamma \max_{a'} Q_{\boldsymbol{\theta}}(S_{t+1}, a')$ moves on every gradient step [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 19.4, pp. 399–401) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.2.6, PDF p. 1188). Modern deep value-based RL stabilizes training through six synergistic mechanisms unified in **Rainbow** [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.5.4, pp. 45–48):
- **Experience Replay ($\mathcal{D}$):** Stores $(s, a, r, s', \text{done})$ tuples and samples minibatches uniformly or via **Prioritized Experience Replay (PER)**, where transition $i$ has priority $p_i = (\lvert \delta_i \rvert + \epsilon_{\text{per}})^\alpha$ and importance-sampling correction weights $w_i = (N \cdot P(i))^{-\beta}$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.5.2.3–2.5.2.4, p. 42).
- **Frozen Target Network ($\boldsymbol{\theta}^-$):** Decouples target evaluation from online parameter updates by holding $\boldsymbol{\theta}^-$ fixed for $C$ steps (or Polyak averaging $\boldsymbol{\theta}^- \leftarrow \tau \boldsymbol{\theta} + (1 - \tau)\boldsymbol{\theta}^-$), yielding the DQN loss [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. (35.14), PDF p. 1189):
  
$$
\mathcal{L}_{\mathrm{DQN}}(\boldsymbol{\theta})  =  \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \Bigl[\bigl(r + \gamma \max_{a'} Q_{\boldsymbol{\theta}^-}(s', a') - Q_{\boldsymbol{\theta}}(s, a)\bigr)^2\Bigr].
$$
- **Double DQN Target:** Eliminates maximization bias by selecting the greedy action with $\boldsymbol{\theta}$ and evaluating it with $\boldsymbol{\theta}^-$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.5.3.2, p. 45):
  
$$
y_i^{\mathrm{DDQN}}  =  r_i + \gamma (1 - \text{done}_i)  Q_{\boldsymbol{\theta}^-}\!\bigl(s'_i, \arg\max_{a'} Q_{\boldsymbol{\theta}}(s'_i, a')\bigr).
$$
- **Dueling Architecture:** Factorizes $Q_{\boldsymbol{\theta}}(s, a)$ into a state-value stream $V_{\boldsymbol{\theta}}(s)$ and an advantage stream $A_{\boldsymbol{\theta}}(s, a)$, enforcing identifiability ($\mathbb{E}_{\pi^*}[A^*] = 0$) via mean subtraction [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.5.4.2, pp. 46–47):
  
$$
Q_{\boldsymbol{\theta}}(s, a)  =  V_{\boldsymbol{\theta}}(s) + \Bigl(A_{\boldsymbol{\theta}}(s, a) - \frac{1}{\lvert \mathcal{A} \rvert} \sum_{a'} A_{\boldsymbol{\theta}}(s, a')\Bigr).
$$
- **Multi-step ($n=3$) Returns, Noisy Nets & Distributional RL (C51):** Reduce bootstrapping bias, provide state-dependent exploration, and learn the full return distribution rather than only its expectation [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.5.4.6, p. 48).

> [!example] Algorithm: Double DQN with Prioritized Experience Replay & Target Network
> **Input:** Environment $\mathcal{M}$, replay buffer $\mathcal{D}$ (capacity $N$), online weights $\boldsymbol{\theta}$, target weights $\boldsymbol{\theta}^- \leftarrow \boldsymbol{\theta}$, step size $\eta$, discount $\gamma$, target sync period $C$, PER exponents $(\alpha, \beta)$.
> **Output:** Trained action-value network $Q_{\boldsymbol{\theta}}(s, a)$.
> **for** episode $= 1, \dots, E$:
>
>   Observe initial state $s_0$
>   **for** $t = 0, 1, \dots$ until terminal:
>
>     Select $a_t$ via $\epsilon$-greedy policy in $Q_{\boldsymbol{\theta}}(s_t, \cdot)$
>     Execute $a_t$, observe $r_{t+1}, s_{t+1}, \text{done}_t$, and store $(s_t, a_t, r_{t+1}, s_{t+1}, \text{done}_t)$ in $\mathcal{D}$ with $p_t = \max_{j} p_j$
>     Sample minibatch $\{i_k\}_{k=1}^B \sim P(i) = p_i^\alpha / \sum_j p_j^\alpha$
>     **for** each sample $i$ in minibatch:
>
>       Compute Double DQN target $y_i \leftarrow r_i + \gamma(1 - \text{done}_i) Q_{\boldsymbol{\theta}^-}(s'_i, \arg\max_{a'} Q_{\boldsymbol{\theta}}(s'_i, a'))$
>       Compute TD error $\delta_i \leftarrow y_i - Q_{\boldsymbol{\theta}}(s_i, a_i)$ and IS weight $w_i \leftarrow (N \cdot P(i))^{-\beta}$
>       Update priority $p_i \leftarrow (\lvert \delta_i \rvert + \epsilon_{\text{per}})^\alpha$
>
>     Update $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \frac{1}{B} \sum_{i} w_i   \nabla_{\boldsymbol{\theta}} \bigl[\tfrac{1}{2}\delta_i^2\bigr]$ (normalized by $\max_j w_j$)
>     **if** $t \bmod C = 0$:
> $\boldsymbol{\theta}^- \leftarrow \boldsymbol{\theta}$

> [!info] Cross-Book Synthesis: Sample Efficiency, Goal Relabeling & Modern DQN
> While Sutton & Barto emphasize theoretical convergence guarantees (e.g., Gradient-TD and Emphatic-TD) for linear off-policy prediction [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 11, PDF pp. 300–304), Murphy RL bridges classical control to modern sample-efficient deep RL: *Bigger, Better, Faster (BBF)* achieves state-of-the-art Atari-100k performance by combining high update-to-data (UTD) ratios, periodic soft weight resets to prevent loss of plasticity, annealing $n$-step returns ($n: 10 \to 3$), and dueling/distributional heads [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.5.4.7, pp. 48–49). For sparse-reward goal-conditioned tasks, *Hindsight Experience Replay (HER)* relabels failed trajectories with achieved terminal states $g' = s_T$ inside the off-policy buffer $\mathcal{D}$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.5.5, p. 49).

**Table:** Per-step computational complexity, memory footprint, and convergence properties ($d$: parameter dimension; $\lvert \mathcal{A} \rvert$: action count; $B$: minibatch size).

| **Method** | **Time / Step** | **Memory** | **Policy** | **Linear / Nonlinear Stability** |
| — | — | — | — | — |
| First-Visit MC | $O(1)$ per step | $O(T)$ episode | On / Off (IS) | Unbiased; converges under coverage |
| $\mathrm{TD}(0)$ / SARSA | $O(d)$ | $O(d)$ | On-policy | Converges w.p.1 (linear, on-policy) |
| $\mathrm{TD}(\lambda)$ | $O(d)$ | $O(d)$ (trace $\boldsymbol{z}$) | On-policy | Converges w.p.1 (linear, on-policy) |
| Q-Learning | $O(\lvert \mathcal{A} \rvert d)$ | $O(d)$ | Off-policy | Diverges in general (Deadly Triad) |
| Gradient-TD (TDC) | $O(d)$ ($\approx 2\times$ TD) | $O(2d)$ ($\boldsymbol{w}, \boldsymbol{v}$) | Off-policy | Converges to $\mathrm{PBE}=0$ (linear) |
| Double DQN + PER | $O(B \cdot \text{fwd/bwd})$ | $O(N)$ buffer | Off-policy | Empirically stabilized via $\mathcal{D}, \boldsymbol{\theta}^-$ |

## Visual Diagnostics & Geometric Structure

Figure `fig:td:deadly_triad_and_dqn` maps the causal anatomy of the Deadly Triad against the stabilizing mechanisms of modern deep value-based RL, while Figure `fig:td:bias_variance_nstep` illustrates the empirical bias–variance trade-off across the $n$-step / $\lambda$ bootstrapping spectrum.

> [!example] Visual Diagram (LaTeX / PDF)
> *Anatomy of the Deadly Triad (left) and the complementary algorithmic stabilizers deployed in Gradient-TD and Rainbow DQN (right) [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 11, PDF p. 286) [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.5.2–2.5.4, pp. 42–48).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Bias–variance trade-off across $\lambda \in [0, 1]$ on the 19-state random walk benchmark (qualitative profile matching [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Fig. 12.8, PDF p. 321)). Intermediate $\lambda$ (or $n$-step horizons) consistently achieves the lowest RMS error.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Unbounded Variance in Ordinary Importance Sampling
> In off-policy MC control, when the behavior policy $b$ explores frequently while $\pi$ is deterministic, the trajectory ratio $\rho_{t:T-1}$ in `(eq:td:is_ratio)` can grow exponentially in episode length $T - t$, yielding infinite variance even when the estimator is unbiased [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 5, PDF p. 129). **Remedy:** Use weighted importance sampling, per-decision IS, or $n$-step Tree Backup / $Q(\sigma)$ methods that avoid IS ratios altogether [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 7, PDF p. 174).

> [!warning] Naive Residual-Gradient / Bellman Error Minimization
> Replacing semi-gradient TD with direct SGD on the Bellman Error $\mathrm{BE}(\boldsymbol{w}) = \lVert \overline{\delta}_{\boldsymbol{w}} \rVert_\mu^2$ fails in two ways: (1) sampling $s'$ once per step yields a biased gradient of $\mathbb{E}[R + \gamma V(S')]^2$ (the double-sampling problem), and (2) the Bellman Error itself is *not learnable* from observed feature-reward trajectories without knowing the underlying state transition probabilities [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 11, PDF pp. 291–298). **Remedy:** Minimize the *Projected* Bellman Error via Gradient-TD (TDC/GTD2) or Emphatic-TD.

> [!warning] Target Network Staleness vs. Loss of Plasticity in Deep Q-Learning
> Updating the frozen target $\boldsymbol{\theta}^-$ too infrequently slows value propagation across long horizons, whereas updating it too frequently reintroduces moving-target oscillations [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 19.4.1, p. 401). Furthermore, high update-to-data (UTD) ratios in sample-efficient DQN can induce dormant neurons and loss of plasticity [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.5.4.7, p. 49). **Remedy:** Use Polyak averaging ($\tau \approx 0.005$), Huber/clipped TD losses, layer normalization, and periodic soft parameter resets for high-UTD regimes.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[Deadly Triad]], [[MC & TD Learning]], [[Q-Learning & DQN]]
- **Prerequisites (Directed $\leftarrow$):** [[Bellman Equations]], [[MC & TD Learning]], [[Q-Learning & DQN]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Deadly Triad]], [[Generalized Advantage Estimation (GAE)]], [[Offline RL (CQL)]], [[Q-Learning & DQN]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Bellman Equations]], [[Deadly Triad]], [[Generalized Advantage Estimation (GAE)]], [[MC & TD Learning]], [[Offline RL (CQL)]], [[Q-Learning & DQN]]
- **Synthesized Sources:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]], [[Source - Understanding Deep Learning]]
