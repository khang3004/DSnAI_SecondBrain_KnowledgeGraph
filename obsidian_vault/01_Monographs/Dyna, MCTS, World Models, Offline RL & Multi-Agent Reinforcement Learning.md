---
type: monograph
domain: "D06"
parent_topic: "[[PT 06 - Policy Gradients, Model-Based RL, MARL & LLM Alignment]]"
sub_topic: "[[ST 06 - Policy Gradients, Actor-Critic, Model-Based RL & MARL]]"
cheatsheet: "[[Cheatsheet - Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]"
part_of: 
  - "[[ST 06 - Policy Gradients, Actor-Critic, Model-Based RL & MARL]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
concepts: 
  - "[[MCTS & MuZero]]"
  - "[[Model-Based RL (Dyna & MBPO)]]"
  - "[[Offline RL (CQL)]]"
prerequisites: 
  - "[[Markov Decision Process (MDP)]]"
  - "[[Model-Based RL (Dyna & MBPO)]]"
  - "[[Multi-Armed Bandits & UCB]]"
  - "[[Q-Learning & DQN]]"
enables: 
  - "[[MCTS & MuZero]]"
related: 
  - "[[Agent Planning & Multi-Agent Patterns]]"
  - "[[Chain-of-Thought & Test-Time Compute]]"
  - "[[MCTS & MuZero]]"
  - "[[Model-Based RL (Dyna & MBPO)]]"
  - "[[Multi-Armed Bandits & UCB]]"
  - "[[Q-Learning & DQN]]"
sources: 
  - "[[Source - Francis Bach]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Reinforcement Learning - An Overview]]"
tags:
  - monograph
  - domain-06
---

# Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning

> [!abstract] Monograph Navigation
> - **Domain:** [[D06 - Reinforcement Learning, Control & LLM Alignment|Reinforcement Learning, Control & LLM Alignment]]
> - **Parent / Sub-Topic:** [[PT 06 - Policy Gradients, Model-Based RL, MARL & LLM Alignment|Policy Gradients, Model-Based RL, MARL & LLM Alignment]] → [[ST 06 - Policy Gradients, Actor-Critic, Model-Based RL & MARL|Policy Gradients, Actor-Critic, Model-Based RL & MARL]]
> - **Visual Cheatsheet:** [[Cheatsheet - Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]
> - **Description:** Synthesizes Dyna planning/acting integration, Monte Carlo Tree Search (PUCT/AlphaZero), latent world models (Dreamer/MuZero) and compounding model error, successor representations, Offline RL conservatism (CQL/Decision Transformer), and Multi-Agent RL equilibria (Nash, CTDE, CFR).

## Problem Formulation & Unified Notation

Model-free reinforcement learning (RL) estimates value functions or policies directly from environment interactions, often requiring millions of samples 
 [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.4, p. 1164). Model-based RL (MBRL) improves sample efficiency by learning an internal approximation of the environment's transition dynamics and reward function, enabling *planning*—any computational process that takes a model as input and produces or improves a policy 
 [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 8, p. 160).

Consider a Markov Decision Process (MDP) $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, R, \gamma, \mu_0)$ with state space $\mathcal{S}$, action space $\mathcal{A}$, transition kernel $\mathcal{P}(s' \mid s, a)$, reward function $R(s, a, s')$, discount factor $\gamma \in [0, 1)$, and initial distribution $\mu_0$. A model of the environment can be either a *distribution model*, which outputs the full conditional probability $p(s', r \mid s, a)$, or a *sample model*, which generates a single draw $(\hat{s}', \hat{r}) \sim \hat{\mathcal{P}}(\cdot \mid s, a)$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 8, pp. 159–160). Planning splits into two regimes:
- **Background (offline) planning** (e.g., Dyna, Dreamer): uses simulated experience from a learned model $\hat{M}$ to update a global policy $\pi_\theta$ or value function $Q_w$ in the background [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 4.3, p. 93).
- **Decision-time (online) planning** (e.g., MCTS, MPC): waits until state $s_t$ is observed and searches forward over a finite horizon $H$ to select the immediate action $a_t$ 
 [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 8, p. 180), 
 [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.4.1, p. 1164).

When real-time interaction is prohibited, **offline RL** learns solely from a static dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^m$ collected by a behavior policy $\pi_b$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 7.7, p. 192), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.5, p. 1171). When $n \ge 2$ agents interact simultaneously, the MDP generalizes to a **Partially Observable Stochastic Game** (POSG) $(\mathcal{I}, \mathcal{S}, \{\mathcal{A}_i\}_{i=1}^n, \mathcal{P}, \{R_i\}_{i=1}^n, \{\mathcal{O}_i\}_{i=1}^n, \gamma)$ over joint actions $\boldsymbol{a} = (a_1, \dots, a_n)$ and agent-specific observation histories $h_t^i = \sigma_i(\hat{h}_t)$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 5.1–5.2, pp. 121, 127).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> | **Concept** | **Sutton & Barto (Ch. 8)** | **Murphy RL (Ch. 4–5, 7)** | **Murphy Adv. (Sec. 35.4–35.5)** |
> | — | — | — | — |
> | Dynamics / Model | $p(s', r \mid s, a)$ (distribution) vs. sample model | $T(s' \mid s, a)$, $M_w(z_t, a_t)$ (latent) | $p(s' \mid s, a)$, $R(s, a)$, $M_\theta$ |
> | Latent World State | Tabular state $s \in \mathcal{S}$ | Deterministic $h_t$, stochastic $z_t$ (Dreamer); $z_t = e_\phi(o_t)$ (MuZero) | Latent $z_t$ (VAE/RSSM), deterministic $h_t$ |
> | Behavior Policy | $b(a \mid s)$ (off-policy target $\pi$) | $\pi_b(a \mid s)$ or $\pi_\beta$ (offline dataset $\mathcal{D}$) | $\pi_b$ (behavior) vs. $\pi_e$ (evaluation) |
> | Planning Horizon | $n$-step lookahead / search depth | Receding horizon $H$, search budget $m$ | Planning horizon $H$, terminal $\hat{V}(s_{t+H})$ |
> | Multi-Agent History | Single-agent MDP (two-player zero-sum in MCTS) | Full history $\hat{h}_t$, agent history $h_t^i = \sigma_i(\hat{h}_t)$, opponents $\pi_{-i}$ | Single-agent focus; off-policy flow $d(s,a)$ |

## Core Mathematical Derivations & Theorems

### Dyna, Probabilistic Ensembles (PETS), and Compounding Error

> [!important] Dyna Integration, MPC Objective, and Compounding Model Error
> In tabular **Dyna-Q** [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 8, pp. 161–164), [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 4.3.2, p. 96), each real transition $(s, a, r, s')$ triggers three updates: (i) *direct RL* on $Q(s, a)$, (ii) *model learning* $\hat{M}(s, a) \leftarrow (s', r)$, and (iii) $N$ *planning updates* on simulated transitions $(\tilde{s}', \tilde{r}) \sim \hat{M}(\tilde{s}, \tilde{a})$ sampled from previously visited pairs (or prioritized by Bellman residual $|\delta|$ in *prioritized sweeping* [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 8, p. 168)).
>
> In continuous control with **Model Predictive Control (MPC)**, the agent solves at each step $t$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 35.46, p. 1164):
>
> $$
> a^*_{t:t+H-1} = \arg\max_{a_{t:t+H-1}} \mathbb{E}\!\left[\sum_{h=0}^{H-1} R(s_{t+h}, a_{t+h}) + \hat{V}(s_{t+H})\right],
>
> $$
>
> and executes only $a^*_t$. To prevent policy exploitation of overconfident point-estimate dynamics, **PETS** (Probabilistic Ensembles with Trajectory Sampling) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.4.4, p. 1168) models $p_\theta(s' \mid s, a)$ via an ensemble of $B$ probabilistic neural networks $\{\mathcal{N}(\mu_{\theta_b}(s, a), \Sigma_{\theta_b}(s, a))\}_{b=1}^B$, separating *aleatoric* uncertainty (intra-member variance $\Sigma_{\theta_b}$) from *epistemic* uncertainty (inter-member disagreement in $\mu_{\theta_b}$).
>
> When a learned model has bounded single-step total variation error $\sup_{s,a} D_{\mathrm{TV}}(\hat{\mathcal{P}}(\cdot \mid s,a)  \|  \mathcal{P}(\cdot \mid s,a)) \le \epsilon_m$ and the policy diverges from the data-collection policy by $\epsilon_\pi$, the return error scales quadratically with horizon:
>
> $$
> \lvert J_{\hat{\mathcal{M}}}(\pi) - J_{\mathcal{M}}(\pi) \rvert \le \frac{2 R_{\max}}{(1-\gamma)^2}\left[\epsilon_\pi + \frac{\gamma}{1-\gamma}\epsilon_m\right], \qquad
> \mathcal{E}_{\text{finite}}(H) = \mathcal{O}(H^2 \epsilon_m).
>
> $$

> [!tip] Why Truncated Rollouts (MBPO) Beat Full-Horizon Imagination
> Because single-step model errors compound quadratically as $\mathcal{O}(H^2 \epsilon_m)$ across long synthetic trajectories, rolling out an imperfect dynamics model from $s_0$ to termination $T$ injects severe bias. Branching short $k$-step model rollouts (e.g., $k \in [1, 5]$) from real states stored in the replay buffer $\mathcal{D}$—the Dyna/MBPO principle—caps model error at $\mathcal{O}(k^2 \epsilon_m)$ while still multiplying effective sample efficiency by $k$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 4.3.2, p. 96).

### Decision-Time Planning: MCTS, UCT, AlphaZero PUCT, and MuZero

Monte Carlo Tree Search (MCTS) builds an asymmetric search tree rooted at $s_t$ through $K$ simulations of four phases: *Selection*, *Expansion*, *Simulation/Evaluation*, and *Backup* 
 [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 8, pp. 185–187), 
 [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 4.2.2, pp. 86–88).

> [!important] UCT, AlphaZero PUCT, and MuZero Latent Search
> At internal tree nodes, **UCT** (Upper Confidence Bounds for Trees) selects actions via [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eq. 4.1, p. 86):
>
> $$
> a_t = \arg\max_{a \in \mathcal{A}(s)} \left[ Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}} \right], \qquad N(s) = \sum_{b} N(s, b).
>
> $$
>
> **AlphaZero** replaces the uniform UCB bonus with a learned neural policy prior $P(s, a) = \pi_\theta(a \mid s)$ via the **PUCT** rule and eliminates Monte Carlo rollouts by evaluating leaf nodes with a value head $V_\theta(s)$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eqs. 4.2–4.4, pp. 86–88):
>
> $$
> a_t = \arg\max_{a \in \mathcal{A}(s)} \left[ Q(s, a) + c_{\text{puct}}  P(s, a) \frac{\sqrt{N(s)}}{1 + N(s, a)} \right], \qquad
> Q(s, a) \leftarrow Q(s, a) + \frac{u - Q(s, a)}{N(s, a)}.
>
> $$
>
> After $K$ simulations, the root visit distribution $\pi^{\text{MCTS}}_s(a) \propto N(s, a)^{1/\tau}$ serves as the policy target, and the network $(V_\theta, \pi_\theta)$ minimizes:
>
> $$
> \mathcal{L}_{\text{AZ}}(\theta) = \mathbb{E}_{(s, \pi^{\text{MCTS}}_s, V^{\text{MCTS}}_s) \sim \mathcal{D}} \left[ \left(V^{\text{MCTS}}(s) - V_\theta(s)\right)^2 - \sum_{a} \pi^{\text{MCTS}}_s(a) \log \pi_\theta(a \mid s) \right].
>
> $$
>
> **MuZero** removes the requirement of a known simulator by learning an encoder $z_t = e_\phi(o_t)$ and latent dynamics/reward model $(z_{t+1}, \hat{r}_t) = M_w(z_t, a_t)$, jointly optimizing $(\theta, w, \phi)$ to match observed rewards $r_t$ alongside MCTS value and policy targets without reconstructing pixels [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eqs. 4.5–4.6, p. 88):
>
> $$
> \mathcal{L}_{\text{MZ}}(\theta, w, \phi) = \mathcal{L}_{\text{AZ}}(\theta, w, \phi) + \mathbb{E}_{\mathcal{D}}\!\left[\left(r_t - \hat{r}_t(e_\phi(o_t), a_t)\right)^2\right].
>
> $$

### Latent World Models (Dreamer RSSM) & Successor Features

Rather than predicting high-dimensional pixels directly or planning only at decision time, **Dreamer** learns a Recurrent State-Space Model (RSSM) combining a deterministic recurrent state $h_t = U(h_{t-1}, a_{t-1}, z_{t-1})$ with a stochastic latent $z_t$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 4.4.1.3, pp. 98–100), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.4.5, p. 1170).

> [!important] Dreamer RSSM ELBO, Imagination Actor-Critic, and Successor Features
> Given conditional prior $\hat{z}_t \sim P(\hat{z}_t \mid h_t)$, posterior encoder $z_t \sim E(z_t \mid h_t, o_t)$, decoder $\hat{o}_t \sim D(\hat{o}_t \mid h_t, z_t)$, and reward head $\hat{r}_t \sim R(\hat{r}_t \mid h_t, z_t)$, the world model minimizes the variational ELBO loss [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eqs. 4.21–4.24, p. 100):
>
> $$
> \mathcal{L}_{\text{WM}} = \mathbb{E}_{q(z_{1:T} \mid h_0, o_{1:T}, a_{1:T})}\!\left[\sum_{t=1}^T \left( -\beta_o \ln D(o_t \mid z_t, h_t) + \beta_z D_{\mathrm{KL}}(E(z_t \mid h_t, o_t) \parallel P(\hat{z}_t \mid h_t)) \right)\right].
>
> $$
>
> Inside latent imagination (unrolling $h_{t+1} = U(h_t, a_t, \hat{z}_t)$ without decoding pixels), the critic and actor optimize GAE $\lambda$-returns $G_t^\lambda = r_t + \gamma\left((1-\lambda)V(h_{t+1}) + \lambda G_{t+1}^\lambda\right)$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eqs. 4.25–4.27, p. 100):
>
> $$
> \mathcal{L}_{\text{critic}} = \sum_{t=1}^T \left(V(h_t) - \mathrm{sg}(G_t^\lambda)\right)^2, \qquad
> \mathcal{L}_{\text{actor}} = -\sum_{t=1}^T \mathrm{sg}\!\left(G_t^\lambda - V(h_t)\right)\log \pi(a_t \mid h_t).
>
> $$
>
> For fast transfer across tasks with linear rewards $R(s, \boldsymbol{w}) = \boldsymbol{\phi}(s)^{\top} \boldsymbol{w}$, **Successor Features (SFs)** decouple dynamics from reward via the discounted feature occupancy [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eqs. 4.60–4.71, pp. 115–116):
>
> $$
> \boldsymbol{\psi}^\pi(s, a) = \mathbb{E}_\pi\!\left[\sum_{t=0}^\infty \gamma^t \boldsymbol{\phi}(s_{t+1}) \mid s_0=s, a_0=a\right], \qquad
> Q^{\pi, \boldsymbol{w}}(s, a) = \boldsymbol{\psi}^\pi(s, a)^{\top} \boldsymbol{w}.
>
> $$
>
> Given a library of policies $\{\pi_i\}_{i=1}^N$ and a new task $\boldsymbol{w}_{\text{new}}$, **Generalized Policy Improvement (GPI)** selects $a^*(s; \boldsymbol{w}_{\text{new}}) = \arg\max_a \max_i \boldsymbol{\psi}^{\pi_i}(s, a)^{\top} \boldsymbol{w}_{\text{new}}$, guaranteeing $Q^{\pi}(s, a) \ge \max_i Q^{\pi_i}(s, a)$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 4.5.3.1, p. 116).

### Offline RL Conservatism & Off-Policy Importance Corrections

In offline RL, standard Q-learning suffers from *extrapolation error*: $\max_{a'} Q(s', a')$ queries out-of-distribution (OOD) actions whose overestimation bias cannot be corrected by online exploration [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 7.7.1, p. 193).

> [!important] CQL, Policy Constraints, Offline MBRL, and DualDICE
> **Conservative Q-Learning (CQL)** [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eqs. 7.56–7.59, p. 195) augments the Bellman error $\mathcal{E}(B, w)$ with a minimax penalty that pushes down $Q$-values under a high-value proposal $\mu(a \mid s)$ while pushing up $Q$-values on dataset actions $\pi_b(a \mid s)$:
>
> $$
> \mathcal{L}_{\text{CQL}}(w) = \alpha \left(\mathbb{E}_{s \sim \mathcal{D},  a \sim \mu(\cdot \mid s)}[Q_w(s, a)] - \mathbb{E}_{(s,a) \sim \mathcal{D}}[Q_w(s, a)] + \mathcal{R}(\mu)\right) + \frac{1}{2}\mathcal{E}(B, w).
>
> $$
>
> Setting $\mathcal{R}(\mu) = -D_{\mathrm{KL}}(\mu \parallel \rho)$ with uniform prior $\rho$ yields the soft log-sum-exp penalty $\log \sum_a \exp(Q_w(s, a))$. Alternatively:
> - **Implicit KL Constraints (AWR/AWAC)**: solve $\pi_{k+1}(a \mid s) \propto \pi_b(a \mid s) \exp(\frac{1}{\alpha} Q^{\pi_k}(s, a))$ and project via advantage-weighted regression [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eqs. 7.51–7.52, p. 194).
> - **Offline MBRL (MOPO/MOREL)**: penalize rewards by model uncertainty $u(s, a)$, $\tilde{R}(s, a) = R(s, a) - \lambda u(s, a)$, yielding a guaranteed lower bound on true return when $D(M_\theta, M^*) \le u(s, a)$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 7.7.2, p. 195).
> - **Sequence Modeling (Decision Transformer)**: autoregressively predicts actions conditioned on return-to-go $\mathrm{RTG}_t = \sum_{k=t}^T r_k$ via $\arg\max_\theta \mathbb{E}_{\mathcal{D}}[\log \pi_\theta(a_t \mid s_{0:t}, a_{0:t-1}, \mathrm{RTG}_{0:t})]$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eq. 7.60, p. 196).
>
> For off-policy evaluation without exponential trajectory importance products $\prod_t \frac{\pi_e(a_t \mid s_t)}{\pi_b(a_t \mid s_t)}$ (the *curse of horizon*), **DualDICE** estimates the marginalized stationary density ratio $\zeta^*(s, a) = p_\pi^\infty(s, a) / p_{\mathcal{D}}(s, a)$ via a convex-concave saddle point [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. 35.56–35.58, p. 1175):
>
> $$
> J(\pi) = \frac{1}{1-\gamma}\mathbb{E}_{(s,a)\sim p_{\mathcal{D}}}[\zeta^*(s, a) R(s, a)].
>
> $$

### Multi-Agent RL: Equilibria, CTDE (VDN/QMIX/MAPPO), and Regret

In a stochastic game, agent $i$'s expected return $U_i(\pi_i, \pi_{-i}) = \mathbb{E}_{\hat{h} \sim p(\cdot \mid \pi)}[\sum_{t=0}^\infty \gamma^t R_i(s_t, \boldsymbol{a}_t, s_{t+1})]$ depends on all opponents' policies $\pi_{-i}$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eqs. 5.11–5.13, p. 127). A joint policy $\pi^*$ is a **Nash Equilibrium (NE)** iff no agent can unilaterally improve: $\forall i, \pi'_i,  U_i(\pi'_i, \pi^*_{-i}) \le U_i(\pi^*)$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eq. 5.22, p. 128). In two-player zero-sum games, closeness to minimax NE is measured by *exploitability*: $\frac{1}{2}[\max_{\pi'_1} J(\pi'_1, \pi_2) - \min_{\pi'_2} J(\pi_1, \pi'_2)]$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eq. 5.21, p. 128).

> [!important] CTDE Value Decomposition (VDN/QMIX), Multi-Agent Actor-Critic, and MMDS
> Under **Centralized Training with Decentralized Execution (CTDE)**, agents condition execution on local histories $h_t^i$ while training critics with global state/central information $c_t$. For cooperative common-reward games, value factorization requires the **Individual-Global-Max (IGM)** condition [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eq. 5.45, p. 136):
>
> $$
> \arg\max_{\boldsymbol{a}} Q_{\text{tot}}(\boldsymbol{h}_t, c_t, \boldsymbol{a}; \theta) = \left(\arg\max_{a^1} Q_1(h_t^1, a^1; \theta_1), \dots, \arg\max_{a^n} Q_n(h_t^n, a^n; \theta_n)\right).
>
> $$
>
> **VDN** enforces IGM via linear additivity $Q_{\text{tot}} = \sum_{i=1}^n Q_i(h_t^i, a_t^i; \theta_i)$, whereas **QMIX** uses a hypernetwork-parameterized monotonic mixing network $Q_{\text{tot}} = f_{\text{mix}}(Q_1, \dots, Q_n; c_t)$ with non-negative weights $\frac{\partial f_{\text{mix}}}{\partial Q_i} \ge 0$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eqs. 5.46–5.49, p. 137).
>
> In policy-gradient MARL (**MAA2C / MAPPO**), the multi-agent policy gradient uses centralized critics $Q_i^\pi(\hat{h}, \boldsymbol{a})$ or $V_i^\pi(\hat{h})$ to reduce variance [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eqs. 5.39–5.40, p. 134):
>
> $$
> \nabla_{\theta_i} J(\theta_{1:n}) \propto \mathbb{E}_{\hat{h} \sim p(\hat{h} \mid \pi),  \boldsymbol{a} \sim \pi}\!\left[\left(Q_i^\pi(\hat{h}, \boldsymbol{a}) - V_i^\pi(\hat{h})\right) \nabla_{\theta_i} \log \pi_i(a_i \mid h_i; \theta_i)\right].
>
> $$
>
> For imperfect-information games, **Magnetic Mirror Descent Search (MMDS)** and Counterfactual Regret Minimization stabilize self-play toward NE via regularized local policy updates toward a magnet $\rho$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eq. 5.70, p. 143):
>
> $$
> \pi_{\text{new}} = \arg\max_{\pi' \in \Delta(\mathcal{A})} \left[ \langle \pi', q \rangle - \alpha D_{\mathrm{KL}}(\pi' \parallel \rho) - \frac{1}{\eta}D_{\mathrm{KL}}(\pi' \parallel \pi) \right].
>
> $$

Beyond flat single-step MDPs, **Hierarchical RL (Options)** models temporally extended macro-actions $\omega = (\mathcal{I}_\omega, \pi_\omega, \beta_\omega)$ over semi-MDPs [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 7.5.1, p. 184), while **MaxEnt Inverse RL** recovers reward functions $R_\theta$ from expert demonstrations $\mathcal{D}_{\text{exp}}$ via energy-based trajectory likelihood $p(\tau) \propto \exp(\sum_{t=0}^{T-1} R_\theta(s_t, a_t))$ 
 [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Eqs. 7.44–7.45, p. 191).

## Algorithmic Implementation & Comparative Synthesis

Algorithm `alg:mbrl:unified` unifies decision-time MCTS (AlphaZero/MuZero) with background Dyna/MBPO rollouts.

> [!example] Algorithm: Unified AlphaZero/MuZero MCTS & Dyna-MBPO Planning Loop
> **Input:** Initial state/obs $o_0$, representation $e_\phi$, dynamics/reward model $M_w$, policy/value $(P_\theta, V_\theta)$, MCTS budget $K$, Dyna rollouts $N$, horizon $k_{\text{branch}}$
> **for** $t = 0, 1, 2, \dots$:
>
>   Encode root latent $z_t \leftarrow e_\phi(o_t)$  *// Identity $e_\phi(o_t)=s_t$ in AlphaZero*
>   **for** $m = 1, \dots, K$  *// Decision-time MCTS*:
>
>     Initialize path $\tau \leftarrow [ ]$, $z \leftarrow z_t$
>     **while** $z$ is expanded in tree:
>
>       Select $a \leftarrow \arg\max_{a'} \left[Q(z, a') + c_{\text{puct}} P_\theta(a' \mid z)\frac{\sqrt{N(z)}}{1 + N(z, a')}\right]$
>       Simulate $(z', \hat{r}) \leftarrow M_w(z, a)$; append $(z, a, \hat{r})$ to $\tau$; $z \leftarrow z'$
>
>     Expand leaf $z$ with prior $P_\theta(\cdot \mid z)$ and evaluate $u \leftarrow V_\theta(z)$
>     **for** $(z_j, a_j, \hat{r}_j) \in \mathrm{reverse}(\tau)$  *// Backup*:
>
>       $u \leftarrow \hat{r}_j + \gamma u$; \quad $N(z_j, a_j) \leftarrow N(z_j, a_j) + 1$
>       $Q(z_j, a_j) \leftarrow Q(z_j, a_j) + \frac{u - Q(z_j, a_j)}{N(z_j, a_j)}$
>
>
>   Sample $a_t \sim \pi^{\text{MCTS}}_{z_t}(a) \propto N(z_t, a)^{1/\tau}$; execute $a_t$, observe $(r_t, o_{t+1})$, store in $\mathcal{D}$
>   **for** $n = 1, \dots, N$  *// Background Dyna / MBPO truncated rollouts*:
>
>     Sample real state $o \sim \mathcal{D}$, encode $\tilde{z}_0 = e_\phi(o)$, unroll $k_{\text{branch}}$ steps via $M_w$ and $\pi_\theta$
>     Update $(\theta, w, \phi)$ via Eq. `(eq:mbrl:muzero_loss)` or Dreamer actor-critic Eqs. `(eq:mbrl:dreamer_wm)`–`(eq:mbrl:dreamer_ac)`

**Table:** Comparative synthesis across model-free, Dyna/MBPO, MCTS, latent world models, and offline RL.

| **Paradigm** | **Planning Mode** | **Sample Eff.** | **Compute** | **Primary Bottleneck** |
| — | — | — | — | — |
| Model-Free (DQN/PPO) | None | Low ($\mathcal{O}(10^7)$) | $\mathcal{O}(1)$ forward | Sample complexity |
| Dyna-Q / MBPO | Background ($k$-step) | High | $\mathcal{O}(1)$ forward | Compounding $\mathcal{O}(k^2\epsilon_m)$ |
| MCTS (AlphaZero/MuZero) | Decision-time (tree) | High (self-play) | $\mathcal{O}(K \cdot d)$ search | Discrete/sampled branching |
| Latent WM (Dreamer) | Background (latent) | Very High | $\mathcal{O}(1)$ policy | Obs. recon. / KL collapse |
| Offline RL (CQL/MOPO) | Static dataset $\mathcal{D}$ | Fixed ($0$ online) | $\mathcal{O}(1)$ forward | OOD extrapolation |

> [!info] Cross-Book Synthesis: Planning, Representation, and Distribution Shift
> Sutton & Barto [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 8, pp. 161, 188) frame model-based and model-free RL as endpoints of a single spectrum: both compute backed-up value targets, differing only in whether experience is real or simulated. Murphy [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 4.4.2, p. 102) highlights the *objective mismatch problem* of pixel-reconstruction world models—maximizing observation likelihood wastes capacity on task-irrelevant distractors—motivating value-equivalent latent models like MuZero [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 4.2.2.2, p. 88) and non-generative self-prediction losses. Finally, Murphy Advanced Topics [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.4.6–35.5, pp. 1171–1175) unifies model error in MBRL with covariate shift in off-policy/offline RL: in both settings, policy improvement shifts the state-action visitation distribution away from the training support $\mathcal{D}$, requiring either epistemic uncertainty penalties (PETS/MOPO) or conservative density-ratio constraints (CQL/DualDICE).

## Visual Diagnostics & Geometric Structure

> [!note] Source Figure
> *Figure:* The general Dyna architecture integrating real experience (direct RL and model learning) with simulated experience (planning updates) [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 8, Fig. 8.1, p. 163). (`reinforcement_learning_introduction_richard_s_su_p185_fig1.png`)

> [!note] Source Figure
> *Figure:* Dreamer RSSM world model factor graph (1-step unroll): deterministic state $h_t$, conditional prior $\hat{z}_t \sim P_0(\cdot \mid h_t)$, posterior encoder $z_t \sim E(\cdot \mid h_t, o_t)$, decoder $D$, and KL/reconstruction losses [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Ch. 4, Fig. 4.5, p. 99). (`reinforement_learning_kevin_murphy_p99_fig1.png`)

> [!note] Source Figure
> *Figure:* Comparison of (a) online on-policy RL, (b) online off-policy RL with replay buffer $\mathcal{D}$, and (c) offline RL with static dataset $\mathcal{D}$ collected by $\pi_\beta$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 7.7, Fig. 7.6, p. 192). (`reinforement_learning_kevin_murphy_p192_fig1.png`)

Figure `fig:mbrl:dyna_world_model` synthesizes the background Dyna/Dreamer latent imagination loop alongside the four-phase decision-time MCTS/MuZero search cycle, while Figure `fig:mbrl:compounding_error` quantifies the $\mathcal{O}(H^2 \epsilon_m)$ compounding error trade-off that motivates truncated rollouts.

> [!example] Visual Diagram (LaTeX / PDF)
> *Unified architecture connecting background planning (Dyna / Dreamer RSSM latent imagination, left) with decision-time planning (4-phase MCTS / MuZero loop, right).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Trade-off between compounding model error $\mathcal{O}(H^2 \epsilon_m)$ and value-function bias reduction across rollout horizon $H$, illustrating why truncated Dyna/MBPO branches minimize total return error.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] The Deadly Triad & Off-Policy Divergence (Baird's Counterexample)
> Combining *function approximation*, *bootstrapping*, and *off-policy* updates—the **deadly triad**—can cause TD(0) and Q-learning weights to diverge to $\infty$ even on a 7-state deterministic MDP with zero rewards and $\gamma = 0.99$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.5.3, Fig. 35.10, p. 1176).
> **Remedy:** Use target networks, gradient-TD / Emphatic-TD corrections, or behavior-regularized offline updates (CQL/IQL).

> [!warning] Model Exploitation, Objective Mismatch & Environment Non-Stationarity
> If a learned model $\hat{M}$ is inaccurate or the environment changes (e.g., blocking/shortcut maze tasks), greedy planning exploits phantom high-reward transitions [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 8, pp. 166–168). Similarly, generative pixel-reconstruction models suffer from *objective mismatch* when capacity is spent on irrelevant distractors [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 4.4.2.1, p. 102).
> **Remedy:** Incorporate exploration bonuses on state-action staleness (Dyna-Q+), uncertainty penalties via PETS ensembles or MOPO ($\tilde{R} = R - \lambda u$), or value-equivalent latent losses (MuZero).

> [!warning] Offline Extrapolation Error & Stochastic Return-Conditioning Bias
> Unconstrained off-policy Q-learning on a static dataset $\mathcal{D}$ overestimates OOD actions; conversely, return-conditioned sequence models (Decision Transformer) suffer from *optimism bias* in stochastic MDPs, mistaking lucky high-return trajectories for optimal actions [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 7.7.1 & 7.7.3, pp. 193, 196).
> **Remedy:** Enforce pessimistic value penalties (CQL, Eq. `(eq:mbrl:cql)`), support constraints (MMD/AWAC), or condition on learned expectiles / latent plans rather than raw RTG.

> [!warning] MARL Non-Stationarity & Independent Learning Oscillation
> In multi-agent environments, simultaneous policy updates render the effective transition dynamics non-stationary from each agent's local perspective, causing independent gradient ascent (IGA) or IPPO to cycle indefinitely (e.g., in Rock-Paper-Scissors) [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 5.3.2.4, p. 135).
> **Remedy:** Use CTDE critics (QMIX/MAPPO), two-timescale learning rates (WoLF-PHC), or regularized self-play (MMDS/CFR, Eq. `(eq:mbrl:mmds)`) with historical opponent pools.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[MCTS & MuZero]], [[Model-Based RL (Dyna & MBPO)]], [[Offline RL (CQL)]]
- **Prerequisites (Directed $\leftarrow$):** [[Markov Decision Process (MDP)]], [[Model-Based RL (Dyna & MBPO)]], [[Multi-Armed Bandits & UCB]], [[Q-Learning & DQN]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[MCTS & MuZero]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Agent Planning & Multi-Agent Patterns]], [[Chain-of-Thought & Test-Time Compute]], [[MCTS & MuZero]], [[Model-Based RL (Dyna & MBPO)]], [[Multi-Armed Bandits & UCB]], [[Q-Learning & DQN]]
- **Synthesized Sources:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]]
