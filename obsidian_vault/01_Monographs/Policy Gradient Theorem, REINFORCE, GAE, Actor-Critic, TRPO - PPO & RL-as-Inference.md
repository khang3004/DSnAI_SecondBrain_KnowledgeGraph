---
type: monograph
domain: "D06"
parent_topic: "[[PT 06 - Policy Gradients, Model-Based RL, MARL & LLM Alignment]]"
sub_topic: "[[ST 06 - Policy Gradients, Actor-Critic, Model-Based RL & MARL]]"
cheatsheet: "[[Cheatsheet - Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]"
part_of: 
  - "[[ST 06 - Policy Gradients, Actor-Critic, Model-Based RL & MARL]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
concepts: 
  - "[[GRPO]]"
  - "[[Generalized Advantage Estimation (GAE)]]"
  - "[[PPO (Proximal Policy Optimization)]]"
  - "[[Policy Gradient Theorem]]"
  - "[[TRPO]]"
prerequisites: 
  - "[[Generalized Advantage Estimation (GAE)]]"
  - "[[KL Divergence]]"
  - "[[Lagrange Duality & KKT]]"
  - "[[MC & TD Learning]]"
  - "[[Markov Decision Process (MDP)]]"
  - "[[Natural Gradient & K-FAC]]"
  - "[[PPO (Proximal Policy Optimization)]]"
  - "[[Policy Gradient Theorem]]"
  - "[[TRPO]]"
enables: 
  - "[[GRPO]]"
  - "[[Generalized Advantage Estimation (GAE)]]"
  - "[[PPO (Proximal Policy Optimization)]]"
  - "[[RLHF (KL-Regularized)]]"
  - "[[RLVR (Verifiable Rewards)]]"
  - "[[TRPO]]"
related: 
  - "[[GRPO]]"
  - "[[Generalized Advantage Estimation (GAE)]]"
  - "[[KL Divergence]]"
  - "[[Lagrange Duality & KKT]]"
  - "[[MC & TD Learning]]"
  - "[[Natural Gradient & K-FAC]]"
  - "[[PPO (Proximal Policy Optimization)]]"
  - "[[Policy Gradient Theorem]]"
  - "[[RLHF (KL-Regularized)]]"
  - "[[RLVR (Verifiable Rewards)]]"
  - "[[Reparameterization Trick]]"
  - "[[TRPO]]"
sources: 
  - "[[Source - Francis Bach]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Reinforcement Learning - An Overview]]"
  - "[[Source - Reinforcement Learning from Human Feedback]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-06
---

# Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO/PPO & RL-as-Inference

> [!abstract] Monograph Navigation
> - **Domain:** [[D06 - Reinforcement Learning, Control & LLM Alignment|Reinforcement Learning, Control & LLM Alignment]]
> - **Parent / Sub-Topic:** [[PT 06 - Policy Gradients, Model-Based RL, MARL & LLM Alignment|Policy Gradients, Model-Based RL, MARL & LLM Alignment]] → [[ST 06 - Policy Gradients, Actor-Critic, Model-Based RL & MARL|Policy Gradients, Actor-Critic, Model-Based RL & MARL]]
> - **Visual Cheatsheet:** [[Cheatsheet - Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
> - **Description:** Proves the Policy Gradient Theorem, derives baseline variance reduction, Generalized Advantage Estimation (GAE(gamma, lambda)), Natural Policy Gradient / TRPO monotonic improvement lower bound, PPO clipped surrogate, SAC maximum-entropy RL, and Control-as-Inference.

## Problem Formulation & Unified Notation

Value-based reinforcement learning methods (such as Q-learning and DQN) derive a policy indirectly by estimating action-values and acting greedily or $\varepsilon$-greedily with respect to them. Consequently, small perturbations in estimated values can cause discontinuous jumps in the induced policy, and continuous or high-dimensional action spaces require intractable inner maximizations [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 13, p. 324) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 19.5, p. 388). Policy-based methods instead parameterize a stochastic policy $\pi_{\boldsymbol{\theta}}(a \mid s)$ directly with parameters $\boldsymbol{\theta} \in \mathbb{R}^{d_\theta}$ and optimize a scalar performance objective $J(\boldsymbol{\theta})$ via stochastic gradient ascent.

Consider a Markov Decision Process (MDP) $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, R, d_0, \gamma)$ with state space $\mathcal{S}$, action space $\mathcal{A}$, transition kernel $\mathcal{P}(s' \mid s, a)$, reward function $r_t = R(s_t, a_t)$, initial state distribution $d_0(s_0)$, and discount factor $\gamma \in [0, 1]$. A trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T)$ generated under $\pi_{\boldsymbol{\theta}}$ has probability density

$$
p_{\boldsymbol{\theta}}(\tau) = d_0(s_0) \prod_{t=0}^{T-1} \pi_{\boldsymbol{\theta}}(a_t \mid s_t)  \mathcal{P}(s_{t+1} \mid s_t, a_t).
$$

The discounted return from step $t$ is $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$. In the episodic setting, the performance objective is $J(\boldsymbol{\theta}) = \mathbb{E}_{s_0 \sim d_0}[ V^{\pi_{\boldsymbol{\theta}}}(s_0)] = \mathbb{E}_{\tau \sim p_{\boldsymbol{\theta}}}[G_0]$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.3.1, p. 1157). In the continuing (average-reward) setting, $J(\boldsymbol{\theta}) = r(\pi_{\boldsymbol{\theta}}) = \lim_{h \to \infty} \frac{1}{h} \sum_{t=1}^h \mathbb{E}[R_t \mid S_0, A_{0:t-1} \sim \pi_{\boldsymbol{\theta}}]$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 13.6, p. 333).

In autoregressive Large Language Model (LLM) post-training (RLHF and RL with Verifiable Rewards), a prompt $x \sim \mathcal{D}$ serves as the initial state $s_0$, and the completion $y = (a_0, \dots, a_{T-1})$ is a sequence of tokens where state $s_t = (x, a_{<t})$ evolves deterministically by concatenation, $\gamma = 1$, and a scalar reward $R(x, y)$ is typically observed only at the terminal token $T$ [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Sec. 6.2, p. 52–53).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> | Concept | Sutton & Barto | Murphy (RL / Adv.) | Prince (UDL) | Lambert (RLHF) |
> | — | — | — | — | — |
> | Policy | $\pi(a \mid s, \boldsymbol{\theta})$ | $\pi_{\boldsymbol{\theta}}(a \mid s)$ | $\pi[a_t \mid s_t, \theta]$ | $\pi_\theta(a \mid s)$ or $\pi_\theta(y \mid x)$ |
> | Critic / Value | $\hat{v}(s, \mathbf{w})$ | $V_{\mathbf{w}}(s)$ / $V_\phi(s)$ | $v[s, \phi]$ | $V_\phi(s_t)$ |
> | State Visitation | $\mu(s)$ (steady/on-policy) | $p_\pi^\gamma(s)$ (discounted) | Implicit in $\mathbb{E}_\tau[\cdot]$ | Rollout batch $s \sim \mathcal{D}$ |
> | Return | $G_t = \sum_{k=t+1}^T \gamma^{k-t-1} R_k$ | $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$ | $r[\tau] = \sum_{t} r_t$ | $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$ |
> | IS Policy Ratio | N/A (on-policy) | $\rho_t = \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\mathrm{old}}}(a_t \mid s_t)}$ | N/A | $\rho(\theta) = \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\mathrm{old}}}(a \mid s)}$ |
> | KL Reg. | N/A | Trust-region constraint | N/A | Reward shaping or loss |
>
>
>
> **Unified Convention:** Throughout this leaf, we adopt $\pi_{\boldsymbol{\theta}}(a \mid s)$ for the actor, $V_{\boldsymbol{\phi}}(s)$ for the critic, $d^{\pi}_{\gamma}(s) = (1-\gamma)\sum_{t=0}^\infty \gamma^t \mathbb{P}(S_t = s \mid \pi)$ for the normalized discounted state visitation distribution, and $r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\mathrm{old}}}(a_t \mid s_t)}$ for the importance-sampling (IS) ratio.

## Core Mathematical Derivations & Theorems

### The Policy Gradient Theorem (Episodic & Continuing)

A central difficulty in differentiating $J(\boldsymbol{\theta}) = \mathbb{E}_{s_0}[ V^{\pi_{\boldsymbol{\theta}}}(s_0)]$ is that $\boldsymbol{\theta}$ affects both the action selection probabilities $\pi_{\boldsymbol{\theta}}(a \mid s)$ and the state visitation distribution $d^{\pi_{\boldsymbol{\theta}}}_\gamma(s)$ through unknown environment dynamics $\mathcal{P}(s' \mid s, a)$. The Policy Gradient Theorem eliminates the derivative of the state distribution [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 13.2, p. 324–325).

> [!important] Theorem: Policy Gradient Theorem (Episodic & Continuing)
> For any differentiable policy $\pi_{\boldsymbol{\theta}}(a \mid s)$ in an episodic MDP (with $\gamma \in (0, 1]$) or an ergodic continuing MDP, the gradient of the performance objective $J(\boldsymbol{\theta})$ is
>
> $$
> \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \propto \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}}  Q^{\pi_{\boldsymbol{\theta}}}(s, a)  \nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a \mid s)
>   = \mathbb{E}_{s \sim \mu,  a \sim \pi_{\boldsymbol{\theta}}}\!\left[  Q^{\pi_{\boldsymbol{\theta}}}(s, a)  \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s) \right],
> $$
>
> where $\mu(s)$ is the discounted state-visitation measure $\sum_{t=0}^\infty \gamma^t \mathbb{P}(s_0 \to s, t, \pi_{\boldsymbol{\theta}})$ (or normalized $d^{\pi_{\boldsymbol{\theta}}}_\gamma(s)$ with proportionality constant $(1-\gamma)^{-1}$) in the episodic case, and the stationary distribution $\mu^{\pi_{\boldsymbol{\theta}}}(s)$ in the continuing case.
>
>
> **Proof (Episodic Case via Unrolling):** Writing $\nabla$ for $\nabla_{\boldsymbol{\theta}}$ and omitting the subscript $\boldsymbol{\theta}$ on $\pi$ for brevity [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (p. 325) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.3.1, p. 1157):
>
> $$
> \begin{aligned}
> \nabla  V^\pi(s) &= \nabla \left[ \sum_{a} \pi(a \mid s)   Q^\pi(s, a) \right]  \\
>   &= \sum_{a} \Big[ \nabla\pi(a \mid s)   Q^\pi(s, a) + \pi(a \mid s)  \nabla  Q^\pi(s, a) \Big]  \\
>   &= \sum_{a} \nabla\pi(a \mid s)   Q^\pi(s, a) + \sum_{a} \pi(a \mid s) \nabla \left[ R(s,a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a)   V^\pi(s') \right]  \\
>   &= \sum_{a} \nabla\pi(a \mid s)   Q^\pi(s, a) + \gamma \sum_{a} \pi(a \mid s) \sum_{s'} \mathcal{P}(s' \mid s, a)  \nabla  V^\pi(s').
> \end{aligned}
> $$
>
> Notice that $\nabla  V^\pi(s')$ on the right-hand side has the exact same form as $\nabla  V^\pi(s)$. Repeatedly unrolling `(eq:pg:unroll_step)` yields
>
> $$
> \nabla  V^\pi(s) = \sum_{x \in \mathcal{S}} \sum_{k=0}^\infty \gamma^k \mathbb{P}(s \to x, k, \pi) \sum_{a} \nabla \pi(a \mid x)   Q^\pi(x, a).
> $$
>
> Taking expectation over $s_0 \sim d_0$ and applying the log-derivative trick $\nabla \pi(a \mid x) = \pi(a \mid x)  \nabla \log \pi(a \mid x)$ gives `(eq:pg:pg_theorem)`.
>
>
> **Continuing Case:** With $J(\boldsymbol{\theta}) = r(\pi) = \sum_s \mu(s) \sum_a \pi(a \mid s) R(s, a)$ and differential value $V^\pi(s) = \sum_a \pi(a \mid s)\big(R(s,a) - r(\pi) + \sum_{s'} \mathcal{P}(s'\mid s,a) V^\pi(s')\big)$, differentiating $V^\pi(s)$ and summing weighted by $\mu(s)$ cancels $\sum_s \mu(s)\nabla  V^\pi(s)$ against $\sum_{s'} \mu(s')\nabla  V^\pi(s')$ via stationarity $\sum_s \mu(s)\sum_a \pi(a\mid s)\mathcal{P}(s'\mid s,a) = \mu(s')$, leaving $\nabla r(\pi) = \sum_s \mu(s)\sum_a \nabla \pi(a \mid s)   Q^\pi(s, a)$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 13.6, p. 334–335).

> [!tip] Why $\nabla_{\boldsymbol{\theta}} d^\pi(s)$ Disappears
> Because environment dynamics $\mathcal{P}(s' \mid s, a)$ do not depend on $\boldsymbol{\theta}$, every change in future state occupancy induced by perturbing $\boldsymbol{\theta}$ propagates through $\nabla  V^\pi(s')$. Unrolling the Bellman recursion absorbs all downstream visitation shifts into the discounted occupancy measure $\mu(s)$, so we only need on-policy samples $s \sim \mu$ and the score function $\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s)$.

### Score-Function Gradient, REINFORCE & Baseline Invariance

An alternative trajectory-level derivation differentiates $J(\boldsymbol{\theta}) = \int p_{\boldsymbol{\theta}}(\tau) R(\tau) \mathrm{d}\tau$ directly ([[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 19.5.1, p. 389); [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Sec. 6.2.1, p. 54)):

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim p_{\boldsymbol{\theta}}}\!\left[ R(\tau)  \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\tau) \right]
  = \mathbb{E}_{\tau \sim p_{\boldsymbol{\theta}}}\!\left[ \left(\sum_{t=0}^{T-1} \gamma^t r_t\right) \sum_{t'=0}^{T-1} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_{t'} \mid s_{t'}) \right],
$$

because $\nabla_{\boldsymbol{\theta}} \log d_0(s_0) = \boldsymbol{0}$ and $\nabla_{\boldsymbol{\theta}} \log \mathcal{P}(s_{t+1} \mid s_t, a_t) = \boldsymbol{0}$. By causality (actions at time $t$ cannot affect rewards at $t' < t$), terms with $t' > t$ have zero expectation, reducing `(eq:pg:score_fn)` to the reward-to-go estimator $\mathbb{E}_\tau\!\left[\sum_{t=0}^{T-1} \gamma^t G_t  \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)\right]$, which defines the Monte Carlo **REINFORCE** update [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 3.1.2–3.1.3, p. 52–53).

> [!important] Theorem: Baseline Invariance & Optimal Control Variate
> Subtracting any state-dependent (action-independent) baseline $b(s)$ from $G_t$ or $Q^\pi(s, a)$ leaves the expected policy gradient strictly unbiased [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 13.4, p. 329) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 19.5.3, p. 392):
>
> $$
> \begin{aligned}
> \mathbb{E}_{a \sim \pi_{\boldsymbol{\theta}}(\cdot \mid s)}\!\left[ b(s)  \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s) \right]
>   &= b(s) \sum_{a \in \mathcal{A}} \pi_{\boldsymbol{\theta}}(a \mid s) \frac{\nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a \mid s)}{\pi_{\boldsymbol{\theta}}(a \mid s)}  \\
>   &= b(s)  \nabla_{\boldsymbol{\theta}} \sum_{a \in \mathcal{A}} \pi_{\boldsymbol{\theta}}(a \mid s) = b(s)  \nabla_{\boldsymbol{\theta}}(1) = \boldsymbol{0}.
> \end{aligned}
> $$
>
> Minimizing $\mathrm{Var}\!\left[(G_t - b(s_t))\nabla_{\boldsymbol{\theta}}\log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)\right]$ with respect to $b(s_t)$ yields the variance-optimal baseline [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Eq. 19.36, p. 392):
>
> $$
> b^*(s_t) = \frac{\mathbb{E}_{a_t \sim \pi_{\boldsymbol{\theta}}}\!\left[ G_t  \lVert \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \rVert_2^2 \right]}{\mathbb{E}_{a_t \sim \pi_{\boldsymbol{\theta}}}\!\left[ \lVert \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \rVert_2^2 \right]}.
> $$
>
> In practice, $b(s_t) \approx  V_{\boldsymbol{\phi}}(s_t)$ is used, turning the weight into an estimate of the **advantage function** $A^\pi(s, a) =  Q^\pi(s, a) -  V^\pi(s)$.

### Generalized Advantage Estimation ($\mathrm{GAE (\gamma, \lambda)$)}

Using the full Monte Carlo return $G_t -  V_{\boldsymbol{\phi}}(s_t)$ is unbiased when $V_{\boldsymbol{\phi}}$ is inaccurate, but suffers high variance. Using a one-step TD residual $\delta_t^V = r_t + \gamma  V_{\boldsymbol{\phi}}(s_{t+1}) -  V_{\boldsymbol{\phi}}(s_t)$ (as in one-step A2C [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 13.5, p. 331)) has low variance but introduces bias whenever $V_{\boldsymbol{\phi}} \ne  V^\pi$. **Generalized Advantage Estimation (GAE)** interpolates smoothly between these extremes [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 3.2.2, p. 57).

> [!important] Derivation of $\mathrm{GAE}(\gamma, \lambda)$
> Define the $k$-step truncated advantage estimator $\hat{A}_t^{(k)} = G_{t:t+k} -  V_{\boldsymbol{\phi}}(s_t) = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V$, where $\delta_{t+l}^V = r_{t+l} + \gamma  V_{\boldsymbol{\phi}}(s_{t+l+1}) -  V_{\boldsymbol{\phi}}(s_{t+l})$ telescopes cleanly. Taking an exponentially weighted average with weights $(1 - \lambda)\lambda^{k-1}$ for $\lambda \in [0, 1]$ gives:
>
> $$
> \begin{aligned}
> \hat{A}_t^{\mathrm{GAE}(\gamma, \lambda)} &= (1 - \lambda) \sum_{k=1}^\infty \lambda^{k-1} \hat{A}_t^{(k)}
>   = (1 - \lambda) \sum_{k=1}^\infty \lambda^{k-1} \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V  \\
>   &= (1 - \lambda) \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V \sum_{k=l+1}^\infty \lambda^{k-1}
>   = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V.
> \end{aligned}
> $$
>
> Setting $\lambda = 0$ recovers the one-step TD advantage $\delta_t^V$ (low variance, higher bias); setting $\lambda = 1$ recovers the Monte Carlo advantage $\sum_{l=0}^\infty \gamma^l \delta_{t+l}^V = G_t -  V_{\boldsymbol{\phi}}(s_t)$ (unbiased, higher variance). Over a finite horizon $T$, `(eq:pg:gae)` is computed via the backward recursion $\hat{A}_t = \delta_t^V + \gamma \lambda \hat{A}_{t+1}$ with $\hat{A}_T = 0$.

### Natural Policy Gradient, Monotonic Improvement & TRPO/PPO

Standard gradient ascent $\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + \eta \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_k)$ measures step size in Euclidean parameter space $\lVert \Delta \boldsymbol{\theta} \rVert_2$, which is not invariant to reparameterization: an identical $\Delta \boldsymbol{\theta}$ can induce an arbitrarily large shift in the action distribution $\pi_{\boldsymbol{\theta}}(\cdot \mid s)$ when variance is small ([[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 3.2.4.1, p. 58–59); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. 35.36, p. 1161)). **Natural Policy Gradient (NPG)** preconditions the gradient by the inverse Fisher Information Matrix (FIM)

$$
F(\boldsymbol{\theta}) = \mathbb{E}_{s \sim d^{\pi_{\boldsymbol{\theta}}}_\gamma,  a \sim \pi_{\boldsymbol{\theta}}}\!\left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s)  \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s)^{\top} \right],
  \qquad
  \Delta \boldsymbol{\theta}_{\mathrm{NPG}} \propto F(\boldsymbol{\theta})^{-1} \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}),
$$

which corresponds to steepest ascent under the local second-order KL metric
$\mathbb{E}_s[D_{\mathrm{KL}}(\pi_{\boldsymbol{\theta}}(\cdot \mid s) \parallel \pi_{\boldsymbol{\theta}+\Delta\boldsymbol{\theta}}(\cdot \mid s))] \approx \frac{1}{2}\Delta\boldsymbol{\theta}^{\top} F(\boldsymbol{\theta})\Delta\boldsymbol{\theta}$.

> [!important] Monotonic Policy Improvement Bound (TRPO) & PPO Clipped Surrogate
> Let $\pi_k = \pi_{\boldsymbol{\theta}_k}$ and let $\pi$ be a candidate policy. Using the performance difference identity $J(\pi) - J(\pi_k) = \frac{1}{1-\gamma}\mathbb{E}_{s \sim d^\pi_\gamma, a \sim \pi}[A^{\pi_k}(s, a)]$ and bounding the mismatch between $d^\pi_\gamma$ and $d^{\pi_k}_\gamma$ yields the monotonic improvement lower bound [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 3.3.1, Eq. 3.85, p. 65) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.3.4, Eq. 35.34, p. 1161):
>
> $$
> \begin{aligned}
> J(\pi) - J(\pi_k) &\ge \underbrace{\frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi_k}_\gamma,  a \sim \pi_k}\!\left[ \frac{\pi(a \mid s)}{\pi_k(a \mid s)} A^{\pi_k}(s, a) \right]}_{L(\pi, \pi_k)}  \\
>   &\quad - \frac{2\gamma C_{\pi,\pi_k}}{(1-\gamma)^2} \mathbb{E}_{s \sim d^{\pi_k}_\gamma}\!\left[ D_{\mathrm{TV}}\!\left(\pi(\cdot \mid s), \pi_k(\cdot \mid s)\right) \right],
> \end{aligned}
> $$
>
> where $C_{\pi,\pi_k} = \max_s \lvert \mathbb{E}_{a \sim \pi}[A^{\pi_k}(s, a)] \rvert$. Applying Pinsker's inequality $D_{\mathrm{TV}}(p, q) \le \sqrt{\frac{1}{2}D_{\mathrm{KL}}(q \parallel p)}$, **TRPO** replaces the penalty with a hard trust-region constraint:
>
> $$
> \boldsymbol{\theta}_{k+1} = \arg\max_{\boldsymbol{\theta}} L(\pi_{\boldsymbol{\theta}}, \pi_k) \quad \text{s.t.} \quad \mathbb{E}_{s \sim d^{\pi_k}_\gamma}\!\left[D_{\mathrm{KL}}(\pi_k(\cdot \mid s) \parallel \pi_{\boldsymbol{\theta}}(\cdot \mid s))\right] \le \delta,
> $$
>
> solved via conjugate gradients on $F_k \mathbf{v} = \mathbf{g}_k$ followed by backtracking line search [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 3.3.2, p. 65).
>
> **Proximal Policy Optimization (PPO)** avoids second-order FIM-vector products by optimizing a first-order **clipped surrogate objective** over the importance ratio $r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\mathrm{old}}}(a_t \mid s_t)}$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 3.3.3, p. 66) [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Sec. 6.2.6, p. 61):
>
> $$
> L^{\mathrm{CLIP}}(\boldsymbol{\theta}) = \mathbb{E}_t\!\left[ \min\!\Big( r_t(\boldsymbol{\theta})\hat{A}_t,  \mathrm{clip}\big(r_t(\boldsymbol{\theta}),  1-\varepsilon,  1+\varepsilon\big)\hat{A}_t \Big) \right].
> $$
>
> Equivalently, $L^{\mathrm{CLIP}}(\boldsymbol{\theta}) = \mathbb{E}_t[\min(r_t(\boldsymbol{\theta})\hat{A}_t, g(\varepsilon, \hat{A}_t))]$, where $g(\varepsilon, A) = (1+\varepsilon)A$ for $A \ge 0$ and $(1-\varepsilon)A$ for $A < 0$. The $\min$ operator acts as a pessimistic lower bound that zeroes out the gradient $\nabla_{\boldsymbol{\theta}} L^{\mathrm{CLIP}}$ whenever the policy moves too far in the direction encouraged by $\hat{A}_t$ ($r_t > 1+\varepsilon$ for $\hat{A}_t > 0$, or $r_t < 1-\varepsilon$ for $\hat{A}_t < 0$).

### Control-as-Inference & Maximum-Entropy RL (SAC)

Policy optimization can alternatively be formulated as **variational Bayesian inference** ([[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 3.6, p. 73–79); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.6, p. 1177–1180)). Introduce binary optimality variables $o_t \in \{0, 1\}$ at each step with emission likelihood $p(o_t = 1 \mid s_t, a_t) = \exp\!\big(\alpha^{-1} r(s_t, a_t)\big)$ for temperature $\alpha > 0$ (assuming $r \le 0$ without loss of generality). Under a uniform action prior $p(a_t \mid s_t)$, the posterior over trajectories conditioned on optimality $o_{1:T} = 1$ is $p(\tau \mid o_{1:T}=1) \propto p(\tau) \exp\!\big(\alpha^{-1} \sum_t r(s_t, a_t)\big)$.

> [!important] Variational ELBO $\iff$ Maximum-Entropy RL & Soft Actor-Critic (SAC)
> Approximating the posterior $p(\tau \mid o_{1:T}=1)$ with a variational trajectory distribution $q_{\boldsymbol{\theta}}(\tau) = d_0(s_0)\prod_t \pi_{\boldsymbol{\theta}}(a_t \mid s_t)\mathcal{P}(s_{t+1} \mid s_t, a_t)$ that shares the true environment dynamics $\mathcal{P}$ yields the reverse-KL Evidence Lower Bound (ELBO):
>
> $$
> -D_{\mathrm{KL}}(q_{\boldsymbol{\theta}}(\tau) \parallel p(\tau \mid o_{1:T}=1)) + \text{const}
>   = \mathbb{E}_{\tau \sim q_{\boldsymbol{\theta}}}\!\left[ \sum_{t=0}^{T-1} \left( \frac{1}{\alpha} r(s_t, a_t) + \mathrm{H}\!\big(\pi_{\boldsymbol{\theta}}(\cdot \mid s_t)\big) \right) \right].
> $$
>
> Multiplying by $\alpha$ recovers the **Maximum-Entropy RL** objective. In **Soft Actor-Critic (SAC)** [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 3.6.8, p. 78–80), two soft Q-networks $Q_{\mathbf{w}_1},  Q_{\mathbf{w}_2}$ are trained off-policy against a clipped twin target $y_t = r_t + \gamma \big(\min_{i=1,2}  Q_{\bar{\mathbf{w}}_i}(s_{t+1}, \tilde{a}_{t+1}) - \alpha \log \pi_{\boldsymbol{\theta}}(\tilde{a}_{t+1} \mid s_{t+1})\big)$ with $\tilde{a}_{t+1} \sim \pi_{\boldsymbol{\theta}}(\cdot \mid s_{t+1})$, and the actor minimizes the KL projection onto the Boltzmann soft-Q policy via the **reparameterization trick** $a_t = f_{\boldsymbol{\theta}}(s_t, \boldsymbol{\xi}_t)$, $\boldsymbol{\xi}_t \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$:
>
> $$
> J_\pi(\boldsymbol{\theta}) = \mathbb{E}_{s_t \sim \mathcal{D},  \boldsymbol{\xi}_t \sim \mathcal{N}}\!\left[ \alpha \log \pi_{\boldsymbol{\theta}}(f_{\boldsymbol{\theta}}(s_t, \boldsymbol{\xi}_t) \mid s_t) - \min_{i=1,2}  Q_{\mathbf{w}_i}(s_t, f_{\boldsymbol{\theta}}(s_t, \boldsymbol{\xi}_t)) \right].
> $$

## Algorithmic Implementation & Comparative Synthesis

Modern policy-gradient systems branch into two dominant paradigms: **critic-based PPO with GAE** (Algorithm `alg:pg:ppo_gae`) for general MDPs and dense-feedback RLHF, and **critic-free Group Relative Policy Optimization (GRPO)** (Algorithm `alg:pg:grpo`) for verifiable-reward LLM reasoning ([[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Alg. 9, p. 66); [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Sec. 6.2.8, p. 66–68)).

> [!example] Algorithm: PPO-Clip with Generalized Advantage Estimation ($\mathrm{GAE}(\gamma, \lambda)$)
> **Input:** Initial actor $\boldsymbol{\theta}$, critic $\boldsymbol{\phi}$, clip $\varepsilon$, discount $\gamma$, GAE $\lambda$, epochs $M$, step sizes $\eta_\theta, \eta_\phi$
> **for** iteration $k = 1, 2, \dots$:
>
>   Set $\boldsymbol{\theta}_{\mathrm{old}} \leftarrow \boldsymbol{\theta}$ and collect trajectory batch $\{(s_t, a_t, r_t, s_{t+1})\}_{t=0}^{T-1}$ under $\pi_{\boldsymbol{\theta}_{\mathrm{old}}}$
>   Compute critic values $v_t =  V_{\boldsymbol{\phi}}(s_t)$ and set $\hat{A}_T \leftarrow 0$
>   **for** $t = T-1, T-2, \dots, 0$:
>
>     $\delta_t \leftarrow r_t + \gamma (1 - \mathrm{done}_t) V_{\boldsymbol{\phi}}(s_{t+1}) - v_t$
>     $\hat{A}_t \leftarrow \delta_t + \gamma \lambda (1 - \mathrm{done}_t) \hat{A}_{t+1}$; \quad $y_t \leftarrow \hat{A}_t + v_t$
>
>   Normalize $\hat{A}_t \leftarrow (\hat{A}_t - \mathrm{mean}(\hat{A})) / (\mathrm{std}(\hat{A}) + 10^{-8})$
>   **for** epoch $m = 1, \dots, M$:
>
>     Compute ratio $r_t(\boldsymbol{\theta}) = \pi_{\boldsymbol{\theta}}(a_t \mid s_t) / \pi_{\boldsymbol{\theta}_{\mathrm{old}}}(a_t \mid s_t)$
>     Update $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \eta_\theta \nabla_{\boldsymbol{\theta}} \frac{1}{T}\sum_{t=0}^{T-1} \big[ \min(r_t(\boldsymbol{\theta})\hat{A}_t,  \mathrm{clip}(r_t(\boldsymbol{\theta}), 1-\varepsilon, 1+\varepsilon)\hat{A}_t) + c_{\mathrm{ent}} \mathrm{H}(\pi_{\boldsymbol{\theta}}(\cdot \mid s_t)) \big]$
>     Update $\boldsymbol{\phi} \leftarrow \boldsymbol{\phi} - \eta_\phi \nabla_{\boldsymbol{\phi}} \frac{1}{T}\sum_{t=0}^{T-1} \frac{1}{2}( V_{\boldsymbol{\phi}}(s_t) - y_t)^2$

> [!example] Algorithm: Group Relative Policy Optimization (GRPO) for Autoregressive LLMs
> **Input:** Initial policy $\boldsymbol{\theta}$, frozen reference $\pi_{\mathrm{ref}}$, group size $G$, clip $\varepsilon$, KL weight $\beta$, epochs $M$
> **for** iteration $k = 1, 2, \dots$:
>
>   Sample prompt batch $\{x\} \sim \mathcal{D}$ and set $\boldsymbol{\theta}_{\mathrm{old}} \leftarrow \boldsymbol{\theta}$
>   For each prompt $x$, sample $G$ completions $\{y_i\}_{i=1}^G \sim \pi_{\boldsymbol{\theta}_{\mathrm{old}}}(\cdot \mid x)$ and score $\{r_i\}_{i=1}^G$
>   Compute group-normalized advantage for all tokens $t \in \{1, \dots, \lvert y_i \rvert\}$:
>   $A_{i,t} \leftarrow \frac{r_i - \mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G}) + 10^{-8}}$
>   **for** epoch $m = 1, \dots, M$:
>
>     For each completion $i$ and token $t$, compute $r_{i,t}(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(y_{i,t} \mid x, y_{i,<t})}{\pi_{\boldsymbol{\theta}_{\mathrm{old}}}(y_{i,t} \mid x, y_{i,<t})}$
>     Maximize $J_{\mathrm{GRPO}}(\boldsymbol{\theta}) = \frac{1}{G}\sum_{i=1}^G \frac{1}{\lvert y_i \rvert}\sum_{t=1}^{\lvert y_i \rvert} \min\!\big(r_{i,t}(\boldsymbol{\theta})A_{i,t},  \mathrm{clip}(r_{i,t}(\boldsymbol{\theta}), 1-\varepsilon, 1+\varepsilon)A_{i,t}\big) - \beta  D_{\mathrm{KL}}(\pi_{\boldsymbol{\theta}} \parallel \pi_{\mathrm{ref}})$

**Table:** Comparative synthesis of policy gradient, actor-critic, trust-region, and LLM alignment algorithms [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Ch. 3) [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Table 3, p. 72).

| Method | On/Off | Critic / Baseline | Trust-Region / IS | Primary Regime & Trade-off |
| — | — | — | — | — |
| REINFORCE | On-policy | None / MC $V(s)$ | None | High variance; simple, unbiased MC baseline |
| RLOO | On-policy | Leave-one-out mean | None | Critic-free multi-sample LLM baseline |
| A2C + GAE | On-policy | Learned $V_{\boldsymbol{\phi}}(s)$ | None (1-step/$\lambda$) | Online updates; $\lambda$ tunes bias–variance |
| TRPO | On-policy | Learned $V_{\boldsymbol{\phi}}(s)$ | Hard KL ball + FIM | Monotonic bound; costly CG Hessian |
| PPO-Clip | Near on-policy | Learned $V_{\boldsymbol{\phi}}(s)$ | Token/step ratio clip | First-order trust region; critic memory cost |
| SAC | Off-policy | Twin Soft-$Q_{\mathbf{w}_{1,2}}$ | Entropy reg. + replay | Sample-efficient continuous control |
| GRPO | Near on-policy | Group $(r_i - \mu_G)/\sigma_G$ | Token ratio clip + KL | Eliminates critic LLM; verifiable rewards |
| GSPO / CISPO | Near on-policy | Group-relative | Seq. / IS weight clip | Fixes token-level IS variance in MoE LLMs |

> [!info] Cross-Book Synthesis: Classical Control vs. Autoregressive LLM RL
> In classical MDPs [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 13) [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Ch. 3), states evolve stochastically and $\gamma < 1$ bounds horizon variance, making a learned critic $V_{\boldsymbol{\phi}}(s)$ essential for per-step credit assignment via GAE. In autoregressive LLM post-training [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Sec. 6.2.8, p. 66–68), transitions are deterministic string concatenations with $\gamma = 1$ and sparse outcome rewards, so a critic LLM doubles memory and struggles with per-token value calibration. **GRPO** replaces $V_{\boldsymbol{\phi}}$ with a Monte Carlo group baseline over $G$ completions per prompt and moves the reference KL penalty from reward shaping ($\tilde{r}_t = r_t - \beta \log \frac{\pi_{\boldsymbol{\theta}}}{\pi_{\mathrm{ref}}}$) directly into the loss. Moreover, removing $\sigma_G$ in **Dr. GRPO** ($\tilde{A}_i = r_i - \mu_G$) is algebraically equivalent to **RLOO** ($A_i^{\mathrm{RLOO}} = r_i - \frac{1}{G-1}\sum_{j \ne i} r_j$) up to the constant factor $\frac{G-1}{G}$ [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Eq. 62, p. 69).

## Visual Diagnostics & Geometric Structure

> [!example] Visual Diagram (LaTeX / PDF)
> *Structural evolution from (a) Actor–Critic with learned value baseline $V_{\boldsymbol{\phi}}$ and $\mathrm{GAE}(\gamma,\lambda)$, to (b) PPO's first-order clipped trust region over importance ratios $r_t(\boldsymbol{\theta})$, to (c) critic-free GRPO using group-normalized outcome advantages and explicit reference KL regularization.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *PPO clipped surrogate $L^{\mathrm{CLIP}}(r_t)$ versus unclipped $r_t \hat{A}_t$ for $\varepsilon = 0.2$. Clipping flattens the objective ($\nabla_{\boldsymbol{\theta}} L^{\mathrm{CLIP}} = \boldsymbol{0}$) only when the policy moves past the trust-region boundary in the direction favored by $\hat{A}_t$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!note] Source Figure
> *Figure:* Empirical variance reduction on the short-corridor gridworld: adding a learned state-value baseline enables REINFORCE to take larger step sizes ($\alpha^\theta = 2^{-9}$) and converge markedly faster than vanilla REINFORCE [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Fig. 13.2, p. 330). (`reinforcement_learning_introduction_richard_s_su_p352_fig1.png`)

> [!note] Source Figure
> *Figure:* Probabilistic graphical model for control-as-inference: states $s_t$ and actions $a_t$ are observed/controlled while binary optimality variables $o_t$ have emission $p(o_t=1 \mid s_t, a_t) = \exp(\alpha^{-1} r(s_t, a_t))$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fig. 35.11, p. 1178). (`probabilisticml_kevinmurphy_advancedtopics_p1212_fig1.png`)

> [!note] Source Figure
> *Figure:* GRPO architecture for autoregressive LLMs: $G$ completions are sampled per prompt, scored by a reward model or verifier, normalized within the group to form $A_i = (r_i - \mu)/\sigma$, and regularized directly by $D_{\mathrm{KL}}(\pi_{\boldsymbol{\theta}} \parallel \pi_{\mathrm{ref}})$ in the loss [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Fig. 25, p. 67). (`reinforcement_learning_from_human_feedback_p73_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] High-Variance Gradients Without Causality or Baselines
> Implementing the raw trajectory score-function `(eq:pg:score_fn)` with total return $R(\tau)$ instead of causal reward-to-go $G_t$ injects future-past cross-terms whose expectation is zero but whose variance grows linearly with horizon $T$. Always pair causal $G_t$ with a state-dependent baseline $V_{\boldsymbol{\phi}}(s_t)$ or $\mathrm{GAE}(\gamma,\lambda)$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 3.1.2–3.1.5, p. 52–54).

> [!warning] Token Dropping & MoE Instability in Token-Level Clipping
> In PPO and GRPO, whenever a token's importance ratio $r_{i,t}(\boldsymbol{\theta})$ exits $[1-\varepsilon, 1+\varepsilon]$, its gradient is zeroed out. For rare, pivotal reasoning tokens initially assigned low probability, this "token dropping" stalls learning; furthermore, per-token IS ratios exhibit severe variance in Mixture-of-Experts (MoE) LLMs [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Sec. 6.2.9–6.2.10, p. 69–71). **Remedies:** Use sequence-level importance ratios (**GSPO**), or clip the IS weights under a stop-gradient (**CISPO**) so every token retains a bounded gradient signal.

> [!warning] GRPO Group Standard-Deviation Collapse
> When all $G$ completions for a prompt receive identical or nearly identical rewards (e.g., all pass or all fail), $\sigma_G \to 0$, inflating $A_i = (r_i - \mu_G)/(\sigma_G + \epsilon)$ and over-weighting low-variance prompts [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Sec. 6.2.8, p. 67–68). **Remedy:** Drop $\sigma_G$ normalization (**Dr. GRPO** / RLOO) or filter zero-variance prompt groups.

> [!warning] Off-Policy Deadly Triad & Soft-Q Overestimation
> Combining off-policy replay, bootstrapping, and function approximation forms the **deadly triad** that can cause critic divergence [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 35.5.3, p. 1177). In off-policy actor-critic methods (DDPG/TD3/SAC), single-critic maximization induces positive overestimation bias. **Remedy:** Maintain Polyak-averaged target networks $\bar{\mathbf{w}}$ and take the minimum over twin critics $\min_{i=1,2} Q_{\bar{\mathbf{w}}_i}$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 3.6.8.3, p. 79).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (5):** [[GRPO]], [[Generalized Advantage Estimation (GAE)]], [[PPO (Proximal Policy Optimization)]], [[Policy Gradient Theorem]], [[TRPO]]
- **Prerequisites (Directed $\leftarrow$):** [[Generalized Advantage Estimation (GAE)]], [[KL Divergence]], [[Lagrange Duality & KKT]], [[MC & TD Learning]], [[Markov Decision Process (MDP)]], [[Natural Gradient & K-FAC]], [[PPO (Proximal Policy Optimization)]], [[Policy Gradient Theorem]], [[TRPO]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[GRPO]], [[Generalized Advantage Estimation (GAE)]], [[PPO (Proximal Policy Optimization)]], [[RLHF (KL-Regularized)]], [[RLVR (Verifiable Rewards)]], [[TRPO]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[GRPO]], [[Generalized Advantage Estimation (GAE)]], [[KL Divergence]], [[Lagrange Duality & KKT]], [[MC & TD Learning]], [[Natural Gradient & K-FAC]], [[PPO (Proximal Policy Optimization)]], [[Policy Gradient Theorem]], [[RLHF (KL-Regularized)]], [[RLVR (Verifiable Rewards)]], [[Reparameterization Trick]], [[TRPO]]
- **Synthesized Sources:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Understanding Deep Learning]]
