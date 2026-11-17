---
type: monograph
domain: "D06"
parent_topic: "[[PT 06 - Bandits, Markov Decision Processes & Value-Based RL]]"
sub_topic: "[[ST 06 - Multi-Armed Bandits, MDPs & Dynamic Programming]]"
cheatsheet: "[[Cheatsheet - MDPs, Bellman Optimality Equations, Policy - Value Iteration & GPI]]"
part_of: 
  - "[[ST 06 - Multi-Armed Bandits, MDPs & Dynamic Programming]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
concepts: 
  - "[[Bellman Equations]]"
  - "[[Markov Decision Process (MDP)]]"
  - "[[Value & Policy Iteration]]"
prerequisites: 
  - "[[Bellman Equations]]"
  - "[[Markov Decision Process (MDP)]]"
enables: 
  - "[[Bellman Equations]]"
  - "[[MC & TD Learning]]"
  - "[[Model-Based RL (Dyna & MBPO)]]"
  - "[[Policy Gradient Theorem]]"
  - "[[Value & Policy Iteration]]"
related: 
  - "[[Bellman Equations]]"
  - "[[Markov Decision Process (MDP)]]"
  - "[[Q-Learning & DQN]]"
  - "[[Value & Policy Iteration]]"
sources: 
  - "[[Source - Francis Bach]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Reinforcement Learning - An Overview]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-06
---

# MDPs, Bellman Optimality Equations, Policy/Value Iteration & GPI

> [!abstract] Monograph Navigation
> - **Domain:** [[D06 - Reinforcement Learning, Control & LLM Alignment|Reinforcement Learning, Control & LLM Alignment]]
> - **Parent / Sub-Topic:** [[PT 06 - Bandits, Markov Decision Processes & Value-Based RL|Bandits, Markov Decision Processes & Value-Based RL]] → [[ST 06 - Multi-Armed Bandits, MDPs & Dynamic Programming|Multi-Armed Bandits, MDPs & Dynamic Programming]]
> - **Visual Cheatsheet:** [[Cheatsheet - MDPs, Bellman Optimality Equations, Policy - Value Iteration & GPI]]
> - **Description:** Formalizes finite MDPs and POMDPs, discounted returns, Bellman expectation and optimality backup operators as gamma-contractions (Banach fixed point), iterative policy evaluation, policy iteration, value iteration, and Generalized Policy Iteration.

## Problem Formulation & Unified Notation

A finite **Markov Decision Process (MDP)** formalizes sequential decision-making under uncertainty when the environment state is fully observable (see [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 3, p. 48) and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.5.1, p. 1128)). At discrete time steps $t = 0, 1, 2, \dots$, an agent observes state $S_t \in \mathcal{S}$, selects an action $A_t \in \mathcal{A}(S_t)$ according to a policy $\pi$, and transitions to a successor state $S_{t+1} \in \mathcal{S}$ while receiving a scalar reward $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$.

> [!important] Finite MDP Tuple and Markov Property
> A finite discounted MDP is specified by the tuple $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, p, R, \gamma, p_0 \rangle$, where $\mathcal{S}$ is a finite state space ($|\mathcal{S}| < \infty$), $\mathcal{A}$ is a finite action space ($|\mathcal{A}| < \infty$), $p_0 \in \Delta(\mathcal{S})$ is the initial state distribution, and $\gamma \in [0, 1)$ is the discount factor. The environment obeys the **Markov property**: the joint distribution of the next state and reward depends only on the current state-action pair $(S_t, A_t)$ and not on the prior history $H_t = (S_0, A_0, R_1, \dots, S_t)$:
>
> $$
> p(s', r \mid s, a) \triangleq \mathbb{P}(S_{t+1} = s', R_{t+1} = r \mid S_t = s, A_t = a), \quad \forall s, s' \in \mathcal{S},  a \in \mathcal{A}(s),  r \in \mathcal{R}.
> $$
>
> Marginalizing over rewards yields the state-transition kernel $p(s' \mid s, a) = \sum_{r \in \mathcal{R}} p(s', r \mid s, a)$, and taking expectations yields the expected immediate reward $R(s, a) \triangleq \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a] = \sum_{s', r} r  p(s', r \mid s, a)$.

When the environment state $w_t$ is hidden and the agent only receives noisy observations $o_t \sim p(o_t \mid w_t)$, the problem generalizes to a **Partially Observable MDP (POMDP)** [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 1.2.1, p. 17)[[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 19.1.3, p. 390). A POMDP reduces to an MDP over continuous *belief states* $b_t(w_t) = p(w_t \mid o_{1:t}, a_{1:t-1})$, which form a sufficient statistic for optimal control, though exact belief-space planning is PSPACE-hard [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 1.2.1, p. 18).

> [!note] Source Figure
> *Figure:* The canonical agent–environment interaction loop in a finite MDP [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 3, Fig. 3.1, p. 48). (`reinforcement_learning_introduction_richard_s_su_p70_fig1.png`)

To evaluate long-horizon behavior, we define the **discounted return** $G_t$ from time $t$ onward:

$$
G_t \triangleq \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma G_{t+1}.
$$

Episodic tasks that terminate at a random horizon $T$ are unified with continuing ($T = \infty$) tasks by introducing a special *absorbing terminal state* $s_{\perp}$ that transitions only to itself with probability $1$ and emits zero reward thereafter [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 3.4, p. 57)[[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.5.3, p. 1131).

> [!note] Source Figure
> *Figure:* MDP represented as a finite state machine (FSM) with three discrete states ($S_0, S_1, S_2$), two actions ($a_0, a_1$), transition probabilities, and sparse non-zero rewards ($+5, -1$) [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Ch. 1, Fig. 1.3, p. 18). (`reinforement_learning_kevin_murphy_p18_fig1.png`)

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Source textbooks index rewards and represent dynamics differently. Table `tab:mdp:notation` establishes a unified translation across all four references.
> 
>
>
>
> | **Concept** | **Sutton & Barto** | **Murphy (RL / Adv.)** | **Prince (UDL)** | **Unified Monograph** |
> | — | — | — | — | — |
> | Joint Dynamics | $p(s', r \mid s, a)$ | $p(s' \mid s, a)$ and $p_R(r \mid s, a, s')$ | $\Pr(s_{t+1} \mid s_t, a_t)$, $r[s_t, a_t]$ | $p(s', r \mid s, a)$ |
> | Reward Indexing | $R_{t+1}$ after $(S_t, A_t)$ | $r_t$ at step $t$ (after $s_t, a_t$) | $r_{t+1}$ after $(s_t, a_t)$ | $R_{t+1}$ (with $R(s, a) = \mathbb{E}[R_{t+1}]$) |
> | State Value | $v_\pi(s)$, $v_*(s)$ | $V_\pi(s)$, $V^*(s)$ (vector $\boldsymbol{v}_\pi$) | $v[s_t \mid \pi]$ | $V^\pi(s)$, $V^*(s)$ (vector $\boldsymbol{v}_\pi$) |
> | Action Value | $q_\pi(s, a)$, $q_*(s, a)$ | $Q_\pi(s, a)$, $Q^*(s, a)$ | $q[s_t, a_t \mid \pi]$ | $Q^\pi(s, a)$, $Q^*(s, a)$ |
> | Policy Transition | $\sum_a \pi(a \mid s) p(s' \mid s, a)$ | $\boldsymbol{T}(s' \mid s)$ (or $\boldsymbol{P}_\pi$) | $\sum_{a_t} \pi[a_t \mid s_t] \Pr(s_{t+1} \mid s_t, a_t)$ | $\boldsymbol{P}_\pi \in \mathbb{R}^{\mid \mathcal{S}\mid  \times \mid \mathcal{S}\mid }$ |

## Core Mathematical Derivations & Theorems

### Policy Value Functions and Bellman Expectation Equations

Given a stationary stochastic policy $\pi(a \mid s) \triangleq \mathbb{P}(A_t = a \mid S_t = s)$, the **state-value function** $V^\pi(s)$ and **action-value function** $Q^\pi(s, a)$ quantify expected discounted returns:

$$
\begin{aligned}
V^\pi(s) &\triangleq \mathbb{E}_\pi\!\left[G_t \mid S_t = s\right] = \sum_{a \in \mathcal{A}(s)} \pi(a \mid s)  Q^\pi(s, a),  \\
  Q^\pi(s, a) &\triangleq \mathbb{E}_\pi\!\left[G_t \mid S_t = s, A_t = a\right] = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} p(s' \mid s, a)  V^\pi(s').
\end{aligned}
$$

The **advantage function** $A^\pi(s, a) \triangleq Q^\pi(s, a) - V^\pi(s)$ measures the relative benefit of taking action $a$ in state $s$ before reverting to $\pi$, satisfying $\mathbb{E}_{a \sim \pi(\cdot \mid s)}[A^\pi(s, a)] = 0$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.5.4, p. 1131).

> [!important] Bellman Expectation Equations & Exact Matrix-Vector Inversion
> Substituting `(eq:mdp:return)` into `(eq:mdp:vpi_def)` and applying the law of total expectation over $A_t \sim \pi(\cdot \mid s)$ and $(S_{t+1}, R_{t+1}) \sim p(\cdot, \cdot \mid s, A_t)$ yields the **Bellman expectation equation** for $V^\pi$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 3, Eq. (3.14), p. 59):
>
> $$
> V^\pi(s) = \sum_{a \in \mathcal{A}(s)} \pi(a \mid s) \sum_{s' \in \mathcal{S},  r \in \mathcal{R}} p(s', r \mid s, a)\left[r + \gamma V^\pi(s')\right] = (\mathcal{T}^\pi V^\pi)(s),
> $$
>
> where $\mathcal{T}^\pi : \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$ is the **Bellman expectation operator**. Defining the policy-induced expected reward vector $\boldsymbol{r}_\pi \in \mathbb{R}^{|\mathcal{S}|}$ with $r_\pi(s) = \sum_a \pi(a \mid s) R(s, a)$ and transition matrix $\boldsymbol{P}_\pi \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}$ with $[\boldsymbol{P}_\pi]_{s, s'} = \sum_a \pi(a \mid s) p(s' \mid s, a)$, `(eq:mdp:bellman_exp_v)` becomes a linear system [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.2.3, p. 34):
>
> $$
> \boldsymbol{v}_\pi = \boldsymbol{r}_\pi + \gamma \boldsymbol{P}_\pi \boldsymbol{v}_\pi \implies (\boldsymbol{I} - \gamma \boldsymbol{P}_\pi)\boldsymbol{v}_\pi = \boldsymbol{r}_\pi.
> $$
>
> Because $\boldsymbol{P}_\pi$ is row-stochastic, its spectral radius satisfies $\rho(\boldsymbol{P}_\pi) = 1$. For $\gamma \in [0, 1)$, all eigenvalues of $\gamma \boldsymbol{P}_\pi$ lie strictly inside the unit disk, so $\boldsymbol{I} - \gamma \boldsymbol{P}_\pi$ is strictly diagonally dominant and nonsingular, yielding the unique closed-form solution:
>
> $$
> \boldsymbol{v}_\pi = (\boldsymbol{I} - \gamma \boldsymbol{P}_\pi)^{-1} \boldsymbol{r}_\pi = \sum_{t=0}^{\infty} \gamma^t \boldsymbol{P}_\pi^t \boldsymbol{r}_\pi.
> $$

### Bellman Optimality Equations and Contraction Mapping Theory

A policy $\pi^*$ is **optimal** if $V^{\pi^*}(s) \ge V^\pi(s)$ for all $s \in \mathcal{S}$ and all policies $\pi$. All optimal policies share the unique optimal state-value function $V^*(s) \triangleq \max_\pi V^\pi(s)$ and optimal action-value function $Q^*(s, a) \triangleq \max_\pi Q^\pi(s, a)$, and every finite MDP admits at least one deterministic optimal policy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.5.5, p. 1132). Their self-consistency conditions form the **Bellman optimality equations** [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 3, Eqs. (3.19)–(3.20), p. 63):

$$
\begin{aligned}
V^*(s) &= \max_{a \in \mathcal{A}(s)} \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma V^*(s')\right] \triangleq (\mathcal{T}^* V^*)(s),  \\
  Q^*(s, a) &= \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma \max_{a' \in \mathcal{A}(s')} Q^*(s', a')\right].
\end{aligned}
$$

> [!important] Banach $\gamma$-Contraction Proof for $\mathcal{T}^*$ and $\mathcal{T}^\pi$
> Let $\mathcal{V} = (\mathbb{R}^{|\mathcal{S}|}, \lVert \cdot \rVert_\infty)$ be the Banach space of bounded value functions equipped with the supremum norm $\lVert V \rVert_\infty = \max_{s \in \mathcal{S}} \lvert V(s) \rvert$. For any two vectors $U, V \in \mathbb{R}^{|\mathcal{S}|}$ and any state $s \in \mathcal{S}$:
>
> $$
> \begin{aligned}
> \lvert (\mathcal{T}^* U)(s) - (\mathcal{T}^* V)(s) \rvert
>   &= \Bigl|\max_{a \in \mathcal{A}(s)} \Bigl[ R(s, a) + \gamma \sum_{s'} p(s' \mid s, a) U(s') \Bigr]  \\
>   &\qquad - \max_{a \in \mathcal{A}(s)} \Bigl[ R(s, a) + \gamma \sum_{s'} p(s' \mid s, a) V(s') \Bigr]\Bigr|  \\
>   &\le \max_{a \in \mathcal{A}(s)} \lvert \gamma \sum_{s' \in \mathcal{S}} p(s' \mid s, a)\left(U(s') - V(s')\right) \rvert  \\
>   &\le \gamma \max_{a \in \mathcal{A}(s)} \sum_{s' \in \mathcal{S}} p(s' \mid s, a) \lVert U - V \rVert_\infty = \gamma \lVert U - V \rVert_\infty,
> \end{aligned}
> $$
>
> where the first inequality uses $\lvert \max_a f(a) - \max_a g(a) \rvert \le \max_a \lvert f(a) - g(a) \rvert$ and the final equality uses $\sum_{s'} p(s' \mid s, a) = 1$. Taking the maximum over $s \in \mathcal{S}$ gives $\lVert \mathcal{T}^* U - \mathcal{T}^* V \rVert_\infty \le \gamma \lVert U - V \rVert_\infty$ (and identically $\lVert \mathcal{T}^\pi U - \mathcal{T}^\pi V \rVert_\infty \le \gamma \lVert U - V \rVert_\infty$). By the **Banach Fixed-Point Theorem**, for $\gamma \in [0, 1)$, $\mathcal{T}^*$ has a unique fixed point $V^* = \mathcal{T}^* V^*$, and iterative updates $V_{k+1} = \mathcal{T}^* V_k$ converge geometrically from any $V_0 \in \mathbb{R}^{|\mathcal{S}|}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.6.1, p. 1134):
>
> $$
> \lVert V_k - V^* \rVert_\infty \le \gamma^k \lVert V_0 - V^* \rVert_\infty.
> $$

> [!tip] Why $V^*$ Turns Long-Term Planning into One-Step Lookahead
> The optimal value function $V^*(s)$ compresses an infinite-horizon tree of future stochastic transitions into a single scalar per state. Once $V^*$ is known, an optimal policy $\pi^*(s) \in \arg\max_{a \in \mathcal{A}(s)} \left[R(s,a) + \gamma \sum_{s'} p(s' \mid s, a) V^*(s')\right]$ requires only a greedy one-step lookahead; $Q^*(s, a)$ goes one step further by caching the lookahead expectations, making action selection $\pi^*(s) \in \arg\max_a Q^*(s, a)$ model-free [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 3, p. 64).

### Policy Improvement Theorem

To guarantee monotonic progress when alternating between evaluation and greedy action selection, Dynamic Programming relies on the **Policy Improvement Theorem** [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 4.2, p. 78)[[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.6.2, p. 1135).

> [!important] Policy Improvement Theorem (Telescopic & Operator Proofs)
> Let $\pi$ and $\pi'$ be any pair of deterministic policies such that for all $s \in \mathcal{S}$,
>
> $$
> Q^\pi(s, \pi'(s)) \ge V^\pi(s).
> $$
>
> Then $\pi'$ is as good as or better than $\pi$: $V^{\pi'}(s) \ge V^\pi(s)$ for all $s \in \mathcal{S}$. Moreover, if strict inequality holds at any state $s$, then $V^{\pi'}(s) > V^\pi(s)$ at that state.
>
> **Proof (Matrix-Operator Neumann Expansion):** In vector notation, `(eq:mdp:pi_cond)` states $\boldsymbol{r}_{\pi'} + \gamma \boldsymbol{P}_{\pi'} \boldsymbol{v}_\pi \ge \boldsymbol{v}_\pi$ componentwise [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.2.3, p. 34). Since $\gamma \boldsymbol{P}_{\pi'}$ has nonnegative entries, it is a monotone linear operator ($\boldsymbol{x} \le \boldsymbol{y} \implies \gamma \boldsymbol{P}_{\pi'} \boldsymbol{x} \le \gamma \boldsymbol{P}_{\pi'} \boldsymbol{y}$). Applying $\mathcal{T}^{\pi'}$ repeatedly gives:
>
> $$
> \boldsymbol{v}_\pi \le \boldsymbol{r}_{\pi'} + \gamma \boldsymbol{P}_{\pi'} \boldsymbol{v}_\pi \le \boldsymbol{r}_{\pi'} + \gamma \boldsymbol{P}_{\pi'} (\boldsymbol{r}_{\pi'} + \gamma \boldsymbol{P}_{\pi'} \boldsymbol{v}_\pi) \le \dots \le \sum_{t=0}^{\infty} \gamma^t \boldsymbol{P}_{\pi'}^t \boldsymbol{r}_{\pi'} + \lim_{t \to \infty} \gamma^t \boldsymbol{P}_{\pi'}^t \boldsymbol{v}_\pi = \boldsymbol{v}_{\pi'},
> $$
>
> where $\lim_{t \to \infty} \gamma^t \boldsymbol{P}_{\pi'}^t \boldsymbol{v}_\pi = \boldsymbol{0}$ since $\gamma < 1$. Setting $\pi'(s) \in \arg\max_{a \in \mathcal{A}(s)} Q^\pi(s, a)$ satisfies `(eq:mdp:pi_cond)` by construction. If $V^{\pi'} = V^\pi$, then $V^\pi = \mathcal{T}^* V^\pi$, so by uniqueness of the fixed point of $\mathcal{T}^*$, $V^\pi = V^*$ and $\pi$ is optimal.

> [!tip] The Discount Factor $\gamma$ as Geometric Survival Probability
> Mathematically, $\gamma \in [0, 1)$ ensures bounded returns and $\gamma$-contraction. Probabilistically, $\gamma$ is equivalent to an undiscounted process in which the agent survives to the next step with probability $\gamma$ and is absorbed into a zero-reward terminal state with probability $1 - \gamma$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.5.3, p. 1131). Setting $\gamma = 0$ yields a completely myopic agent, whereas $\gamma \to 1$ extends the effective planning horizon $1 / (1 - \gamma) \to \infty$.

## Algorithmic Implementation & Comparative Synthesis

Dynamic Programming (DP) transforms the Bellman equations into iterative assignment operators [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 4, pp. 80–84)[[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 19.3.1, p. 396). **Policy Iteration (PI)** alternates full policy evaluation ($V \leftarrow V^\pi$) with greedy policy improvement ($\pi \leftarrow \mathrm{greedy}(V)$). Because a finite MDP has at most $|\mathcal{A}|^{|\mathcal{S}|}$ deterministic policies and each strict improvement eliminates suboptimal policies, PI terminates in finitely many steps. **Value Iteration (VI)** truncates policy evaluation to a single Bellman optimality sweep $V_{k+1} = \mathcal{T}^* V_k$, extracting $\pi^*$ only after convergence.

> [!example] Algorithm: Policy Iteration (PI) with Consistent Tie-Breaking
> **Input:** Finite MDP $\langle \mathcal{S}, \mathcal{A}, p, R, \gamma \rangle$, evaluation tolerance $\theta > 0$
> **Output:** Optimal value function $V^*$ and deterministic optimal policy $\pi^*$
> Initialize $V(s) \leftarrow 0$ and $\pi(s) \in \mathcal{A}(s)$ arbitrarily for all $s \in \mathcal{S}$
> **repeat until** $\text{policy\_stable} = \mathrm{true}$:
>
>    *// 1. Policy Evaluation (Iterative or Exact $(\boldsymbol{I} - \gamma \boldsymbol{P}_\pi)^{-1} \boldsymbol{r}_\pi$)*
>   **repeat until** $\Delta < \theta$:
>
>     $\Delta \leftarrow 0$
>     **for each** $s \in \mathcal{S}$:
>
>       $v_{\mathrm{old}} \leftarrow V(s)$
>       $V(s) \leftarrow \sum_{s', r} p(s', r \mid s, \pi(s)) [r + \gamma V(s')]$
>       $\Delta \leftarrow \max(\Delta, \lvert v_{\mathrm{old}} - V(s) \rvert)$
>
>
>    *// 2. Policy Improvement*
>   $\text{policy\_stable} \leftarrow \mathrm{true}$
>   **for each** $s \in \mathcal{S}$:
>
>     $a_{\mathrm{old}} \leftarrow \pi(s)$
>     $a^* \leftarrow \arg\max_{a \in \mathcal{A}(s)} \sum_{s', r} p(s', r \mid s, a) [r + \gamma V(s')]$
>     **if** $Q_V(s, a^*) > Q_V(s, a_{\mathrm{old}})$:
>
>       $\pi(s) \leftarrow a^*$; \quad $\text{policy\_stable} \leftarrow \mathrm{false}$
>
>
>
> **return** $V, \pi$

> [!example] Algorithm: Value Iteration (VI) with Contraction Stopping Bound
> **Input:** Finite MDP $\langle \mathcal{S}, \mathcal{A}, p, R, \gamma \rangle$, target value error $\epsilon > 0$
> **Output:** $\epsilon$-optimal value function $V$ and greedy policy $\pi$
> Initialize $V(s) \leftarrow 0$ for all $s \in \mathcal{S}$; \quad $\theta \leftarrow \epsilon(1 - \gamma) / (2\gamma)$
> **repeat until** $\Delta < \theta$:
>
>   $\Delta \leftarrow 0$
>   **for each** $s \in \mathcal{S}$:
>
>     $v_{\mathrm{old}} \leftarrow V(s)$
>     $V(s) \leftarrow \max_{a \in \mathcal{A}(s)} \sum_{s', r} p(s', r \mid s, a) [r + \gamma V(s')]$
>     $\Delta \leftarrow \max(\Delta, \lvert v_{\mathrm{old}} - V(s) \rvert)$
>
>
> **for each** $s \in \mathcal{S}$:
>
>   $\pi(s) \leftarrow \arg\max_{a \in \mathcal{A}(s)} \sum_{s', r} p(s', r \mid s, a) [r + \gamma V(s')]$
>
> **return** $V, \pi$

**Table:** Computational complexity and convergence comparison of exact MDP planning algorithms [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 4.7, p. 87)[[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.6, pp. 1134–1136).

| **Method** | **Per-Iteration Cost** | **Convergence Rate** | **Key Trade-off / Scope** |
| — | — | — | — |
| Iterative Policy Eval. | $O(\mid \mathcal{S}\mid ^2)$ | Linear: $O(\gamma^k)$ | Evaluates fixed $\pi$; avoids $O(\mid \mathcal{S}\mid ^3)$ inversion of $(\boldsymbol{I} - \gamma \boldsymbol{P}_\pi)$. |
| Policy Iteration (PI) | $O(\mid \mathcal{S}\mid ^3 + \mid \mathcal{S}\mid ^2 \mid \mathcal{A}\mid )$ | Finite / superlinear | Few outer iterations in practice; exact evaluation is cubic in $\mid \mathcal{S}\mid$. |
| Value Iteration (VI) | $O(\mid \mathcal{S}\mid ^2 \mid \mathcal{A}\mid )$ | Linear: $O(\gamma^k)$ | Cheap sweeps, but slow when $\gamma \to 1$; requires $O\!\left(\frac{\log(1/\epsilon)}{1-\gamma}\right)$ sweeps. |
| Modified PI / GPI | $O(m\mid \mathcal{S}\mid ^2 + \mid \mathcal{S}\mid ^2 \mid \mathcal{A}\mid )$ | Geometric to superlinear | Performs $m$ evaluation sweeps per improvement; interpolates VI ($m=1$) and PI ($m=\infty$). |
| Linear Programming | $\mathrm{poly}(\mid \mathcal{S}\mid , \mid \mathcal{A}\mid )$ | Polynomial (interior pt.) | Primal $\min \boldsymbol{p}_0^{\top} \boldsymbol{v}$ s.t. $\boldsymbol{v} \ge \mathcal{T}^* \boldsymbol{v}$; impractical for large $\mid \mathcal{S}\mid$. |

> [!info] Cross-Book Synthesis: Generalized Policy Iteration, LP, and RTDP
> **Generalized Policy Iteration (GPI)** unifies virtually all RL and DP methods as two competing yet cooperative processes: *evaluation* drives $V \to V^\pi$ while *improvement* drives $\pi \to \mathrm{greedy}(V)$ [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 4.6, p. 86). Policy Iteration and Value Iteration sit at opposite ends of this spectrum, while **Asynchronous DP** and **Real-Time DP (RTDP)** focus Bellman backups on reachable states along simulated trajectories rather than full synchronous sweeps [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Sec. 2.2.2, p. 34). Conversely, Murphy highlights the **Linear Programming (LP)** dual formulation, which maximizes expected reward $\sum_{s,a} d(s,a) R(s,a)$ over normalized occupancy measures $d(s,a)$ satisfying Bellman flow conservation [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.6.3, p. 1136); this dual view forms the mathematical bridge to off-policy RL and constrained MDPs.

## Visual Diagnostics & Geometric Structure

Figure `fig:mdp:gpi_backup` illustrates the geometric interplay of GPI alongside the structural contrast between Bellman expectation ($\mathcal{T}^\pi$) and Bellman optimality ($\mathcal{T}^*$) 1-step backup trees. Figure `fig:mdp:sutton_backups` reproduces the canonical backup diagrams for $V^*$ and $Q^*$, while Figure `fig:mdp:contraction_convergence` quantifies how the discount factor $\gamma$ governs the Banach $\ell_\infty$ error decay of Value Iteration relative to Policy Iteration's finite-step convergence.

> [!example] Visual Diagram (LaTeX / PDF)
> *(a) Generalized Policy Iteration (GPI) interleaves evaluation and greedy improvement until reaching the unique optimal fixed point $(V^*, \pi^*)$. (b) 1-step backup trees contrast expectation over $\pi$ ($\mathcal{T}^\pi$) with maximization over actions ($\mathcal{T}^*$).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!note] Source Figure
> *Figure:* Extracted 300 DPI backup diagrams for $v_*$ and $q_*$ showing the maximization arc at agent decision nodes and stochastic transitions at environment chance nodes [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 3, Fig. 3.4, p. 64). (`reinforcement_learning_introduction_richard_s_su_p86_fig1.png`)

> [!example] Visual Diagram (LaTeX / PDF)
> *Banach $\ell_\infty$ contraction bounds $\lVert V_k - V^* \rVert_\infty / \lVert V_0 - V^* \rVert_\infty \le \gamma^k$ for Value Iteration across discount factors $\gamma \in \{0.5, 0.9, 0.99\}$, contrasted with the rapid superlinear convergence of Policy Iteration.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Policy Oscillation Under Arbitrary Tie-Breaking in Policy Iteration
> In naive Policy Iteration [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Ch. 4, Ex. 4.4, p. 82), if two actions $a_1, a_2 \in \mathcal{A}(s)$ achieve identical optimal value $Q_V(s, a_1) = Q_V(s, a_2)$ and $\arg\max$ breaks ties arbitrarily, $\pi(s)$ can cycle indefinitely between $a_1$ and $a_2$, causing $\text{policy\_stable}$ to remain `false`. **Remedy:** Only update $\pi(s) \leftarrow a^*$ when $Q_V(s, a^*) > Q_V(s, \pi(s))$ strictly (as enforced in Algorithm `alg:mdp:policy_iteration`), or use a fixed deterministic tie-breaking order.

> [!warning] Stopping Criterion Error Amplification as $\gamma \to 1$
> Terminating Value Iteration when $\Delta_k = \lVert V_k - V_{k-1} \rVert_\infty < \theta$ does *not* imply $\lVert V_k - V^* \rVert_\infty < \theta$. By the triangle inequality and $\gamma$-contraction:
>
> $$
> \lVert V_k - V^* \rVert_\infty \le \frac{\gamma}{1 - \gamma} \lVert V_k - V_{k-1} \rVert_\infty.
> $$
>
> When $\gamma = 0.999$, $\frac{\gamma}{1-\gamma} = 999$, so $\Delta_k = 10^{-3}$ still permits a value error of $\approx 1.0$. Furthermore, extracting the greedy policy from $V_k$ guarantees $\lVert V^{\pi_k} - V^* \rVert_\infty \le \frac{2\gamma}{1-\gamma}\lVert V_k - V_{k-1} \rVert_\infty$, motivating the scaled threshold $\theta = \frac{\epsilon(1-\gamma)}{2\gamma}$ in Algorithm `alg:mdp:value_iteration`.

> [!warning] Singularity for $\gamma = 1$ in Continuing Tasks and Curse of Dimensionality
> If $\gamma = 1$ in a continuing (non-episodic) MDP, $\boldsymbol{I} - \boldsymbol{P}_\pi$ is singular because $\boldsymbol{P}_\pi \boldsymbol{1} = \boldsymbol{1}$, causing $(\boldsymbol{I} - \gamma \boldsymbol{P}_\pi)^{-1}$ in `(eq:mdp:neumann)` to diverge and $\mathcal{T}^*$ to lose its contraction property [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 34.5.3, p. 1131). Undiscounted continuing MDPs require the *average-reward* Bellman formulation instead. Additionally, tabular DP scales as $O(|\mathcal{S}|^2 |\mathcal{A}|)$, which becomes intractable under the **curse of dimensionality** when $|\mathcal{S}|$ grows exponentially with state variables, motivating Approximate Dynamic Programming (ADP) and function approximation [[Source - Francis Bach|reinforcement_learning_introduction_richard_s_su]] (Sec. 4.7, p. 87).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[Bellman Equations]], [[Markov Decision Process (MDP)]], [[Value & Policy Iteration]]
- **Prerequisites (Directed $\leftarrow$):** [[Bellman Equations]], [[Markov Decision Process (MDP)]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Bellman Equations]], [[MC & TD Learning]], [[Model-Based RL (Dyna & MBPO)]], [[Policy Gradient Theorem]], [[Value & Policy Iteration]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Bellman Equations]], [[Markov Decision Process (MDP)]], [[Q-Learning & DQN]], [[Value & Policy Iteration]]
- **Synthesized Sources:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]], [[Source - Understanding Deep Learning]]
