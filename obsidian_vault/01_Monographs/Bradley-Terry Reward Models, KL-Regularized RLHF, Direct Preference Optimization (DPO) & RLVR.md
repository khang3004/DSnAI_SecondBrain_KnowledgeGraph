---
type: monograph
domain: "D06"
parent_topic: "[[PT 06 - Policy Gradients, Model-Based RL, MARL & LLM Alignment]]"
sub_topic: "[[ST 06 - Post-Training Alignment - RLHF, Reward Modeling, DPO & RLVR]]"
cheatsheet: "[[Cheatsheet - Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]"
part_of: 
  - "[[ST 06 - Post-Training Alignment - RLHF, Reward Modeling, DPO & RLVR]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
concepts: 
  - "[[Bradley-Terry Reward Model]]"
  - "[[DPO (Direct Preference Optimization)]]"
  - "[[RLHF (KL-Regularized)]]"
  - "[[RLVR (Verifiable Rewards)]]"
prerequisites: 
  - "[[Bradley-Terry Reward Model]]"
  - "[[GRPO]]"
  - "[[KL Divergence]]"
  - "[[PPO (Proximal Policy Optimization)]]"
enables: 
  - "[[Bayesian Personalized Ranking (BPR)]]"
  - "[[DPO (Direct Preference Optimization)]]"
  - "[[RLHF (KL-Regularized)]]"
related: 
  - "[[Bradley-Terry Reward Model]]"
  - "[[Chain-of-Thought & Test-Time Compute]]"
  - "[[DPO (Direct Preference Optimization)]]"
  - "[[GRPO]]"
  - "[[KL Divergence]]"
  - "[[LoRA & QLoRA]]"
  - "[[PPO (Proximal Policy Optimization)]]"
  - "[[RLHF (KL-Regularized)]]"
sources: 
  - "[[Source - Foundations of Large Language Models]]"
  - "[[Source - Reinforcement Learning - An Overview]]"
  - "[[Source - Reinforcement Learning from Human Feedback]]"
  - "[[Source - Speech and Language Processing]]"
  - "[[Source - The Principles of Diffusion Models]]"
tags:
  - monograph
  - domain-06
---

# Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR

> [!abstract] Monograph Navigation
> - **Domain:** [[D06 - Reinforcement Learning, Control & LLM Alignment|Reinforcement Learning, Control & LLM Alignment]]
> - **Parent / Sub-Topic:** [[PT 06 - Policy Gradients, Model-Based RL, MARL & LLM Alignment|Policy Gradients, Model-Based RL, MARL & LLM Alignment]] → [[ST 06 - Post-Training Alignment - RLHF, Reward Modeling, DPO & RLVR|Post-Training Alignment: RLHF, Reward Modeling, DPO & RLVR]]
> - **Visual Cheatsheet:** [[Cheatsheet - Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]
> - **Description:** Derives the Bradley-Terry preference likelihood, outcome vs. process reward models (ORM/PRM), KL-constrained RL objective and closed-form optimal Gibbs policy pi*(y|x), the exact DPO change-of-variables derivation, GRPO/RLVR reasoning training, Constitutional AI, and Goodhart's Law reward over-optimization.

## Problem Formulation & Unified Notation

Post-training alignment steers a pretrained and supervised fine-tuned (SFT) autoregressive policy $\pi_{\mathrm{ref}}(y \mid x)$ toward human preferences or verifiable correctness criteria without sacrificing fluency or general world knowledge. Formally, given a prompt distribution $x \sim \mathcal{D}$ and an autoregressive completion $y = (y_1, \dots, y_T)$ generated under

$$
\pi_\theta(y \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{<t}),
$$

post-training can be viewed either as a *single-turn contextual bandit* with sequence-valued action $y$ and terminal scalar reward $r(x,y)$, or as a *deterministic-transition token-level MDP* with state $s_t = \mathrm{concat}(x, y_{<t})$, action $a_t = y_t$, and transition $p(s_{t+1} \mid s_t, a_t) = \delta(s_{t+1} = \mathrm{concat}(s_t, a_t))$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Ch. 6, pp. 150–151). Unlike classical RL where the environment supplies an exogenous reward, preference-based alignment must first infer a scalar proxy $r_\phi(x,y)$ from pairwise comparisons $(x, y_w, y_l)$ where $y_w \succ y_l$, or bypass explicit reward modeling altogether via closed-form policy-reward reparameterization [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 5, p. 39; Ch. 8, p. 115).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Because LLM alignment bridges NLP, classical RL, and generative modeling, each textbook adopts slightly different symbols for identical mathematical objects. Table `tab:align:notation_tbl` establishes a unified notation used throughout this monograph.
>
>
>
> | **Concept** | **RLHF** | **LLMs** | **Murphy** | **J&M / Diff.** | **Unified** |
> | — | — | — | — | — | — |
> | Prompt / Context | $x$ | $x$ | $s$ ($x$) | $x$ / $c$ | $x \sim \mathcal{D}$ |
> | Completion | $y$ | $y$ | $a$ ($y$) | $o$ / $x_{0:T}$ | $y = y_{1:T}$ |
> | Preference Pair | $y_c \succ y_r$ | $y_a \succ y_b$ | $y_w \succ y_l$ | $o_w \succ o_l$ | $y_w \succ y_l$ |
> | Reward Model | $r_\theta(y\mid x)$ | $r(x,y)$ | $R(x,y)$ | $r_\psi(c,x)$ | $r_\phi(x,y)$ |
> | Reference Policy | $\pi_{\mathrm{ref}}$ | $\pi_{\mathrm{SFT}}$ | $\pi_{\mathrm{ref}}$ | $\pi_{\mathrm{ref}}$ | $\pi_{\mathrm{ref}}(y\mid x)$ |
> | KL Coefficient | $\beta$ | $\beta$ | $\beta$ | $\beta$ | $\beta > 0$ |

## Core Mathematical Derivations & Theorems

### Bradley-Terry Preference Modeling, ORMs, and PRMs

> [!important] Bradley-Terry Likelihood & Pairwise Cross-Entropy Loss
> Under the Bradley-Terry model [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Sec. 8.3.2, p. 218) [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 5, pp. 39–41), each completion $y$ to prompt $x$ possesses a latent strength $p(y \mid x) = \exp(r^*(x,y)) > 0$. The probability that $y_w$ is preferred over $y_l$ is the logistic sigmoid of their reward difference:
>
> $$
> P(y_w \succ y_l \mid x) = \frac{\exp(r^*(x, y_w))}{\exp(r^*(x, y_w)) + \exp(r^*(x, y_l))} = \sigma\!\big(r^*(x, y_w) - r^*(x, y_l)\big),
> $$
>
> where $\sigma(z) = (1 + e^{-z})^{-1}$. Equivalently, the reward gap $\delta = r^*(x, y_w) - r^*(x, y_l)$ equals the log-odds $\log \frac{P(y_w \succ y_l \mid x)}{1 - P(y_w \succ y_l \mid x)}$. Parameterizing $r_\phi(x,y) = h_{\mathrm{last}}(x,y)^\top W_r$ via a linear head on the final Transformer hidden state at the $\langle\mathrm{EOS}\rangle$ token [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 4, p. 180), we train $r_\phi$ by minimizing the negative log-likelihood over preference dataset $\mathcal{D}_{\mathrm{pref}}$:
>
> $$
> \mathcal{L}_{\mathrm{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\mathrm{pref}}} \!\left[ \log \sigma\!\big(r_\phi(x, y_w) - r_\phi(x, y_l) - m(y_w, y_l)\big) \right],
> $$
>
> where $m(y_w, y_l) = 0$ recovers the standard objective and $m(y_w, y_l) > 0$ incorporates the Llama 2 annotator-rating margin [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 15, p. 180).

> [!tip] Shift Invariance & Reward Model Taxonomy
> Because Eq. `(eq:align:bt_prob)` depends solely on $r_\phi(x, y_w) - r_\phi(x, y_l)$, adding any prompt-dependent constant $c(x)$ leaves preference probabilities unchanged. In practice, practitioners distinguish four supervision granularities [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 5, p. 44) [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 4, p. 198) [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Ch. 6, p. 148): (i) **Bradley-Terry RMs** assign a sequence-level scalar score at $\langle\mathrm{EOS}\rangle$ from pairwise comparisons; (ii) **Outcome Reward Models (ORMs)** predict per-token correctness probabilities trained via binary cross-entropy against a final verifiable outcome; (iii) **Process Reward Models (PRMs)** supervise intermediate reasoning step boundaries (e.g., $\{-1, 0, +1\}$ correctness labels) to catch flawed reasoning paths that accidentally reach a correct final answer; and (iv) **Value Functions** $V_\psi(s_t)$ regress expected future return $\mathbb{E}[G_t \mid s_t]$ for on-policy advantage estimation.

### KL-Regularized RLHF Objective & Closed-Form Gibbs Policy

> [!important] Variational Derivation of the Optimal RLHF Policy
> Directly maximizing $\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}[r_\phi(x,y)]$ collapses $\pi_\theta$ onto a single high-reward mode and induces reward hacking. To anchor $\pi_\theta$ near $\pi_{\mathrm{ref}}$, RLHF maximizes the KL-regularized objective [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 8, pp. 110–111) [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Ch. 6, pp. 154–155):
>
> $$
> \max_{\pi}   \mathbb{E}_{x \sim \mathcal{D}} \Big[ \mathbb{E}_{y \sim \pi(\cdot \mid x)}[r_\phi(x,y)] - \beta D_{\mathrm{KL}}\!\big(\pi(y \mid x)  \|  \pi_{\mathrm{ref}}(y \mid x)\big) \Big].
> $$
>
> Expanding $D_{\mathrm{KL}}(\pi  \|  \pi_{\mathrm{ref}}) = \mathbb{E}_{y \sim \pi}\big[\log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}\big]$, negating to form a minimization, dividing by $\beta > 0$, and introducing the prompt-dependent partition function
>
> $$
> Z(x) = \sum_{y} \pi_{\mathrm{ref}}(y \mid x) \exp\!\left(\frac{1}{\beta} r_\phi(x,y)\right),
> $$
>
> we rewrite Eq. `(eq:align:rlhf_obj)` as minimizing
>
> $$
> \min_{\pi}   \mathbb{E}_{x \sim \mathcal{D}} \!\left[ D_{\mathrm{KL}}\!\left( \pi(y \mid x)  \Big\|  \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp\!\left(\frac{1}{\beta} r_\phi(x,y)\right) \right) - \log Z(x) \right].
> $$
>
> Since $\log Z(x)$ is independent of $\pi$ and $D_{\mathrm{KL}}(p  \|  q) \ge 0$ with equality if and only if $p = q$ almost everywhere, the unique global optimum is the exponential-tilted Gibbs policy:
>
> $$
> \pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp\!\left(\frac{1}{\beta} r_\phi(x,y)\right).
> $$

> [!tip] Control as Inference & Posterior Sampling
> Murphy [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Ch. 6, p. 155) highlights that Eq. `(eq:align:gibbs_opt)` is identically Bayes' rule in the control-as-inference framework: treating $\pi_{\mathrm{ref}}(y \mid x)$ as a prior and $\exp(r_\phi(x,y)/\beta)$ as the likelihood $p(O=1 \mid x,y)$ of an optimality event $O=1$, RLHF performs amortized posterior sampling from $p(y \mid x, O=1) \propto \pi_{\mathrm{ref}}(y \mid x)\exp(r_\phi(x,y)/\beta)$.

### Exact DPO Reparameterization & Gradient Mechanics

> [!important] Direct Preference Optimization (DPO) Change of Variables
> Taking logarithms of Eq. `(eq:align:gibbs_opt)` and solving for $r_\phi(x,y)$ expresses the reward exactly in terms of the optimal policy $\pi^*$, the reference $\pi_{\mathrm{ref}}$, and the intractable partition function $Z(x)$ [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 8, pp. 111–113) [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Sec. 8.4.2, p. 222):
>
> $$
> r(x,y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \beta \log Z(x).
> $$
>
> Substituting Eq. `(eq:align:dpo_reparam)` into the Bradley-Terry likelihood `(eq:align:bt_prob)` causes $\beta \log Z(x)$ to cancel identically between $y_w$ and $y_l$:
>
> $$
> P^*(y_w \succ y_l \mid x) = \sigma\!\left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \right).
> $$
>
> Replacing $\pi^*$ with a parametric policy $\pi_\theta$ yields the DPO maximum-likelihood objective:
>
> $$
> \mathcal{L}_{\mathrm{DPO}}(\theta; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\mathrm{pref}}} \!\left[ \log \sigma\!\left( \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l) \right) \right],
> $$
>
> where $\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}$ is the *implicit reward*. Differentiating Eq. `(eq:align:dpo_loss)` with respect to $\theta$ gives
>
> $$
> \nabla_\theta \mathcal{L}_{\mathrm{DPO}}(\theta) = -\beta \mathbb{E}_{(x, y_w, y_l)} \Big[ \underbrace{\sigma\!\big(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\big)}_{\text{misranking weight}} \big( \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \big) \Big].
> $$
>
> The scalar weight $\sigma(\hat{r}_\theta(x,y_l) - \hat{r}_\theta(x,y_w))$ vanishes when the policy already ranks $y_w$ well above $y_l$ and amplifies updates when the implicit reward model misranks the pair [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Sec. 8.4.2, p. 222).

### Transfer to Continuous/Denoising Domains: Diffusion-DPO

In diffusion models, a sample $x_0$ is generated via a $T$-step Markov reverse chain $x_{0:T}$ conditioned on prompt $c$, with $\pi_\theta(x_{0:T} \mid c) = \pi_\theta(x_T \mid c)\prod_{t=1}^T \pi_\theta(x_{t-1} \mid x_t, c)$. Evaluating the marginal likelihood $\pi_\theta(x_0 \mid c)$ is intractable, and backpropagating through all $T \sim 10^2\text{–}10^3$ denoising steps is memory-prohibitive [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 8.5, pp. 250–252). Applying Jensen's inequality to the convex function $-\log \sigma(\cdot)$ over posterior trajectories $x_{1:T} \sim \pi_\theta(\cdot \mid x_0, c)$ and sampling a single timestep $t \sim \mathcal{U}\{1,\dots,T\}$ yields the tractable single-step Diffusion-DPO upper bound:

$$
\tilde{\mathcal{L}}_{\mathrm{Diff\text{-}DPO}}(\theta; \pi_{\mathrm{ref}}) = \mathbb{E}_{(c, x_0^w, x_0^l),  t,  \epsilon \sim \mathcal{N}(0,I)} \!\left[ -\log \sigma\!\big(-\beta T   w(t)   [\Delta_{\mathrm{MSE}}(x_t^w) - \Delta_{\mathrm{MSE}}(x_t^l)]\big) \right],
$$

where $\Delta_{\mathrm{MSE}}(x_t) = \|\hat{\epsilon}_\theta(x_t, t, c) - \epsilon\|_2^2 - \|\hat{\epsilon}_{\mathrm{ref}}(x_t, t, c) - \epsilon\|_2^2$ and $x_t^w, x_t^l$ share the same Gaussian noise $\epsilon$ for variance reduction [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 8.5.4, p. 252).

## Algorithmic Implementation & Comparative Synthesis

### From Critic-Based PPO to Critic-Free RLOO, GRPO, GSPO, and RLVR

While PPO optimizes a clipped surrogate objective with per-token Generalized Advantage Estimation (GAE) from a learned value network $V_\psi(s_t)$, [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 6, pp. 64–72) notes that maintaining a critic comparable in size to the policy doubles memory overhead and introduces value-estimation bias. Modern post-training increasingly favors *group-relative, critic-free* estimators that sample $G$ completions $\{y_1, \dots, y_G\} \sim \pi_{\theta_{\mathrm{old}}}(\cdot \mid x)$ per prompt:
\begin{sloppypar}
- **RLOO (REINFORCE Leave-One-Out):** Computes an unbiased leave-one-out baseline $b(x, y_i) = \frac{1}{G-1}\sum_{j \neq i} R(x, y_j)$ so that $A_i^{\mathrm{RLOO}} = R(x, y_i) - b(x, y_i)$ [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 6, p. 58).
- **GRPO (Group Relative Policy Optimization):** Normalizes rewards within each group via $A_i = \frac{r_i - \mathrm{mean}(\{r_j\}_{j=1}^G)}{\mathrm{std}(\{r_j\}_{j=1}^G)}$, applying an explicit per-token KL penalty in the loss rather than in the reward [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Ch. 6, p. 151) [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 6, pp. 66–67).
- **GSPO (Group Sequence Policy Optimization):** Replaces high-variance token ratios $\rho_{i,t}(\theta)$ with a length-normalized sequence ratio $\rho_i(\theta) = \big(\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\mathrm{old}}}(y_i \mid x)}\big)^{1/|y_i|}$, matching importance sampling to sequence rewards [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 6, p. 70).
- **RLVR (RL with Verifiable Rewards):** Replaces the learned Bradley-Terry RM with a deterministic verifier $R_{\mathrm{ver}}(x,y) \in \{0, 1\}$ (e.g., symbolic math equivalence on `\boxed{}` or code unit tests), enabling large-scale reasoning without reward hacking [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Ch. 6, p. 148) [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 7, pp. 95–96).

\end{sloppypar}

> [!note] Source Figure
> *Figure:* Comparison of (a) three-stage RLHF (PPO) requiring an explicit Reward Model and Value Function versus (b) Direct Preference Optimization (DPO), which fits the policy directly on pairwise preference data via closed-form reparameterization [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 4, Fig. 4.10, p. 193). (`foundations_large_language_models_p200_fig1.png`)

> [!note] Source Figure
> *Figure:* Three baseline-estimation paradigms for token-level advantage $A_t$: (a) critic-free group Monte Carlo baselines (RLOO / GRPO), (b) parametric value network $V_\phi(s_t)$ in PPO, and (c) per-state Monte Carlo rollouts in VinePPO [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Ch. 6, Fig. 6.1, p. 154). (`reinforement_learning_kevin_murphy_p154_fig1.png`)

> [!example] Visual Diagram (LaTeX / PDF)
> *Architectural comparison of post-training pipelines: (top) three-stage RLHF with an explicit Bradley-Terry RM and critic-based PPO; (middle) offline Direct Preference Optimization eliminating both the RM and RL loop; (bottom) RLVR replacing the learned RM with a deterministic verifier and critic-free GRPO.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Algorithm: Unified Post-Training: DPO (Offline) vs. GRPO / RLVR (Online Group-Relative)
> **Input:** Reference policy $\pi_{\mathrm{ref}}$, initial policy $\pi_\theta \leftarrow \pi_{\mathrm{ref}}$, KL coefficient $\beta$, clip $\varepsilon$, mode $\in \{\mathrm{DPO}, \mathrm{GRPO}\}$
> **Output:** Aligned policy $\pi_\theta$
> **if** $\mathrm{mode} = \mathrm{DPO}$:
>
>   Pre-compute and cache $\log \pi_{\mathrm{ref}}(y_w \mid x)$ and $\log \pi_{\mathrm{ref}}(y_l \mid x)$ over $\mathcal{D}_{\mathrm{pref}}$
>   **for** each minibatch $(x, y_w, y_l) \sim \mathcal{D}_{\mathrm{pref}}$:
>
>     Compute $\Delta_{\mathrm{chosen}} \leftarrow \log \pi_\theta(y_w \mid x) - \log \pi_{\mathrm{ref}}(y_w \mid x)$
>     Compute $\Delta_{\mathrm{rej}} \leftarrow \log \pi_\theta(y_l \mid x) - \log \pi_{\mathrm{ref}}(y_l \mid x)$
>     Update $\theta$ by minimizing $\mathcal{L}_{\mathrm{DPO}} = -\log \sigma\!\big(\beta(\Delta_{\mathrm{chosen}} - \Delta_{\mathrm{rej}})\big)$
>
>
> **else**:
>
>   **for** each prompt batch $x \sim \mathcal{D}$:
>
>     Sample $G$ completions $\{y_1, \dots, y_G\} \sim \pi_{\theta_{\mathrm{old}}}(\cdot \mid x)$ and score $r_i = R_{\mathrm{ver}}(x, y_i)$
>     Compute group-relative advantages $A_i \leftarrow \frac{r_i - \mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G}) + \delta}$
>     Update $\theta$ for $K$ epochs maximizing $\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \min(\rho_{i,t} A_i, \mathrm{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)A_i) - \beta \hat{D}_{\mathrm{KL}}$

**Table:** Computational, memory, and statistical comparison of LLM post-training optimizers.

| **Method** | **Models** | **Rollouts** | **Critic** | **On-Policy** | **Primary Trade-off** |
| — | — | — | — | — | — |
| PPO (RLHF) | 4 ($\pi_\theta, \pi_{\mathrm{ref}}, r_\phi, V_\psi$) | $1$ (or $G$) | Yes | Yes | Highest memory; per-token GAE credit |
| RLOO | 3 ($\pi_\theta, \pi_{\mathrm{ref}}, r_\phi$) | $G \ge 2$ | No | Yes | Unbiased LOO baseline; sequence advantage |
| GRPO (RLVR) | 2 ($\pi_\theta, \pi_{\mathrm{ref}}$) | $G \ge 4$ | No | Yes | No RM or critic; excels on math and code |
| DPO (Offline) | 2 (1 cached) | $0$ ($\mathcal{D}$) | No | No | Simplest stack; vulnerable to OOD shift |

> [!info] Cross-Book Synthesis: Online RL, Direct Alignment, and Constitutional AI
> All five sources converge on a fundamental trade-off between *optimization stability* and *on-policy exploration*. While Jurafsky & Martin [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Sec. 8.4.2, p. 222) and *The Principles of Diffusion Models* [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 8.5.1, p. 243) emphasize DPO's elimination of unstable RL loops and multi-model hosting, *Foundations of Large Language Models* [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 4, p. 196) and Lambert [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 8, p. 116) note that offline DPO cannot explore beyond its static preference dataset $\mathcal{D}_{\mathrm{pref}}$, whereas online PPO/GRPO continuously discovers novel reasoning strategies. To scale preference annotation without human bottlenecks, Constitutional AI (CAI / RLAIF) [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 12, p. 156) [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 4, p. 197) uses an LLM guided by a written constitution $\mathcal{C}$ to critique SFT drafts and label pairwise preferences, while variants such as SimPO and KTO modify DPO to remove the reference model or handle unpaired binary feedback.

## Visual Diagnostics, KL Geometry & Goodhart Over-Optimization

As shown in [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 15, pp. 184–185), a central geometric distinction in post-training separates **forward KL** $D_{\mathrm{KL}}(\pi_{\mathrm{data}}  \|  \pi_\theta)$ minimized by SFT from **reverse KL** $D_{\mathrm{KL}}(\pi_\theta  \|  \pi_{\mathrm{ref}})$ minimized by RLHF and DPO. Forward KL is *mode-covering*: forcing $\pi_\theta$ to cover new target modes stretches probability mass across low-density regions and erodes prior knowledge modes (catastrophic forgetting). Reverse KL is *mode-seeking*: $\pi_\theta$ concentrates on high-reward modes while leaving prior modes largely intact (Figure `fig:align:forward_reverse_kl`). To estimate reverse KL in GRPO/PPO without excessive variance, Murphy [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Ch. 6, p. 154) notes three standard Monte Carlo estimators for $r(a) = \pi_\theta(a \mid s)/\pi_{\mathrm{ref}}(a \mid s)$: $k_1 = \log r$, $k_2 = \frac{1}{2}(\log r)^2$ (biased value, unbiased/low-variance gradient), and $k_3 = (r - 1) - \log r$ (unbiased value, but biased gradient).

> [!note] Source Figure
> *Figure:* Forgetting dynamics under forward KL (SFT, top) versus reverse KL (RL, bottom): forward KL stretches probability mass and drains the prior knowledge mode, whereas reverse KL preserves the prior mode while shifting the target mode toward $\pi^*$ [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 15, Fig. 46, p. 184). (`reinforcement_learning_from_human_feedback_p190_fig1.png`)

Even with reverse-KL regularization, [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 14, pp. 174–175) and [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 4, p. 192) show that optimizing a learned proxy reward $R_{\mathrm{proxy}}$ induces **Goodhart's Law**: as the policy drifts further from $\pi_{\mathrm{ref}}$ in KL distance $d = \sqrt{D_{\mathrm{KL}}(\pi_\theta  \|  \pi_{\mathrm{ref}})}$, the proxy score climbs monotonically while the true (gold) human preference score rises initially and then collapses. Figure `fig:align:goodhart_kl_curve` plots this characteristic quadratic divergence $R_{\mathrm{gold}}(d) = \alpha d - \beta d^2$.

> [!example] Visual Diagram (LaTeX / PDF)
> *Goodhart's Law scaling in KL-regularized RLHF: the learned proxy reward $R_{\mathrm{proxy}}$ increases monotonically with $\sqrt{D_{\mathrm{KL}}(\pi_\theta  \|  \pi_{\mathrm{ref}})}$, whereas the true gold reward $R_{\mathrm{gold}}(d) = \alpha d - \beta d^2$ peaks at $d^* = \alpha / (2\beta)$ and degrades under reward hacking.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Reward Hacking & Goodhart Over-Optimization
> **Failure Mode:** Excessive RL steps against a single learned Bradley-Terry RM exploit spurious correlations (e.g., verbosity, sycophancy, or formatting artifacts), driving $R_{\mathrm{proxy}}$ up while true quality $R_{\mathrm{gold}}$ collapses (Figure `fig:align:goodhart_kl_curve`).\\
> **Remedy:** Monitor $\sqrt{D_{\mathrm{KL}}(\pi_\theta  \|  \pi_{\mathrm{ref}})}$ and halt before $d^*$; average an ensemble of diverse RMs $r_{\mathrm{combine}}(x,y) = \frac{1}{K}\sum_{k=1}^K w_k r_k(x,y)$ [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 4, Eq. (4.72), p. 192); or replace KL with $\chi^2$-divergence regularization, which is provably robust to over-optimization [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Ch. 6, p. 154).

> [!warning] DPO Preference Displacement & Likelihood Collapse
> **Failure Mode:** Because $\mathcal{L}_{\mathrm{DPO}}$ only maximizes the *relative* margin $\hat{r}_\theta(x,y_w) - \hat{r}_\theta(x,y_l)$, both $\pi_\theta(y_w \mid x)$ and $\pi_\theta(y_l \mid x)$ can decrease simultaneously while their gap widens, degrading fluency [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 8, p. 115).\\
> **Remedy:** Add a length-normalized negative log-likelihood term on the chosen response, $\mathcal{L}_{\mathrm{DPO+NLL}} = \mathcal{L}_{\mathrm{DPO}} - \frac{\alpha}{|y_w|}\log \pi_\theta(y_w \mid x)$ [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 15, Eq. (159), p. 180).

> [!warning] Token-Level Importance Ratio Instability & Clipped Token Dropping
> **Failure Mode:** In PPO/GRPO on long reasoning chains or sparse MoE models, per-token ratios $\rho_{i,t}(\theta)$ exhibit extreme variance, causing individual tokens to dominate updates or hit the clipping boundary $[1-\varepsilon, 1+\varepsilon]$ and receive zero gradient ("token dropping") [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 6, pp. 69–70).\\
> **Remedy:** Use GSPO's length-normalized sequence-level importance ratio $\rho_i(\theta)$ and compute KL penalties with the low-variance $k_2 = \frac{1}{2}(\log r)^2$ estimator rather than $k_3$ [[Source - Reinforcement Learning - An Overview|reinforement_learning_kevin_murphy]] (Ch. 6, p. 154).

> [!warning] Reference Model Memory Bottleneck in DPO
> **Failure Mode:** Naively co-locating $\pi_\theta$ and $\pi_{\mathrm{ref}}$ during DPO forward passes doubles activation and weight memory.\\
> **Remedy:** Pre-compute and cache $\log \pi_{\mathrm{ref}}(y_w \mid x)$ and $\log \pi_{\mathrm{ref}}(y_l \mid x)$ over the offline dataset before training, cutting peak GPU memory by $50\%$ [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 8, p. 116).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (4):** [[Bradley-Terry Reward Model]], [[DPO (Direct Preference Optimization)]], [[RLHF (KL-Regularized)]], [[RLVR (Verifiable Rewards)]]
- **Prerequisites (Directed $\leftarrow$):** [[Bradley-Terry Reward Model]], [[GRPO]], [[KL Divergence]], [[PPO (Proximal Policy Optimization)]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Bayesian Personalized Ranking (BPR)]], [[DPO (Direct Preference Optimization)]], [[RLHF (KL-Regularized)]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Bradley-Terry Reward Model]], [[Chain-of-Thought & Test-Time Compute]], [[DPO (Direct Preference Optimization)]], [[GRPO]], [[KL Divergence]], [[LoRA & QLoRA]], [[PPO (Proximal Policy Optimization)]], [[RLHF (KL-Regularized)]]
- **Synthesized Sources:** [[Source - Foundations of Large Language Models]], [[Source - Reinforcement Learning - An Overview]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Speech and Language Processing]], [[Source - The Principles of Diffusion Models]]
