---
type: algorithm
domain: "D06"
aliases: 
  - "PPO"
part_of: 
  - "[[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]"
  - "[[Cheatsheet - Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[Generalized Advantage Estimation (GAE)]]"
  - "[[TRPO]]"
enables: 
  - "[[GRPO]]"
  - "[[RLHF (KL-Regularized)]]"
generalizes: []
special_case_of: []
related: 
  - "[[GRPO]]"
  - "[[Generalized Advantage Estimation (GAE)]]"
  - "[[Policy Gradient Theorem]]"
  - "[[RLHF (KL-Regularized)]]"
  - "[[Reparameterization Trick]]"
  - "[[TRPO]]"
contrasts_with: 
  - "[[DPO (Direct Preference Optimization)]]"
dual_to: []
sources: 
  - "[[Source - Francis Bach]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Reinforcement Learning - An Overview]]"
  - "[[Source - Reinforcement Learning from Human Feedback]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - algorithm
  - domain-06
---

# PPO (Proximal Policy Optimization)

> [!abstract] Definition (Algorithm)
> First-order trust-region proxy clipping probability ratios r_t(theta) within [1-eps, 1+eps]; workhorse of RL and RLHF.

$$
L^{\mathrm{CLIP}}(\boldsymbol{\theta}) = \mathbb{E}_t\!\left[\min\bigl(r_t(\boldsymbol{\theta})\hat{A}_t,\; \mathrm{clip}(r_t(\boldsymbol{\theta}), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\bigr)\right]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Generalized Advantage Estimation (GAE)]], [[TRPO]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[GRPO]], [[RLHF (KL-Regularized)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[GRPO]], [[Generalized Advantage Estimation (GAE)]], [[Policy Gradient Theorem]], [[RLHF (KL-Regularized)]], [[Reparameterization Trick]], [[TRPO]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[DPO (Direct Preference Optimization)]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Visual Cheatsheet:** [[Cheatsheet - Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Referenced Across Monographs:** [[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Primary Textbooks:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Understanding Deep Learning]]
