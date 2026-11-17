---
type: theorem
domain: "D06"
aliases: []
part_of: 
  - "[[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]"
  - "[[Cheatsheet - Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[Markov Decision Process (MDP)]]"
enables: 
  - "[[Generalized Advantage Estimation (GAE)]]"
  - "[[TRPO]]"
generalizes: []
special_case_of: []
related: 
  - "[[Generalized Advantage Estimation (GAE)]]"
  - "[[PPO (Proximal Policy Optimization)]]"
  - "[[TRPO]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Francis Bach]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Reinforcement Learning - An Overview]]"
  - "[[Source - Reinforcement Learning from Human Feedback]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - theorem
  - domain-06
---

# Policy Gradient Theorem

> [!abstract] Definition (Theorem)
> Score-function gradient of expected return avoids differentiating environment dynamics via log-derivative trick and baseline subtraction.

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{s\sim\mu,\,a\sim\pi_{\boldsymbol{\theta}}}\!\bigl[A^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(a\mid s)\bigr]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Markov Decision Process (MDP)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Generalized Advantage Estimation (GAE)]], [[TRPO]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Generalized Advantage Estimation (GAE)]], [[PPO (Proximal Policy Optimization)]], [[TRPO]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Visual Cheatsheet:** [[Cheatsheet - Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Referenced Across Monographs:** [[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Primary Textbooks:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Understanding Deep Learning]]
