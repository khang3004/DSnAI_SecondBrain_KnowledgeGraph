---
type: algorithm
domain: "D06"
aliases: 
  - "GAE"
part_of: 
  - "[[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]"
  - "[[Cheatsheet - Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[MC & TD Learning]]"
  - "[[Policy Gradient Theorem]]"
enables: 
  - "[[PPO (Proximal Policy Optimization)]]"
generalizes: []
special_case_of: []
related: 
  - "[[MC & TD Learning]]"
  - "[[PPO (Proximal Policy Optimization)]]"
  - "[[Policy Gradient Theorem]]"
contrasts_with: []
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

# Generalized Advantage Estimation (GAE)

> [!abstract] Definition (Algorithm)
> Exponentially weighted sum of k-step TD residuals trading bias (lambda -> 0) against variance (lambda -> 1).

$$
\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[MC & TD Learning]], [[Policy Gradient Theorem]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[PPO (Proximal Policy Optimization)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[MC & TD Learning]], [[PPO (Proximal Policy Optimization)]], [[Policy Gradient Theorem]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Visual Cheatsheet:** [[Cheatsheet - Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Referenced Across Monographs:** [[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Primary Textbooks:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Understanding Deep Learning]]
