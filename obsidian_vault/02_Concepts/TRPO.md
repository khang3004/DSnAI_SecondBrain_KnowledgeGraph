---
type: algorithm
domain: "D06"
aliases: []
part_of: 
  - "[[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]"
  - "[[Cheatsheet - Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[KL Divergence]]"
  - "[[Lagrange Duality & KKT]]"
  - "[[Natural Gradient & K-FAC]]"
  - "[[Policy Gradient Theorem]]"
enables: 
  - "[[PPO (Proximal Policy Optimization)]]"
generalizes: []
special_case_of: []
related: 
  - "[[KL Divergence]]"
  - "[[Lagrange Duality & KKT]]"
  - "[[Natural Gradient & K-FAC]]"
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

# TRPO

> [!abstract] Definition (Algorithm)
> Monotonic policy improvement under a hard KL trust-region constraint solved with Fisher-vector products and conjugate gradients.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[KL Divergence]], [[Lagrange Duality & KKT]], [[Natural Gradient & K-FAC]], [[Policy Gradient Theorem]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[PPO (Proximal Policy Optimization)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[KL Divergence]], [[Lagrange Duality & KKT]], [[Natural Gradient & K-FAC]], [[PPO (Proximal Policy Optimization)]], [[Policy Gradient Theorem]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Visual Cheatsheet:** [[Cheatsheet - Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Referenced Across Monographs:** [[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Primary Textbooks:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Understanding Deep Learning]]
