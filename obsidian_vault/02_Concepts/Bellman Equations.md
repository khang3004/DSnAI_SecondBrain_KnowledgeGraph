---
type: theorem
domain: "D06"
aliases: []
part_of: 
  - "[[MDPs, Bellman Optimality Equations, Policy - Value Iteration & GPI]]"
  - "[[Cheatsheet - MDPs, Bellman Optimality Equations, Policy - Value Iteration & GPI]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[Markov Decision Process (MDP)]]"
enables: 
  - "[[MC & TD Learning]]"
  - "[[Value & Policy Iteration]]"
generalizes: []
special_case_of: []
related: 
  - "[[Markov Decision Process (MDP)]]"
  - "[[Q-Learning & DQN]]"
  - "[[Value & Policy Iteration]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Francis Bach]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Reinforcement Learning - An Overview]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - theorem
  - domain-06
---

# Bellman Equations

> [!abstract] Definition (Theorem)
> Recursive fixed-point equations for V^pi and Q^*; gamma-contraction in L_infty norm guarantees unique value fixed points.

$$
(\mathcal{T}^* V)(s) = \max_{a\in\mathcal{A}} \sum_{s',r} p(s',r\mid s,a)\bigl[r + \gamma V(s')\bigr]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Markov Decision Process (MDP)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[MC & TD Learning]], [[Value & Policy Iteration]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Markov Decision Process (MDP)]], [[Q-Learning & DQN]], [[Value & Policy Iteration]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[MDPs, Bellman Optimality Equations, Policy - Value Iteration & GPI]]
- **Visual Cheatsheet:** [[Cheatsheet - MDPs, Bellman Optimality Equations, Policy - Value Iteration & GPI]]
- **Referenced Across Monographs:** [[MDPs, Bellman Optimality Equations, Policy - Value Iteration & GPI]]
- **Primary Textbooks:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]], [[Source - Understanding Deep Learning]]
