---
type: algorithm
domain: "D06"
aliases: 
  - "UCB"
part_of: 
  - "[[Multi-Armed & Contextual Bandits, UCB, Thompson Sampling & Regret Bounds]]"
  - "[[Cheatsheet - Multi-Armed & Contextual Bandits, UCB, Thompson Sampling & Regret Bounds]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[Concentration Inequalities]]"
enables: 
  - "[[MCTS & MuZero]]"
generalizes: []
special_case_of: []
related: 
  - "[[Concentration Inequalities]]"
  - "[[MCTS & MuZero]]"
contrasts_with: 
  - "[[Thompson Sampling]]"
dual_to: []
sources: 
  - "[[Source - Francis Bach]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Reinforcement Learning - An Overview]]"
tags:
  - algorithm
  - domain-06
---

# Multi-Armed Bandits & UCB

> [!abstract] Definition (Algorithm)
> Stateless exploration-exploitation trade-off; Upper Confidence Bound (UCB) achieves logarithmic O(log T) regret via optimism under uncertainty.

$$
A_t = \arg\max_a \left[\hat{Q}_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}}\right]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Concentration Inequalities]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[MCTS & MuZero]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Concentration Inequalities]], [[MCTS & MuZero]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Thompson Sampling]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Multi-Armed & Contextual Bandits, UCB, Thompson Sampling & Regret Bounds]]
- **Visual Cheatsheet:** [[Cheatsheet - Multi-Armed & Contextual Bandits, UCB, Thompson Sampling & Regret Bounds]]
- **Referenced Across Monographs:** [[Multi-Armed & Contextual Bandits, UCB, Thompson Sampling & Regret Bounds]]
- **Primary Textbooks:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]]
