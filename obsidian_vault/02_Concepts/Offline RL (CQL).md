---
type: algorithm
domain: "D06"
aliases: 
  - "CQL"
  - "Offline RL"
part_of: 
  - "[[Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]"
  - "[[Cheatsheet - Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[Q-Learning & DQN]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Q-Learning & DQN]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Francis Bach]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Reinforcement Learning - An Overview]]"
tags:
  - algorithm
  - domain-06
---

# Offline RL (CQL)

> [!abstract] Definition (Algorithm)
> Conservative Q-Learning penalizes OOD action values outside the behavior dataset support to prevent extrapolation overestimation.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Q-Learning & DQN]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Q-Learning & DQN]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]
- **Visual Cheatsheet:** [[Cheatsheet - Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]
- **Referenced Across Monographs:** [[Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]
- **Primary Textbooks:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]]
