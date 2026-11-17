---
type: algorithm
domain: "D06"
aliases: 
  - "MBPO"
  - "World Models"
part_of: 
  - "[[Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]"
  - "[[Cheatsheet - Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[Markov Decision Process (MDP)]]"
enables: 
  - "[[MCTS & MuZero]]"
generalizes: []
special_case_of: []
related: 
  - "[[MCTS & MuZero]]"
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

# Model-Based RL (Dyna & MBPO)

> [!abstract] Definition (Algorithm)
> Learns dynamics model to generate short k-step branched rollouts (MBPO), bounding compounding model error.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Markov Decision Process (MDP)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[MCTS & MuZero]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[MCTS & MuZero]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]
- **Visual Cheatsheet:** [[Cheatsheet - Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]
- **Referenced Across Monographs:** [[Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]
- **Primary Textbooks:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]]
