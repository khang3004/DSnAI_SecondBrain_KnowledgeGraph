---
type: algorithm
domain: "D06"
aliases: 
  - "MCTS"
  - "MuZero"
  - "AlphaZero"
part_of: 
  - "[[Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]"
  - "[[Cheatsheet - Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[Model-Based RL (Dyna & MBPO)]]"
  - "[[Multi-Armed Bandits & UCB]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Agent Planning & Multi-Agent Patterns]]"
  - "[[Chain-of-Thought & Test-Time Compute]]"
  - "[[Model-Based RL (Dyna & MBPO)]]"
  - "[[Multi-Armed Bandits & UCB]]"
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

# MCTS & MuZero

> [!abstract] Definition (Algorithm)
> Decision-time planning via PUCT tree search over learned latent value, policy, and reward dynamics without pixel reconstruction.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Model-Based RL (Dyna & MBPO)]], [[Multi-Armed Bandits & UCB]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Agent Planning & Multi-Agent Patterns]], [[Chain-of-Thought & Test-Time Compute]], [[Model-Based RL (Dyna & MBPO)]], [[Multi-Armed Bandits & UCB]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]
- **Visual Cheatsheet:** [[Cheatsheet - Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]
- **Referenced Across Monographs:** [[Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning]]
- **Primary Textbooks:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]]
