---
type: algorithm
domain: "D06"
aliases: 
  - "Q-Learning"
  - "DQN"
part_of: 
  - "[[Monte Carlo, SARSA, Q-Learning, DQN, Eligibility Traces & the Deadly Triad]]"
  - "[[Cheatsheet - Monte Carlo, SARSA, Q-Learning, DQN, Eligibility Traces & the Deadly Triad]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[MC & TD Learning]]"
enables: 
  - "[[Deadly Triad]]"
  - "[[Offline RL (CQL)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Bellman Equations]]"
  - "[[Deadly Triad]]"
  - "[[MC & TD Learning]]"
  - "[[Offline RL (CQL)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Francis Bach]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Reinforcement Learning - An Overview]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - algorithm
  - domain-06
---

# Q-Learning & DQN

> [!abstract] Definition (Algorithm)
> Off-policy TD control learning Q*; stabilized at scale via experience replay, target networks, and Double-Q.

$$
y_t = r_t + \gamma \max_{a'} Q_{\bar{\boldsymbol{w}}}(s_{t+1}, a')
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[MC & TD Learning]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Deadly Triad]], [[Offline RL (CQL)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Bellman Equations]], [[Deadly Triad]], [[MC & TD Learning]], [[Offline RL (CQL)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Monte Carlo, SARSA, Q-Learning, DQN, Eligibility Traces & the Deadly Triad]]
- **Visual Cheatsheet:** [[Cheatsheet - Monte Carlo, SARSA, Q-Learning, DQN, Eligibility Traces & the Deadly Triad]]
- **Referenced Across Monographs:** [[Monte Carlo, SARSA, Q-Learning, DQN, Eligibility Traces & the Deadly Triad]]
- **Primary Textbooks:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]], [[Source - Understanding Deep Learning]]
