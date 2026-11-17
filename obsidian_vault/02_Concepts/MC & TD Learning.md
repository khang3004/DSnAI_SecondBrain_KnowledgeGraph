---
type: algorithm
domain: "D06"
aliases: 
  - "TD Learning"
  - "Temporal Difference"
part_of: 
  - "[[Monte Carlo, SARSA, Q-Learning, DQN, Eligibility Traces & the Deadly Triad]]"
  - "[[Cheatsheet - Monte Carlo, SARSA, Q-Learning, DQN, Eligibility Traces & the Deadly Triad]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[Bellman Equations]]"
enables: 
  - "[[Generalized Advantage Estimation (GAE)]]"
  - "[[Q-Learning & DQN]]"
generalizes: []
special_case_of: []
related: 
  - "[[Generalized Advantage Estimation (GAE)]]"
  - "[[Q-Learning & DQN]]"
contrasts_with: 
  - "[[Value & Policy Iteration]]"
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

# MC & TD Learning

> [!abstract] Definition (Algorithm)
> Model-free value estimation: unbiased high-variance Monte Carlo returns vs. biased low-variance bootstrapped TD(0) targets.

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Bellman Equations]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Generalized Advantage Estimation (GAE)]], [[Q-Learning & DQN]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Generalized Advantage Estimation (GAE)]], [[Q-Learning & DQN]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Value & Policy Iteration]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Monte Carlo, SARSA, Q-Learning, DQN, Eligibility Traces & the Deadly Triad]]
- **Visual Cheatsheet:** [[Cheatsheet - Monte Carlo, SARSA, Q-Learning, DQN, Eligibility Traces & the Deadly Triad]]
- **Referenced Across Monographs:** [[Monte Carlo, SARSA, Q-Learning, DQN, Eligibility Traces & the Deadly Triad]]
- **Primary Textbooks:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]], [[Source - Understanding Deep Learning]]
