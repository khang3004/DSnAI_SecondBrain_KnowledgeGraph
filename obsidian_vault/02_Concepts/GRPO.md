---
type: algorithm
domain: "D06"
aliases: []
part_of: 
  - "[[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]"
  - "[[Cheatsheet - Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[PPO (Proximal Policy Optimization)]]"
enables: 
  - "[[RLVR (Verifiable Rewards)]]"
generalizes: []
special_case_of: []
related: 
  - "[[PPO (Proximal Policy Optimization)]]"
  - "[[RLVR (Verifiable Rewards)]]"
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

# GRPO

> [!abstract] Definition (Algorithm)
> Group Relative Policy Optimization drops the critic network and normalizes rewards across G sampled completions per prompt.

$$
A_{i,t} = \frac{r_i - \mu_G}{\sigma_G + 10^{-8}}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[PPO (Proximal Policy Optimization)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[RLVR (Verifiable Rewards)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[PPO (Proximal Policy Optimization)]], [[RLVR (Verifiable Rewards)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Visual Cheatsheet:** [[Cheatsheet - Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Referenced Across Monographs:** [[Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO - PPO & RL-as-Inference]]
- **Primary Textbooks:** [[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Understanding Deep Learning]]
