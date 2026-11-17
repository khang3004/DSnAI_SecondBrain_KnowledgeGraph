---
type: object
domain: "D06"
aliases: []
part_of: 
  - "[[Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]"
  - "[[Cheatsheet - Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: []
enables: 
  - "[[Bayesian Personalized Ranking (BPR)]]"
  - "[[DPO (Direct Preference Optimization)]]"
  - "[[RLHF (KL-Regularized)]]"
generalizes: []
special_case_of: []
related: 
  - "[[DPO (Direct Preference Optimization)]]"
  - "[[RLHF (KL-Regularized)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Foundations of Large Language Models]]"
  - "[[Source - Reinforcement Learning - An Overview]]"
  - "[[Source - Reinforcement Learning from Human Feedback]]"
  - "[[Source - Speech and Language Processing]]"
  - "[[Source - The Principles of Diffusion Models]]"
tags:
  - object
  - domain-06
---

# Bradley-Terry Reward Model

> [!abstract] Definition (Object)
> Pairwise preference likelihood modeling P(y_w > y_l | x) = sigmoid(r(x,y_w) - r(x,y_l)) for scalar reward training.

$$
P(y_w \succ y_l \mid x) = \sigma\bigl(r^*(x,y_w) - r^*(x,y_l)\bigr)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Bayesian Personalized Ranking (BPR)]], [[DPO (Direct Preference Optimization)]], [[RLHF (KL-Regularized)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[DPO (Direct Preference Optimization)]], [[RLHF (KL-Regularized)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]
- **Visual Cheatsheet:** [[Cheatsheet - Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]
- **Referenced Across Monographs:** [[Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]
- **Primary Textbooks:** [[Source - Foundations of Large Language Models]], [[Source - Reinforcement Learning - An Overview]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Speech and Language Processing]], [[Source - The Principles of Diffusion Models]]
