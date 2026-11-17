---
type: algorithm
domain: "D06"
aliases: 
  - "RLVR"
part_of: 
  - "[[Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]"
  - "[[Cheatsheet - Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[GRPO]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Chain-of-Thought & Test-Time Compute]]"
  - "[[GRPO]]"
contrasts_with: 
  - "[[RLHF (KL-Regularized)]]"
dual_to: []
sources: 
  - "[[Source - Foundations of Large Language Models]]"
  - "[[Source - Reinforcement Learning - An Overview]]"
  - "[[Source - Reinforcement Learning from Human Feedback]]"
  - "[[Source - Speech and Language Processing]]"
  - "[[Source - The Principles of Diffusion Models]]"
tags:
  - algorithm
  - domain-06
---

# RLVR (Verifiable Rewards)

> [!abstract] Definition (Algorithm)
> Trains reasoning policies via GRPO against deterministic programmatic verifiers (math/code unit tests) instead of learned reward models.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[GRPO]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Chain-of-Thought & Test-Time Compute]], [[GRPO]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[RLHF (KL-Regularized)]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]
- **Visual Cheatsheet:** [[Cheatsheet - Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]
- **Referenced Across Monographs:** [[Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]
- **Primary Textbooks:** [[Source - Foundations of Large Language Models]], [[Source - Reinforcement Learning - An Overview]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Speech and Language Processing]], [[Source - The Principles of Diffusion Models]]
