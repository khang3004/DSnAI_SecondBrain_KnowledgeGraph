---
type: algorithm
domain: "D06"
aliases: 
  - "RLHF"
part_of: 
  - "[[Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]"
  - "[[Cheatsheet - Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[Bradley-Terry Reward Model]]"
  - "[[KL Divergence]]"
  - "[[PPO (Proximal Policy Optimization)]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Bradley-Terry Reward Model]]"
  - "[[KL Divergence]]"
  - "[[LoRA & QLoRA]]"
  - "[[PPO (Proximal Policy Optimization)]]"
contrasts_with: 
  - "[[DPO (Direct Preference Optimization)]]"
  - "[[RLVR (Verifiable Rewards)]]"
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

# RLHF (KL-Regularized)

> [!abstract] Definition (Algorithm)
> Maximizes learned reward while penalizing KL divergence from reference SFT policy to prevent reward hacking.

$$
\max_\pi \; \mathbb{E}_{y\sim\pi(\cdot|x)}[r_\phi(x,y)] - \beta D_{\mathrm{KL}}\bigl(\pi(y|x)\,\|\,\pi_{\mathrm{ref}}(y|x)\bigr)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Bradley-Terry Reward Model]], [[KL Divergence]], [[PPO (Proximal Policy Optimization)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Bradley-Terry Reward Model]], [[KL Divergence]], [[LoRA & QLoRA]], [[PPO (Proximal Policy Optimization)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[DPO (Direct Preference Optimization)]], [[RLVR (Verifiable Rewards)]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]
- **Visual Cheatsheet:** [[Cheatsheet - Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]
- **Referenced Across Monographs:** [[Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]
- **Primary Textbooks:** [[Source - Foundations of Large Language Models]], [[Source - Reinforcement Learning - An Overview]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Speech and Language Processing]], [[Source - The Principles of Diffusion Models]]
