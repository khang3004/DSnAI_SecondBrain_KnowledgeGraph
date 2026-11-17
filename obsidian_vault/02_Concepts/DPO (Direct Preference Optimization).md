---
type: algorithm
domain: "D06"
aliases: 
  - "DPO"
part_of: 
  - "[[Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]"
  - "[[Cheatsheet - Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]"
  - "[[D06 - Reinforcement Learning, Control & LLM Alignment]]"
prerequisites: 
  - "[[Bradley-Terry Reward Model]]"
  - "[[KL Divergence]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Bradley-Terry Reward Model]]"
  - "[[KL Divergence]]"
  - "[[LoRA & QLoRA]]"
contrasts_with: 
  - "[[PPO (Proximal Policy Optimization)]]"
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

# DPO (Direct Preference Optimization)

> [!abstract] Definition (Algorithm)
> Reparameterizes the closed-form Gibbs optimum of KL-RLHF into a direct binary cross-entropy loss on policy log-ratios without an explicit RM.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)\right]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Bradley-Terry Reward Model]], [[KL Divergence]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Bradley-Terry Reward Model]], [[KL Divergence]], [[LoRA & QLoRA]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[PPO (Proximal Policy Optimization)]], [[RLHF (KL-Regularized)]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]
- **Visual Cheatsheet:** [[Cheatsheet - Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]
- **Referenced Across Monographs:** [[Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR]]
- **Primary Textbooks:** [[Source - Foundations of Large Language Models]], [[Source - Reinforcement Learning - An Overview]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Speech and Language Processing]], [[Source - The Principles of Diffusion Models]]
