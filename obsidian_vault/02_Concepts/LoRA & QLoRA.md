---
type: algorithm
domain: "D05"
aliases: 
  - "LoRA"
  - "QLoRA"
part_of: 
  - "[[Self-Supervised Pretraining, Scaling Laws, Instruction Fine-Tuning & LoRA - QLoRA]]"
  - "[[Cheatsheet - Self-Supervised Pretraining, Scaling Laws, Instruction Fine-Tuning & LoRA - QLoRA]]"
  - "[[D05 - Transformers, Language Modeling & LLM Engineering]]"
prerequisites: 
  - "[[Low-Rank Approximation (Eckart-Young)]]"
  - "[[Singular Value Decomposition (SVD)]]"
  - "[[Transformer Architecture]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Chinchilla Scaling Laws]]"
  - "[[DPO (Direct Preference Optimization)]]"
  - "[[Low-Rank Approximation (Eckart-Young)]]"
  - "[[RLHF (KL-Regularized)]]"
  - "[[Singular Value Decomposition (SVD)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Foundations of Large Language Models]]"
  - "[[Source - LLM Engineer’s Handbook]]"
  - "[[Source - Reinforcement Learning from Human Feedback]]"
  - "[[Source - Speech and Language Processing]]"
tags:
  - algorithm
  - domain-05
---

# LoRA & QLoRA

> [!abstract] Definition (Algorithm)
> Parameter-efficient adaptation via frozen 4-bit NF4 base weights plus trainable low-rank bypasses (alpha/r) B A.

$$
\boldsymbol{W} = \boldsymbol{W}_0 + \frac{\alpha}{r}\boldsymbol{B}\boldsymbol{A}, \quad \boldsymbol{B}=\boldsymbol{0},\;\boldsymbol{A}\sim\mathcal{N}(0,\sigma^2)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Low-Rank Approximation (Eckart-Young)]], [[Singular Value Decomposition (SVD)]], [[Transformer Architecture]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Chinchilla Scaling Laws]], [[DPO (Direct Preference Optimization)]], [[Low-Rank Approximation (Eckart-Young)]], [[RLHF (KL-Regularized)]], [[Singular Value Decomposition (SVD)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Self-Supervised Pretraining, Scaling Laws, Instruction Fine-Tuning & LoRA - QLoRA]]
- **Visual Cheatsheet:** [[Cheatsheet - Self-Supervised Pretraining, Scaling Laws, Instruction Fine-Tuning & LoRA - QLoRA]]
- **Referenced Across Monographs:** [[Self-Supervised Pretraining, Scaling Laws, Instruction Fine-Tuning & LoRA - QLoRA]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Foundations of Large Language Models]], [[Source - LLM Engineer’s Handbook]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Speech and Language Processing]]
