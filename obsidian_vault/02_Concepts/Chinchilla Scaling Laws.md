---
type: theorem
domain: "D05"
aliases: []
part_of: 
  - "[[Self-Supervised Pretraining, Scaling Laws, Instruction Fine-Tuning & LoRA - QLoRA]]"
  - "[[Cheatsheet - Self-Supervised Pretraining, Scaling Laws, Instruction Fine-Tuning & LoRA - QLoRA]]"
  - "[[D05 - Transformers, Language Modeling & LLM Engineering]]"
prerequisites: 
  - "[[Transformer Architecture]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[LoRA & QLoRA]]"
  - "[[Transformer Architecture]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Foundations of Large Language Models]]"
  - "[[Source - LLM Engineer’s Handbook]]"
  - "[[Source - Reinforcement Learning from Human Feedback]]"
  - "[[Source - Speech and Language Processing]]"
tags:
  - theorem
  - domain-05
---

# Chinchilla Scaling Laws

> [!abstract] Definition (Theorem)
> Compute-optimal pretraining allocates tokens D proportional to parameters N (D approx 20 N) along C approx 6ND.

$$
L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}, \quad C \approx 6ND
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Transformer Architecture]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[LoRA & QLoRA]], [[Transformer Architecture]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Self-Supervised Pretraining, Scaling Laws, Instruction Fine-Tuning & LoRA - QLoRA]]
- **Visual Cheatsheet:** [[Cheatsheet - Self-Supervised Pretraining, Scaling Laws, Instruction Fine-Tuning & LoRA - QLoRA]]
- **Referenced Across Monographs:** [[Self-Supervised Pretraining, Scaling Laws, Instruction Fine-Tuning & LoRA - QLoRA]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Foundations of Large Language Models]], [[Source - LLM Engineer’s Handbook]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Speech and Language Processing]]
