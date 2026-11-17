---
type: algorithm
domain: "D05"
aliases: []
part_of: 
  - "[[Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]"
  - "[[Cheatsheet - Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]"
  - "[[D05 - Transformers, Language Modeling & LLM Engineering]]"
prerequisites: 
  - "[[KV Cache & PagedAttention]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Chain-of-Thought & Test-Time Compute]]"
  - "[[KV Cache & PagedAttention]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Foundations of Large Language Models]]"
  - "[[Source - Reinforcement Learning from Human Feedback]]"
  - "[[Source - Speech and Language Processing]]"
tags:
  - algorithm
  - domain-05
---

# Speculative Decoding

> [!abstract] Definition (Algorithm)
> Fast draft model proposes gamma tokens verified in parallel by target model with exact rejection sampling preserving target distribution.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[KV Cache & PagedAttention]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Chain-of-Thought & Test-Time Compute]], [[KV Cache & PagedAttention]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]
- **Visual Cheatsheet:** [[Cheatsheet - Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]
- **Referenced Across Monographs:** [[Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Foundations of Large Language Models]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Speech and Language Processing]]
