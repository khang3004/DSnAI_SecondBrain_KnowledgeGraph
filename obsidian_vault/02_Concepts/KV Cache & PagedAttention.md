---
type: algorithm
domain: "D05"
aliases: []
part_of: 
  - "[[Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]"
  - "[[Cheatsheet - Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]"
  - "[[D05 - Transformers, Language Modeling & LLM Engineering]]"
prerequisites: 
  - "[[GQA & MQA]]"
  - "[[Self-Attention & Multi-Head Attention]]"
enables: 
  - "[[Speculative Decoding]]"
generalizes: []
special_case_of: []
related: 
  - "[[Context Memory & Compaction]]"
  - "[[FlashAttention]]"
  - "[[GQA & MQA]]"
  - "[[Speculative Decoding]]"
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

# KV Cache & PagedAttention

> [!abstract] Definition (Algorithm)
> Caches past K,V tensors to avoid O(L^2) recomputation per token; PagedAttention eliminates fragmentation via OS-style virtual memory blocks.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[GQA & MQA]], [[Self-Attention & Multi-Head Attention]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Speculative Decoding]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Context Memory & Compaction]], [[FlashAttention]], [[GQA & MQA]], [[Speculative Decoding]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]
- **Visual Cheatsheet:** [[Cheatsheet - Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]
- **Referenced Across Monographs:** [[Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Foundations of Large Language Models]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Speech and Language Processing]]
