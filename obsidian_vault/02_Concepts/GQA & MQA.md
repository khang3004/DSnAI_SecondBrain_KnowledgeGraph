---
type: architecture
domain: "D05"
aliases: []
part_of: 
  - "[[Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]"
  - "[[Cheatsheet - Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]"
  - "[[D05 - Transformers, Language Modeling & LLM Engineering]]"
prerequisites: 
  - "[[Self-Attention & Multi-Head Attention]]"
enables: 
  - "[[KV Cache & PagedAttention]]"
generalizes: []
special_case_of: []
related: 
  - "[[KV Cache & PagedAttention]]"
  - "[[Self-Attention & Multi-Head Attention]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Speech and Language Processing]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - architecture
  - domain-05
---

# GQA & MQA

> [!abstract] Definition (Architecture)
> Grouped-Query and Multi-Query Attention share K/V heads across Q-head groups to slash autoregressive KV cache bandwidth.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Self-Attention & Multi-Head Attention]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[KV Cache & PagedAttention]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[KV Cache & PagedAttention]], [[Self-Attention & Multi-Head Attention]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]
- **Visual Cheatsheet:** [[Cheatsheet - Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]
- **Referenced Across Monographs:** [[Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Speech and Language Processing]], [[Source - Understanding Deep Learning]]
