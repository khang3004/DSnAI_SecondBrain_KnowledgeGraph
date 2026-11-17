---
type: architecture
domain: "D05"
aliases: 
  - "RoPE"
  - "Rotary Position Embedding"
part_of: 
  - "[[Words, Byte-Pair Encoding (BPE), Distributional Embeddings & Positional Encodings]]"
  - "[[Cheatsheet - Words, Byte-Pair Encoding (BPE), Distributional Embeddings & Positional Encodings]]"
  - "[[D05 - Transformers, Language Modeling & LLM Engineering]]"
prerequisites: []
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[BPE & Subword Tokenization]]"
  - "[[Self-Attention & Multi-Head Attention]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Speech and Language Processing]]"
tags:
  - architecture
  - domain-05
---

# RoPE & Positional Embeddings

> [!abstract] Definition (Architecture)
> Rotary Position Embedding encodes relative token displacement m-n directly into Q-K inner products via 2D rotation blocks.

$$
\langle \boldsymbol{R}_m \boldsymbol{q}, \boldsymbol{R}_n \boldsymbol{k} \rangle = \boldsymbol{q}^\top \boldsymbol{R}_{n-m} \boldsymbol{k}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[BPE & Subword Tokenization]], [[Self-Attention & Multi-Head Attention]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Words, Byte-Pair Encoding (BPE), Distributional Embeddings & Positional Encodings]]
- **Visual Cheatsheet:** [[Cheatsheet - Words, Byte-Pair Encoding (BPE), Distributional Embeddings & Positional Encodings]]
- **Referenced Across Monographs:** [[Words, Byte-Pair Encoding (BPE), Distributional Embeddings & Positional Encodings]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Probabilistic Machine Learning]], [[Source - Speech and Language Processing]]
