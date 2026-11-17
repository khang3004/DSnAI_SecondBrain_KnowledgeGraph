---
type: architecture
domain: "D05"
aliases: 
  - "Scaled Dot-Product Attention"
  - "Multi-Head Attention"
part_of: 
  - "[[Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]"
  - "[[Cheatsheet - Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]"
  - "[[D05 - Transformers, Language Modeling & LLM Engineering]]"
prerequisites: []
enables: 
  - "[[FlashAttention]]"
  - "[[GQA & MQA]]"
  - "[[KV Cache & PagedAttention]]"
  - "[[Transformer Architecture]]"
  - "[[Vision Transformer (ViT)]]"
generalizes: []
special_case_of: []
related: 
  - "[[FlashAttention]]"
  - "[[GQA & MQA]]"
  - "[[Graph Neural Networks (MPNN)]]"
  - "[[RoPE & Positional Embeddings]]"
  - "[[Transformer Architecture]]"
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

# Self-Attention & Multi-Head Attention

> [!abstract] Definition (Architecture)
> Content-addressed bilinear routing over Q, K, V scaled by 1/sqrt(d_k) across parallel representation subspaces.

$$
\mathrm{Attn}(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = \mathrm{softmax}\!\left(\frac{\boldsymbol{Q}\boldsymbol{K}^\top}{\sqrt{d_k}} + \boldsymbol{M}\right)\boldsymbol{V}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[FlashAttention]], [[GQA & MQA]], [[KV Cache & PagedAttention]], [[Transformer Architecture]], [[Vision Transformer (ViT)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[FlashAttention]], [[GQA & MQA]], [[Graph Neural Networks (MPNN)]], [[RoPE & Positional Embeddings]], [[Transformer Architecture]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]
- **Visual Cheatsheet:** [[Cheatsheet - Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]
- **Referenced Across Monographs:** [[Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Speech and Language Processing]], [[Source - Understanding Deep Learning]]
