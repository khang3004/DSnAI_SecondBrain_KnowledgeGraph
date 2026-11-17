---
type: architecture
domain: "D05"
aliases: []
part_of: 
  - "[[Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]"
  - "[[Cheatsheet - Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]"
  - "[[D05 - Transformers, Language Modeling & LLM Engineering]]"
prerequisites: 
  - "[[BatchNorm, LayerNorm & RMSNorm]]"
  - "[[Residual Connections]]"
  - "[[Self-Attention & Multi-Head Attention]]"
enables: 
  - "[[CLIP & Vision-Language Alignment]]"
  - "[[Chinchilla Scaling Laws]]"
  - "[[LoRA & QLoRA]]"
generalizes: []
special_case_of: []
related: 
  - "[[BatchNorm, LayerNorm & RMSNorm]]"
  - "[[Chinchilla Scaling Laws]]"
  - "[[Object Detection (R-CNN, SSD & DETR)]]"
  - "[[Residual Connections]]"
  - "[[Self-Attention & Multi-Head Attention]]"
  - "[[Vision Transformer (ViT)]]"
contrasts_with: 
  - "[[RNN, LSTM & GRU]]"
  - "[[State Space Models (S4 & Mamba)]]"
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

# Transformer Architecture

> [!abstract] Definition (Architecture)
> Pre-LN residual backbone interleaving Multi-Head Attention (mixing tokens) and SwiGLU FFN (mixing channels).

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[BatchNorm, LayerNorm & RMSNorm]], [[Residual Connections]], [[Self-Attention & Multi-Head Attention]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[CLIP & Vision-Language Alignment]], [[Chinchilla Scaling Laws]], [[LoRA & QLoRA]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[BatchNorm, LayerNorm & RMSNorm]], [[Chinchilla Scaling Laws]], [[Object Detection (R-CNN, SSD & DETR)]], [[Residual Connections]], [[Self-Attention & Multi-Head Attention]], [[Vision Transformer (ViT)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[RNN, LSTM & GRU]], [[State Space Models (S4 & Mamba)]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]
- **Visual Cheatsheet:** [[Cheatsheet - Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]
- **Referenced Across Monographs:** [[Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Speech and Language Processing]], [[Source - Understanding Deep Learning]]
