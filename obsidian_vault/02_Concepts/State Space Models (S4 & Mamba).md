---
type: architecture
domain: "D03"
aliases: 
  - "Mamba"
  - "S4"
  - "Selective SSM"
part_of: 
  - "[[Classical State-Space Models, Structured SSMs & Selective Scan (Mamba)]]"
  - "[[Cheatsheet - Classical State-Space Models, Structured SSMs & Selective Scan (Mamba)]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[Kalman Filter]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Kalman Filter]]"
contrasts_with: 
  - "[[RNN, LSTM & GRU]]"
  - "[[Transformer Architecture]]"
dual_to: []
sources: 
  - "[[Source - Foundations of Large Language Models]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Report - Mamba When Transformer is Not All You Need - AIVN]]"
tags:
  - architecture
  - domain-03
---

# State Space Models (S4 & Mamba)

> [!abstract] Definition (Architecture)
> Discretized continuous linear/selective SSMs achieving O(L log L) or O(LN) parallel training and O(1) constant-memory autoregressive inference.

$$
\boldsymbol{h}_t = \bar{\boldsymbol{A}}_t\boldsymbol{h}_{t-1} + \bar{\boldsymbol{B}}_t x_t, \quad y_t = \boldsymbol{C}_t\boldsymbol{h}_t
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Kalman Filter]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Kalman Filter]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[RNN, LSTM & GRU]], [[Transformer Architecture]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Classical State-Space Models, Structured SSMs & Selective Scan (Mamba)]]
- **Visual Cheatsheet:** [[Cheatsheet - Classical State-Space Models, Structured SSMs & Selective Scan (Mamba)]]
- **Referenced Across Monographs:** [[Classical State-Space Models, Structured SSMs & Selective Scan (Mamba)]]
- **Primary Textbooks:** [[Source - Foundations of Large Language Models]], [[Source - Probabilistic Machine Learning]], [[Source - Report - Mamba When Transformer is Not All You Need - AIVN]]
