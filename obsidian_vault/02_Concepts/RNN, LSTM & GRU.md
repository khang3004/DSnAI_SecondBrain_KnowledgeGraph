---
type: architecture
domain: "D03"
aliases: 
  - "RNN"
  - "LSTM"
  - "GRU"
part_of: 
  - "[[Recurrent Neural Networks, BPTT, LSTMs - GRUs & Long-Range Gradient Flow]]"
  - "[[Cheatsheet - Recurrent Neural Networks, BPTT, LSTMs - GRUs & Long-Range Gradient Flow]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[Automatic Differentiation (AD)]]"
enables: []
generalizes: []
special_case_of: []
related: []
contrasts_with: 
  - "[[State Space Models (S4 & Mamba)]]"
  - "[[Transformer Architecture]]"
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Speech and Language Processing]]"
tags:
  - architecture
  - domain-03
---

# RNN, LSTM & GRU

> [!abstract] Definition (Architecture)
> Fixed-state recurrent sequence models using additive cell-state highways (LSTM/GRU) to mitigate vanishing gradients during BPTT.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Automatic Differentiation (AD)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | — |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[State Space Models (S4 & Mamba)]], [[Transformer Architecture]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Recurrent Neural Networks, BPTT, LSTMs - GRUs & Long-Range Gradient Flow]]
- **Visual Cheatsheet:** [[Cheatsheet - Recurrent Neural Networks, BPTT, LSTMs - GRUs & Long-Range Gradient Flow]]
- **Referenced Across Monographs:** [[Recurrent Neural Networks, BPTT, LSTMs - GRUs & Long-Range Gradient Flow]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Probabilistic Machine Learning]], [[Source - Speech and Language Processing]]
