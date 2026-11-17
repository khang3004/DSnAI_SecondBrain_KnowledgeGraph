---
type: algorithm
domain: "D03"
aliases: []
part_of: 
  - "[[Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]"
  - "[[Cheatsheet - Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[MLP & Universal Approximation]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Residual Connections]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - algorithm
  - domain-03
---

# He & Xavier Initialization

> [!abstract] Definition (Algorithm)
> Variance-preserving weight scaling preventing exponential 2^l explosion or collapse across deep forward/backward passes.

$$
\mathrm{Var}(W) = \frac{2}{D_{\mathrm{in}}} \text{ (He/ReLU)} \quad\text{vs.}\quad \frac{2}{D_{\mathrm{in}}+D_{\mathrm{out}}} \text{ (Xavier)}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[MLP & Universal Approximation]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Residual Connections]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]
- **Visual Cheatsheet:** [[Cheatsheet - Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]
- **Referenced Across Monographs:** [[Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Understanding Deep Learning]]
