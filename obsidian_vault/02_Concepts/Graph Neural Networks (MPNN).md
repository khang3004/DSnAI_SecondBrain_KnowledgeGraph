---
type: architecture
domain: "D03"
aliases: 
  - "GNN"
  - "Message Passing Neural Network"
part_of: 
  - "[[Graph Neural Networks, Message Passing & Graph Convolutional Architectures]]"
  - "[[Cheatsheet - Graph Neural Networks, Message Passing & Graph Convolutional Architectures]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: []
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Self-Attention & Multi-Head Attention]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - architecture
  - domain-03
---

# Graph Neural Networks (MPNN)

> [!abstract] Definition (Architecture)
> Permutation-equivariant message passing over graph neighborhoods; expressivity bounded by the 1-WL isomorphism test.

$$
\boldsymbol{h}_v^{(l+1)} = \phi\Bigl(\boldsymbol{h}_v^{(l)}, \bigoplus_{u\in\mathcal{N}(v)} \psi(\boldsymbol{h}_v^{(l)}, \boldsymbol{h}_u^{(l)}, \boldsymbol{e}_{uv})\Bigr)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Self-Attention & Multi-Head Attention]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Graph Neural Networks, Message Passing & Graph Convolutional Architectures]]
- **Visual Cheatsheet:** [[Cheatsheet - Graph Neural Networks, Message Passing & Graph Convolutional Architectures]]
- **Referenced Across Monographs:** [[Graph Neural Networks, Message Passing & Graph Convolutional Architectures]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
