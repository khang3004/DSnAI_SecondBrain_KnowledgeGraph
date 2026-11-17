---
type: algorithm
domain: "D01"
aliases: []
part_of: 
  - "[[Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]"
  - "[[Cheatsheet - Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[KL Divergence]]"
  - "[[Wasserstein Distance]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Bregman & Legendre Duality]]"
  - "[[Wasserstein Distance]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Optimal Transport in Large-Scale]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
tags:
  - algorithm
  - domain-01
---

# Sinkhorn Algorithm

> [!abstract] Definition (Algorithm)
> Entropic regularization turns OT into strictly convex matrix scaling solvable in O(n^2) GPU-friendly alternating projections.

$$
\min_{\boldsymbol{P}\in\Pi(\boldsymbol{a},\boldsymbol{b})} \langle \boldsymbol{C}, \boldsymbol{P}\rangle - \varepsilon H(\boldsymbol{P}) \implies \boldsymbol{P}^* = \mathrm{diag}(\boldsymbol{u})\boldsymbol{K}\mathrm{diag}(\boldsymbol{v})
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[KL Divergence]], [[Wasserstein Distance]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Bregman & Legendre Duality]], [[Wasserstein Distance]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]
- **Visual Cheatsheet:** [[Cheatsheet - Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]
- **Referenced Across Monographs:** [[Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]
- **Primary Textbooks:** [[Source - Optimal Transport in Large-Scale]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]]
