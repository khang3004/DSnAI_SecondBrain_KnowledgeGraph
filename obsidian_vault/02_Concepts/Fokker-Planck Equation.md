---
type: theorem
domain: "D04"
aliases: []
part_of: 
  - "[[Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]"
  - "[[Cheatsheet - Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
prerequisites: 
  - "[[Score-Based SDE]]"
enables: 
  - "[[Probability Flow ODE]]"
generalizes: []
special_case_of: []
related: 
  - "[[Probability Flow ODE]]"
  - "[[Score-Based SDE]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
  - "[[Source - Tutorial on Diffusion Models for Imaging and Vision]]"
tags:
  - theorem
  - domain-04
---

# Fokker-Planck Equation

> [!abstract] Definition (Theorem)
> Deterministic PDE governing density evolution of an Ito SDE; rewriting diffusion as a transport term yields the Probability Flow ODE.

$$
\partial_t p_t = -\nabla\cdot(\boldsymbol{f}p_t) + \frac{1}{2}g^2\Delta p_t
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Score-Based SDE]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Probability Flow ODE]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Probability Flow ODE]], [[Score-Based SDE]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]
- **Visual Cheatsheet:** [[Cheatsheet - Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]
- **Referenced Across Monographs:** [[Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]
- **Primary Textbooks:** [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]]
