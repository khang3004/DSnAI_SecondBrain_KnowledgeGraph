---
type: object
domain: "D04"
aliases: 
  - "PF-ODE"
part_of: 
  - "[[Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]"
  - "[[Cheatsheet - Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
prerequisites: 
  - "[[Fokker-Planck Equation]]"
  - "[[Score-Based SDE]]"
enables: 
  - "[[Consistency Models]]"
  - "[[Flow Matching & Rectified Flow]]"
generalizes: []
special_case_of: []
related: 
  - "[[Consistency Models]]"
  - "[[Flow Matching & Rectified Flow]]"
  - "[[Fokker-Planck Equation]]"
  - "[[Normalizing Flows]]"
  - "[[Score-Based SDE]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
  - "[[Source - Tutorial on Diffusion Models for Imaging and Vision]]"
tags:
  - object
  - domain-04
---

# Probability Flow ODE

> [!abstract] Definition (Object)
> Deterministic ODE sharing the exact marginal densities p_t(x) of the reverse SDE while enabling exact likelihood and fast ODE solvers.

$$
\frac{\mathrm{d}\tilde{\boldsymbol{x}}_t}{\mathrm{d}t} = \boldsymbol{f}(\tilde{\boldsymbol{x}}_t,t) - \frac{1}{2}g^2(t)\nabla_{\boldsymbol{x}}\log p_t(\tilde{\boldsymbol{x}}_t)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Fokker-Planck Equation]], [[Score-Based SDE]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Consistency Models]], [[Flow Matching & Rectified Flow]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Consistency Models]], [[Flow Matching & Rectified Flow]], [[Fokker-Planck Equation]], [[Normalizing Flows]], [[Score-Based SDE]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]
- **Visual Cheatsheet:** [[Cheatsheet - Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]
- **Referenced Across Monographs:** [[Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]
- **Primary Textbooks:** [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]]
