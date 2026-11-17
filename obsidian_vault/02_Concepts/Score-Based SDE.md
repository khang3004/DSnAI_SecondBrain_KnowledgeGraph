---
type: theorem
domain: "D04"
aliases: []
part_of: 
  - "[[Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]"
  - "[[Cheatsheet - Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
prerequisites: 
  - "[[DDPM]]"
  - "[[Score Matching (DSM & SSM)]]"
enables: 
  - "[[Fokker-Planck Equation]]"
  - "[[Probability Flow ODE]]"
generalizes: []
special_case_of: []
related: 
  - "[[DDPM]]"
  - "[[Flow Matching & Rectified Flow]]"
  - "[[Fokker-Planck Equation]]"
  - "[[Probability Flow ODE]]"
  - "[[Score Matching (DSM & SSM)]]"
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

# Score-Based SDE

> [!abstract] Definition (Theorem)
> Continuous-time stochastic differential equation (VP/VE-SDE) whose time-reversal uses the learned score via Anderson's theorem.

$$
\mathrm{d}\bar{\boldsymbol{x}}_t = \bigl[\boldsymbol{f}(\bar{\boldsymbol{x}}_t,t) - g^2(t)\nabla_{\boldsymbol{x}}\log p_t(\bar{\boldsymbol{x}}_t)\bigr]\mathrm{d}t + g(t)\,\mathrm{d}\bar{\boldsymbol{w}}_t
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[DDPM]], [[Score Matching (DSM & SSM)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Fokker-Planck Equation]], [[Probability Flow ODE]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[DDPM]], [[Flow Matching & Rectified Flow]], [[Fokker-Planck Equation]], [[Probability Flow ODE]], [[Score Matching (DSM & SSM)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]
- **Visual Cheatsheet:** [[Cheatsheet - Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]
- **Referenced Across Monographs:** [[Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]
- **Primary Textbooks:** [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]]
