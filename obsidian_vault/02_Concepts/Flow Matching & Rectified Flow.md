---
type: algorithm
domain: "D04"
aliases: 
  - "Conditional Flow Matching"
  - "Rectified Flow"
part_of: 
  - "[[Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]"
  - "[[Cheatsheet - Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
prerequisites: 
  - "[[Probability Flow ODE]]"
  - "[[Wasserstein Distance]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Consistency Models]]"
  - "[[Normalizing Flows]]"
  - "[[Probability Flow ODE]]"
  - "[[Score-Based SDE]]"
  - "[[Wasserstein Distance]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
  - "[[Source - Tutorial on Diffusion Models for Imaging and Vision]]"
tags:
  - algorithm
  - domain-04
---

# Flow Matching & Rectified Flow

> [!abstract] Definition (Algorithm)
> Simulation-free regression of continuous velocity fields along straight OT-like interpolation paths x_t = (1-t)x_0 + t x_1.

$$
\mathcal{L}_{\mathrm{CFM}} = \mathbb{E}_{t,\boldsymbol{x}_0,\boldsymbol{x}_1}\!\left[\|\boldsymbol{v}_{\boldsymbol{\phi}}(\boldsymbol{x}_t,t) - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\|_2^2\right]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Probability Flow ODE]], [[Wasserstein Distance]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Consistency Models]], [[Normalizing Flows]], [[Probability Flow ODE]], [[Score-Based SDE]], [[Wasserstein Distance]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]
- **Visual Cheatsheet:** [[Cheatsheet - Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]
- **Referenced Across Monographs:** [[Score SDEs, Itô - Girsanov - Fokker-Planck Equations & Flow Matching]]
- **Primary Textbooks:** [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]]
