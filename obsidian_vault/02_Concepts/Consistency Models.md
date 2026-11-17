---
type: architecture
domain: "D04"
aliases: []
part_of: 
  - "[[Classifier - Classifier-Free Guidance, DDIM - DPM-Solver ODEs & Consistency Models]]"
  - "[[Cheatsheet - Classifier - Classifier-Free Guidance, DDIM - DPM-Solver ODEs & Consistency Models]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
prerequisites: 
  - "[[DDPM]]"
  - "[[Probability Flow ODE]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[DDPM]]"
  - "[[Flow Matching & Rectified Flow]]"
  - "[[Probability Flow ODE]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
tags:
  - architecture
  - domain-04
---

# Consistency Models

> [!abstract] Definition (Architecture)
> Maps any point on a PF-ODE trajectory directly to its clean origin x_0 in 1-4 steps via self-consistency distillation.

$$
\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) = \boldsymbol{x}_0 \quad \forall t \in [\varepsilon, T]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[DDPM]], [[Probability Flow ODE]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[DDPM]], [[Flow Matching & Rectified Flow]], [[Probability Flow ODE]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Classifier - Classifier-Free Guidance, DDIM - DPM-Solver ODEs & Consistency Models]]
- **Visual Cheatsheet:** [[Cheatsheet - Classifier - Classifier-Free Guidance, DDIM - DPM-Solver ODEs & Consistency Models]]
- **Referenced Across Monographs:** [[Classifier - Classifier-Free Guidance, DDIM - DPM-Solver ODEs & Consistency Models]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]]
