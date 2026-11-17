---
type: algorithm
domain: "D04"
aliases: 
  - "CFG"
part_of: 
  - "[[Classifier - Classifier-Free Guidance, DDIM - DPM-Solver ODEs & Consistency Models]]"
  - "[[Cheatsheet - Classifier - Classifier-Free Guidance, DDIM - DPM-Solver ODEs & Consistency Models]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
prerequisites: 
  - "[[DDPM]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[CLIP & Vision-Language Alignment]]"
  - "[[DDPM]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
tags:
  - algorithm
  - domain-04
---

# Classifier-Free Guidance (CFG)

> [!abstract] Definition (Algorithm)
> Extrapolates conditional vs. unconditional score predictions to trade diversity for sharp prompt alignment without an external classifier.

$$
\tilde{\boldsymbol{\varepsilon}}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,c) = (1+w)\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,c) - w\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,\varnothing)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[DDPM]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[CLIP & Vision-Language Alignment]], [[DDPM]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Classifier - Classifier-Free Guidance, DDIM - DPM-Solver ODEs & Consistency Models]]
- **Visual Cheatsheet:** [[Cheatsheet - Classifier - Classifier-Free Guidance, DDIM - DPM-Solver ODEs & Consistency Models]]
- **Referenced Across Monographs:** [[Classifier - Classifier-Free Guidance, DDIM - DPM-Solver ODEs & Consistency Models]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]]
