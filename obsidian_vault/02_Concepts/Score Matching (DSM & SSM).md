---
type: algorithm
domain: "D04"
aliases: 
  - "Denoising Score Matching"
  - "Stein Score"
part_of: 
  - "[[Energy-Based Models, Partition Functions, Contrastive Divergence & Score Matching]]"
  - "[[Cheatsheet - Energy-Based Models, Partition Functions, Contrastive Divergence & Score Matching]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
prerequisites: 
  - "[[Automatic Differentiation (AD)]]"
  - "[[Energy-Based Models (EBM)]]"
enables: 
  - "[[DDPM]]"
  - "[[Score-Based SDE]]"
generalizes: []
special_case_of: []
related: 
  - "[[Automatic Differentiation (AD)]]"
  - "[[DDPM]]"
  - "[[Energy-Based Models (EBM)]]"
  - "[[MCMC & HMC Sampling]]"
  - "[[Score-Based SDE]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
  - "[[Source - Tutorial on Diffusion Models for Imaging and Vision]]"
tags:
  - algorithm
  - domain-04
---

# Score Matching (DSM & SSM)

> [!abstract] Definition (Algorithm)
> Learns the Stein score grad_x log p(x) without Z_theta via denoising perturbation (DSM) or Hutchinson trace slicing (SSM).

$$
\mathcal{L}_{\mathrm{DSM}} = \mathbb{E}_{\boldsymbol{x},\tilde{\boldsymbol{x}}}\!\left[\frac{1}{2}\left\|\boldsymbol{s}_{\boldsymbol{\theta}}(\tilde{\boldsymbol{x}}) + \frac{\tilde{\boldsymbol{x}}-\boldsymbol{x}}{\sigma^2}\right\|_2^2\right]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Automatic Differentiation (AD)]], [[Energy-Based Models (EBM)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[DDPM]], [[Score-Based SDE]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Automatic Differentiation (AD)]], [[DDPM]], [[Energy-Based Models (EBM)]], [[MCMC & HMC Sampling]], [[Score-Based SDE]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Energy-Based Models, Partition Functions, Contrastive Divergence & Score Matching]]
- **Visual Cheatsheet:** [[Cheatsheet - Energy-Based Models, Partition Functions, Contrastive Divergence & Score Matching]]
- **Referenced Across Monographs:** [[Energy-Based Models, Partition Functions, Contrastive Divergence & Score Matching]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]]
