---
type: object
domain: "D04"
aliases: 
  - "EBM"
part_of: 
  - "[[Energy-Based Models, Partition Functions, Contrastive Divergence & Score Matching]]"
  - "[[Cheatsheet - Energy-Based Models, Partition Functions, Contrastive Divergence & Score Matching]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
prerequisites: []
enables: 
  - "[[Score Matching (DSM & SSM)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Graphical Models (BN & MRF)]]"
  - "[[MCMC & HMC Sampling]]"
  - "[[Score Matching (DSM & SSM)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
  - "[[Source - Tutorial on Diffusion Models for Imaging and Vision]]"
tags:
  - object
  - domain-04
---

# Energy-Based Models (EBM)

> [!abstract] Definition (Object)
> Unnormalized Gibbs densities p(x) = exp(-E(x))/Z; circumvents intractable partition function Z via contrastive divergence or score matching.

$$
p_{\boldsymbol{\theta}}(\boldsymbol{x}) = \frac{e^{-E_{\boldsymbol{\theta}}(\boldsymbol{x})}}{Z_{\boldsymbol{\theta}}}, \quad \nabla_{\boldsymbol{x}}\log p_{\boldsymbol{\theta}}(\boldsymbol{x}) = -\nabla_{\boldsymbol{x}}E_{\boldsymbol{\theta}}(\boldsymbol{x})
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Score Matching (DSM & SSM)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Graphical Models (BN & MRF)]], [[MCMC & HMC Sampling]], [[Score Matching (DSM & SSM)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Energy-Based Models, Partition Functions, Contrastive Divergence & Score Matching]]
- **Visual Cheatsheet:** [[Cheatsheet - Energy-Based Models, Partition Functions, Contrastive Divergence & Score Matching]]
- **Referenced Across Monographs:** [[Energy-Based Models, Partition Functions, Contrastive Divergence & Score Matching]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]]
