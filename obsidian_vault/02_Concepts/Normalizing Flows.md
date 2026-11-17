---
type: architecture
domain: "D04"
aliases: []
part_of: 
  - "[[Change-of-Variables, Coupling - Autoregressive Normalizing Flows & Neural ODEs]]"
  - "[[Cheatsheet - Change-of-Variables, Coupling - Autoregressive Normalizing Flows & Neural ODEs]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
prerequisites: 
  - "[[Matrix Calculus & Jacobians]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Flow Matching & Rectified Flow]]"
  - "[[Matrix Calculus & Jacobians]]"
  - "[[Probability Flow ODE]]"
contrasts_with: 
  - "[[Variational Autoencoder (VAE)]]"
dual_to: []
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - architecture
  - domain-04
---

# Normalizing Flows

> [!abstract] Definition (Architecture)
> Invertible diffeomorphic compositions enabling exact likelihood evaluation via change-of-variables and triangular/coupling Jacobians.

$$
\log p_X(\boldsymbol{x}) = \log p_Z(\boldsymbol{f}^{-1}(\boldsymbol{x})) - \sum_{k=1}^K \log \left|\det \boldsymbol{J}_{\boldsymbol{f}_k}(\boldsymbol{z}_{k-1})\right|
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Matrix Calculus & Jacobians]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Flow Matching & Rectified Flow]], [[Matrix Calculus & Jacobians]], [[Probability Flow ODE]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Variational Autoencoder (VAE)]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Change-of-Variables, Coupling - Autoregressive Normalizing Flows & Neural ODEs]]
- **Visual Cheatsheet:** [[Cheatsheet - Change-of-Variables, Coupling - Autoregressive Normalizing Flows & Neural ODEs]]
- **Referenced Across Monographs:** [[Change-of-Variables, Coupling - Autoregressive Normalizing Flows & Neural ODEs]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Understanding Deep Learning]]
