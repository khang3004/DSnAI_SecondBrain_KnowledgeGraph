---
type: object
domain: "D01"
aliases: []
part_of: 
  - "[[Probability Spaces, Multivariate Gaussians & Exponential Families]]"
  - "[[Cheatsheet - Probability Spaces, Multivariate Gaussians & Exponential Families]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Multivariate Gaussian & Schur Complement]]"
enables: 
  - "[[Generalized Linear Models (GLM)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Generalized Linear Models (GLM)]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
  - "[[Variational Inference & ELBO]]"
contrasts_with: []
dual_to: 
  - "[[Bregman & Legendre Duality]]"
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - object
  - domain-01
---

# Exponential Families

> [!abstract] Definition (Object)
> Parametric family whose log-partition function A(eta) acts as cumulant generator and induces Legendre-dual mean parameters.

$$
p(\boldsymbol{x}\mid\boldsymbol{\eta}) = h(\boldsymbol{x})\exp\bigl(\langle\boldsymbol{\eta}, \boldsymbol{T}(\boldsymbol{x})\rangle - A(\boldsymbol{\eta})\bigr), \quad \nabla A(\boldsymbol{\eta}) = \mathbb{E}[\boldsymbol{T}(\boldsymbol{x})]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Multivariate Gaussian & Schur Complement]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Generalized Linear Models (GLM)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Generalized Linear Models (GLM)]], [[Multivariate Gaussian & Schur Complement]], [[Variational Inference & ELBO]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | [[Bregman & Legendre Duality]] |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Probability Spaces, Multivariate Gaussians & Exponential Families]]
- **Visual Cheatsheet:** [[Cheatsheet - Probability Spaces, Multivariate Gaussians & Exponential Families]]
- **Referenced Across Monographs:** [[Probability Spaces, Multivariate Gaussians & Exponential Families]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
