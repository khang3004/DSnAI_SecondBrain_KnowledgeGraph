---
type: object
domain: "D01"
aliases: 
  - "Multivariate Normal"
  - "Schur Complement"
part_of: 
  - "[[Probability Spaces, Multivariate Gaussians & Exponential Families]]"
  - "[[Cheatsheet - Probability Spaces, Multivariate Gaussians & Exponential Families]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Eigendecomposition & Spectral Theorem]]"
enables: 
  - "[[Exponential Families]]"
  - "[[Gaussian Mixture Models (GMM)]]"
  - "[[Gaussian Processes]]"
  - "[[Kalman Filter]]"
  - "[[Mahalanobis Distance]]"
generalizes: []
special_case_of: []
related: 
  - "[[Exponential Families]]"
  - "[[Gaussian Mixture Models (GMM)]]"
  - "[[Gaussian Processes]]"
  - "[[Graphical Models (BN & MRF)]]"
  - "[[Kalman Filter]]"
  - "[[Mahalanobis Distance]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - object
  - domain-01
---

# Multivariate Gaussian & Schur Complement

> [!abstract] Definition (Object)
> Maximum-entropy distribution under fixed mean/covariance; closed under marginalization (in Sigma) and conditioning (in Precision Lambda).

$$
\boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Lambda}_{aa}^{-1} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Eigendecomposition & Spectral Theorem]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Exponential Families]], [[Gaussian Mixture Models (GMM)]], [[Gaussian Processes]], [[Kalman Filter]], [[Mahalanobis Distance]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Exponential Families]], [[Gaussian Mixture Models (GMM)]], [[Gaussian Processes]], [[Graphical Models (BN & MRF)]], [[Kalman Filter]], [[Mahalanobis Distance]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Probability Spaces, Multivariate Gaussians & Exponential Families]]
- **Visual Cheatsheet:** [[Cheatsheet - Probability Spaces, Multivariate Gaussians & Exponential Families]]
- **Referenced Across Monographs:** [[Probability Spaces, Multivariate Gaussians & Exponential Families]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
