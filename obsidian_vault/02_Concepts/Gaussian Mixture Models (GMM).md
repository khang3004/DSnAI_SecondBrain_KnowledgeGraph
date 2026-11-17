---
type: object
domain: "D02"
aliases: 
  - "GMM"
part_of: 
  - "[[PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]"
  - "[[Cheatsheet - PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[Multivariate Gaussian & Schur Complement]]"
enables: 
  - "[[EM Algorithm]]"
generalizes: []
special_case_of: []
related: 
  - "[[EM Algorithm]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
  - "[[Variational Inference & ELBO]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - object
  - domain-02
---

# Gaussian Mixture Models (GMM)

> [!abstract] Definition (Object)
> Latent-variable density model combining K Gaussian components; universal smooth density approximator trained via EM.

$$
p(\boldsymbol{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\boldsymbol{x}\mid\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Multivariate Gaussian & Schur Complement]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[EM Algorithm]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[EM Algorithm]], [[Multivariate Gaussian & Schur Complement]], [[Variational Inference & ELBO]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]
- **Visual Cheatsheet:** [[Cheatsheet - PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]
- **Referenced Across Monographs:** [[PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
