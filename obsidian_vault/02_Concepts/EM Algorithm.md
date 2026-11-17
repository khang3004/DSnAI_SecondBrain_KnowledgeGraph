---
type: algorithm
domain: "D02"
aliases: []
part_of: 
  - "[[PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]"
  - "[[Cheatsheet - PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[Gaussian Mixture Models (GMM)]]"
  - "[[Variational Inference & ELBO]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Gaussian Mixture Models (GMM)]]"
  - "[[Variational Inference & ELBO]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - algorithm
  - domain-02
---

# EM Algorithm

> [!abstract] Definition (Algorithm)
> Alternating coordinate ascent on the ELBO: E-step sets exact posterior q(z)=p(z|x,theta); M-step maximizes expected complete log-likelihood.

$$
\text{E: } q^{(t)} = p(\boldsymbol{z}\mid\boldsymbol{x},\boldsymbol{\theta}^{(t)}), \quad \text{M: } \boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta}} \mathbb{E}_{q^{(t)}}[\log p(\boldsymbol{x},\boldsymbol{z}\mid\boldsymbol{\theta})]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Gaussian Mixture Models (GMM)]], [[Variational Inference & ELBO]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Gaussian Mixture Models (GMM)]], [[Variational Inference & ELBO]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]
- **Visual Cheatsheet:** [[Cheatsheet - PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]
- **Referenced Across Monographs:** [[PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
