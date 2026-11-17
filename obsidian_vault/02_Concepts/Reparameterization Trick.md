---
type: algorithm
domain: "D02"
aliases: []
part_of: 
  - "[[Variational Inference, Evidence Lower Bound (ELBO) & Expectation Propagation]]"
  - "[[Cheatsheet - Variational Inference, Evidence Lower Bound (ELBO) & Expectation Propagation]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[Automatic Differentiation (AD)]]"
  - "[[Variational Inference & ELBO]]"
enables: 
  - "[[Variational Autoencoder (VAE)]]"
generalizes: []
special_case_of: []
related: 
  - "[[PPO (Proximal Policy Optimization)]]"
  - "[[Variational Autoencoder (VAE)]]"
  - "[[Variational Inference & ELBO]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - algorithm
  - domain-02
---

# Reparameterization Trick

> [!abstract] Definition (Algorithm)
> Expresses random sample z = g(theta, eps) with parameter-free base noise eps to obtain low-variance pathwise gradients.

$$
\nabla_{\boldsymbol{\phi}}\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z})}[f(\boldsymbol{z})] = \mathbb{E}_{p(\boldsymbol{\varepsilon})}\!\left[\nabla_{\boldsymbol{z}}f(\boldsymbol{z})\nabla_{\boldsymbol{\phi}}g_{\boldsymbol{\phi}}(\boldsymbol{\varepsilon},\boldsymbol{x})\right]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Automatic Differentiation (AD)]], [[Variational Inference & ELBO]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Variational Autoencoder (VAE)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[PPO (Proximal Policy Optimization)]], [[Variational Autoencoder (VAE)]], [[Variational Inference & ELBO]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Variational Inference, Evidence Lower Bound (ELBO) & Expectation Propagation]]
- **Visual Cheatsheet:** [[Cheatsheet - Variational Inference, Evidence Lower Bound (ELBO) & Expectation Propagation]]
- **Referenced Across Monographs:** [[Variational Inference, Evidence Lower Bound (ELBO) & Expectation Propagation]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
