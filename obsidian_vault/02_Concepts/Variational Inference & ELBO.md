---
type: algorithm
domain: "D02"
aliases: 
  - "ELBO"
  - "Variational Inference"
part_of: 
  - "[[Variational Inference, Evidence Lower Bound (ELBO) & Expectation Propagation]]"
  - "[[Cheatsheet - Variational Inference, Evidence Lower Bound (ELBO) & Expectation Propagation]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[KL Divergence]]"
enables: 
  - "[[DDPM]]"
  - "[[EM Algorithm]]"
  - "[[Reparameterization Trick]]"
  - "[[Variational Autoencoder (VAE)]]"
generalizes: []
special_case_of: []
related: 
  - "[[EM Algorithm]]"
  - "[[Exponential Families]]"
  - "[[Gaussian Mixture Models (GMM)]]"
  - "[[KL Divergence]]"
  - "[[Reparameterization Trick]]"
  - "[[Variational Autoencoder (VAE)]]"
contrasts_with: 
  - "[[MCMC & HMC Sampling]]"
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

# Variational Inference & ELBO

> [!abstract] Definition (Algorithm)
> Converts intractable posterior inference into optimization by maximizing the Evidence Lower Bound (minimizing Reverse KL to posterior).

$$
\log p(\boldsymbol{x}) = \mathrm{ELBO}(q;\boldsymbol{\theta}) + D_{\mathrm{KL}}\bigl(q(\boldsymbol{z})\,\|\,p(\boldsymbol{z}\mid\boldsymbol{x})\bigr)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[KL Divergence]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[DDPM]], [[EM Algorithm]], [[Reparameterization Trick]], [[Variational Autoencoder (VAE)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[EM Algorithm]], [[Exponential Families]], [[Gaussian Mixture Models (GMM)]], [[KL Divergence]], [[Reparameterization Trick]], [[Variational Autoencoder (VAE)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[MCMC & HMC Sampling]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Variational Inference, Evidence Lower Bound (ELBO) & Expectation Propagation]]
- **Visual Cheatsheet:** [[Cheatsheet - Variational Inference, Evidence Lower Bound (ELBO) & Expectation Propagation]]
- **Referenced Across Monographs:** [[Variational Inference, Evidence Lower Bound (ELBO) & Expectation Propagation]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
