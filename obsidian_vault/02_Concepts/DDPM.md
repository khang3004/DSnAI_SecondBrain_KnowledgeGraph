---
type: architecture
domain: "D04"
aliases: 
  - "Denoising Diffusion Probabilistic Model"
part_of: 
  - "[[Denoising Diffusion Probabilistic Models (DDPM) & Noise-Conditional Score Networks]]"
  - "[[Cheatsheet - Denoising Diffusion Probabilistic Models (DDPM) & Noise-Conditional Score Networks]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
prerequisites: 
  - "[[Score Matching (DSM & SSM)]]"
  - "[[Variational Inference & ELBO]]"
enables: 
  - "[[Classifier-Free Guidance (CFG)]]"
  - "[[Consistency Models]]"
  - "[[Score-Based SDE]]"
generalizes: []
special_case_of: []
related: 
  - "[[Classifier-Free Guidance (CFG)]]"
  - "[[Consistency Models]]"
  - "[[Score Matching (DSM & SSM)]]"
  - "[[Score-Based SDE]]"
  - "[[Semantic & Instance Segmentation (FCN, U-Net & SAM)]]"
  - "[[Variational Autoencoder (VAE)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
  - "[[Source - Tutorial on Diffusion Models for Imaging and Vision]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - architecture
  - domain-04
---

# DDPM

> [!abstract] Definition (Architecture)
> Discrete-time Markovian noising and learned reverse Gaussian denoising; equivalent to multi-scale denoising score matching.

$$
\mathcal{L}_{\mathrm{simple}}(\boldsymbol{\theta}) = \mathbb{E}_{t,\boldsymbol{x}_0,\boldsymbol{\varepsilon}}\!\left[\left\|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\sqrt{\bar{\alpha}_t}\boldsymbol{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\varepsilon}, t)\right\|_2^2\right]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Score Matching (DSM & SSM)]], [[Variational Inference & ELBO]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Classifier-Free Guidance (CFG)]], [[Consistency Models]], [[Score-Based SDE]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Classifier-Free Guidance (CFG)]], [[Consistency Models]], [[Score Matching (DSM & SSM)]], [[Score-Based SDE]], [[Semantic & Instance Segmentation (FCN, U-Net & SAM)]], [[Variational Autoencoder (VAE)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Denoising Diffusion Probabilistic Models (DDPM) & Noise-Conditional Score Networks]]
- **Visual Cheatsheet:** [[Cheatsheet - Denoising Diffusion Probabilistic Models (DDPM) & Noise-Conditional Score Networks]]
- **Referenced Across Monographs:** [[Denoising Diffusion Probabilistic Models (DDPM) & Noise-Conditional Score Networks]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]], [[Source - Understanding Deep Learning]]
