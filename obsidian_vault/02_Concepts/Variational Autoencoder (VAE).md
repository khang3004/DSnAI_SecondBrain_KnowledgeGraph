---
type: architecture
domain: "D04"
aliases: 
  - "VAE"
part_of: 
  - "[[Autoencoders, Variational Autoencoders (VAEs), Reparameterization & VQ-VAE]]"
  - "[[Cheatsheet - Autoencoders, Variational Autoencoders (VAEs), Reparameterization & VQ-VAE]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
prerequisites: 
  - "[[Reparameterization Trick]]"
  - "[[Variational Inference & ELBO]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Autoencoders]]"
  - "[[DDPM]]"
  - "[[Information Bottleneck]]"
  - "[[Reparameterization Trick]]"
  - "[[Variational Inference & ELBO]]"
contrasts_with: 
  - "[[GAN & Mode Collapse]]"
  - "[[Normalizing Flows]]"
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
  - "[[Source - Tutorial on Diffusion Models for Imaging and Vision]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - architecture
  - domain-04
---

# Variational Autoencoder (VAE)

> [!abstract] Definition (Architecture)
> Amortized variational generative model optimizing the ELBO via encoder inference network and reparameterized latent sampling.

$$
\mathcal{L}_{\mathrm{VAE}} = \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z})] - D_{\mathrm{KL}}\bigl(q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x}) \,\|\, p(\boldsymbol{z})\bigr)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Reparameterization Trick]], [[Variational Inference & ELBO]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Autoencoders]], [[DDPM]], [[Information Bottleneck]], [[Reparameterization Trick]], [[Variational Inference & ELBO]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[GAN & Mode Collapse]], [[Normalizing Flows]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Autoencoders, Variational Autoencoders (VAEs), Reparameterization & VQ-VAE]]
- **Visual Cheatsheet:** [[Cheatsheet - Autoencoders, Variational Autoencoders (VAEs), Reparameterization & VQ-VAE]]
- **Referenced Across Monographs:** [[Autoencoders, Variational Autoencoders (VAEs), Reparameterization & VQ-VAE]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]], [[Source - Understanding Deep Learning]]
