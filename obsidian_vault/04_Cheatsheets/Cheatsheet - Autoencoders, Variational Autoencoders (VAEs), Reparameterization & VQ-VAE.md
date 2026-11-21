---
type: cheatsheet
domain: "D04"
monograph: "[[Autoencoders, Variational Autoencoders (VAEs), Reparameterization & VQ-VAE]]"
part_of: 
  - "[[Autoencoders, Variational Autoencoders (VAEs), Reparameterization & VQ-VAE]]"
  - "[[ST 04 - Autoencoders, Variational Autoencoders & Normalizing Flows]]"
concepts: 
  - "[[Autoencoders]]"
  - "[[Variational Autoencoder (VAE)]]"
tags:
  - cheatsheet
  - mindmap
  - domain-04
---

# Cheatsheet — Autoencoders, Variational Autoencoders (VAEs), Reparameterization & VQ-VAE

> [!summary] 10-Second Visual Hook
> Compress $\boldsymbol{x}$ to a prior-regularized latent bottleneck $\boldsymbol{z}$; sample via reparameterization or discrete codebooks

- **Full Monograph:** [[Autoencoders, Variational Autoencoders (VAEs), Reparameterization & VQ-VAE]]
- **Sub-Topic MOC:** [[ST 04 - Autoencoders, Variational Autoencoders & Normalizing Flows]]
- **Atomic Concepts:** [[Autoencoders]], [[Variational Autoencoder (VAE)]]

## Interactive Mindmap Overview

```mermaid
flowchart LR
  ROOT["Autoencoders, Variational Autoencoders (VAEs), R"]
  B1["1. WHAT deterministic vs. variational"]
  ROOT --> B1
  B2["2. ELBO likelihood lower bound gap"]
  ROOT --> B2
  B3["3. HOW pathwise reparameterization trick"]
  ROOT --> B3
  B4["4. beta-VAE VQ-VAE rate vs. discrete codes"]
  ROOT --> B4
  B5["5. COMPARISON model taxonomy"]
  ROOT --> B5
  B6["6. CONNECTIONS EM, information diffusion"]
  ROOT --> B6
  C1(("Autoencoders"))
  ROOT -.-> C1
  C2(("Variational Autoencoder (VAE)"))
  ROOT -.-> C2
```

## Branch Summary & Key Equations

- **1. WHAT — deterministic vs. variational**
  $$
  p_\theta(\boldsymbol{x},\boldsymbol{z})=p(\boldsymbol{z}) p_\theta(\boldsymbol{x}\mid\boldsymbol{z})
  $$
- **2. ELBO — likelihood lower bound & gap**
  $$
  \mathcal L_{\mathrm{ELBO}}(\theta,\phi;\boldsymbol{x})=\underbrace{\mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}[\log p_\theta(\boldsymbol{x}\mid\boldsymbol{z})]}_{\text{Reconstruction }(-D)}-\underbrace{D_{\mathrm{KL}}(q_\phi(\boldsymbol{z}\mid\boldsymbol{x}) \parallel p(\boldsymbol{z}))}_{\text{Prior Matching / Rate }R}
  $$
- **3. HOW — pathwise reparameterization trick**
  $$
  \boldsymbol{z}=g_\phi(\boldsymbol{\epsilon},\boldsymbol{x})=\boldsymbol{\mu}_\phi(\boldsymbol{x})+\boldsymbol{\sigma}_\phi(\boldsymbol{x})\odot\boldsymbol{\epsilon}
  $$
- **4. $\beta$-VAE & VQ-VAE — rate vs. discrete codes**
- **5. COMPARISON — model taxonomy**
- **6. CONNECTIONS — EM, information & diffusion**

## Related Sources

[[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]], [[Source - Understanding Deep Learning]]
