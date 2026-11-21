---
type: cheatsheet
domain: "D02"
monograph: "[[PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]"
part_of: 
  - "[[PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]"
  - "[[ST 02 - PCA, Factor Analysis, Mixture Models & EM]]"
concepts: 
  - "[[EM Algorithm]]"
  - "[[Gaussian Mixture Models (GMM)]]"
  - "[[Principal Component Analysis (PCA)]]"
tags:
  - cheatsheet
  - mindmap
  - domain-02
---

# Cheatsheet — PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm

> [!summary] 10-Second Visual Hook
> Subspaces & soft clusters via alternating E-step (tighten ELBO) and M-step (climb surrogate)

- **Full Monograph:** [[PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]
- **Sub-Topic MOC:** [[ST 02 - PCA, Factor Analysis, Mixture Models & EM]]
- **Atomic Concepts:** [[EM Algorithm]], [[Gaussian Mixture Models (GMM)]], [[Principal Component Analysis (PCA)]]

## Interactive Mindmap Overview

```mermaid
flowchart LR
  ROOT["PCA, Gaussian Mixture Models the Expectation-Max"]
  B1["1. PCA three equivalent views"]
  ROOT --> B1
  B2["2. PPCA FACTOR ANALYSIS generative subspace"]
  ROOT --> B2
  B3["3. GENERAL EM ELBO monotone ascent"]
  ROOT --> B3
  B4["4. GMM vs. K-MEANS soft to hard limit"]
  ROOT --> B4
  B5["5. COMPLEXITY SCALING which solver"]
  ROOT --> B5
  B6["6. LATENT TAXONOMY one shallow template"]
  ROOT --> B6
  C1(("EM Algorithm"))
  ROOT -.-> C1
  C2(("Gaussian Mixture Models (GMM)"))
  ROOT -.-> C2
  C3(("Principal Component Analysis (PCA)"))
  ROOT -.-> C3
```

## Branch Summary & Key Equations

- **1. PCA — three equivalent views**
  $$
  \boldsymbol{S} = \frac{1}{N}\tilde{\boldsymbol{X}}^{\top}\tilde{\boldsymbol{X}} = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top}
  $$
- **2. PPCA & FACTOR ANALYSIS — generative subspace**
  $$
  \boldsymbol{z}_n\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I}_M),  \boldsymbol{x}_n\mid\boldsymbol{z}_n\sim\mathcal{N}(\boldsymbol{W}\boldsymbol{z}_n+\boldsymbol{\mu}, \boldsymbol{\Psi})
  $$
- **3. GENERAL EM — ELBO & monotone ascent**
  $$
  \log p(\boldsymbol{X}\mid\boldsymbol{\theta}) = \mathcal{L}(q,\boldsymbol{\theta}) + D_{\mathrm{KL}}(q(\boldsymbol{Z}) \parallel p(\boldsymbol{Z}\mid\boldsymbol{X},\boldsymbol{\theta}))
  $$
- **4. GMM vs. K-MEANS — soft to hard limit**
  $$
  r_{nk} = \frac{\pi_k\mathcal{N}(\boldsymbol{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j\mathcal{N}(\boldsymbol{x}_n\mid\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)},  N_k = \sum_{n=1}^N r_{nk}
  $$
- **5. COMPLEXITY & SCALING — which solver?**
- **6. LATENT TAXONOMY — one shallow template**

## Related Sources

[[Source - Foundations]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
