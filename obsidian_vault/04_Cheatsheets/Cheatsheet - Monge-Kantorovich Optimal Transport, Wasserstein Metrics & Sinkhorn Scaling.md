---
type: cheatsheet
domain: "D01"
monograph: "[[Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]"
part_of: 
  - "[[Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]"
  - "[[ST 01 - Information Theory, Divergences & Optimal Transport]]"
concepts: 
  - "[[Kantorovich Duality]]"
  - "[[Sinkhorn Algorithm]]"
  - "[[Wasserstein Distance]]"
tags:
  - cheatsheet
  - mindmap
  - domain-01
---

# Cheatsheet — Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling

> [!summary] 10-Second Visual Hook
> Compare distributions horizontally along ground metrics; regularize with entropy to scale in $O(n^2)$

- **Full Monograph:** [[Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]
- **Sub-Topic MOC:** [[ST 01 - Information Theory, Divergences & Optimal Transport]]
- **Atomic Concepts:** [[Kantorovich Duality]], [[Sinkhorn Algorithm]], [[Wasserstein Distance]]

## Interactive Mindmap Overview

```mermaid
flowchart LR
  ROOT["Monge-Kantorovich Optimal Transport, Wasserstein"]
  B1["1. MONGE vs. KANTOROVICH maps vs. plans"]
  ROOT --> B1
  B2["2. DUALITY WASSERSTEIN-p c-transforms"]
  ROOT --> B2
  B3["3. BRENIER DYNAMIC OT convex potentials"]
  ROOT --> B3
  B4["4. ENTROPIC OT SINKHORN matrix scaling"]
  ROOT --> B4
  B5["5. SOLVERS COMPLEXITY scaling trade-offs"]
  ROOT --> B5
  B6["6. GENERATIVE ML FLOWS where OT rules"]
  ROOT --> B6
  C1(("Kantorovich Duality"))
  ROOT -.-> C1
  C2(("Sinkhorn Algorithm"))
  ROOT -.-> C2
  C3(("Wasserstein Distance"))
  ROOT -.-> C3
```

## Branch Summary & Key Equations

- **1. MONGE vs. KANTOROVICH — maps vs. plans**
  $$
  \mathrm{OT}_c(\mu,\nu)=\inf_{\pi\in\Pi(\mu,\nu)}\int c(\boldsymbol{x},\boldsymbol{y}) \mathrm{d}\pi
  $$
- **2. DUALITY & WASSERSTEIN-$p$ — $c$-transforms**
  $$
  \mathrm{OT}_c(\mu,\nu)=\sup_{f(\boldsymbol{x})+g(\boldsymbol{y})\le c}\mathbb{E}_\mu[f]+\mathbb{E}_\nu[g]=\sup_{f\in\mathcal F_c}\mathbb{E}_\mu[f]+\mathbb{E}_\nu[f^c]
  $$
- **3. BRENIER & DYNAMIC OT — convex potentials**
  $$
  T^\star(\boldsymbol{x})=\nabla u(\boldsymbol{x}),  u(\boldsymbol{x})=\tfrac12\lVert \boldsymbol{x} \rVert_2^2-f^\star(\boldsymbol{x}) \text{convex}
  $$
- **4. ENTROPIC OT & SINKHORN — matrix scaling**
  $$
  \mathrm{EOT}_\varepsilon(\mu,\nu)=\inf_{\pi\in\Pi(\mu,\nu)}\langle c, \pi \rangle+\varepsilon D_{\mathrm{KL}}(\pi \parallel \mu\otimes\nu)
  $$
- **5. SOLVERS & COMPLEXITY — scaling trade-offs**
- **6. GENERATIVE ML & FLOWS — where OT rules**

## Related Sources

[[Source - Optimal Transport in Large-Scale]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]]
