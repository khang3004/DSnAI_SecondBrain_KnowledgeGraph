---
type: cheatsheet
domain: "D01"
monograph: "[[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]"
part_of: 
  - "[[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]"
  - "[[ST 01 - Vector Spaces, Spectral Theory & Matrix Factorizations]]"
concepts: 
  - "[[Condition Number]]"
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Low-Rank Approximation (Eckart-Young)]]"
  - "[[Moore-Penrose Pseudoinverse]]"
  - "[[Singular Value Decomposition (SVD)]]"
tags:
  - cheatsheet
  - mindmap
  - domain-01
---

# Cheatsheet — Spectral Theory, Singular Value Decomposition & Low-Rank Approximations

> [!summary] 10-Second Visual Hook
> Every matrix = rotate $\cdot$ stretch $\cdot$ rotate; keep only the biggest stretches

- **Full Monograph:** [[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Sub-Topic MOC:** [[ST 01 - Vector Spaces, Spectral Theory & Matrix Factorizations]]
- **Atomic Concepts:** [[Condition Number]], [[Eigendecomposition & Spectral Theorem]], [[Low-Rank Approximation (Eckart-Young)]], [[Moore-Penrose Pseudoinverse]], [[Singular Value Decomposition (SVD)]]

## Interactive Mindmap Overview

```mermaid
flowchart LR
  ROOT["Spectral Theory, Singular Value Decomposition Lo"]
  B1["1. WHAT the three-move machine"]
  ROOT --> B1
  B2["2. EVD vs SVD which one"]
  ROOT --> B2
  B3["3. KEY THEOREM EckartYoungMirsky"]
  ROOT --> B3
  B4["4. CONNECTIONS one object, many names"]
  ROOT --> B4
  B5["5. COST which solver"]
  ROOT --> B5
  B6["6. USE-CASES where you will meet it"]
  ROOT --> B6
  C1(("Condition Number"))
  ROOT -.-> C1
  C2(("Eigendecomposition Spectral Theorem"))
  ROOT -.-> C2
  C3(("Low-Rank Approximation (Eckart-Young)"))
  ROOT -.-> C3
  C4(("Moore-Penrose Pseudoinverse"))
  ROOT -.-> C4
  C5(("Singular Value Decomposition (SVD)"))
  ROOT -.-> C5
```

## Branch Summary & Key Equations

- **1. WHAT — the three-move machine**
  $$
  \boldsymbol{A}=\boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^{\top}
  $$
- **2. EVD vs SVD — which one?**
- **3. KEY THEOREM — Eckart–Young–Mirsky**
  $$
  \boldsymbol{A}_k=\sum_{i=1}^{k}\sigma_i\boldsymbol{u}_i\boldsymbol{v}_i^{\top}
  $$
- **4. CONNECTIONS — one object, many names**
- **5. COST — which solver?**
- **6. USE-CASES — where you will meet it**

## Related Sources

[[Source - Deep Learning]], [[Source - E D I T I O N]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
