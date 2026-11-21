---
type: cheatsheet
domain: "D02"
monograph: "[[Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]"
part_of: 
  - "[[Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]"
  - "[[ST 02 - Linear, Regularized & Generalized Linear Models]]"
concepts: 
  - "[[Bias-Variance Trade-off]]"
  - "[[Double Descent]]"
  - "[[Rademacher Complexity & VC Dimension]]"
tags:
  - cheatsheet
  - mindmap
  - domain-02
---

# Cheatsheet — Bias-Variance Decomposition, Overfitting, Generalization & Double Descent

> [!summary] 10-Second Visual Hook
> Test risk = $\mathrm{Bias}^2 + \mathrm{Var} + \sigma_\epsilon^2$; variance blows up at $P=N$ then falls again via min-norm interpolation

- **Full Monograph:** [[Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]
- **Sub-Topic MOC:** [[ST 02 - Linear, Regularized & Generalized Linear Models]]
- **Atomic Concepts:** [[Bias-Variance Trade-off]], [[Double Descent]], [[Rademacher Complexity & VC Dimension]]

## Interactive Mindmap Overview

```mermaid
flowchart LR
  ROOT["Bias-Variance Decomposition, Overfitting, Genera"]
  B1["1. WHAT Exact Squared-Error Decomposition"]
  ROOT --> B1
  B2["2. FOUR REGIMES ACROSS P/N"]
  ROOT --> B2
  B3["3. KEY THEOREM Min-Norm MarchenkoPastur"]
  ROOT --> B3
  B4["4. CONNECTIONS VC, Rademacher Implicit Bias"]
  ROOT --> B4
  B5["5. COST Validation Primal/Dual Solvers"]
  ROOT --> B5
  B6["6. USE-CASES Navigating Capacity in Practice"]
  ROOT --> B6
  C1(("Bias-Variance Trade-off"))
  ROOT -.-> C1
  C2(("Double Descent"))
  ROOT -.-> C2
  C3(("Rademacher Complexity VC Dimension"))
  ROOT -.-> C3
```

## Branch Summary & Key Equations

- **1. WHAT — Exact Squared-Error Decomposition**
  $$
  \mathbb{E}_{\mathcal D,\epsilon}\!\big[(y-\hat f_{\mathcal D}(\boldsymbol{x}))^2\big]=\mathrm{Bias}^2+\mathrm{Var}+\sigma_\epsilon^2
  $$
- **2. FOUR REGIMES ACROSS $P/N$**
- **3. KEY THEOREM — Min-Norm & Marchenko–Pastur**
  $$
  \hat{\boldsymbol{w}}_{\min}=\arg\min_{\boldsymbol{\Phi}\boldsymbol{w}=\boldsymbol{y}}\lVert \boldsymbol{w} \rVert_2=\boldsymbol{\Phi}^{+}\boldsymbol{y}=\boldsymbol{\Phi}^{\top}(\boldsymbol{\Phi}\boldsymbol{\Phi}^{\top})^{-1}\boldsymbol{y}
  $$
- **4. CONNECTIONS — VC, Rademacher & Implicit Bias**
- **5. COST — Validation & Primal/Dual Solvers**
- **6. USE-CASES — Navigating Capacity in Practice**

## Related Sources

[[Source - Deep Learning]], [[Source - Foundations]], [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Machine Learning co ban]], [[Source - Understanding Deep Learning]]
