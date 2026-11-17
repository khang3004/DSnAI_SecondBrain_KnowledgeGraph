---
type: cheatsheet
domain: "D01"
monograph: "[[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]"
part_of: 
  - "[[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]"
  - "[[ST 01 - Vector Spaces, Spectral Theory & Matrix Factorizations]]"
concepts: 
  - "[[Cauchy-Schwarz Inequality]]"
  - "[[Inner Product & Norms]]"
  - "[[Orthogonal Projection]]"
  - "[[QR Factorization]]"
  - "[[Vector Space & Subspaces]]"
tags:
  - cheatsheet
  - mindmap
  - domain-01
---

# Cheatsheet — Vector Spaces, Norms, Orthogonality & Least-Squares Projections

> [!summary] 10-Second Visual Hook
> Every vector splits into an in-subspace shadow $\hat{\boldsymbol{b}}$ and a perpendicular residual $\boldsymbol{e}\perp W$

- **Full Monograph:** [[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]
- **Sub-Topic MOC:** [[ST 01 - Vector Spaces, Spectral Theory & Matrix Factorizations]]
- **Atomic Concepts:** [[Cauchy-Schwarz Inequality]], [[Inner Product & Norms]], [[Orthogonal Projection]], [[QR Factorization]], [[Vector Space & Subspaces]]

## Interactive Mindmap Overview

```mermaid
flowchart LR
  ROOT["Vector Spaces, Norms, Orthogonality Least-Square"]
  B1["1. FOUR FUNDAMENTAL SUBSPACES"]
  ROOT --> B1
  B2["2. NORMS SPD INNER PRODUCTS"]
  ROOT --> B2
  B3["3. ORTHOGONAL PROJECTOR ALGEBRA"]
  ROOT --> B3
  B4["4. LEAST SQUARES, MIN-NORM boldsymbolA+"]
  ROOT --> B4
  B5["5. SOLVERS ORTHOGONALIZATION COST"]
  ROOT --> B5
  B6["6. ML USE-CASES where you meet it"]
  ROOT --> B6
  C1(("Cauchy-Schwarz Inequality"))
  ROOT -.-> C1
  C2(("Inner Product Norms"))
  ROOT -.-> C2
  C3(("Orthogonal Projection"))
  ROOT -.-> C3
  C4(("QR Factorization"))
  ROOT -.-> C4
  C5(("Vector Space Subspaces"))
  ROOT -.-> C5
```

## Branch Summary & Key Equations

- **1. FOUR FUNDAMENTAL SUBSPACES**
  $$
  \mathbb{R}^n = \mathrm{Row}(\boldsymbol{A}) \oplus \mathrm{null}(\boldsymbol{A})
  $$
- **2. NORMS & SPD INNER PRODUCTS**
  $$
  \langle \boldsymbol{x}, \boldsymbol{y} \rangle_{\boldsymbol{M}} = \boldsymbol{x}^{\top}\boldsymbol{M}\boldsymbol{y} (\boldsymbol{M}\succ\boldsymbol{0})
  $$
- **3. ORTHOGONAL PROJECTOR ALGEBRA**
  $$
  \boldsymbol{P}_W = \boldsymbol{B}(\boldsymbol{B}^{\top}\boldsymbol{B})^{-1}\boldsymbol{B}^{\top} = \boldsymbol{Q}\boldsymbol{Q}^{\top} \quad (\boldsymbol{Q}^{\top}\boldsymbol{Q}=\boldsymbol{I}_r)
  $$
- **4. LEAST SQUARES, MIN-NORM & \boldsymbol{A}^{+}**
- **5. SOLVERS & ORTHOGONALIZATION COST**
- **6. ML USE-CASES — where you meet it**

## Related Sources

[[Source - Deep Learning]], [[Source - E D I T I O N]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
