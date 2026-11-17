---
type: cheatsheet
domain: "D01"
monograph: "[[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]"
part_of: 
  - "[[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]"
  - "[[ST 01 - Information Theory, Divergences & Optimal Transport]]"
concepts: 
  - "[[Information Bottleneck]]"
  - "[[KL Divergence]]"
  - "[[Mutual Information & DPI]]"
  - "[[Shannon Entropy]]"
  - "[[f-Divergence]]"
tags:
  - cheatsheet
  - mindmap
  - domain-01
---

# Cheatsheet — Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck

> [!summary] 10-Second Visual Hook
> Measure uncertainty in nats/bits, compare densities via convex ratios, and squeeze nuisance via bottlenecks

- **Full Monograph:** [[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]
- **Sub-Topic MOC:** [[ST 01 - Information Theory, Divergences & Optimal Transport]]
- **Atomic Concepts:** [[Information Bottleneck]], [[KL Divergence]], [[Mutual Information & DPI]], [[Shannon Entropy]], [[f-Divergence]]

## Interactive Mindmap Overview

```mermaid
flowchart LR
  ROOT["Shannon Entropy, KL Divergence, f-Divergences In"]
  B1["1. ENTROPY MAXENT uncertainty coding"]
  ROOT --> B1
  B2["2. KL CROSS-ENTROPY M- vs. I-projection"]
  ROOT --> B2
  B3["3. MUTUAL INFO, DPI FANO channel limits"]
  ROOT --> B3
  B4["4. f-DIVERGENCES PINSKER ratio geometry"]
  ROOT --> B4
  B5["5. INFORMATION BOTTLENECK IB, VIB CEB"]
  ROOT --> B5
  B6["6. ML CONNECTIONS one variational root"]
  ROOT --> B6
  C1(("Information Bottleneck"))
  ROOT -.-> C1
  C2(("KL Divergence"))
  ROOT -.-> C2
  C3(("Mutual Information DPI"))
  ROOT -.-> C3
  C4(("Shannon Entropy"))
  ROOT -.-> C4
  C5(("f-Divergence"))
  ROOT -.-> C5
```

## Branch Summary & Key Equations

- **1. ENTROPY & MAXENT — uncertainty & coding**
  $$
  \mathrm{H}(X)=-\sum_{x}p(x)\ln p(x)
  $$
- **2. KL & CROSS-ENTROPY — M- vs. I-projection**
  $$
  H_{\mathrm{ce}}(p,q)=\mathrm{H}(p)+D_{\mathrm{KL}}(p \parallel q),\quad D_{\mathrm{KL}}(p \parallel q)\ge 0
  $$
- **3. MUTUAL INFO, DPI & FANO — channel limits**
  $$
  \mathrm{I}(X;Y)=D_{\mathrm{KL}}(p(x,y) \parallel p(x)p(y))=\mathrm{H}(X)-\mathrm{H}(X|Y)\ge 0
  $$
- **4. f-DIVERGENCES & PINSKER — ratio geometry**
- **5. INFORMATION BOTTLENECK — IB, VIB & CEB**
  $$
  \min_{e(t|x)} \mathcal L_{\mathrm{IB}}=\beta \mathrm{I}(X;T)-\mathrm{I}(T;Y)
  $$
- **6. ML CONNECTIONS — one variational root**

## Related Sources

[[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
