---
type: cheatsheet
domain: "D03"
monograph: "[[Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]"
part_of: 
  - "[[Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]"
  - "[[ST 03 - Feedforward Networks, Normalization, Residuals & KANs]]"
concepts: 
  - "[[BatchNorm, LayerNorm & RMSNorm]]"
  - "[[He & Xavier Initialization]]"
  - "[[Residual Connections]]"
  - "[[Weight Decay & Dropout]]"
tags:
  - cheatsheet
  - mindmap
  - domain-03
---

# Cheatsheet — Initialization, Batch/LayerNorm, Residual Connections, Weight Decay & Dropout

> [!summary] 10-Second Visual Hook
> Calibrate variance at $t=0$, standardize features, open an $\boldsymbol{I}$ gradient highway, filter flat directions

- **Full Monograph:** [[Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]
- **Sub-Topic MOC:** [[ST 03 - Feedforward Networks, Normalization, Residuals & KANs]]
- **Atomic Concepts:** [[BatchNorm, LayerNorm & RMSNorm]], [[He & Xavier Initialization]], [[Residual Connections]], [[Weight Decay & Dropout]]

## Interactive Mindmap Overview

```mermaid
flowchart LR
  ROOT["Initialization, Batch/LayerNorm, Residual Connec"]
  B1["1. INIT Variance Preservation at t=0"]
  ROOT --> B1
  B2["2. NORMALIZATION Which Axes to Slice"]
  ROOT --> B2
  B3["3. RESIDUALS The boldsymbolI Gradient Highway"]
  ROOT --> B3
  B4["4. PRE-LN vs. POST-LN Transformer Conditioning"]
  ROOT --> B4
  B5["5. REGULARIZATION Curvature Ensembles"]
  ROOT --> B5
  B6["6. DATA TARGET REGULARIZERS"]
  ROOT --> B6
  C1(("BatchNorm, LayerNorm RMSNorm"))
  ROOT -.-> C1
  C2(("He Xavier Initialization"))
  ROOT -.-> C2
  C3(("Residual Connections"))
  ROOT -.-> C3
  C4(("Weight Decay Dropout"))
  ROOT -.-> C4
```

## Branch Summary & Key Equations

- **1. INIT — Variance Preservation at $t=0$**
  $$
  \mathrm{Var}(f_i^{(l)})=D_{l-1}\sigma_l^2 \mathbb{E}\bigl[(h_j^{(l-1)})^2\bigr]
  $$
- **2. NORMALIZATION — Which Axes to Slice?**
- **3. RESIDUALS — The $\boldsymbol{I}$ Gradient Highway**
  $$
  \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}^{(l)}}=\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}^{(L)}}\!\left(\boldsymbol{I}+\frac{\partial }{\partial \boldsymbol{x}^{(l)}}\sum_{k=l}^{L-1}\mathcal{F}_k\right)
  $$
- **4. PRE-LN vs. POST-LN — Transformer Conditioning**
- **5. REGULARIZATION — Curvature & Ensembles**
- **6. DATA & TARGET REGULARIZERS**

## Related Sources

[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Understanding Deep Learning]]
