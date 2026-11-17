---
type: cheatsheet
domain: "D07"
monograph: "[[Iterative ML Systems Design, Feature Engineering, Leakage, Distribution Shift & Continual Learning]]"
part_of: 
  - "[[Iterative ML Systems Design, Feature Engineering, Leakage, Distribution Shift & Continual Learning]]"
  - "[[ST 07 - ML Systems Design, Data Pipelines & Distribution Shifts]]"
concepts: 
  - "[[Distribution Shift & Drift]]"
  - "[[FTI Architecture & Leakage Prevention]]"
tags:
  - cheatsheet
  - mindmap
  - domain-07
---

# Cheatsheet — Iterative ML Systems Design, Feature Engineering, Leakage, Distribution Shift & Continual Learning

> [!summary] 10-Second Visual Hook
> Decouple Feature–Training–Inference, enforce split-invariance, detect $P(X,Y)$ shift & retrain continually

- **Full Monograph:** [[Iterative ML Systems Design, Feature Engineering, Leakage, Distribution Shift & Continual Learning]]
- **Sub-Topic MOC:** [[ST 07 - ML Systems Design, Data Pipelines & Distribution Shifts]]
- **Atomic Concepts:** [[Distribution Shift & Drift]], [[FTI Architecture & Leakage Prevention]]

## Interactive Mindmap Overview

```mermaid
flowchart LR
  ROOT["Iterative ML Systems Design, Feature Engineering"]
  B1["1. WHAT Decoupled FTI Architecture"]
  ROOT --> B1
  B2["2. IMBALANCE SPLIT-INVARIANT FEATURES"]
  ROOT --> B2
  B3["3. SHIFT TAXONOMY Factorizing P(X,Y)"]
  ROOT --> B3
  B4["4. DRIFT METRICS KS, PSI Kernel MMD"]
  ROOT --> B4
  B5["5. SERVING COMPRESSION TRADE-OFFS"]
  ROOT --> B5
  B6["6. CONTINUAL LEARNING Stateful vs. Stateless"]
  ROOT --> B6
  C1(("Distribution Shift Drift"))
  ROOT -.-> C1
  C2(("FTI Architecture Leakage Prevention"))
  ROOT -.-> C2
```

## Branch Summary & Key Equations

- **1. WHAT — Decoupled FTI Architecture**
- **2. IMBALANCE & SPLIT-INVARIANT FEATURES**
  $$
  \mathcal{L}_{\mathrm{FL}}(p_t)=-\alpha_t(1-p_t)^\gamma\log(p_t),\quad W_i^{\mathrm{eff}}=\frac{1-\beta}{1-\beta^{N_i}}
  $$
- **3. SHIFT TAXONOMY — Factorizing P(X,Y)**
- **4. DRIFT METRICS — KS, PSI & Kernel MMD**
  $$
  \mathrm{PSI}=\sum_{b=1}^{B}(q_b-p_b)\ln\frac{q_b}{p_b}
  $$
- **5. SERVING & COMPRESSION TRADE-OFFS**
- **6. CONTINUAL LEARNING — Stateful vs. Stateless**

## Related Sources

[[Source - Designing Machine Learning Systems]], [[Source - LLM Engineer’s Handbook]]
