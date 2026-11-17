---
type: cheatsheet
domain: "D06"
monograph: "[[Multi-Armed & Contextual Bandits, UCB, Thompson Sampling & Regret Bounds]]"
part_of: 
  - "[[Multi-Armed & Contextual Bandits, UCB, Thompson Sampling & Regret Bounds]]"
  - "[[ST 06 - Multi-Armed Bandits, MDPs & Dynamic Programming]]"
concepts: 
  - "[[Multi-Armed Bandits & UCB]]"
  - "[[Thompson Sampling]]"
tags:
  - cheatsheet
  - mindmap
  - domain-06
---

# Cheatsheet — Multi-Armed & Contextual Bandits, UCB, Thompson Sampling & Regret Bounds

> [!summary] 10-Second Visual Hook
> Earn while you learn: trade off immediate reward $\hat{\mu}_t(a)$ against uncertainty bonus $\delta_t(a)$

- **Full Monograph:** [[Multi-Armed & Contextual Bandits, UCB, Thompson Sampling & Regret Bounds]]
- **Sub-Topic MOC:** [[ST 06 - Multi-Armed Bandits, MDPs & Dynamic Programming]]
- **Atomic Concepts:** [[Multi-Armed Bandits & UCB]], [[Thompson Sampling]]

## Interactive Mindmap Overview

```mermaid
flowchart LR
  ROOT["Multi-Armed Contextual Bandits, UCB, Thompson Sa"]
  B1["1. WHAT evaluative feedback regret"]
  ROOT --> B1
  B2["2. HOW four exploration mechanisms"]
  ROOT --> B2
  B3["3. KEY BOUNDS LaiRobbins minimax"]
  ROOT --> B3
  B4["4. CONNECTIONS Contexts, BAMDP OPE"]
  ROOT --> B4
  B5["5. COST time, memory scaling"]
  ROOT --> B5
  B6["6. USE-CASES where bandits shine"]
  ROOT --> B6
  C1(("Multi-Armed Bandits UCB"))
  ROOT -.-> C1
  C2(("Thompson Sampling"))
  ROOT -.-> C2
```

## Branch Summary & Key Equations

- **1. WHAT — evaluative feedback & regret**
  $$
  L_T = \sum_{t=1}^T \ell_t = \sum_{a=1}^K \mathbb{E}[N_{T+1}(a)] \Delta_a
  $$
- **2. HOW — four exploration mechanisms**
- **3. KEY BOUNDS — Lai–Robbins & minimax**
  $$
  \liminf_{T\to\infty}\frac{L_T}{\ln T} \ge \sum_{\Delta_a>0}\frac{\Delta_a}{D_{\mathrm{KL}}(p_R(a) \parallel p_R(a^*))}
  $$
- **4. CONNECTIONS — Contexts, BAMDP & OPE**
- **5. COST — time, memory & scaling**
- **6. USE-CASES — where bandits shine**

## Related Sources

[[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]]
