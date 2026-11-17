---
type: cheatsheet
domain: "D06"
monograph: "[[MDPs, Bellman Optimality Equations, Policy - Value Iteration & GPI]]"
part_of: 
  - "[[MDPs, Bellman Optimality Equations, Policy - Value Iteration & GPI]]"
  - "[[ST 06 - Multi-Armed Bandits, MDPs & Dynamic Programming]]"
concepts: 
  - "[[Bellman Equations]]"
  - "[[Markov Decision Process (MDP)]]"
  - "[[Value & Policy Iteration]]"
tags:
  - cheatsheet
  - mindmap
  - domain-06
---

# Cheatsheet — MDPs, Bellman Optimality Equations, Policy/Value Iteration & GPI

> [!summary] 10-Second Visual Hook
> Compress infinite-horizon trees into $V^*(s)$ via $\gamma$-contractions; alternate evaluation and greedy improvement

- **Full Monograph:** [[MDPs, Bellman Optimality Equations, Policy - Value Iteration & GPI]]
- **Sub-Topic MOC:** [[ST 06 - Multi-Armed Bandits, MDPs & Dynamic Programming]]
- **Atomic Concepts:** [[Bellman Equations]], [[Markov Decision Process (MDP)]], [[Value & Policy Iteration]]

## Interactive Mindmap Overview

```mermaid
flowchart LR
  ROOT["MDPs, Bellman Optimality Equations, Policy/Value"]
  B1["1. WHAT Markov decision process"]
  ROOT --> B1
  B2["2. BELLMAN expectation vs. optimality"]
  ROOT --> B2
  B3["3. KEY THEOREMS Banach improvement"]
  ROOT --> B3
  B4["4. CONNECTIONS GPI, LP dual POMDPs"]
  ROOT --> B4
  B5["5. COST exact planning algorithms"]
  ROOT --> B5
  B6["6. GPI GEOMETRY competing pulls"]
  ROOT --> B6
  C1(("Bellman Equations"))
  ROOT -.-> C1
  C2(("Markov Decision Process (MDP)"))
  ROOT -.-> C2
  C3(("Value Policy Iteration"))
  ROOT -.-> C3
```

## Branch Summary & Key Equations

- **1. WHAT — Markov decision process**
  $$
  G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1} = R_{t+1} + \gamma G_{t+1}
  $$
- **2. BELLMAN — expectation vs. optimality**
- **3. KEY THEOREMS — Banach & improvement**
  $$
  \lVert \mathcal{T}^* U - \mathcal{T}^* V \rVert_\infty \le \gamma\lVert U - V \rVert_\infty
  $$
- **4. CONNECTIONS — GPI, LP dual & POMDPs**
- **5. COST — exact planning algorithms**
- **6. GPI GEOMETRY — competing pulls**

## Related Sources

[[Source - Francis Bach]], [[Source - Probabilistic Machine Learning]], [[Source - Reinforcement Learning - An Overview]], [[Source - Understanding Deep Learning]]
