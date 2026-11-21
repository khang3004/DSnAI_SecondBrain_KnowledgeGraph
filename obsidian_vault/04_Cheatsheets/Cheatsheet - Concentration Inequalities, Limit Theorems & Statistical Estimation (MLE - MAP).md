---
type: cheatsheet
domain: "D01"
monograph: "[[Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]"
part_of: 
  - "[[Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]"
  - "[[ST 01 - Probability Spaces, Gaussians & Statistical Estimation]]"
concepts: 
  - "[[Concentration Inequalities]]"
  - "[[Law of Large Numbers & CLT]]"
  - "[[Maximum Likelihood & Fisher Info]]"
tags:
  - cheatsheet
  - mindmap
  - domain-01
---

# Cheatsheet — Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE/MAP)

> [!summary] 10-Second Visual Hook
> Sample means concentrate exponentially; rescaled errors turn Gaussian; priors shrink variance

- **Full Monograph:** [[Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]
- **Sub-Topic MOC:** [[ST 01 - Probability Spaces, Gaussians & Statistical Estimation]]
- **Atomic Concepts:** [[Concentration Inequalities]], [[Law of Large Numbers & CLT]], [[Maximum Likelihood & Fisher Info]]

## Interactive Mindmap Overview

```mermaid
flowchart LR
  ROOT["Concentration Inequalities, Limit Theorems Stati"]
  B1["1. CONCENTRATION finite-N tail bounds"]
  ROOT --> B1
  B2["2. LIMIT THEOREMS Ntoinfty laws"]
  ROOT --> B2
  B3["3. FISHER CRLB curvature efficiency"]
  ROOT --> B3
  B4["4. MLE vs. MAP vs. BAYES point posterior"]
  ROOT --> B4
  B5["5. DECISION TESTING loss, LRT Bayes"]
  ROOT --> B5
  B6["6. COST SOLVERS how to compute"]
  ROOT --> B6
  C1(("Concentration Inequalities"))
  ROOT -.-> C1
  C2(("Law of Large Numbers CLT"))
  ROOT -.-> C2
  C3(("Maximum Likelihood Fisher Info"))
  ROOT -.-> C3
```

## Branch Summary & Key Equations

- **1. CONCENTRATION — finite-$N$ tail bounds**
- **2. LIMIT THEOREMS — $N\to\infty$ laws**
- **3. FISHER & CRLB — curvature & efficiency**
  $$
  \boldsymbol{F}(\theta)=\mathrm{Cov}_\theta[\boldsymbol{s}]=-\mathbb{E}_\theta[\nabla^2_\theta\log p(X\mid\theta)]
  $$
- **4. MLE vs. MAP vs. BAYES — point & posterior**
- **5. DECISION & TESTING — loss, LRT & Bayes**
- **6. COST & SOLVERS — how to compute**

## Related Sources

[[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Machine Learning co ban]], [[Source - Probabilistic Machine Learning]]
