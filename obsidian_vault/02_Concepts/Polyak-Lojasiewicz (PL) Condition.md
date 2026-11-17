---
type: object
domain: "D01"
aliases: 
  - "PL Condition"
part_of: 
  - "[[Gradient Descent Dynamics - Strong Convexity, Smoothness, PL Condition & Saddle Escape]]"
  - "[[Cheatsheet - Gradient Descent Dynamics - Strong Convexity, Smoothness, PL Condition & Saddle Escape]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Gradient Descent]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Gradient Descent]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Fall 2022]]"
  - "[[Source - Foundations]]"
  - "[[Source - Handbook of Convergence Theorems]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - object
  - domain-01
---

# Polyak-Lojasiewicz (PL) Condition

> [!abstract] Definition (Object)
> Non-convex gradient-dominance condition guaranteeing linear convergence of GD without requiring convexity or unique minima.

$$
\frac{1}{2}\|\nabla f(\boldsymbol{x})\|_2^2 \ge \mu\bigl(f(\boldsymbol{x}) - f^*\bigr)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Gradient Descent]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Gradient Descent]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Gradient Descent Dynamics - Strong Convexity, Smoothness, PL Condition & Saddle Escape]]
- **Visual Cheatsheet:** [[Cheatsheet - Gradient Descent Dynamics - Strong Convexity, Smoothness, PL Condition & Saddle Escape]]
- **Referenced Across Monographs:** [[Gradient Descent Dynamics - Strong Convexity, Smoothness, PL Condition & Saddle Escape]]
- **Primary Textbooks:** [[Source - Fall 2022]], [[Source - Foundations]], [[Source - Handbook of Convergence Theorems]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
