---
type: object
domain: "D02"
aliases: []
part_of: 
  - "[[Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]"
  - "[[Cheatsheet - Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[Exponential Families]]"
enables: []
generalizes: 
  - "[[Linear & Ridge Regression]]"
  - "[[Logistic Regression]]"
special_case_of: []
related: 
  - "[[Exponential Families]]"
  - "[[Logistic Regression]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Speech and Language Processing]]"
tags:
  - object
  - domain-02
---

# Generalized Linear Models (GLM)

> [!abstract] Definition (Object)
> Unifies OLS, Logistic, Softmax, and Poisson regression via exponential-family response and canonical link g(mu)=eta.

$$
\mathbb{E}[y\mid\boldsymbol{x}] = g^{-1}(\boldsymbol{w}^\top\boldsymbol{x}) = \nabla A(\boldsymbol{w}^\top\boldsymbol{x})
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Exponential Families]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | [[Linear & Ridge Regression]], [[Logistic Regression]] |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Exponential Families]], [[Logistic Regression]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]
- **Visual Cheatsheet:** [[Cheatsheet - Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]
- **Referenced Across Monographs:** [[Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]], [[Source - Speech and Language Processing]]
