---
type: algorithm
domain: "D02"
aliases: 
  - "Ridge Regression"
  - "OLS"
part_of: 
  - "[[Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]"
  - "[[Cheatsheet - Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[Orthogonal Projection]]"
  - "[[Singular Value Decomposition (SVD)]]"
enables: []
generalizes: []
special_case_of: 
  - "[[Generalized Linear Models (GLM)]]"
related: 
  - "[[Bias-Variance Trade-off]]"
  - "[[Double Descent]]"
  - "[[Moore-Penrose Pseudoinverse]]"
  - "[[Orthogonal Projection]]"
  - "[[Proximal & Subgradient Methods]]"
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
  - algorithm
  - domain-02
---

# Linear & Ridge Regression

> [!abstract] Definition (Algorithm)
> Closed-form OLS projection and L2-regularized Ridge shrinking small singular directions to control variance.

$$
\hat{\boldsymbol{w}}_{\mathrm{ridge}} = (\boldsymbol{X}^\top\boldsymbol{X} + \lambda\boldsymbol{I})^{-1}\boldsymbol{X}^\top\boldsymbol{y}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Orthogonal Projection]], [[Singular Value Decomposition (SVD)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | [[Generalized Linear Models (GLM)]] |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Bias-Variance Trade-off]], [[Double Descent]], [[Moore-Penrose Pseudoinverse]], [[Orthogonal Projection]], [[Proximal & Subgradient Methods]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]
- **Visual Cheatsheet:** [[Cheatsheet - Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]
- **Referenced Across Monographs:** [[Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]], [[Source - Speech and Language Processing]]
