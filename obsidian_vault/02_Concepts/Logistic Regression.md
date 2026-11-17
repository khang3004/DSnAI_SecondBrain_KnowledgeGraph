---
type: algorithm
domain: "D02"
aliases: []
part_of: 
  - "[[Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]"
  - "[[Cheatsheet - Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[Maximum Likelihood & Fisher Info]]"
enables: []
generalizes: []
special_case_of: 
  - "[[Generalized Linear Models (GLM)]]"
related: 
  - "[[Generalized Linear Models (GLM)]]"
  - "[[Maximum Likelihood & Fisher Info]]"
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

# Logistic Regression

> [!abstract] Definition (Algorithm)
> Linear classifier modeling log-odds via sigmoid/softmax link; strictly convex cross-entropy loss trained by Newton/IRLS or GD.

$$
P(y=1\mid\boldsymbol{x}) = \sigma(\boldsymbol{w}^\top\boldsymbol{x}), \quad \mathcal{L}_{\mathrm{CE}} = -\sum_i \bigl[y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\bigr]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Maximum Likelihood & Fisher Info]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | [[Generalized Linear Models (GLM)]] |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Generalized Linear Models (GLM)]], [[Maximum Likelihood & Fisher Info]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]
- **Visual Cheatsheet:** [[Cheatsheet - Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]
- **Referenced Across Monographs:** [[Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]], [[Source - Speech and Language Processing]]
