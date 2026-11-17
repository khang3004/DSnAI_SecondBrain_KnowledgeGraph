---
type: algorithm
domain: "D02"
aliases: 
  - "PCA"
part_of: 
  - "[[PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]"
  - "[[Cheatsheet - PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Singular Value Decomposition (SVD)]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Autoencoders]]"
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Low-Rank Approximation (Eckart-Young)]]"
  - "[[Singular Value Decomposition (SVD)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - algorithm
  - domain-02
---

# Principal Component Analysis (PCA)

> [!abstract] Definition (Algorithm)
> Orthogonal linear projection maximizing retained variance and minimizing reconstruction MSE simultaneously via top-k eigenvectors/SVD.

$$
\max_{\boldsymbol{W}^\top\boldsymbol{W}=\boldsymbol{I}_k} \mathrm{Tr}(\boldsymbol{W}^\top\boldsymbol{S}\boldsymbol{W})
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Eigendecomposition & Spectral Theorem]], [[Singular Value Decomposition (SVD)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Autoencoders]], [[Eigendecomposition & Spectral Theorem]], [[Low-Rank Approximation (Eckart-Young)]], [[Singular Value Decomposition (SVD)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]
- **Visual Cheatsheet:** [[Cheatsheet - PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]
- **Referenced Across Monographs:** [[PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
