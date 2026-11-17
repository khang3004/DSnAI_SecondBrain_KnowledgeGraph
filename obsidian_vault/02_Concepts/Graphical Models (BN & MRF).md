---
type: object
domain: "D02"
aliases: 
  - "Bayesian Networks"
  - "Markov Random Fields"
part_of: 
  - "[[Probabilistic Graphical Models, Message Passing & Gaussian Processes]]"
  - "[[Cheatsheet - Probabilistic Graphical Models, Message Passing & Gaussian Processes]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: []
enables: 
  - "[[MCMC & HMC Sampling]]"
generalizes: []
special_case_of: []
related: 
  - "[[Energy-Based Models (EBM)]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - object
  - domain-02
---

# Graphical Models (BN & MRF)

> [!abstract] Definition (Object)
> Factorizes high-dimensional distributions via DAGs (Bayesian Networks) or undirected cliques (Markov Random Fields / Hammersley-Clifford).

$$
p(\boldsymbol{x}) = \frac{1}{Z}\prod_{c\in\mathcal{C}} \psi_c(\boldsymbol{x}_c)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[MCMC & HMC Sampling]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Energy-Based Models (EBM)]], [[Multivariate Gaussian & Schur Complement]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Probabilistic Graphical Models, Message Passing & Gaussian Processes]]
- **Visual Cheatsheet:** [[Cheatsheet - Probabilistic Graphical Models, Message Passing & Gaussian Processes]]
- **Referenced Across Monographs:** [[Probabilistic Graphical Models, Message Passing & Gaussian Processes]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]]
