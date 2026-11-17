---
type: algorithm
domain: "D02"
aliases: 
  - "MCMC"
  - "Hamiltonian Monte Carlo"
  - "Langevin Dynamics"
part_of: 
  - "[[Monte Carlo Integration, Metropolis-Hastings, Gibbs, HMC & Langevin Sampling]]"
  - "[[Cheatsheet - Monte Carlo Integration, Metropolis-Hastings, Gibbs, HMC & Langevin Sampling]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[Graphical Models (BN & MRF)]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Energy-Based Models (EBM)]]"
  - "[[Score Matching (DSM & SSM)]]"
contrasts_with: 
  - "[[Variational Inference & ELBO]]"
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Tutorial on Diffusion Models for Imaging and Vision]]"
tags:
  - algorithm
  - domain-02
---

# MCMC & HMC Sampling

> [!abstract] Definition (Algorithm)
> Ergodic Markov chains (Metropolis-Hastings, Gibbs) and symplectic Hamiltonian dynamics (HMC/NUTS) for asymptotically exact posterior sampling.

$$
H(\boldsymbol{q},\boldsymbol{p}) = U(\boldsymbol{q}) + \frac{1}{2}\boldsymbol{p}^\top\boldsymbol{M}^{-1}\boldsymbol{p}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Graphical Models (BN & MRF)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Energy-Based Models (EBM)]], [[Score Matching (DSM & SSM)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Variational Inference & ELBO]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Monte Carlo Integration, Metropolis-Hastings, Gibbs, HMC & Langevin Sampling]]
- **Visual Cheatsheet:** [[Cheatsheet - Monte Carlo Integration, Metropolis-Hastings, Gibbs, HMC & Langevin Sampling]]
- **Referenced Across Monographs:** [[Monte Carlo Integration, Metropolis-Hastings, Gibbs, HMC & Langevin Sampling]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Tutorial on Diffusion Models for Imaging and Vision]]
