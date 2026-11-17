---
type: theorem
domain: "D01"
aliases: []
part_of: 
  - "[[Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]"
  - "[[Cheatsheet - Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Lagrange Duality & KKT]]"
enables: 
  - "[[Wasserstein GAN (WGAN)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Lagrange Duality & KKT]]"
  - "[[Wasserstein GAN (WGAN)]]"
contrasts_with: []
dual_to: 
  - "[[Wasserstein Distance]]"
sources: 
  - "[[Source - Optimal Transport in Large-Scale]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
tags:
  - theorem
  - domain-01
---

# Kantorovich Duality

> [!abstract] Definition (Theorem)
> Dual formulation of Optimal Transport over c-concave potentials; reduces W_1 to maximization over 1-Lipschitz critics.

$$
W_1(\mu, \nu) = \sup_{\|f\|_{\mathrm{Lip}}\le 1} \left(\mathbb{E}_\mu[f] - \mathbb{E}_\nu[f]\right)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Lagrange Duality & KKT]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Wasserstein GAN (WGAN)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Lagrange Duality & KKT]], [[Wasserstein GAN (WGAN)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | [[Wasserstein Distance]] |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]
- **Visual Cheatsheet:** [[Cheatsheet - Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]
- **Referenced Across Monographs:** [[Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]
- **Primary Textbooks:** [[Source - Optimal Transport in Large-Scale]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]]
