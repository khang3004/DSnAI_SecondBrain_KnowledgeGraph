---
type: metric
domain: "D01"
aliases: 
  - "Optimal Transport"
  - "Earth Mover's Distance"
part_of: 
  - "[[Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]"
  - "[[Cheatsheet - Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: []
enables: 
  - "[[Flow Matching & Rectified Flow]]"
  - "[[Sinkhorn Algorithm]]"
  - "[[Wasserstein GAN (WGAN)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Flow Matching & Rectified Flow]]"
  - "[[Sinkhorn Algorithm]]"
  - "[[Wasserstein GAN (WGAN)]]"
contrasts_with: 
  - "[[KL Divergence]]"
  - "[[f-Divergence]]"
dual_to: 
  - "[[Kantorovich Duality]]"
sources: 
  - "[[Source - Optimal Transport in Large-Scale]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
tags:
  - metric
  - domain-01
---

# Wasserstein Distance

> [!abstract] Definition (Metric)
> Optimal transport metric lifting ground geometry to distributions; remains smooth and informative even on disjoint supports.

$$
W_p(\mu, \nu) = \left(\inf_{\gamma \in \Pi(\mu,\nu)} \int c(x,y)^p\,\mathrm{d}\gamma(x,y)\right)^{1/p}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Flow Matching & Rectified Flow]], [[Sinkhorn Algorithm]], [[Wasserstein GAN (WGAN)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Flow Matching & Rectified Flow]], [[Sinkhorn Algorithm]], [[Wasserstein GAN (WGAN)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[KL Divergence]], [[f-Divergence]] |
| Undirected ($\rightleftarrows$) | **Dual To** | [[Kantorovich Duality]] |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]
- **Visual Cheatsheet:** [[Cheatsheet - Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]
- **Referenced Across Monographs:** [[Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling]]
- **Primary Textbooks:** [[Source - Optimal Transport in Large-Scale]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]]
