---
type: architecture
domain: "D04"
aliases: 
  - "GAN"
part_of: 
  - "[[GAN Game Dynamics, Mode Collapse, f-GANs & Wasserstein - OT-GANs]]"
  - "[[Cheatsheet - GAN Game Dynamics, Mode Collapse, f-GANs & Wasserstein - OT-GANs]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
prerequisites: []
enables: 
  - "[[Wasserstein GAN (WGAN)]]"
generalizes: []
special_case_of: []
related: []
contrasts_with: 
  - "[[Variational Autoencoder (VAE)]]"
  - "[[Wasserstein GAN (WGAN)]]"
dual_to: []
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Optimal Transport in Large-Scale]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - architecture
  - domain-04
---

# GAN & Mode Collapse

> [!abstract] Definition (Architecture)
> Implicit minimax game between generator and discriminator; minimizes JS divergence at optimum but suffers vanishing gradients and mode collapse.

$$
\min_G\max_D \; \mathbb{E}_{\boldsymbol{x}\sim p_{\mathrm{data}}}[\log D(\boldsymbol{x})] + \mathbb{E}_{\boldsymbol{z}\sim p_{\boldsymbol{z}}}[\log(1 - D(G(\boldsymbol{z})))]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Wasserstein GAN (WGAN)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | — |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Variational Autoencoder (VAE)]], [[Wasserstein GAN (WGAN)]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[GAN Game Dynamics, Mode Collapse, f-GANs & Wasserstein - OT-GANs]]
- **Visual Cheatsheet:** [[Cheatsheet - GAN Game Dynamics, Mode Collapse, f-GANs & Wasserstein - OT-GANs]]
- **Referenced Across Monographs:** [[GAN Game Dynamics, Mode Collapse, f-GANs & Wasserstein - OT-GANs]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Optimal Transport in Large-Scale]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
