---
type: architecture
domain: "D04"
aliases: []
part_of: 
  - "[[GAN Game Dynamics, Mode Collapse, f-GANs & Wasserstein - OT-GANs]]"
  - "[[Cheatsheet - GAN Game Dynamics, Mode Collapse, f-GANs & Wasserstein - OT-GANs]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
prerequisites: 
  - "[[GAN & Mode Collapse]]"
  - "[[Kantorovich Duality]]"
  - "[[Wasserstein Distance]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Kantorovich Duality]]"
  - "[[Wasserstein Distance]]"
contrasts_with: 
  - "[[GAN & Mode Collapse]]"
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

# Wasserstein GAN (WGAN)

> [!abstract] Definition (Architecture)
> Replaces JS discriminator with a 1-Lipschitz Kantorovich critic via gradient penalty or spectral norm for stable Wasserstein-1 training.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[GAN & Mode Collapse]], [[Kantorovich Duality]], [[Wasserstein Distance]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Kantorovich Duality]], [[Wasserstein Distance]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[GAN & Mode Collapse]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[GAN Game Dynamics, Mode Collapse, f-GANs & Wasserstein - OT-GANs]]
- **Visual Cheatsheet:** [[Cheatsheet - GAN Game Dynamics, Mode Collapse, f-GANs & Wasserstein - OT-GANs]]
- **Referenced Across Monographs:** [[GAN Game Dynamics, Mode Collapse, f-GANs & Wasserstein - OT-GANs]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Optimal Transport in Large-Scale]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
