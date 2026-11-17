---
type: metric
domain: "D01"
aliases: 
  - "Kullback-Leibler Divergence"
  - "Forward KL"
  - "Reverse KL"
part_of: 
  - "[[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]"
  - "[[Cheatsheet - Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Shannon Entropy]]"
enables: 
  - "[[DPO (Direct Preference Optimization)]]"
  - "[[Maximum Likelihood & Fisher Info]]"
  - "[[Mutual Information & DPI]]"
  - "[[RLHF (KL-Regularized)]]"
  - "[[Sinkhorn Algorithm]]"
  - "[[TRPO]]"
  - "[[Variational Inference & ELBO]]"
generalizes: []
special_case_of: 
  - "[[f-Divergence]]"
related: 
  - "[[DPO (Direct Preference Optimization)]]"
  - "[[Maximum Likelihood & Fisher Info]]"
  - "[[RLHF (KL-Regularized)]]"
  - "[[Shannon Entropy]]"
  - "[[TRPO]]"
  - "[[Variational Inference & ELBO]]"
contrasts_with: 
  - "[[Wasserstein Distance]]"
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - metric
  - domain-01
---

# KL Divergence

> [!abstract] Definition (Metric)
> Asymmetric information divergence: Forward KL (M-projection, mode-covering) vs. Reverse KL (I-projection, mode-seeking).

$$
D_{\mathrm{KL}}(p \| q) = \mathbb{E}_p\!\left[\log\frac{p(X)}{q(X)}\right] \ge 0
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Shannon Entropy]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[DPO (Direct Preference Optimization)]], [[Maximum Likelihood & Fisher Info]], [[Mutual Information & DPI]], [[RLHF (KL-Regularized)]], [[Sinkhorn Algorithm]], [[TRPO]], [[Variational Inference & ELBO]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | [[f-Divergence]] |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[DPO (Direct Preference Optimization)]], [[Maximum Likelihood & Fisher Info]], [[RLHF (KL-Regularized)]], [[Shannon Entropy]], [[TRPO]], [[Variational Inference & ELBO]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Wasserstein Distance]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]
- **Visual Cheatsheet:** [[Cheatsheet - Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]
- **Referenced Across Monographs:** [[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
