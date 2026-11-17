---
type: home_moc
tags:
  - moc
  - home
---

# AI/ML SecondBrain Knowledge Graph

> [!info] Graph Architecture (Directed + Undirected)
> - **[[00_Concept_Graph_Index]]**: Atomic concepts, theorems, algorithms, objects, architectures, and metrics.
> - **Directed Edges:** `prerequisites` ($\leftarrow$), `enables` ($\rightarrow$), `generalizes` ($\supset$), `special_case_of` ($\subset$), `part_of`.
> - **Undirected Edges:** `related` ($\leftrightarrow$), `contrasts_with` ($\leftrightarrow$), `dual_to` ($\rightleftarrows$).

## 7 Knowledge Domains

- **D01:** [[D01 - Mathematical Foundations for Machine Intelligence|Mathematical Foundations for Machine Intelligence]]
- **D02:** [[D02 - Classical, Kernel & Probabilistic Machine Learning|Classical, Kernel & Probabilistic Machine Learning]]
- **D03:** [[D03 - Neural Architectures, Representations & Regularization|Neural Architectures, Representations & Regularization]]
- **D04:** [[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion|Deep Generative Modeling: VAEs, Flows, EBMs, GANs & Diffusion]]
- **D05:** [[D05 - Transformers, Language Modeling & LLM Engineering|Transformers, Language Modeling & LLM Engineering]]
- **D06:** [[D06 - Reinforcement Learning, Control & LLM Alignment|Reinforcement Learning, Control & LLM Alignment]]
- **D07:** [[D07 - Agentic AI Architectures, Protocols & Production ML Systems|Agentic AI Architectures, Protocols & Production ML Systems]]

## Core Cross-Domain Bridges

- **Geometry & Optimization:** [[Singular Value Decomposition (SVD)]] ↔ [[Low-Rank Approximation (Eckart-Young)]] → [[LoRA & QLoRA]]
- **Information & Alignment:** [[KL Divergence]] → [[Variational Inference & ELBO]] → [[Variational Autoencoder (VAE)]] / [[TRPO]] → [[RLHF (KL-Regularized)]] ↔ [[DPO (Direct Preference Optimization)]]
- **Transport & Diffusion:** [[Wasserstein Distance]] ⇄ [[Kantorovich Duality]] → [[Wasserstein GAN (WGAN)]] / [[Score-Based SDE]] → [[Probability Flow ODE]] → [[Flow Matching & Rectified Flow]]
- **Transformers & Agents:** [[Self-Attention & Multi-Head Attention]] → [[Transformer Architecture]] → [[Chain-of-Thought & Test-Time Compute]] → [[ReAct & Tool-Use Loop]] → [[Model Context Protocol (MCP)]] ↔ [[Agent-to-Agent Protocol (A2A)]]
