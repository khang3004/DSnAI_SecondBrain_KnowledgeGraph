---
type: architecture
domain: "D07"
aliases: 
  - "FTI Architecture"
  - "Data Leakage"
part_of: 
  - "[[Iterative ML Systems Design, Feature Engineering, Leakage, Distribution Shift & Continual Learning]]"
  - "[[Cheatsheet - Iterative ML Systems Design, Feature Engineering, Leakage, Distribution Shift & Continual Learning]]"
  - "[[D07 - Agentic AI Architectures, Protocols & Production ML Systems]]"
prerequisites: []
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Distribution Shift & Drift]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Designing Machine Learning Systems]]"
  - "[[Source - LLM Engineer’s Handbook]]"
tags:
  - architecture
  - domain-07
---

# FTI Architecture & Leakage Prevention

> [!abstract] Definition (Architecture)
> Decouples Feature, Training, and Inference pipelines with point-in-time as-of joins and split-invariant transforms.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Distribution Shift & Drift]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Iterative ML Systems Design, Feature Engineering, Leakage, Distribution Shift & Continual Learning]]
- **Visual Cheatsheet:** [[Cheatsheet - Iterative ML Systems Design, Feature Engineering, Leakage, Distribution Shift & Continual Learning]]
- **Referenced Across Monographs:** [[Iterative ML Systems Design, Feature Engineering, Leakage, Distribution Shift & Continual Learning]]
- **Primary Textbooks:** [[Source - Designing Machine Learning Systems]], [[Source - LLM Engineer’s Handbook]]
