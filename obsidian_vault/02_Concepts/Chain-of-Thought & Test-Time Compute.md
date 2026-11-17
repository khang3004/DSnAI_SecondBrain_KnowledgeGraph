---
type: concept
domain: "D05"
aliases: 
  - "CoT"
  - "Test-Time Scaling"
part_of: 
  - "[[Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]"
  - "[[Cheatsheet - Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]"
  - "[[D05 - Transformers, Language Modeling & LLM Engineering]]"
prerequisites: []
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[MCTS & MuZero]]"
  - "[[RLVR (Verifiable Rewards)]]"
  - "[[ReAct & Tool-Use Loop]]"
  - "[[Speculative Decoding]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Foundations of Large Language Models]]"
  - "[[Source - Reinforcement Learning from Human Feedback]]"
  - "[[Source - Speech and Language Processing]]"
tags:
  - concept
  - domain-05
---

# Chain-of-Thought & Test-Time Compute

> [!abstract] Definition (Concept)
> Allocates extra inference tokens/verifiers (CoT, self-consistency, MCTS) to trade latency for reasoning accuracy.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[MCTS & MuZero]], [[RLVR (Verifiable Rewards)]], [[ReAct & Tool-Use Loop]], [[Speculative Decoding]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]
- **Visual Cheatsheet:** [[Cheatsheet - Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]
- **Referenced Across Monographs:** [[Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Foundations of Large Language Models]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Speech and Language Processing]]
