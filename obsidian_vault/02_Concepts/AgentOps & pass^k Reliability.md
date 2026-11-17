---
type: metric
domain: "D07"
aliases: 
  - "AgentOps"
  - "pass^k"
part_of: 
  - "[[Outside-In Agent Evaluation, LLM-as-a-Judge, Observability, Guardrails & AgentOps CI - CD]]"
  - "[[Cheatsheet - Outside-In Agent Evaluation, LLM-as-a-Judge, Observability, Guardrails & AgentOps CI - CD]]"
  - "[[D07 - Agentic AI Architectures, Protocols & Production ML Systems]]"
prerequisites: 
  - "[[ReAct & Tool-Use Loop]]"
  - "[[Workflow DAGs]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Distribution Shift & Drift]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - A practical]]"
  - "[[Source - Agent Quality]]"
  - "[[Source - Designing Machine Learning Systems]]"
  - "[[Source - Prototype to]]"
  - "[[Source - Reinforcement Learning from Human Feedback]]"
tags:
  - metric
  - domain-07
---

# AgentOps & pass^k Reliability

> [!abstract] Definition (Metric)
> Distinguishes demo pass@k from production SLA consistency pass^k = p_step^(kH) under OpenTelemetry tracing and guardrails.

$$
\mathrm{pass}^k = p_{\tau}^k \approx p_{\mathrm{step}}^{kH}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[ReAct & Tool-Use Loop]], [[Workflow DAGs]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Distribution Shift & Drift]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Outside-In Agent Evaluation, LLM-as-a-Judge, Observability, Guardrails & AgentOps CI - CD]]
- **Visual Cheatsheet:** [[Cheatsheet - Outside-In Agent Evaluation, LLM-as-a-Judge, Observability, Guardrails & AgentOps CI - CD]]
- **Referenced Across Monographs:** [[Outside-In Agent Evaluation, LLM-as-a-Judge, Observability, Guardrails & AgentOps CI - CD]]
- **Primary Textbooks:** [[Source - A practical]], [[Source - Agent Quality]], [[Source - Designing Machine Learning Systems]], [[Source - Prototype to]], [[Source - Reinforcement Learning from Human Feedback]]
