---
type: monograph
domain: "D07"
parent_topic: "[[PT 07 - Production ML Systems, Evaluation, Observability & AgentOps]]"
sub_topic: "[[ST 07 - Evaluation, Observability, Guardrails & AgentOps]]"
cheatsheet: "[[Cheatsheet - Outside-In Agent Evaluation, LLM-as-a-Judge, Observability, Guardrails & AgentOps CI - CD]]"
part_of: 
  - "[[ST 07 - Evaluation, Observability, Guardrails & AgentOps]]"
  - "[[D07 - Agentic AI Architectures, Protocols & Production ML Systems]]"
concepts: 
  - "[[AgentOps & pass^k Reliability]]"
prerequisites: 
  - "[[ReAct & Tool-Use Loop]]"
  - "[[Workflow DAGs]]"
enables: []
related: 
  - "[[Distribution Shift & Drift]]"
sources: 
  - "[[Source - A practical]]"
  - "[[Source - Agent Quality]]"
  - "[[Source - Designing Machine Learning Systems]]"
  - "[[Source - Prototype to]]"
  - "[[Source - Reinforcement Learning from Human Feedback]]"
tags:
  - monograph
  - domain-07
---

# Outside-In Agent Evaluation, LLM-as-a-Judge, Observability, Guardrails & AgentOps CI/CD

> [!abstract] Monograph Navigation
> - **Domain:** [[D07 - Agentic AI Architectures, Protocols & Production ML Systems|Agentic AI Architectures, Protocols & Production ML Systems]]
> - **Parent / Sub-Topic:** [[PT 07 - Production ML Systems, Evaluation, Observability & AgentOps|Production ML Systems, Evaluation, Observability & AgentOps]] → [[ST 07 - Evaluation, Observability, Guardrails & AgentOps|Evaluation, Observability, Guardrails & AgentOps]]
> - **Visual Cheatsheet:** [[Cheatsheet - Outside-In Agent Evaluation, LLM-as-a-Judge, Observability, Guardrails & AgentOps CI - CD]]
> - **Description:** Covers non-deterministic agent evaluation hierarchies (end-to-end outcome vs. trajectory/process), deterministic metrics vs. LLM-as-a-Judge & human-in-the-loop calibration, OpenTelemetry logging/tracing/metrics pillars, defense-in-depth guardrails, canary/shadow rollouts, and the Agent Quality Flywheel.

## Problem Formulation & Unified Notation

Deploying autonomous AI agents into production breaks the foundational assumption of classical software engineering: deterministic verification against a static specification. Whereas traditional software executes a fixed control-flow graph, an active agent couples a stochastic foundation model policy $\pi_\theta$ with multi-step planning, external tool invocation $\mathcal{T}$, and mutable short- and long-term memory $\mathcal{M}$ [[Source - Agent Quality|agent_quality]] (pp. 11–13). Across a horizon of $H$ reasoning-and-action turns, the agent generates a trajectory

$$
\tau = \bigl(s_0,  z_1, a_1, o_1,  z_2, a_2, o_2,  \dots,  z_H, a_H, o_H,  y\bigr),
$$

where $s_0$ is the user intent and system context, $z_t$ is the internal reasoning state ("Thought"), $a_t \in \mathcal{A} \cup \mathcal{T}$ is the selected action or parameterized tool call, $o_t$ is the external environment observation, and $y$ is the final user-facing output [[Source - Agent Quality|agent_quality]] (pp. 12, 19–20). Because stochasticity compounds at every step $t \in \{1,\dots,H\}$ and external APIs introduce non-stationary state, evaluation shifts from deterministic *verification* ("Did we build the product right?") to probabilistic *validation* ("Did we build the right product?") [[Source - Agent Quality|agent_quality]] (p. 16).

### The Four Pillars of Agent Quality & Outside-In Hierarchy

To anchor technical metrics in business value, agent quality is evaluated across four coupled pillars [[Source - Agent Quality|agent_quality]] (pp. 14–15):
- **Effectiveness (Goal Achievement):** Black-box task completion against user intent (e.g., pull-request acceptance rate, transaction completion, CSAT).
- **Efficiency (Operational Cost):** Resource footprint per resolved task, measured by total tokens $N_{\mathrm{tok}}(\tau)$, end-to-end latency $L(\tau)$ (P50/P99), and trajectory length $H$.
- **Robustness (Reliability):** Graceful degradation under noisy inputs, API timeouts, malformed tool schemas, and ambiguous prompts.
- **Safety & Alignment (Trustworthiness):** Non-negotiable compliance with policy boundaries, prompt-injection resistance, PII protection, and demographic fairness.

The *Outside-In* evaluation hierarchy operationalizes these pillars in two stages [[Source - Agent Quality|agent_quality]] (pp. 17–20): first scoring the black-box final outcome $y$ against task intent, and only then opening the "glass box" to diagnose intermediate trajectory failures in planning ($z_t$), tool selection/parameterization ($a_t$), tool-response interpretation ($o_t$), or retrieval-augmented generation (RAG) grounding.

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> | **Concept** | **Agent Quality / Ops** | **RLHF / ML Systems** | **Unified Notation** |
> | — | — | — | — |
> | Execution unit | Trajectory (Thought $\to$ Action $\to$ Obs) | Rollout / Harness + Sandbox run | $\tau = (s_0, \{z_t, a_t, o_t\}_{t=1}^H, y)$ |
> | Telemetry record | OpenTelemetry Trace & Spans | Evaluation Harness Log | Trace $\mathcal{G}_\tau = (\mathcal{V}_{\mathrm{span}}, \mathcal{E})$ |
> | Automated scorer | LLM-as-a-Judge / Agent-as-a-Judge | Grader / Reward Model | $J_\phi(\tau) \in [0,1]$ or $J_\phi(\tau_A, \tau_B)$ |
> | Runtime defense | Layered Guardrail Plugin | Confidence Threshold / Deferral | Gate $G(\cdot) \to \{\mathrm{allow}, \mathrm{block}, \mathrm{HITL}\}$ |
> | Subgroup audit | Quality Slice Dashboard | Slice-Based Eval & Model Card | Slice risk $R_g(\pi)$ over $g \in \mathcal{G}_{\mathrm{slice}}$ |
>
>
> **Source Mapping:** Agent Quality & Ops terminology from [[Source - Agent Quality|agent_quality]] (pp. 12, 23–29, 37, 41) and [[Source - A practical|practical_guide_building_agents]] (p. 25); RLHF & ML Systems terminology from [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 16, pp. 197–198) and [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (pp. 205, 371).

## Core Mathematical Derivations & Theorems

### Compounding Horizon Decay: `pass@k` vs. `pass^ k`

In single-turn classification, error rates are additive across samples. In multi-step agent trajectories, errors compound multiplicatively across the horizon $H$.

> [!important] Derivation: Horizon Reliability Decay and Production Consistency
> Suppose an agent must execute $H$ sequential dependent steps (e.g., plan $\to$ tool call $\to$ observation parsing $\to$ synthesis) where each step succeeds independently with conditional probability $p_{\mathrm{step}} \in (0,1)$. The probability that a single trajectory $\tau$ succeeds end-to-end is
>
> $$
> p_\tau = \prod_{t=1}^{H} p_t \approx p_{\mathrm{step}}^{H}.
>
> $$
>
> To characterize non-deterministic capability versus production reliability across $k$ independent rollouts $\tau^{(1)},\dots,\tau^{(k)} \sim \pi_\theta(\cdot \mid s_0)$, we distinguish two estimators:
>
> $$
> \begin{aligned}
> \mathrm{pass@}k &= \mathbb{P}\bigl(\exists  i \in \{1,\dots,k\} : \mathrm{Success}(\tau^{(i)}) = 1\bigr) = 1 - (1 - p_\tau)^k,  \\
>   \mathrm{pass^}k &= \mathbb{P}\bigl(\forall  i \in \{1,\dots,k\} : \mathrm{Success}(\tau^{(i)}) = 1\bigr) = p_\tau^k = p_{\mathrm{step}}^{k H}.
> \end{aligned}
>
> $$
>
> For $p_{\mathrm{step}} = 0.95$ and $H = 10$ steps, $p_\tau = 0.95^{10} \approx 0.599$. Over $k=5$ trials, $\mathrm{pass@}5 \approx 0.990$ (looks production-ready in a demo), whereas $\mathrm{pass^}5 \approx 0.077$ (fails $>92\%$ of 5-run SLA streaks).

This exponential gap between $\mathrm{pass@}k$ and $\mathrm{pass^}k$ explains why prototype demos mislead engineering teams [[Source - Prototype to|prototype_production]] (p. 6, 38) and why production CI/CD gates must enforce multi-run consistency alongside mean task success [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 16, p. 198).

### LLM-as-a-Judge Debiasing, Cohen's $\kappa$, and Statistical Power

Automated metrics (ROUGE, BLEU, BERTScore) act as fast, low-cost regression trend filters in CI/CD, but fail to capture reasoning quality or tool-use validity [[Source - Agent Quality|agent_quality]] (p. 22). Scaling qualitative judgment requires LLM-as-a-Judge (for final responses) and Agent-as-a-Judge (for full execution traces) [[Source - Agent Quality|agent_quality]] (pp. 23–25). However, pointwise 1–5 scoring exhibits high variance and calibration drift, while naive pairwise comparison suffers from position bias (favoring the first response) and verbosity bias.

> [!important] Symmetric Pairwise Estimator and HITL Alignment ($\kappa$)
> Given two agent versions $A$ and $B$ evaluated on query $x$, let $J_\phi(y_1, y_2 \mid x) \in \{1, 0.5, 0\}$ denote the judge score favoring slot 1, tie, or slot 2. To cancel first-slot position bias $\beta_{\mathrm{pos}}$, evaluate both permutations and compute the debiased win rate:
>
> $$
> \hat{w}_{A \succ B}(x) = \frac{1}{2}\Bigl[J_\phi(y_A, y_B \mid x) + \bigl(1 - J_\phi(y_B, y_A \mid x)\bigr)\Bigr].
>
> $$
>
> To validate that the automated judge aligns with Human-in-the-Loop (HITL) domain experts on a calibrated Golden Set $\mathcal{D}_{\mathrm{gold}}$ of size $N$, compute chance-corrected inter-rater agreement via Cohen's $\kappa$:
>
> $$
> \kappa = \frac{p_o - p_e}{1 - p_e}, \qquad p_o = \frac{1}{N}\sum_{i=1}^{N}\mathbb{I}[c_{\mathrm{LLM}}^{(i)} = c_{\mathrm{HITL}}^{(i)}], \quad p_e = \sum_{c \in \mathcal{C}} \hat{p}_{\mathrm{LLM}}(c) \hat{p}_{\mathrm{HITL}}(c).
>
> $$
>
> When comparing a candidate agent $\pi_{\mathrm{cand}}$ against production $\pi_{\mathrm{prod}}$ over $N$ i.i.d. Golden Set tasks with observed win-rate difference $\Delta \hat{w} = \hat{w} - 0.5$ and per-prompt variance $\sigma_w^2$, a two-sided CI/CD promotion gate at significance level $\alpha$ and power $1-\beta$ requires sample size
>
> $$
> N \ge \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \sigma_w^2}{(\Delta w_{\min})^2},
>
> $$
>
> where $\Delta w_{\min}$ is the minimum detectable improvement above the 0.25–1.5 percentage-point evaluation noise floor observed in reasoning-era benchmarks [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 16, p. 198).

> [!tip] Why Pairwise Judging Beats Pointwise Scoring
> [[Source - Agent Quality|agent_quality]] (p. 24) emphasizes that a small prompt or model change rarely shifts a noisy 1–5 pointwise score beyond its calibration jitter. Forcing a relative choice between Answer $A$ (production) and Answer $B$ (candidate) cancels shared query difficulty and yields a high-signal win/loss/tie rate, directly analogous to how RLHF preference models reduce noise during post-training [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 16, p. 199).

### Slice-Based Risk, Simpson's Paradox, and Confidence Deferral

Aggregate pass rates routinely conceal catastrophic failures on minority user cohorts, rare languages, or high-stakes tool categories [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (pp. 205–207).

> [!important] Simpson's Reversal Condition and Optimal HITL Deferral
> Partition the evaluation distribution into slices $g \in \mathcal{G}_{\mathrm{slice}}$ with traffic weights $w_g = \mathbb{P}(g) > 0$. The aggregate success rate of policy $\pi$ is $S(\pi) = \sum_g w_g S_g(\pi)$. Simpson's paradox occurs when candidate $\pi_B$ beats $\pi_A$ on every individual slice ($S_g(\pi_B) > S_g(\pi_A) \forall g$), yet loses in aggregate ($S(\pi_B) < S(\pi_A)$) because evaluation traffic weights $w_g^{(A)} \neq w_g^{(B)}$ differ across slices [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (pp. 206–207).
>
> To prevent silent slice regressions and bound safety risk, each prediction or tool call with calibrated confidence $\hat{p}(x) \in [0,1]$ (verified via reliability diagrams or Platt scaling [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (p. 204)) is routed to HITL review whenever $\hat{p}(x) < \tau_{\mathrm{conf}}(r_a)$, where $r_a \in \{\mathrm{low}, \mathrm{med}, \mathrm{high}\}$ is the tool risk tier [[Source - A practical|practical_guide_building_agents]] (pp. 26, 31). Under false-execution cost $C_{\mathrm{err}}(r_a)$ and human-review cost $C_{\mathrm{HITL}}$, the Bayes-optimal deferral threshold satisfies
>
> $$
> \tau_{\mathrm{conf}}(r_a) = 1 - \frac{C_{\mathrm{HITL}}}{C_{\mathrm{err}}(r_a)}.
>
> $$
>
> High-risk irreversible tools ($C_{\mathrm{err}} \gg C_{\mathrm{HITL}}$, e.g., `execute\_payment` or `delete\_database\_entry`) drive $\tau_{\mathrm{conf}} \to 1$, mandating unconditional HITL approval [[Source - Agent Quality|agent_quality]] (p. 27); [[Source - A practical|practical_guide_building_agents]] (p. 31).

## Algorithmic Implementation & Comparative Synthesis

### Outside-In Trajectory Evaluator & Guarded Canary Gate

Production AgentOps unifies offline CI/CD gating [[Source - Prototype to|prototype_production]] (pp. 12–15) with online canary monitoring and automated rollback [[Source - Prototype to|prototype_production]] (pp. 16, 20–24). Algorithm `alg:ops:outside_in_gate` formalizes this end-to-end promotion pipeline.

> [!example] Algorithm: Outside-In Trajectory Evaluator & Guarded Canary Promotion Gate
> **Input:** Candidate agent $\pi_{\mathrm{cand}}$, production baseline $\pi_{\mathrm{prod}}$, Golden Set $\mathcal{D}_{\mathrm{gold}}$ with slices $\mathcal{G}_{\mathrm{slice}}$, risk-tiered guardrails $G$, thresholds $(\tau_{\mathrm{win}}, \tau_{\mathrm{slice}}, \tau_{\mathrm{err}}, \kappa_{\min})$.
> **Output:** Deployment decision $\in \{Promote, Block, Rollback\}$.
>
>  *// Phase 1: Offline Pre-Merge CI Gate (Outside-In + Glass-Box)*
> **for each** $(x, y^\star, \tau^\star, g) \in \mathcal{D}_{\mathrm{gold}}$:
>
>   Execute $\pi_{\mathrm{cand}}(x)$ in isolated sandbox with OpenTelemetry tracing $\to (\hat{y}, \hat{\tau})$
>   Run deterministic checks: schema validity, regex/PII filter, perturbation invariance
>   **if** safety/PII violation $\lor$ automated metric drops below regression floor:
>
>     **return** Block (Phase 1 deterministic/safety failure)
>
>   Compute debiased pairwise win $\hat{w}(x)$ via Eq. `(eq:ops:debiased_win)` and Agent-as-a-Judge trace score $J_{\mathrm{trace}}(\hat{\tau}, \tau^\star)$
>
> **if** $\kappa(J_\phi, \mathrm{HITL}) < \kappa_{\min}$ **or** $\min_{g \in \mathcal{G}_{\mathrm{slice}}} S_g(\pi_{\mathrm{cand}}) < \tau_{\mathrm{slice}}$ **or** $\bar{w} < \tau_{\mathrm{win}}$:
>
>   **return** Block (Insufficient slice quality or judge calibration)
>
>  *// Phase 2–3: Canary Rollout (1\% $\to$ 10\% $\to$ 100\%) with Guardrails & OTel*
> Route $1\%$ live traffic to $\pi_{\mathrm{cand}}$ with concurrent optimistic `@input\_guardrail` and tool-risk gates
> Monitor P99 latency, error rate $e_{\mathrm{can}}$, and guardrail tripwire rate over window $\Delta t$
> **if** $e_{\mathrm{can}} > \tau_{\mathrm{err}}$ **or** prompt-injection tripwire spike **or** high-risk tool anomaly:
>
>   Trigger circuit-breaker feature flag; capture OTel trace $\mathcal{G}_{\hat{\tau}}$ into $\mathcal{D}_{\mathrm{gold}}$
>   **return** Rollback (Instant revert to $\pi_{\mathrm{prod}}$ + Evolve loop)
>
> **else**:
>
>   Promote $\pi_{\mathrm{cand}}$ to $100\%$ traffic and update Model Card
>   **return** Promote

### Comparative Synthesis of Evaluation & Guardrail Mechanisms

No single evaluator or guardrail suffices in isolation. Table `tab:ops:evaluators` synthesizes the latency, cost, and diagnostic trade-offs across the five evaluation layers and defense-in-depth guardrails.

**Table:** Comparative synthesis of agent evaluation and runtime guardrail mechanisms across latency, cost, and diagnostic scope [[Source - Agent Quality|agent_quality]] (pp. 21–29); [[Source - A practical|practical_guide_building_agents]] (pp. 25–31); [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (pp. 201–207).

| **Mechanism** | **Latency** | **Unit Cost** | **Failure Mode Caught** | **Primary Lifecycle Stage** |
| — | — | — | — | — |
| Rules / Regex / Schema | $<1$ ms | Negligible | SQL injection, PII patterns, malformed JSON args | Runtime input/output + Pre-merge CI |
| Automated Metrics (ROUGE, BERTScore) | $5$–$50$ ms | Very Low | Surface regression vs. Golden Set | Pre-merge CI trend filter |
| Perturbation & Slicing Tests | Offline batch | Low–Med | Noise sensitivity, Simpson's paradox, subgroup bias | Offline CI + Staging CD |
| LLM / Agent-as-a-Judge | $0.5$–$3$ s | Moderate | Flawed plan $z_t$, wrong tool $a_t$, bad tone | Post-merge CD + Async prod sampling |
| HITL / Reviewer UI | Minutes–Hours | High | Subtle domain errors, high-risk tool sign-off | Golden Set curation + High-risk runtime |

> [!info] Cross-Book Synthesis: Offline Rigor Meets Online AgentOps
> [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (pp. 199–207) establishes that offline evaluation must go beyond aggregate accuracy through simple heuristics/baselines, perturbation tests, invariance tests (e.g., demographic feature removal), directional expectation tests, and slice-based audits. [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 16, pp. 197–202) warns that agentic benchmarks depend sensitively on the surrounding harness, sandbox, prompt format, and inference token budget, while public leaderboards suffer from rapid saturation and 8-gram training contamination. [[Source - Prototype to|prototype_production]] (pp. 13–25) bridges these offline limits to production via a three-phase CI/CD funnel (Pre-Merge CI $\to$ Staging Dogfooding $\to$ HITL-Gated Prod), safe canary/blue-green rollouts, and the *Observe $\to$ Act $\to$ Evolve* loop that converts every production failure into a permanent regression test within hours.

## Visual Diagnostics, Observability Architecture & AgentOps Flywheel

### Holistic Evaluation & OpenTelemetry Observability Pillars

Just as a gourmet chef cannot be judged solely by a temperature checklist ("Line Cook" monitoring), an autonomous agent requires *true observability* built on three OpenTelemetry pillars [[Source - Agent Quality|agent_quality]] (pp. 31–40):
- **Logs (The Agent's Diary):** Structured JSON events capturing prompts, tool arguments, intermediate reasoning, and errors, scrubbed for PII.
- **Traces (Following the Footsteps):** Causal Directed Acyclic Graphs of nested spans (`invocation` $\to$ `agent\_run` $\to$ `call\_llm` $\to$ `execute\_tool`) linking every step to a single `trace\_id`.
- **Metrics (The Health Report):** Split into *System Metrics* (P50/P99 latency, error rate, tokens/task, API cost) for SRE operational dashboards, and second-order *Quality Metrics* (correctness, trajectory adherence, safety) for AgentOps dashboards [[Source - Agent Quality|agent_quality]] (pp. 38–41).

> [!note] Source Figure
> *Figure:* Holistic Outside-In evaluation hierarchy separating Output Evaluation (black-box task success, CSAT) from Process Evaluation (glass-box planning, tool use, memory), paired with hybrid judgment methods and a mandatory Responsible AI gate [[Source - Agent Quality|agent_quality]] (Ch. 2, p. 18). (`agent_quality_p18_fig1.png`)

> [!note] Source Figure
> *Figure:* OpenTelemetry agent trace waterfall showing causal span nesting (`invocation` $\to$ `agent\_run` $\to$ `call\_llm` $\to$ `execute\_tool get\_weather`), token counts, and sub-millisecond vs. LLM latency attribution [[Source - Agent Quality|agent_quality]] (Ch. 3, p. 37). (`agent_quality_p37_fig1.png`)

### Closed-Loop AgentOps Flywheel & Multi-Environment Governance

Operationalizing quality requires closing the loop between pre-production CI/CD gates, layered defense-in-depth guardrails [[Source - A practical|practical_guide_building_agents]] (pp. 25–31), live OpenTelemetry tracing, and continuous Golden Set evolution [[Source - Agent Quality|agent_quality]] (pp. 45–46); [[Source - Prototype to|prototype_production]] (pp. 22–25, 36–37). Figure `fig:ops:agentops_flywheel_tikz` and Figure `fig:ops:agentops_envs` illustrate the closed-loop architecture and enterprise environment topology.

> [!example] Visual Diagram (LaTeX / PDF)
> *Closed-loop AgentOps Flywheel integrating pre-deployment CI/CD gates, defense-in-depth runtime guardrails, OpenTelemetry trace collection, debiased LLM/HITL judgment, and automated regression-set evolution [[Source - Agent Quality|agent_quality]] (pp. 45–46); [[Source - Prototype to|prototype_production]] (pp. 22–25).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!note] Source Figure
> *Figure:* End-to-end AgentOps reference topology spanning Data Lake/Mesh governance, Development (AI Model Gateway, Guardrails), Staging (simulation and scale eval), Production (A/B serving, observability, RAI alerts), and centralized Tool/Agent Registries [[Source - Prototype to|prototype_production]] (p. 37). (`prototype_production_p37_fig1.png`)

### Quantitative Trade-Off: Multi-Turn Consistency Decay

Figure `fig:ops:pass_k_variance` plots the analytical divergence derived in Eqs. `(eq:ops:horizon_decay)`–`(eq:ops:pass_pow_k)` across trajectory horizon $H \in [1, 15]$ for per-step accuracies $p_{\mathrm{step}} \in \{0.95, 0.90\}$. Even a $95\%$ per-step tool-calling policy drops below $50\%$ single-run task success by $H=14$ steps and exhibits near-zero 5-run consistency ($\mathrm{pass^}5$), underscoring why step-level process evaluation and idempotent retries are mandatory.

> [!example] Visual Diagram (LaTeX / PDF)
> *Compounding trajectory decay across horizon $H$: optimistic capability ($\mathrm{pass@}5$) remains above $0.95$ even when single-run reliability ($\mathrm{pass@}1$) and 5-run production consistency ($\mathrm{pass^}5$) collapse exponentially.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] LLM-as-a-Judge Position, Verbosity & Self-Preference Bias
> Uncalibrated LLM judges systematically favor the first candidate in a prompt, reward verbose responses over concise correct ones, and exhibit self-preference for outputs from the same model family [[Source - Agent Quality|agent_quality]] (p. 24). **Remedy:** Always run symmetric slot-swapped pairwise evaluation (Eq. `(eq:ops:debiased_win)`), enforce length-neutral rubrics, and audit judge agreement against HITL domain experts via Cohen's $\kappa$ (Eq. `(eq:ops:cohen_kappa)`) on every Golden Set release.

> [!warning] Benchmark Saturation, Harness Drift & 8-Gram Contamination
> Public benchmarks saturate rapidly and cannot distinguish genuine capability gains from web-scraped leakage or prompt-format overfitting [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 16, pp. 198–202). Moreover, changing the agent harness, sandbox, or inference token budget can swing scores by more than model weight updates. **Remedy:** Run 8-gram and 50-character substring decontamination, maintain private held-out Golden Sets, and pair static benchmarks with perturbation tests [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (pp. 201–202).

> [!warning] Simpson's Paradox & Aggregate Canary Masking
> A candidate agent can improve overall task success rate while severely degrading a minority demographic, mobile traffic, or paid-user slice [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (pp. 205–207). Because error costs escalate by roughly an order of magnitude per lifecycle stage [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (p. 371), shipping an un-sliced regression is costly. **Remedy:** Enforce per-slice minimum thresholds ($\min_g S_g$) in CI/CD (Algorithm `alg:ops:outside_in_gate`) and publish updated Model Cards documenting unitary and intersectional results [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (pp. 371–372).

> [!warning] Non-Idempotent Tool Retries & Guardrail Latency Spikes
> Retrying a failed network call on a non-idempotent tool (e.g., `initiate\_refund`) causes duplicate financial transactions [[Source - Prototype to|prototype_production]] (p. 21), while running sequential LLM guardrails on every turn inflates P99 latency. **Remedy:** Enforce idempotency keys on all write tools with exponential backoff, execute `@input\_guardrail` checks concurrently via optimistic execution [[Source - A practical|practical_guide_building_agents]] (p. 31), and pair high-risk tools with circuit-breaker feature flags and HITL approval pauses [[Source - Prototype to|prototype_production]] (p. 22); [[Source - Agent Quality|agent_quality]] (p. 27).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (1):** [[AgentOps & pass^k Reliability]]
- **Prerequisites (Directed $\leftarrow$):** [[ReAct & Tool-Use Loop]], [[Workflow DAGs]]
- **Enables / Unlocks (Directed $\rightarrow$):** None
- **Related Concepts (Undirected $\leftrightarrow$):** [[Distribution Shift & Drift]]
- **Synthesized Sources:** [[Source - A practical]], [[Source - Agent Quality]], [[Source - Designing Machine Learning Systems]], [[Source - Prototype to]], [[Source - Reinforcement Learning from Human Feedback]]
