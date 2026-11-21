---
type: monograph
domain: "D07"
parent_topic: "[[PT 07 - Production ML Systems, Evaluation, Observability & AgentOps]]"
sub_topic: "[[ST 07 - ML Systems Design, Data Pipelines & Distribution Shifts]]"
cheatsheet: "[[Cheatsheet - Iterative ML Systems Design, Feature Engineering, Leakage, Distribution Shift & Continual Learning]]"
part_of: 
  - "[[ST 07 - ML Systems Design, Data Pipelines & Distribution Shifts]]"
  - "[[D07 - Agentic AI Architectures, Protocols & Production ML Systems]]"
concepts: 
  - "[[Distribution Shift & Drift]]"
  - "[[FTI Architecture & Leakage Prevention]]"
prerequisites: []
enables: []
related: 
  - "[[AgentOps & pass^k Reliability]]"
  - "[[Distribution Shift & Drift]]"
  - "[[FTI Architecture & Leakage Prevention]]"
sources: 
  - "[[Source - Designing Machine Learning Systems]]"
  - "[[Source - LLM Engineer’s Handbook]]"
tags:
  - monograph
  - domain-07
---

# Iterative ML Systems Design, Feature Engineering, Leakage, Distribution Shift & Continual Learning

> [!abstract] Monograph Navigation
> - **Domain:** [[D07 - Agentic AI Architectures, Protocols & Production ML Systems|Agentic AI Architectures, Protocols & Production ML Systems]]
> - **Parent / Sub-Topic:** [[PT 07 - Production ML Systems, Evaluation, Observability & AgentOps|Production ML Systems, Evaluation, Observability & AgentOps]] → [[ST 07 - ML Systems Design, Data Pipelines & Distribution Shifts|ML Systems Design, Data Pipelines & Distribution Shifts]]
> - **Visual Cheatsheet:** [[Cheatsheet - Iterative ML Systems Design, Feature Engineering, Leakage, Distribution Shift & Continual Learning]]
> - **Description:** Synthesizes iterative ML systems architecture, the FTI (Feature-Training-Inference) pipeline separation, sampling/labeling/class imbalance, data leakage diagnostics, model compression & batch-vs-online serving, covariate/label/concept shift mathematics, and continual retraining.

## Problem Formulation & Unified Notation

Production machine learning systems differ fundamentally from research prototypes: whereas research optimizes offline predictive accuracy on static benchmark splits, production systems must satisfy coupled operational and business constraints under continuous data evolution [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 1, p. 32). Following [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 2, pp. 46–52), an ML system is engineered iteratively around four non-functional pillars—reliability, scalability, maintainability, and adaptability—while explicitly mapping offline ML metrics (cross-entropy loss, F1, AUC-ROC, p99 latency) to business objectives (conversion rate, take-rate, fraud loss prevented).

At the architectural level, monolithic batch scripts that couple feature extraction, model training, and inference inside a single pipeline fail to scale horizontally across heterogeneous compute tiers and impede feature reuse [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 1, pp. 10–12 (PDF pp. 39–41)). Both [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 1, pp. 13–15 (PDF pp. 42–44)) and [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 7, pp. 204–205 (PDF pp. 224–225)) converge on decoupling the lifecycle into three independent, artifact-mediated pipelines—the **Feature–Training–Inference (FTI)** architecture:
- **Feature Pipeline ($\mathcal{P}_F$):** Ingests raw events from transactional databases, NoSQL document stores (e.g., MongoDB), or real-time streams, computes deterministic transformations $\boldsymbol{x}_t = \phi(\boldsymbol{r}_{\le t})$, and writes versioned feature vectors and labels $(\boldsymbol{x}_t, y_t)$ to a *feature store* or logical vector warehouse (e.g., Qdrant) [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 1–2, pp. 14, 47–48 (PDF pp. 43, 76–77)).
- **Training Pipeline ($\mathcal{P}_T$):** Queries point-in-time consistent snapshots $\mathcal{D}_{\mathrm{train}} = \{(\boldsymbol{x}_i, y_i)\}_{i=1}^N$ from the feature store, optimizes parameters $\theta^* \in \Theta$, logs lineage metadata (e.g., via ZenML artifacts and Comet ML), and publishes versioned weights to a *model registry* [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 2, pp. 39–45 (PDF pp. 68–74)).
- **Inference Pipeline ($\mathcal{P}_I$):** Loads a registered model $f_{\theta^*}$ and retrieves online/batch features from the feature store (or vector embeddings for RAG) to emit predictions $\hat{y}_t = f_{\theta^*}(\boldsymbol{x}_t)$ synchronously via REST/gRPC or asynchronously in batch, logging inputs and predictions for drift monitoring [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 7–8, pp. 197–199, 251 (PDF pp. 217–219, 271)).

Let $\mathcal{X} \subseteq \mathbb{R}^d$ denote the engineered feature space and $\mathcal{Y}$ the target space. Training data are drawn from a source joint distribution $P_{\mathrm{src}}(X, Y)$, whereas production inference queries arrive from a time-varying target distribution $P_{\mathrm{tgt}}^{(t)}(X, Y)$ at epoch $t$. Table `tab:mlsys:notation` reconciles the notation and systems terminology across our primary sources.

> [!info] Cross-Book Notation & Pedagogical Reconciliation
>
> | **Concept** | **Designing ML Systems (Ch. 3–9)** | **LLM Engineer's Handbook (Ch. 1–3)** | **Unified Monograph Convention** |
> | --- | --- | --- | --- |
> | Pipeline Decoupling | Unified batch/stream feature store eliminating train-serving skew | FTI pattern: Feature, Training, Inference pipelines via Feature Store | $\mathcal{P}_F \to \mathcal{S}_F \to \mathcal{P}_T \to \mathcal{R}_M \to \mathcal{P}_I$ with versioned contracts |
> | Raw vs. Vector Storage | Relational (OLTP/OLAP) vs. NoSQL; row- vs. column-major (Parquet) | MongoDB (unstructured raw docs) + Qdrant (chunked vector index) | Raw warehouse $\mathcal{W}_{\mathrm{raw}}$ feeding logical feature/vector store $\mathcal{S}_F$ |
> | Lineage & Artifact | Model store + data/model lineage tracking (MLflow, Stitch Fix) | ZenML step artifacts, metadata store, Hugging Face registry | Versioned tuple $(v_{\mathrm{data}}, v_{\phi}, v_{\theta})$ |
> | Distribution Shift | $P(X,Y)$ split into Covariate $P(X)$, Label $P(Y)$, Concept $P(Y\mid X)$ | Prompt/query monitoring (Opik) and CT/CI/CD retrain triggers | $P_{\mathrm{src}}(X,Y)$ vs. $P_{\mathrm{tgt}}^{(t)}(X,Y)$ monitored on sliding windows $W_t$ |
> | Model Updating | Stateless retraining (from scratch) vs. stateful training (fine-tuning) | Continuous Training (CT) triggered by new data/artifacts | Stateless $\theta^{(k)} \leftarrow \mathcal{A}(\mathcal{D}_{t-L:t})$ vs. stateful $\theta^{(k)} \leftarrow \mathcal{A}(\theta^{(k-1)}, \mathcal{D}_{t-1:t})$ |
>
> *Sources reconciled:* [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 3–9) and [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 1–3).

## Core Mathematical Derivations & Theorems

### Class Imbalance, Resampling Bounds, and Loss Reweighting

Real-world production tasks (fraud detection, rare-event click prediction, fault diagnosis) exhibit severe class imbalance where minority class priors $\pi_c = \mathbb{P}(Y = c) \ll 1/K$ [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 4, pp. 102–105 (PDF pp. 122–125)). Standard empirical risk minimization $\mathcal{L}(X; \theta) = \frac{1}{N} \sum_{i=1}^N \ell(\boldsymbol{x}_i, y_i; \theta)$ weights every instance equally, biasing gradient updates toward majority classes and yielding degenerate decision boundaries.

Data-level resampling alters the training distribution via random undersampling (RUS), random oversampling (ROS), Tomek links (removing majority-class members of opposite-class nearest-neighbor pairs), or SMOTE (synthesizing minority points via convex combinations $\tilde{\boldsymbol{x}} = \boldsymbol{x}_i + \lambda(\boldsymbol{x}_{\mathrm{nn}} - \boldsymbol{x}_i)$, $\lambda \sim \mathcal{U}(0,1)$) [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 4, pp. 109–110 (PDF pp. 129–130)). However, distance-based resampling degrades in high-dimensional spaces ($d \gg 100$) due to distance concentration, and evaluation must *never* be performed on resampled splits. Algorithm-level methods preserve the true data distribution while modifying the objective.

> [!important] Cost-Sensitive, Class-Balanced, and Focal Loss Formulations
> Let $C_{ij} \ge 0$ denote the domain cost of predicting class $j$ when the true class is $i$ ($C_{ii} = 0$), and let $P(j \mid \boldsymbol{x}; \theta)$ be the model's predicted posterior probability [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 4, pp. 111–112 (PDF pp. 131–132)):
> - **Cost-Sensitive Loss:** For an instance $\boldsymbol{x}$ of true class $i$, the expected misclassification cost is
>
> $$
> \mathcal{L}_{\mathrm{CS}}(\boldsymbol{x}, i; \theta) = \sum_{j=1}^{K} C_{ij}  P(j \mid \boldsymbol{x}; \theta).
> $$
>
> - **Class-Balanced Loss:** Assigning inverse-frequency class weights $W_i = N / N_i$ (where $N_i$ is the sample count of class $i$) yields $\mathcal{L}_{\mathrm{CB}}(\boldsymbol{x}, i; \theta) = W_i \sum_{j=1}^{K} P(j \mid \boldsymbol{x}; \theta) \mathrm{Loss}(\boldsymbol{x}, j)$. Accounting for sample overlap via the *effective number of samples* $E_{N_i} = (1 - \beta^{N_i})/(1 - \beta)$ with $\beta \in [0, 1)$ refines the weight to $W_i^{\mathrm{eff}} = (1 - \beta)/(1 - \beta^{N_i})$.
> - **Focal Loss:** To down-weight well-classified majority examples and concentrate gradients on hard examples, let $p_t \in (0, 1]$ be the model's predicted probability for the ground-truth class. With focusing parameter $\gamma \ge 0$ and balancing factor $\alpha_t$, Focal Loss is defined as
>
> $$
> \mathcal{L}_{\mathrm{FL}}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t).
> $$
>
>   For an easy example with $p_t = 0.9$ and $\gamma = 2$, the modulating factor $(1 - p_t)^2 = 0.01$ suppresses its loss contribution by $100\times$ relative to standard cross-entropy ($\gamma = 0$), preventing easy majority-class gradients from overwhelming the minority class.

> [!tip] Why Resampling Requires Post-Training Calibration
> Oversampling minority examples by factor $r > 1$ (or applying class weights $W_i$) shifts the effective training prior from $\pi_1$ to $\tilde{\pi}_1 = \frac{r \pi_1}{r \pi_1 + (1 - \pi_1)}$. Consequently, a classifier trained with cross-entropy outputs shifted posterior probabilities $\tilde{p}(y=1 \mid \boldsymbol{x})$ rather than true calibrated probabilities. Before thresholding for downstream business decisions, probabilities must be recalibrated via Bayes' rule odds correction or Platt/isotonic scaling [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 4, p. 110; Ch. 6, p. 184 (PDF pp. 130, 204)).

### Split-Invariant Feature Transformations and Causal Leakage

Feature scaling normalizes heterogeneous feature magnitudes (e.g., age $\in [20, 40]$ vs. income $\in [10^4, 1.5\times 10^5]$) via min-max rescaling to $[a, b]$ or $z$-score standardization, while $\log(1+x)$ transforms compress heavy-tailed skew [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 5, pp. 126–127 (PDF pp. 146–147)). Crucially, every transformation operator $\mathcal{T}_{\psi}$ parameterized by dataset statistics $\psi$ (mean $\mu$, standard deviation $\sigma$, quantiles, imputation medians) must satisfy **split invariance**:

$$
\psi^* = \mathcal{A}_{\mathrm{fit}}(\mathcal{D}_{\mathrm{train}}), \qquad
  \tilde{\boldsymbol{x}} = \mathcal{T}_{\psi^*}(\boldsymbol{x}) \quad \forall  \boldsymbol{x} \in \mathcal{D}_{\mathrm{train}} \cup \mathcal{D}_{\mathrm{val}} \cup \mathcal{D}_{\mathrm{test}}.
$$

Computing $\psi^*$ over $\mathcal{D}_{\mathrm{train}} \cup \mathcal{D}_{\mathrm{test}}$ prior to splitting leaks test-distribution moments into training representations. For time-indexed data $\{(\boldsymbol{x}_t, y_t)\}_{t=1}^T$, random splitting violates temporal causality; splits must obey $t_{\mathrm{train}} < t_{\mathrm{val}} < t_{\mathrm{test}}$ with point-in-time feature joins restricted to $\tau \le t$ [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 5, pp. 137–138 (PDF pp. 157–158)).

### Distribution Shift Taxonomy: Factorizing $P(X, Y)$

In supervised production ML, a model learns $P_{\mathrm{src}}(Y \mid X)$ from samples drawn from $P_{\mathrm{src}}(X, Y)$. Distribution shift occurs when $P_{\mathrm{tgt}}(X, Y) \ne P_{\mathrm{src}}(X, Y)$. Factorizing the joint distribution in two equivalent ways yields three canonical shift regimes [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 8, pp. 237–240 (PDF pp. 257–260)):

> [!important] Canonical Factorization of Joint Distribution Shift
> Using $P(X, Y) = P(Y \mid X) P(X) = P(X \mid Y) P(Y)$, distribution shifts decompose as:
> - **Covariate Shift ($P(X)$ shifts, $P(Y \mid X)$ invariant):**
>
> $$
> P_{\mathrm{tgt}}(X) \ne P_{\mathrm{src}}(X), \qquad P_{\mathrm{tgt}}(Y \mid X) = P_{\mathrm{src}}(Y \mid X).
> $$
>
>   The input feature density changes (e.g., younger patient demographics at inference than in training), while the true conditional risk given features is unchanged. Under shared support ($\mathrm{supp}(P_{\mathrm{tgt}}) \subseteq \mathrm{supp}(P_{\mathrm{src}})$), unbiased risk estimation under $P_{\mathrm{tgt}}$ is recovered via *importance weighting*:
>
> $$
> \mathbb{E}_{(\boldsymbol{x}, y) \sim P_{\mathrm{tgt}}}\!\bigl[\ell(\boldsymbol{x}, y; \theta)\bigr] = \mathbb{E}_{(\boldsymbol{x}, y) \sim P_{\mathrm{src}}}\!\left[\frac{P_{\mathrm{tgt}}(\boldsymbol{x})}{P_{\mathrm{src}}(\boldsymbol{x})} \ell(\boldsymbol{x}, y; \theta)\right].
> $$
>
> - **Label Shift ($P(Y)$ shifts, $P(X \mid Y)$ invariant):**
>
> $$
> P_{\mathrm{tgt}}(Y) \ne P_{\mathrm{src}}(Y), \qquad P_{\mathrm{tgt}}(X \mid Y) = P_{\mathrm{src}}(X \mid Y).
> $$
>
>   Typical when $Y \to X$ is causal (e.g., disease prevalence surges during an outbreak while symptom profiles given disease remain constant). Black Box Shift Estimation (BBSE) estimates target class priors $\boldsymbol{\pi}_{\mathrm{tgt}}$ without target labels by inverting the source confusion matrix $\boldsymbol{C}_{\mathrm{src}}$ against predicted target label marginals $\hat{\boldsymbol{\mu}}_{\mathrm{tgt}} = \boldsymbol{C}_{\mathrm{src}} \boldsymbol{w}$ where $w_y = P_{\mathrm{tgt}}(y)/P_{\mathrm{src}}(y)$.
> - **Concept Drift ($P(Y \mid X)$ shifts, $P(X)$ invariant):**
>
> $$
> P_{\mathrm{tgt}}(Y \mid X) \ne P_{\mathrm{src}}(Y \mid X), \qquad P_{\mathrm{tgt}}(X) = P_{\mathrm{src}}(X).
> $$
>
>   The underlying input-to-output mapping changes (e.g., ride-share demand or housing prices for identical features before vs. during a macroeconomic shock). Because $P(X)$ is unchanged, unsupervised input monitoring cannot detect concept drift; delayed ground-truth labels $Y$ or proxy feedback signals are required.

### Statistical Two-Sample Tests and Drift Metrics

When ground-truth labels $Y$ are unavailable or delayed in production, monitoring pipelines track divergence between a reference window $P$ (source) and a sliding production window $Q$ (target) [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 8, pp. 242–247 (PDF pp. 262–267)).

> [!important] KS Test, Population Stability Index (PSI), and Maximum Mean Discrepancy
> - **Kolmogorov–Smirnov (KS) Two-Sample Test (1D):** Given empirical cumulative distribution functions $F_{1,n}(x)$ (source, size $n$) and $F_{2,m}(x)$ (target, size $m$), the nonparametric KS statistic is
>
> $$
> D_{n,m} = \sup_{x \in \mathbb{R}} \lvert F_{1,n}(x) - F_{2,m}(x) \rvert.
> $$
>
>   The null hypothesis $H_0: P = Q$ is rejected at significance level $\alpha$ if $D_{n,m} > c(\alpha)\sqrt{\frac{n+m}{nm}}$, where $c(\alpha) = \sqrt{-\frac{1}{2}\ln(\alpha/2)}$. KS is restricted to 1D scalar features or predictions and can trigger excessive false alarms at large $n, m$ [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 8, p. 244 (PDF p. 264)).
> - **Population Stability Index (PSI):** Partitioning a scalar feature or model score into $B$ bins with reference proportions $p_b$ and production proportions $q_b$ ($\sum_b p_b = \sum_b q_b = 1$), PSI equals the symmetrized Kullback–Leibler (Jeffreys) divergence:
>
> $$
> \mathrm{PSI}(P, Q) = \sum_{b=1}^{B} (q_b - p_b)\ln\!\left(\frac{q_b}{p_b}\right) = D_{\mathrm{KL}}(Q \parallel P) + D_{\mathrm{KL}}(P \parallel Q).
> $$
>
>   Standard production thresholds classify $\mathrm{PSI} < 0.10$ as stable, $0.10 \le \mathrm{PSI} < 0.25$ as moderate shift (warning), and $\mathrm{PSI} \ge 0.25$ as significant shift (retrain trigger).
> - **Maximum Mean Discrepancy (MMD, Multivariate):** For high-dimensional features $\boldsymbol{x} \in \mathbb{R}^d$, let $\mathcal{H}$ be a Reproducing Kernel Hilbert Space (RKHS) with kernel $k(\boldsymbol{x}, \boldsymbol{x}') = \langle \phi(\boldsymbol{x}), \phi(\boldsymbol{x}') \rangle_{\mathcal{H}}$. The squared MMD is
>
> $$
> \begin{aligned}
>     \mathrm{MMD}^2(\mathcal{H}, P, Q)
>     &= \lVert \mathbb{E}_{X \sim P}[\phi(X)] - \mathbb{E}_{Y \sim Q}[\phi(Y)] \rVert_{\mathcal{H}}^2 \\
>     &= \mathbb{E}[k(X, X')] + \mathbb{E}[k(Y, Y')] - 2 \mathbb{E}[k(X, Y)].
>   \end{aligned}
> $$
>
>   For characteristic kernels (e.g., Gaussian RBF), $\mathrm{MMD}(\mathcal{H}, P, Q) = 0 \iff P = Q$. In practice, dimensionality reduction (PCA or untuned autoencoder projections) precedes two-sample testing to preserve statistical power [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 8, p. 244 (PDF p. 264)).

## Algorithmic Implementation & Comparative Synthesis

Production continual learning bridges feature engineering, drift detection, and champion–challenger deployment. Algorithm `alg:mlsys:drift_retrain` formalizes a unified point-in-time feature join, sliding-window drift detector, and stateful/stateless retraining loop with champion–challenger gating [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 8–9, pp. 246, 264–266, 277–278 (PDF pp. 266, 284–286, 297–298)).

> [!example] Algorithm: Point-in-Time Feature Join, Sliding-Window Drift Detection & Continual Retraining
> **Input:** Event stream $\mathcal{E}$, Feature Store $\mathcal{S}_F$, Champion model $\theta_{\mathrm{champ}}$, reference window $\mathcal{W}_{\mathrm{ref}}$, thresholds $(\tau_{\mathrm{warn}}, \tau_{\mathrm{crit}})$, mode $\in \{\mathrm{stateful}, \mathrm{stateless}\}$
> **Output:** Updated production model $\theta_{\mathrm{prod}}$ and logged lineage artifact
> **for** each serving epoch $t = 1, 2, \dots$:
>
>   Construct point-in-time feature vector $\boldsymbol{x}_t \leftarrow \mathrm{AsOfJoin}(\mathcal{S}_F, \text{entity}_t, \tau \le t)$  *// Zero future leakage*
>   Emit prediction $\hat{y}_t \leftarrow f_{\theta_{\mathrm{champ}}}(\boldsymbol{x}_t)$ and log $(\boldsymbol{x}_t, \hat{y}_t, t)$ to monitoring store
>   Aggregate sliding window $\mathcal{W}_t \leftarrow \{\boldsymbol{x}_s : t - \Delta \le s \le t\}$
>   Compute univariate $\mathrm{PSI}_j(\mathcal{W}_{\mathrm{ref}}, \mathcal{W}_t)$ via `(eq:mlsys:psi)` and multivariate $\widehat{\mathrm{MMD}}^2(\mathcal{W}_{\mathrm{ref}}, \mathcal{W}_t)$ via `(eq:mlsys:mmd)`
>   **if** $\max_j \mathrm{PSI}_j \ge \tau_{\mathrm{crit}}$ **or** $\widehat{\mathrm{MMD}}^2$ $p$-value $< \alpha$:
>
>     Trigger drift alert and assemble labeled window $\mathcal{D}_{\mathrm{new}}$ via delayed natural labels (Log-and-Wait)
>     **if** $\mathrm{mode} = \mathrm{stateful}$:
>
>       $\theta_{\mathrm{chal}} \leftarrow \mathrm{FineTune}(\theta_{\mathrm{champ}}, \mathcal{D}_{\mathrm{new}})$  *// Incremental checkpoint update*
>
> **else**:
>
>       $\theta_{\mathrm{chal}} \leftarrow \mathrm{TrainFromScratch}(\mathcal{D}_{t-L:t})$  *// Stateless full-window retrain*
>
>     Evaluate $\theta_{\mathrm{chal}}$ vs. $\theta_{\mathrm{champ}}$ on holdout temporal split $\mathcal{D}_{\mathrm{eval}}$
>     **if** $\mathrm{Metric}(\theta_{\mathrm{chal}}, \mathcal{D}_{\mathrm{eval}}) > \mathrm{Metric}(\theta_{\mathrm{champ}}, \mathcal{D}_{\mathrm{eval}}) + \delta_{\min}$:
>
>       Promote $\theta_{\mathrm{champ}} \leftarrow \theta_{\mathrm{chal}}$ in Model Registry with ZenML lineage $(\mathcal{D}_{\mathrm{new}}, \theta_{\mathrm{champ}})$
>       Update reference baseline $\mathcal{W}_{\mathrm{ref}} \leftarrow \mathcal{W}_t$

### Serving Paradigms and Model Compression Trade-Offs

Choosing between batch prediction, online prediction with batch features, and streaming prediction (online + streaming features) governs both feature-store architecture and tail latency [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 7, pp. 197–205 (PDF pp. 217–225)). When synchronous inference exceeds latency SLAs (e.g., $< 20\text{ ms}$ p50), four model compression techniques trade minimal accuracy for substantial throughput gains [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 7, pp. 206–211 (PDF pp. 226–231)): low-rank factorization (replacing $m \times n$ convolutions with depthwise-separable $m \times k$ and $k \times n$ factors), knowledge distillation (training a compact student to match a teacher's soft logits), pruning (zeroing low-magnitude weights or structured channels), and quantization (FP32 $\to$ FP16/BF16/INT8 fixed-point arithmetic). Table `tab:mlsys:serving_compression` summarizes these operational regimes.

**Table:** Serving Architectures and Model Compression Trade-Offs [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 7, pp. 197–211)

| **Paradigm / Method** | **Feature Access** | **Latency / Compute Profile** | **Primary Failure Mode / Trade-Off** |
| --- | --- | --- | --- |
| Batch Prediction | Batch warehouse only | High throughput; minutes–hours lag | Stale predictions under rapid user-state changes |
| Online (Batch Features) | Precomputed KV store | Low latency (ms); CPU/GPU serving | Cannot react to sub-minute session context |
| Streaming Prediction | Batch KV + Flink/stream | Sub-100 ms; unified stream engine | Train–serving skew if batch/stream code diverges |
| Knowledge Distillation | N/A (architecture) | $2\times$–$4\times$ faster student inference | Student capacity ceiling on complex tasks |
| INT8 Quantization | N/A (precision) | Up to $7\times$ latency cut, $8\times$ throughput | Rounding underflow/overflow without QAT |

> [!info] Cross-Book Synthesis: Feature Stores, Lineage, and Continual Retraining
> Synthesizing [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 9–10, pp. 265–278, 321–325) and [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 1–2, pp. 13–15, 39–48), production maturity progresses through four stages: manual updates (Stage 1), scheduled stateless retraining (Stage 2), automated stateful incremental training (Stage 3, yielding up to $45\times$ compute savings in Grubhub's production deployment), and drift-triggered continual learning (Stage 4). While [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 9, p. 277) highlights the *Log-and-Wait* pattern—reusing online-extracted features during retraining to eliminate train–serving skew—[[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 2, pp. 39–45) operationalizes this contract via ZenML step artifacts and metadata tracking, pairing MongoDB (raw unstructured documents) with Qdrant (chunked vector embeddings) as a logical feature store for LLM and RAG pipelines.

## Visual Diagnostics & Geometric Structure

Figure `fig:mlsys:fti_and_drift_loop` illustrates the end-to-end decoupled FTI architecture integrated with a point-in-time feature store and a closed-loop drift-monitoring controller that routes Covariate, Label, and Concept shifts to stateful or stateless retraining.

> [!example] Visual Diagram (LaTeX / PDF)
> *Decoupled Feature–Training–Inference (FTI) architecture with Point-in-Time Feature Store and closed-loop drift monitoring. Covariate ($P(X)$), Label ($P(Y)$), and Concept ($P(Y|X)$) shifts detected on sliding windows trigger champion–challenger retraining.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:mlsys:fti_book` shows the canonical FTI separation from [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 1, p. 13 (PDF p. 42)), while Figure `fig:mlsys:compression_book` and Figure `fig:mlsys:stateful_book` present empirical model compression speedups and stateful vs. stateless retraining windows from [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 7, p. 211; Ch. 9, p. 266).

> [!note] Source Figure
> *Figure:* Decoupled Feature–Training–Inference (FTI) pipeline architecture communicating exclusively through a versioned Feature Store and Model Registry [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 1, p. 13 (PDF p. 42)). (`llm_engineer_s_handbook_p42_fig1.png`)

> [!note] Source Figure
> *Figure:* Empirical latency (p50, ms) and throughput (inferences/s on 32 CPU cores) across DistilBERT distillation, dynamic shape inputs, and INT8 quantization [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 7, p. 211 (PDF p. 231)). (`chip_huyen_designing_machine_learning_systems_p231_fig1.png`)

> [!note] Source Figure
> *Figure:* Comparison of stateless retraining over sliding data windows versus stateful incremental training from prior checkpoints [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 9, p. 266 (PDF p. 286)). (`chip_huyen_designing_machine_learning_systems_p286_fig1.png`)

To visualize drift-detection dynamics over time, Figure `fig:mlsys:drift_psi_monitoring` plots a 30-day production trajectory of the Population Stability Index (PSI) and two-sample KS statistic across stability, warning ($\mathrm{PSI}=0.10$), and critical ($\mathrm{PSI}=0.25$) regimes, demonstrating post-retrain baseline recovery after Day 19.

> [!example] Visual Diagram (LaTeX / PDF)
> *Sliding-window Population Stability Index (PSI) and Kolmogorov–Smirnov ($D_{n,m}$) trajectory over 30 production days. Crossing $\mathrm{PSI} = 0.25$ on Day 17 triggers challenger evaluation and stateful retraining on Day 19, resetting the reference distribution.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Temporal, Group, and Preprocessing Data Leakage
> Randomly splitting time-series or user-session datasets leaks future market or trend regimes into training, inflating offline validation metrics while collapsing in production [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 5, pp. 137–140 (PDF pp. 157–160)). Similarly, deduplication must precede splitting (barring near-duplicate leakage such as the 3.3\%–10\% train–test overlap discovered in CIFAR-10/100), and grouped entities (e.g., multiple medical scans from one patient) must be assigned atomically to a single split.

> [!warning] Training–Serving Skew and Missing-Value Sentinel Collisions
> Maintaining separate batch (SQL/Spark) and streaming (Python/C++) feature pipelines induces silent divergence in aggregation windows or edge-case handling [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 5, p. 126; Ch. 7, p. 204). Furthermore, imputing missing numerical features with valid domain values (e.g., setting missing `age` or `num_children` to $0$) creates an unidentifiable collision between true zeros and missingness; always pair imputation with an explicit binary missingness indicator $m_j = \mathbb{I}[x_j\text{ is NaN}]$.

> [!warning] Cumulative Monitoring Blindness and Catastrophic Forgetting
> Evaluating accuracy cumulatively since deployment masks sudden transient outages (e.g., a 2-hour accuracy collapse diluted by weeks of healthy logs); always monitor *sliding* windows alongside multi-scale seasonality decomposition [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 8, pp. 246–247 (PDF pp. 266–267)). Conversely, updating neural networks sample-by-sample in production induces catastrophic forgetting and underutilizes parallel hardware; update in micro-batches ($512$–$1024$ samples) gated by champion–challenger evaluation [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 9, p. 264 (PDF p. 284)).

> [!warning] Quantization Rounding Underflow and High-Dimensional SMOTE
> Post-training INT8 quantization narrows representable dynamic range; outliers outside clipping bounds saturate or underflow to zero, causing severe accuracy degradation unless mitigated via quantization-aware training (QAT) or activation-range calibration [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 7, p. 210 (PDF p. 230)). Likewise, applying $k$-NN-based SMOTE or Tomek links directly in high-dimensional embedding spaces fails due to distance concentration; prefer algorithm-level focal or class-balanced losses.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[Distribution Shift & Drift]], [[FTI Architecture & Leakage Prevention]]
- **Prerequisites (Directed $\leftarrow$):** None
- **Enables / Unlocks (Directed $\rightarrow$):** None
- **Related Concepts (Undirected $\leftrightarrow$):** [[AgentOps & pass^k Reliability]], [[Distribution Shift & Drift]], [[FTI Architecture & Leakage Prevention]]
- **Synthesized Sources:** [[Source - Designing Machine Learning Systems]], [[Source - LLM Engineer’s Handbook]]
