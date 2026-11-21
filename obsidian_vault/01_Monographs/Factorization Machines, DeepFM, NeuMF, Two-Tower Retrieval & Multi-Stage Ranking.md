---
type: monograph
domain: "D07"
parent_topic: "[[PT 07 - Recommender Systems, Collaborative Filtering & Neural Ranking]]"
sub_topic: "[[ST 07 - Deep CTR Models, Two-Tower Retrieval & Production Ranking Pipelines]]"
cheatsheet: "[[Cheatsheet - Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]"
part_of: 
  - "[[ST 07 - Deep CTR Models, Two-Tower Retrieval & Production Ranking Pipelines]]"
  - "[[D07 - Agentic AI Architectures, Protocols & Production ML Systems]]"
concepts: 
  - "[[Factorization Machines & DeepFM]]"
  - "[[NDCG, MAP & Calibration]]"
  - "[[Two-Tower Retrieval & Multi-Stage Ranking]]"
prerequisites: 
  - "[[Factorization Machines & DeepFM]]"
  - "[[HNSW & IVF-PQ Indexing]]"
  - "[[MLP & Universal Approximation]]"
  - "[[Matrix Factorization (ALS & SGD)]]"
enables: 
  - "[[Two-Tower Retrieval & Multi-Stage Ranking]]"
related: 
  - "[[Bayesian Personalized Ranking (BPR)]]"
  - "[[Factorization Machines & DeepFM]]"
  - "[[NDCG, MAP & Calibration]]"
  - "[[Sparse vs Dense Retrieval & RRF]]"
  - "[[Two-Tower Retrieval & Multi-Stage Ranking]]"
sources: 
  - "[[Source - Designing Machine Learning Systems]]"
  - "[[Source - Dive into Deep Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - monograph
  - domain-07
---

# Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking

> [!abstract] Monograph Navigation
> - **Domain:** [[D07 - Agentic AI Architectures, Protocols & Production ML Systems|Agentic AI Architectures, Protocols & Production ML Systems]]
> - **Parent / Sub-Topic:** [[PT 07 - Recommender Systems, Collaborative Filtering & Neural Ranking|Recommender Systems, Collaborative Filtering & Neural Ranking]] → [[ST 07 - Deep CTR Models, Two-Tower Retrieval & Production Ranking Pipelines|Deep CTR Models, Two-Tower Retrieval & Production Ranking Pipelines]]
> - **Visual Cheatsheet:** [[Cheatsheet - Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]
> - **Description:** Linear-time O(kd) Factorization Machines, Wide & Deep / DeepFM feature interactions, Neural Collaborative Filtering (NeuMF), sequential recommendation, Two-Tower ANN candidate generation, multi-stage funnel (Retrieval -> Scoring -> Reranking), cold-start, position bias, and offline/online metrics (NDCG@K, MAP, Recall@K).

## Problem Formulation & Unified Notation

Industrial recommendation and advertising systems must score $N \sim 10^6\text{--}10^8$ candidate items for a user context $\boldsymbol{x} \in \mathbb{R}^d$ within a strict $\sim 50\text{--}100 \mathrm{ms}$ latency SLA, incorporating rich sparse categorical features (user history, device, item metadata, time) [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 21, pp. 933–936), [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 6, pp. 152–174). Modern architectures solve this via two-stage or multi-stage funnels: fast *Two-Tower ANN retrieval* followed by expressive *feature-interaction CTR/CVR rankers* (FM, DeepFM, NeuMF, DCN) and diversity/business-rule *re-ranking* [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 22, pp. 775–786), [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 6, pp. 170–175).

> [!info] Cross-Book Notation: Feature-Rich CTR & Multi-Stage Ranking
>
> | **Concept** | **D2L / Murphy** | **Chip Huyen (ML Systems)** | **Unified Monograph** |
> | --- | --- | --- | --- |
> | Sparse Feature Vector | $\boldsymbol{x} \in \mathbb{R}^d$ (one-hot/multi-hot) | User/Item/Context features | $\boldsymbol{x} \in \mathbb{R}^d$, latent $\boldsymbol{v}_i \in \mathbb{R}^k$ |
> | CTR Prediction | $\hat{y} = \sigma(\text{logit}(\boldsymbol{x}))$ | Multi-objective $P(\text{click}), P(\text{buy})$ | $\hat{y}(\boldsymbol{x}) = \sigma(\phi(\boldsymbol{x}))$ |
> | Funnel Stages | Retrieval $\to$ Ranking | Candidate gen $\to$ Scoring $\to$ Rerank | $10^7 \xrightarrow{\text{ANN}} 10^3 \xrightarrow{\text{DeepFM}} 10^1$ |
> | Ranking Metric | Recall@$K$, NDCG@$K$ | Business KPI + engagement | $\mathrm{NDCG}@K = \frac{\mathrm{DCG}@K}{\mathrm{IDCG}@K}$ |

## Core Mathematical Derivations

### Factorization Machines ($O(kd)$) & DeepFM / NeuMF

> [!important] Factorization Machines (FM), Linear-Time Trick & DeepFM
> Naive degree-2 polynomial regression requires $O(d^2)$ parameters and fails on sparse feature pairs never co-occurring in training data. *Factorization Machines (FM)* factorize every pairwise interaction weight $w_{ij} \approx \langle \boldsymbol{v}_i, \boldsymbol{v}_j \rangle$ ($\boldsymbol{v}_i \in \mathbb{R}^k$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 22, pp. 775–777), [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 21, p. 935):
>
> $$
> \hat{y}_{\mathrm{FM}}(\boldsymbol{x}) = w_0 + \sum_{i=1}^d w_i x_i + \sum_{i=1}^d \sum_{j=i+1}^d \langle \boldsymbol{v}_i, \boldsymbol{v}_j \rangle  x_i x_j.
> $$
>
> Using the algebraic identity $\sum_{i<j} \langle \boldsymbol{v}_i, \boldsymbol{v}_j \ranglex_i x_j = \frac{1}{2}\sum_{f=1}^k \Bigl[\bigl(\sum_{i=1}^d v_{i,f} x_i\bigr)^2 - \sum_{i=1}^d v_{i,f}^2 x_i^2\Bigr]$, FM evaluates all $\binom{d}{2}$ interactions in exact **linear time** $O(k d)$ (or $O(k \cdot \mathrm{nnz}(\boldsymbol{x}))$ for sparse inputs) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 22, p. 776).
>
> *DeepFM* shares the same embedding table $\boldsymbol{V}$ between a low-order FM component and a deep MLP component: $\hat{y}_{\mathrm{DeepFM}} = \sigma\bigl(y_{\mathrm{FM}}(\boldsymbol{x}) + y_{\mathrm{DNN}}(\boldsymbol{e})\bigr)$, capturing both explicit degree-2 crosses and arbitrary high-order nonlinearities end-to-end without manual feature engineering [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 21, p. 935). Similarly, *NeuMF* fuses Generalized Matrix Factorization ($\boldsymbol{p}_u^{\mathrm{G}}\odot\boldsymbol{q}_i^{\mathrm{G}}$) with an MLP tower over concatenated embeddings [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 21, p. 935).

### Two-Tower Retrieval, Multi-Stage Ranking & Evaluation (NDCG@K)

> [!important] Two-Tower In-Batch Softmax & Discounted Cumulative Gain (NDCG@K)
> For $10^7$-scale retrieval, heavy cross-attention or DeepFM over $(u, i)$ pairs is intractable. *Two-Tower* models decouple user/context encoder $u(\boldsymbol{x}_u) \in \mathbb{R}^k$ from item encoder $v(\boldsymbol{x}_i) \in \mathbb{R}^k$, scoring $s(u,i) = \langle u(\boldsymbol{x}_u), v(\boldsymbol{x}_i) \rangle/\tau$ so item vectors can be indexed offline in HNSW/IVF-PQ for sub-linear $O(\log N)$ ANN retrieval [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 6, pp. 170–175). Training uses in-batch softmax with log-Q sampling bias correction:
>
> $$
> \mathcal{L}_{\mathrm{TT}} = -\log\frac{\exp(s(u, i^+) - \log Q(i^+))}{\exp(s(u, i^+) - \log Q(i^+)) + \sum_{j \in \mathcal{B}^-} \exp(s(u, j) - \log Q(j))},
> $$
>
> where $Q(j)$ is item $j$'s sampling probability in mini-batch $\mathcal{B}$. Ranking quality at top-$K$ is evaluated via *Normalized Discounted Cumulative Gain* [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 22, pp. 782–784):
>
> $$
> \mathrm{DCG}@K = \sum_{r=1}^K \frac{2^{\mathrm{rel}_r} - 1}{\log_2(r + 1)}, \qquad \mathrm{NDCG}@K = \frac{\mathrm{DCG}@K}{\mathrm{IDCG}@K} \in [0, 1].
> $$
>

> [!tip] Why Two-Stage Funnels Exist: Decoupling Recall from Interaction Precision
> A Two-Tower dot product cannot model arbitrary nonlinear crosses between a user's real-time query and an item's attributes, yet it supports $O(\log N)$ ANN indexing [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 6, pp. 170–174). Placing Two-Tower (recall-optimized) before DeepFM/DCN (precision/calibration-optimized) achieves both million-item coverage and rich feature interaction within milliseconds.

## Visual Diagnostics & Production Pitfalls

> [!example] Visual Diagram (LaTeX / PDF)
> *Multi-stage production recommendation funnel balancing latency, recall, and cross-feature interaction.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!warning] Position Bias, Feedback Loops & Retrieval-Ranking Train-Serve Skew
> Users disproportionately click top-ranked slots regardless of relevance (*position bias*); training CTR models without position propensity features or Inverse Propensity Scoring (IPS) amplifies rich-get-richer feedback loops and filter bubbles [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 4 & 8, pp. 111–113, 293). In Two-Tower in-batch negatives, omitting the $-\log Q(j)$ correction over-penalizes popular items! Finally, training the ranker only on items that passed the old ranker induces *sample selection bias*—mitigated by logging randomized exploration slates [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Ch. 6, pp. 172–174).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[Factorization Machines & DeepFM]], [[NDCG, MAP & Calibration]], [[Two-Tower Retrieval & Multi-Stage Ranking]]
- **Prerequisites (Directed $\leftarrow$):** [[Factorization Machines & DeepFM]], [[HNSW & IVF-PQ Indexing]], [[MLP & Universal Approximation]], [[Matrix Factorization (ALS & SGD)]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Two-Tower Retrieval & Multi-Stage Ranking]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Bayesian Personalized Ranking (BPR)]], [[Factorization Machines & DeepFM]], [[NDCG, MAP & Calibration]], [[Sparse vs Dense Retrieval & RRF]], [[Two-Tower Retrieval & Multi-Stage Ranking]]
- **Synthesized Sources:** [[Source - Designing Machine Learning Systems]], [[Source - Dive into Deep Learning]], [[Source - Probabilistic Machine Learning]]
