---
type: monograph
domain: "D07"
parent_topic: "[[PT 07 - Recommender Systems, Collaborative Filtering & Neural Ranking]]"
sub_topic: "[[ST 07 - Collaborative Filtering, Matrix Factorization & Implicit Feedback Ranking]]"
cheatsheet: "[[Cheatsheet - Collaborative Filtering, Matrix Factorization (ALS - SGD), AutoRec & Bayesian Personalized Ranking (BPR)]]"
part_of: 
  - "[[ST 07 - Collaborative Filtering, Matrix Factorization & Implicit Feedback Ranking]]"
  - "[[D07 - Agentic AI Architectures, Protocols & Production ML Systems]]"
concepts: 
  - "[[AutoRec]]"
  - "[[Bayesian Personalized Ranking (BPR)]]"
  - "[[Collaborative Filtering (User-Item)]]"
  - "[[Matrix Factorization (ALS & SGD)]]"
prerequisites: 
  - "[[Autoencoders]]"
  - "[[Bradley-Terry Reward Model]]"
  - "[[Low-Rank Approximation (Eckart-Young)]]"
  - "[[Matrix Factorization (ALS & SGD)]]"
  - "[[Singular Value Decomposition (SVD)]]"
enables: 
  - "[[AutoRec]]"
  - "[[Bayesian Personalized Ranking (BPR)]]"
  - "[[Factorization Machines & DeepFM]]"
related: 
  - "[[AutoRec]]"
  - "[[Bayesian Personalized Ranking (BPR)]]"
  - "[[Collaborative Filtering (User-Item)]]"
  - "[[Matrix Factorization (ALS & SGD)]]"
  - "[[NDCG, MAP & Calibration]]"
sources: 
  - "[[Source - Dive into Deep Learning]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - monograph
  - domain-07
---

# Collaborative Filtering, Matrix Factorization (ALS/SGD), AutoRec & Bayesian Personalized Ranking (BPR)

> [!abstract] Monograph Navigation
> - **Domain:** [[D07 - Agentic AI Architectures, Protocols & Production ML Systems|Agentic AI Architectures, Protocols & Production ML Systems]]
> - **Parent / Sub-Topic:** [[PT 07 - Recommender Systems, Collaborative Filtering & Neural Ranking|Recommender Systems, Collaborative Filtering & Neural Ranking]] → [[ST 07 - Collaborative Filtering, Matrix Factorization & Implicit Feedback Ranking|Collaborative Filtering, Matrix Factorization & Implicit Feedback Ranking]]
> - **Visual Cheatsheet:** [[Cheatsheet - Collaborative Filtering, Matrix Factorization (ALS - SGD), AutoRec & Bayesian Personalized Ranking (BPR)]]
> - **Description:** Explicit vs. implicit feedback, neighborhood user-user/item-item CF, biased SVD matrix completion via ALS and SGD, weighted implicit ALS, AutoRec, and pairwise Bayesian Personalized Ranking (BPR) with negative sampling.

## Problem Formulation & Unified Notation

Given $M$ users $\mathcal{U}$ and $N$ items $\mathcal{I}$, recommender systems observe a sparse interaction matrix $\boldsymbol{Y} \in \mathbb{R}^{M \times N}$ over observed pairs $\Omega = \{(u, i) : y_{ui} \text{ observed}\}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 22, pp. 771–773), [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 21, pp. 933–935). In *explicit feedback* ($y_{ui} \in \{1,\dots,5\}$), missing entries are Not Missing At Random (MNAR); in *implicit feedback* (clicks, watches, purchases), only positive interactions $\mathcal{I}_u^+$ are observed while unobserved pairs mix true negatives with unexposed positives [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 22, pp. 774–779), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 18, pp. 237–244).

> [!info] Cross-Book Notation: Collaborative Filtering & Matrix Completion
>
> | **Concept** | **Murphy (PML1 Ch. 22)** | **Tiep Vu (Ch. 18–19)** | **Unified Monograph** |
> | --- | --- | --- | --- |
> | Utility / Rating Matrix | $\boldsymbol{Y} \in \mathbb{R}^{N_u \times N_m}$ | $\boldsymbol{Y} \in \mathbb{R}^{N \times M}$ (items $\times$ users) | $\boldsymbol{Y} \in \mathbb{R}^{M \times N}$ (users $\times$ items) |
> | Latent Factors | $\boldsymbol{U} \in \mathbb{R}^{N_u \times K}$, $\boldsymbol{V} \in \mathbb{R}^{N_m \times K}$ | $\boldsymbol{X}$ (items), $\boldsymbol{W}$ (users) | $\boldsymbol{p}_u \in \mathbb{R}^K$ (user), $\boldsymbol{q}_i \in \mathbb{R}^K$ (item) |
> | Bias Terms | $\mu + b_u + b_i$ | User mean $\bar{y}_u$ normalization | $\hat{y}_{ui} = \mu + b_u + b_i + \boldsymbol{p}_u^{\top} \boldsymbol{q}_i$ |

## Core Mathematical Derivations

### Neighborhood Collaborative Filtering & Biased Matrix Factorization

> [!important] Item-Item Similarity, Biased Matrix Factorization & Alternating Least Squares
> Neighborhood collaborative filtering predicts missing ratings from cosine or Pearson similarity between mean-centered user or item vectors [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 18, pp. 238–249):
>
> $$
> \hat{y}_{ui} = \bar{y}_u + \frac{\sum_{j \in \mathcal{N}_u(i)} \mathrm{sim}(i, j) (y_{uj} - \bar{y}_u)}{\sum_{j \in \mathcal{N}_u(i)} |\mathrm{sim}(i, j)|}.
> $$
>
> To capture global low-rank preference structure across sparse entries $\Omega$, *Biased Matrix Factorization* models $\hat{y}_{ui} = \mu + b_u + b_i + \boldsymbol{p}_u^{\top} \boldsymbol{q}_i$ and minimizes regularized squared error over $\Omega$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 22, pp. 773–776), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 19, pp. 253–264):
>
> $$
> \mathcal{L}_{\mathrm{MF}} = \sum_{(u,i)\in\Omega} \bigl(y_{ui} - \mu - b_u - b_i - \boldsymbol{p}_u^{\top} \boldsymbol{q}_i\bigr)^2 + \lambda\Bigl(\sum_u (\lVert \boldsymbol{p}_u \rVert_2^2 + b_u^2) + \sum_i (\lVert \boldsymbol{q}_i \rVert_2^2 + b_i^2)\Bigr).
> $$
>
> Although jointly non-convex in $(\boldsymbol{P}, \boldsymbol{Q})$, fixing $\boldsymbol{Q}$ makes $\mathcal{L}_{\mathrm{MF}}$ strictly convex quadratic in each $\boldsymbol{p}_u$, yielding *Alternating Least Squares (ALS)* closed-form updates [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 22, pp. 775–776):
>
> $$
> \boldsymbol{p}_u \gets \Bigl(\sum_{i \in \Omega_u} \boldsymbol{q}_i \boldsymbol{q}_i^{\top} + \lambda \boldsymbol{I}_K\Bigr)^{-1} \sum_{i \in \Omega_u} (y_{ui} - \mu - b_u - b_i) \boldsymbol{q}_i.
> $$
>
> For *implicit feedback* with confidence weights $c_{ui} = 1 + \alpha r_{ui}$ and preference indicators $p_{ui} = \mathbb{I}(r_{ui} > 0)$, weighted ALS sums over all $N$ items using the Gram matrix trick $\boldsymbol{Q}^{\top}\boldsymbol{Q} + \sum_{i \in \mathcal{I}_u^+} (c_{ui}-1)\boldsymbol{q}_i\boldsymbol{q}_i^{\top}$ in $O(|\Omega| K^2 + M K^3)$ instead of $O(MNK^2)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 22, pp. 777–779).

### AutoRec & Bayesian Personalized Ranking (BPR)

> [!important] AutoRec Reconstruction & Pairwise BPR Objective
> *AutoRec* replaces linear dot products with a single-hidden-layer autoencoder over sparse item rating vectors $\boldsymbol{y}^{(i)} \in \mathbb{R}^M$, backpropagating masked reconstruction loss only on observed entries $\Omega$ [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 21, p. 935):
>
> $$
> \mathcal{L}_{\mathrm{AutoRec}} = \sum_{i=1}^N \lVert \bigl(\boldsymbol{y}^{(i)} - f(\boldsymbol{W}\sigma(\boldsymbol{V}\boldsymbol{y}^{(i)} + \boldsymbol{b}_1) + \boldsymbol{b}_2)\bigr) \odot \boldsymbol{m}^{(i)} \rVert_2^2 + \frac{\lambda}{2}(\lVert \boldsymbol{W} \rVert_F^2 + \lVert \boldsymbol{V} \rVert_F^2).
> $$
>
> For top-$K$ recommendation from implicit positives $\mathcal{I}_u^+$, pointwise regression treats missing items as absolute zeros, whereas *Bayesian Personalized Ranking (BPR)* maximizes the posterior probability of pairwise orderings $(i \succ_u j)$ for $i \in \mathcal{I}_u^+$, $j \notin \mathcal{I}_u^+$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 22, pp. 779–782):
>
> $$
> \mathcal{L}_{\mathrm{BPR}} = -\sum_{(u, i, j) \in \mathcal{D}_S} \log \sigma\bigl(\hat{y}_{ui} - \hat{y}_{uj}\bigr) + \lambda \lVert \Theta \rVert_2^2,
> $$
>
> directly optimizing a smooth surrogate of Area Under the ROC Curve (AUC).

> [!tip] Why Item-Item CF Beats User-User and Why BPR Matches Top-K Ranking
> In large catalogs, user tastes drift rapidly and $|U| \gg |I|$, whereas item-item co-occurrence similarities remain stable and precomputable [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 18, pp. 244–248). Moreover, BPR directly penalizes ranking inversions ($\hat{y}_{uj} > \hat{y}_{ui}$) rather than wasting capacity calibrating absolute score magnitudes on unobserved negatives [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 22, pp. 780–782).

## Algorithmic Pipeline & Pitfalls

> [!example] Algorithm: Stochastic Gradient BPR with Uniform/Popularity Negative Sampling
> **Input:** Implicit positives $\{\mathcal{I}_u^+\}_{u=1}^M$, latent dim $K$, learning rate $\eta$, regularization $\lambda$
> **Output:** Embeddings $\boldsymbol{P} \in \mathbb{R}^{M\times K}$, $\boldsymbol{Q} \in \mathbb{R}^{N\times K}$
> Initialize $\boldsymbol{P}, \boldsymbol{Q} \sim \mathcal{N}(0, 0.01^2)$
> **for** $t = 1, \dots, T$:
>
>   Sample user $u \sim \mathcal{U}(\mathcal{U})$, positive item $i \sim \mathcal{U}(\mathcal{I}_u^+)$, and negative item $j \notin \mathcal{I}_u^+$
>   Compute margin $\Delta = \boldsymbol{p}_u^{\top}(\boldsymbol{q}_i - \boldsymbol{q}_j) + b_i - b_j$ and weight $w = \sigma(-\Delta)$
>   $\boldsymbol{p}_u \gets \boldsymbol{p}_u + \eta\bigl(w(\boldsymbol{q}_i - \boldsymbol{q}_j) - \lambda \boldsymbol{p}_u\bigr)$
>   $\boldsymbol{q}_i \gets \boldsymbol{q}_i + \eta\bigl(w\boldsymbol{p}_u - \lambda \boldsymbol{q}_i\bigr)$,  $\boldsymbol{q}_j \gets \boldsymbol{q}_j + \eta\bigl(-w\boldsymbol{p}_u - \lambda \boldsymbol{q}_j\bigr)$
>
> **return** $\boldsymbol{P}, \boldsymbol{Q}$

> [!warning] MNAR Exposure Bias, Popularity Collapse & Cold-Start
> Observed explicit ratings suffer from *Missing Not At Random (MNAR)* selection bias: users rate items they expect to like or hate, inflating naive RMSE estimates [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 22, pp. 774–775). In BPR, uniform negative sampling mostly draws trivial tail items ($\sigma(-\Delta) \approx 0$), causing vanishing gradients; adaptive hard-negative sampling or popularity-weighted negatives is essential. Finally, pure collaborative filtering cannot score brand-new users or items with zero interactions (*cold-start*), motivating feature-rich hybrid models [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 18, pp. 248–250).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (4):** [[AutoRec]], [[Bayesian Personalized Ranking (BPR)]], [[Collaborative Filtering (User-Item)]], [[Matrix Factorization (ALS & SGD)]]
- **Prerequisites (Directed $\leftarrow$):** [[Autoencoders]], [[Bradley-Terry Reward Model]], [[Low-Rank Approximation (Eckart-Young)]], [[Matrix Factorization (ALS & SGD)]], [[Singular Value Decomposition (SVD)]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[AutoRec]], [[Bayesian Personalized Ranking (BPR)]], [[Factorization Machines & DeepFM]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[AutoRec]], [[Bayesian Personalized Ranking (BPR)]], [[Collaborative Filtering (User-Item)]], [[Matrix Factorization (ALS & SGD)]], [[NDCG, MAP & Calibration]]
- **Synthesized Sources:** [[Source - Dive into Deep Learning]], [[Source - Machine Learning co ban]], [[Source - Probabilistic Machine Learning]]
