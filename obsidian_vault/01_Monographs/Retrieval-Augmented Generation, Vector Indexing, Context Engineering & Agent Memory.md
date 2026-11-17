---
type: monograph
domain: "D07"
parent_topic: "[[PT 07 - Agentic Reasoning, Memory, RAG, MCP - A2A & Workflows]]"
sub_topic: "[[ST 07 - RAG, Context Engineering, Memory & Tool - Agent Protocols]]"
cheatsheet: "[[Cheatsheet - Retrieval-Augmented Generation, Vector Indexing, Context Engineering & Agent Memory]]"
part_of: 
  - "[[ST 07 - RAG, Context Engineering, Memory & Tool - Agent Protocols]]"
  - "[[D07 - Agentic AI Architectures, Protocols & Production ML Systems]]"
concepts: 
  - "[[Context Memory & Compaction]]"
  - "[[HNSW & IVF-PQ Indexing]]"
  - "[[Sparse vs Dense Retrieval & RRF]]"
prerequisites: 
  - "[[Sparse vs Dense Retrieval & RRF]]"
enables: 
  - "[[HNSW & IVF-PQ Indexing]]"
  - "[[Two-Tower Retrieval & Multi-Stage Ranking]]"
related: 
  - "[[CLIP & Vision-Language Alignment]]"
  - "[[Context Memory & Compaction]]"
  - "[[HNSW & IVF-PQ Indexing]]"
  - "[[KV Cache & PagedAttention]]"
  - "[[Sparse vs Dense Retrieval & RRF]]"
  - "[[Two-Tower Retrieval & Multi-Stage Ranking]]"
sources: 
  - "[[Source - Agents Book]]"
  - "[[Source - Engineering -]]"
  - "[[Source - LLM Engineer’s Handbook]]"
  - "[[Source - Speech and Language Processing]]"
tags:
  - monograph
  - domain-07
---

# Retrieval-Augmented Generation, Vector Indexing, Context Engineering & Agent Memory

> [!abstract] Monograph Navigation
> - **Domain:** [[D07 - Agentic AI Architectures, Protocols & Production ML Systems|Agentic AI Architectures, Protocols & Production ML Systems]]
> - **Parent / Sub-Topic:** [[PT 07 - Agentic Reasoning, Memory, RAG, MCP - A2A & Workflows|Agentic Reasoning, Memory, RAG, MCP/A2A & Workflows]] → [[ST 07 - RAG, Context Engineering, Memory & Tool - Agent Protocols|RAG, Context Engineering, Memory & Tool/Agent Protocols]]
> - **Visual Cheatsheet:** [[Cheatsheet - Retrieval-Augmented Generation, Vector Indexing, Context Engineering & Agent Memory]]
> - **Description:** Unifies sparse BM25 vs. dense bi-encoder retrieval, HNSW/IVF vector DB indexing, pre/post-retrieval reranking (query expansion, RRF, cross-encoders), dynamic Context Engineering assembly, short-term Session compaction, and long-term declarative vs. procedural Memory consolidation.

## Problem Formulation & Unified Notation

Large Language Models (LLMs) are inherently stateless functions whose reasoning at inference turn $t$ is strictly bounded by the tokens inside a finite context window of capacity $B_{\max}$ [[Source - Engineering -|context_engineering_sessions_memory]] (p. 7). To ground generation in proprietary, dynamic, or user-specific knowledge without retraining model weights, production agentic systems unify two complementary mechanisms: *Retrieval-Augmented Generation (RAG)* over an external document corpus $\mathcal{C} = \{d_1, \dots, d_N\}$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 11, PDF p. 275), and *Context Engineering* over stateful session histories $\mathcal{S}_t$ and consolidated long-term memories $\mathcal{M}_t$ [[Source - Engineering -|context_engineering_sessions_memory]] (pp. 7–10).

Formally, given a user query $q_t$ at turn $t$, a retrieval pipeline selects an ordered subset of passages $\mathcal{R}_{\mathrm{rag}}(q_t, \mathcal{C}) = (d_{\pi(1)}, \dots, d_{\pi(k)}) \subset \mathcal{C}$, while a memory manager retrieves relevant persisted facts or workflows $\mathcal{R}_{\mathrm{mem}}(q_t, \mathcal{M}_t)$ and a session compactor produces a bounded conversational trace $\mathcal{K}_{\mathrm{comp}}(\mathcal{S}_t)$. Context Engineering then solves a token-budgeted payload assembly problem:

$$
\begin{aligned}
  C_t &= \mathcal{A}\!\Big(I_{\mathrm{sys}},  \mathcal{T},  \mathcal{R}_{\mathrm{mem}}(q_t, \mathcal{M}_t),  \mathcal{K}_{\mathrm{comp}}(\mathcal{S}_t),  \mathcal{R}_{\mathrm{rag}}(q_t, \mathcal{C}),  q_t\Big), \\
  &\qquad \text{s.t.} \quad \lvert C_t \rvert_{\mathrm{tok}} \le B_{\max} - B_{\mathrm{out}},
\end{aligned}
$$

where $I_{\mathrm{sys}}$ denotes system instructions, $\mathcal{T}$ denotes tool schemas, and $B_{\mathrm{out}}$ reserves tokens for model generation [[Source - Engineering -|context_engineering_sessions_memory]] (pp. 8–9). The LLM subsequently generates output tokens $x_1, \dots, x_n$ autoregressively under the conditional distribution

$$
p(x_1, \dots, x_n \mid C_t) = \prod_{i=1}^{n} p(x_i \mid C_t, x_{<i}),
$$

which grounds factual claims in $\mathcal{R}_{\mathrm{rag}}(q_t, \mathcal{C})$ and personalizes behavior via $\mathcal{R}_{\mathrm{mem}}(q_t, \mathcal{M}_t)$ (see [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 11, PDF p. 276) for the baseline RAG formulation).

> [!note] Source Figure
> *Figure:* Unified information flow between the Session Store (raw turn events), Memory Manager (proactive/reactive read and asynchronous extraction/consolidation), External Knowledge Bases (RAG vector databases), and the Agent execution loop [[Source - Engineering -|context_engineering_sessions_memory]] (p. 30, Fig. 5). (`context_engineering_sessions_memory_p30_fig1.png`)

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> | Concept | Source-Specific Terminology | Unified Monograph Notation |
> | — | — | — |
> | Retrieval Unit | Document $d \in \mathcal{C}$ [[Source - Speech and Language Processing]] (Ch. 11, PDF p. 264); cleaned/chunked text + metadata [[Source - LLM Engineer’s Handbook]] (Ch. 4, PDF p. 148) | Indexed chunk $d \in \mathcal{C}$ with embedding $\boldsymbol{z}_d \in \mathbb{R}^m$ and metadata $\mu_d$. |
> | Short-Term State | "Session" (verbatim/compacted turn events) [[Source - Engineering -]] (pp. 12, 27); "short-term memory" in agent frameworks | Session history $\mathcal{S}_t$ with compacted representation $\mathcal{K}_{\mathrm{comp}}(\mathcal{S}_t)$. |
> | Long-Term Memory | Extracted, consolidated facts/playbooks across sessions [[Source - Engineering -]] (pp. 27, 35) | Persistent store $\mathcal{M}_t = \mathcal{M}_t^{\mathrm{decl}} \cup \mathcal{M}_t^{\mathrm{proc}}$. |
> | Vector Storage | Standalone ANN index (e.g., FAISS) [[Source - Speech and Language Processing]] (Ch. 11, PDF p. 275); production Vector DB + logical Feature Store (Qdrant + ZenML) [[Source - LLM Engineer’s Handbook]] (Ch. 4, PDF pp. 144, 157) | Managed Vector DB supporting CRUD, sharding, metadata filtering, and ANN graph/quantized indices. |

## Core Mathematical Derivations & Theorems

### Sparse Lexical Retrieval: TF-IDF and Okapi BM25

Classic lexical retrieval represents queries and documents as high-dimensional sparse vectors indexed by vocabulary terms $t \in \mathcal{V}$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 11, PDF pp. 263–264). Because raw term counts $\mathrm{count}(t, d)$ scale linearly and over-emphasize repetitive words, logarithmic term frequency and inverse document frequency ($\mathrm{idf}_t$) attenuate common terms across a corpus of $N = \lvert \mathcal{C} \rvert$ documents.

> [!important] Okapi BM25 Lexical Scoring
> Let $\mathrm{df}_t$ denote the number of documents in $\mathcal{C}$ containing term $t$, $\lvert d \rvert$ the token length of document $d$, and $\lvert d_{\mathrm{avg}} \rvert = \frac{1}{N}\sum_{d' \in \mathcal{C}} \lvert d' \rvert$ the mean corpus document length. Following [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 11, PDF pp. 265–268, Eqs. (11.4)–(11.5), (11.12)), logarithmic TF-IDF defines
>
> $$
> \mathrm{tf}_{t,d} = \begin{cases} 1 + \log_{10} \mathrm{count}(t,d) & \text{if } \mathrm{count}(t,d) > 0, \\ 0 & \text{otherwise,} \end{cases}
>   \qquad \mathrm{idf}_t = \log_{10}\!\left(\frac{N}{\mathrm{df}_t}\right).
> $$
>
> Okapi BM25 replaces linear cosine normalization with bounded term-frequency saturation ($k \in [1.2, 2.0]$) and length normalization ($b \in [0, 1]$, typically $b = 0.75$):
>
> $$
> \mathrm{score}_{\mathrm{BM25}}(q, d) = \sum_{t \in q} \underbrace{\log\!\left(\frac{N}{\mathrm{df}_t}\right)}_{\text{IDF weight}} \cdot \underbrace{\frac{\mathrm{tf}_{t,d} (k + 1)}{\mathrm{tf}_{t,d} + k\!\left(1 - b + b \frac{\lvert d \rvert}{\lvert d_{\mathrm{avg}} \rvert}\right)}}_{\text{length-normalized saturated TF}},
> $$
>
> or equivalently without the $(k+1)$ constant factor in the simplified form of [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 11, PDF p. 268, Eq. (11.12)). Setting $k = 0$ collapses BM25 to binary term presence weighted by IDF, whereas $b = 0$ disables document-length normalization.

An *inverted index* maps each vocabulary term $t$ to a postings list of $(d, \mathrm{tf}_{t,d})$ pairs, enabling exact sparse scoring in time proportional to the postings length of query terms rather than $O(N)$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 11, PDF p. 268).

### Dense Retrieval: Bi-Encoders, Cross-Encoders, and ColBERT

Sparse lexical matching fails when query and document use synonymous phrasing (vocabulary mismatch). Dense retrieval maps text into continuous $m$-dimensional semantic spaces via Transformer encoders [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF pp. 136–140).

> [!important] Bi-Encoder Inner Product vs. Cross-Encoder vs. ColBERT MaxSim
> Let $E_Q$ and $E_D$ denote query and document encoders producing $L_2$-normalized embeddings $\boldsymbol{z}_q = E_Q(q) \in \mathbb{R}^m$ and $\boldsymbol{z}_d = E_D(d) \in \mathbb{R}^m$ (e.g., from `[CLS]` or mean pooling) [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 11, PDF p. 273, Eq. (11.19)):
>
> $$
> s_{\mathrm{bi}}(q, d) = \langle \boldsymbol{z}_q, \boldsymbol{z}_d \rangle = \frac{E_Q(q)^{\top} E_D(d)}{\lVert E_Q(q) \rVert_2 \lVert E_D(d) \rVert_2}.
> $$
>
> Because $E_D(d)$ is independent of $q$, all $N$ document vectors are precomputed offline and indexed for sub-linear Approximate Nearest Neighbor (ANN) search. Conversely, a *cross-encoder* concatenates $q$ and $d$ into a single sequence $[q; `[SEP]`; d]$ and applies full self-attention across all $L_q + L_d$ tokens [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF p. 154):
>
> $$
> s_{\mathrm{cross}}(q, d) = \sigma\!\left(\boldsymbol{w}^{\top} \mathrm{BERT}\!\big([q; `[SEP]`; d]\big)_{`[CLS]`}\right).
> $$
>
> Bridging both regimes, *ColBERT* late interaction precomputes per-token normalized document embeddings $\boldsymbol{E}_{d} = (\boldsymbol{E}_{d,1}, \dots, \boldsymbol{E}_{d,m_d})$ and computes a sum of maximum token similarities (*MaxSim*) at query time [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 11, PDF p. 274, Eq. (11.20)):
>
> $$
> s_{\mathrm{ColBERT}}(q, d) = \sum_{i=1}^{N_q} \max_{j=1,\dots,m_d} \langle \boldsymbol{E}_{q,i}, \boldsymbol{E}_{d,j} \rangle.
> $$

> [!note] Source Figure
> *Figure:* Architectural contrast between a Bi-Encoder (independent encoding and cosine similarity in $[-1, 1]$ enabling offline vector indexing) and a Cross-Encoder (joint BERT attention and classification head outputting $[0, 1]$ relevance scores for top-$K$ reranking) [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF p. 154, Fig. 4.7). (`llm_engineer_s_handbook_p154_fig1.png`)

### Hybrid Fusion (RRF) and Ranked Retrieval Evaluation

Because BM25 excels at exact entity/code matching while dense bi-encoders capture semantic paraphrases, production RAG systems execute both retrievers in parallel and fuse their rankings [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF p. 152).

> [!important] Reciprocal Rank Fusion (RRF) & IR Evaluation Metrics
> Given a set of rankers $\mathcal{R} = \{\text{BM25}, \text{Dense}\}$, where $\mathrm{rank}_r(d) \in \{1, 2, \dots\}$ is the rank of chunk $d$ under ranker $r$, Reciprocal Rank Fusion aggregates rankings without requiring score-distribution calibration:
>
> $$
> \mathrm{RRF}(d; q) = \sum_{r \in \mathcal{R}} \frac{1}{\kappa + \mathrm{rank}_r(d)}, \qquad \kappa = 60.
> $$
>
> To evaluate ranked lists against ground-truth relevance judgments $\mathrm{rel}_i \in \{0, 1, \dots\}$ over a query set $Q$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 11, PDF pp. 269–272, Eqs. (11.14)–(11.17)), let $R_r(q)$ be the set of relevant documents for query $q$ up to cutoff $r$:
>
> $$
> \begin{aligned}
> \mathrm{Recall}@k &= \frac{\lvert \text{relevant docs in top } k \rvert}{\lvert \text{total relevant docs} \rvert}, &
>   \mathrm{MRR} &= \frac{1}{\lvert Q \rvert} \sum_{q \in Q} \frac{1}{\mathrm{rank}_{\mathrm{first}}(q)},  \\
>   \mathrm{MAP} &= \frac{1}{\lvert Q \rvert} \sum_{q \in Q} \frac{1}{\lvert R_r(q) \rvert} \sum_{d \in R_r(q)} \mathrm{Precision}_{\mathrm{rank}(d)}, &
>   \mathrm{nDCG}@k &= \frac{1}{\mathrm{IDCG}@k} \sum_{i=1}^{k} \frac{2^{\mathrm{rel}_i} - 1}{\log_2(i + 1)},
> \end{aligned}
> $$
>
> where $\mathrm{IDCG}@k$ is the ideal discounted cumulative gain of the oracle ranking.

### Approximate Nearest Neighbor Indexing: HNSW vs. IVF-PQ

Exact $k$-nearest neighbor search over $N$ vectors in $\mathbb{R}^m$ requires $O(Nm)$ distance evaluations, which becomes prohibitive at $N \ge 10^6$ [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF p. 144). Vector databases therefore rely on Approximate Nearest Neighbor (ANN) indices.

> [!tip] HNSW Multi-Layer Graph Navigation vs. IVF-PQ Quantization
> **Hierarchical Navigable Small World (HNSW)** constructs a hierarchy of proximity graphs $\{G_0, G_1, \dots, G_L\}$ where layer $0$ contains all $N$ vectors and each upper layer $l$ retains an exponentially decaying subset (probability $p \propto e^{-l / m_L}$) [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF p. 145). Each node maintains at most $M$ edges ($2M$ at layer $0$). A query enters at the top layer and greedily traverses long-range skip edges down to layer $0$, where a beam search of width $\text{efSearch} \ge k$ refines the candidate set in $O(\log N)$ expected hops.
>
> In contrast, **IVF-PQ** partitions $\mathbb{R}^m$ into $C_{\mathrm{list}}$ Voronoi cells via $k$-means clustering (Inverted File) and compresses residual vectors by splitting $\mathbb{R}^m$ into $S$ subspaces of dimension $m/S$, quantizing each sub-vector to a $b_{\mathrm{bits}}$-bit centroid code (Product Quantization) [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF p. 145). IVF-PQ reduces RAM footprint by $10\times$–$30\times$ at the cost of quantization distortion, whereas HNSW achieves higher recall@10 at the cost of storing both full vectors and graph adjacency lists in memory.

### Context Budget Optimization, Compaction, and Memory Consolidation

In multi-turn agentic workflows, unbounded session histories $\mathcal{S}_t$ exhaust $B_{\max}$ and dilute attention. [[Source - Engineering -|context_engineering_sessions_memory]] (pp. 23–26) formalize three session compaction triggers and a four-stage long-term memory lifecycle.

> [!important] Token-Budgeted Assembly, Compaction Triggers & Memory Consolidation
> Partition the context budget into $B_{\mathrm{sys}} + B_{\mathrm{mem}} + B_{\mathrm{sess}} + B_{\mathrm{rag}} + B_{\mathrm{out}} \le B_{\max}$. When raw session tokens $\lvert \mathcal{S}_t \rvert_{\mathrm{tok}}$ exceed a threshold, a compaction operator $\mathcal{K}_{\mathrm{comp}}$ fires under one of three triggers [[Source - Engineering -|context_engineering_sessions_memory]] (pp. 25–26):
> - **Count-based**: $\lvert \mathcal{S}_t \rvert_{\mathrm{tok}} > \tau_{\mathrm{tok}}$ or turn count $t \equiv 0 \pmod{\Delta_{\mathrm{interval}}}$ (retaining an overlap window of $o$ recent turns).
> - **Time-based**: Idle duration $\Delta t_{\mathrm{idle}} > \tau_{\mathrm{idle}}$ triggers asynchronous background summarization.
> - **Event-based**: Semantic detection of sub-task completion compresses completed tool trajectories into a summary event.
>
> Long-term memory generation operates asynchronously on $\mathcal{S}_t$ via extraction $\mathcal{E}_\theta(\mathcal{S}_t) \to \{m_{\mathrm{cand}}\}$ filtered by domain topic schemas, followed by a consolidation operator $\mathcal{U}_{\mathrm{cons}}$ against existing memories $\mathcal{M}_{t-1}$ [[Source - Engineering -|context_engineering_sessions_memory]] (pp. 42–47):
>
> $$
> \mathcal{M}_t = \mathcal{U}_{\mathrm{cons}}\!\big(\mathcal{M}_{t-1},  \mathcal{E}_\theta(\mathcal{S}_t)\big) \in \{\mathrm{Create}(m_{\mathrm{new}}),  \mathrm{Merge}(m_{\mathrm{old}}, m_{\mathrm{new}}),  \mathrm{Delete}(m_{\mathrm{invalid}})\}.
> $$

## Algorithmic Implementation & Comparative Synthesis

Production RAG pipelines decouple offline ingestion (ETL $\to$ cleaning $\to$ sliding-window or small-to-big chunking $\to$ bi-encoder embedding $\to$ vector DB + feature store sync via Change Data Capture) from online inference (pre-retrieval query optimization $\to$ hybrid retrieval $\to$ post-retrieval reranking and compression) [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF pp. 147–166).

> [!note] Source Figure
> *Figure:* The three optimization stages of Advanced RAG: Pre-retrieval (document cleaning/chunking and query routing/rewriting), Retrieval (bi-encoder embedding and vector DB similarity search), and Post-retrieval (cross-encoder reranking and prompt compression) [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF p. 147, Fig. 4.5). (`llm_engineer_s_handbook_p147_fig1.png`)

> [!example] Algorithm: Hybrid BM25+Dense RRF Retrieval with Session Compaction & Context Assembly
> **Input:** User query $q_t$, session $\mathcal{S}_{t-1}$, memory store $\mathcal{M}_{t-1}$, corpus $\mathcal{C}$ (BM25 + HNSW indices), budgets $(B_{\max}, B_{\mathrm{out}})$, RRF constant $\kappa=60$, retrieval depth $K$, rerank cutoff $k$
> **Output:** Assembled prompt $C_t$, updated session $\mathcal{S}_t$, async memory update job
>
>  *// 1. Pre-Retrieval: Query Rewriting & Agentic Routing*
> $q'_t \leftarrow \mathrm{RewriteExpand}(q_t, \mathcal{S}_{t-1})$  *// Resolve coreferences / HyDE if needed*
> $\mathrm{route} \leftarrow \mathrm{QueryRouter}(q'_t)$  *// Select VectorDB, Web Search, or No-RAG*
>
>  *// 2. Parallel Hybrid Retrieval & RRF Fusion*
> **if** $\mathrm{route} == \mathrm{VectorDB}$:
>
>   $L_{\mathrm{bm25}} \leftarrow \mathrm{InvertedIndexSearch}(q'_t, \mathcal{C}, K)$  *// Eq. `(eq:rag:bm25)`*
>   $L_{\mathrm{dense}} \leftarrow \mathrm{HNSWSearch}(E_Q(q'_t), \mathcal{C}, K, \mathrm{efSearch})$  *// Eq. `(eq:rag:biencoder)`*
>   Compute $\mathrm{RRF}(d; q'_t) = \sum_{r \in \{\mathrm{bm25}, \mathrm{dense}\}} (\kappa + \mathrm{rank}_r(d))^{-1}$ for $d \in L_{\mathrm{bm25}} \cup L_{\mathrm{dense}}$
>   $D_{\mathrm{cand}} \leftarrow \mathrm{TopK}(\mathrm{RRF}, 2k)$
>   $\mathcal{R}_{\mathrm{rag}} \leftarrow \mathrm{TopK}\!\big(\{s_{\mathrm{cross}}(q_t, d) : d \in D_{\mathrm{cand}}\},  k\big)$  *// Cross-encoder rerank, Eq. `(eq:rag:crossencoder)`*
>
> **else**:
>
>   $\mathcal{R}_{\mathrm{rag}} \leftarrow \mathrm{ExternalToolSearch}(q'_t)$  *// e.g., Firecrawl web search*
>
>  *// 3. Memory Retrieval & Session Compaction*
> $\mathcal{R}_{\mathrm{mem}} \leftarrow \mathrm{RetrieveMemory}(q'_t, \mathcal{M}_{t-1})$  *// Proactive callback or Memory-as-a-Tool*
> **if** $\lvert \mathcal{S}_{t-1} \rvert_{\mathrm{tok}} > \tau_{\mathrm{tok}}$:
>
>   $\mathcal{K}_{\mathrm{comp}} \leftarrow \mathrm{RecursiveSummarize}(\mathcal{S}_{t-1}[1 : t-o]) \cup \mathcal{S}_{t-1}[t-o+1 : t-1]$
>
> **else**:
>
>   $\mathcal{K}_{\mathrm{comp}} \leftarrow \mathcal{S}_{t-1}$
>
>  *// 4. Token-Budgeted Assembly & Async Memory Consolidation*
> Assemble $C_t \leftarrow \mathcal{A}(I_{\mathrm{sys}}, \mathcal{T}, \mathcal{R}_{\mathrm{mem}}, \mathcal{K}_{\mathrm{comp}}, \mathcal{R}_{\mathrm{rag}}, q_t)$ respecting $B_{\max} - B_{\mathrm{out}}$
> Generate response $y_t \sim p(\cdot \mid C_t)$ and append $(q_t, y_t)$ to $\mathcal{S}_t$
> Enqueue async job: $\mathcal{M}_t \leftarrow \mathcal{U}_{\mathrm{cons}}(\mathcal{M}_{t-1}, \mathcal{E}_\theta(\mathcal{S}_t))$
> **return** $C_t, \mathcal{S}_t$

**Table:** Computational complexity and systems trade-offs across retrieval and reranking stages for $N$ chunks of dimension $m$, average postings length $L_p$, HNSW degree $M$, and candidate set $K$.

| Method | Query Time | Memory Footprint | Recall@10 | Operational Role |
| — | — | — | — | — |
| BM25 (Inverted Index) | $O(\lvert q \rvert \cdot L_p)$ | Sparse postings | Moderate (exact) | Lexical anchor; entities, IDs, code |
| Flat Exact Search | $O(N \cdot m)$ | $O(N \cdot m)$ | $100\%$ (exact) | Small corpora ($N < 10^4$) or ground truth |
| IVF-PQ | $O(\frac{N}{C_{\mathrm{list}}} S + C_{\mathrm{list}} m)$ | $O(N \cdot S \cdot b_{\mathrm{bits}})$ | $80\%$–$92\%$ | Billion-scale RAM-constrained indices |
| HNSW Graph | $O(M \cdot \mathrm{efSearch} \cdot \log N)$ | $O(N(m + M))$ | $95\%$–$99\%$ | Default low-latency vector DB index |
| Cross-Encoder Rerank | $O(K \cdot (L_q + L_d)^2)$ | Model weights | Highest precision | Stage-2 rescoring of top-$K$ ($K \le 50$) |

> [!info] Cross-Book Synthesis: Declarative vs. Procedural Memory & Agentic Routing
> Synthesizing [[Source - Engineering -|context_engineering_sessions_memory]] (pp. 35–38, 58–65), [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF pp. 149–153), and [[Source - Agents Book|ai_agents_2025_illustrated_guidebook]] (pp. 36–41) reveals three orthogonal axes of agentic retrieval and memory design:
> - **Declarative vs. Procedural Memory**: Declarative memory stores "knowing what"—either *semantic* world/user facts (e.g., structured user profile) or *episodic* interaction logs (collections pattern). Procedural memory stores "knowing how"—reusable tool-call playbooks distilled from successful trajectories, providing fast online adaptation via in-context learning rather than slow offline RLHF/fine-tuning [[Source - Engineering -|context_engineering_sessions_memory]] (pp. 35, 64–65).
> - **Proactive vs. Reactive Retrieval**: Proactive retrieval (e.g., ADK `PreloadMemoryTool` callbacks) injects memories before every turn, guaranteeing availability at the cost of per-turn latency. Reactive retrieval (`Memory-as-a-Tool`) lets the LLM decide when to invoke `load\_memory(query)`, saving unnecessary lookups on chitchat turns at the expense of an extra tool-call round trip [[Source - Engineering -|context_engineering_sessions_memory]] (pp. 58–60).
> - **Agentic RAG Routing**: Rather than unconditionally querying a single vector store, an agentic router dispatches queries across heterogeneous sources—vector DBs for internal docs, SQL databases for structured records, or live web crawlers (e.g., Firecrawl) for real-time events [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF p. 149); [[Source - Agents Book|ai_agents_2025_illustrated_guidebook]] (pp. 36–38).

## Visual Diagnostics & Geometric Structure

Figure `fig:rag:rag_memory_pipeline` integrates the two-stage hybrid retrieval pipeline from [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF pp. 147–155) with the token-budgeted Context Engineering assembly manager from [[Source - Engineering -|context_engineering_sessions_memory]] (pp. 8–10, 25–30).

> [!example] Visual Diagram (LaTeX / PDF)
> *End-to-end Hybrid RAG and Context Engineering architecture: query rewriting/routing feeds parallel sparse BM25 and dense HNSW retrieval, fused via Reciprocal Rank Fusion (RRF) and cross-encoder reranking into a token-budgeted Context Assembly Manager.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

To quantify the indexing latency-accuracy trade-off discussed in [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF pp. 144–146), Figure `fig:rag:recall_latency_hnsw` illustrates the characteristic Pareto frontier of Recall@10 versus Queries Per Second (QPS) across exact flat search, quantized IVF-PQ, and graph-based HNSW as the beam width $\text{efSearch}$ increases.

> [!example] Visual Diagram (LaTeX / PDF)
> *Characteristic ANN indexing Pareto frontier ($N = 10^6$ vectors, $m = 768$): HNSW achieves $\ge 97\%$ Recall@10 at $>900$ QPS by navigating hierarchical small-world edges, whereas IVF-PQ trades $6$–$10\%$ recall for compact quantized storage and Exact Flat Scan saturates throughput at $\sim 15$ QPS.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Chunk Boundary Severance & Embedding Topic Dilution
> Fixed-size chunking without overlap severs sentences and cross-references at chunk boundaries, while overly large chunks aggregate multiple topics into a single blurred mean-pooled vector [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF pp. 148–149).
> **Remedy:** Apply sliding-window overlap ($10\%$–$20\%$ of chunk tokens) or *small-to-big* decoupling: index small, semantically focused child chunks ($128$–$256$ tokens) for sharp ANN matching, while storing the parent passage ($512$–$1024$ tokens) in metadata for LLM generation.

> [!warning] Pre-Filtering vs. Post-Filtering Collapse in HNSW
> In filtered vector search [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF pp. 145, 152), applying metadata predicates *after* HNSW search (post-filtering) yields fewer than $k$ results when predicate selectivity is low ($<5\%$). Conversely, naive *pre-filtering* can disconnect the HNSW graph, forcing fallback to $O(N)$ brute-force scanning.
> **Remedy:** Use inline predicate-aware graph traversal (ACORN / filtered-HNSW) or maintain partitioned sub-indices for high-cardinality tenant/scope keys (e.g., `user\_id` in memory stores).

> [!warning] Dialogue Injection & Stale Memory Lineage
> Injecting retrieved long-term memories directly into the `user` turn history risks *dialogue injection*, where the LLM hallucinates that it or the user explicitly uttered the memory in the current session [[Source - Engineering -|context_engineering_sessions_memory]] (p. 63). Furthermore, consolidating memories without tracking provenance/lineage propagates stale or contradictory facts across sessions [[Source - Engineering -|context_engineering_sessions_memory]] (pp. 49–52).
> **Remedy:** Inject declarative memories into structured system instructions (with explicit provenance timestamps and source IDs) rather than raw dialogue turns, and execute LLM-based consolidation (`Merge`/`Delete`) asynchronously against golden-set precision/recall evaluations [[Source - Engineering -|context_engineering_sessions_memory]] (pp. 61–65).

> [!warning] Context Window Saturation & Attention Dilution
> Stuffing excessive low-ranked RAG chunks or uncompacted tool outputs into $C_t$ increases TTFT (time-to-first-token) quadratically during prefill and degrades reasoning accuracy—a failure mode where relevant evidence buried in the middle of long contexts is ignored [[Source - Engineering -|context_engineering_sessions_memory]] (pp. 22–24); [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 4, PDF p. 153).
> **Remedy:** Enforce strict token budgets $B_{\mathrm{rag}}$ and $B_{\mathrm{sess}}$, apply cross-encoder reranking with a relevance threshold to prune noisy chunks, and trigger asynchronous recursive session compaction before $\lvert \mathcal{S}_t \rvert_{\mathrm{tok}}$ saturates $B_{\max}$.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[Context Memory & Compaction]], [[HNSW & IVF-PQ Indexing]], [[Sparse vs Dense Retrieval & RRF]]
- **Prerequisites (Directed $\leftarrow$):** [[Sparse vs Dense Retrieval & RRF]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[HNSW & IVF-PQ Indexing]], [[Two-Tower Retrieval & Multi-Stage Ranking]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[CLIP & Vision-Language Alignment]], [[Context Memory & Compaction]], [[HNSW & IVF-PQ Indexing]], [[KV Cache & PagedAttention]], [[Sparse vs Dense Retrieval & RRF]], [[Two-Tower Retrieval & Multi-Stage Ranking]]
- **Synthesized Sources:** [[Source - Agents Book]], [[Source - Engineering -]], [[Source - LLM Engineer’s Handbook]], [[Source - Speech and Language Processing]]
