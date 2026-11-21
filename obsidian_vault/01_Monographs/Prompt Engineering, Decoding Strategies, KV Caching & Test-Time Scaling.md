---
type: monograph
domain: "D05"
parent_topic: "[[PT 05 - LLM Pretraining, Adaptation, Prompting & Efficient Inference]]"
sub_topic: "[[ST 05 - Prompting, Decoding, KV Caching & Test-Time Compute]]"
cheatsheet: "[[Cheatsheet - Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]"
part_of: 
  - "[[ST 05 - Prompting, Decoding, KV Caching & Test-Time Compute]]"
  - "[[D05 - Transformers, Language Modeling & LLM Engineering]]"
concepts: 
  - "[[Chain-of-Thought & Test-Time Compute]]"
  - "[[KV Cache & PagedAttention]]"
  - "[[Speculative Decoding]]"
prerequisites: 
  - "[[GQA & MQA]]"
  - "[[KV Cache & PagedAttention]]"
  - "[[Self-Attention & Multi-Head Attention]]"
enables: 
  - "[[Speculative Decoding]]"
related: 
  - "[[Chain-of-Thought & Test-Time Compute]]"
  - "[[Context Memory & Compaction]]"
  - "[[FlashAttention]]"
  - "[[GQA & MQA]]"
  - "[[KV Cache & PagedAttention]]"
  - "[[MCTS & MuZero]]"
  - "[[RLVR (Verifiable Rewards)]]"
  - "[[ReAct & Tool-Use Loop]]"
  - "[[Speculative Decoding]]"
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Foundations of Large Language Models]]"
  - "[[Source - Reinforcement Learning from Human Feedback]]"
  - "[[Source - Speech and Language Processing]]"
tags:
  - monograph
  - domain-05
---

# Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling

> [!abstract] Monograph Navigation
> - **Domain:** [[D05 - Transformers, Language Modeling & LLM Engineering|Transformers, Language Modeling & LLM Engineering]]
> - **Parent / Sub-Topic:** [[PT 05 - LLM Pretraining, Adaptation, Prompting & Efficient Inference|LLM Pretraining, Adaptation, Prompting & Efficient Inference]] → [[ST 05 - Prompting, Decoding, KV Caching & Test-Time Compute|Prompting, Decoding, KV Caching & Test-Time Compute]]
> - **Visual Cheatsheet:** [[Cheatsheet - Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling]]
> - **Description:** Synthesizes in-context learning, Chain-of-Thought/Tree-of-Thought prompting, greedy/beam/temperature/top-k/top-p sampling, prefill vs. decode arithmetic intensity, KV-cache memory math & PagedAttention, speculative decoding, quantization, and inference-time search/scaling.

## Problem Formulation & Unified Notation

Let $\mathcal{V}$ denote a discrete token vocabulary of size $|\mathcal{V}|$. Given an input prompt $\boldsymbol{x} = (x_0, x_1, \dots, x_m) \in \mathcal{V}^{m+1}$ (with $x_0 = \langle\mathrm{SOS}\rangle$), an autoregressive decoder-only Transformer parameterized by $\theta$ defines a conditional probability distribution over an output sequence $\boldsymbol{y} = (y_1, \dots, y_n) \in \mathcal{V}^n$ via the probability chain rule [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, pp. 211–213):

$$
\log \Pr_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})
  = \sum_{i=1}^{n} \log \Pr_{\theta}(y_i \mid \boldsymbol{x}, \boldsymbol{y}_{<i}),
  \qquad \boldsymbol{y}_{<i} \triangleq (y_1, \dots, y_{i-1}).
$$

At decoding step $i$, let $i' = m + i$ denote the total context length of the concatenated sequence $[\boldsymbol{x}, \boldsymbol{y}_{<i}]$. The Transformer stack produces hidden representations $\boldsymbol{H} = \mathrm{Dec}([\boldsymbol{x}, \boldsymbol{y}_{<i}]) \in \mathbb{R}^{i' \times d}$, and the unembedding matrix $\boldsymbol{W}_o \in \mathbb{R}^{d \times |\mathcal{V}|}$ maps the terminal hidden vector $\boldsymbol{h}_{m+i} \in \mathbb{R}^d$ to unnormalized logits $\boldsymbol{u}_{i} = \boldsymbol{W}_o^{\top} \boldsymbol{h}_{m+i} \in \mathbb{R}^{|\mathcal{V}|}$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Sec. 7.6, p. 196). The next-token categorical distribution is

$$
\Pr_{\theta}(\cdot \mid \boldsymbol{x}, \boldsymbol{y}_{<i})
  = \mathrm{softmax}(\boldsymbol{u}_i)
  = \mathrm{softmax}\!\left((\boldsymbol{H}\boldsymbol{W}_o)_{m+i}\right).
$$

LLM inference decomposes into two coupled sub-problems: (i) *model computation*, evaluating Eq. `(eq:decode:softmax_logits)` under strict latency and HBM bandwidth constraints via Key-Value (KV) caching and parallel verification; and (ii) *search and conditioning*, shaping $\boldsymbol{x}$ (in-context learning, Chain-of-Thought) and traversing or scaling the output space $\mathcal{V}^n$ (sampling, beam search, test-time verification).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Source textbooks adopt distinct conventions for logits, temperature, sequence indexing, and verification models. Table `tab:decode:notation` reconciles these notations into a single unified formalism used throughout this monograph.
>
> | **Unified Symbol** | **Meaning in This Monograph** | **Source Textbook Equivalents** |
> | --- | --- | --- |
> | $\boldsymbol{x}, \boldsymbol{y}_{<i}, y_i$ | Prompt ($m{+}1$ tokens), prefix, next token | $\boldsymbol{x}, \boldsymbol{y}_{<i}, y_i$ [[Source - Foundations of Large Language Models]] (p. 211); $w_{<t}, w_t$ [[Source - Speech and Language Processing]] (p. 196); $s_i, a_i$ [[Source - Reinforcement Learning from Human Feedback]] (p. 103) |
> | $\boldsymbol{u}_i \in \mathbb{R}^{\mid \mathcal{V}\mid }$ | Unnormalized next-token logit vector | $(\boldsymbol{H}\boldsymbol{W}_o)_{m+i}$ [[Source - Foundations of Large Language Models]] (p. 212); $\boldsymbol{u}$ [[Source - Speech and Language Processing]] (p. 196); `logits` [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]] (p. 381) |
> | $\tau > 0$ | Softmax temperature scaling parameter | $\tau$ [[Source - Speech and Language Processing]] (p. 198); $T/\tau$ [[Source - Foundations of Large Language Models]] (p. 224); `temperature` [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]] (p. 384) |
> | $p(\cdot), q(\cdot)$ | Target (verifier) and draft distributions | $\Pr_p(\cdot), \Pr_q(\cdot)$ [[Source - Foundations of Large Language Models]] (p. 226) |
> | $b, s, n_{\mathrm{layers}}, n_{\mathrm{kv}}, d_h$ | Batch, seq len, layers, KV heads, head dim | $L$ layers, $d$ hidden dim [[Source - Foundations of Large Language Models]] (p. 216) |

## In-Context Learning, Prompting & Mechanistic Circuits

Unlike gradient-based fine-tuning, *in-context learning* (ICL) adapts model behavior during the forward pass at inference time with zero parameter updates ($\Delta\theta = \boldsymbol{0}$, see [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 3, p. 106)). Prompting spans a spectrum of conditioning regimes:
- **Zero-Shot, One-Shot & Few-Shot ICL:** In zero-shot prompting, the input $\boldsymbol{x}$ specifies task instructions and role constraints. One-shot and few-shot prompting prepend $k$ input–output demonstrations $\{(x^{(j)}, y^{(j)})\}_{j=1}^k$ to $\boldsymbol{x}$, inducing an empirical task mapping directly in the residual stream [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 3, pp. 106–108).
- **Chain-of-Thought (CoT) & Problem Decomposition:** Rather than mapping $\boldsymbol{x} \mapsto \boldsymbol{y}_{\mathrm{ans}}$ directly, CoT conditions generation on an explicit intermediate reasoning trajectory $\boldsymbol{z} = (z_1, \dots, z_T)$ before emitting the final answer: $\Pr(\boldsymbol{y}_{\mathrm{ans}} \mid \boldsymbol{x}) = \sum_{\boldsymbol{z}} \Pr(\boldsymbol{y}_{\mathrm{ans}} \mid \boldsymbol{x}, \boldsymbol{z}) \Pr(\boldsymbol{z} \mid \boldsymbol{x})$ [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 3, pp. 122–125). Decomposing multi-step problems allocates serial Transformer compute across intermediate token positions.
- **Self-Consistency & Tree-of-Thoughts (ToT):** Sampling $M$ diverse CoT paths $\{\boldsymbol{z}^{(m)}, \boldsymbol{y}_{\mathrm{ans}}^{(m)}\}_{m=1}^M$ and marginalizing via majority vote approximates the intractable sum over $\boldsymbol{z}$, whereas ToT structures intermediate thoughts as a search tree with lookahead and backtracking ([[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 3, pp. 125–141)).

> [!tip] Mechanistic Interpretability: Logit Lens, SAEs & Induction Heads
> How does a frozen Transformer execute ICL without weight updates? Mechanistic interpretability probes the residual stream across layers $l \in \{1, \dots, L\}$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 10, pp. 248–251):
> - **The Logit Lens:** Projecting intermediate hidden states $\boldsymbol{h}_t^{(l)}$ directly through the unembedding matrix $\mathrm{softmax}(\boldsymbol{W}_o^{\top} \boldsymbol{h}_t^{(l)})$ reveals a layer-wise refinement workspace where early layers resolve entity attributes and later layers format the target language output.
> - **Sparse Autoencoders (SAEs):** Decompose polysemantic neuron activations into sparse, monosemantic feature dictionaries for causal intervention.
> - **Induction Heads:** Two-layer attention circuits implementing the pattern-completion rule $A B \dots A \to B$ (and its fuzzy semantic generalization $A^* B^* \dots A \to B$). When attending from the current token $A$, a previous-token head writes $A$'s identity into $B$'s key, allowing the induction head to perform *prefix matching* back to the prior $A$ and *copy* the subsequent token $B$ (Fig. `fig:decode:induction_head`). Ablating induction heads causally degrades few-shot ICL performance [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 10, pp. 250–251).

> [!note] Source Figure
> *Figure:* Induction head circuit implementing in-context pattern completion: attending from the second occurrence of *vintage*, the prefix-matching mechanism locates the prior occurrence and the copying mechanism boosts the probability of *cars*. Source: [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 10, Fig. 10.7, p. 251 (PDF p. 259)). (`speech_language_processing_aug26_p259_fig1.png`)

## Core Mathematical Derivations: Decoding & Sampling

Exact maximum a posteriori (MAP) decoding $\arg\max_{\boldsymbol{y}} \log \Pr_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})$ over a search tree of depth $n_{\max}$ and branching factor $|\mathcal{V}|$ requires evaluating $|\mathcal{V}|^{n_{\max}}$ sequences, which is computationally intractable [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, pp. 219–220). Practical decoding algorithms prune the level-$i$ candidate set via $\mathcal{Y}_i = \mathrm{Prune}(\mathcal{Y}_{i-1} \times \mathcal{V})$.

> [!important] Deterministic Search: Greedy Decoding & Beam Search
> At step $i$, the cumulative log-probability of extending prefix $\boldsymbol{y}_{<i}$ with token $y_i$ decomposes additively:
>
> $$
> \log \Pr(\boldsymbol{y}_{1:i} \mid \boldsymbol{x}) = \log \Pr(\boldsymbol{y}_{<i} \mid \boldsymbol{x}) + \log \Pr(y_i \mid \boldsymbol{x}, \boldsymbol{y}_{<i}).
> $$
>
> - **Greedy Decoding:** Sets $|\mathcal{Y}_i| = 1$ by selecting $\hat{y}_i = \arg\max_{y_i \in \mathcal{V}} \log \Pr(y_i \mid \boldsymbol{x}, \boldsymbol{y}_{<i})$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Sec. 7.6.1, p. 196). Greedy decoding is $O(n|\mathcal{V}|)$ but is locally myopic and prone to repetitive degeneration.
> - **Beam Search:** Maintains the top-$K$ hypotheses $\mathcal{Y}_i = \mathrm{argTopK}_{\boldsymbol{y} \in \mathcal{Y}_{i-1} \times \mathcal{V}} \log \Pr(\boldsymbol{y} \mid \boldsymbol{x})$ with beam width $K$ (typically $K \in \{2, 4\}$) [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, pp. 221–222). Beam search excels in constrained sequence-to-sequence tasks (e.g., machine translation) but often produces generic, low-entropy text in open-ended generation.

To restore diversity and match human-like entropy profiles, stochastic decoding samples $y_i \sim p_{\tau}(\cdot \mid \boldsymbol{x}, \boldsymbol{y}_{<i})$ after shaping and truncating the logit distribution.

> [!important] Temperature Scaling, Top-$k$ & Top-$p$ (Nucleus) Filtering
> Given logits $\boldsymbol{u} \in \mathbb{R}^{|\mathcal{V}|}$ and temperature $\tau > 0$, the temperature-scaled distribution is [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Sec. 5.3.1, pp. 380–387), [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Sec. 7.6.3, p. 198):
>
> $$
> p_{\tau}(w_j) = \frac{\exp(u_j / \tau)}{\sum_{v=1}^{|\mathcal{V}|} \exp(u_v / \tau)}, \qquad
>   \lim_{\tau \to 0^+} p_{\tau}(w_j) = \mathbb{I}[j = \arg\max_v u_v], \quad
>   \lim_{\tau \to \infty} p_{\tau}(w_j) = \frac{1}{|\mathcal{V}|}.
> $$
>
> Sampling directly from $p_{\tau}$ across all $|\mathcal{V}|$ tokens exposes generation to the *unreliable tail* (accumulated probability mass over thousands of implausible tokens). Two truncation filters eliminate this tail:
> - **Top-$k$ Filtering:** Retain the $k$ highest-logit indices $\mathcal{V}_k = \mathrm{argTopK}(\boldsymbol{u}, k)$ and mask all other logits to $-\infty$ prior to softmax [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Sec. 5.3.2, pp. 388–393):
>
> $$
> \tilde{u}_j = \begin{cases} u_j & \text{if } j \in \mathcal{V}_k, \\ -\infty & \text{otherwise}, \end{cases}
>     \qquad p_{\tau, k}(w_j) = \frac{\exp(\tilde{u}_j / \tau)}{\sum_{v \in \mathcal{V}_k} \exp(u_v / \tau)}.
> $$
>
> - **Top-$p$ (Nucleus) Filtering:** Sort tokens in descending probability order $p_{\tau}(w_{(1)}) \ge p_{\tau}(w_{(2)}) \ge \dots \ge p_{\tau}(w_{(|\mathcal{V}|)})$ and define the minimal nucleus $\mathcal{V}_p = \{w_{(1)}, \dots, w_{(m^*)}\}$ whose cumulative mass meets threshold $p \in (0, 1]$ [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, pp. 222–224), [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Sec. 7.6.4, p. 200):
>
> $$
> m^* = \min \left\{ m \in \{1, \dots, |\mathcal{V}|\} : \sum_{r=1}^{m} p_{\tau}(w_{(r)}) \ge p \right\},
>     \qquad p_{\tau, p}(w_j) = \frac{p_{\tau}(w_j) \mathbb{I}[w_j \in \mathcal{V}_p]}{\sum_{v \in \mathcal{V}_p} p_{\tau}(v)}.
> $$
>

> [!tip] Why Nucleus ($p$) Adapts Better Than Fixed $k$
> When the model is highly certain (e.g., completing a deterministic idiom), $p_{\tau}(w_{(1)}) \approx 0.95$, so a fixed $k = 50$ forces 49 junk tokens into the candidate pool, whereas top-$p$ with $p = 0.9$ automatically collapses $|\mathcal{V}_p| = 1$. Conversely, in high-entropy contexts (e.g., starting a story), probability mass spreads across hundreds of valid continuations; top-$p$ expands $|\mathcal{V}_p|$ dynamically while still excising the long tail.

> [!example] Algorithm: Temperature-Scaled Top-$k$ / Top-$p$ Nucleus Sampling
> **Input:** Context $\boldsymbol{x}$, model $\theta$, temperature $\tau \ge 0$, top-$k$ bound $k \le |\mathcal{V}|$, nucleus $p \in (0, 1]$, max tokens $n_{\max}$
> **Output:** Generated sequence $\boldsymbol{y}$
> Initialize $\boldsymbol{y} \leftarrow ()$, $\mathrm{cache} \leftarrow \mathrm{Dec}_{\mathrm{kv}}(\boldsymbol{x})$  *// Prefill phase*
> **for** $i \leftarrow 1$ to $n_{\max}$:
>
>   Compute logits $\boldsymbol{u}_i \in \mathbb{R}^{|\mathcal{V}|}$ using $\mathrm{cache}$
>   **if** $\tau = 0$:
>
>     $y_i \leftarrow \arg\max_{v \in \mathcal{V}} u_{i,v}$  *// Greedy fallback*
>
>   **else**:
>
>     **if** $k < |\mathcal{V}|$:
>
>       $\mathcal{V}_k \leftarrow \mathrm{argTopK}(\boldsymbol{u}_i, k)$;  set $u_{i,v} \leftarrow -\infty$ for $v \notin \mathcal{V}_k$
>
>     $\boldsymbol{p} \leftarrow \mathrm{softmax}(\boldsymbol{u}_i / \tau)$
>     Sort $\boldsymbol{p}$ descending: $p_{(1)} \ge p_{(2)} \ge \dots$; find $m^* = \min \{m : \sum_{r=1}^m p_{(r)} \ge p\}$
>     $\mathcal{V}_p \leftarrow \{(1), \dots, (m^*)\}$;  zero out $p_v$ for $v \notin \mathcal{V}_p$ and renormalize $\boldsymbol{p} \leftarrow \boldsymbol{p} / \sum_{v \in \mathcal{V}_p} p_v$
>     Sample $y_i \sim \mathrm{Cat}(\boldsymbol{p})$
>
>   Append $y_i$ to $\boldsymbol{y}$ and update $\mathrm{cache}$
>   **if** $y_i = \langle\mathrm{EOS}\rangle$:
> **break**
>
> **return** $\boldsymbol{y}$

## KV Caching, Prefill vs. Decode & Efficient Serving

In causal self-attention, keys $\boldsymbol{k}_{t}$ and values $\boldsymbol{v}_{t}$ depend only on tokens $\le t$ and never change in subsequent steps. Recomputing $\boldsymbol{K}_{1:i'}, \boldsymbol{V}_{1:i'}$ from scratch at every step incurs $O(n^2)$ redundant work. Storing past key–value pairs in a *KV cache* $\mathrm{cache} = \{(\boldsymbol{K}^{(l)}, \boldsymbol{V}^{(l)})\}_{l=1}^{n_{\mathrm{layers}}}$ splits inference into two hardware-distinct phases [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, pp. 214–217) (Fig. `fig:decode:prefill_decode_book`):
- **Prefilling (Compute-Bound):** Processes all $m$ prompt tokens in parallel via $\mathrm{cache} = \mathrm{Dec}_{\mathrm{kv}}(\boldsymbol{x})$, executing dense GEMMs with high arithmetic intensity and dominating Time-to-First-Token (TTFT).
- **Decoding (Memory-Bandwidth-Bound):** Generates one token at a time, appending $(\boldsymbol{k}_{i'}, \boldsymbol{v}_{i'})$ to $\mathrm{cache}$ and attending $\boldsymbol{q}_{i'}$ against all $i'$ cached keys/values, governing Inter-Token Latency (ITL).

> [!note] Source Figure
> *Figure:* Two-phase LLM inference: (a) Prefilling processes all $m$ input tokens in parallel to populate the initial KV cache; (b) Decoding generates one token per step, appending $(\boldsymbol{k}_{m+n}, \boldsymbol{v}_{m+n})$ to the KV cache and attending $\boldsymbol{q}_{m+n}$ across all prior keys and values. Source: [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, Fig. 5.3, p. 217). (`foundations_large_language_models_p217_fig1.png`)

> [!important] KV Cache Memory Footprint & Arithmetic Intensity
> For a Transformer with $n_{\mathrm{layers}}$ layers, $n_{\mathrm{kv}}$ key-value heads per layer, head dimension $d_h$, batch size $b$, total sequence length $s = m + n$, and $p_{\mathrm{bytes}}$ bytes per element (e.g., $p_{\mathrm{bytes}} = 2$ for FP16/BF16), the total KV cache footprint is
>
> $$
> M_{\mathrm{KV}} = 2 \cdot b \cdot s \cdot n_{\mathrm{layers}} \cdot n_{\mathrm{kv}} \cdot d_h \cdot p_{\mathrm{bytes}} \quad [\text{bytes}],
> $$
>
> where the factor of $2$ accounts for storing both $\boldsymbol{K}$ and $\boldsymbol{V}$ tensors. Architectural variants reduce $n_{\mathrm{kv}}$ relative to query heads $n_q$:
> - **Multi-Head Attention (MHA):** $n_{\mathrm{kv}} = n_q$ (baseline $1\times$).
> - **Grouped-Query Attention (GQA):** $1 < n_{\mathrm{kv}} < n_q$, sharing each KV head across $n_q / n_{\mathrm{kv}}$ query heads (e.g., $8\times$ memory reduction for $n_q=64, n_{\mathrm{kv}}=8$).
> - **Multi-Query Attention (MQA):** $n_{\mathrm{kv}} = 1$, achieving maximal $n_q\times$ KV reduction.
>
> **Arithmetic Intensity Gap:** During a single decode step over $b$ sequences, each linear projection performs $\approx 2 b P_{\mathrm{layer}}$ FLOPs while loading $\approx P_{\mathrm{layer}} \cdot p_{\mathrm{bytes}}$ weight bytes plus $M_{\mathrm{KV}} / n_{\mathrm{layers}}$ cache bytes from HBM. For small $b$, arithmetic intensity $\mathcal{I}_{\mathrm{decode}} = \frac{\mathrm{FLOPs}}{\mathrm{Bytes}} \approx \frac{2b}{p_{\mathrm{bytes}}} \sim O(b)$ sits far below modern GPU ridge points ($\sim 200$–$300 \mathrm{FLOPs/byte}$), making decoding strictly memory-bandwidth-bound.

To overcome HBM capacity and bandwidth bottlenecks in production serving, modern inference engines combine four complementary techniques [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, pp. 230–241):
- **PagedAttention:** Partitions each sequence's KV cache into fixed-size non-contiguous physical blocks managed via a block table (analogous to OS virtual memory), eliminating external fragmentation and enabling copy-on-write prefix sharing [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, pp. 237–238).
- **Continuous Batching & Chunked Prefilling:** Iteration-level scheduling injects new requests into active batches as soon as sequences emit $\langle\mathrm{EOS}\rangle$, while chunked prefilling splits long prompts into fixed-size chunks overlapped with decode steps to bound ITL variance [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, pp. 235–239).
- **IO-Aware Tiling (FlashAttention):** Tiles $\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}$ into GPU SRAM and maintains running $(\max, \text{sum})$ statistics for online softmax, avoiding $O(s^2)$ HBM materialization of the attention matrix.
- **KV & Weight Quantization:** Compresses weights (e.g., INT4/FP8) and KV caches (INT8/FP8) to halve or quarter HBM traffic per decode step, trading minor calibration error for proportional TPS gains [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, pp. 240–241).

> [!example] Visual Diagram (LaTeX / PDF)
> *Unified inference architecture: (Left) Compute-bound prefill populates the paged KV cache; (Center) Memory-bandwidth-bound autoregressive decode appends one token per step; (Right) Speculative decoding amortizes target-model HBM reads by verifying $\tau$ draft tokens in a single parallel pass.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:decode:kv_memory_scaling` quantifies the impact of Eq. `(eq:decode:kv_memory)` for a 70B-class Transformer ($n_{\mathrm{layers}}=80, d_h=128$, FP16 $p_{\mathrm{bytes}}=2$, single request $b=1$) across MHA ($n_{\mathrm{kv}}=64$), GQA-8 ($n_{\mathrm{kv}}=8$), GQA-4 ($n_{\mathrm{kv}}=4$), and MQA ($n_{\mathrm{kv}}=1$). At $s=128\mathrm{k}$ tokens, MHA consumes $320 \mathrm{GiB}$ for a single sequence's KV cache alone, whereas GQA-8 reduces this to $40 \mathrm{GiB}$ and MQA to $5 \mathrm{GiB}$.

> [!example] Visual Diagram (LaTeX / PDF)
> *KV cache memory scaling (Eq. `(eq:decode:kv_memory)`) for a 70B-class Transformer ($n_{\mathrm{layers}}=80$, $d_h=128$, FP16, $b=1$). Under MHA ($n_{\mathrm{kv}}=64$), a single $128\mathrm{k}$-token context requires $320 \mathrm{GiB}$, exceeding a single $80 \mathrm{GiB}$ GPU by $4\times$, whereas GQA-8 reduces this to $40 \mathrm{GiB}$ and MQA to $5 \mathrm{GiB}$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Speculative Decoding: Lossless Rejection Sampling

Because autoregressive decoding is memory-bandwidth-bound, evaluating $\tau + 1$ tokens in parallel on the target model $p(\cdot)$ takes nearly the same wall-clock time as evaluating a single token. Speculative decoding exploits this by using a small, fast draft model $q(\cdot)$ to autoregressively propose $\tau$ candidate tokens $\hat{y}_{i+1}, \dots, \hat{y}_{i+\tau}$, which $p(\cdot)$ verifies in one parallel forward pass [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, pp. 225–228).

> [!important] Proof of Exact Distribution Equivalence (Lossless Speculative Sampling)
> Let $q(y)$ and $p(y)$ denote the draft and target next-token distributions at position $t$ given prefix $(\boldsymbol{x}, \boldsymbol{y}_{<i}, \hat{\boldsymbol{y}}_{1:t-1})$. Sample $\hat{y} \sim q(\cdot)$ and $r \sim \mathcal{U}(0, 1)$:
> - If $r \le \frac{p(\hat{y})}{q(\hat{y})}$ (equivalently, always accept when $q(\hat{y}) \le p(\hat{y})$), **accept** $\hat{y}$.
> - Otherwise, **reject** $\hat{y}$ and all subsequent draft tokens, and resample $\bar{y}$ from the normalized residual distribution
>
> $$
> p_{\mathrm{res}}(y) = \frac{\max(0,  p(y) - q(y))}{Z}, \qquad Z = \sum_{v \in \mathcal{V}} \max(0,  p(v) - q(v)) = 1 - \sum_{v \in \mathcal{V}} \min(p(v), q(v)).
> $$
>
> **Theorem (Losslessness):** The emitted token $Y$ satisfies $\Pr(Y = y) = p(y)$ for all $y \in \mathcal{V}$.
>
> *Proof.* By the law of total probability, the overall acceptance probability is $\alpha = \sum_{v \in \mathcal{V}} q(v) \min\!\left(1, \frac{p(v)}{q(v)}\right) = \sum_{v \in \mathcal{V}} \min(p(v), q(v)) = 1 - Z$. For any $y \in \mathcal{V}$:
>
> $$
> \begin{aligned}
> \Pr(Y = y) &= \Pr(\text{accept } y) + \Pr(\text{reject}) \cdot p_{\mathrm{res}}(y)  \\
>   &= q(y)\min\!\left(1, \frac{p(y)}{q(y)}\right) + Z \cdot \frac{\max(0,  p(y) - q(y))}{Z}  \\
>   &= \min(p(y), q(y)) + \max(0,  p(y) - q(y)) = p(y). \qquad \square
> \end{aligned}
> $$
>
> If all $\tau$ draft tokens are accepted, the target model emits one bonus token from $p(\cdot \mid \boldsymbol{x}, \boldsymbol{y}_{<i}, \hat{\boldsymbol{y}}_{1:\tau})$ at zero extra forward-pass cost, yielding up to $\tau + 1$ exact target tokens per verification step.

> [!example] Algorithm: Lossless Speculative Decoding (Draft–Verify–Resample)
> **Input:** Prefix $(\boldsymbol{x}, \boldsymbol{y}_{<i})$, draft model $q$, target model $p$, lookahead $\tau \ge 1$
> **Output:** Accepted tokens $(\tilde{y}_{i+1}, \dots, \tilde{y}_{i+n_a+1})$ distributed identically to $p$
> **for** $t \leftarrow 1$ to $\tau$:
>
>   Sample $\hat{y}_{i+t} \sim q(\cdot \mid \boldsymbol{x}, \boldsymbol{y}_{<i}, \hat{y}_{i+1:i+t-1})$
>
> Evaluate $p(\cdot \mid \boldsymbol{x}, \boldsymbol{y}_{<i}, \hat{y}_{i+1:i+t-1})$ in parallel for $t = 1, \dots, \tau + 1$
> Initialize $n_a \leftarrow \tau$
> **for** $t \leftarrow 1$ to $\tau$:
>
>   Draw $r_t \sim \mathcal{U}(0, 1)$
>   **if** $r_t > \frac{p(\hat{y}_{i+t})}{q(\hat{y}_{i+t})}$:
>
>     $n_a \leftarrow t - 1$
>     Sample $\bar{y}_{i+n_a+1} \sim p_{\mathrm{res}}(\cdot)$ via Eq. `(eq:decode:residual_dist)`
>     **break**
>
> **if** $n_a = \tau$:
>
>   Sample bonus token $\bar{y}_{i+\tau+1} \sim p(\cdot \mid \boldsymbol{x}, \boldsymbol{y}_{<i}, \hat{y}_{i+1:i+\tau})$
>
> **return** $(\hat{y}_{i+1}, \dots, \hat{y}_{i+n_a}, \bar{y}_{i+n_a+1})$

## Test-Time Scaling, Verifiers & RLVR Reasoning

Beyond pretraining and fine-tuning scaling laws, *inference-time (test-time) scaling* improves accuracy on complex reasoning tasks by allocating additional compute during generation without updating parameters $\theta$ ([[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, pp. 241–251); [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 7, pp. 101–105)). Test-time compute scales along two primary axes:
- **Parallel Scaling (Best-of-$N$ & Majority Voting):** Sample $N$ independent rollouts $\{\boldsymbol{y}^{(1)}, \dots, \boldsymbol{y}^{(N)}\}$ and aggregate via unweighted majority vote (Self-Consistency) or select $\arg\max_{m} V(\boldsymbol{x}, \boldsymbol{y}^{(m)})$ using a learned verifier $V$ [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, pp. 244–248).
- **Sequential Scaling & Step-Level Search:** Extend a single reasoning trajectory via self-refinement or explore partial reasoning states $\boldsymbol{a}_{\le i} = (a_1, \dots, a_i)$ with a *Process Reward Model* (PRM), which scores intermediate steps $V(\boldsymbol{x}, \boldsymbol{a}_{\le i})$ rather than only terminal outcomes (Outcome Reward Models, ORMs), enabling beam search or MCTS over thought steps [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, pp. 248–251).
- **RLVR-Elicited Long Thinking:** Reinforcement Learning with Verifiable Rewards (RLVR) trains policies $\pi_{\theta}$ against deterministic binary verifiers $r \in \{0, \gamma\}$ (e.g., exact math extraction via `\boxed{}` or unit-test execution, Fig. `fig:decode:rlvr_loop`), teaching models to allocate long CoT budgets, backtrack, and self-verify natively at test time [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 7, pp. 101–112).

> [!note] Source Figure
> *Figure:* Reinforcement Learning with Verifiable Rewards (RLVR) loop: instead of a learned subjective reward model, a deterministic verifier scores completions $a_i$ with $r_i \in \{0, \gamma\}$, eliciting extended test-time reasoning chains. Source: [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 7, Fig. 28, p. 97 (PDF p. 103)). (`reinforcement_learning_from_human_feedback_p103_fig1.png`)

Figure `fig:decode:test_time_scaling` illustrates the empirical scaling regimes of test-time compute: naive majority voting saturates early as $N$ grows, whereas PRM-guided Best-of-$N$ and RLVR sequential deliberation continue converting inference FLOPs into higher pass@1 accuracy.

> [!example] Visual Diagram (LaTeX / PDF)
> *Conceptual test-time compute scaling laws across inference budgets $N$ (illustrative log-linear trends synthesized from [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, pp. 242–251) and [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 7, pp. 101–111)). Majority voting plateaus under correlated errors, while PRM-guided search and RLVR sequential deliberation sustain monotonic gains.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

**Table:** Complexity and hardware bottleneck synthesis across LLM inference mechanisms.

| **Mechanism** | **Compute / Step** | **Memory / Req.** | **Bottleneck & Primary Trade-off** |
| --- | --- | --- | --- |
| Prefill (no cache) | $O(m^2 d + m d^2)$ | $O(b m n_{\mathrm{layers}} n_{\mathrm{kv}} d_h)$ | Compute-bound; sets TTFT |
| Naive Decode (no cache) | $O(s^2 d + s d^2)$ | $O(1)$ cache | $O(n^3)$ total compute; unusable for long $s$ |
| KV-Cached Decode | $O(s d + d^2)$ | $O(b s n_{\mathrm{layers}} n_{\mathrm{kv}} d_h)$ | HBM bandwidth-bound; sets ITL / TPS |
| GQA / MQA | $O(s d + d^2)$ | $\frac{n_{\mathrm{kv}}}{n_q} \times$ MHA | Cuts KV HBM traffic $n_q / n_{\mathrm{kv}}\times$ |
| PagedAttention | $O(s d + d^2)$ | $\sim 0\%$ internal waste | Eliminates fragmentation; enables prefix sharing |
| Speculative Decoding | $O(\tau \cdot C_q + C_p)$ | $+ M_{\mathrm{KV}}(q)$ | Lossless $\sim 2$–$3\times$ speedup if $\alpha$ is high |
| Test-Time Search ($N$) | $O(N \cdot C_{\mathrm{seq}})$ | $O(N \cdot M_{\mathrm{KV}})$ | Trades $N\times$ inference FLOPs for pass@1 |

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] KV Cache Memory Exhaustion & Contiguous Fragmentation
> Pre-allocating contiguous KV buffers for $s_{\max}$ tokens per request wastes up to $60$–$80\%$ of GPU VRAM to internal and external fragmentation [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, p. 237). Under high concurrency, this triggers out-of-memory (OOM) evictions before compute is saturated. *Remedy:* Use PagedAttention block tables, GQA/MQA, and chunked prefilling to cap per-iteration memory spikes.

> [!warning] Logit Overflow, Temperature Collapse & Top-$p$ Empty Support
> Dividing logits by tiny $\tau \to 0^+$ causes $\exp(u_j / \tau)$ to overflow FP16/BF16 ($> 65504$) into `NaN`. Conversely, applying top-$p$ without safeguarding the top-1 token can yield an empty set if numerical rounding drops $p_{(1)}$ below $p$, or if $-\infty$ masking is applied after softmax without renormalization [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Sec. 5.3.3, pp. 394–395). *Remedy:* Route $\tau = 0$ directly to $\arg\max$, apply log-sum-exp stabilization ($\boldsymbol{u} - \max \boldsymbol{u}$), and always retain at least $w_{(1)}$ in $\mathcal{V}_p$.

> [!warning] Speculative Decoding Acceptance Collapse
> If the draft model $q$ diverges from $p$ (e.g., domain shift, mismatched tokenizer, or high temperature $\tau > 1$ amplifying tail differences), the expected acceptance rate $\alpha = \sum_v \min(p(v), q(v))$ drops sharply, causing wasted draft compute and slower-than-baseline latency. *Remedy:* Dynamically adapt lookahead $\tau$ based on rolling acceptance rates or use self-speculative / Medusa-style heads aligned with $p$.

> [!warning] Overthinking & Verifier Misalignment in Test-Time Scaling
> In RLVR and PRM-guided search, unconstrained thinking budgets induce *overthinking* (repetitive self-verification loops on trivial prompts that inflate latency and degrade accuracy) or reward hacking against imperfect PRMs [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 7, pp. 110–111), [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 5, p. 251). *Remedy:* Apply curriculum length penalties during RLVR training and calibrate PRM scores against ground-truth outcome verifiers.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[Chain-of-Thought & Test-Time Compute]], [[KV Cache & PagedAttention]], [[Speculative Decoding]]
- **Prerequisites (Directed $\leftarrow$):** [[GQA & MQA]], [[KV Cache & PagedAttention]], [[Self-Attention & Multi-Head Attention]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Speculative Decoding]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Chain-of-Thought & Test-Time Compute]], [[Context Memory & Compaction]], [[FlashAttention]], [[GQA & MQA]], [[KV Cache & PagedAttention]], [[MCTS & MuZero]], [[RLVR (Verifiable Rewards)]], [[ReAct & Tool-Use Loop]], [[Speculative Decoding]]
- **Synthesized Sources:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Foundations of Large Language Models]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Speech and Language Processing]]
