---
type: monograph
domain: "D05"
parent_topic: "[[PT 05 - Tokenization, Embeddings, Attention & Transformer Architectures]]"
sub_topic: "[[ST 05 - Subword Tokenization & Vector Semantics]]"
cheatsheet: "[[Cheatsheet - Words, Byte-Pair Encoding (BPE), Distributional Embeddings & Positional Encodings]]"
part_of: 
  - "[[ST 05 - Subword Tokenization & Vector Semantics]]"
  - "[[D05 - Transformers, Language Modeling & LLM Engineering]]"
concepts: 
  - "[[BPE & Subword Tokenization]]"
  - "[[RoPE & Positional Embeddings]]"
prerequisites: []
enables: []
related: 
  - "[[BPE & Subword Tokenization]]"
  - "[[RoPE & Positional Embeddings]]"
  - "[[Self-Attention & Multi-Head Attention]]"
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Speech and Language Processing]]"
tags:
  - monograph
  - domain-05
---

# Words, Byte-Pair Encoding (BPE), Distributional Embeddings & Positional Encodings

> [!abstract] Monograph Navigation
> - **Domain:** [[D05 - Transformers, Language Modeling & LLM Engineering|Transformers, Language Modeling & LLM Engineering]]
> - **Parent / Sub-Topic:** [[PT 05 - Tokenization, Embeddings, Attention & Transformer Architectures|Tokenization, Embeddings, Attention & Transformer Architectures]] → [[ST 05 - Subword Tokenization & Vector Semantics|Subword Tokenization & Vector Semantics]]
> - **Visual Cheatsheet:** [[Cheatsheet - Words, Byte-Pair Encoding (BPE), Distributional Embeddings & Positional Encodings]]
> - **Description:** Synthesizes Unicode/subword tokenization, the BPE merge algorithm, sparse PPMI vs. dense skip-gram/CBOW Word2Vec embeddings, sliding-window causal dataset construction, and sinusoidal vs. learned vs. Rotary Position Embeddings (RoPE).

## Problem Formulation & Unified Notation

Modern sequence models and large language models (LLMs) operate over continuous tensor spaces $\mathbb{R}^{N \times d}$, whereas human language arrives as discrete, variable-length character streams. Bridging this discrete–continuous divide requires a three-stage mathematical pipeline: (1) deterministic segmentation of raw UTF-8 byte streams into a finite subword vocabulary $\mathcal{V}$ of size $\lvert \mathcal{V} \rvert$; (2) projection of discrete token indices $w_t \in \{1,\dots,\lvert \mathcal{V} \rvert\}$ into dense semantic vectors $\boldsymbol{e}_t \in \mathbb{R}^d$ that preserve distributional similarity; and (3) injection of sequence order so that permutation-equivariant self-attention layers can distinguish token positions $t \in \{1,\dots,N\}$.

Let $\mathcal{C}$ denote a training corpus over an alphabet $\Sigma_0$ (typically the $256$ raw UTF-8 bytes $\{0,\dots,255\}$ in byte-level schemes). A tokenizer defines an encoding map $\tau: \Sigma_0^* \to \mathcal{V}^N$ and an inverse decoding map $\tau^{-1}: \mathcal{V}^N \to \Sigma_0^*$ satisfying $\tau^{-1}(\tau(s)) = s$ for every valid UTF-8 string $s$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 2, pp. 40–43 (PDF pp. 48–51)). Given a tokenized sequence $\boldsymbol{w} = (w_1, \dots, w_N) \in \{1,\dots,\lvert \mathcal{V} \rvert\}^N$, each token index $w_t$ corresponds to a one-hot basis vector $\boldsymbol{o}_t \in \{0,1\}^{\lvert \mathcal{V} \rvert}$ with $(\boldsymbol{o}_t)_j = \mathbb{I}[j = w_t]$. Multiplying by the learnable token embedding matrix $\boldsymbol{E} \in \mathbb{R}^{\lvert \mathcal{V} \rvert \times d}$ performs a linear row-selection lookup:

$$
\boldsymbol{e}_t = \boldsymbol{E}^{\top} \boldsymbol{o}_t = \boldsymbol{E}_{w_t, :} \in \mathbb{R}^d,
$$

as formalized both as a one-hot matrix product in [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 7.4, p. 192 (PDF p. 200)) and as an equivalent PyTorch `nn.Embedding` lookup layer in [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 2.7, pp. 126–128). Finally, positional information is combined either additively at the input layer, $\boldsymbol{X}_0 = \boldsymbol{E}_{\boldsymbol{w}} + \boldsymbol{P}_{1:N} \in \mathbb{R}^{N \times d}$, or multiplicatively inside each attention head via relative rotation operators $\boldsymbol{R}_{\Theta, t}^d$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 7.4, p. 193 (PDF p. 201)).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> The three primary references adopt distinct conventions for vocabulary size, embedding dimension, target-versus-context matrices, and sequence indices. Table `tab:tok:notation` establishes a unified symbol set used throughout this monograph.
>
> | **Concept** | **Unified** | **J&M** | **Raschka** | **Murphy Intro** |
> | --- | --- | --- | --- | --- |
> | Vocabulary size | $\lvert \mathcal{V} \rvert$ | $\lvert V \rvert$ | `vocab_size` | $M$ (Sec. 20.5, p. 705) |
> | Embedding dim. | $d$ | $d$ | `output_dim` | $D$ (Sec. 20.5, p. 705) |
> | Context length | $N$ | $N$ | `max_length` | $T$ (Sec. 20.5.2, p. 708) |
> | Embedding matrix | $\boldsymbol{E} \in \mathbb{R}^{\lvert \mathcal{V} \rvert\times d}$ | $\boldsymbol{E}$ | `emb.weight` | $\boldsymbol{V}, \boldsymbol{U} \in \mathbb{R}^{D \times M}$ |
> | Target / context vec. | $\boldsymbol{w}_i, \boldsymbol{c}_j \in \mathbb{R}^d$ | $\boldsymbol{w}, \boldsymbol{c}$ | $\boldsymbol{e}_t$ (tied) | $\boldsymbol{v}_i, \boldsymbol{u}_j \in \mathbb{R}^D$ |
> | Neg. samples count | $k$ | $k$ | — (autoregressive) | $K$ (Sec. 20.5.2.3, p. 709) |
> | Context half-window | $L$ | $L$ | `stride` (sliding) | $m$ or $h$ |

## Core Mathematical Derivations & Theorems

### Unicode, Byte-Pair Encoding (BPE), and Edit Distance

Representing arbitrary multilingual text cannot rely on ASCII ($7$-bit, $128$ codes) or fixed word-level dictionaries, which suffer from out-of-vocabulary (OOV) failures and morphological explosion across polysynthetic or agglutinative languages [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 2.2–2.3, pp. 37–41 (PDF pp. 45–49)). Unicode assigns each character a unique code point in $\mathtt{U+0000}$–$\mathtt{U+10FFFF}$ ($1{,}114{,}111$ slots), and UTF-8 encodes each code point into $1$ to $4$ bytes in a prefix-free, self-synchronizing, ASCII-compatible format.

> [!important] BPE Greedy Pair-Merge Objective & Minimum Edit Distance
> Let $\mathcal{C}_{\mathrm{pre}}$ be a multiset of pre-tokenized strings segmented at whitespace/punctuation boundaries (e.g., via regex pre-tokenization [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 2.6, p. 55 (PDF p. 63))), where each string $s \in \mathcal{C}_{\mathrm{pre}}$ has corpus multiplicity $f(s)$ and is initialized as a sequence of base symbols from $\Sigma_0$. At merge iteration $m \in \{1,\dots,k\}$, given current vocabulary $\mathcal{V}_{m-1}$, BPE selects the adjacent symbol pair $(a^*, b^*) \in \mathcal{V}_{m-1} \times \mathcal{V}_{m-1}$ maximizing total corpus co-occurrence frequency:
>
> $$
> (a^*, b^*) = \arg\max_{(a, b) \in \mathcal{V}_{m-1}^2} \sum_{s \in \mathcal{C}_{\mathrm{pre}}} f(s) \cdot \mathrm{count}(ab \text{ in } s),
> $$
>
> replaces every non-overlapping occurrence of $(a^*, b^*)$ with a new merged symbol $z_m = a^* b^*$, and sets $\mathcal{V}_m = \mathcal{V}_{m-1} \cup \{z_m\}$, yielding $\lvert \mathcal{V}_k \rvert = \lvert \Sigma_0 \rvert + k$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 2.4.1, pp. 43–44 (PDF pp. 51–52)).
>
> To measure lexical similarity or align noisy/misspelled strings prior to tokenization, the **Minimum Edit Distance** (Levenshtein) between source string $X_{1:n}$ and target string $Y_{1:m}$ satisfies the dynamic programming recurrence for $D[i, j]$ with $D[0,0]=0$, $D[i,0]=\sum_{r=1}^i \mathrm{del}(X_r)$, $D[0,j]=\sum_{s=1}^j \mathrm{ins}(Y_s)$:
>
> $$
> D[i, j] = \min \begin{cases}
>     D[i-1, j] + \mathrm{del}(X_i), \\
>     D[i, j-1] + \mathrm{ins}(Y_j), \\
>     D[i-1, j-1] + \mathrm{sub}(X_i, Y_j),
>   \end{cases}
> $$
>
> where standard Levenshtein weights set $\mathrm{del}=\mathrm{ins}=1$ and $\mathrm{sub}(x,y) = 2 \mathbb{I}[x \neq y]$ (or $1$ for unit substitution cost) [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 2.9, pp. 62–63 (PDF pp. 70–71)).

> [!tip] Why Subword BPE Eliminates `<|unk|>` Without Character-Level Blowup
> Pure word-level tokenizers require an `<|unk|>` fallback for unseen compounds (e.g., `someunknownPlace` or `Akwirw ier` in [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 2.5, pp. 104–106)), destroying morphological and lexical signal. Conversely, pure character-level models inflate sequence length $N$, forcing $O(N^2)$ self-attention to expend capacity learning spelling. BPE interpolates smoothly: frequent words compress into single tokens, rare words decompose into reusable morphemes (`low` + `er`), and arbitrary unseen bytes fall back to the base alphabet $\Sigma_0$.

### Distributional Semantics, Cosine Similarity, and SGNS

By the distributional hypothesis [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 5.2, p. 131 (PDF p. 139)), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 20.5, p. 705 (PDF p. 735)), words occurring in similar local contexts share semantic or syntactic roles. Measuring vector similarity via the raw dot product $\boldsymbol{v}^{\top} \boldsymbol{w}$ is biased toward frequent words whose co-occurrence counts produce larger Euclidean norms $\lVert \boldsymbol{v} \rVert_2$. Dividing by the norms yields scale-invariant **cosine similarity**:

$$
\mathrm{sim}_{\cos}(\boldsymbol{v}, \boldsymbol{w}) = \frac{\boldsymbol{v}^{\top} \boldsymbol{w}}{\lVert \boldsymbol{v} \rVert_2 \lVert \boldsymbol{w} \rVert_2} = 1 - \frac{1}{2}\lVert \hat{\boldsymbol{v}} - \hat{\boldsymbol{w}} \rVert_2^2,
$$

where $\hat{\boldsymbol{v}} = \boldsymbol{v}/\lVert \boldsymbol{v} \rVert_2$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 5.4, p. 135 (PDF p. 143)), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 20.5.1, p. 707 (PDF p. 737)).

While sparse count vectors of dimension $\lvert \mathcal{V} \rvert$ capture co-occurrence directly, dense low-rank embeddings $\boldsymbol{w}_i, \boldsymbol{c}_j \in \mathbb{R}^d$ ($d \ll \lvert \mathcal{V} \rvert$) generalize better across rare contexts. In **Skip-Gram with Negative Sampling (SGNS)**, rather than normalizing a full $\lvert \mathcal{V} \rvert$-way softmax over all context words, the model trains a binary logistic classifier to distinguish true co-occurring pairs $(w, c_{\mathrm{pos}})$ within window $\pm L$ from $k$ noise pairs $(w, c_{\mathrm{neg}, i})$ drawn from a smoothed unigram distribution [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 5.5.2, pp. 139–141 (PDF pp. 147–149)), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 20.5.2.3, p. 709 (PDF p. 739)).

> [!important] SGNS Objective, Smoothed Noise Distribution, and Exact Gradients
> For a target word $w \in \mathbb{R}^d$ (row of target matrix $\boldsymbol{W} \in \mathbb{R}^{\lvert \mathcal{V} \rvert\times d}$) and context word $c \in \mathbb{R}^d$ (row of context matrix $\boldsymbol{C} \in \mathbb{R}^{\lvert \mathcal{V} \rvert\times d}$), SGNS models the probability that $(w, c)$ is a genuine corpus pair ($D=1$) via the logistic sigmoid $\sigma(z) = (1 + e^{-z})^{-1}$:
>
> $$
> P(D=1 \mid w, c) = \sigma(\boldsymbol{c}^{\top} \boldsymbol{w}), \qquad P(D=0 \mid w, c) = 1 - \sigma(\boldsymbol{c}^{\top} \boldsymbol{w}) = \sigma(-\boldsymbol{c}^{\top} \boldsymbol{w}).
> $$
>
> Noise words $c_{\mathrm{neg}, 1}, \dots, c_{\mathrm{neg}, k}$ are sampled independently from the $\alpha$-smoothed unigram distribution with $\alpha = 3/4$:
>
> $$
> P_\alpha(w') = \frac{\mathrm{count}(w')^\alpha}{\sum_{u \in \mathcal{V}} \mathrm{count}(u)^\alpha},
> $$
>
> which upweights rare words relative to raw unigram frequencies [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Eq. (5.19), p. 140 (PDF p. 148)). Minimizing the negative log-likelihood of one positive pair and $k$ negative samples defines the per-instance SGNS loss:
>
> $$
> \mathcal{L}_{\mathrm{SGNS}}(\boldsymbol{w}, \boldsymbol{c}_{\mathrm{pos}}, \{\boldsymbol{c}_{\mathrm{neg},i}\}_{i=1}^k)
>   = -\log \sigma(\boldsymbol{c}_{\mathrm{pos}}^{\top} \boldsymbol{w}) - \sum_{i=1}^k \log \sigma(-\boldsymbol{c}_{\mathrm{neg},i}^{\top} \boldsymbol{w}).
> $$
>
> Using $\frac{\mathrm{d}}{\mathrm{d}z}\log\sigma(z) = 1 - \sigma(z)$ and $\frac{\mathrm{d}}{\mathrm{d}z}\log\sigma(-z) = -\sigma(z)$, the exact gradients with respect to $\boldsymbol{c}_{\mathrm{pos}}$, $\boldsymbol{c}_{\mathrm{neg},i}$, and $\boldsymbol{w}$ are:
>
> $$
> \begin{aligned}
> \frac{\partial \mathcal{L}_{\mathrm{SGNS}}}{\partial \boldsymbol{c}_{\mathrm{pos}}} &= \bigl(\sigma(\boldsymbol{c}_{\mathrm{pos}}^{\top} \boldsymbol{w}) - 1\bigr)\boldsymbol{w},  \\
>   \frac{\partial \mathcal{L}_{\mathrm{SGNS}}}{\partial \boldsymbol{c}_{\mathrm{neg},i}} &= \sigma(\boldsymbol{c}_{\mathrm{neg},i}^{\top} \boldsymbol{w})\boldsymbol{w},  \\
>   \frac{\partial \mathcal{L}_{\mathrm{SGNS}}}{\partial \boldsymbol{w}} &= \bigl(\sigma(\boldsymbol{c}_{\mathrm{pos}}^{\top} \boldsymbol{w}) - 1\bigr)\boldsymbol{c}_{\mathrm{pos}} + \sum_{i=1}^k \sigma(\boldsymbol{c}_{\mathrm{neg},i}^{\top} \boldsymbol{w})\boldsymbol{c}_{\mathrm{neg},i}.
> \end{aligned}
> $$
>

### Implicit Matrix Factorization: Shifted PMI SVD and GloVe

Although Word2Vec is framed as predictive neural classification, both Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 20.5.1–20.5.3, pp. 706–710) and Jurafsky & Martin [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 5.5.3, p. 142) highlight a deep equivalence between SGNS and classical count-matrix factorization.

> [!important] Levy & Goldberg Shifted-PMI Equivalence and GloVe Objective
> Let $x_{ij}$ denote the corpus co-occurrence count of target word $i$ and context word $j$, with marginals $x_i = \sum_j x_{ij}$, $x_j = \sum_i x_{ij}$, and total pairs $\lvert D \rvert = \sum_{i,j} x_{ij}$. Summing Eq. `(eq:tok:sgns_loss)` over the corpus yields an expected objective whose stationary point with respect to $z_{ij} = \boldsymbol{w}_i^{\top} \boldsymbol{c}_j$ satisfies:
>
> $$
> \frac{\partial }{\partial z_{ij}} \Bigl[ x_{ij}\log\sigma(z_{ij}) + k \cdot x_i P_\alpha(j) \log\sigma(-z_{ij}) \Bigr] = 0
>    \implies 
>   x_{ij}(1 - \sigma(z_{ij})) - k  x_i P_\alpha(j)\sigma(z_{ij}) = 0.
> $$
>
> Solving for $e^{z_{ij}} = \sigma(z_{ij}) / (1 - \sigma(z_{ij}))$ gives the exact optimal inner product:
>
> $$
> \boldsymbol{w}_i^{\top} \boldsymbol{c}_j = \log\!\left(\frac{x_{ij}}{x_i P_\alpha(j)}\right) - \log k = \mathrm{PMI}_\alpha(i, j) - \log k,
> $$
>
> where $\mathrm{PMI}(i, j) = \log \frac{P(i,j)}{P(i)P(j)}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. (20.143), p. 707 (PDF p. 737)). Thus, SGNS implicitly performs a rank-$d$ factorization of the $\log k$-shifted PMI matrix.
>
> **GloVe** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 20.5.3, Eq. (20.156), p. 710 (PDF p. 740)) makes this factorization explicit via weighted least squares on log co-occurrence counts with learnable scalar biases $b_i, c_j$:
>
> $$
> \mathcal{L}_{\mathrm{GloVe}} = \sum_{i,j \in \mathcal{V}} h(x_{ij})\bigl(\boldsymbol{u}_j^{\top} \boldsymbol{v}_i + b_i + c_j - \log x_{ij}\bigr)^2, \qquad
>   h(x) = \begin{cases} (x/x_{\max})^{0.75} & x < x_{\max}, \\ 1 & x \ge x_{\max}, \end{cases}
> $$
>
> where $x_{\max} = 100$ prevents ultra-frequent stopword pairs from dominating the squared loss.

### Positional Encodings: Absolute, Sinusoidal, and RoPE

Self-attention layers compute pairwise interactions $\boldsymbol{q}_m^{\top} \boldsymbol{k}_n$ that are inherently permutation-equivariant: permuting the input rows of $\boldsymbol{E}_{\boldsymbol{w}}$ permutes the outputs identically. As noted in [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 2.8, pp. 129–131), Transformers restore sequence order through three primary positional mechanisms:
- **Learned Absolute Positional Embeddings**: Used in GPT-2/3 and detailed in [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 2.8, pp. 131–136) and [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 7.4, p. 193), a learnable table $\boldsymbol{E}_{\mathrm{pos}} \in \mathbb{R}^{N_{\max} \times d}$ is added to token embeddings: $\boldsymbol{X}_0[t, :] = \boldsymbol{E}[w_t, :] + \boldsymbol{E}_{\mathrm{pos}}[t, :]$.
- **Sinusoidal Positional Encodings**: To avoid parameterizing every position up to $N_{\max}$ and enable length generalization, the original Transformer assigns fixed multi-frequency waves across dimension pairs $2i, 2i+1$ ($0 \le i < d/2$):

$$
\mathrm{PE}_{(p, 2i)} = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \qquad
  \mathrm{PE}_{(p, 2i+1)} = \cos\!\left(\frac{p}{10000^{2i/d}}\right).
$$
- **Rotary Position Embeddings (RoPE)**: Rather than adding position at the input layer, RoPE rotates query and key vectors inside each attention head by position-dependent angles [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 7.4, p. 193 (PDF p. 201)).

> [!important] Linear Relative-Shift Property of Sinusoids and RoPE Inner-Product Invariance
> For any fixed offset $\Delta p$, each $(2i, 2i+1)$ pair of sinusoidal coordinates at position $p + \Delta p$ is a linear orthogonal rotation of the coordinates at $p$ with angular frequency $\omega_i = 10000^{-2i/d}$:
>
> $$
> \begin{bmatrix} \mathrm{PE}_{(p+\Delta p, 2i)} \\ \mathrm{PE}_{(p+\Delta p, 2i+1)} \end{bmatrix}
>   = \begin{bmatrix} \cos(\omega_i \Delta p) & \sin(\omega_i \Delta p) \\ -\sin(\omega_i \Delta p) & \cos(\omega_i \Delta p) \end{bmatrix}
>   \begin{bmatrix} \mathrm{PE}_{(p, 2i)} \\ \mathrm{PE}_{(p, 2i+1)} \end{bmatrix}.
> $$
>
> RoPE elevates this rotation directly into the attention dot product. Partitioning $\boldsymbol{q}_m, \boldsymbol{k}_n \in \mathbb{R}^{d_h}$ into $d_h/2$ two-dimensional subspaces (identified with $\mathbb{C}^{d_h/2}$), RoPE applies the block-diagonal rotation matrix $\boldsymbol{R}_{\Theta, m}^{d_h} = \mathrm{diag}\bigl(\boldsymbol{R}(\theta_0 m), \dots, \boldsymbol{R}(\theta_{d_h/2-1} m)\bigr)$ where $\theta_i = 10000^{-2i/d_h}$:
>
> $$
> \tilde{\boldsymbol{q}}_m = \boldsymbol{R}_{\Theta, m}^{d_h} \boldsymbol{q}_m, \qquad
>   \tilde{\boldsymbol{k}}_n = \boldsymbol{R}_{\Theta, n}^{d_h} \boldsymbol{k}_n.
> $$
>
> Because rotation matrices satisfy $(\boldsymbol{R}_{\Theta, m}^{d_h})^{\top} \boldsymbol{R}_{\Theta, n}^{d_h} = \boldsymbol{R}_{\Theta, n-m}^{d_h}$, the attention logit depends *strictly* on the relative displacement $n - m$:
>
> $$
> \tilde{\boldsymbol{q}}_m^{\top} \tilde{\boldsymbol{k}}_n = \boldsymbol{q}_m^{\top} \bigl(\boldsymbol{R}_{\Theta, m}^{d_h}\bigr)^{\top} \boldsymbol{R}_{\Theta, n}^{d_h} \boldsymbol{k}_n = \boldsymbol{q}_m^{\top} \boldsymbol{R}_{\Theta, n-m}^{d_h} \boldsymbol{k}_n.
> $$
>

## Algorithmic Implementation & Comparative Synthesis

Algorithm `alg:tok:bpe` unifies the two-phase BPE procedure from Jurafsky & Martin [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 2.4, pp. 43–45) with the byte-level `tiktoken` encoding pipeline in Raschka [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 2.5, pp. 102–106).

> [!example] Algorithm: Byte-Level BPE Merge Training and Greedy Encoding
> **Input:** Corpus $\mathcal{C}$, target merges $k$, base byte alphabet $\Sigma_0 = \{0,\dots,255\}$, test string $s_{\mathrm{test}}$
> **Output:** Vocabulary $\mathcal{V}_k$, ordered merge table $\mathcal{M} = [(a_m, b_m, r_m)]_{m=1}^k$, token IDs $\boldsymbol{w}$
>  *// Phase 1: Training (Inducing Vocabulary and Merge Ranks)*
> Pre-tokenize $\mathcal{C}$ via regex into word-frequency multiset $\mathcal{C}_{\mathrm{pre}} = \{(s, f(s))\}$
> Convert each $s$ to UTF-8 byte sequence; initialize $\mathcal{V}_0 \leftarrow \Sigma_0$, $\mathcal{M} \leftarrow []$
> **for** $m = 1$ to $k$:
>
>   Compute pair counts $C(a, b) \leftarrow \sum_{(s, f) \in \mathcal{C}_{\mathrm{pre}}} f \cdot \mathrm{count}((a, b) \text{ in } s)$
>   **if** $C$ is empty:
> **break**
>   $(a^*, b^*) \leftarrow \arg\max_{(a,b)} C(a, b)$
>   $\mathcal{V}_m \leftarrow \mathcal{V}_{m-1} \cup \{a^* b^*\}$; append $(a^*, b^*, m)$ to $\mathcal{M}$
>   Replace all non-overlapping adjacent $(a^*, b^*)$ with $a^* b^*$ in every $s \in \mathcal{C}_{\mathrm{pre}}$
>
>  *// Phase 2: Greedy Rank-Ordered Encoding*
> Segment $s_{\mathrm{test}}$ via regex into pre-tokens; map each pre-token to UTF-8 bytes
> **for each** byte sequence $\boldsymbol{u} = (u_1, \dots, u_L)$:
>
>   **while** $\lvert \boldsymbol{u} \rvert \ge 2$:
>
>     Find adjacent pair $(u_j, u_{j+1})$ with minimum merge rank $r_m$ in $\mathcal{M}$
>     **if** no adjacent pair exists in $\mathcal{M}$:
> **break**
>     Merge all occurrences of $(u_j, u_{j+1})$ into $u_j u_{j+1}$ in $\boldsymbol{u}$
>
>   Append vocabulary IDs of $\boldsymbol{u}$ to $\boldsymbol{w}$
>
> **return** $\mathcal{V}_k, \mathcal{M}, \boldsymbol{w}$

**Table:** Asymptotic time and space complexity across tokenization, static embedding, and positional encoding methods ($\lvert \mathcal{C} \rvert$: corpus tokens; $k$: BPE merges; $N$: sequence length; $d$: embedding dimension).

| **Method** | **Time Complexity** | **Space / Params** | **Key Operational Trade-off** |
| --- | --- | --- | --- |
| BPE Training (naive) | $O(k \cdot \lvert \mathcal{C} \rvert_{\mathrm{uniq}})$ | $O(\lvert \mathcal{V} \rvert + \lvert \mathcal{C} \rvert_{\mathrm{uniq}})$ | Heap-indexed pair counts reduce update cost to $O(\log \lvert \mathcal{V} \rvert)$ per occurrence. |
| BPE Encoding | $O(N \log N)$ or $O(N \cdot k_{\mathrm{local}})$ | $O(\lvert \mathcal{V} \rvert)$ | Deterministic greedy merge by training rank; no `<\|unk\|>` tokens needed. |
| SGNS (Word2Vec) | $O(\lvert \mathcal{C} \rvert \cdot L \cdot k \cdot d)$ | $2\lvert \mathcal{V} \rvert d$ | Avoids $O(\lvert \mathcal{V} \rvert)$ softmax via $k$ negative samples; static per type. |
| GloVe | $O(\lvert \mathrm{nnz}(\boldsymbol{X}) \rvert \cdot d)$ / epoch | $2\lvert \mathcal{V} \rvert(d+1)$ | Trains on non-zero co-occurrences $\lvert \mathrm{nnz}(\boldsymbol{X}) \rvert \ll \lvert \mathcal{V} \rvert^2$; global statistics. |
| Learned Abs. PE | $O(N d)$ | $N_{\max} d$ | Expressive for $t \le N_{\max}$; cannot extrapolate beyond $N_{\max}$. |
| Sinusoidal PE | $O(N d)$ | $0$ (buffers) | Parameter-free; linear relative-shift property, weak long-range extrapolation. |
| RoPE | $O(N d)$ per layer | $0$ (buffers) | Exact relative inner product $\tilde{\boldsymbol{q}}_m^{\top} \tilde{\boldsymbol{k}}_n$; supports base-$\theta$ context scaling. |

> [!info] Cross-Book Synthesis: Static Vectors vs. Contextual LLM Input Pipelines
> Jurafsky & Martin [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 5, pp. 139–143) and Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 20.5, pp. 707–711) treat embeddings primarily as *static* lexical representations learned via SGNS, GloVe, or shifted-PMI SVD, where word analogies ($\boldsymbol{v}_{\text{queen}} \approx \boldsymbol{v}_{\text{king}} - \boldsymbol{v}_{\text{man}} + \boldsymbol{v}_{\text{woman}}$) emerge from log-linear co-occurrence geometry (justified theoretically by the RAND-WALK latent discourse model [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 20.5.5, p. 711)). By contrast, Raschka [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 2.6–2.8, pp. 107–139) constructs embeddings as the trainable first layer of an autoregressive Transformer: a sliding window of length `max_length` and stride $s$ extracts input–target tensor pairs $(\boldsymbol{X}_{t:t+N-1}, \boldsymbol{X}_{t+1:t+N})$, `nn.Embedding` maps IDs to $\mathbb{R}^d$, and positional vectors are added before self-attention. In modern LLMs, $\boldsymbol{E}$ is frequently weight-tied to the unembedding head $\boldsymbol{U} = \boldsymbol{E}^{\top}$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 7.5, pp. 194–195 (PDF pp. 202–203)), unifying input representation and next-token logit projection.

## Visual Diagnostics & Geometric Structure

Figure `fig:tok:pipeline_tikz` synthesizes the complete end-to-end transformation from raw UTF-8 text through byte-level BPE, embedding lookup, and positional injection into the Transformer input tensor $\boldsymbol{X}_0$.

> [!example] Visual Diagram (LaTeX / PDF)
> *End-to-end LLM input representation pipeline: raw UTF-8 text undergoes regex pre-tokenization, byte-level BPE merging, vocabulary ID mapping, linear embedding lookup $\boldsymbol{E}$, and positional injection (additive $\boldsymbol{P}_{1:N}$ or multiplicative RoPE $\boldsymbol{R}_{\Theta,t}^{d_h}$) to yield $\boldsymbol{X}_0 \in \mathbb{R}^{N \times d}$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

When encountering rare or out-of-vocabulary strings, BPE gracefully backs off along its merge hierarchy without emitting `<|unk|>`, as illustrated in Figure `fig:tok:bpe_unk`.

> [!note] Source Figure
> *Figure:* Subword fallback in GPT-2 byte-level BPE (`tiktoken`): the unseen string `"Akwirw ier"` is segmented into known subwords and individual characters with exact round-trip decodability [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 2.5, Fig. 2.11, p. 105). (`build_large_language_model_from_scratch_p105_fig1.png`)

In static distributional embedding spaces, CBOW and Skip-Gram represent dual formulations of local context prediction (Figure `fig:tok:cbow_skipgram`), while SGNS gradient updates (Figure `fig:tok:sgns_grad`) pull target vectors toward observed context embeddings and push them away from $k$ negative noise samples.

> [!note] Source Figure
> *Figure:* Comparison of Word2Vec architectures for window half-size $m=2$: (a) Continuous Bag of Words (CBOW) averages context vectors to predict $w_t$; (b) Skip-Gram predicts surrounding context words $w_{t\pm j}$ from center word $w_t$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 20.5.2, Fig. 20.44, p. 708 (PDF p. 738)). (`probabilisticml_kevinmurphy_introduction_p738_fig1.png`)

> [!note] Source Figure
> *Figure:* Geometric intuition of one SGNS gradient descent step from Eqs. `(eq:tok:sgns_grad_pos)`–`(eq:tok:sgns_grad_w)`: increasing $\boldsymbol{c}_{\mathrm{pos}}^{\top} \boldsymbol{w}$ for observed pair (*apricot*, *jam*) while decreasing $\boldsymbol{c}_{\mathrm{neg},i}^{\top} \boldsymbol{w}$ for $k=2$ noise words (*matrix*, *Tolstoy*) [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 5.5.2, Fig. 5.7, p. 141 (PDF p. 149)). (`speech_language_processing_aug26_p149_fig1.png`)

Figure `fig:tok:sinusoid_spectrum` plots the multi-scale sinusoidal basis $\mathrm{PE}_{(p, 2i)} = \sin(p / 10000^{2i/d})$ for $d=512$ across positions $p \in [0, 100]$. Low-index dimensions ($i=0$, wavelength $2\pi$) oscillate rapidly to resolve fine local offsets, whereas higher-index dimensions ($i=64, 128$) vary slowly to encode coarse global position.

> [!example] Visual Diagram (LaTeX / PDF)
> *Multi-scale frequency spectrum of sinusoidal positional encodings ($d=512$, Eq. `(eq:tok:sinusoid)`). High-frequency dimensions ($i=0$) distinguish adjacent tokens, while low-frequency dimensions ($i=128$) track long-range position without wrapping.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Multilingual Over-Segmentation and Fertility Inflation
> When a BPE tokenizer is trained on an English-dominated corpus, non-English scripts receive far fewer merge rules, inflating *token fertility* (tokens per word) by $2\times$–$5\times$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 2.4.3, pp. 46–47 (PDF pp. 54–55)). For example, a $16$-word Spanish sentence can fragment into $33$ tokens (splitting *hondo* into `h` + `ondo`), whereas its $14$-word English translation requires only $19$ tokens. Because self-attention scales as $O(N^2)$, over-segmentation directly increases inference latency, exhausts finite context windows, and degrades cross-lingual semantic quality. **Remedy**: Balance training corpora across target languages during BPE induction and monitor per-language fertility.

> [!warning] Unsmoothed Noise Sampling ($\alpha=1$) and Raw Dot-Product Bias
> Using unsmoothed unigram probabilities ($\alpha=1$) in SGNS over-samples high-frequency stopwords as negatives while starving rare content words of contrastive updates; setting $\alpha=0.75$ in Eq. `(eq:tok:noise_dist)` boosts rare-word sampling probability (e.g., from $0.01$ to $0.03$ in a two-word illustration) [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 5.5.2, p. 140 (PDF p. 148)). Similarly, ranking word similarity by unnormalized dot product $\boldsymbol{v}^{\top} \boldsymbol{w}$ conflates semantic relatedness with word frequency; always normalize via cosine similarity `(eq:tok:cosine)` or evaluate shifted PMI `(eq:tok:shifted_pmi)`.

> [!warning] Learned Positional Extrapolation Failure and Context Truncation
> Learned absolute positional tables $\boldsymbol{E}_{\mathrm{pos}} \in \mathbb{R}^{N_{\max} \times d}$ suffer from two failure modes: (1) outer positions $t \approx N_{\max}$ receive far fewer gradient updates if training documents are shorter than $N_{\max}$, leaving those rows under-trained [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 7.4, p. 193 (PDF p. 201)); and (2) sequences exceeding $N_{\max}$ cannot be indexed and must be truncated [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 2.8, p. 135). **Remedy**: Use RoPE `(eq:tok:rope_rel)` with base-$\theta$ frequency scaling (YaRN / NTK-aware interpolation) when context-length extrapolation is required.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[BPE & Subword Tokenization]], [[RoPE & Positional Embeddings]]
- **Prerequisites (Directed $\leftarrow$):** None
- **Enables / Unlocks (Directed $\rightarrow$):** None
- **Related Concepts (Undirected $\leftrightarrow$):** [[BPE & Subword Tokenization]], [[RoPE & Positional Embeddings]], [[Self-Attention & Multi-Head Attention]]
- **Synthesized Sources:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Probabilistic Machine Learning]], [[Source - Speech and Language Processing]]
