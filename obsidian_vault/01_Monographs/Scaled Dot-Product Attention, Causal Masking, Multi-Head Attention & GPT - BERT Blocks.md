---
type: monograph
domain: "D05"
parent_topic: "[[PT 05 - Tokenization, Embeddings, Attention & Transformer Architectures]]"
sub_topic: "[[ST 05 - Attention Mechanisms & Transformer Architectures]]"
cheatsheet: "[[Cheatsheet - Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]"
part_of: 
  - "[[ST 05 - Attention Mechanisms & Transformer Architectures]]"
  - "[[D05 - Transformers, Language Modeling & LLM Engineering]]"
concepts: 
  - "[[FlashAttention]]"
  - "[[GQA & MQA]]"
  - "[[Self-Attention & Multi-Head Attention]]"
  - "[[Transformer Architecture]]"
prerequisites: 
  - "[[BatchNorm, LayerNorm & RMSNorm]]"
  - "[[Residual Connections]]"
  - "[[Self-Attention & Multi-Head Attention]]"
enables: 
  - "[[CLIP & Vision-Language Alignment]]"
  - "[[Chinchilla Scaling Laws]]"
  - "[[FlashAttention]]"
  - "[[GQA & MQA]]"
  - "[[KV Cache & PagedAttention]]"
  - "[[LoRA & QLoRA]]"
  - "[[Transformer Architecture]]"
  - "[[Vision Transformer (ViT)]]"
related: 
  - "[[BatchNorm, LayerNorm & RMSNorm]]"
  - "[[Chinchilla Scaling Laws]]"
  - "[[FlashAttention]]"
  - "[[GQA & MQA]]"
  - "[[Graph Neural Networks (MPNN)]]"
  - "[[KV Cache & PagedAttention]]"
  - "[[Object Detection (R-CNN, SSD & DETR)]]"
  - "[[Residual Connections]]"
  - "[[RoPE & Positional Embeddings]]"
  - "[[Self-Attention & Multi-Head Attention]]"
  - "[[Transformer Architecture]]"
  - "[[Vision Transformer (ViT)]]"
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Speech and Language Processing]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-05
---

# Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT/BERT Blocks

> [!abstract] Monograph Navigation
> - **Domain:** [[D05 - Transformers, Language Modeling & LLM Engineering|Transformers, Language Modeling & LLM Engineering]]
> - **Parent / Sub-Topic:** [[PT 05 - Tokenization, Embeddings, Attention & Transformer Architectures|Tokenization, Embeddings, Attention & Transformer Architectures]] → [[ST 05 - Attention Mechanisms & Transformer Architectures|Attention Mechanisms & Transformer Architectures]]
> - **Visual Cheatsheet:** [[Cheatsheet - Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT - BERT Blocks]]
> - **Description:** Derives scaled dot-product attention (1/sqrt(d_k) variance proof), causal lower-triangular masking, Multi-Head / Grouped-Query Attention (MHA/GQA), Pre-LN residual blocks with SwiGLU/GELU MLPs, and encoder (BERT) vs. decoder (GPT) vs. encoder-decoder designs.

## Problem Formulation & Unified Notation

Given an input sequence of $N$ token embeddings, a Transformer layer computes a length-preserving contextualized representation $\tilde{\boldsymbol{X}} = \mathrm{TransformerLayer}(\boldsymbol{X}) \in \mathbb{R}^{N \times d_{\mathrm{model}}}$ without sequential recurrence [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 12, Eq. 12.1, p. 377). Unlike convolutional or recurrent layers whose receptive fields grow linearly with depth or time, self-attention establishes direct, data-dependent routing paths between every pair of positions $(i, j) \in \{1, \dots, N\}^2$ in $O(1)$ sequential steps [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 12, Sec. 12.2, p. 222).

Let $\boldsymbol{X} \in \mathbb{R}^{N \times d_{\mathrm{model}}}$ denote the row-major input matrix whose $i$-th row is $\boldsymbol{x}_i^{\top} \in \mathbb{R}^{1 \times d_{\mathrm{model}}}$. Single-head attention projects $\boldsymbol{X}$ into three distinct subspaces—queries $\boldsymbol{Q}$, keys $\boldsymbol{K}$, and values $\boldsymbol{V}$—via learned parameter matrices $\boldsymbol{W}_Q \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$, $\boldsymbol{W}_K \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$, and $\boldsymbol{W}_V \in \mathbb{R}^{d_{\mathrm{model}} \times d_v}$:

$$
\boldsymbol{Q} = \boldsymbol{X}\boldsymbol{W}_Q \in \mathbb{R}^{N \times d_k}, \qquad
  \boldsymbol{K} = \boldsymbol{X}\boldsymbol{W}_K \in \mathbb{R}^{N \times d_k}, \qquad
  \boldsymbol{V} = \boldsymbol{X}\boldsymbol{W}_V \in \mathbb{R}^{N \times d_v}.
$$

Separating $\boldsymbol{W}_Q$ and $\boldsymbol{W}_K$ breaks the symmetry of $\boldsymbol{X}\boldsymbol{U}\boldsymbol{U}^{\top}\boldsymbol{X}^{\top}$, enabling asymmetric token affinities such as "chisel" attending strongly to "tool" without forcing the converse [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 12, p. 380).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Source textbooks differ in matrix orientation (row-major vs. column-major), bias conventions, and LayerNorm placement. Table `tab:attn:notation` establishes a canonical translation across all five synthesized references.
>
> | **Concept** | **Canonical Here** | **Raschka / J&M** | **Bishop (Ch. 12)** | **Prince / Murphy** |
> | --- | --- | --- | --- | --- |
> | Input $\boldsymbol{X}$ | $\boldsymbol{X} \in \mathbb{R}^{N \times d_{\mathrm{model}}}$ (row-major) | $\mathcal{X} \in \mathbb{R}^{B \times T \times d_{\mathrm{in}}}$ (Raschka); $\boldsymbol{X} \in \mathbb{R}^{N \times d}$ (J&M) | $\boldsymbol{X} \in \mathbb{R}^{N \times D}$ (row $\boldsymbol{x}_n^{\top}$, Fig. 12.3) | $\boldsymbol{X} \in \mathbb{R}^{D \times N}$ col-major (Prince); $\boldsymbol{x}_{1:n}$ (Murphy) |
> | QKV weights | $\boldsymbol{W}_Q, \boldsymbol{W}_K, \boldsymbol{W}_V$ | `W_query`, `W_key`, `W_value`; $\boldsymbol{W}^{Q,K,V}$ | $\boldsymbol{W}^{(q)}, \boldsymbol{W}^{(k)}, \boldsymbol{W}^{(v)}$ (bias absorbed) | $\boldsymbol{\Omega}_{q,k,v}$ + $\boldsymbol{\beta}$ (Prince); $\boldsymbol{W}_i^{(q,k,v)}$ (Murphy) |
> | Head dims | $H$ heads; $d_k = d_v = d_{\mathrm{model}}/H$ | `num_heads`, `head_dim` ($d_{\mathrm{out}}/H$) | $H$ heads; $D_k, D_v = D/H$ | $H$ heads, $D_q = D/H$ (Prince); $h, p_k, p_v$ (Murphy) |
> | Softmax | Row $\mathrm{softmax}(\boldsymbol{Q}\boldsymbol{K}^{\top}/\sqrt{d_k})\boldsymbol{V}$ | Row-wise softmax over keys | Row $\mathrm{softmax}(\boldsymbol{Q}\boldsymbol{K}^{\top}/\sqrt{D_k})\boldsymbol{V}$ | Col $\boldsymbol{V}\cdot\mathrm{softmax}(\boldsymbol{K}^{\top}\boldsymbol{Q}/\sqrt{D_q})$ (Prince) |
> | Norm | Pre-LN: $\boldsymbol{X} + F(\mathrm{LN}(\boldsymbol{X}))$ | Pre-LN in GPT block (Raschka L4.6; J&M Fig. 7.7) | Post-LN (Eq. 12.20) & Pre-LN (Eq. 12.21) | Post-LN $\boldsymbol{X} \leftarrow \mathrm{LN}(\boldsymbol{X} + F(\boldsymbol{X}))$ (Prince Eq. 12.13) |

## Core Mathematical Derivations & Theorems

### Scaled Dot-Product Attention & Variance Stabilization

> [!important] Theorem: Variance Scaling of Dot-Product Logits & Softmax Jacobians
> Let $\boldsymbol{q}, \boldsymbol{k} \in \mathbb{R}^{d_k}$ be independent random vectors whose entries satisfy $\mathbb{E}[q_r] = \mathbb{E}[k_r] = 0$, $\mathrm{Var}(q_r) = \mathrm{Var}(k_r) = 1$, and $\mathrm{Cov}(q_r, q_s) = \mathrm{Cov}(k_r, k_s) = 0$ for $r \neq s$. Then the unscaled dot-product logit $s = \boldsymbol{q}^{\top} \boldsymbol{k} = \sum_{r=1}^{d_k} q_r k_r$ satisfies:
>
> $$
> \mathbb{E}[s] = \sum_{r=1}^{d_k} \mathbb{E}[q_r]\mathbb{E}[k_r] = 0, \qquad
>   \mathrm{Var}(s) = \sum_{r=1}^{d_k} \mathrm{Var}(q_r k_r) = \sum_{r=1}^{d_k} \Big(\mathbb{E}[q_r^2]\mathbb{E}[k_r^2] - (\mathbb{E}[q_r]\mathbb{E}[k_r])^2\Big) = d_k.
> $$
>
> Scaling the logits by $1/\sqrt{d_k}$ restores unit variance independent of head width: $\mathrm{Var}(\boldsymbol{q}^{\top} \boldsymbol{k} / \sqrt{d_k}) = d_k / d_k = 1$.
>
> Moreover, for any logit vector $\boldsymbol{s} \in \mathbb{R}^N$ and attention probability vector $\boldsymbol{p} = \mathrm{softmax}(\boldsymbol{s})$ with $p_i = \exp(s_i) / \sum_{m=1}^N \exp(s_m)$, the softmax Jacobian has entries
>
> $$
> \frac{\partial p_i}{\partial s_j} = p_i (\delta_{ij} - p_j),
> $$
>
> where $\delta_{ij}$ is the Kronecker delta. As $\mathrm{Var}(s)$ grows with $d_k$, the maximum logit dominates ($p_{i^*} \to 1$, $p_{j \neq i^*} \to 0$), driving $\frac{\partial p_i}{\partial s_j} \to 0$ exponentially and stalling gradient flow [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 12, Eq. 12.14, p. 382) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 12, Sec. 12.3.2, p. 228).
>
> The resulting **Scaled Dot-Product Attention** operator is:
>
> $$
> \mathrm{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \mathrm{softmax}\!\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V} \in \mathbb{R}^{N \times d_v},
> $$
>
> where $\mathrm{softmax}$ is applied row-wise so each output row $\boldsymbol{y}_i^{\top} = \sum_{j=1}^N A_{ij} \boldsymbol{v}_j^{\top}$ is a convex combination of value vectors ($A_{ij} \ge 0$, $\sum_{j=1}^N A_{ij} = 1$).

> [!tip] Attention as Differentiable Dictionary Lookup & Kernel Smoothing
> Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.4.1–15.4.2, pp. 549–551) frames attention as a soft dictionary lookup and a parametric Nadaraya–Watson kernel smoother: each query $\boldsymbol{q}_i$ evaluates a similarity kernel $\kappa(\boldsymbol{q}_i, \boldsymbol{k}_j) = \exp(\boldsymbol{q}_i^{\top} \boldsymbol{k}_j / \sqrt{d_k})$ against all keys $\{\boldsymbol{k}_j\}_{j=1}^N$, normalizes by $\sum_m \kappa(\boldsymbol{q}_i, \boldsymbol{k}_m)$ to form a partition of unity on the simplex $\Delta^{N-1}$, and returns the conditional expectation of the values $\{\boldsymbol{v}_j\}$. Prince [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 12.2, Fig. 12.1, p. 223) emphasizes that unlike fixed MLP weights, attention weights $A_{ij}$ are dynamic, input-dependent routing coefficients.

### Causal Lower-Triangular Masking & Autoregressive Factorization

Autoregressive language models factorize the joint sequence likelihood as $\mathbb{P}(t_1, \dots, t_N) = \mathbb{P}(t_1)\prod_{i=2}^N \mathbb{P}(t_i \mid t_1, \dots, t_{i-1})$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Eq. 12.15, p. 237). During parallel training on the full sequence $\boldsymbol{X}$, position $i$ must be strictly prevented from attending to future positions $j > i$.

> [!important] Causal Masking Equivalence & Zero Future-Token Leakage
> Define the upper-triangular additive causal mask $\boldsymbol{M} \in \{0, -\infty\}^{N \times N}$ with entries $M_{ij} = 0$ if $j \le i$ and $M_{ij} = -\infty$ if $j > i$. Causal scaled dot-product attention computes:
>
> $$
> \boldsymbol{A}_{\mathrm{causal}} = \mathrm{softmax}\!\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}} + \boldsymbol{M}\right).
> $$
>
> For any future position $j > i$, $\lim_{M_{ij} \to -\infty} \exp(S_{ij} + M_{ij}) = 0$, so $(\boldsymbol{A}_{\mathrm{causal}})_{ij} = 0$ and the softmax denominator sums exclusively over past and current tokens $m \in \{1, \dots, i\}$:
>
> $$
> (\boldsymbol{A}_{\mathrm{causal}})_{ij} = \frac{\exp(S_{ij}) \mathbb{I}[j \le i]}{\sum_{m=1}^{i} \exp(S_{im})}.
> $$
>
> Raschka [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 3, Sec. 3.5.1, pp. 200–205) demonstrates that post-softmax zeroing ($\boldsymbol{A} \odot \boldsymbol{L}_{\mathrm{tril}}$) followed by row-wise renormalization is mathematically identical to Eq. `(eq:attn:causal_denom)` because masked terms cancel from the ratio, but pre-softmax $-\infty$ masking achieves this in a single fused pass with zero future-token influence on the normalization constant.

> [!note] Source Figure
> *Figure:* Parallel matrix-form causal self-attention: $\boldsymbol{X}$ is projected into $\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}$; $\boldsymbol{Q}\boldsymbol{K}^{\top}$ computes all pairwise affinities simultaneously; upper-triangular $-\infty$ masking enforces causality before softmax; and multiplication by $\boldsymbol{V}$ yields contextual outputs $\boldsymbol{A}$. Source: [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 7, Fig. 7.11, PDF p. 198). (`speech_language_processing_aug26_p198_fig2.png`)

### Multi-Head Attention, GQA, and Pre-LN Transformer Blocks

A single attention head averages over all token relationships with one similarity metric, whereas natural language requires tracking syntactic dependencies, coreference, and local phrasing simultaneously [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 12.1.6, p. 383).

> [!important] Multi-Head Attention (MHA) & Fused Tensor Splitting
> With $H$ parallel heads ($h = 1, \dots, H$), each head learns independent projections $\boldsymbol{W}_{Q,h}, \boldsymbol{W}_{K,h} \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$ and $\boldsymbol{W}_{V,h} \in \mathbb{R}^{d_{\mathrm{model}} \times d_v}$ (typically $d_k = d_v = d_{\mathrm{model}}/H$):
>
> $$
> \begin{aligned}
> \boldsymbol{H}_h &= \mathrm{Attention}\!\left(\boldsymbol{X}\boldsymbol{W}_{Q,h},  \boldsymbol{X}\boldsymbol{W}_{K,h},  \boldsymbol{X}\boldsymbol{W}_{V,h}\right) \in \mathbb{R}^{N \times d_v},  \\
>   \mathrm{MHA}(\boldsymbol{X}) &= \mathrm{Concat}(\boldsymbol{H}_1, \dots, \boldsymbol{H}_H) \boldsymbol{W}_O \in \mathbb{R}^{N \times d_{\mathrm{model}}},
> \end{aligned}
> $$
>
> where $\boldsymbol{W}_O \in \mathbb{R}^{(H d_v) \times d_{\mathrm{model}}}$ mixes subspaces across heads [[Source - Foundations|deep_learning_foundations_concepts]] (Eq. 12.15–12.19, p. 383) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. 15.55–15.56, p. 557). In practice [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Listing 3.5, pp. 227–232), all $H$ projections are fused into single dense layers $\boldsymbol{W}_Q, \boldsymbol{W}_K, \boldsymbol{W}_V \in \mathbb{R}^{d_{\mathrm{model}} \times (H d_k)}$ and reshaped via $[B, T, d_{\mathrm{model}}] \to [B, T, H, d_k] \xrightarrow{\mathrm{transpose}} [B, H, T, d_k]$.
>
> **Grouped-Query Attention (GQA) & Cross-Attention:** To shrink the autoregressive KV-cache by a factor of $H / H_{\mathrm{kv}}$, GQA partitions the $H$ query heads into $H_{\mathrm{kv}}$ groups that share key-value heads ($H_{\mathrm{kv}}=1$ recovers Multi-Query Attention, MQA). In encoder-decoder cross-attention [[Source - Foundations|deep_learning_foundations_concepts]] (Fig. 12.19, p. 407), queries come from decoder states ($\boldsymbol{Q} = \boldsymbol{X}_{\mathrm{dec}}\boldsymbol{W}_Q$) while keys and values come from the encoder output ($\boldsymbol{K} = \boldsymbol{Z}_{\mathrm{enc}}\boldsymbol{W}_K$, $\boldsymbol{V} = \boldsymbol{Z}_{\mathrm{enc}}\boldsymbol{W}_V$).

Because attention outputs are convex combinations of linearly transformed inputs, a Transformer layer pairs MHA with a position-wise nonlinear $\mathrm{MLP}$ (GELU in GPT-2 [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Sec. 4.3, p. 274) or gated SwiGLU in modern LLMs), residual shortcuts, and LayerNorm (or RMSNorm). While the original Transformer [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Eq. 12.13, p. 230) used **Post-LN**, modern LLMs adopt **Pre-LN** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Listing 4.6, pp. 292–293) [[Source - Foundations|deep_learning_foundations_concepts]] (Eq. 12.21–12.23, pp. 385–386) so the residual highway remains unscaled:

$$
\begin{aligned}
\boldsymbol{Z}^{(\ell)} &= \boldsymbol{X}^{(\ell)} + \mathrm{MHA}\!\left(\mathrm{LN}_1\!\left(\boldsymbol{X}^{(\ell)}\right)\right),  \\
  \boldsymbol{X}^{(\ell+1)} &= \boldsymbol{Z}^{(\ell)} + \mathrm{MLP}\!\left(\mathrm{LN}_2\!\left(\boldsymbol{Z}^{(\ell)}\right)\right).
\end{aligned}
$$

At the output of layer $L$, a final normalization feeds the language modeling head, which often ties the unembedding matrix to the transpose of the token embedding matrix: $\boldsymbol{U} = \boldsymbol{E}^{\top} \in \mathbb{R}^{d_{\mathrm{model}} \times |V|}$, yielding logits $\boldsymbol{u}_i = \boldsymbol{h}_i^{(L)\top}\boldsymbol{E}^{\top}$ and next-token probabilities $\boldsymbol{y}_i = \mathrm{softmax}(\boldsymbol{u}_i)$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Eq. 7.47–7.48, p. 203).

> [!note] Source Figure
> *Figure:* Full encoder–decoder Transformer architecture with zoom-ins on Multi-Head Attention and Scaled Dot-Product Attention (optional causal mask). Source: [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.5.4, Fig. 15.26, PDF p. 560). (`probabilisticml_kevinmurphy_introduction_p560_fig1.png`)

## Algorithmic Implementation & Comparative Synthesis

Algorithm `alg:attn:causal_mha_preln` consolidates the batched tensor-splitting MHA implementation [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Listing 3.5) and Pre-LN Transformer block [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Listing 4.6) with optional causal masking.

> [!example] Algorithm: Batched Causal Multi-Head Self-Attention & Pre-LN Transformer Block
> **Input:** Input tensor $\mathcal{X} \in \mathbb{R}^{B \times T \times d}$; weights $\boldsymbol{W}_Q, \boldsymbol{W}_K, \boldsymbol{W}_V, \boldsymbol{W}_O \in \mathbb{R}^{d \times d}$; head count $H$ with $d_k = d / H$; causal flag $\mathrm{is\_causal} \in \{\mathrm{true}, \mathrm{false}\}$; dropout rate $p$.
> **Output:** Transformed tensor $\mathcal{Y} \in \mathbb{R}^{B \times T \times d}$.
> $\tilde{\mathcal{X}} \leftarrow \mathrm{LayerNorm}_1(\mathcal{X})$  *// Pre-LN preserves unscaled residual highway*
> $\mathcal{Q}, \mathcal{K}, \mathcal{V} \leftarrow \tilde{\mathcal{X}}\boldsymbol{W}_Q, \tilde{\mathcal{X}}\boldsymbol{W}_K, \tilde{\mathcal{X}}\boldsymbol{W}_V$  *// Shape: $[B, T, d]$*
> Reshape & transpose $\mathcal{Q}, \mathcal{K}, \mathcal{V}$ to $[B, H, T, d_k]$
> $\mathcal{S} \leftarrow (\mathcal{Q}   \mathcal{K}^{\top}) / \sqrt{d_k}$  *// Pairwise scaled logits: $[B, H, T, T]$*
> **if** $\mathrm{is\_causal}$:
>
>   Set $S_{b,h,i,j} \leftarrow -\infty$ for all $j > i$  *// Upper-triangular causal mask*
>
> $\mathcal{A} \leftarrow \mathrm{Dropout}_p\!\big(\mathrm{softmax}(\mathcal{S}, \mathrm{dim}=-1)\big)$  *// Row-wise attention weights*
> $\mathcal{H} \leftarrow \mathcal{A}   \mathcal{V}$  *// Shape: $[B, H, T, d_k]$*
> Transpose & reshape $\mathcal{H}$ back to $[B, T, d]$, then $\mathcal{O} \leftarrow \mathcal{H}\boldsymbol{W}_O$
> $\mathcal{Z} \leftarrow \mathcal{X} + \mathrm{Dropout}_p(\mathcal{O})$  *// First residual shortcut*
> $\mathcal{Y} \leftarrow \mathcal{Z} + \mathrm{Dropout}_p\!\big(\mathrm{MLP}(\mathrm{LayerNorm}_2(\mathcal{Z}))\big)$  *// Position-wise GELU/SwiGLU + shortcut*
> **return** $\mathcal{Y}$

### Computational Complexity & Hardware-Aware Variants

Standard self-attention incurs $O(N^2 d_{\mathrm{model}} + N d_{\mathrm{model}}^2)$ FLOPs and materializes an $O(H N^2)$ attention matrix in GPU high-bandwidth memory (HBM) [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 12.1.8, p. 386). Table `tab:attn:complexity` compares exact and approximate attention regimes synthesized in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.5.5, 15.6, pp. 561–566) and [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 12.9, p. 241).

**Table:** Complexity comparison of exact and sub-quadratic attention mechanisms ($d = d_{\mathrm{model}}$, $M$ = random feature dimension) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.5.5, 15.6, pp. 561–566).

| **Attention Variant** | **Time (FLOPs)** | **HBM Memory** | **Seq. Path** | **Mechanism & Trade-off** |
| --- | --- | --- | --- | --- |
| Dense Scaled MHA | $O(N^2 d + N d^2)$ | $O(H N^2 + N d)$ | $O(1)$ | Exact global routing; HBM-bound for large $N$. |
| FlashAttention (Tiled) | $O(N^2 d + N d^2)$ | $O(N d)$ | $O(1)$ | Exact; fuses QK^T–softmax–V in SRAM via online softmax rescaling. |
| Local / Strided Sparse | $O(N w d + N d^2)$ | $O(H N w + N d)$ | $O(N/w)$ | Window size $w \ll N$; loses arbitrary-range single-hop mixing. |
| LSH (Reformer) | $O(N d \log N)$ | $O(N d \log N)$ | $O(1)$ | Locality-sensitive hashing buckets (Murphy Sec. 15.6.2); requires $\boldsymbol{Q}=\boldsymbol{K}$. |
| Kernel / Random Feat. (Performer) | $O(N M d)$ | $O(N M + M d)$ | $O(1)$ | Positive random features $\phi(\boldsymbol{q})^{\top}\phi(\boldsymbol{k})$ (Murphy Eq. 15.66–15.68); approximate. |

> [!info] Cross-Book Synthesis: GPT (Decoder) vs. BERT (Encoder) vs. Seq2Seq & ViT
> Across [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 12.3–12.4), [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 12.6–12.10), and [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 7 & 9), the same Transformer block specializes into four canonical macro-architectures:
>
> | **Family** | **Canonical Model** | **Attention Mask** | **Pretraining Obj.** | **Primary Strength & Caveat** |
> | --- | --- | --- | --- | --- |
> | Decoder-only | GPT-2 / GPT-3 / LLaMA | Causal ($j \le i$) | Next-token prediction (NTP) | Uses 100\% of tokens per step; supports KV-cache generation; unidirectional context only. |
> | Encoder-only | BERT / RoBERTa | Bidirectional (full) | Masked LM (15\% tokens) | Rich bidirectional representations for classification/NER; cannot autoregressively generate. |
> | Encoder–Decoder | Original Transformer / T5 | Enc: full; Dec: causal + cross | Span corruption / Seq2Seq MLE | Decouples source encoding from target generation via cross-attention. |
> | Vision (ViT) | ViT-B/16 (Bishop Fig. 12.22) | Bidirectional (+ `[CLS]`) | Supervised / MIM on $P \times P$ patches | Global spatial mixing from layer 1; lacks CNN translation equivariance inductive bias. |

> [!note] Source Figure
> *Figure:* Stacked GPT-style decoder Transformer: shifted token embeddings plus positional encodings pass through $L$ masked Transformer layers, followed by shared linear-softmax (LSM) language modeling heads. Source: [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 12.3.1, Fig. 12.15, p. 400). (`deep_learning_foundations_concepts_p400_fig1.png`)

## Visual Diagnostics & Geometric Structure

Figure `fig:attn:mha_gpt_block` contrasts the tensor routing of a Pre-LN GPT decoder block (causal mask, unscaled residual stream) against a bidirectional BERT encoder block.

> [!example] Visual Diagram (LaTeX / PDF)
> *Architectural comparison of a Pre-LN GPT decoder block (left, unscaled residual stream + upper-triangular $-\infty$ causal mask) versus a Post-LN bidirectional BERT encoder block (right, full $T \times T$ attention).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:attn:scaling_softmax_variance` quantifies both the variance-stabilizing role of $1/\sqrt{d_k}$ scaling on softmax sensitivity and the HBM memory savings of SRAM-tiled FlashAttention over naive $O(H N^2)$ materialization.

> [!example] Visual Diagram (LaTeX / PDF)
> *Left: Unscaled dot-product logits grow in standard deviation as $\sqrt{d_k}$, driving softmax into gradient-vanishing saturation, whereas $1/\sqrt{d_k}$ scaling preserves unit variance. Right: Log-log HBM memory scaling ($B=1, H=32, d_{\mathrm{model}}=4096$, FP16) showing quadratic blowup of naive $N \times N$ score materialization versus linear SRAM-tiled FlashAttention.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Pitfall 1: Unscaled Logits & Softmax Saturation in Wide Heads
> Omitting the $1/\sqrt{d_k}$ factor causes $\mathrm{Var}(\boldsymbol{q}^{\top}\boldsymbol{k}) = d_k$ to grow linearly with head width [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 12.1.5, p. 382) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Sec. 12.3.2, p. 228). In FP16/BF16 mixed precision, large logits push $\mathrm{softmax}$ into near-one-hot distributions where $\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j) \approx 0$ and $\exp(S_{ij})$ risks overflow. **Remedy:** Always scale by $1/\sqrt{d_k}$ and compute the softmax log-sum-exp reduction in FP32 after subtracting $\max_j S_{ij}$ row-wise.

> [!warning] Pitfall 2: Post-Softmax Masking Leakage & All-Masked `NaN`s
> Zeroing out upper-triangular entries *after* $\mathrm{softmax}$ breaks row stochasticity ($\sum_j A_{ij} < 1$) unless explicitly renormalized [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Sec. 3.5.1, pp. 200–204). Conversely, when combining causal masks with padding masks via $-\infty$, if every key in a row is masked ($S_{ij} = -\infty  \forall j$), $\exp(-\infty)/\sum \exp(-\infty) = 0/0$ produces `NaN` gradients across the batch. **Remedy:** Apply $-\infty$ (or a large negative finite sentinel in FP16) *before* $\mathrm{softmax}$, and ensure every non-padding query attends to at least itself ($j=i$).

> [!warning] Pitfall 3: Post-LN Warmup Instability & KV-Cache Memory Wall
> Post-LN (Eq. 12.20 in [[Source - Foundations|deep_learning_foundations_concepts]] (p. 385)) passes the residual highway through LayerNorm at every block, attenuating or amplifying gradients in deep stacks and requiring delicate learning-rate warmup [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Sec. 4.5, p. 293). During autoregressive inference, caching $\boldsymbol{K}, \boldsymbol{V}$ across $L$ layers consumes $2 B L H_{\mathrm{kv}} N d_k \cdot b_{\mathrm{dtype}}$ bytes, quickly saturating GPU HBM. **Remedy:** Use Pre-LN (or RMSNorm) for stable deep training, and pair GQA ($H_{\mathrm{kv}} \ll H$) with FlashAttention tiling to eliminate $O(H N^2)$ training memory.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (4):** [[FlashAttention]], [[GQA & MQA]], [[Self-Attention & Multi-Head Attention]], [[Transformer Architecture]]
- **Prerequisites (Directed $\leftarrow$):** [[BatchNorm, LayerNorm & RMSNorm]], [[Residual Connections]], [[Self-Attention & Multi-Head Attention]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[CLIP & Vision-Language Alignment]], [[Chinchilla Scaling Laws]], [[FlashAttention]], [[GQA & MQA]], [[KV Cache & PagedAttention]], [[LoRA & QLoRA]], [[Transformer Architecture]], [[Vision Transformer (ViT)]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[BatchNorm, LayerNorm & RMSNorm]], [[Chinchilla Scaling Laws]], [[FlashAttention]], [[GQA & MQA]], [[Graph Neural Networks (MPNN)]], [[KV Cache & PagedAttention]], [[Object Detection (R-CNN, SSD & DETR)]], [[Residual Connections]], [[RoPE & Positional Embeddings]], [[Self-Attention & Multi-Head Attention]], [[Transformer Architecture]], [[Vision Transformer (ViT)]]
- **Synthesized Sources:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Speech and Language Processing]], [[Source - Understanding Deep Learning]]
