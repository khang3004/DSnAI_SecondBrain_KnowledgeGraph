---
type: monograph
domain: "D05"
parent_topic: "[[PT 05 - LLM Pretraining, Adaptation, Prompting & Efficient Inference]]"
sub_topic: "[[ST 05 - Pretraining, Scaling Laws & Parameter-Efficient Fine-Tuning]]"
cheatsheet: "[[Cheatsheet - Self-Supervised Pretraining, Scaling Laws, Instruction Fine-Tuning & LoRA - QLoRA]]"
part_of: 
  - "[[ST 05 - Pretraining, Scaling Laws & Parameter-Efficient Fine-Tuning]]"
  - "[[D05 - Transformers, Language Modeling & LLM Engineering]]"
concepts: 
  - "[[Chinchilla Scaling Laws]]"
  - "[[LoRA & QLoRA]]"
prerequisites: 
  - "[[Low-Rank Approximation (Eckart-Young)]]"
  - "[[Singular Value Decomposition (SVD)]]"
  - "[[Transformer Architecture]]"
enables: []
related: 
  - "[[Chinchilla Scaling Laws]]"
  - "[[DPO (Direct Preference Optimization)]]"
  - "[[LoRA & QLoRA]]"
  - "[[Low-Rank Approximation (Eckart-Young)]]"
  - "[[RLHF (KL-Regularized)]]"
  - "[[Singular Value Decomposition (SVD)]]"
  - "[[Transformer Architecture]]"
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Foundations of Large Language Models]]"
  - "[[Source - LLM Engineer’s Handbook]]"
  - "[[Source - Reinforcement Learning from Human Feedback]]"
  - "[[Source - Speech and Language Processing]]"
tags:
  - monograph
  - domain-05
---

# Self-Supervised Pretraining, Scaling Laws, Instruction Fine-Tuning & LoRA/QLoRA

> [!abstract] Monograph Navigation
> - **Domain:** [[D05 - Transformers, Language Modeling & LLM Engineering|Transformers, Language Modeling & LLM Engineering]]
> - **Parent / Sub-Topic:** [[PT 05 - LLM Pretraining, Adaptation, Prompting & Efficient Inference|LLM Pretraining, Adaptation, Prompting & Efficient Inference]] → [[ST 05 - Pretraining, Scaling Laws & Parameter-Efficient Fine-Tuning|Pretraining, Scaling Laws & Parameter-Efficient Fine-Tuning]]
> - **Visual Cheatsheet:** [[Cheatsheet - Self-Supervised Pretraining, Scaling Laws, Instruction Fine-Tuning & LoRA - QLoRA]]
> - **Description:** Covers causal LM cross-entropy and perplexity, MLM/denoising objectives, Kaplan vs. Chinchilla compute-optimal scaling laws, distributed parallelism (DP/TP/PP/ZeRO), SFT chat templates, and low-rank adaptation (LoRA/QLoRA).

## Problem Formulation & Unified Notation

Modern large language models (LLMs) are trained through a two-stage paradigm: (1) *self-supervised pretraining* on trillions of unlabeled tokens to acquire broad linguistic and world knowledge, followed by (2) *supervised fine-tuning* (SFT) and *parameter-efficient fine-tuning* (PEFT) to align next-token continuation behavior with structured instruction following [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 1–2, pp. 7–65) and [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 4, pp. 33–38).

Let $\mathcal{V}$ denote a discrete token vocabulary of size $\lvert \mathcal{V} \rvert$, and let $\mathcal{D} = \{\boldsymbol{x}^{(k)}\}_{k=1}^{M}$ denote a pretraining corpus where each sequence $\boldsymbol{x} = (x_0, x_1, \dots, x_m) \in \mathcal{V}^{m+1}$ is modeled by a neural network with parameters $\theta \in \mathbb{R}^N$. In autoregressive *Causal Language Modeling* (CLM), the joint probability factorizes via the chain rule as

$$
\mathbb{P}_\theta(\boldsymbol{x}) = \prod_{i=0}^{m-1} \mathbb{P}_\theta(x_{i+1} \mid x_0, \dots, x_i).
$$

In the post-training SFT stage, we are given an instruction dataset $\mathcal{D}_{\text{SFT}} = \{(\boldsymbol{c}^{(k)}, \boldsymbol{y}^{(k)})\}_{k=1}^{M_{\text{SFT}}}$ formatted via a chat template (e.g., ChatML or Alpaca), where $\boldsymbol{c}$ is the prompt/context prefix and $\boldsymbol{y}$ is the target assistant response [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, pp. 208–209) [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 4, pp. 33–35). Rather than updating all $N$ parameters of every linear projection $\boldsymbol{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, Low-Rank Adaptation (LoRA) freezes $\boldsymbol{W}_0$ (or its 4-bit NormalFloat quantized counterpart $\tilde{\boldsymbol{W}}_0$ in QLoRA) and optimizes a low-rank residual $\Delta \boldsymbol{W} = \frac{\alpha}{r}\boldsymbol{B}\boldsymbol{A}$ with $r \ll \min(d_{\text{in}}, d_{\text{out}})$ [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (App. E, pp. 777–795) [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, pp. 213–215).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> The five primary sources adopt distinct indexing conventions, logarithm bases, and matrix orientations across pretraining, scaling laws, and LoRA. Table `tab:pretrain:notation` establishes a unified notation used throughout this monograph.
>
>
>
>
> | Concept | Source Conventions | Unified Monograph Notation |
> | — | — | — |
> | Token Sequence | $x_0, \dots, x_m$ [[Source - Foundations of Large Language Models]] (Ch. 1, p. 7); $w_1, \dots, w_N$ [[Source - Speech and Language Processing]] (Ch. 3, p. 76); `inputs`/`targets` (Raschka) | $\boldsymbol{x} = (x_0, \dots, x_m) \in \mathcal{V}^{m+1}$ |
> | Perplexity Base | $\text{PPL}(W) = 2^{H(p,m)}$ (base 2) [[Source - Speech and Language Processing]] (Ch. 3, p. 88) vs. $\exp(\mathcal{L}_{\text{CE}})$ (base $e$) [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]] (Ch. 5, p. 348) | $\text{PPL}(\boldsymbol{x}) = \exp(\mathcal{L}_{\text{CE}})$ (nats) |
> | Scaling Law | $L(N,D) = aN^b + cD^d + \epsilon_\infty$ ($b,d < 0$) [[Source - Foundations of Large Language Models]] (Ch. 2, p. 65) | $L(N,D) = E + A N^{-\alpha} + B D^{-\beta}$ ($\alpha,\beta > 0$) |
> | LoRA Orientation | Row-vector $\boldsymbol{x}\boldsymbol{A}\boldsymbol{B}$ with $\boldsymbol{A}\in\mathbb{R}^{d_{\text{in}}\times r}$ [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]] (App. E, p. 792) vs. column-vector [[Source - LLM Engineer’s Handbook]] (Ch. 5, p. 214) | Column-vector $\boldsymbol{h} = \boldsymbol{W}_0\boldsymbol{x} + \frac{\alpha}{r}\boldsymbol{B}\boldsymbol{A}\boldsymbol{x}$, $\boldsymbol{A}\in\mathbb{R}^{r\times d_{\text{in}}}$, $\boldsymbol{B}\in\mathbb{R}^{d_{\text{out}}\times r}$ |
> | Prompt Masking | `ignore\_index=-100` [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]] (Ch. 7, p. 539); user-turn vs. final-turn [[Source - Reinforcement Learning from Human Feedback]] (Ch. 4, p. 37) | Binary mask $m_i \in \{0,1\}$ (target $-100$ when $m_i = 0$) |

## Core Mathematical Derivations & Theorems

### Information-Theoretic Bridge: Cross-Entropy, Perplexity, and Pretraining Objectives

> [!important] Causal LM Cross-Entropy, Maximum Likelihood, and Perplexity
> Let $p^{\text{gold}}_{i+1} \in \{0,1\}^{\lvert \mathcal{V} \rvert}$ denote the one-hot indicator of the ground-truth next token $x_{i+1}$, and let $p_{\theta, i+1} = \mathrm{softmax}(\boldsymbol{z}_{i}) \in \Delta^{\lvert \mathcal{V} \rvert-1}$ denote the decoder's predicted probability distribution given prefix $x_{\le i} = (x_0, \dots, x_i)$. The sequence cross-entropy loss is [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 1, pp. 7–8) [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 5, pp. 334–348)
>
> $$
> \mathcal{L}_{\text{CE}}(\theta; \boldsymbol{x})
>   = \frac{1}{m}\sum_{i=0}^{m-1} H\!\left(p^{\text{gold}}_{i+1},  p_{\theta, i+1}\right)
>   = -\frac{1}{m}\sum_{i=0}^{m-1} \log \mathbb{P}_\theta(x_{i+1} \mid x_0, \dots, x_i).
>
> $$
>
> Minimizing $\sum_{\boldsymbol{x}\in\mathcal{D}} \mathcal{L}_{\text{CE}}(\theta; \boldsymbol{x})$ is algebraically identical to Maximum Likelihood Estimation (MLE):
>
> $$
> \hat{\theta}_{\text{MLE}}
>   = \arg\min_\theta \sum_{\boldsymbol{x}\in\mathcal{D}} \mathcal{L}_{\text{CE}}(\theta; \boldsymbol{x})
>   = \arg\max_\theta \sum_{\boldsymbol{x}\in\mathcal{D}} \sum_{i=0}^{m-1} \log \mathbb{P}_\theta(x_{i+1} \mid x_0, \dots, x_i).
>
> $$
>
> By the Shannon–McMillan–Breiman theorem for a stationary ergodic language process $p$, the empirical cross-entropy rate of a model $m = \mathbb{P}_\theta$ converges to $H(p, m) = -\lim_{n\to\infty} \frac{1}{n}\log m(w_1\dots w_n) \ge H(p)$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 3, p. 87). Exponentiating Eq. `(eq:pretrain:ce_loss)` yields *perplexity* (PPL), the length-normalized inverse probability of the sequence [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 3, p. 76) [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 5, p. 348):
>
> $$
> \text{PPL}_\theta(\boldsymbol{x})
>   = \exp\!\big(\mathcal{L}_{\text{CE}}(\theta; \boldsymbol{x})\big)
>   = \left(\prod_{i=0}^{m-1} \mathbb{P}_\theta(x_{i+1} \mid x_0, \dots, x_i)\right)^{-\frac{1}{m}}.
>
> $$

> [!tip] Perplexity as Effective Vocabulary Branching Factor
> If a model assigns uniform probability $1/K$ to $K$ plausible next tokens at every step, then $\mathbb{P}_\theta(x_{i+1}\mid x_{\le i}) = 1/K$ and $\text{PPL}_\theta(\boldsymbol{x}) = (K^{-m})^{-1/m} = K$. Thus, perplexity measures the *weighted average branching factor* of the model's uncertainty [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 3, p. 77). At initialization over a GPT-2 vocabulary of $\lvert \mathcal{V} \rvert = 50{,}257$, $\mathcal{L}_{\text{CE}} \approx \ln(50{,}257) \approx 10.82$ nats, giving $\text{PPL} \approx 48{,}726$; the untrained model is effectively guessing uniformly across nearly the entire vocabulary [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 5, p. 348).

Beyond autoregressive CLM, self-supervised pretraining encompasses *Masked Language Modeling* (MLM) for bidirectional encoders (e.g., BERT) and *Permuted/Denoising* objectives for encoder–decoder models (e.g., T5, BART) [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 1, pp. 9–21). In MLM, a random subset $\mathcal{M} \subset \{0, \dots, m\}$ (typically $15\%$ of tokens) is corrupted into $\tilde{\boldsymbol{x}}$ (80\% `[MASK]`, 10\% random, 10\% unchanged) and reconstructed using full bidirectional context:

$$
\mathcal{L}_{\text{MLM}}(\theta; \boldsymbol{x})
  = -\frac{1}{\lvert \mathcal{M} \rvert} \sum_{i \in \mathcal{M}} \log \mathbb{P}_\theta(x_i \mid \tilde{\boldsymbol{x}}).
$$

As illustrated in Figure `fig:pretrain:clm_mlm_plm`, CLM enforces a lower-triangular causal attention mask so each token attends strictly to its prefix, whereas MLM permits unmasked bidirectional attention across all tokens at the cost of a pretrain–finetune discrepancy from `[MASK]` tokens during downstream adaptation [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 1, pp. 13–20).

> [!note] Source Figure
> *Figure:* Self-attention visibility masks comparing Causal Language Modeling (lower-triangular prefix conditioning), Masked Language Modeling (bidirectional conditioning over corrupted positions), and Permuted Language Modeling [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 1, Fig. 1.3, p. 13 (PDF p. 20)). (`foundations_large_language_models_p20_fig1.png`)

### Compute-Optimal Scaling Laws: Kaplan vs. Chinchilla

Empirical scaling laws relate test cross-entropy loss $L(N, D)$ to non-embedding parameter count $N$, training token count $D$, and compute budget $C \approx 6ND$ FLOPs (accounting for one forward pass of $2ND$ FLOPs and one backward pass of $4ND$ FLOPs) [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 2, pp. 63–65).

> [!important] Lagrangian Derivation of the Chinchilla Compute-Optimal Frontier
> Following Hoffmann et al. (2022) as presented in [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 2, p. 65), test loss decomposes into an irreducible entropy floor $E$ plus power-law capacity and data terms:
>
> $$
> L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta},
>
> $$
>
> with empirically fitted constants $E = 1.69$, $A = 406.4$, $\alpha = 0.34$, $B = 410.7$, and $\beta = 0.28$. Given a fixed compute budget $C = 6ND$, we minimize $L(N,D)$ via the Lagrangian
>
> $$
> \Lambda(N, D, \lambda) = E + A N^{-\alpha} + B D^{-\beta} + \lambda (6ND - C).
>
> $$
>
> Setting $\frac{\partial \Lambda}{\partial N} = 0$ and $\frac{\partial \Lambda}{\partial D} = 0$ yields $-\alpha A N^{-\alpha-1} + 6\lambda D = 0$ and $-\beta B D^{-\beta-1} + 6\lambda N = 0$. Multiplying the first by $N$ and the second by $D$ equates both to $6\lambda ND = \lambda C$:
>
> $$
> \alpha A N^{-\alpha} = \beta B D^{-\beta}.
>
> $$
>
> Substituting $D = C/(6N)$ into Eq. `(eq:pretrain:balance)` gives the closed-form compute-optimal allocation:
>
> $$
> N^*(C) = G \left(\frac{C}{6}\right)^{a}, \qquad
>   D^*(C) = G^{-1} \left(\frac{C}{6}\right)^{b},
>
> $$
>
> where $G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}$, $a = \frac{\beta}{\alpha+\beta} \approx 0.452$, and $b = \frac{\alpha}{\alpha+\beta} \approx 0.548$ (yielding the symmetric Chinchilla rule-of-thumb $N^* \propto C^{0.5}$, $D^* \propto C^{0.5}$, or $\approx 20$ tokens per parameter).

In contrast, Kaplan et al. (2020) held learning-rate schedules fixed across token horizons and inferred $N^* \propto C^{0.73}$ and $D^* \propto C^{0.27}$, which led early LLMs (e.g., GPT-3 175B on 300B tokens) to be severely undertrained in tokens [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 2, pp. 64–65). Figure `fig:pretrain:chinchilla_scaling` plots the Chinchilla loss surface and isoFLOP frontier.

> [!example] Visual Diagram (LaTeX / PDF)
> *IsoFLOP test loss curves $L(N, C/(6N))$ under the Chinchilla parametric law [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 2, Eq. (2.39), p. 65). Left of the crimson compute-optimal frontier ($N < N^*$), models are capacity-bottlenecked (overtrained in tokens, favored for inference-heavy deployment); right of the frontier ($N > N^*$), models are data-starved (undertrained).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

### Low-Rank Adaptation (LoRA) and 4-Bit NormalFloat Quantization (QLoRA)

Full fine-tuning of an LLM updates all $N$ weights, requiring FP32 master weights, gradients, and AdamW first/second moments ($\approx 16$ bytes/parameter, or $112\text{ GB}$ for a 7B model before activations) [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, pp. 211–212). LoRA exploits the empirical observation that task-specific weight updates $\Delta \boldsymbol{W}$ exhibit low intrinsic rank, as detailed in [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (App. E, pp. 776–779).

> [!important] LoRA Forward Pass, Zero-Perturbation Initialization, and Gradient Flow
> For any linear layer $\boldsymbol{h}_0 = \boldsymbol{W}_0 \boldsymbol{x}$ with frozen pretrained matrix $\boldsymbol{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, LoRA introduces trainable low-rank factors $\boldsymbol{A} \in \mathbb{R}^{r \times d_{\text{in}}}$ and $\boldsymbol{B} \in \mathbb{R}^{d_{\text{out}} \times r}$ with $r \ll \min(d_{\text{in}}, d_{\text{out}})$ and scaling hyperparameter $\alpha > 0$:
>
> $$
> \boldsymbol{h} = \boldsymbol{W}_0 \boldsymbol{x} + \frac{\alpha}{r}\boldsymbol{B}\boldsymbol{A}\boldsymbol{x}.
>
> $$
>
> At initialization ($t=0$), $\boldsymbol{A}$ is sampled from $\mathcal{N}(0, \sigma^2)$ (or Kaiming uniform) while $\boldsymbol{B} = \boldsymbol{0}$, guaranteeing $\Delta \boldsymbol{W}_0 = \frac{\alpha}{r}\boldsymbol{B}\boldsymbol{A} = \boldsymbol{0}$ so the adapted model reproduces the pretrained outputs identically before step 1 [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (App. E, pp. 792–795); see also [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, p. 214). Differentiating the loss $\mathcal{L}$ with respect to $(\boldsymbol{A}, \boldsymbol{B})$ yields
>
> $$
> \frac{\partial \mathcal{L}}{\partial \boldsymbol{B}} = \frac{\alpha}{r} \left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{h}}\right) (\boldsymbol{A}\boldsymbol{x})^{\top}, \qquad
>   \frac{\partial \mathcal{L}}{\partial \boldsymbol{A}} = \frac{\alpha}{r} \boldsymbol{B}^{\top} \left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{h}}\right) \boldsymbol{x}^{\top}.
>
> $$
>
> Because $\boldsymbol{A} \neq \boldsymbol{0}$, $\frac{\partial \mathcal{L}}{\partial \boldsymbol{B}} \neq \boldsymbol{0}$ at $t=0$, allowing $\boldsymbol{B}$ to move off zero on the first optimizer step and subsequently backpropagate non-zero gradients into $\boldsymbol{A}$. Trainable parameters per adapted matrix drop from $d_{\text{out}} d_{\text{in}}$ to $r(d_{\text{in}} + d_{\text{out}})$—for example, targeting all linear projections of Llama 3 8B with $r=16$ requires only $42\text{M}$ trainable parameters ($0.5196\%$ of $8\text{B}$) [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, p. 215). At inference, $\boldsymbol{W}_{\text{eff}} = \boldsymbol{W}_0 + \frac{\alpha}{r}\boldsymbol{B}\boldsymbol{A}$ merges into a single matrix with zero added latency [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (App. E, p. 779).

QLoRA extends LoRA by compressing the frozen base weights $\boldsymbol{W}_0$ into 4-bit *NormalFloat* (NF4), an information-theoretically optimal quantile data type for zero-mean normally distributed pretrained weights [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, p. 215). To eliminate the $0.5\text{ bits/param}$ overhead of FP32 blockwise quantization scales, QLoRA applies *double quantization* (quantizing the first-level scaling constants to FP8) and pairs this with *paged AdamW 8-bit* optimizers that offload optimizer state spikes to CPU RAM via unified memory. During forward/backward passes, NF4 blocks are dequantized on the fly to BF16 to compute $\tilde{\boldsymbol{W}}_0\boldsymbol{x}$ alongside the BF16 LoRA branch $\frac{\alpha}{r}\boldsymbol{B}\boldsymbol{A}\boldsymbol{x}$, cutting 7B fine-tuning memory from $15.6\text{ GB}$ (LoRA) to $9.3\text{ GB}$ (QLoRA) at a $\sim 30\%$ throughput cost [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, p. 215). Figure `fig:pretrain:sft_lora_qlora_arch` contrasts Full Fine-Tuning, LoRA, and QLoRA at the module level, while Figure `fig:pretrain:lora_qlora_tikz` details the joint QLoRA forward/backward dataflow and SFT loss-masking pipeline.

> [!note] Source Figure
> *Figure:* Module-level comparison of Full Fine-Tuning (all FP16 weights updated), LoRA (frozen FP16 base $\boldsymbol{W}$ plus trainable low-rank adapters $\boldsymbol{A}, \boldsymbol{B}$), and QLoRA (frozen 4-bit NF4 quantized $\tilde{\boldsymbol{W}}$ plus BF16 adapters) [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, Fig. 5.9, p. 211 (PDF p. 240)). (`llm_engineer_s_handbook_p240_fig1.png`)

> [!example] Visual Diagram (LaTeX / PDF)
> *Unified SFT loss-masking and QLoRA computation graph. Prompt and user-turn tokens are masked with $-100$ before sequence packing, while forward/backward passes dequantize $\tilde{\boldsymbol{W}}_0$ to BF16 and restrict optimizer states solely to $(\boldsymbol{A}, \boldsymbol{B})$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Algorithmic Implementation & Comparative Synthesis

### Instruction-Masked SFT with QLoRA and Gradient Checkpointing

In instruction fine-tuning, raw JSONL pairs (Alpaca, ShareGPT, or OpenAI schemas) are serialized via a chat template with explicit role delimiters (e.g., `<|im\_start|>system`, `user`, `assistant`, `<|im\_end|>`) [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, pp. 208–209). To prevent the model from wasting capacity memorizing user prompts, target tokens corresponding to system and user turns are replaced with PyTorch's `ignore\_index=-100` (Figure `fig:pretrain:alpaca_masking`), restricting the cross-entropy gradient to assistant response tokens [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 7, pp. 539–540) [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 4, p. 37).

> [!note] Source Figure
> *Figure:* Alpaca-style instruction loss masking: input tokens are fed intact to the model, while instruction/prompt positions in the target tensor are replaced by $-100$ so cross-entropy is evaluated solely on the response and `<|endoftext|>` token [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 7, Fig. 7.13, p. 539). (`build_large_language_model_from_scratch_p539_fig1.png`)

Algorithm `alg:pretrain:qlora_sft` unifies instruction masking, sequence packing with block-diagonal attention masks, NF4 dequantization, and gradient checkpointing (trading $\sim 1$ extra forward pass per checkpointed block for $O(\sqrt{L})$ activation memory) [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, pp. 215–219).

> [!example] Algorithm: Instruction-Masked SFT with QLoRA & Gradient Checkpointing
> **Input:** Pretrained weights $\boldsymbol{W}_0$, SFT dataset $\mathcal{D}_{\text{SFT}}$, rank $r$, scale $\alpha$, learning rate $\eta$
> **Output:** Trained LoRA adapter weights $\{(\boldsymbol{A}_\ell, \boldsymbol{B}_\ell)\}_{\ell=1}^L$
> Quantize base linear layers $\boldsymbol{W}_0 \to \tilde{\boldsymbol{W}}_0$ in 4-bit NF4 with double-quantized FP8 scales
> **for** each target linear module $\ell \in \{Q, K, V, O, \text{MLP}\}$:
>
>   Initialize $\boldsymbol{A}_\ell \sim \mathcal{N}(0, \sigma^2)$ (BF16), $\boldsymbol{B}_\ell \leftarrow \boldsymbol{0}$ (BF16); freeze $\tilde{\boldsymbol{W}}_{0,\ell}$
>
> Format $\mathcal{D}_{\text{SFT}}$ with chat template; set target $y_t \leftarrow -100$ on prompt/user tokens
> Pack short sequences up to $L_{\max}$ with block-diagonal causal attention masks
> **for** step $= 1, \dots, T$:
>
>   Sample packed minibatch $(\boldsymbol{X}, \boldsymbol{Y})$
>   **for** layer $\ell = 1, \dots, L$  *// Forward pass with activation checkpointing*:
>
>     Dequantize $\tilde{\boldsymbol{W}}_{0,\ell} \to \boldsymbol{W}_{0,\ell}^{\text{BF16}}$ on-the-fly; compute $\boldsymbol{H}_\ell = \boldsymbol{W}_{0,\ell}^{\text{BF16}}\boldsymbol{X}_\ell + \frac{\alpha}{r}\boldsymbol{B}_\ell\boldsymbol{A}_\ell\boldsymbol{X}_\ell$
>     Discard intermediate intra-block activations; retain block boundary checkpoints only
>
>   Compute masked loss $\mathcal{L}_{\text{SFT}} = -\frac{1}{\sum_t \mathbb{I}[y_t \neq -100]} \sum_{t: y_t \neq -100} \log \mathbb{P}(y_t \mid x_{\le t})$
>   Recompute intra-block activations during backward pass; accumulate $\nabla_{\boldsymbol{A}_\ell, \boldsymbol{B}_\ell}\mathcal{L}_{\text{SFT}}$
>   Update $\{(\boldsymbol{A}_\ell, \boldsymbol{B}_\ell)\}$ via Paged AdamW 8-bit with cosine warmup/decay schedule

### Distributed Pretraining Parallelism & Memory Complexity

At pretraining scale, single-GPU memory cannot hold model states or activations. Three orthogonal parallelism axes—Data Parallelism (DP), Tensor Parallelism (TP), and Pipeline Parallelism (PP)—alongside ZeRO optimizer-state sharding and QLoRA govern system memory [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 2, pp. 60–63) [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, pp. 211–215), as summarized in Table `tab:pretrain:complexity`.

**Table:** Memory footprint and systems trade-offs across pretraining and fine-tuning regimes.

| Training Regime | Memory / Param | 7B VRAM | Communication / Compute Trade-off |
| — | — | — | — |
| Full FT (FP32 AdamW) | $16\text{ B}$ ($4\text{W}+4\text{G}+8\text{Opt}$) | $\sim 112\text{ GB}$ (+act.) | Zero recompute; prohibitive single-GPU footprint [[Source - LLM Engineer’s Handbook]] (Ch. 5, p. 212) |
| DP / ZeRO-3 ($P$ GPUs) | $16/P\text{ B}$ (sharded) | $\sim 14\text{ GB}$ ($P=8$) | All-gather / reduce-scatter overhead per layer [[Source - Foundations of Large Language Models]] (Ch. 2, p. 61) |
| LoRA (FP16 + AdamW) | $2\text{ B} + O(r/d)$ | $14\text{–}15.6\text{ GB}$ | Non-destructive; multi-tenant adapter swapping [[Source - LLM Engineer’s Handbook]] (Ch. 5, p. 215) |
| QLoRA (NF4 + Paged 8b) | $\sim 0.5\text{ B} + O(r/d)$ | $9.1\text{–}9.3\text{ GB}$ | $\sim 30\%$ slower due to on-the-fly NF4 dequantization [[Source - LLM Engineer’s Handbook]] (Ch. 5, p. 215) |

> [!info] Cross-Book Synthesis: Prompt Masking, Learning Rates, and Sequence Packing
> Comparing post-training recipes reveals three critical design choices:
> - **Prompt Masking vs. Full-Sequence Loss**: Lambert distinguishes *user-turn masking* (training on all assistant turns) from *final-turn masking* (masking all prior context) [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 4, p. 37). Conversely, Raschka notes empirical findings (Shi et al., 2024) where omitting the $-100$ mask can aid short-instruction tasks, making masking an empirical hyperparameter [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 7, p. 540).
> - **Learning-Rate Scaling**: Because SFT operates on smaller datasets with a strong pretrained initialization, [[Source - Reinforcement Learning from Human Feedback|reinforcement_learning_from_human_feedback]] (Ch. 4, p. 37) recommends an SFT peak learning rate $10\times$–$100\times$ lower than pretraining (e.g., OLMo 2 drops from $3\times 10^{-4}$ in pretraining to $1\times 10^{-5}$ in SFT), whereas sequence packing increases effective tokens per batch and permits $5\text{–}8\times 10^{-5}$.
> - **LoRA Hyperparameters**: [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, p. 214) recommends targeting all attention ($Q, K, V, O$) and MLP linear layers with $\alpha = 2r$ (e.g., $r=16, \alpha=32$) and weight decay $0.01$ over $1\text{–}5$ epochs.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Rank Collapse, Over-Scaling ($\alpha/r$), and Catastrophic Forgetting
> Setting $\alpha/r \gg 2$ amplifies the low-rank update $\frac{\alpha}{r}\boldsymbol{B}\boldsymbol{A}\boldsymbol{x}$ relative to $\boldsymbol{W}_0\boldsymbol{x}$, destabilizing early SFT steps and inducing catastrophic forgetting of pretraining capabilities [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, pp. 212–214). Conversely, setting $r$ too small on heterogeneous multi-domain corpora starves adapter capacity. **Remedy**: Fix $\alpha = 2r$ (or use rank-stabilized $\alpha/\sqrt{r}$ scaling) and sweep $r \in \{8, 16, 32, 64\}$ with LoRA dropout $0.05$–$0.1$ on held-out validation loss.

> [!warning] Non-Associativity of Distributed Low-Precision Reductions
> In distributed mixed-precision (FP16/BF16/FP8) pretraining, floating-point addition is non-associative: $(a+b)+c \neq a+(b+c)$. Accumulating gradients across $P$ workers in different reduction orders introduces numerical divergence, overflow, and underflow across nodes [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 2, p. 63). **Remedy**: Maintain FP32 master weights and gradient accumulation buffers, use BF16 (wider dynamic range than FP16) for forward/backward activations, and apply gradient norm clipping [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 5, p. 364).

> [!warning] Cross-Sample Attention Leakage in Packing and Template Mismatch
> Concatenating multiple short instructions into a single $L_{\max}$ row without resetting causal attention masks causes later samples to attend to earlier unrelated samples, corrupting gradients [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, p. 218). Similarly, fine-tuning an already instruct-tuned model with a mismatched chat template degrades formatting compliance [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, p. 209). **Remedy**: Enforce block-diagonal causal masks and reset rotary position IDs at each packed sample boundary; always match the base/instruct model's exact tokenizer chat template.

> [!warning] Undertrained Scaling Regimes and Benchmark Contamination
> Allocating compute to oversized $N$ with insufficient $D$ ($N \gg N^*$) leaves the model on the suboptimal right branch of Figure `fig:pretrain:chinchilla_scaling` [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Ch. 2, p. 65). In SFT, failing to deduplicate and decontaminate synthetic/web instruction data against evaluation benchmarks inflates apparent generalization [[Source - LLM Engineer’s Handbook|llm_engineer_s_handbook]] (Ch. 5, pp. 213–214). **Remedy**: Size pretraining runs via $N^* \approx (C/6)^{0.5}$ (or overtrain $D \gg D^*$ when inference cost dominates) and run MinHash/exact-substring decontamination before SFT.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[Chinchilla Scaling Laws]], [[LoRA & QLoRA]]
- **Prerequisites (Directed $\leftarrow$):** [[Low-Rank Approximation (Eckart-Young)]], [[Singular Value Decomposition (SVD)]], [[Transformer Architecture]]
- **Enables / Unlocks (Directed $\rightarrow$):** None
- **Related Concepts (Undirected $\leftrightarrow$):** [[Chinchilla Scaling Laws]], [[DPO (Direct Preference Optimization)]], [[LoRA & QLoRA]], [[Low-Rank Approximation (Eckart-Young)]], [[RLHF (KL-Regularized)]], [[Singular Value Decomposition (SVD)]], [[Transformer Architecture]]
- **Synthesized Sources:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Foundations of Large Language Models]], [[Source - LLM Engineer’s Handbook]], [[Source - Reinforcement Learning from Human Feedback]], [[Source - Speech and Language Processing]]
