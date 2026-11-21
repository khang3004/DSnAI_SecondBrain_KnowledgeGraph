---
type: monograph
domain: "D03"
parent_topic: "[[PT 03 - Convolutional, Graph, Recurrent & State-Space Architectures]]"
sub_topic: "[[ST 03 - Recurrent Networks & Selective State-Space Models (Mamba)]]"
cheatsheet: "[[Cheatsheet - Recurrent Neural Networks, BPTT, LSTMs - GRUs & Long-Range Gradient Flow]]"
part_of: 
  - "[[ST 03 - Recurrent Networks & Selective State-Space Models (Mamba)]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
concepts: 
  - "[[RNN, LSTM & GRU]]"
prerequisites: 
  - "[[Automatic Differentiation (AD)]]"
enables: []
related: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Speech and Language Processing]]"
tags:
  - monograph
  - domain-03
---

# Recurrent Neural Networks, BPTT, LSTMs/GRUs & Long-Range Gradient Flow

> [!abstract] Monograph Navigation
> - **Domain:** [[D03 - Neural Architectures, Representations & Regularization|Neural Architectures, Representations & Regularization]]
> - **Parent / Sub-Topic:** [[PT 03 - Convolutional, Graph, Recurrent & State-Space Architectures|Convolutional, Graph, Recurrent & State-Space Architectures]] → [[ST 03 - Recurrent Networks & Selective State-Space Models (Mamba)|Recurrent Networks & Selective State-Space Models (Mamba)]]
> - **Visual Cheatsheet:** [[Cheatsheet - Recurrent Neural Networks, BPTT, LSTMs - GRUs & Long-Range Gradient Flow]]
> - **Description:** Derives unfolded recurrent computation graphs, Backpropagation Through Time (BPTT), spectral-radius causes of vanishing/exploding gradients, gradient clipping, and constant-error-carousel gating in LSTMs and GRUs.

## Problem Formulation & Unified Notation

Sequence modeling requires architectures that process variable-length ordered data $(\boldsymbol{x}_1, \boldsymbol{x}_2, \dots, \boldsymbol{x}_T)$ with $\boldsymbol{x}_t \in \mathbb{R}^{d_x}$ without fixing a finite context window. Unlike feedforward networks or 1-D convolutions with bounded receptive fields, a Recurrent Neural Network (RNN) maintains a stateful hidden representation $\boldsymbol{h}_t \in \mathbb{R}^{d_h}$ that summarizes the entire input history $\boldsymbol{x}_{1:t}$ by sharing parameters across every time step [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 10, PDF pp. 388–389).

In the canonical Elman simple recurrent network (SRN) [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 14.1, PDF pp. 310–311), forward propagation from an initial state $\boldsymbol{h}_0 \in \mathbb{R}^{d_h}$ (typically $\boldsymbol{h}_0 = \boldsymbol{0}$) proceeds for $t = 1, \dots, T$ via

$$
\begin{aligned}
\boldsymbol{a}_t &= \boldsymbol{W}_{hh}\boldsymbol{h}_{t-1} + \boldsymbol{W}_{hx}\boldsymbol{x}_t + \boldsymbol{b}_h,
   \\
  \boldsymbol{h}_t &= \phi(\boldsymbol{a}_t),
   \\
  \boldsymbol{o}_t &= \boldsymbol{W}_{oh}\boldsymbol{h}_t + \boldsymbol{b}_o, \qquad \hat{\boldsymbol{y}}_t = \mathrm{softmax}(\boldsymbol{o}_t),
\end{aligned}
$$

where $\boldsymbol{W}_{hh} \in \mathbb{R}^{d_h \times d_h}$ is the recurrent hidden-to-hidden transition matrix, $\boldsymbol{W}_{hx} \in \mathbb{R}^{d_h \times d_x}$ is the input-to-hidden projection, $\boldsymbol{W}_{oh} \in \mathbb{R}^{d_y \times d_h}$ is the hidden-to-output projection, $\phi(\cdot)$ is an element-wise nonlinearity (commonly $\tanh$), and $\boldsymbol{b}_h \in \mathbb{R}^{d_h}$, $\boldsymbol{b}_o \in \mathbb{R}^{d_y}$ are bias vectors [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.2, PDF p. 396). Given a target sequence $\boldsymbol{y}_{1:T}$ and per-step loss $\ell(\boldsymbol{y}_t, \boldsymbol{o}_t)$ (e.g., cross-entropy negative log-likelihood $-\log p_{\text{model}}(\boldsymbol{y}_t \mid \boldsymbol{x}_{1:t})$), the sequence objective is

$$
\mathcal{L}(\theta) = \sum_{t=1}^{T} \ell(\boldsymbol{y}_t, \boldsymbol{o}_t)
  \quad\text{or normalized}\quad
  \bar{\mathcal{L}}(\theta) = \frac{1}{T} \sum_{t=1}^{T} \ell(\boldsymbol{y}_t, \boldsymbol{o}_t).
$$

The same recurrent state mechanism supports four primary sequence topologies (see [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Sec. 14.6, PDF p. 326) and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.2.1–15.2.3, PDF pp. 533–538)): (i) *sequence labeling* (aligned many-to-many mapping $\boldsymbol{x}_{1:T} \mapsto \boldsymbol{y}_{1:T}$), (ii) *sequence classification* (many-to-one mapping $\boldsymbol{x}_{1:T} \mapsto \boldsymbol{y}_T$ using $\boldsymbol{h}_T$), (iii) *autoregressive generation / language modeling* (factorizing $p(\boldsymbol{y}_{1:T}) = \prod_{t=1}^T p(\boldsymbol{y}_t \mid \boldsymbol{y}_{1:t-1})$ with $\boldsymbol{x}_t = \boldsymbol{y}_{t-1}$), and (iv) *encoder–decoder seq2seq* (encoding $\boldsymbol{x}_{1:T_x}$ into a context vector and decoding $\boldsymbol{y}_{1:T_y}$). Because hidden-to-hidden recurrence allows unbounded state dependence, finite-size RNNs with rational weights are Turing-universal ([[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.1, PDF p. 394); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.2.1, PDF p. 534)).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Source textbooks use conflicting matrix letters (notably swapping $\boldsymbol{U}$ and $\boldsymbol{W}$) and differing tensor layouts (column vectors vs. batch-first row matrices): Goodfellow et al. [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 10), Jurafsky & Martin [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 14), and Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.2). This monograph adopts the explicit subscript convention $\boldsymbol{W}_{hh}, \boldsymbol{W}_{hx}, \boldsymbol{W}_{oh}$ and column vectors $\boldsymbol{x}_t, \boldsymbol{h}_t, \boldsymbol{c}_t$.
>
> | **Concept** | **Goodfellow et al. (2017)** | **Jurafsky & Martin (2026)** | **Murphy Intro (2025)** |
> | --- | --- | --- | --- |
> | Time-step index | Superscript $\boldsymbol{x}^{(t)}, \boldsymbol{h}^{(t)}, \boldsymbol{o}^{(t)}$ ($t=1\dots\tau$) | Subscript $\boldsymbol{x}_t, \boldsymbol{h}_t, \boldsymbol{y}_t$ ($t=1\dots n$) | Subscript $\boldsymbol{x}_t, \boldsymbol{h}_t, \boldsymbol{o}_t$ or batch $\boldsymbol{X}_t, \boldsymbol{H}_t$ |
> | Input $\to$ Hidden | $\boldsymbol{U}$ | $\boldsymbol{W}$ | $\boldsymbol{W}_{hx}$ (or $\boldsymbol{W}_{xh}$ in vec2seq) |
> | Hidden $\to$ Hidden | $\boldsymbol{W}$ | $\boldsymbol{U}$ | $\boldsymbol{W}_{hh}$ |
> | Hidden $\to$ Output | $\boldsymbol{V}$ | $\boldsymbol{V}$ | $\boldsymbol{W}_{oh}$ (or $\boldsymbol{W}_{hy}$) |
> | LSTM cell / gates | Cell $s_i^{(t)}$, forget $f_i^{(t)}$, input $g_i^{(t)}$, output $q_i^{(t)}$ | Context $\boldsymbol{c}_t$, candidate $\boldsymbol{g}_t$, add $\boldsymbol{i}_t$, forget $\boldsymbol{f}_t$, output $\boldsymbol{o}_t$ | Memory $\boldsymbol{C}_t$, candidate $\tilde{\boldsymbol{C}}_t$, gates $\boldsymbol{F}_t, \boldsymbol{I}_t, \boldsymbol{O}_t$ |
> | GRU gates | Update $u_i^{(t)}$, reset $r_i^{(t)}$ | — (focuses on LSTM) | Reset $\boldsymbol{R}_t$, update $\boldsymbol{Z}_t$, candidate $\tilde{\boldsymbol{H}}_t$ |

> [!note] Source Figure
> *Figure:* Time-unfolded computational graph of an Elman RNN over $T=3$ steps, illustrating parameter sharing across $\boldsymbol{W}_{hx}, \boldsymbol{W}_{hh}, \boldsymbol{W}_{oh}$, the initial state $\boldsymbol{h}_0$, logits $\boldsymbol{o}_t$, targets $\boldsymbol{y}_t$, and total loss $\mathcal{L}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 15.9, PDF p. 540). (`probabilisticml_kevinmurphy_introduction_p540_fig1.png`)

## Core Mathematical Derivations & Theorems

### Backpropagation Through Time (BPTT)

Unfolding the cyclic recurrence in `(eq:rnn:elman_state)` across $t=1,\dots,T$ yields a directed acyclic computational graph (Figure `fig:rnn:unrolled_graph`) to which reverse-mode automatic differentiation applies directly.

> [!important] BPTT Recursive Gradient Derivation
> For the Elman RNN in `(eq:rnn:elman_pre)`–`(eq:rnn:seq_loss)` with softmax cross-entropy loss $\ell(\boldsymbol{y}_t, \boldsymbol{o}_t)$, define the output error signal $\boldsymbol{\delta}_t^o := \nabla_{\boldsymbol{o}_t}\mathcal{L} = \hat{\boldsymbol{y}}_t - \boldsymbol{y}_t$ (where $\boldsymbol{y}_t$ is one-hot) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.2.2, PDF p. 400). At the final step $T$, $\boldsymbol{h}_T$ only feeds $\boldsymbol{o}_T$, so
>
> $$
> \nabla_{\boldsymbol{h}_T}\mathcal{L} = \boldsymbol{W}_{oh}^{\top} \nabla_{\boldsymbol{o}_T}\mathcal{L}.
> $$
>
> For $t = T-1, \dots, 1$, $\boldsymbol{h}_t$ has two descendants ($\boldsymbol{o}_t$ and $\boldsymbol{h}_{t+1}$). Applying the multivariable chain rule gives the backward recursion:
>
> $$
> \nabla_{\boldsymbol{h}_t}\mathcal{L}
>   = \left(\frac{\partial \boldsymbol{h}_{t+1}}{\partial \boldsymbol{h}_t}\right)^{\top} \nabla_{\boldsymbol{h}_{t+1}}\mathcal{L}
>   + \left(\frac{\partial \boldsymbol{o}_t}{\partial \boldsymbol{h}_t}\right)^{\top} \nabla_{\boldsymbol{o}_t}\mathcal{L}
>   = \boldsymbol{W}_{hh}^{\top} \mathrm{diag}\big(\phi'(\boldsymbol{a}_{t+1})\big) \nabla_{\boldsymbol{h}_{t+1}}\mathcal{L}
>   + \boldsymbol{W}_{oh}^{\top} \nabla_{\boldsymbol{o}_t}\mathcal{L}.
> $$
>
> Unrolling `(eq:rnn:bptt_rec)` expresses the sensitivity of the loss at step $t$ to an earlier hidden state $\boldsymbol{h}_k$ ($k < t$) through a product of $t-k$ step-to-step Jacobians $\boldsymbol{J}_j := \frac{\partial \boldsymbol{h}_j}{\partial \boldsymbol{h}_{j-1}} = \mathrm{diag}\big(\phi'(\boldsymbol{a}_j)\big)\boldsymbol{W}_{hh}$:
>
> $$
> \frac{\partial \boldsymbol{h}_t}{\partial \boldsymbol{h}_k} = \prod_{j=k+1}^{t} \boldsymbol{J}_j = \prod_{j=k+1}^{t} \mathrm{diag}\big(\phi'(\boldsymbol{a}_j)\big)\boldsymbol{W}_{hh}.
> $$
>
> Summing contributions across all time steps yields the exact parameter gradients [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Eqs. 10.22–10.28, PDF p. 401):
>
> $$
> \begin{aligned}
> \nabla_{\boldsymbol{W}_{oh}}\mathcal{L} &= \sum_{t=1}^{T} (\nabla_{\boldsymbol{o}_t}\mathcal{L})\boldsymbol{h}_t^{\top}, &
>   \nabla_{\boldsymbol{W}_{hh}}\mathcal{L} &= \sum_{t=1}^{T} \mathrm{diag}\big(\phi'(\boldsymbol{a}_t)\big)(\nabla_{\boldsymbol{h}_t}\mathcal{L})\boldsymbol{h}_{t-1}^{\top}, \\
>   \nabla_{\boldsymbol{W}_{hx}}\mathcal{L} &= \sum_{t=1}^{T} \mathrm{diag}\big(\phi'(\boldsymbol{a}_t)\big)(\nabla_{\boldsymbol{h}_t}\mathcal{L})\boldsymbol{x}_t^{\top}, &
>   \nabla_{\boldsymbol{b}_h}\mathcal{L} &= \sum_{t=1}^{T} \mathrm{diag}\big(\phi'(\boldsymbol{a}_t)\big)\nabla_{\boldsymbol{h}_t}\mathcal{L}.
> \end{aligned}
> $$
>
> Equivalently, writing $\boldsymbol{h}_t = f(\boldsymbol{x}_t, \boldsymbol{h}_{t-1}, \boldsymbol{w}_h)$ yields the recursive sensitivity [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. 15.22, PDF p. 541):
>
> $$
> \frac{\partial \boldsymbol{h}_t}{\partial \boldsymbol{w}_h} = \frac{\partial f(\boldsymbol{x}_t,\boldsymbol{h}_{t-1},\boldsymbol{w}_h)}{\partial \boldsymbol{w}_h} + \sum_{i=1}^{t-1} \left(\prod_{j=i+1}^t \frac{\partial f(\boldsymbol{x}_j,\boldsymbol{h}_{j-1},\boldsymbol{w}_h)}{\partial \boldsymbol{h}_{j-1}}\right)\frac{\partial f(\boldsymbol{x}_i,\boldsymbol{h}_{i-1},\boldsymbol{w}_h)}{\partial \boldsymbol{w}_h}.
> $$
>

### Spectral Radius Analysis of Vanishing and Exploding Gradients

The Jacobian product `(eq:rnn:jacobian_prod)` governs whether long-range credit assignment succeeds or collapses exponentially with lag $\Delta = t - k$.

> [!important] Spectral Radius & Eigenvalue-Power Decay/Explosion
> Consider first a linearized or constant-Jacobian regime where $\boldsymbol{J}_j = \boldsymbol{J}$ is independent of $j$ (e.g., $\phi(z)=z$ so $\boldsymbol{J} = \boldsymbol{W}_{hh}$), and suppose $\boldsymbol{W}_{hh}$ admits an eigendecomposition $\boldsymbol{W}_{hh} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{-1}$ with eigenvalues $\{\lambda_m\}_{m=1}^{d_h}$ and spectral radius $\rho(\boldsymbol{W}_{hh}) := \max_m \lvert \lambda_m \rvert$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.7–10.8, PDF pp. 417–420). Over $\Delta = t - k$ steps,
>
> $$
> \boldsymbol{J}^{\Delta} = \boldsymbol{Q}\boldsymbol{\Lambda}^{\Delta}\boldsymbol{Q}^{-1}.
> $$
>
> If a gradient perturbation $\delta \boldsymbol{v}$ lies along a unit eigenvector $\boldsymbol{v}$ with eigenvalue $\lambda$, backpropagating $\Delta$ steps scales the perturbation to $\delta \boldsymbol{J}^{\Delta}\boldsymbol{v} = \delta \lambda^{\Delta}\boldsymbol{v}$, whose norm is $\delta\lvert \lambda \rvert^{\Delta}$:
> - **Vanishing ($\lvert \lambda \rvert < 1$):** $\lvert \lambda \rvert^{\Delta} \to 0$ exponentially in $\Delta$; long-term gradient components are drowned out by short-term components.
> - **Exploding ($\lvert \lambda \rvert > 1$):** $\lvert \lambda \rvert^{\Delta} \to \infty$ exponentially in $\Delta$; parameter updates encounter steep "cliff" walls in the loss surface [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.11.1, PDF p. 429).
>
> In the general nonlinear case $\boldsymbol{J}_j = \mathrm{diag}(\phi'(\boldsymbol{a}_j))\boldsymbol{W}_{hh}$, let $\gamma := \sup_z \lvert \phi'(z) \rvert$ ($\gamma = 1$ for $\tanh$, $\gamma = 1/4$ for logistic $\sigma$) and let $\sigma_{\max}(\boldsymbol{W}_{hh}) = \lVert \boldsymbol{W}_{hh} \rVert_2$ be the largest singular value of $\boldsymbol{W}_{hh}$. By submultiplicativity of the spectral norm,
>
> $$
> \lVert \frac{\partial \boldsymbol{h}_t}{\partial \boldsymbol{h}_k} \rVert_2
>   \le \prod_{j=k+1}^{t} \lVert \mathrm{diag}(\phi'(\boldsymbol{a}_j)) \rVert_2 \lVert \boldsymbol{W}_{hh} \rVert_2
>   \le \big(\gamma \sigma_{\max}(\boldsymbol{W}_{hh})\big)^{t-k}.
> $$
>
> Thus $\sigma_{\max}(\boldsymbol{W}_{hh}) < 1/\gamma$ is a *sufficient condition* for gradients to vanish exponentially for all input sequences [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.2.6, PDF p. 541). Conversely, $\rho(\boldsymbol{J}) > 1$ along active non-saturated modes is necessary for gradient explosion.

> [!tip] Why Vanilla RNNs Cannot Simultaneously Store and Compute
> In an Elman RNN, the single vector $\boldsymbol{h}_t$ and weight matrix $\boldsymbol{W}_{hh}$ must perform two conflicting roles at every step: (1) execute local nonlinear feature extraction for the current prediction $\boldsymbol{o}_t$, and (2) preserve distant context across dozens or hundreds of intervening tokens [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Sec. 14.5, PDF p. 315). When $\boldsymbol{W}_{hh}$ is contractive ($\rho < 1$), finite-precision states exponentially forget past inputs; when $\rho > 1$ to retain memory, slight perturbations trigger chaotic oscillations or gradient cliffs [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.8, PDF p. 420).

### Constant Error Carousel (CEC) & Gated Architectures: LSTM and GRU

Rather than tuning $\boldsymbol{W}_{hh}$ to sit precariously on the $\rho = 1$ boundary, gated RNNs introduce *additive* state highways with input-dependent, dynamically learned time constants ([[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.10, PDF pp. 423–427); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.2.7, PDF p. 542)).

> [!important] LSTM Equations & Constant Error Carousel Highway
> The Long Short-Term Memory (LSTM) augments the hidden output $\boldsymbol{h}_t \in \mathbb{R}^{d_h}$ with an explicit memory cell $\boldsymbol{c}_t \in \mathbb{R}^{d_h}$ regulated by three sigmoidal gates ($\boldsymbol{f}_t, \boldsymbol{i}_t, \boldsymbol{o}_t \in (0,1)^{d_h}$) and a candidate state $\tilde{\boldsymbol{c}}_t \in (-1,1)^{d_h}$ [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Eqs. 14.20–14.27, PDF p. 324); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eqs. 15.27–15.32, PDF pp. 543–544):
>
> $$
> \begin{aligned}
> \boldsymbol{f}_t &= \sigma(\boldsymbol{W}_{fx}\boldsymbol{x}_t + \boldsymbol{W}_{fh}\boldsymbol{h}_{t-1} + \boldsymbol{b}_f) && \text{(forget gate)},  \\
>   \boldsymbol{i}_t &= \sigma(\boldsymbol{W}_{ix}\boldsymbol{x}_t + \boldsymbol{W}_{ih}\boldsymbol{h}_{t-1} + \boldsymbol{b}_i) && \text{(input / add gate)},  \\
>   \boldsymbol{o}_t &= \sigma(\boldsymbol{W}_{ox}\boldsymbol{x}_t + \boldsymbol{W}_{oh}\boldsymbol{h}_{t-1} + \boldsymbol{b}_o) && \text{(output gate)},  \\
>   \tilde{\boldsymbol{c}}_t &= \tanh(\boldsymbol{W}_{cx}\boldsymbol{x}_t + \boldsymbol{W}_{ch}\boldsymbol{h}_{t-1} + \boldsymbol{b}_c) && \text{(candidate memory)},  \\
>   \boldsymbol{c}_t &= \boldsymbol{f}_t \odot \boldsymbol{c}_{t-1} + \boldsymbol{i}_t \odot \tilde{\boldsymbol{c}}_t && \text{(additive cell-state update)},  \\
>   \boldsymbol{h}_t &= \boldsymbol{o}_t \odot \tanh(\boldsymbol{c}_t) && \text{(gated hidden output)}.
> \end{aligned}
> $$
>
> Differentiating $\boldsymbol{c}_t$ with respect to $\boldsymbol{c}_{t-1}$ along the direct cell-state highway (holding the gate activations fixed as functions of $\boldsymbol{h}_{t-1}$) yields the diagonal Jacobian
>
> $$
> \left.\frac{\partial \boldsymbol{c}_t}{\partial \boldsymbol{c}_{t-1}}\right|_{\text{direct}} = \mathrm{diag}(\boldsymbol{f}_t)
>   \implies
>   \left.\frac{\partial \boldsymbol{c}_t}{\partial \boldsymbol{c}_k}\right|_{\text{direct}} = \mathrm{diag}\!\left(\prod_{j=k+1}^{t} \boldsymbol{f}_j\right).
> $$
>
> This additive self-loop forms the *Constant Error Carousel (CEC)*: whenever the forget gate remains open ($f_{j,m} \approx 1$, $i_{j,m} \approx 0$), error gradients propagate backward across $t-k$ steps with near-unit gain and *zero* repeated multiplication by $\boldsymbol{W}_{hh}$ or $\tanh'(\cdot)$.

The **Gated Recurrent Unit (GRU)** simplifies the LSTM by merging the cell state and hidden state ($\boldsymbol{h}_t$ only) and coupling the forget and input decisions into a single update gate $\boldsymbol{z}_t$ alongside a reset gate $\boldsymbol{r}_t$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.10.2, PDF p. 427); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.2.7.1, PDF p. 543):

$$
\begin{aligned}
\boldsymbol{r}_t &= \sigma(\boldsymbol{W}_{rx}\boldsymbol{x}_t + \boldsymbol{W}_{rh}\boldsymbol{h}_{t-1} + \boldsymbol{b}_r), \qquad
  \boldsymbol{z}_t = \sigma(\boldsymbol{W}_{zx}\boldsymbol{x}_t + \boldsymbol{W}_{zh}\boldsymbol{h}_{t-1} + \boldsymbol{b}_z),  \\
  \tilde{\boldsymbol{h}}_t &= \tanh\big(\boldsymbol{W}_{hx}\boldsymbol{x}_t + \boldsymbol{W}_{hh}(\boldsymbol{r}_t \odot \boldsymbol{h}_{t-1}) + \boldsymbol{b}_h\big),  \\
  \boldsymbol{h}_t &= \boldsymbol{z}_t \odot \boldsymbol{h}_{t-1} + (\boldsymbol{1} - \boldsymbol{z}_t) \odot \tilde{\boldsymbol{h}}_t.
\end{aligned}
$$

When $z_{t,m} \to 1$, dimension $m$ copies $h_{t-1,m}$ unchanged; when $r_{t,m} \to 0$, the candidate state ignores past hidden activity and behaves like a feedforward layer on $\boldsymbol{x}_t$.

## Algorithmic Implementation & Comparative Synthesis

Full BPTT over a sequence of length $T$ requires $O(T)$ sequential steps and $O(T d_h)$ memory to cache intermediate activations, making long sequences ($T \gg 10^3$) prohibitive ([[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.2, PDF p. 396); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.2.5, PDF p. 541)). In practice, training combines *Truncated BPTT (TBPTT)* over contiguous chunks of length $K$ (with hidden-state carryover) and *global $L_2$-norm gradient clipping* [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.11.1, PDF pp. 429–430) to prevent gradient cliffs.

> [!example] Algorithm: Truncated BPTT with Hidden-State Carryover and Norm Gradient Clipping
> **Input:** Ordered sequence $(\boldsymbol{x}_{1:T}, \boldsymbol{y}_{1:T})$, chunk length $K$, clipping threshold $\nu > 0$, learning rate $\eta$, initial parameters $\theta$
> **Output:** Updated parameters $\theta$
> Initialize recurrent state $\boldsymbol{h}_0 \leftarrow \boldsymbol{0}$ (and $\boldsymbol{c}_0 \leftarrow \boldsymbol{0}$ for LSTM)
> **for** $s = 1, K+1, 2K+1, \dots \le T$:
>
>   $e \leftarrow \min(s + K - 1, T)$
>   Detach $\boldsymbol{h}_{s-1}$ (and $\boldsymbol{c}_{s-1}$) from prior computation graph to bound BPTT horizon to $K$
>   **for** $t = s$ to $e$:
>
>     Compute $(\boldsymbol{h}_t, \boldsymbol{o}_t)$ via forward cell equations `(eq:rnn:elman_pre)`–`(eq:rnn:elman_out)` or `(eq:rnn:lstm_f)`–`(eq:rnn:lstm_h)`
>
>   $\mathcal{L}_{\text{chunk}} \leftarrow \frac{1}{e - s + 1} \sum_{t=s}^{e} \ell(\boldsymbol{y}_t, \boldsymbol{o}_t)$
>   Compute $\boldsymbol{g} \leftarrow \nabla_{\theta} \mathcal{L}_{\text{chunk}}$ via reverse-mode BPTT from $t=e$ down to $t=s$
>   **if** $\lVert \boldsymbol{g} \rVert_2 > \nu$:
>
>     $\boldsymbol{g} \leftarrow \frac{\nu}{\lVert \boldsymbol{g} \rVert_2} \boldsymbol{g}$  *// Rescale step norm while preserving descent direction*
>
>   $\theta \leftarrow \theta - \eta \boldsymbol{g}$

**Table:** Complexity and gradient-flow comparison across recurrent architectures (input dim $d_x$, hidden dim $d_h$, chunk/sequence length $K \le T$).

| **Architecture** | **Recurrent Params** | **Time / Step** | **BPTT Mem.** | **Long-Range Mechanism** |
| --- | --- | --- | --- | --- |
| Elman SRN | $d_h(d_x + d_h + 1)$ | $O(d_h(d_x + d_h))$ | $O(K d_h)$ | None ($\boldsymbol{J}_t$ product decays/explodes) |
| LSTM (4 gates) | $4 d_h(d_x + d_h + 1)$ | $O(4 d_h(d_x + d_h))$ | $O(K d_h)$ | Additive cell $\boldsymbol{c}_t$ (CEC, $\prod \boldsymbol{f}_t$) |
| GRU (3 gates) | $3 d_h(d_x + d_h + 1)$ | $O(3 d_h(d_x + d_h))$ | $O(K d_h)$ | Convex interpolation $\boldsymbol{z}_t \odot \boldsymbol{h}_{t-1}$ |
| Echo State Net | $d_y d_h$ trainable | $O(d_h^2)$ | $O(1)$ | Fixed $\boldsymbol{W}_{hh}$ with $\rho(\boldsymbol{W}_{hh}) \approx 1$ |
| Bidirectional RNN | $2\times$ base cell | $2\times$ base cell | $O(2 T d_h)$ | Forward + backward states (non-causal) |

> [!info] Cross-Book Synthesis: Alternative Remedies for Long-Range Dependencies
> The three source texts highlight complementary strategies beyond LSTM/GRU gating:
> - **Reservoir Computing (ESNs):** Fix random $\boldsymbol{W}_{hx}, \boldsymbol{W}_{hh}$ near the edge of stability ($\rho \approx 1$) and train only linear readout $\boldsymbol{W}_{oh}$ via convex regression [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.8, PDF pp. 419–420); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.2.6, PDF p. 541).
> - **Multi-Scale Leaky Units & Skip Connections:** Linear self-connections $\mu_t = \alpha \mu_{t-1} + (1-\alpha)v_t$ or $d$-step skip edges reduce effective path length from $T$ to $T/d$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.9, PDF pp. 421–423).
> - **Information-Flow Regularization:** Pascanu et al. penalize $\Omega = \sum_t \big(\lVert \boldsymbol{J}_t^{\top} \nabla_{\boldsymbol{h}_t}\mathcal{L} \rVert_2 / \lVert \nabla_{\boldsymbol{h}_t}\mathcal{L} \rVert_2 - 1\big)^2$ alongside gradient clipping [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.11.2, PDF pp. 430–431).
> - **Stacked & Bidirectional Architectures:** Deepening RNNs vertically or running dual forward/backward chains captures both left and right context in non-causal tasks [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Sec. 14.4, PDF pp. 320–323); [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.3, PDF p. 409).

## Visual Diagnostics & Geometric Structure

Figure `fig:rnn:lstm_gates_tikz` isolates the additive Constant Error Carousel highway inside an LSTM cell, while Figure `fig:rnn:eigenvalue_decay` quantifies the exponential divergence of $\lvert \lambda \rvert^T$ predicted by `(eq:rnn:eig_power)`. Figures `fig:rnn:lstm_murphy` and `fig:rnn:gru_murphy` show the corresponding textbook computation graphs for LSTM and GRU cells.

> [!example] Visual Diagram (LaTeX / PDF)
> *LSTM cell architecture highlighting the additive Constant Error Carousel (CEC) highway $\boldsymbol{c}_t = \boldsymbol{f}_t \odot \boldsymbol{c}_{t-1} + \boldsymbol{i}_t \odot \tilde{\boldsymbol{c}}_t$ and gated output $\boldsymbol{h}_t = \boldsymbol{o}_t \odot \tanh(\boldsymbol{c}_t)$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Exponential scaling of Jacobian eigenvalue power $\lvert \lambda \rvert^T$ across sequence lag $T$. Only $\lvert \lambda \rvert \approx 1$ (e.g., an open LSTM forget gate $f_t \approx 1$) preserves gradient magnitude over long horizons.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!note] Source Figure
> *Figure:* LSTM cell computation graph showing the forget gate $\boldsymbol{F}_t$, input gate $\boldsymbol{I}_t$, candidate memory $\tilde{\boldsymbol{C}}_t$, output gate $\boldsymbol{O}_t$, and horizontal memory highway $\boldsymbol{C}_{t-1} \to \boldsymbol{C}_t$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 15.11, PDF p. 544). (`probabilisticml_kevinmurphy_introduction_p544_fig1.png`)

> [!note] Source Figure
> *Figure:* GRU cell computation graph showing the reset gate $\boldsymbol{R}_t$, update gate $\boldsymbol{Z}_t$, candidate state $\tilde{\boldsymbol{H}}_t$, and convex interpolation $\boldsymbol{H}_t = \boldsymbol{Z}_t \odot \boldsymbol{H}_{t-1} + (\boldsymbol{1} - \boldsymbol{Z}_t) \odot \tilde{\boldsymbol{H}}_t$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Fig. 15.10, PDF p. 542). (`probabilisticml_kevinmurphy_introduction_p542_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Zero-Initialized Forget-Gate Bias ($\boldsymbol{b}_f = \boldsymbol{0}$)
> Initializing LSTM forget-gate biases to $\boldsymbol{b}_f = \boldsymbol{0}$ yields $f_t \approx \sigma(0) = 0.5$ early in training, causing the CEC product $\prod_{j=1}^{20} f_j \approx 0.5^{20} \approx 9.5 \times 10^{-7}$ to vanish before learning begins [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.2.7.2, PDF p. 544). **Remedy:** Initialize $\boldsymbol{b}_f = \boldsymbol{1}$ (or $1.0$–$2.0$) so $f_t \approx 0.73$–$0.88$ by default [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.10.2, PDF p. 427).

> [!warning] Exposure Bias Under Strict Teacher Forcing
> Training autoregressive RNNs with teacher forcing feeds ground-truth $\boldsymbol{y}_{t-1}$ at step $t$, whereas open-loop inference feeds the model's own sample $\hat{\boldsymbol{y}}_{t-1}$. Errors compound rapidly when the model encounters self-generated states outside the training manifold [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.2.1, PDF p. 399). **Remedy:** Apply *scheduled sampling* to anneal from ground-truth inputs to model-generated inputs [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.2.4, PDF p. 540).

> [!warning] Gradient Clipping Bias & TBPTT Horizon Blindness
> Per-minibatch norm clipping ($\boldsymbol{g} \leftarrow \frac{\nu}{\lVert \boldsymbol{g} \rVert_2}\boldsymbol{g}$) introduces a heuristic bias relative to true-gradient clipping because minibatches with large norms are down-weighted [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 10.11.1, PDF p. 430). Furthermore, TBPTT($K$) cannot propagate credit beyond $K$ steps, and carrying $\boldsymbol{h}_{s-1}$ across non-contiguous or shuffled batches injects corrupted context [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 15.2.5, PDF p. 541). **Remedy:** Reset $\boldsymbol{h}_0 = \boldsymbol{0}$ at every sequence boundary and match $K$ to the longest required dependency span.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (1):** [[RNN, LSTM & GRU]]
- **Prerequisites (Directed $\leftarrow$):** [[Automatic Differentiation (AD)]]
- **Enables / Unlocks (Directed $\rightarrow$):** None
- **Related Concepts (Undirected $\leftrightarrow$):** None
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Probabilistic Machine Learning]], [[Source - Speech and Language Processing]]
