---
type: monograph
domain: "D03"
parent_topic: "[[PT 03 - Convolutional, Graph, Recurrent & State-Space Architectures]]"
sub_topic: "[[ST 03 - Recurrent Networks & Selective State-Space Models (Mamba)]]"
cheatsheet: "[[Cheatsheet - Classical State-Space Models, Structured SSMs & Selective Scan (Mamba)]]"
part_of: 
  - "[[ST 03 - Recurrent Networks & Selective State-Space Models (Mamba)]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
concepts: 
  - "[[Kalman Filter]]"
  - "[[State Space Models (S4 & Mamba)]]"
prerequisites: 
  - "[[Kalman Filter]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
enables: 
  - "[[State Space Models (S4 & Mamba)]]"
related: 
  - "[[Kalman Filter]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
  - "[[State Space Models (S4 & Mamba)]]"
sources: 
  - "[[Source - Foundations of Large Language Models]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Report - Mamba When Transformer is Not All You Need - AIVN]]"
tags:
  - monograph
  - domain-03
---

# Classical State-Space Models, Structured SSMs & Selective Scan (Mamba)

> [!abstract] Monograph Navigation
> - **Domain:** [[D03 - Neural Architectures, Representations & Regularization|Neural Architectures, Representations & Regularization]]
> - **Parent / Sub-Topic:** [[PT 03 - Convolutional, Graph, Recurrent & State-Space Architectures|Convolutional, Graph, Recurrent & State-Space Architectures]] → [[ST 03 - Recurrent Networks & Selective State-Space Models (Mamba)|Recurrent Networks & Selective State-Space Models (Mamba)]]
> - **Visual Cheatsheet:** [[Cheatsheet - Classical State-Space Models, Structured SSMs & Selective Scan (Mamba)]]
> - **Description:** Bridges classical linear-Gaussian Kalman filtering/smoothing, continuous-to-discrete ZOH discretization of LTI state-space models (S4), convolutional-vs-recurrent duality, input-dependent selective SSMs (Mamba/S6), and hardware-aware parallel scan.

## Problem Formulation & Unified Notation

Sequence modeling across control theory, Bayesian time-series analysis, and modern foundation models centers on a common structural abstraction: a latent dynamical state summarizes past history so that future predictions or emissions depend only on the current state and instantaneous input. In probabilistic machine learning, a discrete-time *Linear-Gaussian State-Space Model (LG-SSM)*—also termed a *Linear Dynamical System (LDS)*—defines a Markovian chain over latent states $\boldsymbol{z}_t \in \mathbb{R}^{N_z}$ generating noisy observations $\boldsymbol{y}_t \in \mathbb{R}^{N_y}$ given optional control inputs $\boldsymbol{u}_t \in \mathbb{R}^{N_u}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 8, Eq. (8.1), p. 359; Ch. 29, Eqs. (29.73)–(29.74), p. 1006):

$$
\begin{aligned}
\boldsymbol{z}_t &= \boldsymbol{F}_t \boldsymbol{z}_{t-1} + \boldsymbol{B}_t \boldsymbol{u}_t + \boldsymbol{b}_t + \boldsymbol{q}_t, & \boldsymbol{q}_t &\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{Q}_t),  \\
  \boldsymbol{y}_t &= \boldsymbol{H}_t \boldsymbol{z}_t + \boldsymbol{D}_t \boldsymbol{u}_t + \boldsymbol{d}_t + \boldsymbol{r}_t, & \boldsymbol{r}_t &\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{R}_t),
\end{aligned}
$$

with Gaussian prior $p(\boldsymbol{z}_1 \mid \boldsymbol{u}_1) = \mathcal{N}(\boldsymbol{z}_1 \mid \boldsymbol{m}_1, \boldsymbol{V}_1)$ (or $\boldsymbol{z}_0 \sim \mathcal{N}(\boldsymbol{\mu}_{0|0}, \boldsymbol{\Sigma}_{0|0})$). Given parameters $\theta_t = (\boldsymbol{F}_t, \boldsymbol{H}_t, \boldsymbol{B}_t, \boldsymbol{D}_t, \boldsymbol{Q}_t, \boldsymbol{R}_t)$, exact Bayesian *filtering* computes $p(\boldsymbol{z}_t \mid \boldsymbol{y}_{1:t}, \boldsymbol{u}_{1:t})$ online, whereas *smoothing* computes $p(\boldsymbol{z}_t \mid \boldsymbol{y}_{1:T}, \boldsymbol{u}_{1:T})$ offline [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 8.1.1, p. 359).

In contrast, deep neural *State-Space Models (SSMs)* such as S4 and Mamba begin from a deterministic, continuous-time linear time-invariant (LTI) ODE mapping a 1D input signal $x(t) \in \mathbb{R}$ (applied independently per channel $d \in \{1,\dots,D\}$) through a latent state $\boldsymbol{h}(t) \in \mathbb{R}^N$ to an output $y(t) \in \mathbb{R}$ [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (Eqs. (1a)–(1b), PDF p. 5):

$$
\begin{aligned}
\dot{\boldsymbol{h}}(t) &= \boldsymbol{A}\boldsymbol{h}(t) + \boldsymbol{B}x(t),  \\
  y(t) &= \boldsymbol{C}\boldsymbol{h}(t) + Dx(t),
\end{aligned}
$$

where $\boldsymbol{A} \in \mathbb{R}^{N \times N}$ is the state transition matrix, $\boldsymbol{B} \in \mathbb{R}^{N \times 1}$ is the input projection, $\boldsymbol{C} \in \mathbb{R}^{1 \times N}$ is the output projection, and $D \in \mathbb{R}$ acts as a residual skip connection. Discretizing `(eq:ssm:ct_state)`–`(eq:ssm:ct_out)` at step size (timescale) $\Delta > 0$ yields a deterministic recurrence $\boldsymbol{h}_t = \bar{\boldsymbol{A}}\boldsymbol{h}_{t-1} + \bar{\boldsymbol{B}}x_t$, $y_t = \boldsymbol{C}\boldsymbol{h}_t + Dx_t$ [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (Eqs. (4a)–(4b), PDF p. 6). Meanwhile, large language model architectures frame this same recurrence as a *fixed-size recurrent memory* alternative to the $O(L)$-growing key-value (KV) cache of standard softmax self-attention [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Sec. 2.3.2–2.3.3, pp. 68–79).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Because control theory, probabilistic graphical models, and deep sequence modeling use conflicting symbols for inputs, outputs, and states, Table `tab:ssm:notation_map` establishes a unified dictionary across all three sources.
> 
>
>
>
> | **Concept** | **Murphy Adv. (LDS)** | **Mamba / S4** | **LLM Found.** | **Unified Role** |
> | — | — | — | — | — |
> | Latent state | $\boldsymbol{z}_t \in \mathbb{R}^{N_z}$ | $\boldsymbol{h}_t \in \mathbb{R}^{N}$ | $\boldsymbol{h}_i \in \mathbb{R}^d$ | Markovian summary of history |
> | Input / control | $\boldsymbol{u}_t \in \mathbb{R}^{N_u}$ | $x_t \in \mathbb{R}^D$ | $\boldsymbol{x}_i \in \mathbb{R}^d$ | Driving signal at step $t$ |
> | Observed output | $\boldsymbol{y}_t \in \mathbb{R}^{N_y}$ | $y_t \in \mathbb{R}^D$ | $\boldsymbol{y}_i$ / $\mathrm{Att}_i$ | Emitted or transformed token |
> | State transition | $\boldsymbol{F}_t \in \mathbb{R}^{N_z \times N_z}$ | $\boldsymbol{A}$ (cont.), $\bar{\boldsymbol{A}}$ (disc.) | Recurrent cell $f$ | Governs memory decay/rotation |
> | Input matrix | $\boldsymbol{B}_t \in \mathbb{R}^{N_z \times N_u}$ | $\boldsymbol{B}$ (cont.), $\bar{\boldsymbol{B}}$ (disc.) | $\boldsymbol{W}_K, \boldsymbol{W}_V$ | Writes input into latent state |
> | Output matrix | $\boldsymbol{H}_t \in \mathbb{R}^{N_y \times N_z}$ | $\boldsymbol{C} \in \mathbb{R}^{1 \times N}$ | $\boldsymbol{q}_i$ (query) | Reads latent state into output |
> | Sequence length | $T$ | $L$ (or $n$) | $m$ (or $L$) | Total number of time steps |

## Core Mathematical Derivations & Theorems

### Exact Bayesian Filtering & RTS Smoothing in Linear-Gaussian SSMs

For an LG-SSM `(eq:ssm:lds_trans)`–`(eq:ssm:lds_obs)`, the joint distribution $p(\boldsymbol{y}_{1:T}, \boldsymbol{z}_{1:T} \mid \boldsymbol{u}_{1:T})$ is a single multivariate Gaussian of dimension $(N_y + N_z)T$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 8.2, p. 364). Exploiting the chain-structured Markov property reduces exact posterior inference from $O(T^3 N_z^3)$ naive Gaussian conditioning to $O(T N_z^3)$ recursive message passing.

> [!important] Kalman Filter Predict–Update & RTS Smoother Recursions
> Assume the filtered belief at step $t-1$ is Gaussian: $p(\boldsymbol{z}_{t-1} \mid \boldsymbol{y}_{1:t-1}, \boldsymbol{u}_{1:t-1}) = \mathcal{N}(\boldsymbol{z}_{t-1} \mid \boldsymbol{\mu}_{t-1|t-1}, \boldsymbol{\Sigma}_{t-1|t-1})$.
>
> **1. Prediction Step (Chapman–Kolmogorov):** Marginalizing the joint predictive distribution $p(\boldsymbol{z}_{t-1}, \boldsymbol{z}_t \mid \boldsymbol{y}_{1:t-1})$ over $\boldsymbol{z}_{t-1}$ via linear-Gaussian identities yields [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. (8.33)–(8.41), pp. 366–367):
>
> $$
> \begin{aligned}
> \boldsymbol{\mu}_{t|t-1} &= \boldsymbol{F}_t \boldsymbol{\mu}_{t-1|t-1} + \boldsymbol{B}_t \boldsymbol{u}_t + \boldsymbol{b}_t,  \\
>   \boldsymbol{\Sigma}_{t|t-1} &= \boldsymbol{F}_t \boldsymbol{\Sigma}_{t-1|t-1} \boldsymbol{F}_t^{\top} + \boldsymbol{Q}_t.
> \end{aligned}
>
> $$
>
>
> **2. Measurement Update (Bayes' Rule):** Forming the joint Gaussian over $(\boldsymbol{z}_t, \boldsymbol{y}_t)$ conditioned on $\boldsymbol{y}_{1:t-1}$ gives expected observation $\hat{\boldsymbol{y}}_t = \boldsymbol{H}_t\boldsymbol{\mu}_{t|t-1} + \boldsymbol{D}_t\boldsymbol{u}_t + \boldsymbol{d}_t$, innovation covariance $\boldsymbol{S}_t$, and cross-covariance $\boldsymbol{C}_t = \boldsymbol{\Sigma}_{t|t-1}\boldsymbol{H}_t^{\top}$. Conditioning on $\boldsymbol{y}_t$ yields [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. (8.25)–(8.31), (8.46)–(8.52), pp. 366–368):
>
> $$
> \begin{aligned}
> \boldsymbol{S}_t &= \boldsymbol{H}_t \boldsymbol{\Sigma}_{t|t-1} \boldsymbol{H}_t^{\top} + \boldsymbol{R}_t, \qquad \boldsymbol{K}_t = \boldsymbol{\Sigma}_{t|t-1}\boldsymbol{H}_t^{\top} \boldsymbol{S}_t^{-1},  \\
>   \boldsymbol{\mu}_{t|t} &= \boldsymbol{\mu}_{t|t-1} + \boldsymbol{K}_t (\boldsymbol{y}_t - \hat{\boldsymbol{y}}_t),  \\
>   \boldsymbol{\Sigma}_{t|t} &= \boldsymbol{\Sigma}_{t|t-1} - \boldsymbol{K}_t \boldsymbol{H}_t \boldsymbol{\Sigma}_{t|t-1} = \boldsymbol{\Sigma}_{t|t-1} - \boldsymbol{K}_t \boldsymbol{S}_t \boldsymbol{K}_t^{\top}.
> \end{aligned}
>
> $$
>
> Moreover, the exact log-marginal likelihood accumulates the one-step-ahead predictive log-densities: $\log p(\boldsymbol{y}_{1:T} \mid \boldsymbol{u}_{1:T}) = \sum_{t=1}^T \log \mathcal{N}(\boldsymbol{y}_t \mid \hat{\boldsymbol{y}}_t, \boldsymbol{S}_t)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eq. (8.36), p. 367).
>
> **3. Rauch–Tung–Striebel (RTS) Backwards Smoother:** Using the Markov property $p(\boldsymbol{z}_t \mid \boldsymbol{z}_{t+1}, \boldsymbol{y}_{1:T}) = p(\boldsymbol{z}_t \mid \boldsymbol{z}_{t+1}, \boldsymbol{y}_{1:t})$ and conditioning the two-slice filtered joint $p(\boldsymbol{z}_t, \boldsymbol{z}_{t+1} \mid \boldsymbol{y}_{1:t})$ on $\boldsymbol{z}_{t+1}$, then marginalizing over $p(\boldsymbol{z}_{t+1} \mid \boldsymbol{y}_{1:T}) = \mathcal{N}(\boldsymbol{\mu}_{t+1|T}, \boldsymbol{\Sigma}_{t+1|T})$, yields the backwards recursion for $t = T-1,\dots,1$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. (8.67)–(8.84), pp. 370–371):
>
> $$
> \begin{aligned}
> \boldsymbol{J}_t &= \boldsymbol{\Sigma}_{t|t} \boldsymbol{F}_t^{\top} \boldsymbol{\Sigma}_{t+1|t}^{-1},  \\
>   \boldsymbol{\mu}_{t|T} &= \boldsymbol{\mu}_{t|t} + \boldsymbol{J}_t (\boldsymbol{\mu}_{t+1|T} - \boldsymbol{\mu}_{t+1|t}),  \\
>   \boldsymbol{\Sigma}_{t|T} &= \boldsymbol{\Sigma}_{t|t} + \boldsymbol{J}_t (\boldsymbol{\Sigma}_{t+1|T} - \boldsymbol{\Sigma}_{t+1|t})\boldsymbol{J}_t^{\top}.
> \end{aligned}
>
> $$

> [!tip] Kalman Gain as Inverse Signal-to-Noise Ratio
> In the scalar observation limit ($\boldsymbol{H}_t = 1$), the Kalman gain reduces to $k_t = \Sigma_{t|t-1} / (\Sigma_{t|t-1} + R_t)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 8.2.2.2, p. 366). When sensor noise $R_t$ is large relative to prior state uncertainty $\Sigma_{t|t-1}$, $k_t \to 0$ and the filter trusts its dynamical prediction $\boldsymbol{\mu}_{t|t-1}$; when $R_t \to 0$, $k_t \to 1$ and the state snaps to the measurement. Crucially, the Kalman gain $\boldsymbol{K}_t$ is an *uncertainty-weighted* input gate—a classical probabilistic precursor to the input-dependent discretization step $\Delta_t$ in Mamba.

### Continuous-to-Discrete ZOH Discretization & Convolutional Duality

Deep neural SSMs parameterize dynamics in continuous time `(eq:ssm:ct_state)`–`(eq:ssm:ct_out)` so that learned parameters $(\boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C})$ are invariant to sampling resolution, and then map them to discrete recurrences via a learnable step size $\Delta > 0$ [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (PDF pp. 5–6).

> [!important] Zero-Order Hold (ZOH) Discretization & Global Convolution Dual
> Suppose $x(t)$ is held piecewise-constant over $[t_k, t_k + \Delta]$ with value $x_k$. Multiplying `(eq:ssm:ct_state)` by the integrating factor $e^{-\boldsymbol{A}t}$ and integrating from $t_k$ to $t_k + \Delta$ gives:
>
> $$
> \begin{aligned}
> \boldsymbol{h}(t_k + \Delta) &= e^{\boldsymbol{A}\Delta}\boldsymbol{h}(t_k) + \left(\int_0^{\Delta} e^{\boldsymbol{A}\tau} \mathrm{d}\tau\right)\boldsymbol{B} x_k.
> \end{aligned}
>
> $$
>
> Evaluating the matrix integral when $\boldsymbol{A}$ is invertible yields the exact *Zero-Order Hold (ZOH)* parameterization [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (PDF p. 6):
>
> $$
> \bar{\boldsymbol{A}} = \exp(\Delta\boldsymbol{A}), \qquad \bar{\boldsymbol{B}} = (\Delta\boldsymbol{A})^{-1}\bigl(\exp(\Delta\boldsymbol{A}) - \boldsymbol{I}\bigr)\Delta\boldsymbol{B}.
>
> $$
>
> For comparison, first-order Euler discretization gives $\bar{\boldsymbol{A}} = \boldsymbol{I} + \Delta\boldsymbol{A}$, $\bar{\boldsymbol{B}} = \Delta\boldsymbol{B}$, whereas the trapezoidal *Bilinear* (Tustin) method yields $\bar{\boldsymbol{A}} = (\boldsymbol{I} - \frac{\Delta}{2}\boldsymbol{A})^{-1}(\boldsymbol{I} + \frac{\Delta}{2}\boldsymbol{A})$ and $\bar{\boldsymbol{B}} = (\boldsymbol{I} - \frac{\Delta}{2}\boldsymbol{A})^{-1}\Delta\boldsymbol{B}$ [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (PDF pp. 5–6).
>
> **Recurrent–Convolutional Duality (LTI):** Setting $\boldsymbol{h}_{-1} = \boldsymbol{0}$ and omitting the skip term $Dx_t$, unrolling the discrete recurrence $\boldsymbol{h}_t = \bar{\boldsymbol{A}}\boldsymbol{h}_{t-1} + \bar{\boldsymbol{B}}x_t$, $y_t = \boldsymbol{C}\boldsymbol{h}_t$ gives [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (Eqs. (5)–(7), PDF p. 6):
>
> $$
> y_k = \sum_{j=0}^{k} \boldsymbol{C}\bar{\boldsymbol{A}}^{k-j}\bar{\boldsymbol{B}} x_j \quad \Longleftrightarrow \quad \boldsymbol{y} = \bar{\boldsymbol{K}} * \boldsymbol{x}, \quad \bar{\boldsymbol{K}} = \bigl(\boldsymbol{C}\bar{\boldsymbol{B}},  \boldsymbol{C}\bar{\boldsymbol{A}}\bar{\boldsymbol{B}},  \dots,  \boldsymbol{C}\bar{\boldsymbol{A}}^{L-1}\bar{\boldsymbol{B}}\bigr).
>
> $$
>
> Because $\bar{\boldsymbol{K}} \in \mathbb{R}^L$ is independent of the input sequence $\boldsymbol{x}$, training can evaluate `(eq:ssm:conv_dual)` in parallel across all $L$ positions in $O(L \log L)$ time via Fast Fourier Transforms (FFT), while autoregressive inference executes the recurrence step-by-step in $O(1)$ time and $O(N)$ state memory per channel [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (PDF pp. 6–7).

### HiPPO Initialization, Structured S4, and Selective Scan ($S6$)

Randomly initialized LTI state matrices $\boldsymbol{A}$ suffer from vanishing or exploding gradients over long horizons, mirroring classical RNNs [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (PDF p. 7). The *HiPPO* (High-Order Polynomial Projection Operators) framework resolves this by choosing $\boldsymbol{A} \in \mathbb{R}^{N \times N}$ so that $\boldsymbol{h}(t)$ optimally projects the cumulative input history onto orthogonal Legendre polynomials [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (PDF p. 7):

$$
A_{nk} = - \begin{cases}
    (2n + 1)^{1/2}(2k + 1)^{1/2}, & n > k, \\
    n + 1, & n = k, \\
    0, & n < k.
  \end{cases}
$$

HiPPO initialization alone boosts sequential MNIST accuracy from $60\%$ to $98\%$, and *S4 (Structured State Spaces)* decomposes $\boldsymbol{A}$ into Normal Plus Low-Rank (NPLR) form, applying diagonalization, the Woodbury identity, and Cauchy kernel evaluation to reduce kernel computation from $O(N^2 L)$ to $O(N + L)$ [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (PDF pp. 7–9).

However, any LTI model—whether S4 or a global convolution—applies the *exact same* parameters $(\bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \boldsymbol{C})$ to every token $x_t$. It cannot perform content-based filtering (e.g., ignoring filler words or selectively gating induction heads) [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (PDF pp. 9–10). Mamba ($S6$) resolves this by making the SSM parameters functions of the input $x_t \in \mathbb{R}^D$:

$$
\begin{aligned}
  \boldsymbol{B}_t &= s_B(x_t) = \mathrm{Linear}_N(x_t), \qquad
  \boldsymbol{C}_t = s_C(x_t) = \mathrm{Linear}_N(x_t), \\
  \Delta_t &= \mathrm{softplus}\bigl(\text{param} + \mathrm{Broadcast}_D(\mathrm{Linear}_1(x_t))\bigr),
\end{aligned}
$$

followed by ZOH discretization $\bar{\boldsymbol{A}}_t = \exp(\Delta_t \boldsymbol{A})$, $\bar{\boldsymbol{B}}_t = (\Delta_t \boldsymbol{A})^{-1}(\exp(\Delta_t \boldsymbol{A}) - \boldsymbol{I})\Delta_t \boldsymbol{B}_t$ (with diagonal $\boldsymbol{A} \in \mathbb{R}^{D \times N}$) [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (PDF pp. 10–11).

> [!important] Associative Parallel Scan for Time-Varying Recurrences
> Because $(\bar{\boldsymbol{A}}_t, \bar{\boldsymbol{B}}_t, \boldsymbol{C}_t)$ depend on $t$, the convolutional dual `(eq:ssm:conv_dual)` no longer holds. Nevertheless, the time-varying linear recurrence $\boldsymbol{h}_t = \bar{\boldsymbol{A}}_t \boldsymbol{h}_{t-1} + \bar{\boldsymbol{B}}_t x_t$ remains solvable in parallel via an *associative parallel prefix scan* [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (PDF p. 13). Define elements $c_t = (\bar{\boldsymbol{A}}_t, \boldsymbol{b}_t)$ with $\boldsymbol{b}_t = \bar{\boldsymbol{B}}_t x_t$, and the binary operator $\bullet$:
>
> $$
> (\boldsymbol{M}_2, \boldsymbol{v}_2) \bullet (\boldsymbol{M}_1, \boldsymbol{v}_1) \triangleq (\boldsymbol{M}_2 \boldsymbol{M}_1,  \boldsymbol{M}_2 \boldsymbol{v}_1 + \boldsymbol{v}_2).
>
> $$
>
> Since matrix multiplication is associative and distributes over vector addition, $\bullet$ is associative: $(c_3 \bullet c_2) \bullet c_1 = c_3 \bullet (c_2 \bullet c_1)$. Consequently, all prefix states $\boldsymbol{h}_t$ (the second component of $c_t \bullet \dots \bullet c_1$) can be computed in $O(\log L)$ parallel depth ($O(L/T)$ on $T$ processors) [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (PDF p. 13). Strikingly, the exact same associative scan framework reduces classical Kalman filtering and RTS smoothing from $O(T)$ sequential steps to $O(\log T)$ parallel span on GPUs [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 8.2.3.4, p. 372).

## Algorithmic Implementation & Comparative Synthesis

Algorithms `alg:ssm:kalman` and `alg:ssm:mamba_s6` present the complete classical Kalman Filter + RTS Smoother and Mamba's hardware-aware Selective Scan ($S6$).

> [!example] Algorithm: Kalman Filter (Forward) and RTS Smoother (Backward) for LG-SSMs [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Alg. 8.1–8.2, Sec. 8.2.3, pp. 369–371)
> **Input:** Dynamics $(\boldsymbol{F}_{1:T}, \boldsymbol{B}_{1:T}, \boldsymbol{b}_{1:T}, \boldsymbol{Q}_{1:T})$, emissions $(\boldsymbol{H}_{1:T}, \boldsymbol{D}_{1:T}, \boldsymbol{d}_{1:T}, \boldsymbol{R}_{1:T})$, inputs $\boldsymbol{u}_{1:T}$, observations $\boldsymbol{y}_{1:T}$, prior $(\boldsymbol{\mu}_{0|0}, \boldsymbol{\Sigma}_{0|0})$.
> **Output:** Smoothed posteriors $\{(\boldsymbol{\mu}_{t|T}, \boldsymbol{\Sigma}_{t|T})\}_{t=1}^T$ and marginal log-likelihood $\ell_{1:T}$.
> $\ell_{1:T} \gets 0$
> **for** $t = 1, \dots, T$:
>
>    *// Forward Prediction Step*
>   $\boldsymbol{\mu}_{t|t-1} \gets \boldsymbol{F}_t \boldsymbol{\mu}_{t-1|t-1} + \boldsymbol{B}_t \boldsymbol{u}_t + \boldsymbol{b}_t$
>   $\boldsymbol{\Sigma}_{t|t-1} \gets \boldsymbol{F}_t \boldsymbol{\Sigma}_{t-1|t-1} \boldsymbol{F}_t^{\top} + \boldsymbol{Q}_t$
>    *// Forward Measurement Update Step*
>   $\hat{\boldsymbol{y}}_t \gets \boldsymbol{H}_t \boldsymbol{\mu}_{t|t-1} + \boldsymbol{D}_t \boldsymbol{u}_t + \boldsymbol{d}_t$; \quad $\boldsymbol{S}_t \gets \boldsymbol{H}_t \boldsymbol{\Sigma}_{t|t-1} \boldsymbol{H}_t^{\top} + \boldsymbol{R}_t$
>   Solve $\boldsymbol{S}_t^{\top} \boldsymbol{K}_t^{\top} = \boldsymbol{H}_t \boldsymbol{\Sigma}_{t|t-1}$ for $\boldsymbol{K}_t$
>   $\boldsymbol{\mu}_{t|t} \gets \boldsymbol{\mu}_{t|t-1} + \boldsymbol{K}_t(\boldsymbol{y}_t - \hat{\boldsymbol{y}}_t)$; \quad $\boldsymbol{\Sigma}_{t|t} \gets \boldsymbol{\Sigma}_{t|t-1} - \boldsymbol{K}_t \boldsymbol{S}_t \boldsymbol{K}_t^{\top}$
>   $\ell_{1:T} \gets \ell_{1:T} + \log \mathcal{N}(\boldsymbol{y}_t \mid \hat{\boldsymbol{y}}_t, \boldsymbol{S}_t)$
>
>  *// Backward RTS Smoothing Pass*
> **for** $t = T-1, \dots, 1$:
>
>   Solve $\boldsymbol{\Sigma}_{t+1|t}^{\top} \boldsymbol{J}_t^{\top} = \boldsymbol{F}_t \boldsymbol{\Sigma}_{t|t}$ for $\boldsymbol{J}_t$
>   $\boldsymbol{\mu}_{t|T} \gets \boldsymbol{\mu}_{t|t} + \boldsymbol{J}_t(\boldsymbol{\mu}_{t+1|T} - \boldsymbol{\mu}_{t+1|t})$
>   $\boldsymbol{\Sigma}_{t|T} \gets \boldsymbol{\Sigma}_{t|t} + \boldsymbol{J}_t(\boldsymbol{\Sigma}_{t+1|T} - \boldsymbol{\Sigma}_{t+1|t})\boldsymbol{J}_t^{\top}$
>
> **return** $\{(\boldsymbol{\mu}_{t|T}, \boldsymbol{\Sigma}_{t|T})\}_{t=1}^T, \ell_{1:T}$

> [!example] Algorithm: Hardware-Aware Selective Scan ($S6$) with SRAM Kernel Fusion [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (Sec. 2.1–2.2, PDF pp. 10–13)
> **Input:** Input tensor $\boldsymbol{X} \in \mathbb{R}^{B \times L \times D}$ in HBM; diagonal $\boldsymbol{A} \in \mathbb{R}^{D \times N}$, $\boldsymbol{b}_\Delta \in \mathbb{R}^D$ in HBM.
> **Output:** Output tensor $\boldsymbol{Y} \in \mathbb{R}^{B \times L \times D}$ in HBM.
> Compute projections $\boldsymbol{B} = \mathrm{Linear}_N(\boldsymbol{X}) \in \mathbb{R}^{B \times L \times N}$, $\boldsymbol{C} = \mathrm{Linear}_N(\boldsymbol{X}) \in \mathbb{R}^{B \times L \times N}$, and $\boldsymbol{\Delta} = \mathrm{softplus}(\boldsymbol{b}_\Delta + \mathrm{Broadcast}_D(\mathrm{Linear}_1(\boldsymbol{X}))) \in \mathbb{R}^{B \times L \times D}$
>  *// Fused CUDA Kernel: Load $(\boldsymbol{A}, \boldsymbol{\Delta}, \boldsymbol{B}, \boldsymbol{C}, \boldsymbol{X})$ tiles from HBM to SRAM*
> Discretize in SRAM: $\bar{\boldsymbol{A}}_{b,l,d,n} \gets \exp(\Delta_{b,l,d} A_{d,n})$ and $\bar{\boldsymbol{B}}_{b,l,d,n} \gets \Delta_{b,l,d} B_{b,l,n}$ (ZOH/Euler)
> Execute associative parallel scan `(eq:ssm:assoc_op)` in SRAM over sequence length $L$: $\boldsymbol{h}_{b,l,d} = \bar{\boldsymbol{A}}_{b,l,d} \odot \boldsymbol{h}_{b,l-1,d} + \bar{\boldsymbol{B}}_{b,l,d} x_{b,l,d}$
> Project output in SRAM: $y_{b,l,d} \gets \sum_{n=1}^N C_{b,l,n} h_{b,l,d,n}$
> Write only $\boldsymbol{Y} \in \mathbb{R}^{B \times L \times D}$ back to HBM (discard intermediate $\bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \boldsymbol{h}$)
>  *// Backward Pass: Recompute $(\bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \boldsymbol{h})$ in SRAM from saved $(\boldsymbol{X}, \boldsymbol{\Delta}, \boldsymbol{B}, \boldsymbol{C})$ to avoid HBM I/O*
> **return** $\boldsymbol{Y}$

**Table:** Asymptotic complexity and expressivity comparison across sequence length $L$ (or $T$), channel width $D$, and state dimension $N$ (or $N_z$), synthesized from [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (Table 1, PDF p. 3; pp. 8–13), [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Sec. 2.3.2–2.3.3, pp. 68–79), and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 8.2.3.4, p. 372).

| **Architecture** | **Parallel** | **Train** | **Infer/Tok** | **State / Cache** | **Content Selection** |
| — | — | — | — | — | — |
| Transformer | Yes | $O(L^2 D)$ | $O(L D)$ | $O(L D)$ (KV) | Yes (Softmax $QK^{\top}$) |
| Linear Attention | Yes | $O(L D N)$ | $O(D N)$ | $O(D N)$ | Partial (kernel $\phi$) |
| LTI S4 | Yes (FFT) | $O(L \log L \cdot D)$ | $O(D N)$ | $O(D N)$ | No (LTI $\bar{\boldsymbol{A}},\bar{\boldsymbol{B}},\boldsymbol{C}$) |
| Mamba ($S6$) | Yes (Scan) | $O(L D N)$ | $O(D N)$ | $O(D N)$ | Yes ($\Delta_t, \boldsymbol{B}_t, \boldsymbol{C}_t$) |
| Kalman / RTS | Yes (Scan) | $O(T N_z^3)$ / $O(\log T)$ | $O(N_z^3)$ | $O(N_z^2)$ | Gain $\boldsymbol{K}_t$ (SNR) |

> [!info] Cross-Book Synthesis: Fixed-Size Recurrent Memory vs. Unbounded KV Cache
> In [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Sec. 2.3.2–2.3.3, pp. 70–79), sequence models are categorized by how they compress context: standard Transformers maintain an exact, non-compressive KV cache of size $O(L)$, whereas linear attention ($\boldsymbol{\mu}_i = \boldsymbol{\mu}_{i-1} + \boldsymbol{k}_i^{\prime\top} \boldsymbol{v}_i$) and recurrent models compress arbitrarily long histories into a fixed-size state. Similarly, [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 29.13, pp. 1038–1041) distinguishes purely stochastic Deep Markov Models from *Recurrent State-Space Models (RSSMs)* that pair a deterministic recurrent backbone $\boldsymbol{h}_t$ with stochastic latents $\boldsymbol{z}_t$ for multi-step forecasting. Mamba bridges these two lineages: like linear attention and RSSMs, it maintains a fixed $O(DN)$ recurrent state during decoding, yet via input-dependent $(\Delta_t, \boldsymbol{B}_t, \boldsymbol{C}_t)$ it learns *what to write into and erase from* that finite state, achieving $100\%$ accuracy on Induction Heads up to $1\text{M}$ tokens where LTI S4 collapses past $1\text{K}$ tokens [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (Sec. 4, PDF pp. 14–15); [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Sec. 2.3.6.3, p. 93).

## Visual Diagnostics & Geometric Structure

Figure `fig:ssm:kalman_vs_rts` illustrates the foundational distinction between online causal filtering and offline bidirectional smoothing in a 2D tracking LG-SSM: incorporating future observations via the RTS backwards pass shrinks the $95\%$ posterior covariance ellipses and cuts trajectory MSE from $3.13$ to $1.71$.

> [!note] Source Figure
> *Figure:* Kalman filtering vs. RTS smoothing on a 2D linear-Gaussian tracking task: (a) true trajectory (blue squares) and noisy observations (green circles); (b) online Kalman filter posterior means (red) and $95\%$ confidence ellipses ($\text{MSE} = 3.13$); (c) offline RTS smoother ($\text{MSE} = 1.71$) with visibly tighter uncertainty ellipses [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 8, Fig. 8.2, p. 360). (`probabilisticml_kevinmurphy_advancedtopics_p394_fig2.png`)

Figure `fig:ssm:three_views` depicts the three equivalent mathematical realizations of an LTI state-space model—continuous-time ODE, discrete-time recurrence, and global causal convolution—while Figure `fig:ssm:zoh_parallel_scan` formalizes how ZOH discretization and the associative parallel scan unify training parallelism with $O(1)$ per-token inference.

> [!note] Source Figure
> *Figure:* Three representations of an LTI State-Space Model: continuous-time differential system $(\boldsymbol{A},\boldsymbol{B},\boldsymbol{C},\boldsymbol{D})$ for irregular sampling, discretized recurrent form $(\bar{\boldsymbol{A}},\bar{\boldsymbol{B}},\bar{\boldsymbol{C}},\bar{\boldsymbol{D}})$ for $O(1)$ per-token inference, and convolutional kernel $\bar{\boldsymbol{K}} = (\bar{\boldsymbol{C}}\bar{\boldsymbol{A}}^k\bar{\boldsymbol{B}})$ for parallel training [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (PDF p. 7). (`mamba_p7_fig1.png`)

> [!example] Visual Diagram (LaTeX / PDF)
> *End-to-end execution pipeline of Selective State-Space Models ($S6$): continuous parameters $(\boldsymbol{A},\boldsymbol{B},\boldsymbol{C})$ are discretized via input-dependent $\Delta_t$ into $(\bar{\boldsymbol{A}}_t,\bar{\boldsymbol{B}}_t,\boldsymbol{C}_t)$, enabling hardware-fused parallel scan during training and $O(1)$ stateful decoding at inference.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

At the macro-architectural level, Mamba interleaves the $S6$ selective scan with a local 1D depthwise convolution and SiLU ($\sigma$) multiplicative gating, merging the H3 and Gated MLP designs (Figure `fig:ssm:mamba_block`). Figure `fig:ssm:scaling_plot` contrasts the asymptotic per-layer compute and memory scaling of Transformers, LTI S4, and Mamba across context lengths up to $128\text{K}$ tokens.

> [!note] Source Figure
> *Figure:* The Mamba block architecture synthesized by combining H3 (Conv + SSM) with Gated MLP multiplicative SiLU ($\sigma$) projections [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (Fig. 4, PDF p. 14). (`mamba_p14_fig1.png`)

> [!example] Visual Diagram (LaTeX / PDF)
> *Asymptotic scaling of sequence architectures across context length $L$: Mamba eliminates both the quadratic $O(L^2)$ training bottleneck and the linear $O(L)$ inference KV-cache growth of standard Transformers [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (PDF pp. 3, 8–13); [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Sec. 2.3.2, pp. 68–70).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Loss of Covariance Symmetry & Positive Definiteness in Kalman Updates
> Computing the filtered covariance via $\boldsymbol{\Sigma}_{t|t} = \boldsymbol{\Sigma}_{t|t-1} - \boldsymbol{K}_t \boldsymbol{H}_t \boldsymbol{\Sigma}_{t|t-1}$ (Eq. `(eq:ssm:kf_upd_cov)`) in finite-precision arithmetic can destroy symmetry and produce negative eigenvalues when $\boldsymbol{R}_t$ is small or $\boldsymbol{H}_t$ is ill-conditioned, causing subsequent Cholesky factorizations to fail [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 8.2.3.1, p. 370). Similarly, the two-filter smoother suffers numerical singularities at $t=T$ where $\boldsymbol{\Sigma}_{b,T} = \infty \boldsymbol{I}$. **Remedy:** Use the symmetric Joseph-form update $\boldsymbol{\Sigma}_{t|t} = (\boldsymbol{I} - \boldsymbol{K}_t\boldsymbol{H}_t)\boldsymbol{\Sigma}_{t|t-1}(\boldsymbol{I} - \boldsymbol{K}_t\boldsymbol{H}_t)^{\top} + \boldsymbol{K}_t \boldsymbol{R}_t \boldsymbol{K}_t^{\top}$ or a square-root/QR Kalman filter, prefer the RTS smoother over two-filter smoothing, and solve $\boldsymbol{S}_t^{\top} \boldsymbol{K}_t^{\top} = \boldsymbol{H}_t \boldsymbol{\Sigma}_{t|t-1}$ via linear solvers rather than explicit matrix inversion $\boldsymbol{S}_t^{-1}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Fn. 6, p. 366; Sec. 8.2.3.3, p. 372).

> [!warning] Unstable Eigenvalues & Exploding Matrix Powers in $\bar{\boldsymbol{A}}^k$
> If any eigenvalue of the continuous transition matrix $\boldsymbol{A}$ has positive real part ($\mathrm{Re}(\lambda_i(\boldsymbol{A})) > 0$), then $|\lambda_i(\bar{\boldsymbol{A}})| = |e^{\Delta\lambda_i}| > 1$, causing the state $\boldsymbol{h}_t$ and convolution kernel $\boldsymbol{C}\bar{\boldsymbol{A}}^k\bar{\boldsymbol{B}}$ to diverge exponentially with sequence length $L$. **Remedy:** Parameterize diagonal $\boldsymbol{A}$ in log-space as $\boldsymbol{A} = -\exp(\boldsymbol{A}_{\log})$ so $\boldsymbol{A} < 0$ strictly, initialize with HiPPO-LegS (Eq. `(eq:ssm:hippo)`) or its diagonalized S4D spectrum, and enforce $\Delta_t > 0$ via $\mathrm{softplus}$ [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (PDF pp. 7, 11).

> [!warning] HBM Memory-Bandwidth Bottleneck in Naive Selective Scans
> A naive PyTorch implementation of $S6$ materializes the expanded discrete tensors $\bar{\boldsymbol{A}}, \bar{\boldsymbol{B}} \in \mathbb{R}^{B \times L \times D \times N}$ and hidden states $\boldsymbol{h} \in \mathbb{R}^{B \times L \times D \times N}$ in GPU HBM ($\sim 1.5\text{ TB/s}$ bandwidth, $40\text{ GB}$), incurring an $O(BLDN)$ HBM I/O bottleneck that nullifies the $O(L)$ FLOP advantage. **Remedy:** Fuse discretization, associative scan, and output projection inside GPU SRAM ($\sim 19\text{ TB/s}$ bandwidth, $20\text{ MB}$) and recompute intermediate states during the backward pass rather than storing them in HBM [[Source - Report - Mamba When Transformer is Not All You Need - AIVN|mamba]] (Sec. 2.2.1–2.2.3, PDF pp. 11–12).

> [!warning] Finite-State Compression Bottleneck on Exact Associative Recall
> Because an SSM compresses the entire prefix $x_{1:t}$ into a fixed $O(DN)$-dimensional state $\boldsymbol{h}_t$, its memory capacity is fundamentally bounded compared to an $O(LD)$ Transformer KV cache [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Sec. 2.3.3.3, p. 79). On exact verbatim recall benchmarks (e.g., needle-in-a-haystack or multi-query associative recall) where the number of distinct key-value facts exceeds the effective state capacity $DN$, pure SSMs degrade unless state dimension $N$ is scaled or hybrid attention layers are interleaved [[Source - Foundations of Large Language Models|foundations_large_language_models]] (Sec. 2.3.6.3, p. 93).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[Kalman Filter]], [[State Space Models (S4 & Mamba)]]
- **Prerequisites (Directed $\leftarrow$):** [[Kalman Filter]], [[Multivariate Gaussian & Schur Complement]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[State Space Models (S4 & Mamba)]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Kalman Filter]], [[Multivariate Gaussian & Schur Complement]], [[State Space Models (S4 & Mamba)]]
- **Synthesized Sources:** [[Source - Foundations of Large Language Models]], [[Source - Probabilistic Machine Learning]], [[Source - Report - Mamba When Transformer is Not All You Need - AIVN]]
