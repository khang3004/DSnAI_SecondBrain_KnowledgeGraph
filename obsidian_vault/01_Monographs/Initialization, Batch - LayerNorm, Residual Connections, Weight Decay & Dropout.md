---
type: monograph
domain: "D03"
parent_topic: "[[PT 03 - MLPs, Kolmogorov-Arnold Networks, Residuals & Regularization]]"
sub_topic: "[[ST 03 - Feedforward Networks, Normalization, Residuals & KANs]]"
cheatsheet: "[[Cheatsheet - Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]"
part_of: 
  - "[[ST 03 - Feedforward Networks, Normalization, Residuals & KANs]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
concepts: 
  - "[[BatchNorm, LayerNorm & RMSNorm]]"
  - "[[He & Xavier Initialization]]"
  - "[[Residual Connections]]"
  - "[[Weight Decay & Dropout]]"
prerequisites: 
  - "[[MLP & Universal Approximation]]"
enables: 
  - "[[Semantic & Instance Segmentation (FCN, U-Net & SAM)]]"
  - "[[Transformer Architecture]]"
related: 
  - "[[AdamW]]"
  - "[[BatchNorm, LayerNorm & RMSNorm]]"
  - "[[Bias-Variance Trade-off]]"
  - "[[CNN & Equivariance]]"
  - "[[He & Xavier Initialization]]"
  - "[[Residual Connections]]"
  - "[[Transformer Architecture]]"
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-03
---

# Initialization, Batch/LayerNorm, Residual Connections, Weight Decay & Dropout

> [!abstract] Monograph Navigation
> - **Domain:** [[D03 - Neural Architectures, Representations & Regularization|Neural Architectures, Representations & Regularization]]
> - **Parent / Sub-Topic:** [[PT 03 - MLPs, Kolmogorov-Arnold Networks, Residuals & Regularization|MLPs, Kolmogorov-Arnold Networks, Residuals & Regularization]] → [[ST 03 - Feedforward Networks, Normalization, Residuals & KANs|Feedforward Networks, Normalization, Residuals & KANs]]
> - **Visual Cheatsheet:** [[Cheatsheet - Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]
> - **Description:** Synthesizes Xavier/He variance-preserving initialization, BatchNorm vs. LayerNorm vs. RMSNorm gradient conditioning, ResNet highway gradient flow and ensemble-of-paths view, L2 weight decay, and inverted dropout.

## Problem Formulation & Unified Notation

Training deep neural networks of depth $L \gg 1$ requires simultaneously solving three coupled conditioning problems: (i) preserving signal and gradient variance at initialization so forward activations and backward sensitivities neither vanish nor explode; (ii) maintaining well-conditioned intermediate feature distributions and smooth loss landscapes throughout optimization via normalization and residual pathways; and (iii) constraining effective model capacity via explicit and stochastic regularization so that overparameterized networks generalize.

Consider an $L$-layer feedforward network mapping an input $\boldsymbol{x} = \boldsymbol{h}^{(0)} \in \mathbb{R}^{D_0}$ to an output $\hat{\boldsymbol{y}} \in \mathbb{R}^{D_L}$. At layer $l \in \{1, \dots, L\}$ with fan-in $D_{l-1}$ and fan-out $D_l$, the pre-activation $\boldsymbol{f}^{(l)} \in \mathbb{R}^{D_l}$ and post-activation $\boldsymbol{h}^{(l)} \in \mathbb{R}^{D_l}$ evolve according to:

$$
\boldsymbol{f}^{(l)} = \boldsymbol{W}^{(l)} \boldsymbol{h}^{(l-1)} + \boldsymbol{b}^{(l)}, \qquad
  \boldsymbol{h}^{(l)} = \phi\!\left(\boldsymbol{f}^{(l)}\right),
$$

where $\boldsymbol{W}^{(l)} \in \mathbb{R}^{D_l \times D_{l-1}}$ is the weight matrix, $\boldsymbol{b}^{(l)} \in \mathbb{R}^{D_l}$ is the bias vector, and $\phi(\cdot)$ is an element-wise nonlinearity (e.g., $\mathrm{ReLU}(z) = \max(0, z)$, $\tanh$, or GELU). For a mini-batch of $N$ examples (or 4D feature maps $\boldsymbol{X} \in \mathbb{R}^{N \times C \times H \times W}$ with batch size $N$, channels $C$, and spatial dimensions $H \times W$), training minimizes the regularized empirical risk:

$$
\tilde{\mathcal{J}}(\boldsymbol{\theta}) = \frac{1}{N}\sum_{n=1}^{N} \ell\!\left(f(\boldsymbol{x}_n; \boldsymbol{\theta}),  \boldsymbol{y}_n\right) + \lambda \Omega(\boldsymbol{\theta}),
$$

where $\boldsymbol{\theta} = \{\boldsymbol{W}^{(l)}, \boldsymbol{b}^{(l)}\}_{l=1}^L$ collects all trainable parameters, $\Omega(\boldsymbol{\theta})$ is a parameter norm penalty (typically applied only to weights $\boldsymbol{W}^{(l)}$), and $\lambda \ge 0$ governs regularization strength.

> [!info] Cross-Book Notation & Pedagogical Reconciliation
>
> | Concept | Prince | Goodfellow et al. | Bishop & Bishop | Unified Monograph Notation |
> | --- | --- | --- | --- | --- |
> | Layer weights & biases | $\boldsymbol{\Omega}_k, \boldsymbol{\beta}_k$ | $\boldsymbol{W}^{(l)}, \boldsymbol{b}^{(l)}$ | $\boldsymbol{W}, \boldsymbol{w}_0$ (or $\boldsymbol{W}^{(l)}$) | $\boldsymbol{W}^{(l)} \in \mathbb{R}^{D_l \times D_{l-1}}$, $\boldsymbol{b}^{(l)} \in \mathbb{R}^{D_l}$ |
> | Pre-/post-activations | $\boldsymbol{f}_k, \boldsymbol{h}_k = a[\boldsymbol{f}_{k-1}]$ | $\boldsymbol{a}^{(l)}, \boldsymbol{h}^{(l)} = g(\boldsymbol{a}^{(l)})$ | $\boldsymbol{a}, \boldsymbol{z} = h(\boldsymbol{a})$ | $\boldsymbol{f}^{(l)}, \boldsymbol{h}^{(l)} = \phi(\boldsymbol{f}^{(l)})$ |
> | Norm scale & shift | $\gamma_c, \delta_c$ (per channel) | $\gamma, \beta$ | $\gamma_i, \beta_i$ | $\boldsymbol{\gamma}, \boldsymbol{\beta}$ (`scale`, `shift` in Raschka) |
> | $L_2$ penalty coeff. | $\lambda \sum_j \phi_j^2$ | $\frac{\alpha}{2}\lVert \boldsymbol{w} \rVert_2^2$ | $\frac{\lambda}{2}\lVert \boldsymbol{w} \rVert_2^2$ | $\frac{\lambda}{2}\sum_l \lVert \boldsymbol{W}^{(l)} \rVert_F^2$ |
> | Dropout mask | Random zeroing | $\boldsymbol{d} \in \{0,1\}^n$, $\frac{1}{2}\boldsymbol{W}$ at test | Bernoulli mask | $\boldsymbol{m} \sim \mathrm{Bern}(p)$, inverted scaling $1/p$ |
> | Residual block | $\boldsymbol{f}_k[\boldsymbol{x}] + \boldsymbol{x}$ | Skip connections | $\boldsymbol{z}_l = \mathcal{F}_l(\boldsymbol{z}_{l-1}) + \boldsymbol{z}_{l-1}$ | $\boldsymbol{x}^{(l+1)} = \boldsymbol{x}^{(l)} + \mathcal{F}_l(\boldsymbol{x}^{(l)})$ |
>
> *Reconciled Sources:* [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 7, 9, 11), [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 7), [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 7.4, 9), and [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 4.2, 4.4).

## Core Mathematical Derivations & Theorems

### Variance-Preserving Initialization: Xavier/Glorot and Kaiming/He

At initialization, biases are set to $\boldsymbol{b}^{(l)} = \boldsymbol{0}$ and weights are drawn i.i.d. from a zero-mean distribution with variance $\sigma_l^2 = \mathrm{Var}(W_{ij}^{(l)})$. If $\sigma_l^2$ is even slightly miscalibrated, repeated multiplication across $L$ layers causes exponential blow-up or collapse of both forward activations and backward gradients [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 7, PDF pp. 122–125).

> [!important] Forward & Backward Variance Preservation: Xavier vs. Kaiming/He
> Assume $\boldsymbol{h}^{(l-1)}$ has independent, identically distributed components with $\mathbb{E}[h_j^{(l-1)}] = 0$ (or symmetric pre-activations $f_j^{(l-1)}$) and $W_{ij}^{(l)}$ are i.i.d. with $\mathbb{E}[W_{ij}^{(l)}] = 0$, $\mathrm{Var}(W_{ij}^{(l)}) = \sigma_l^2$, independent of $\boldsymbol{h}^{(l-1)}$. Since $\boldsymbol{b}^{(l)} = \boldsymbol{0}$, each pre-activation $f_i^{(l)} = \sum_{j=1}^{D_{l-1}} W_{ij}^{(l)} h_j^{(l-1)}$ satisfies $\mathbb{E}[f_i^{(l)}] = 0$ and:
>
> $$
> \mathrm{Var}\!\left(f_i^{(l)}\right) = \sum_{j=1}^{D_{l-1}} \mathrm{Var}\!\left(W_{ij}^{(l)} h_j^{(l-1)}\right)
>   = \sum_{j=1}^{D_{l-1}} \mathbb{E}\!\left[\left(W_{ij}^{(l)}\right)^2\right] \mathbb{E}\!\left[\left(h_j^{(l-1)}\right)^2\right]
>   = D_{l-1} \sigma_l^2 \mathbb{E}\!\left[\left(h_j^{(l-1)}\right)^2\right].
> $$
>
> **Case 1: Linear / $\tanh$ regime (Xavier/Glorot).** In the linear regime around $0$ ($\phi(z) \approx z$), $\mathbb{E}[(h_j^{(l-1)})^2] = \mathrm{Var}(f_j^{(l-1)})$. Enforcing $\mathrm{Var}(f_i^{(l)}) = \mathrm{Var}(f_j^{(l-1)})$ requires $D_{l-1}\sigma_l^2 = 1$. Repeating the argument for the backward gradient $\boldsymbol{\delta}^{(l-1)} = (\boldsymbol{W}^{(l)})^{\top} \boldsymbol{\delta}^{(l)} \odot \phi'(\boldsymbol{f}^{(l-1)})$ yields $D_l \sigma_l^2 = 1$. Averaging the fan-in and fan-out constraints gives **Xavier/Glorot initialization**:
>
> $$
> \sigma_{\text{Xavier}}^2 = \frac{2}{D_{l-1} + D_l}, \qquad
>   W_{ij}^{(l)} \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{D_{l-1}+D_l}},  \sqrt{\frac{6}{D_{l-1}+D_l}}\right).
> $$
>
> **Case 2: ReLU nonlinearity (Kaiming/He).** For $\phi(z) = \mathrm{ReLU}(z) = \max(0, z)$ and a symmetric zero-mean pre-activation density $p(f) = p(-f)$, half of the pre-activations are clipped to zero, so the second moment of the post-activation is exactly half the pre-activation variance [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 7, Eq. (7.30), PDF p. 123):
>
> $$
> \mathbb{E}\!\left[\left(h_j^{(l-1)}\right)^2\right]
>   = \int_0^{\infty} f^2  p(f) \mathrm{d}f
>   = \frac{1}{2}\int_{-\infty}^{\infty} f^2  p(f) \mathrm{d}f
>   = \frac{1}{2}\mathrm{Var}\!\left(f_j^{(l-1)}\right).
> $$
>
> Substituting `(eq:normres:relu_half)` into `(eq:normres:var_forward)` gives $\mathrm{Var}(f_i^{(l)}) = \frac{1}{2} D_{l-1} \sigma_l^2 \mathrm{Var}(f_j^{(l-1)})$. Preserving variance across all $L$ layers therefore requires **Kaiming/He initialization**:
>
> $$
> \sigma_{\text{He, fwd}}^2 = \frac{2}{D_{l-1}} \quad (\text{fan-in}), \qquad
>   \sigma_{\text{He, bwd}}^2 = \frac{2}{D_l} \quad (\text{fan-out}), \qquad
>   \sigma_{\text{He, avg}}^2 = \frac{4}{D_{l-1} + D_l}.
> $$
>

> [!tip] Why ReLU Needs the $\sqrt{2}$ Weight Standard Deviation Factor
> A ReLU unit sets negative pre-activations to zero, discarding $50\%$ of the signal energy at every layer. Without the factor of $2$ in $\sigma_l^2 = 2/D_{l-1}$ (i.e., $\sqrt{2}$ in weight standard deviation), the activation variance decays geometrically as $(1/2)^L$; after $L = 30$ layers, variance shrinks by $2^{-30} \approx 9.3 \times 10^{-10}$, starving both forward representations and backward gradients.

### Normalization Layers: BatchNorm, LayerNorm, RMSNorm & GroupNorm

 Even with exact initialization, parameter updates during training alter layer statistics and induce ill-conditioned curvature. As shown in [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 7.4, pp. 224–229), [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 11, PDF p. 218), and [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 4.2, PDF pp. 257–268), normalization layers standardize activations along specific tensor axes and re-introduce learnable affine degrees of freedom $\boldsymbol{\gamma}, \boldsymbol{\beta}$.

For a 4D activation tensor $\boldsymbol{X} \in \mathbb{R}^{N \times C \times H \times W}$ (where $N$ is batch size, $C$ is channels/features, and $H, W$ are spatial or sequence dimensions), each normalization family computes mean $\mu_i$ and variance $\sigma_i^2$ over an index set $\mathcal{S}_i$ specific to element $i = (n, c, h, w)$:

$$
\mu_i = \frac{1}{|\mathcal{S}_i|}\sum_{k \in \mathcal{S}_i} X_k, \qquad
  \sigma_i^2 = \frac{1}{|\mathcal{S}_i|}\sum_{k \in \mathcal{S}_i} (X_k - \mu_i)^2, \qquad
  \hat{X}_i = \frac{X_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}, \qquad
  Y_i = \gamma_c \hat{X}_i + \beta_c.
$$

> [!important] Exact Normalization Axes, BatchNorm Jacobian Coupling & RMSNorm
> - **BatchNorm (BN):** $\mathcal{S}_i^{\text{BN}} = \{(n', c, h', w') : n' \in [N],  h' \in [H],  w' \in [W]\}$ with $|\mathcal{S}_i^{\text{BN}}| = NHW$. During training, BN couples all $N$ batch elements via $\mu_c, \sigma_c^2$. Differentiating $\hat{x}_n = (x_n - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$ with respect to $x_n$ yields the coupled mini-batch Jacobian:
>
> $$
> \frac{\partial \mathcal{L}}{\partial x_n} = \frac{1}{\sqrt{\sigma_B^2 + \epsilon}} \left[ \frac{\partial \mathcal{L}}{\partial \hat{x}_n} - \frac{1}{N}\sum_{m=1}^N \frac{\partial \mathcal{L}}{\partial \hat{x}_m} - \frac{\hat{x}_n}{N}\sum_{m=1}^N \frac{\partial \mathcal{L}}{\partial \hat{x}_m} \hat{x}_m \right].
> $$
>
>   At inference, BN replaces mini-batch statistics with exponential moving averages $\hat{\mu}_{\text{run}}, \hat{\sigma}_{\text{run}}^2$ accumulated during training.
> - **LayerNorm (LN):** $\mathcal{S}_i^{\text{LN}} = \{(n, c', h', w') : c' \in [C],  h' \in [H],  w' \in [W]\}$ (or across the last embedding dimension $D$ for sequence tensors $\boldsymbol{X} \in \mathbb{R}^{N \times T \times D}$, as implemented via `dim=-1, unbiased=False` in [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 4.2, Listing 4.2)). LN is completely independent across batch elements and requires no train/eval running statistics.
> - **GroupNorm (GN) & InstanceNorm (IN):** Partition $C$ channels into $G$ groups of $C/G$ channels: $\mathcal{S}_i^{\text{GN}} = \{(n, c', h', w') : \lfloor c'/(C/G)\rfloor = \lfloor c/(C/G)\rfloor,  h' \in [H],  w' \in [W]\}$. Setting $G=1$ recovers LN; setting $G=C$ recovers InstanceNorm [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 11, Fig. 11.14, PDF p. 218).
> - **RMSNorm:** Used in modern LLMs (e.g., LLaMA, Gemma) for computational efficiency [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (PDF p. 705), RMSNorm drops mean centering and normalizes solely by the root mean square over the feature dimension $D$:
>
> $$
> \bar{x}_i = \frac{x_i}{\mathrm{RMS}(\boldsymbol{x})}, \qquad
>     \mathrm{RMS}(\boldsymbol{x}) = \sqrt{\frac{1}{D}\sum_{j=1}^D x_j^2 + \epsilon}, \qquad
>     y_i = \gamma_i \bar{x}_i.
> $$
>
>   Eliminating the mean subtraction saves one reduction pass over $D$ elements while preserving re-scaling invariance: $\mathrm{RMS}(\alpha\boldsymbol{x}) = \alpha \mathrm{RMS}(\boldsymbol{x})$ for $\alpha > 0$.

### Residual Connections: Gradient Highway, Shattered Gradients & Pre-LN vs. Post-LN

In deep plain networks, even with He initialization, gradients become spatially uncorrelated across nearby inputs — a phenomenon termed *shattered gradients* [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 11, Fig. 11.3, PDF p. 202), [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 9.5, Fig. 9.12, p. 274). Residual networks (ResNets) resolve this by adding an identity skip connection around each nonlinear residual block $\mathcal{F}_l$:

$$
\boldsymbol{x}^{(l+1)} = \boldsymbol{x}^{(l)} + \mathcal{F}_l\!\left(\boldsymbol{x}^{(l)};  \boldsymbol{\theta}_l\right).
$$

> [!important] Residual Gradient Highway, Ensemble of Paths & $1/\sqrt{L}$ Variance Scaling
> Unrolling `(eq:normres:residual_block)` from layer $l$ to $L$ yields $\boldsymbol{x}^{(L)} = \boldsymbol{x}^{(l)} + \sum_{k=l}^{L-1} \mathcal{F}_k(\boldsymbol{x}^{(k)})$. Applying the chain rule to a scalar loss $\mathcal{L}$ gives the exact backward gradient:
>
> $$
> \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}^{(l)}}
>   = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}^{(L)}} \left( \boldsymbol{I} + \frac{\partial }{\partial \boldsymbol{x}^{(l)}} \sum_{k=l}^{L-1} \mathcal{F}_k\!\left(\boldsymbol{x}^{(k)}\right) \right)
>   = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}^{(L)}} \prod_{k=l}^{L-1} \left( \boldsymbol{I} + \frac{\partial \mathcal{F}_k(\boldsymbol{x}^{(k)})}{\partial \boldsymbol{x}^{(k)}} \right).
> $$
>
> The additive identity $\boldsymbol{I}$ creates an unimpeded **gradient highway**: $\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}^{(L)}}$ propagates directly to any earlier layer $l$ without being attenuated by repeated matrix multiplications [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 4.4, PDF pp. 280–287). Expanding the nested composition across $L$ blocks reveals an implicit ensemble of $2^L$ computational paths of lengths $0$ through $L$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 9.5, Eq. (9.40), p. 276).
>
> **Variance Growth & $1/\sqrt{L}$ Scaling:** If $\boldsymbol{x}^{(l)}$ and $\mathcal{F}_l(\boldsymbol{x}^{(l)})$ are uncorrelated with $\mathrm{Var}(\boldsymbol{x}^{(l)}) = \sigma_x^2$ and $\mathrm{Var}(\mathcal{F}_l) = \sigma_F^2$, then $\mathrm{Var}(\boldsymbol{x}^{(l+1)}) = \mathrm{Var}(\boldsymbol{x}^{(l)}) + \mathrm{Var}(\mathcal{F}_l)$ doubles at each block if $\sigma_F^2 = \sigma_x^2$, causing exponential variance growth $2^L$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 11, PDF p. 206). To keep $\mathrm{Var}(\boldsymbol{x}^{(L)}) = \mathcal{O}(1)$ at depth $L$ without normalization, residual branches must be scaled by $1/\sqrt{L}$ (or $1/\sqrt{2L}$ in Transformers with two sub-layers per block), yielding $\mathrm{Var}(\boldsymbol{x}^{(L)}) = \mathrm{Var}(\boldsymbol{x}^{(0)}) + \sum_{l=0}^{L-1} \frac{1}{L}\sigma_F^2 = \sigma_0^2 + \sigma_F^2$.

In Transformer architectures, placement of LayerNorm/RMSNorm relative to the residual addition critically governs gradient conditioning [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (PDF p. 704):
- **Post-LN:** $\boldsymbol{x}^{(l+1)} = \mathrm{Norm}\!\left(\boldsymbol{x}^{(l)} + \mathcal{F}_l(\boldsymbol{x}^{(l)})\right)$. The identity path passes *through* the normalization Jacobian at every block, causing large gradients near the output layer and requiring careful learning-rate warmup.
- **Pre-LN:** $\boldsymbol{x}^{(l+1)} = \boldsymbol{x}^{(l)} + \mathcal{F}_l\!\left(\mathrm{Norm}(\boldsymbol{x}^{(l)})\right)$. The identity skip connection remains completely unmodulated from input to output, bounding gradient norms across depth and enabling stable training without aggressive warmup.

### Regularization: Weight Decay, Dropout, Label Smoothing, Mixup & Early Stopping

> [!important] $L_2$ Weight Decay Curvature Filtering & Inverted Dropout Ensemble
> **1. $L_2$ Weight Decay as Spectral Curvature Filtering:** Consider minimizing $\tilde{\mathcal{J}}(\boldsymbol{w}) = \mathcal{J}(\boldsymbol{w}) + \frac{\lambda}{2}\lVert \boldsymbol{w} \rVert_2^2$. A single gradient step yields $\boldsymbol{w} \leftarrow (1 - \eta\lambda)\boldsymbol{w} - \eta \nabla_{\boldsymbol{w}}\mathcal{J}(\boldsymbol{w})$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 7.1.1, Eq. (7.5), p. 231). Expanding $\mathcal{J}(\boldsymbol{w})$ to second order around the unregularized minimum $\boldsymbol{w}^*$ with Hessian $\boldsymbol{H} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\top}$ gives:
>
> $$
> (\boldsymbol{H} + \lambda\boldsymbol{I})\tilde{\boldsymbol{w}} = \boldsymbol{H}\boldsymbol{w}^*
>   \implies \tilde{w}_i' = \frac{\Lambda_{ii}}{\Lambda_{ii} + \lambda} (w_i^*)' \quad \text{in the eigenbasis of } \boldsymbol{H}.
> $$
>
> Directions with high curvature ($\Lambda_{ii} \gg \lambda$) are preserved ($\frac{\Lambda_{ii}}{\Lambda_{ii}+\lambda} \approx 1$), whereas flat, low-curvature directions ($\Lambda_{ii} \ll \lambda$) are shrunk toward zero [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 7.1.1, pp. 232–234), [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 9.2, p. 262).
>
> **2. Inverted Dropout & Geometric-Mean Ensemble:** During training, each unit in $\boldsymbol{h}^{(l)}$ is multiplied by an independent Bernoulli mask $m_j \sim \mathrm{Bern}(p)$ and rescaled by $1/p$ (*inverted dropout*) so that $\mathbb{E}[\tilde{h}_j^{(l)}] = h_j^{(l)}$ and no weight rescaling is needed at test time. For a softmax classifier with weights $\boldsymbol{W}$ and binary mask $\boldsymbol{d} \in \{0,1\}^D$ ($p=1/2$), the geometric mean of the $2^D$ sub-networks is [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 7.12, Eqs. (7.60)–(7.66), p. 264):
>
> $$
> \tilde{P}_{\text{ens}}(y \mid \boldsymbol{v}) \propto \left(\prod_{\boldsymbol{d}\in\{0,1\}^D} \exp\!\left(\boldsymbol{W}_{y,:}(\boldsymbol{d}\odot\boldsymbol{v}) + b_y\right)\right)^{2^{-D}}
>   = \exp\!\left(\frac{1}{2}\boldsymbol{W}_{y,:}\boldsymbol{v} + b_y\right),
> $$
>
> showing that weight scaling at inference is *exact* for single-layer softmax/linear models and a strong approximation for deep nonlinear networks.

Complementing weight decay and dropout, three widely used regularizers act directly on targets, inputs, or optimization trajectories [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 9, pp. 142–157), [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 9.3, pp. 266–268):
- **Label Smoothing:** Replaces one-hot target $\boldsymbol{y} \in \{0,1\}^K$ with $\tilde{\boldsymbol{y}} = (1 - \varepsilon)\boldsymbol{y} + \frac{\varepsilon}{K}\boldsymbol{1}$, preventing logit divergence ($\lVert \boldsymbol{z} \rVert \to \infty$) and improving calibration.
- **Mixup:** Constructs convex combinations $\tilde{\boldsymbol{x}} = \alpha\boldsymbol{x}_i + (1-\alpha)\boldsymbol{x}_j$, $\tilde{\boldsymbol{y}} = \alpha\boldsymbol{y}_i + (1-\alpha)\boldsymbol{y}_j$ with $\alpha \sim \mathrm{Beta}(\nu,\nu)$, enforcing linear interpolation between training manifolds.
- **Early Stopping:** Halting SGD after $T$ iterations acts like $L_2$ regularization with effective penalty $\lambda \propto (\eta T)^{-1}$ along low-curvature Hessian eigenvectors 
[[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 9.3.1, Fig. 9.8, p. 268).

## Algorithmic Implementation & Comparative Synthesis

Algorithm `alg:normres:preln_bn` details the exact forward mechanics of a Pre-LN RMSNorm Residual Block (standard in modern LLMs) alongside the dual train/eval execution paths of BatchNorm.

> [!example] Algorithm: Pre-LN RMSNorm Residual Block & BatchNorm Train/Eval Execution
> **Input:** Input $\boldsymbol{x}^{(l)} \in \mathbb{R}^D$ (or mini-batch $\boldsymbol{X} \in \mathbb{R}^{N \times C \times H \times W}$); RMSNorm gain $\boldsymbol{\gamma}$; BN params $(\boldsymbol{\gamma}_{\text{bn}}, \boldsymbol{\beta}_{\text{bn}}, \hat{\boldsymbol{\mu}}, \hat{\boldsymbol{\sigma}}^2)$, momentum $\rho \in (0,1)$, $\epsilon = 10^{-5}$.
> **Output:** Updated hidden state $\boldsymbol{x}^{(l+1)}$ or normalized tensor $\boldsymbol{Y}$.
>
> **Procedure 1: Pre-LN RMSNorm Residual Block**
> $\mathrm{rms} \leftarrow \sqrt{\frac{1}{D}\sum_{j=1}^D \left(x_j^{(l)}\right)^2 + \epsilon}$
> $\bar{\boldsymbol{x}} \leftarrow \boldsymbol{\gamma} \odot \left(\boldsymbol{x}^{(l)} / \mathrm{rms}\right)$  *// No mean centering; invariant to positive rescaling*
> $\boldsymbol{x}^{(l+1)} \leftarrow \boldsymbol{x}^{(l)} + \mathcal{F}_l(\bar{\boldsymbol{x}})$  *// Unmodulated identity skip path*
>
> **Procedure 2: BatchNorm Forward Pass (Train vs. Eval)**
> **if** `training == True`:
>
>   Compute batch statistics $\boldsymbol{\mu}_B, \boldsymbol{\sigma}_B^2$ over $(N, H, W)$ for each channel $c \in [C]$
>   $\hat{\boldsymbol{X}}_{:,c,:,:} \leftarrow (\boldsymbol{X}_{:,c,:,:} - \mu_{B,c}) / \sqrt{\sigma_{B,c}^2 + \epsilon}$
>   Update running buffers: $\hat{\boldsymbol{\mu}} \leftarrow (1-\rho)\hat{\boldsymbol{\mu}} + \rho\boldsymbol{\mu}_B$,   $\hat{\boldsymbol{\sigma}}^2 \leftarrow (1-\rho)\hat{\boldsymbol{\sigma}}^2 + \frac{N}{N-1}\rho\boldsymbol{\sigma}_B^2$
>
> **else**:
>
>   $\hat{\boldsymbol{X}}_{:,c,:,:} \leftarrow (\boldsymbol{X}_{:,c,:,:} - \hat{\mu}_c) / \sqrt{\hat{\sigma}_c^2 + \epsilon}$  *// Frozen running stats; no batch coupling*
>
> $\boldsymbol{Y}_{:,c,:,:} \leftarrow \gamma_{\text{bn},c}\hat{\boldsymbol{X}}_{:,c,:,:} + \beta_{\text{bn},c}$

**Table:** Comparison of Normalization Schemes across Axes, Complexity, and Inductive Properties.

| Method | Reduction Axes | Params | Train/Eval Match | Primary Domain & Trade-offs |
| --- | --- | --- | --- | --- |
| BatchNorm | $(N, H, W)$ | $2C$ | No (EMA) | Vision CNNs; smooths loss landscape, fails at small $N$ |
| LayerNorm | $(C, H, W)$ / $D$ | $2D$ | Yes | Transformers/NLP; batch-independent, mean-centered |
| RMSNorm | Feature dim $D$ | $D$ | Yes | Modern LLMs; faster than LN, drops shift invariance |
| GroupNorm | $(C/G, H, W)$ | $2C$ | Yes | Detection/segmentation with small $N$; interpolates LN–IN |

> [!info] Cross-Book Synthesis: Why Normalization and Residuals Co-Evolve
> Prince [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 11, PDF p. 218) highlights Santurkar et al.'s finding that BatchNorm's primary benefit is not reducing internal covariate shift, but rather smoothing the loss surface and bounding gradient Lipschitzness. Bishop & Bishop [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 9.5, Fig. 9.14, p. 276) show the exact same landscape-smoothing phenomenon for residual connections in 56-layer networks. Meanwhile, Raschka [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (Ch. 4.2, 4.4, PDF pp. 266, 287) demonstrates empirically in GPT-style Transformers that combining Pre-LN with shortcut connections eliminates vanishing gradients across depth without batch-size dependence.

## Visual Diagnostics & Geometric Structure

Figure `fig:normres:shattered` illustrates the shattered-gradients pathology in plain ReLU networks as depth grows from $1$ to $24$ hidden layers, while Figure `fig:normres:norm_cubes` visualizes the tensor-slicing geometry of BatchNorm, GhostNorm, LayerNorm, GroupNorm, and InstanceNorm. Figure `fig:normres:loss_surface` contrasts the chaotic non-convex landscape of a 56-layer plain network against the smooth basin induced by residual connections.

> [!note] Source Figure
> *Figure:* Shattered gradients in plain ReLU networks: as depth increases from 1 to 24 layers, the gradient $\mathrm{d}y/\mathrm{d}x$ becomes white-noise-like and its spatial autocorrelation collapses to zero [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 11, Fig. 11.3, PDF p. 202). (`understandingdl_simonprince_p202_fig2.png`)

> [!note] Source Figure
> *Figure:* Normalization tensor-slicing schemes across batch, channel, and spatial/position axes for BatchNorm, GhostNorm, LayerNorm, GroupNorm, and InstanceNorm [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 11, Fig. 11.14, PDF p. 218). (`understandingdl_simonprince_p218_fig1.png`)

> [!note] Source Figure
> *Figure:* Loss surface visualization for a 56-layer network (a) without residual connections and (b) with residual connections, demonstrating dramatic smoothing of local minima and ridges [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 9.5, Fig. 9.14, p. 276). (`deep_learning_foundations_concepts_p293_fig1.png`)

Figure `fig:normres:pre_post_ln` compares the computational graphs of Post-LN and Pre-LN residual blocks, highlighting how Pre-LN keeps the identity gradient highway unobstructed. Figure `fig:normres:var_prop` plots theoretical forward activation variance across $30$ ReLU layers under Xavier ($\sigma^2 = 1/D$), over-scaled ($\sigma^2 = 4/D$), and Kaiming/He ($\sigma^2 = 2/D$) initialization, alongside residual gradient norm preservation.

> [!example] Visual Diagram (LaTeX / PDF)
> *Post-LN vs. Pre-LN residual block topology. In Post-LN, the skip connection passes through $\mathrm{Norm}$ at every layer; in Pre-LN, the identity path bypasses normalization, preserving $\boldsymbol{I}$ in the backward Jacobian.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Forward pre-activation variance propagation across 30 ReLU layers ($\mathrm{Var}(f^{(0)})=1$). Omitting the factor of $2$ in He initialization causes $2^{-l}$ exponential collapse; over-scaling causes $2^l$ explosion.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Post-LN Warmup Instability
> In Post-LN Transformers, gradients near the output layer scale as $\mathcal{O}(1)$ while deeper layers receive attenuated updates, causing divergence at standard learning rates without a linear learning-rate warmup. **Remedy:** Prefer Pre-LN (or Pre-RMSNorm) with a final normalization layer before the output projection, or apply a linear warmup over the first $1$–$5\%$ of training steps [[Source - Build a Large Language Model (From Scratch) (for sam speaks)|build_large_language_model_from_scratch]] (PDF p. 704).

> [!warning] BatchNorm Batch-Size Collapse & Train/Eval Mismatch
> When mini-batch size $N \le 4$ (e.g., high-resolution object detection or 3D medical imaging), mini-batch variance estimates $\sigma_B^2$ become noisy and biased, degrading accuracy; additionally, forgetting `model.eval()` causes inference predictions to depend on test-batch composition. **Remedy:** Switch to GroupNorm ($G=32$) or LayerNorm for small-batch regimes, and enforce `model.eval()` before validation/deployment [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 11, PDF p. 217).

> [!warning] Dropout Variance Shift Before BatchNorm
> Placing Dropout immediately before BatchNorm creates a train/eval variance mismatch: during training, inverted dropout multiplies feature variance by $1/p$, which BatchNorm absorbs into $\sigma_B^2$; at evaluation, dropout is disabled (variance drops by $p$), yet BatchNorm uses the inflated training running variance $\hat{\sigma}_{\text{run}}^2$. **Remedy:** Apply Dropout *after* normalization + nonlinearity (or omit Dropout in BatchNorm-heavy convolutional blocks).

> [!warning] Forgetting the $\sqrt{2}$ Factor in ReLU Initialization
> Applying Xavier/Glorot initialization ($\sigma_w^2 = 1/D_{\text{in}}$) to deep ReLU or GELU networks without skip connections shrinks activation variance by $1/2$ per layer, collapsing gradients in early layers before training begins. **Remedy:** Use Kaiming/He initialization ($\sigma_w^2 = 2/D_{\text{in}}$) for ReLU-family activations [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 7, Eq. (7.31), PDF p. 124).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (4):** [[BatchNorm, LayerNorm & RMSNorm]], [[He & Xavier Initialization]], [[Residual Connections]], [[Weight Decay & Dropout]]
- **Prerequisites (Directed $\leftarrow$):** [[MLP & Universal Approximation]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Semantic & Instance Segmentation (FCN, U-Net & SAM)]], [[Transformer Architecture]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[AdamW]], [[BatchNorm, LayerNorm & RMSNorm]], [[Bias-Variance Trade-off]], [[CNN & Equivariance]], [[He & Xavier Initialization]], [[Residual Connections]], [[Transformer Architecture]]
- **Synthesized Sources:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Understanding Deep Learning]]
