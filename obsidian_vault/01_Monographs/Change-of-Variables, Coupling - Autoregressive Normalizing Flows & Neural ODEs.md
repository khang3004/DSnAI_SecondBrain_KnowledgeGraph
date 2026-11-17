---
type: monograph
domain: "D04"
parent_topic: "[[PT 04 - Autoencoders, VAEs, Normalizing Flows, EBMs & GANs]]"
sub_topic: "[[ST 04 - Autoencoders, Variational Autoencoders & Normalizing Flows]]"
cheatsheet: "[[Cheatsheet - Change-of-Variables, Coupling - Autoregressive Normalizing Flows & Neural ODEs]]"
part_of: 
  - "[[ST 04 - Autoencoders, Variational Autoencoders & Normalizing Flows]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
concepts: 
  - "[[Normalizing Flows]]"
prerequisites: 
  - "[[Matrix Calculus & Jacobians]]"
enables: []
related: 
  - "[[Flow Matching & Rectified Flow]]"
  - "[[Matrix Calculus & Jacobians]]"
  - "[[Probability Flow ODE]]"
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - The Principles of Diffusion Models]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-04
---

# Change-of-Variables, Coupling/Autoregressive Normalizing Flows & Neural ODEs

> [!abstract] Monograph Navigation
> - **Domain:** [[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion|Deep Generative Modeling: VAEs, Flows, EBMs, GANs & Diffusion]]
> - **Parent / Sub-Topic:** [[PT 04 - Autoencoders, VAEs, Normalizing Flows, EBMs & GANs|Autoencoders, VAEs, Normalizing Flows, EBMs & GANs]] → [[ST 04 - Autoencoders, Variational Autoencoders & Normalizing Flows|Autoencoders, Variational Autoencoders & Normalizing Flows]]
> - **Visual Cheatsheet:** [[Cheatsheet - Change-of-Variables, Coupling - Autoregressive Normalizing Flows & Neural ODEs]]
> - **Description:** Derives exact log-likelihood training via change-of-variables Jacobian determinants, RealNVP/Glow affine coupling, MAF/IAF triangular Jacobians, continuous normalizing flows (Neural ODEs), and Hutchinson trace estimators.

## Problem Formulation & Unified Notation

Unlike variational autoencoders (VAEs), which optimize a variational lower bound on the marginal log-likelihood, or generative adversarial networks (GANs), which bypass explicit density evaluation altogether, *normalizing flows* (NFs) construct tractable generative models that support both exact likelihood evaluation and fast ancestral sampling [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 23, p. 829 (PDF p. 863)), [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, p. 547 (PDF p. 559)). A normalizing flow models a complex target distribution $p_X(\boldsymbol{x})$ over $\mathbb{R}^D$ by transforming a simple, analytically tractable base distribution $p_Z(\boldsymbol{z})$ (typically an isotropic Gaussian $\mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_D)$) through a smooth, invertible diffeomorphism $f_\theta : \mathbb{R}^D \to \mathbb{R}^D$:

$$
\boldsymbol{x} = f_\theta(\boldsymbol{z}), \qquad \boldsymbol{z} \sim p_Z(\boldsymbol{z}), \qquad \boldsymbol{z} = f_\theta^{-1}(\boldsymbol{x}) = g_\theta(\boldsymbol{x}).
$$

The term *normalizing flow* reflects the dual directionality of the map: the *generative* (forward) direction $f_\theta$ pushes base samples $\boldsymbol{z}$ toward data $\boldsymbol{x}$, while the *normalizing* (inverse) direction $g_\theta = f_\theta^{-1}$ maps complex data $\boldsymbol{x}$ back to a normalized base distribution $p_Z$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, p. 549 (PDF p. 561)), [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 16, p. 305 (PDF p. 319)).

Because $f_\theta$ is a bijection with $\dim(\boldsymbol{z}) = \dim(\boldsymbol{x}) = D$, the exact log-density of any observation $\boldsymbol{x} \in \mathbb{R}^D$ is given by the change-of-variables formula:

$$
\log p_X(\boldsymbol{x}; \theta) = \log p_Z\!\left(f_\theta^{-1}(\boldsymbol{x})\right) + \log \lvert \det \boldsymbol{J}_{f_\theta^{-1}}(\boldsymbol{x}) \rvert = \log p_Z(\boldsymbol{z}) - \log \lvert \det \boldsymbol{J}_{f_\theta}(\boldsymbol{z}) \rvert,
$$

where $\boldsymbol{J}_{f_\theta}(\boldsymbol{z}) = \frac{\partial f_\theta(\boldsymbol{z})}{\partial \boldsymbol{z}} \in \mathbb{R}^{D \times D}$ denotes the Jacobian matrix. Given i.i.d. training data $\mathcal{D} = \{\boldsymbol{x}^{(n)}\}_{n=1}^N$, parameters $\theta$ are learned by exact maximum likelihood estimation (MLE), minimizing the forward Kullback–Leibler divergence $D_{\mathrm{KL}}(p_{\mathrm{data}} \parallel p_X(\cdot; \theta))$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 16, p. 320 (PDF p. 334)), [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 5.1.1, p. 130 (PDF p. 135)):

$$
\theta^\star = \arg\max_\theta \frac{1}{N}\sum_{n=1}^N \left[ \log p_Z\!\left(f_\theta^{-1}(\boldsymbol{x}^{(n)})\right) + \log \lvert \det \boldsymbol{J}_{f_\theta^{-1}}(\boldsymbol{x}^{(n)}) \rvert \right].
$$

To make `(eq:flow:mle)` computationally viable in high dimensions, an invertible architecture must satisfy two structural criteria: (i) $f_\theta$ and $f_\theta^{-1}$ must be closed-form or numerically tractable, and (ii) $\log \lvert \det \boldsymbol{J}_{f_\theta} \rvert$ must be computable in $O(D)$ or $O(D^2)$ time rather than the generic $O(D^3)$ cost of dense matrix determinants [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, p. 548 (PDF p. 560)).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Source textbooks adopt distinct conventions for latent variables, forward versus inverse directions, and continuous-time indexing. Table `tab:flow:notation` reconciles these conventions into the unified notation used throughout this monograph.
>
> 
> {
>
> | **Concept** | **Murphy Adv.** | **Bishop** | **Prince** | **Unified Notation** |
> | — | — | — | — | — |
> | Base latent | $\boldsymbol{u} \sim p(\boldsymbol{u})$ | $\boldsymbol{z} \sim p_z(\boldsymbol{z})$ | $\boldsymbol{z} \sim \Pr(\boldsymbol{z})$ | $\boldsymbol{z} \sim p_Z(\boldsymbol{z})$ |
> | Forward / Inverse | $\boldsymbol{x} = f(\boldsymbol{u})$, $\boldsymbol{u} = g(\boldsymbol{x})$ | $\boldsymbol{x} = f(\boldsymbol{z}, \boldsymbol{w})$, $\boldsymbol{z} = g(\boldsymbol{x}, \boldsymbol{w})$ | $\boldsymbol{x} = \mathrm{f}[\boldsymbol{z}, \boldsymbol{\phi}]$, $\boldsymbol{z} = \mathrm{f}^{-1}[\boldsymbol{x}, \boldsymbol{\phi}]$ | $\boldsymbol{x} = f_\theta(\boldsymbol{z})$, $\boldsymbol{z} = f_\theta^{-1}(\boldsymbol{x})$ |
> | Coupling split | $\boldsymbol{u} = (\boldsymbol{u}_A, \boldsymbol{u}_B)$, $\Theta(\boldsymbol{u}_B)$ | $\boldsymbol{z} = (\boldsymbol{z}_A, \boldsymbol{z}_B)$, $s(\boldsymbol{z}_A), b(\boldsymbol{z}_A)$ | $\boldsymbol{h} = [\boldsymbol{h}_1^{\top}, \boldsymbol{h}_2^{\top}]^{\top}$, $\boldsymbol{\beta}, \boldsymbol{\gamma}$ | $\boldsymbol{z} = (\boldsymbol{z}_A, \boldsymbol{z}_B)$, $\boldsymbol{s}(\boldsymbol{z}_A), \boldsymbol{t}(\boldsymbol{z}_A)$ |
> | ODE vector field | $\frac{\mathrm{d}\boldsymbol{x}}{\mathrm{d}t} = F(\boldsymbol{x}(t), t)$ | $\frac{\mathrm{d}\boldsymbol{z}(t)}{\mathrm{d}t} = f(\boldsymbol{z}(t), \boldsymbol{w})$ | — | $\frac{\mathrm{d}\boldsymbol{z}(t)}{\mathrm{d}t} = \boldsymbol{v}_\theta(\boldsymbol{z}(t), t)$ |
> | Time convention | $t=0$ base $\to$ $t=T$ data | $t=0$ base $\to$ $t=T$ data | — | $t=0$ base $\to$ $t=T$ data |
> }

## Core Mathematical Derivations & Theorems

> [!important] Change-of-Variables Theorem & Compositional Additivity
> Let $f : \mathbb{R}^D \to \mathbb{R}^D$ be a $C^1$-diffeomorphism with inverse $g = f^{-1}$, and let $\boldsymbol{z} \sim p_Z(\boldsymbol{z})$. For any measurable subset $\mathcal{A} \subseteq \mathbb{R}^D$, conservation of probability mass requires
>
> $$
> \int_{\mathcal{A}} p_X(\boldsymbol{x}) \mathrm{d}\boldsymbol{x} = \int_{g(\mathcal{A})} p_Z(\boldsymbol{z}) \mathrm{d}\boldsymbol{z}.
>
> $$
>
> Applying the multivariate substitution $\boldsymbol{z} = g(\boldsymbol{x})$ with infinitesimal volume element $\mathrm{d}\boldsymbol{z} = \lvert \det \boldsymbol{J}_g(\boldsymbol{x}) \rvert \mathrm{d}\boldsymbol{x}$ yields
>
> $$
> \int_{\mathcal{A}} p_X(\boldsymbol{x}) \mathrm{d}\boldsymbol{x} = \int_{\mathcal{A}} p_Z(g(\boldsymbol{x})) \lvert \det \boldsymbol{J}_g(\boldsymbol{x}) \rvert \mathrm{d}\boldsymbol{x} \implies p_X(\boldsymbol{x}) = p_Z(g(\boldsymbol{x})) \lvert \det \boldsymbol{J}_g(\boldsymbol{x}) \rvert.
>
> $$
>
> By the inverse function theorem, differentiating $f(g(\boldsymbol{x})) = \boldsymbol{x}$ gives $\boldsymbol{J}_f(g(\boldsymbol{x})) \boldsymbol{J}_g(\boldsymbol{x}) = \boldsymbol{I}_D$, whence $\det \boldsymbol{J}_g(\boldsymbol{x}) = \left(\det \boldsymbol{J}_f(\boldsymbol{z})\right)^{-1}$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, Ex. 18.1, p. 559 (PDF p. 571)).
>
> Furthermore, if $f = f_L \circ \dots \circ f_1$ is a composition of $L$ bijections with $\boldsymbol{h}_0 = \boldsymbol{z}$, $\boldsymbol{h}_k = f_k(\boldsymbol{h}_{k-1})$, and $\boldsymbol{h}_L = \boldsymbol{x}$, then $g = f_1^{-1} \circ \dots \circ f_L^{-1}$, and by the chain rule and multiplicativity of determinants:
>
> $$
> \log p_X(\boldsymbol{x}) = \log p_Z(\boldsymbol{z}) - \sum_{k=1}^L \log \lvert \det \boldsymbol{J}_{f_k}(\boldsymbol{h}_{k-1}) \rvert = \log p_Z(\boldsymbol{z}) + \sum_{k=1}^L \log \lvert \det \boldsymbol{J}_{f_k^{-1}}(\boldsymbol{h}_k) \rvert.
>
> $$

> [!tip] Jacobian Expansion, Compression & Density Conservation
> In 1D, `(eq:flow:cov_proof)` reduces to $p_X(x) = p_Z(z) \lvert f'(z) \rvert^{-1}$. Because total probability mass is conserved, stretching a local interval ($\lvert f'(z) \rvert > 1$) dilutes the probability density ($p_X(x) < p_Z(z)$), whereas compressing an interval ($\lvert f'(z) \rvert < 1$) concentrates probability mass ($p_X(x) > p_Z(z)$) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 16, p. 304 (PDF p. 318)). In $\mathbb{R}^D$, $\lvert \det \boldsymbol{J}_f(\boldsymbol{z}) \rvert$ measures the local $D$-dimensional volume scaling factor of the parallelotope spanned by the columns of $\boldsymbol{J}_f(\boldsymbol{z})$.

### Affine Coupling Layers & Invertible Convolutions (RealNVP & Glow)

Linear transformations $\boldsymbol{x} = \boldsymbol{A}\boldsymbol{z} + \boldsymbol{b}$ are invertible when $\det\boldsymbol{A} \ne 0$, yet closed under composition: any stack of affine maps preserves Gaussianity [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 23, p. 832), [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, p. 549). To introduce arbitrary non-linearities while retaining a triangular Jacobian, *RealNVP* (Real-valued Non-Volume-Preserving) partitions $\boldsymbol{z} = (\boldsymbol{z}_A, \boldsymbol{z}_B) \in \mathbb{R}^d \times \mathbb{R}^{D-d}$, leaving $\boldsymbol{z}_A$ unchanged and applying an elementwise affine map to $\boldsymbol{z}_B$ conditioned on $\boldsymbol{z}_A$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, pp. 549–551), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 23, p. 835):

> [!important] Affine Coupling Layer & Glow $1\times 1$ LU Convolution
> For a partition $\boldsymbol{z} = (\boldsymbol{z}_A, \boldsymbol{z}_B)$, the forward and inverse RealNVP coupling transformations are
>
> $$
> \begin{cases}
>     \boldsymbol{x}_A = \boldsymbol{z}_A, \\
>     \boldsymbol{x}_B = \exp(\boldsymbol{s}(\boldsymbol{z}_A)) \odot \boldsymbol{z}_B + \boldsymbol{t}(\boldsymbol{z}_A),
>   \end{cases}
>   \Longleftrightarrow
>   \begin{cases}
>     \boldsymbol{z}_A = \boldsymbol{x}_A, \\
>     \boldsymbol{z}_B = \exp(-\boldsymbol{s}(\boldsymbol{x}_A)) \odot (\boldsymbol{x}_B - \boldsymbol{t}(\boldsymbol{x}_A)),
>   \end{cases}
>
> $$
>
> where $\boldsymbol{s}, \boldsymbol{t} : \mathbb{R}^d \to \mathbb{R}^{D-d}$ are arbitrary (non-invertible) neural networks and $\odot$ is the Hadamard product. Differentiating `(eq:flow:realnvp)` yields a block lower-triangular Jacobian:
>
> $$
> \boldsymbol{J}_f(\boldsymbol{z}) = \begin{bmatrix} \boldsymbol{I}_d & \boldsymbol{0} \\ \frac{\partial \boldsymbol{x}_B}{\partial \boldsymbol{z}_A} & \mathrm{diag}\!\left(\exp(\boldsymbol{s}(\boldsymbol{z}_A))\right) \end{bmatrix} \implies \log \lvert \det \boldsymbol{J}_f(\boldsymbol{z}) \rvert = \sum_{j=1}^{D-d} s_j(\boldsymbol{z}_A).
>
> $$
>
> Notice that the off-diagonal block $\frac{\partial \boldsymbol{x}_B}{\partial \boldsymbol{z}_A}$ never enters the determinant, so $\boldsymbol{s}$ and $\boldsymbol{t}$ require neither invertibility nor Jacobian evaluation, and both $f$ and $f^{-1}$ execute in a single parallel pass [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, p. 551 (PDF p. 563)).
>
> Because $\boldsymbol{z}_A$ is copied verbatim, coupling layers must alternate partitions (e.g., via index reversal or learned channel mixing). *Glow* replaces fixed permutations on $H \times W \times C$ tensors with an invertible $1\times 1$ convolution $\boldsymbol{x}_{h,w,:} = \boldsymbol{W}\boldsymbol{z}_{h,w,:}$ ($\boldsymbol{W} \in \mathbb{R}^{C \times C}$), whose log-determinant is $HW \log\lvert \det \boldsymbol{W} \rvert$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, p. 551 (PDF p. 563)), [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 16, p. 318 (PDF p. 332)). Parameterizing $\boldsymbol{W}$ via an LU decomposition
>
> $$
> \boldsymbol{W} = \boldsymbol{P}\boldsymbol{L}\left(\boldsymbol{U} + \mathrm{diag}(\boldsymbol{s})\right),
>
> $$
>
> where $\boldsymbol{P}$ is a fixed permutation matrix, $\boldsymbol{L}$ is lower-triangular with ones on the diagonal, and $\boldsymbol{U}$ is strictly upper-triangular, reduces the determinant cost from $O(C^3)$ to $O(C)$ via $\log\lvert \det \boldsymbol{W} \rvert = \sum_{c=1}^C \log\lvert s_c \rvert$.

### Autoregressive Flows: MAF versus IAF Duality

Instead of splitting dimensions into two blocks, an *autoregressive flow* applies a scalar bijection $h(\cdot; \theta_i) : \mathbb{R} \to \mathbb{R}$ to each coordinate $i \in \{1, \dots, D\}$ conditioned on all preceding coordinates [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 23, pp. 837–841 (PDF pp. 871–875)), [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, pp. 552–553 (PDF pp. 564–565)). For an affine scalar bijection with log-scale $\alpha_i$ and shift $\mu_i$, two dual parameterizations arise:

> [!important] MAF vs. IAF Equations & Triangular Jacobian
> **Masked Autoregressive Flow (MAF):** Conditions on the *data* prefix $\boldsymbol{x}_{1:i-1}$:
>
> $$
> x_i = z_i \exp(\alpha_i(\boldsymbol{x}_{1:i-1})) + \mu_i(\boldsymbol{x}_{1:i-1}) \Longleftrightarrow z_i = \left(x_i - \mu_i(\boldsymbol{x}_{1:i-1})\right)\exp(-\alpha_i(\boldsymbol{x}_{1:i-1})).
>
> $$
>
> **Inverse Autoregressive Flow (IAF):** Conditions on the *latent* prefix $\boldsymbol{z}_{1:i-1}$:
>
> $$
> x_i = z_i \exp(\alpha_i(\boldsymbol{z}_{1:i-1})) + \mu_i(\boldsymbol{z}_{1:i-1}) \Longleftrightarrow z_i = \left(x_i - \mu_i(\boldsymbol{z}_{1:i-1})\right)\exp(-\alpha_i(\boldsymbol{z}_{1:i-1})).
>
> $$
>
> In both cases, $\frac{\partial x_i}{\partial z_j} = 0$ for all $j > i$, so $\boldsymbol{J}_f$ is lower-triangular and $\log\lvert \det\boldsymbol{J}_f \rvert = \sum_{i=1}^D \alpha_i$ in $O(D)$ operations once the conditioner outputs are computed. Using masked MLPs (MADE), *MAF* evaluates all $\alpha_i(\boldsymbol{x}_{1:i-1}), \mu_i(\boldsymbol{x}_{1:i-1})$ in $O(1)$ parallel forward passes (fast density evaluation for MLE) but requires $D$ sequential steps to sample $\boldsymbol{x}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 23, p. 840 (PDF p. 874)). Conversely, *IAF* samples $\boldsymbol{x}$ in $O(1)$ parallel pass (ideal for variational posteriors and WaveNet distillation) but requires $D$ sequential steps to invert $\boldsymbol{x} \to \boldsymbol{z}$ for arbitrary data density evaluation [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 23, p. 841 (PDF p. 875)), [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 16, p. 313 (PDF p. 327)).

### Continuous Normalizing Flows, Neural ODEs & Continuity Equation

Consider a residual flow layer $\boldsymbol{z}_{k+1} = \boldsymbol{z}_k + \Delta t \boldsymbol{v}_\theta(\boldsymbol{z}_k, t_k)$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 5.1.2, p. 132 (PDF p. 137)). Taking the continuous-time limit $\Delta t \to 0$ yields a *Neural Ordinary Differential Equation (Neural ODE)*:

$$
\frac{\mathrm{d}\boldsymbol{z}(t)}{\mathrm{d}t} = \boldsymbol{v}_\theta(\boldsymbol{z}(t), t), \qquad t \in [0, T], \qquad \boldsymbol{z}(0) \sim p_Z, \quad \boldsymbol{x} = \boldsymbol{z}(T).
$$

By the Picard–Lindelöf theorem, if $\boldsymbol{v}_\theta(\cdot, t)$ is uniformly Lipschitz continuous in $\boldsymbol{z}$ and continuous in $t$, then `(eq:flow:node)` admits a unique solution trajectory $\boldsymbol{z}(t) = \Psi_{0 \to t}(\boldsymbol{z}(0))$ for all $t \in [0, T]$, and the time-$T$ flow map $f_\theta = \Psi_{0 \to T}$ is guaranteed to be a diffeomorphism whose inverse $f_\theta^{-1} = \Psi_{T \to 0}$ is obtained by integrating `(eq:flow:node)` backward from $T$ to $0$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 23, p. 845 (PDF p. 879)).

> [!important] Instantaneous Change of Variables & Continuity Equation Equivalence
> Discretizing `(eq:flow:node)` over a small step $\Delta t > 0$ via forward Euler gives $\boldsymbol{z}_{t+\Delta t} = \boldsymbol{z}_t + \Delta t \boldsymbol{v}_\theta(\boldsymbol{z}_t, t) + O(\Delta t^2)$, whose Jacobian is $\boldsymbol{J} = \boldsymbol{I}_D + \Delta t \nabla_{\boldsymbol{z}}\boldsymbol{v}_\theta(\boldsymbol{z}_t, t) + O(\Delta t^2)$ [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (App. B.1, p. 405 (PDF p. 410)). Using the first-order determinant expansion $\det(\boldsymbol{I}_D + \Delta t \boldsymbol{A}) = 1 + \Delta t\mathrm{Tr}(\boldsymbol{A}) + O(\Delta t^2)$ and $\log(1 + u) = u + O(u^2)$:
>
> $$
> \begin{aligned}
> \log p_{t+\Delta t}(\boldsymbol{z}_{t+\Delta t}) &= \log p_t(\boldsymbol{z}_t) - \log \lvert \det\!\left(\boldsymbol{I}_D + \Delta t \nabla_{\boldsymbol{z}}\boldsymbol{v}_\theta(\boldsymbol{z}_t, t)\right) \rvert + O(\Delta t^2)  \\
>   &= \log p_t(\boldsymbol{z}_t) - \Delta t \mathrm{Tr}\!\left(\nabla_{\boldsymbol{z}}\boldsymbol{v}_\theta(\boldsymbol{z}_t, t)\right) + O(\Delta t^2).
> \end{aligned}
>
> $$
>
> Dividing by $\Delta t$ and taking $\Delta t \to 0$ yields the *Instantaneous Change-of-Variables Formula* [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, Eq. (18.28), p. 557 (PDF p. 569)), [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 5.1.2, p. 133 (PDF p. 138)):
>
> $$
> \frac{\mathrm{d}}{\mathrm{d}t}\log p_t(\boldsymbol{z}(t)) = -\mathrm{Tr}\!\left(\frac{\partial \boldsymbol{v}_\theta(\boldsymbol{z}(t), t)}{\partial \boldsymbol{z}(t)}\right) = -\nabla_{\boldsymbol{z}} \cdot \boldsymbol{v}_\theta(\boldsymbol{z}(t), t).
>
> $$
>
> Integrating from $t=0$ to $t=T$ gives the exact log-likelihood without any determinant or architectural restriction on $\boldsymbol{v}_\theta$:
>
> $$
> \log p_X(\boldsymbol{x}) = \log p_Z(\boldsymbol{z}(0)) - \int_0^T \nabla_{\boldsymbol{z}} \cdot \boldsymbol{v}_\theta(\boldsymbol{z}(t), t) \mathrm{d}t.
>
> $$
>
> **Link to Continuity & Fokker–Planck Equations:** Expanding the total derivative $\frac{\mathrm{d}}{\mathrm{d}t}\log p_t(\boldsymbol{z}(t)) = \partial_t \log p_t + \boldsymbol{v}_\theta^{\top} \nabla_{\boldsymbol{z}}\log p_t$ in `(eq:flow:inst_cov)` and multiplying by $p_t$ yields the *Continuity Equation* (Liouville's theorem for probability conservation) [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (App. B.1–B.2, pp. 404–411 (PDF pp. 409–416)):
>
> $$
> \frac{\partial p_t(\boldsymbol{z})}{\partial t} + \nabla_{\boldsymbol{z}} \cdot \left(p_t(\boldsymbol{z}) \boldsymbol{v}_\theta(\boldsymbol{z}, t)\right) = 0,
>
> $$
>
> which is precisely the zero-diffusion ($g(t)=0$) deterministic limit of the Fokker–Planck equation $\partial_t p_t = -\nabla \cdot (\boldsymbol{v}_\theta p_t) + \frac{1}{2}g^2(t)\Delta p_t$.

**Adjoint Sensitivity Method:** Backpropagating through adaptive ODE solver steps incurs $O(\text{NFE})$ memory. Instead, the *adjoint sensitivity method* defines the adjoint state $\boldsymbol{a}(t) = \frac{\partial \mathcal{L}}{\partial \boldsymbol{z}(t)}$, which evolves backward via $\frac{\mathrm{d}\boldsymbol{a}(t)}{\mathrm{d}t} = -\boldsymbol{a}(t)^{\top} \frac{\partial \boldsymbol{v}_\theta(\boldsymbol{z}(t), t)}{\partial \boldsymbol{z}(t)}$, allowing parameter gradients $\nabla_\theta \mathcal{L} = -\int_T^0 \boldsymbol{a}(t)^{\top} \frac{\partial \boldsymbol{v}_\theta}{\partial \theta} \mathrm{d}t$ to be computed in $O(1)$ memory [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, pp. 555–556 (PDF pp. 567–568)).

**Hutchinson Trace Estimator (FFJORD):** Computing the exact divergence $\mathrm{Tr}(\boldsymbol{J}_{\boldsymbol{v}_\theta})$ requires $D$ reverse-mode automatic differentiation passes ($O(D^2)$ cost). *FFJORD* reduces this to $O(D)$ via Hutchinson's unbiased trace estimator ([[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, Eq. (18.29), p. 558); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 23, p. 845)):

$$
\mathrm{Tr}\!\left(\boldsymbol{J}_{\boldsymbol{v}_\theta}(\boldsymbol{z}, t)\right) = \mathbb{E}_{\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})}\!\left[\boldsymbol{\epsilon}^{\top} \boldsymbol{J}_{\boldsymbol{v}_\theta}(\boldsymbol{z}, t) \boldsymbol{\epsilon}\right] \approx \frac{1}{M}\sum_{m=1}^M \boldsymbol{\epsilon}_m^{\top} \left(\boldsymbol{J}_{\boldsymbol{v}_\theta}(\boldsymbol{z}, t) \boldsymbol{\epsilon}_m\right),
$$

where $\mathbb{E}[\boldsymbol{\epsilon}] = \boldsymbol{0}$ and $\mathrm{Cov}(\boldsymbol{\epsilon}) = \boldsymbol{I}_D$ (e.g., Rademacher $\{\pm 1\}^D$ or $\mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_D)$), and the vector-Jacobian product $\boldsymbol{\epsilon}^{\top} \boldsymbol{J}_{\boldsymbol{v}_\theta}$ is evaluated in a single reverse-mode AD pass.

## Algorithmic Implementation & Comparative Synthesis

Algorithm `alg:flow:ffjord` details one training iteration of an unconstrained FFJORD Continuous Normalizing Flow, integrating the augmented state $(\boldsymbol{z}(t), \Delta\log p(t))$ backward from data $\boldsymbol{x}$ to base $\boldsymbol{z}(0)$ and computing gradients via the adjoint ODE.

> [!example] Algorithm: FFJORD Continuous Normalizing Flow Training Step [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, pp. 556–558), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 23, p. 845)
> **Input:** Data batch $\{\boldsymbol{x}^{(n)}\}_{n=1}^B$, neural velocity field $\boldsymbol{v}_\theta(\boldsymbol{z}, t)$, horizon $T$, base density $p_Z$, Hutchinson samples $M$.
> **Output:** Updated parameters $\theta$.
> Sample noise vectors $\boldsymbol{\epsilon}^{(n,m)} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_D)$ (or Rademacher) for $n \in [B], m \in [M]$
> **for** each sample $\boldsymbol{x}^{(n)}$ in parallel:
>
>   Define augmented dynamics $\mathcal{F}(\boldsymbol{z}, t) = \left[\boldsymbol{v}_\theta(\boldsymbol{z}, t), -\frac{1}{M}\sum_{m=1}^M (\boldsymbol{\epsilon}^{(n,m)})^{\top} \nabla_{\boldsymbol{z}}(\boldsymbol{v}_\theta(\boldsymbol{z}, t)^{\top} \boldsymbol{\epsilon}^{(n,m)})\right]$
>   Integrate backward from $t=T$ to $t=0$ with initial state $[\boldsymbol{x}^{(n)},  0]$:
>   $[\boldsymbol{z}^{(n)}(0),  \Delta\ell^{(n)}] \leftarrow \mathrm{ODESolve}([\boldsymbol{x}^{(n)}, 0], \mathcal{F}, T \to 0)$
>   Compute exact log-likelihood: $\log p_\theta(\boldsymbol{x}^{(n)}) = \log p_Z(\boldsymbol{z}^{(n)}(0)) - \Delta\ell^{(n)}$
>
> Form negative log-likelihood loss $\mathcal{L}(\theta) = -\frac{1}{B}\sum_{n=1}^B \log p_\theta(\boldsymbol{x}^{(n)})$
> Solve adjoint ODE backward/forward with $\boldsymbol{a}(0) = \frac{\partial \mathcal{L}}{\partial \boldsymbol{z}(0)}$ to obtain $\nabla_\theta \mathcal{L}$ in $O(1)$ memory
> Update $\theta \leftarrow \mathrm{OptimizerStep}(\theta, \nabla_\theta \mathcal{L})$

For high-resolution images, discrete flows (RealNVP, Glow) avoid ODE solvers by employing a *multiscale architecture* [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 16, pp. 316–318 (PDF pp. 330–332)): at each scale, a squeeze operation trades spatial resolution ($2\times 2$ patches) for $4\times$ channel depth, followed by alternating ActNorm, $1\times 1$ invertible convolutions, and affine coupling layers; half of the channels are periodically factored out directly into $\boldsymbol{z}$ to reduce memory and computation.

**Table:** Computational and structural comparison of exact-likelihood normalizing flow families [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 23, pp. 835–845), [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, pp. 549–558).

| **Flow Family** | **Sampling $\boldsymbol{z} \to \boldsymbol{x}$** | **Density $\boldsymbol{x} \to \boldsymbol{z}$** | **Log-Det Cost** | **Key Trade-Off** |
| — | — | — | — | — |
| Coupling (RealNVP / Glow) | $O(1)$ parallel pass | $O(1)$ parallel pass | $O(D)$ (diag) + $O(C)$ ($1{\times}1$ LU) | Symmetric speed; requires deep stacking for cross-dim mixing |
| MAF | $O(D)$ sequential | $O(1)$ parallel pass | $O(D)$ (triangular) | Fast MLE training; slow $O(D)$ ancestral sampling |
| IAF | $O(1)$ parallel pass | $O(D)$ sequential | $O(D)$ (triangular) | Fast sampling (VAE/distillation); slow MLE on data |
| Residual Flow (iResNet) | $O(K)$ fixed-point | $O(1)$ forward pass | Unbiased power series | Unconstrained layers; requires Lipschitz $< 1$ spectral norm |
| Continuous (FFJORD) | $O(\text{NFE})$ ODE solve | $O(\text{NFE})$ ODE solve | $O(D)$ per NFE (Hutchinson) | Free-form $\boldsymbol{v}_\theta$, $O(1)$ memory; high NFE if stiff |

> [!info] Cross-Book Synthesis: Likelihood Objectives & Bridge to Flow Matching
> **Forward vs. Reverse KL Training:** As shown in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 23, pp. 831–832) and [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 16, p. 320), flows support two complementary objectives. Given data samples $\boldsymbol{x} \sim p_{\mathrm{data}}$, MLE minimizes $D_{\mathrm{KL}}(p_{\mathrm{data}} \parallel p_\theta)$, requiring fast inverse evaluation $f_\theta^{-1}(\boldsymbol{x})$ (favoring RealNVP, Glow, MAF). Conversely, when approximating an unnormalized target density $q(\boldsymbol{x})$ (e.g., a VAE posterior or WaveNet teacher), one minimizes the reverse KL $D_{\mathrm{KL}}(p_\theta \parallel q)$ using reparameterized samples $\boldsymbol{x} = f_\theta(\boldsymbol{z})$, requiring fast forward sampling (favoring IAF).
>
> **From CNFs to Flow Matching & Diffusion:** Both Bishop [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, p. 558) and Diffusion Principles [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 5.1.2, pp. 133–135) highlight that FFJORD suffers from costly ODE integration during training. *Flow Matching* resolves this bottleneck by regressing $\boldsymbol{v}_\theta(\boldsymbol{z}_t, t)$ directly onto conditional vector fields along prescribed probability paths $p_t$ satisfying the continuity equation `(eq:flow:continuity)`, eliminating ODE solves during training while retaining exact likelihood evaluation at inference.

## Visual Diagnostics & Geometric Structure

> [!example] Visual Diagram (LaTeX / PDF)
> *Architectural comparison of (a) an affine coupling layer with block-triangular Jacobian and symmetric $O(1)$ parallel forward/inverse passes, and (b) the computational asymmetry between Masked Autoregressive Flows (MAF) and Inverse Autoregressive Flows (IAF).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:flow:density_warp` illustrates the 1D change-of-variables mechanism for the smooth global diffeomorphism $x = f(z) = z + 0.45\sin(1.8z)$, whose derivative $f'(z) = 1 + 0.81\cos(1.8z) \ge 0.19 > 0$ guarantees strict monotonicity. Where $f'(z) > 1$ (near $z=0$), the base Gaussian $\mathcal{N}(0,1)$ is stretched, depressing $p_X(0)$ below $p_Z(0)$; where $f'(z) < 1$ (near $z \approx \pm 1.75$), probability mass is compressed, creating two symmetric modes in $p_X(x)$.

> [!example] Visual Diagram (LaTeX / PDF)
> *1D change-of-variables density warping under the $C^\infty$ diffeomorphism $x = f(z) = z + 0.45\sin(1.8z)$. Stretching near the origin ($f'(0)=1.81$) splits the unimodal Gaussian base into a bimodal target density via local Jacobian compression around $z \approx \pm 1.75$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

In two dimensions, Figure `fig:flow:realnvp_moons` shows how composing RealNVP affine coupling layers progressively deforms an isotropic Gaussian into a nonlinear, multimodal two-moons distribution. Figure `fig:flow:multiscale` illustrates the multiscale factor-out architecture used in RealNVP and Glow to scale to high-dimensional images, and Figure `fig:flow:cnf_evolution` visualizes a Continuous Normalizing Flow smoothly transporting probability mass along non-crossing ODE integral curves.

> [!note] Source Figure
> *Figure:* RealNVP normalizing flow applied to the two-moons dataset, showing the base Gaussian (a), intermediate warped densities across stacked coupling layers (b–e), and empirical target samples (f) [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, Fig. 18.3, p. 552 (PDF p. 564)). (`deep_learning_foundations_concepts_p564_fig1.png`)

> [!note] Source Figure
> *Figure:* Multiscale flow architecture: latent dimensions are partitioned into blocks and progressively factored out at coarser scales, reducing intermediate tensor dimensionality while preserving exact invertibility [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 16, Fig. 16.10, p. 317 (PDF p. 331)). (`understandingdl_simonprince_p331_fig1.png`)

> [!note] Source Figure
> *Figure:* Continuous normalizing flow (Neural ODE) smoothly transforming a unimodal Gaussian $p(z(0))$ into a multimodal distribution $p(z(T))$ along non-crossing flow lines; density decreases where trajectories diverge ($\nabla\cdot\boldsymbol{v}_\theta > 0$) and increases where they converge ($\nabla\cdot\boldsymbol{v}_\theta < 0$) [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, Fig. 18.6, p. 558 (PDF p. 570)). (`deep_learning_foundations_concepts_p570_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Discrete Data & Dequantization Singularity
> Training a continuous density model $p_X(\boldsymbol{x})$ via MLE on discrete data (e.g., 8-bit quantized pixel intensities $\boldsymbol{x} \in \{0, \dots, 255\}^D$) is ill-posed: the flow can place arbitrarily narrow Dirac-like spikes on discrete lattice points, driving $\log p_X(\boldsymbol{x}) \to +\infty$ while collapsing between grid points [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 16, p. 318 (PDF p. 332)). **Remedy:** Add continuous noise $\boldsymbol{u} \in [0, 1)^D$ to form $\tilde{\boldsymbol{x}} = \boldsymbol{x} + \boldsymbol{u}$ (*uniform dequantization*), or learn a conditional flow $q(\boldsymbol{u}|\boldsymbol{x})$ (*variational dequantization*) so that $\mathbb{E}_{q(\boldsymbol{u}|\boldsymbol{x})}[\log p_X(\boldsymbol{x}+\boldsymbol{u}) - \log q(\boldsymbol{u}|\boldsymbol{x})]$ lower-bounds the discrete log-probability.

> [!warning] Exponential Scale Explosion & Floating-Point Non-Invertibility
> In RealNVP/MAF, unconstrained log-scales $s_j(\boldsymbol{z}_A)$ can grow large, causing $\exp(s_j)$ to overflow to $\infty$ or underflow to $0$ in float32, which destroys numerical invertibility ($f_\theta^{-1}(f_\theta(\boldsymbol{z})) \ne \boldsymbol{z}$) [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 18, p. 550). **Remedy:** Bound the scale via $s_j = \alpha \tanh(\tilde{s}_j)$ or replace $\exp(s_j)$ with $\sigma(\tilde{s}_j + 2)$ (Glow), and initialize coupling layers as identity maps using *ActNorm* (data-dependent per-channel shift/scale initialized on the first minibatch).

> [!warning] Stiff ODEs, Adjoint Reversibility Drift & High NFE
> In CNFs, an unregularized neural velocity field $\boldsymbol{v}_\theta(\boldsymbol{z}, t)$ may learn convoluted, high-curvature trajectories, causing adaptive Runge–Kutta solvers to take hundreds of function evaluations (NFE) and inducing numerical drift between forward and backward ODE trajectories in the adjoint pass. **Remedy:** Add kinetic energy ($\int_0^T \lVert \boldsymbol{v}_\theta \rVert_2^2 \mathrm{d}t$) and Jacobian Frobenius norm ($\int_0^T \lVert \boldsymbol{J}_{\boldsymbol{v}_\theta} \rVert_F^2 \mathrm{d}t$) regularization (RNODE), or replace MLE-based CNF training with Simulation-Free Flow Matching [[Source - The Principles of Diffusion Models|theprinciple_diffusion_models]] (Sec. 5.1.2, p. 134).

> [!warning] Topological Obstructions & Out-of-Distribution Likelihood Paradox
> Because diffeomorphisms preserve topological invariants (connectedness, homotopy), mapping a unimodal Gaussian $\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ to a disconnected multi-mode target forces $\lVert \boldsymbol{J}_{f_\theta} \rVert$ to blow up in low-density bridges between modes. Moreover, deep flows trained on high-dimensional images (e.g., CIFAR-10) famously assign *higher* exact likelihoods to simpler out-of-distribution datasets (e.g., SVHN) because local Jacobian volume terms $\log\lvert \det\boldsymbol{J}_{f_\theta^{-1}}(\boldsymbol{x}) \rvert$ dominate the latent prior term $\log p_Z(\boldsymbol{z})$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 23, p. 846 (PDF p. 880)). **Remedy:** Never threshold raw flow likelihoods $\log p_X(\boldsymbol{x})$ alone for OOD detection; use likelihood ratios against a background model or typicality tests $\lvert \log p_X(\boldsymbol{x}) + H(p_X) \rvert$.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (1):** [[Normalizing Flows]]
- **Prerequisites (Directed $\leftarrow$):** [[Matrix Calculus & Jacobians]]
- **Enables / Unlocks (Directed $\rightarrow$):** None
- **Related Concepts (Undirected $\leftrightarrow$):** [[Flow Matching & Rectified Flow]], [[Matrix Calculus & Jacobians]], [[Probability Flow ODE]]
- **Synthesized Sources:** [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - The Principles of Diffusion Models]], [[Source - Understanding Deep Learning]]
