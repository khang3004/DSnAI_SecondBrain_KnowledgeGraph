---
type: monograph
domain: "D04"
parent_topic: "[[PT 04 - Autoencoders, VAEs, Normalizing Flows, EBMs & GANs]]"
sub_topic: "[[ST 04 - Energy-Based Models, Score Matching & Adversarial Networks]]"
cheatsheet: "[[Cheatsheet - GAN Game Dynamics, Mode Collapse, f-GANs & Wasserstein - OT-GANs]]"
part_of: 
  - "[[ST 04 - Energy-Based Models, Score Matching & Adversarial Networks]]"
  - "[[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion]]"
concepts: 
  - "[[GAN & Mode Collapse]]"
  - "[[Wasserstein GAN (WGAN)]]"
prerequisites: 
  - "[[GAN & Mode Collapse]]"
  - "[[Kantorovich Duality]]"
  - "[[Wasserstein Distance]]"
enables: 
  - "[[Wasserstein GAN (WGAN)]]"
related: 
  - "[[Kantorovich Duality]]"
  - "[[Wasserstein Distance]]"
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Optimal Transport in Large-Scale]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-04
---

# GAN Game Dynamics, Mode Collapse, f-GANs & Wasserstein/OT-GANs

> [!abstract] Monograph Navigation
> - **Domain:** [[D04 - Deep Generative Modeling - VAEs, Flows, EBMs, GANs & Diffusion|Deep Generative Modeling: VAEs, Flows, EBMs, GANs & Diffusion]]
> - **Parent / Sub-Topic:** [[PT 04 - Autoencoders, VAEs, Normalizing Flows, EBMs & GANs|Autoencoders, VAEs, Normalizing Flows, EBMs & GANs]] → [[ST 04 - Energy-Based Models, Score Matching & Adversarial Networks|Energy-Based Models, Score Matching & Adversarial Networks]]
> - **Visual Cheatsheet:** [[Cheatsheet - GAN Game Dynamics, Mode Collapse, f-GANs & Wasserstein - OT-GANs]]
> - **Description:** Derives the minimax JS-divergence theorem, explains vanishing gradients on disjoint low-dimensional manifolds, introduces f-GAN variational divergence minimization, Wasserstein-1 Kantorovich-Rubinstein duality (WGAN/WGAN-GP), and minibatch/sliced OT-GANs.

## Problem Formulation & Unified Notation

Implicit generative models learn a target data distribution $p_{\mathrm{data}}$ over $\mathcal{X} \subseteq \mathbb{R}^d$ without specifying a tractable density function. Instead, a latent variable $z \sim p_Z$ (typically $p_Z = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ or $\mathcal{U}([-1,1]^{d_z})$ with $d_z \ll d$) is passed through a deterministic, differentiable generator $G_\theta : \mathcal{Z} \to \mathcal{X}$ parameterized by $\theta$. As formulated in [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 894) and [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 17, p. 534), this induces the pushforward model distribution $q_\theta = (G_\theta)_\# p_Z$. Because $q_\theta(x)$ is intractable in closed form, Generative Adversarial Networks (GANs) introduce a second network—a discriminator $D_\phi : \mathcal{X} \to (0,1)$ or real-valued critic $f_\phi : \mathcal{X} \to \mathbb{R}$—that estimates a divergence or Integral Probability Metric (IPM) solely from samples [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 896).

In the classical zero-sum minimax formulation [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 15, p. 277) [[Source - Optimal Transport in Large-Scale|optimal_transport]] (p. 44), the two players solve:

$$
\min_{\theta} \max_{\phi} V(\phi, \theta)
  := \mathbb{E}_{x \sim p_{\mathrm{data}}}\!\left[\log D_\phi(x)\right]
   + \mathbb{E}_{z \sim p_Z}\!\left[\log\!\left(1 - D_\phi(G_\theta(z))\right)\right].
$$

More generally, GANs need not be zero-sum; [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 905) separates the discriminator loss $L_D(\phi, \theta)$ and generator loss $L_G(\phi, \theta)$ into the coupled nested optimization:

$$
\min_{\phi} L_D(\phi, \theta), \qquad \min_{\theta} L_G(\phi, \theta),
$$

where $L_D(\phi, \theta) = \mathbb{E}_{p_{\mathrm{data}}}[g(D_\phi(x))] + \mathbb{E}_{p_Z}[h(D_\phi(G_\theta(z)))]$ and $L_G(\phi, \theta) = \mathbb{E}_{p_Z}[\ell(D_\phi(G_\theta(z)))]$ for scalar functions $g, h, \ell : \mathbb{R} \to \mathbb{R}$.

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> The four synthesized sources adopt distinct conventions for generator/discriminator symbols, probability measures, and optimization signs. Table `tab:gan:notation` reconciles these notations into the unified conventions used throughout this monograph.
>
>
>
> {
>
> | **Concept** | **Unified** | **Murphy Adv. (Ch. 26)** | **Prince (Ch. 15)** | **Bishop (Ch. 17) / OT** |
> | — | — | — | — | — |
> | Data dist. | $p_{\mathrm{data}}(x)$ | $p^*(x)$ | $\Pr(x)$ | $p_{\mathrm{data}}(x)$ / $P$ |
> | Model dist. | $q_\theta(x)$ | $q_\theta(x)$ | $\Pr(x^*)$ | $p_G(x)$ / $(T_\phi)_\# p_Z$ |
> | Generator | $G_\theta(z)$ | $G_\theta(z)$ | $g[z, \theta]$ | $g(z, w)$ / $T_\phi(z)$ |
> | Disc. / Critic | $D_\phi(x),  f_\phi(x)$ | $D_\phi(x)$ | $\mathrm{sig}[f[x,\phi]],  f[x,\phi]$ | $d(x, \phi)$ / $f_\omega(x)$ |
> | Objective sign | $\min_\theta \max_\phi V$ | $\min_\theta \max_\phi V$ ($\times \frac{1}{2}$) | $\arg\max_\theta \min_\phi$ (NLL) | $\max_w \min_\phi E_{\mathrm{GAN}}$ |
>
> }

## Core Mathematical Derivations & Theorems

### Optimal Discriminator, Jensen–Shannon Equilibrium, and Gradient Saturation

> [!important] Theorem 1: Minimax Optimal Discriminator & Jensen–Shannon Divergence
> For a fixed generator $G_\theta$ inducing density $q_\theta(x)$ with full support, the inner maximization of `(eq:gan:minimax)` over all measurable functions $D : \mathcal{X} \to (0,1)$ has the unique pointwise maximizer [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 15, p. 281) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 897):
>
> $$
> D^*_\theta(x) = \frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x) + q_\theta(x)}.
>
> $$
>
> **Proof.** Writing $V(G_\theta, D)$ as an integral over $\mathcal{X}$:
>
> $$
>
>   V(G_\theta, D) = \int_{\mathcal{X}} \Big[ p_{\mathrm{data}}(x)\log D(x) + q_\theta(x)\log(1 - D(x)) \Big] \mathrm{d}x.
>
> $$
>
> For any $(a, b) \in \mathbb{R}_+^2 \setminus \{(0,0)\}$, the scalar function $y \mapsto a\log y + b\log(1-y)$ on $(0,1)$ has derivative $\frac{a}{y} - \frac{b}{1-y} = 0$ at $y^* = \frac{a}{a+b}$, and second derivative $-\frac{a}{y^2} - \frac{b}{(1-y)^2} < 0$. Setting $a = p_{\mathrm{data}}(x)$ and $b = q_\theta(x)$ yields `(eq:gan:opt_disc)`. Equivalently, in density-ratio form, $D^*_\theta(x)/(1 - D^*_\theta(x)) = p_{\mathrm{data}}(x)/q_\theta(x)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 896).
>
> Substituting $D^*_\theta$ back into `(eq:gan:minimax)` and letting $m_\theta = \frac{1}{2}(p_{\mathrm{data}} + q_\theta)$ gives:
>
> $$
> \begin{aligned}
> V(G_\theta, D^*_\theta)
>   &= \mathbb{E}_{p_{\mathrm{data}}}\!\left[\log \frac{p_{\mathrm{data}}(x)}{2m_\theta(x)}\right]
>    + \mathbb{E}_{q_\theta}\!\left[\log \frac{q_\theta(x)}{2m_\theta(x)}\right]  \\
>   &= D_{\mathrm{KL}}(p_{\mathrm{data}} \parallel m_\theta) + D_{\mathrm{KL}}(q_\theta \parallel m_\theta) - 2\log 2
>    = 2 D_{\mathrm{JS}}(p_{\mathrm{data}} \parallel q_\theta) - \log 4.
> \end{aligned}
>
> $$
>
> Since $D_{\mathrm{JS}}(p_{\mathrm{data}} \parallel q_\theta) \ge 0$ with equality if and only if $q_\theta = p_{\mathrm{data}}$, the global minimax equilibrium is attained at $q_\theta = p_{\mathrm{data}}$ and $D^*_\theta(x) \equiv \frac{1}{2}$, with game value $-\log 4$ (or $-\log 2$ under the $\frac{1}{2}$-weighted objective of [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 897)).  $\square$

> [!tip] Mode Dropping, Quality vs. Coverage, and Non-Saturating Loss
> Expanding $D_{\mathrm{JS}}(p_{\mathrm{data}} \parallel q_\theta) = \frac{1}{2}D_{\mathrm{KL}}(p_{\mathrm{data}} \parallel m_\theta) + \frac{1}{2}D_{\mathrm{KL}}(q_\theta \parallel m_\theta)$ reveals an asymmetry in practice [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 15, pp. 281–282): the second term penalizes regions where $q_\theta(x)$ is high but $p_{\mathrm{data}}(x)$ is low (enforcing *quality*), whereas the first term penalizes regions where $p_{\mathrm{data}}(x)$ is high but $q_\theta(x)$ is low (enforcing *coverage*). Because the generator only controls $q_\theta$, minimizing the saturating loss $\mathbb{E}_{p_Z}[\log(1 - D_\phi(G_\theta(z)))]$ suffers from vanishing gradients when $D_\phi(G_\theta(z)) \approx 0$ (early in training or when $D_\phi$ is too strong). Replacing it with the *non-saturating* generator loss $L_G^{\mathrm{NS}}(\phi, \theta) = -\mathbb{E}_{p_Z}[\log D_\phi(G_\theta(z))]$ provides strong gradients near $D_\phi \approx 0$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 17, p. 538) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 905), though it shifts the effective divergence toward a reverse-KL-like objective that can exacerbate mode dropping.

### $f$-GANs: Variational Divergence Minimization

Beyond the Bernoulli log-loss, any proper binary scoring rule corresponds to an $f$-divergence [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, pp. 897–900). For a convex, lower-semicontinuous generator $f : \mathbb{R}_+ \to \mathbb{R}$ with $f(1) = 0$, the $f$-divergence is $D_f(p_{\mathrm{data}} \| q_\theta) = \int q_\theta(x) f\!\left(\frac{p_{\mathrm{data}}(x)}{q_\theta(x)}\right)\mathrm{d}x$.

> [!important] Theorem 2: $f$-GAN Variational Lower Bound via Fenchel Duality
> Let $f^\dagger(t) := \sup_{u \in \mathrm{dom}(f)} \{ut - f(u)\}$ denote the Fenchel–Legendre convex conjugate of $f$. By the Fenchel–Moreau theorem, $f(u) = \sup_{t \in \mathrm{dom}(f^\dagger)} \{ut - f^\dagger(t)\}$. Substituting $u = p_{\mathrm{data}}(x)/q_\theta(x)$ and exchanging integration and supremum over a variational critic class $\mathcal{T} \subseteq \{T : \mathcal{X} \to \mathrm{dom}(f^\dagger)\}$ yields [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 899):
>
> $$
> D_f(p_{\mathrm{data}} \| q_\theta)
>   \ge \sup_{T \in \mathcal{T}} \Big\{ \mathbb{E}_{x \sim p_{\mathrm{data}}}[T(x)] - \mathbb{E}_{x \sim q_\theta}[f^\dagger(T(x))] \Big\}.
>
> $$
>
> The bound is tight whenever the optimal critic $T^*(x) = f'\!\left(\frac{p_{\mathrm{data}}(x)}{q_\theta(x)}\right)$ lies in $\mathcal{T}$. Parameterizing $T_\phi(x) = a_f(V_\phi(x))$ with an output activation $a_f : \mathbb{R} \to \mathrm{dom}(f^\dagger)$ gives the $f$-GAN minimax game $\min_\theta \max_\phi \big(\mathbb{E}_{p_{\mathrm{data}}}[T_\phi(x)] - \mathbb{E}_{p_Z}[f^\dagger(T_\phi(G_\theta(z)))]\big)$. Table `tab:gan:fgan` lists standard instantiations.
>
>
>
> {
>
> | **Divergence** | **Generator $f(u)$** | **Conjugate $f^\dagger(t)$** | **Optimal Critic $T^*(x)$ ($r = p_{\mathrm{data}}/q_\theta$)** |
> | — | — | — | — |
> | Forward KL | $u\log u$ | $\exp(t - 1)$ | $1 + \log r(x)$ |
> | Reverse KL | $-\log u$ | $-1 - \log(-t)$ | $-1 / r(x)$ |
> | JSD | $u\log u - (u+1)\log\frac{u+1}{2}$ | $-\log(2 - e^t)$ | $\log\frac{2r(x)}{r(x)+1}$ |
> | Pearson $\chi^2$ | $(u - 1)^2$ | $\frac{1}{4}t^2 + t$ | $2(r(x) - 1)$ |
>
> }

### Disjoint-Support Failure, Continuity Theorem, and Kantorovich–Rubinstein Duality

In high-dimensional spaces $\mathcal{X} = \mathbb{R}^d$, natural data $p_{\mathrm{data}}$ and generator outputs $q_\theta = (G_\theta)_\# p_Z$ typically concentrate on low-dimensional manifolds ($d_z \ll d$) whose intersection has measure zero [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 15, p. 282). Because $f$-divergences depend on the density ratio $p_{\mathrm{data}}(x)/q_\theta(x)$, they fail catastrophically when supports are disjoint [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 903); [[Source - Optimal Transport in Large-Scale|optimal_transport]] (p. 45).

> [!important] Theorem 3: Disjoint-Support Saturation vs. Wasserstein Continuity & Duality
> **(a) Disjoint-Support Counterexample.** Let $P = (0, \mathcal{U}(0,1))$ and $Q_\theta = (\theta, \mathcal{U}(0,1))$ in $\mathbb{R}^2$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (pp. 45–48). For any $\theta \neq 0$, $\mathrm{supp}(P) \cap \mathrm{supp}(Q_\theta) = \emptyset$, so:
>
> $$
> D_{\mathrm{KL}}(P \parallel Q_\theta) = +\infty, \qquad
>   D_{\mathrm{JS}}(P \parallel Q_\theta) = \begin{cases} \log 2 & \text{if } \theta \neq 0, \\ 0 & \text{if } \theta = 0, \end{cases}
>   \qquad
>   W_{1}(P, Q_\theta) = \lvert \theta \rvert.
>
> $$
>
> Thus $\mathrm{JSD}$ is discontinuous at $\theta = 0$ and has zero gradient everywhere $\theta \neq 0$, whereas $W_1$ is continuous everywhere and provides a constant unit gradient $\partial_\theta W_1 = \mathrm{sign}(\theta)$ for $\theta \neq 0$.
>
> **(b) Primal and Kantorovich–Rubinstein Dual.** The $1$-Wasserstein (Earth Mover's) distance between $p_{\mathrm{data}}$ and $q_\theta$ is [[Source - Optimal Transport in Large-Scale|optimal_transport]] (pp. 46–50) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 15, p. 285):
>
> $$
> W_{1}(p_{\mathrm{data}}, q_\theta)
>   = \inf_{\pi \in \Pi(p_{\mathrm{data}}, q_\theta)} \int_{\mathcal{X}\times\mathcal{X}} \lVert x - y \rVert_2 \mathrm{d}\pi(x, y)
>   = \sup_{\lVert f \rVert_{\mathrm{Lip}} \le 1} \Big\{ \mathbb{E}_{x \sim p_{\mathrm{data}}}[f(x)] - \mathbb{E}_{y \sim q_\theta}[f(y)] \Big\},
>
> $$
>
> where $\Pi(p_{\mathrm{data}}, q_\theta)$ is the set of couplings with marginals $p_{\mathrm{data}}, q_\theta$, and $\lVert f \rVert_{\mathrm{Lip}} \le 1$ means $\lvert f(x) - f(y) \rvert \le \lVert x - y \rVert_2$ for all $x, y \in \mathcal{X}$. This places $W_1$ inside the family of Integral Probability Metrics (IPMs), which measure density *differences* rather than ratios [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 900).
>
> **(c) Continuity Theorem (Arjovsky et al., 2017).** If $G_\theta(z)$ is continuous in $\theta$, then $\theta \mapsto W_{1}(p_{\mathrm{data}}, q_\theta)$ is continuous everywhere. Moreover, if $G_\theta$ is locally Lipschitz in $\theta$ with $\mathbb{E}_{p_Z}[L(\theta, z)] < \infty$, then $W_{1}(p_{\mathrm{data}}, q_\theta)$ is continuous everywhere and differentiable almost everywhere [[Source - Optimal Transport in Large-Scale|optimal_transport]] (p. 48).  $\square$

Figure `fig:gan:kl_js_vs_w1` plots the exact behavior of $\mathrm{KL}$, $\mathrm{JSD}$, and $W_1$ for the parallel-line family `(eq:gan:disjoint_counterexample)`, and Figure `fig:gan:disjoint_support_murphy` illustrates how a smooth parametric neural critic $D_\phi$ partially mitigates—but does not eliminate—the density-ratio singularity compared to an IPM critic.

> [!example] Visual Diagram (LaTeX / PDF)
> *Disjoint-support convergence failure on $P = (0,\mathcal{U}(0,1))$ vs. $Q_\theta = (\theta,\mathcal{U}(0,1))$: $\mathrm{KL}$ diverges and $\mathrm{JSD} \equiv \log 2$ has zero gradient for all $\theta \neq 0$, whereas $W_1(P, Q_\theta) = |\theta|$ is continuous everywhere and yields a constant unit gradient toward $\theta = 0$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (pp. 45–48).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!note] Source Figure
> *Figure:* Disjoint-support behavior: (a) Exact $\mathrm{KL}(p^* \| q_\theta) = \infty$ cannot distinguish $q(\theta_1)$ from $q(\theta_2)$; (b) a smooth parametric MLP critic $D_\phi$ approximates the singular ratio $p^*/q_\theta$ and supplies a non-zero input gradient $\nabla_x D_\phi(x)$, whereas IPMs (such as $W_1$) build smoothness directly into the metric via the $1$-Lipschitz constraint [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, Fig. 26.4, p. 903). (`probabilisticml_kevinmurphy_advancedtopics_p937_fig1.png`)

### Continuous vs. Discrete Game Dynamics: DiracGAN Stability

Even when the loss landscape is well-defined, alternating or simultaneous gradient descent in `(eq:gan:general_game)` need not converge to a Nash equilibrium. Writing the continuous-time ODEs $\dot{\phi} = -\nabla_\phi L_D(\phi, \theta)$ and $\dot{\theta} = -\nabla_\theta L_G(\phi, \theta)$ gives the game Jacobian [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 910):

$$
\boldsymbol{J}(\phi, \theta) =
  \begin{bmatrix}
    -\nabla^2_{\phi\phi} L_D(\phi, \theta) & -\nabla^2_{\theta\phi} L_D(\phi, \theta) \\
    -\nabla^2_{\phi\theta} L_G(\phi, \theta) & -\nabla^2_{\theta\theta} L_G(\phi, \theta)
  \end{bmatrix}.
$$

For the canonical **DiracGAN** ($p_{\mathrm{data}} = \delta_0$, $q_\theta = \delta_\theta$, linear discriminator $D_\phi(x) = \phi x$), evaluating $\boldsymbol{J}$ at the unique Nash equilibrium $(\phi, \theta) = (0,0)$ gives $\boldsymbol{J}(0,0) = \begin{bmatrix} 0 & \ell'(0) \\ -\ell'(0) & 0 \end{bmatrix}$ with purely imaginary eigenvalues $\lambda_\pm = \pm i\ell'(0)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 911). Consequently, $\frac{\mathrm{d}}{\mathrm{d}t}(\theta^2 + \phi^2) = 0$: continuous dynamics orbit in closed circles, while discrete simultaneous Euler updates with step size $h > 0$ spiral outward unless regularized.

## Algorithmic Implementation & Comparative Synthesis

### Enforcing $1$-Lipschitzness: Weight Clipping, WGAN-GP, and Spectral Normalization

Approximating the Kantorovich–Rubinstein supremum `(eq:gan:w1_primal_dual)` with a neural critic $f_\phi$ requires enforcing $\lVert \nabla_x f_\phi(x) \rVert_2 \le 1$ almost everywhere. Three principal mechanisms achieve this:
- **Weight Clipping (WGAN):** Clamping weights $\phi \leftarrow \mathrm{clip}(\phi, -c, c)$ (e.g., $c = 0.01$) guarantees bounded Lipschitzness [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 15, p. 285), but severely restricts critic capacity and induces pathological weight saturation at $\pm c$.
- **Gradient Penalty (WGAN-GP):** Exploiting the property that the optimal $W_1$ critic has unit gradient norm along straight lines between coupled real and generated points, WGAN-GP adds a soft penalty along random interpolants $\hat{x} = \epsilon x + (1-\epsilon)\tilde{x}$ with $\epsilon \sim \mathcal{U}(0,1)$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 17, p. 538) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 15, p. 285):
  
$$
L_D^{\mathrm{WGAN\text{-}GP}}(\phi, \theta)
    = \mathbb{E}_{\tilde{x} \sim q_\theta}[f_\phi(\tilde{x})] - \mathbb{E}_{x \sim p_{\mathrm{data}}}[f_\phi(x)]
    + \eta \mathbb{E}_{\hat{x}}\!\left[\left(\lVert \nabla_{\hat{x}} f_\phi(\hat{x}) \rVert_2 - 1\right)^2\right].
$$
- **Spectral Normalization (SN-GAN):** Normalizing each layer's weight matrix $\boldsymbol{W}_l$ by its largest singular value, $\boldsymbol{W}_l^{\mathrm{SN}} := \boldsymbol{W}_l / \sigma_{\max}(\boldsymbol{W}_l)$ (via one power-iteration step per update), enforces $\lVert f_\phi \rVert_{\mathrm{Lip}} \le \prod_l \lVert \boldsymbol{W}_l^{\mathrm{SN}} \rVert_2 = 1$ for $1$-Lipschitz activations without per-sample second-order derivatives [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 918).

> [!example] Algorithm: WGAN-GP Alternating Training (with Spectral-Norm Variant)
> **Input:** Data distribution $p_{\mathrm{data}}$, prior $p_Z$, critic steps $n_{\mathrm{critic}}$ (default $5$), penalty weight $\eta$ (default $10$), batch size $m$, Adam $(\alpha, \beta_1, \beta_2)$ with low $\beta_1 \in [0, 0.5]$.
> **Output:** Trained generator parameters $\theta$ and critic parameters $\phi$.
> Initialize $\theta, \phi$
> **while** $\theta$ has not converged:
>
>   **for** $t = 1, \dots, n_{\mathrm{critic}}$:
>
>     Sample real minibatch $\{x^{(i)}\}_{i=1}^m \sim p_{\mathrm{data}}$ and latent minibatch $\{z^{(i)}\}_{i=1}^m \sim p_Z$
>     Sample $\epsilon^{(i)} \sim \mathcal{U}(0, 1)$ for $i = 1, \dots, m$
>     $\tilde{x}^{(i)} \leftarrow G_\theta(z^{(i)})$,\quad $\hat{x}^{(i)} \leftarrow \epsilon^{(i)} x^{(i)} + (1 - \epsilon^{(i)})\tilde{x}^{(i)}$
>     $L_D^{(i)} \leftarrow f_\phi(\tilde{x}^{(i)}) - f_\phi(x^{(i)}) + \eta \left(\lVert \nabla_{\hat{x}^{(i)}} f_\phi(\hat{x}^{(i)}) \rVert_2 - 1\right)^2$
>     Update $\phi \leftarrow \mathrm{Adam}\!\left(\nabla_\phi \frac{1}{m}\sum_{i=1}^m L_D^{(i)},  \phi,  \alpha, \beta_1, \beta_2\right)$
>     *(SN-GAN variant: omit $\eta$ term and apply $\boldsymbol{W}_l \leftarrow \boldsymbol{W}_l / \sigma_{\max}(\boldsymbol{W}_l)$ per layer)*
>
>   Sample fresh latent minibatch $\{z^{(i)}\}_{i=1}^m \sim p_Z$
>   $L_G \leftarrow -\frac{1}{m}\sum_{i=1}^m f_\phi(G_\theta(z^{(i)}))$
>   Update $\theta \leftarrow \mathrm{Adam}(\nabla_\theta L_G,  \theta,  \alpha, \beta_1, \beta_2)$

### Primal Optimal Transport GANs: mOT-GAN, Partial OT, and Sliced OT-GAN

Dual WGAN formulations are tied to the Euclidean $c(x,y) = \lVert x-y \rVert_2$ cost, which can poorly reflect perceptual similarity in high-dimensional pixel spaces [[Source - Optimal Transport in Large-Scale|optimal_transport]] (p. 53). Direct primal OT-GANs replace the dual critic with an empirical transport cost $\mathrm{OT}_c(\hat{q}_\theta, \hat{p}_{\mathrm{data}})$ over $M$ generated and $n$ real samples, but exact linear programming costs $\mathcal{O}(\max\{M,n\}^3)$ and entropic Sinkhorn costs $\mathcal{O}(\max\{M,n\}^2)$, which is prohibitive for $n \sim 10^6$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (pp. 55–57).
- **Minibatch OT-GAN (mOT-GAN):** Averages OT across $k$ uniformly drawn minibatches of size $m \le \min\{M, n\}$ [[Source - Optimal Transport in Large-Scale|optimal_transport]] (pp. 59–62):
  
$$
\min_{\theta} \frac{1}{k} \sum_{i=1}^{k} \mathrm{OT}_c\!\left((G_\theta)_\# P_{z_i^m},  P_{X_i^m}\right),
$$

  reducing per-step complexity to $\mathcal{O}(km^2)$ with Sinkhorn. However, small-batch sampling induces *misspecified matchings* (cross-mode pairings absent from the global OT plan), which can be alleviated via Minibatch Partial OT ($\mathrm{m\text{-}POT}_s$) transporting only a fraction $s \in (0,1]$ of mass at $\mathcal{O}(k(m+1)^2)$ cost [[Source - Optimal Transport in Large-Scale|optimal_transport]] (pp. 64–69).
- **Sliced OT-GAN ($\mathrm{SW}_p$):** Projects distributions onto $L$ random $1$-D directions $u \in \mathbb{S}^{d-1}$, where $1$-D OT reduces to sorting in $\mathcal{O}(n \log n)$ time with dimension-free sample complexity [[Source - Optimal Transport in Large-Scale|optimal_transport]] (pp. 83–84). For images, vectorizing into $\mathbb{R}^d$ discards spatial structure and incurs $\mathcal{O}(Ld)$ memory; *Convolution Sliced-OT* replaces vector projections with strided/dilated convolutional slicers [[Source - Optimal Transport in Large-Scale|optimal_transport]] (pp. 84–88).

**Table:** Comparative synthesis of adversarial and optimal-transport generative objectives [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, pp. 897–918) [[Source - Optimal Transport in Large-Scale|optimal_transport]] (pp. 51–88).

| **Method** | **Discrepancy** | **Critic Constraint** | **Step Complexity** | **Disjoint Support** |
| — | — | — | — | — |
| Minimax GAN | $\mathrm{JSD}$ | $D_\phi \in (0,1)$ | $\mathcal{O}(m)$ | Saturates ($\log 2$) |
| $f$-GAN | $D_f$ | $T_\phi \in \mathrm{dom}(f^\dagger)$ | $\mathcal{O}(m)$ | Singular / smooth MLP |
| WGAN-GP | $W_1$ (dual) | Soft $\lVert \nabla_{\hat{x}} f_\phi \rVert_2 \approx 1$ | $\mathcal{O}(n_{\mathrm{critic}} m)$ (2nd-order) | Smooth ($\lVert \cdot \rVert_2$) |
| SN-GAN | Hinge / $W_1$ | $\sigma_{\max}(\boldsymbol{W}_l) = 1$ | $\mathcal{O}(n_{\mathrm{critic}} m)$ (power iter.) | Smooth |
| mOT-GAN | Primal $\mathrm{OT}_c$ | None (Sinkhorn) | $\mathcal{O}(km^2)$ | Smooth (misspec. risk) |
| Sliced OT-GAN | $\mathrm{SW}_p$ | None ($L$ projections) | $\mathcal{O}(L m \log m)$ | Smooth ($\mathcal{O}(m\log m)$) |

> [!info] Cross-Book Synthesis: Architectural Stabilization & Quality–Diversity Trade-offs
> Across [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 15, pp. 286–288), [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 17, pp. 539–540), and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, pp. 917–918), loss stabilization alone is insufficient for high-resolution synthesis without architectural inductive biases:
> - **Progressive Growing & StyleGAN:** Training starts at $4\times 4$ resolution and smoothly fades in higher-resolution residual blocks up to $1024\times 1024$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 15, p. 286). StyleGAN further maps $z \sim p_Z$ through an MLP to an intermediate disentangled latent $w \in \mathcal{W}$ that modulates feature maps via Adaptive Instance Normalization (AdaIN) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 15, pp. 295–297).
> - **Truncation Trick:** Sampling $z$ from a truncated distribution (or interpolating $w' = \bar{w} + \tau(w - \bar{w})$ with $\tau \in (0,1]$) trades sample diversity for perceptual fidelity [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 15, p. 287).
> - **Regularization vs. Capacity:** Excessive discriminator regularization (e.g., over-strong gradient penalties or spectral norm) guarantees stability but can degrade sample sharpness [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 918).

## Visual Diagnostics & Geometric Structure

Figure `fig:gan:minimax_vs_wgan` contrasts the standard minimax GAN feedback loop against the $1$-Lipschitz WGAN-GP / Spectral-Norm critic architecture, highlighting how removing the sigmoid bottleneck and enforcing bounded Lipschitzness eliminates gradient saturation.

> [!example] Visual Diagram (LaTeX / PDF)
> *Architectural contrast between (a) the classical minimax GAN with sigmoid saturation and (b) the $1$-Lipschitz WGAN-GP / Spectral-Norm critic yielding well-behaved gradients [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, pp. 900–906) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 15, pp. 282–285).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:gan:mode_collapse_murphy` visualizes mode collapse and mode hopping on a $16$-component Gaussian mixture: as discriminator loss drops from $1.252$ to $0.305$, generator loss climbs and samples collapse onto one or two modes. Figure `fig:gan:mot_misspecified` illustrates the geometric origin of misspecified matchings in minibatch OT-GAN.

> [!note] Source Figure
> *Figure:* Mode collapse and mode hopping in a $16$-Gaussian mixture: (a) true data density; (b–f) generator collapse onto a tiny subset of modes as discriminator loss decreases and generator loss increases [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, Fig. 26.6, p. 908). (`probabilisticml_kevinmurphy_advancedtopics_p942_fig1.png`)

> [!note] Source Figure
> *Figure:* Misspecified matching in minibatch OT-GAN: (left) exact global matching between empirical measures $\mu_n$ and $\nu_n$; (right) minibatch subsampling $(P_{X^m}, P_{Y^m})$ induces false cross-mode pairings (red arrows), motivating minibatch partial OT [[Source - Optimal Transport in Large-Scale|optimal_transport]] (p. 66). (`optimal_transport_p67_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Mode Collapse, Mode Hopping, and Oscillatory Limit Cycles
> **Failure Mode:** As shown by the DiracGAN Jacobian `(eq:gan:game_jacobian)`, unregularized alternating/simultaneous gradient descent exhibits purely imaginary eigenvalues $\lambda_\pm = \pm i\ell'(0)$ and fails to converge [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, pp. 910–911). In multimodal data, the generator either collapses onto a single high-density mode or hops indefinitely between modes without ever increasing support coverage (Figure `fig:gan:mode_collapse_murphy`).
>
> **Remedy:** Lower Adam momentum ($\beta_1 \in [0, 0.5]$ instead of $0.9$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 908), apply instance noise or $R_1$/gradient penalties to damp imaginary eigenvalues, and use progressive growing or minibatch discrimination [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 15, pp. 286–287).

> [!warning] Weight Clipping Collapse vs. WGAN-GP BatchNorm Incompatibility
> **Failure Mode:** Enforcing $1$-Lipschitzness via weight clipping $[-c, c]$ forces critic weights toward the boundary $\pm c$, reducing the critic to a near-linear function and stalling learning. Conversely, WGAN-GP penalizes $\lVert \nabla_{\hat{x}} f_\phi(\hat{x}) \rVert_2$ independently per sample $\hat{x}$; placing `BatchNorm` inside the critic introduces cross-sample dependencies within a minibatch, invalidating the per-sample gradient penalty [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 26, p. 918).
>
> **Remedy:** Replace `BatchNorm` in WGAN-GP critics with `LayerNorm` or `GroupNorm`, or use Spectral Normalization $\boldsymbol{W}_l / \sigma_{\max}(\boldsymbol{W}_l)$ which enforces global Lipschitzness without second-order gradient penalties.

> [!warning] Misspecified Minibatch OT Matchings and Sliced-OT Vectorization
> **Failure Mode:** In mOT-GAN `(eq:gan:mot)`, using $k=1$ and small batch size $m$ forces every generated point in $P_{z^m}$ to match some real point in $P_{X^m}$ even when modes are missing from the subsample, producing biased gradients toward the barycenter (Figure `fig:gan:mot_misspecified`) [[Source - Optimal Transport in Large-Scale|optimal_transport]] (pp. 60, 64–66). In Sliced OT-GAN, flattening high-resolution images to $\mathbb{R}^d$ destroys spatial locality and requires $\mathcal{O}(Ld)$ projection memory [[Source - Optimal Transport in Large-Scale|optimal_transport]] (p. 84).
>
> **Remedy:** Increase $m$ and $k$ or adopt Minibatch Partial OT ($\mathrm{m\text{-}POT}_s$) to discard unmatchable mass [[Source - Optimal Transport in Large-Scale|optimal_transport]] (pp. 67–69); for image synthesis, replace vector projections with strided/dilated Convolution Sliced-OT [[Source - Optimal Transport in Large-Scale|optimal_transport]] (pp. 87–88).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[GAN & Mode Collapse]], [[Wasserstein GAN (WGAN)]]
- **Prerequisites (Directed $\leftarrow$):** [[GAN & Mode Collapse]], [[Kantorovich Duality]], [[Wasserstein Distance]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Wasserstein GAN (WGAN)]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Kantorovich Duality]], [[Wasserstein Distance]]
- **Synthesized Sources:** [[Source - Foundations]], [[Source - Optimal Transport in Large-Scale]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
