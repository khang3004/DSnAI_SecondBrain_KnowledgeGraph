---
type: monograph
domain: "D02"
parent_topic: "[[PT 02 - Latent Variables, Graphical Models, VI & MCMC Sampling]]"
sub_topic: "[[ST 02 - Graphical Models, Gaussian Processes, Variational Inference & MCMC]]"
cheatsheet: "[[Cheatsheet - Probabilistic Graphical Models, Message Passing & Gaussian Processes]]"
part_of: 
  - "[[ST 02 - Graphical Models, Gaussian Processes, Variational Inference & MCMC]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
concepts: 
  - "[[Gaussian Processes]]"
  - "[[Graphical Models (BN & MRF)]]"
prerequisites: 
  - "[[Mercer Kernels & RKHS]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
enables: 
  - "[[MCMC & HMC Sampling]]"
related: 
  - "[[Energy-Based Models (EBM)]]"
  - "[[Mercer Kernels & RKHS]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - monograph
  - domain-02
---

# Probabilistic Graphical Models, Message Passing & Gaussian Processes

> [!abstract] Monograph Navigation
> - **Domain:** [[D02 - Classical, Kernel & Probabilistic Machine Learning|Classical, Kernel & Probabilistic Machine Learning]]
> - **Parent / Sub-Topic:** [[PT 02 - Latent Variables, Graphical Models, VI & MCMC Sampling|Latent Variables, Graphical Models, VI & MCMC Sampling]] → [[ST 02 - Graphical Models, Gaussian Processes, Variational Inference & MCMC|Graphical Models, Gaussian Processes, Variational Inference & MCMC]]
> - **Visual Cheatsheet:** [[Cheatsheet - Probabilistic Graphical Models, Message Passing & Gaussian Processes]]
> - **Description:** Covers directed DAGs (d-separation) vs. undirected MRFs (Hammersley-Clifford), belief propagation/sum-product message passing, and non-parametric Gaussian Process regression, marginal likelihood, and sparse inducing-point scaling.

## Problem Formulation & Unified Notation

High-dimensional probabilistic modeling faces an exponential curse of dimensionality: a joint distribution over $D$ discrete variables with $K$ states requires $K^D - 1$ parameters in general [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 11, p. 346), while a joint distribution over continuous functions $f: \mathcal{X} \to \mathbb{R}$ is infinite-dimensional [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.2, p. 574). Two complementary paradigms render inference and learning tractable:
- **Probabilistic Graphical Models (PGMs)** encode conditional independence (CI) structure via graphs $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, factorizing $p(\boldsymbol{x})$ into local conditional probability distributions (CPDs) on Directed Acyclic Graphs (DAGs) or non-negative clique potentials $\psi_c(\boldsymbol{x}_c)$ on Undirected Markov Random Fields (MRFs) and bipartite factor graphs [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 4, pp. 149, 165, 202).
- **Gaussian Processes (GPs)** place a coherent, non-parametric Gaussian prior directly over function space, $f \sim \mathcal{GP}(m, k)$, so that any finite collection $\boldsymbol{f}_X = [f(\boldsymbol{x}_1),\dots,f(\boldsymbol{x}_N)]^{\top}$ is jointly Gaussian with covariance $\boldsymbol{K}_{X,X} = [k(\boldsymbol{x}_i, \boldsymbol{x}_j)]_{i,j=1}^N$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 18, p. 684). Viewed as a graphical model, a GP forms a dense Gaussian MRF over latent function values $\boldsymbol{f}_X$ and test evaluations $\boldsymbol{f}_*$ with noisy local emissions $y_i \mid f_i \sim \mathcal{N}(f_i, \sigma_y^2)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.2, Fig. 17.6, p. 574).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> {
>
> | **Concept** | **Murphy (Adv / Intro)** | **Bishop / Goodfellow** | **Unified Notation** |
> | — | — | — | — |
> | DAG Factorization | $p(\boldsymbol{x}) = \prod_i p(x_i \mid \boldsymbol{x}_{\mathrm{pa}(i)})$ | $p(\boldsymbol{x}) = \prod_k p(x_k \mid \mathrm{pa}(k))$ | $p(\boldsymbol{x}) = \prod_{i=1}^D p(x_i \mid \boldsymbol{x}_{\mathrm{pa}(i)})$ |
> | Clique Potential | $\psi_c(\boldsymbol{x}_c; \boldsymbol{\theta}_c)$ | $\phi_c(\boldsymbol{x}_c)$ (Goodfellow Ch. 16) | $\psi_c(\boldsymbol{x}_c) = \exp(-E_c(\boldsymbol{x}_c)) > 0$ |
> | Partition Function | $Z(\boldsymbol{\theta}) = \sum_{\boldsymbol{x}} \prod_c \psi_c(\boldsymbol{x}_c)$ | $Z = \int \tilde{p}(\boldsymbol{x}) \mathrm{d}\boldsymbol{x}$ | $Z = \sum_{\boldsymbol{x}} \prod_{c \in \mathcal{C}} \psi_c(\boldsymbol{x}_c)$ |
> | BP Message | $m_{s \to t}(z_t)$ | $\mu_{\alpha \to x}(x)$ (standard) | $m_{s \to t}(x_t)$ / $m_{f \to x}(x)$ |
> | GP Covariance | $\boldsymbol{K}_{\sigma} = \boldsymbol{K} + \sigma_y^2 \boldsymbol{I}$ | — | $\boldsymbol{K}_y = \boldsymbol{K}_{X,X} + \sigma_y^2 \boldsymbol{I}_N$ |
> | Inducing Matrix | $\boldsymbol{Q}_{A,B} = \boldsymbol{K}_{A,Z}\boldsymbol{K}_{Z,Z}^{-1}\boldsymbol{K}_{Z,B}$ | — | $\boldsymbol{Q}_{A,B} = \boldsymbol{K}_{A,Z}\boldsymbol{K}_{Z,Z}^{-1}\boldsymbol{K}_{Z,B}$, $M \ll N$ |
> }

## Core Mathematical Derivations & Theorems

### Directed Acyclic Graphs: d-Separation, Explaining Away & Markov Blankets

> [!important] Global Markov Property (d-Separation) & Markov Blanket
> Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be a DAG encoding $p(\boldsymbol{x}) = \prod_{i=1}^D p(x_i \mid \boldsymbol{x}_{\mathrm{pa}(i)})$. An undirected path $P$ between $A$ and $B$ is **blocked** (d-separated) by evidence set $C$ if and only if $P$ contains at least one triple $(s, m, t)$ satisfying [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 4, Eq. (4.17), p. 149) [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 11, p. 341):
> - **Chain (head-to-tail)** $s \to m \to t$ or $s \leftarrow m \leftarrow t$, with $m \in C$;
> - **Fork (tail-to-tail)** $s \leftarrow m \to t$, with $m \in C$;
> - **Collider (v-structure, head-to-head)** $s \to m \leftarrow t$, where *neither* $m$ nor any descendant $\mathrm{desc}(m)$ is in $C$.
>
> Sets $A$ and $B$ are d-separated given $C$ ($\boldsymbol{x}_A \perp\!\!\!\perp_{\mathcal{G}} \boldsymbol{x}_B \mid \boldsymbol{x}_C$) iff every path between $A$ and $B$ is blocked by $C$. Furthermore, the conditional distribution of $x_i$ given all other nodes $\boldsymbol{x}_{-i}$ depends only on its **Markov blanket** $\mathrm{MB}(i)$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 11, p. 348) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 4, Fig. 4.7, p. 154):
>
> $$
> p(x_i \mid \boldsymbol{x}_{-i})
> = \frac{p(x_i \mid \boldsymbol{x}_{\mathrm{pa}(i)}) \prod_{k \in \mathrm{ch}(i)} p(x_k \mid \boldsymbol{x}_{\mathrm{pa}(k)})}{\int p(x_i \mid \boldsymbol{x}_{\mathrm{pa}(i)}) \prod_{k \in \mathrm{ch}(i)} p(x_k \mid \boldsymbol{x}_{\mathrm{pa}(k)}) \mathrm{d}x_i},
> \quad
> \mathrm{MB}(i) = \mathrm{pa}(i) \cup \mathrm{ch}(i) \cup \mathrm{coparents}(i).
>
> $$

> [!tip] Bayes Ball & Explaining Away in V-Structures
> In a chain $X \to Y \to Z$ or fork $X \leftarrow Y \to Z$, observing $Y$ blocks information flow because $p(x, z \mid y) = p(x \mid y)p(z \mid y)$. In a collider $X \to Y \leftarrow Z$, $X$ and $Z$ are marginally independent ($p(x, z) = p(x)p(z)$), so an unobserved collider blocks the path. Conditioning on $Y$ (or a descendant $Y'$) couples $X$ and $Z$ via **explaining away**: in Bishop's fuel-gauge network ($B \to G \leftarrow F$ with battery $B$, fuel $F$, gauge $G$), observing an empty gauge $G=0$ raises $p(F=0 \mid G=0) \approx 0.257$ above the prior $p(F=0)=0.1$, whereas subsequently observing a flat battery $B=0$ explains away the gauge reading and drops $p(F=0 \mid G=0, B=0) \approx 0.111$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 11, Eqs. (11.34)–(11.35), p. 342).

### Undirected MRFs, Hammersley–Clifford Theorem & Factor Graphs

> [!important] Hammersley–Clifford Theorem & Gibbs Representation
> Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be an undirected graph with maximal cliques $\mathcal{C}$. A strictly positive distribution $p(\boldsymbol{x}) > 0$ satisfies the Markov properties of $\mathcal{G}$ (where $\boldsymbol{x}_A \perp\!\!\!\perp \boldsymbol{x}_B \mid \boldsymbol{x}_C$ whenever $C$ separates $A$ and $B$ by graph separation) if and only if $p$ factorizes as a Gibbs distribution over $\mathcal{C}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 4, Eqs. (4.69)–(4.73), p. 166) [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 16, Eq. (16.7), p. 570):
>
> $$
> p(\boldsymbol{x} \mid \boldsymbol{\theta}) = \frac{1}{Z(\boldsymbol{\theta})} \prod_{c \in \mathcal{C}} \psi_c(\boldsymbol{x}_c; \boldsymbol{\theta}_c)
> = \frac{1}{Z(\boldsymbol{\theta})} \exp\!\left(- \sum_{c \in \mathcal{C}} E_c(\boldsymbol{x}_c; \boldsymbol{\theta}_c)\right),
> \quad
> Z(\boldsymbol{\theta}) = \sum_{\boldsymbol{x}} \prod_{c \in \mathcal{C}} \psi_c(\boldsymbol{x}_c; \boldsymbol{\theta}_c),
>
> $$
>
> where $\psi_c(\boldsymbol{x}_c) = \exp(-E_c(\boldsymbol{x}_c)) > 0$ are positive clique potentials and $Z(\boldsymbol{\theta})$ is the partition function. Unlike DAGs, an undirected node's Markov blanket is simply its immediate graph neighborhood $\mathrm{nbr}(i)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 4, p. 165).

To unify directed and undirected models and resolve clique-factorization ambiguity (e.g., a $3$-clique $\{a,b,c\}$ could represent a single $3$-way factor $f(a,b,c)$ or three pairwise factors $f_1(a,b)f_2(b,c)f_3(a,c)$), we construct a bipartite **factor graph** $\mathcal{F} = (\mathcal{V}_{\mathrm{var}}, \mathcal{V}_{\mathrm{fac}}, \mathcal{E})$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 16, Fig. 16.13, p. 580) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 4, p. 202). Converting a DAG to an MRF requires **moralization**: adding undirected edges between all unmarried parents sharing a child and dropping edge directions, ensuring every CPD $p(x_i \mid \boldsymbol{x}_{\mathrm{pa}(i)})$ fits inside a clique [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 4, p. 196). Conversely, converting an MRF to a DAG requires **triangulation** (adding chords to all cycles of length $\ge 4$ to form a chordal graph) followed by topological edge orientation [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Ch. 16, Fig. 16.12, p. 579).

### Exact Tree Belief Propagation, Loopy BP & Junction Tree Triangulation

> [!important] Sum-Product & Max-Product Belief Propagation on Trees
> Consider a pairwise tree MRF $p(\boldsymbol{z} \mid \boldsymbol{y}) \propto \prod_s \psi_s(z_s) \prod_{(s,t) \in \mathcal{E}} \psi_{st}(z_s, z_t)$. Rooting the tree and orienting edges away from the root, the **Sum-Product** algorithm computes exact posterior marginals $\mathrm{bel}_s(z_s) = p(z_s \mid \boldsymbol{y})$ in two passes (leaves-to-root post-order, then root-to-leaves pre-order) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 9, Eqs. (9.58)–(9.62), pp. 414–415):
>
> $$
> \mathrm{bel}_s(z_s) \propto \psi_s(z_s) \prod_{u \in \mathrm{nbr}(s)} m_{u \to s}(z_s),
> \qquad
> m_{s \to t}(z_t) = \sum_{z_s} \psi_{st}(z_s, z_t)   \psi_s(z_s) \prod_{u \in \mathrm{nbr}(s) \setminus \{t\}} m_{u \to s}(z_s).
>
> $$
>
> Using the identity $\prod_{u \in \mathrm{nbr}(s) \setminus \{t\}} m_{u \to s}(z_s) = \mathrm{bel}_s(z_s) / m_{t \to s}(z_s)$ avoids redundant products. Replacing $\sum_{z_s}$ with $\max_{z_s}$ yields the **Max-Product** algorithm (operating in the max-product semiring, or max-sum in log-space), which computes exact max-marginals $\zeta_i(k) = \max_{\boldsymbol{z}_{-i}} p(z_i = k, \boldsymbol{z}_{-i} \mid \boldsymbol{y})$ for MAP decoding [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 9, Eq. (9.64), p. 416).

When $\mathcal{G}$ contains cycles, two routes exist:
- **Exact Inference via Junction Tree Algorithm (JTA):** Symbolic variable elimination adds fill-in edges to produce a *chordal* (triangulated) graph; maximal cliques are organized into a *junction tree* (clique tree) satisfying the running intersection property, on which tree message passing is exact in time exponential in the treewidth $w^*$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 9, p. 434).
- **Approximate Inference via Loopy BP:** Applying Eq. `(eq:pgmgp:sum_product)` iteratively on cyclic graphs corresponds to finding stationary points of the Bethe free energy. Damped asynchronous updates and priority-scheduled **residual BP** (ranking edges by $\lVert \log m_{s \to t}^{(k+1)} - \log m_{s \to t}^{(k)} \rVert_{\infty}$) substantially improve convergence over synchronous flooding [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 9, Eq. (9.80), pp. 421–422).

### Gaussian Processes: Weight-Space Duality, Kernels & Closed-Form Inference

> [!important] Weight-Space vs. Function-Space Duality & Closed-Form GP Posterior
> Consider Bayesian linear regression $y = f(\boldsymbol{x}) + \epsilon$ with $f(\boldsymbol{x}) = \boldsymbol{w}^{\top} \boldsymbol{\phi}(\boldsymbol{x})$, $\boldsymbol{w} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma}_w)$, and $\epsilon \sim \mathcal{N}(0, \sigma_y^2)$. Defining $\boldsymbol{K}_{X,X} = \boldsymbol{\Phi}\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^{\top}$ and $\boldsymbol{k}_* = \boldsymbol{\Phi}\boldsymbol{\Sigma}_w\boldsymbol{\phi}(\boldsymbol{x}_*)$ with kernel $k(\boldsymbol{x}, \boldsymbol{x}') = \boldsymbol{\phi}(\boldsymbol{x})^{\top} \boldsymbol{\Sigma}_w \boldsymbol{\phi}(\boldsymbol{x}')$, applying the matrix inversion lemma to the weight-space posterior $\boldsymbol{w} \mid \mathcal{D} \sim \mathcal{N}(\sigma_y^{-2}\boldsymbol{A}^{-1} \boldsymbol{\Phi}^{\top} \boldsymbol{y}, \boldsymbol{A}^{-1})$ ($\boldsymbol{A} = \sigma_y^{-2}\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi} + \boldsymbol{\Sigma}_w^{-1}$) yields the identical function-space GP posterior predictive distribution [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.2.4, Eqs. (17.41)–(17.46), p. 577) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 18, p. 695):
>
> $$
> \begin{aligned}
> p(\boldsymbol{f}_* \mid \boldsymbol{X}_*, \mathcal{D}) &= \mathcal{N}(\boldsymbol{f}_* \mid \boldsymbol{\mu}_*, \boldsymbol{\Sigma}_*), \\
> \boldsymbol{\mu}_* &= \boldsymbol{m}(\boldsymbol{X}_*) + \boldsymbol{K}_{*,X} \boldsymbol{K}_y^{-1} \bigl(\boldsymbol{y} - \boldsymbol{m}(\boldsymbol{X})\bigr), \qquad
> \boldsymbol{\Sigma}_* = \boldsymbol{K}_{*,*} - \boldsymbol{K}_{*,X} \boldsymbol{K}_y^{-1} \boldsymbol{K}_{X,*},
> \end{aligned}
>
> $$
>
> where $\boldsymbol{K}_y = \boldsymbol{K}_{X,X} + \sigma_y^2 \boldsymbol{I}_N$. Function-space inference remains valid even when $\boldsymbol{\phi}(\boldsymbol{x})$ is infinite-dimensional, as for the stationary **Squared Exponential (SE/RBF)** and **Matérn-$\nu$** kernels ($r = \lVert \boldsymbol{x} - \boldsymbol{x}' \rVert_2$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 18, Eqs. (18.8)–(18.13), pp. 684–685):
>
> $$
> k_{\mathrm{SE}}(r; \ell) = \sigma_f^2 \exp\!\left(-\frac{r^2}{2\ell^2}\right), \qquad
> k_{\mathrm{Mat\acute{e}rn}}(r; \nu, \ell) = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu} r}{\ell}\right)^{\nu} K_{\nu}\!\left(\frac{\sqrt{2\nu} r}{\ell}\right).
>
> $$
>
> Sample paths from a Matérn-$\nu$ GP are $k$-times mean-square differentiable iff $\nu > k$: $\nu = 1/2$ yields rough Ornstein–Uhlenbeck paths ($C^0$), $\nu = 3/2$ yields $C^1$ paths, and $\nu \to \infty$ recovers the $C^\infty$ SE kernel. Replacing $r^2/\ell^2$ by $\sum_{d=1}^D (x_d - x'_d)^2/\ell_d^2$ gives the Automatic Relevance Determination (ARD) kernel [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 18, Eq. (18.11), p. 684).

### Log Marginal Likelihood (Occam's Razor) & Inducing-Point Sparse GPs

> [!important] Empirical Bayes Marginal Likelihood & Inducing-Point Sparse GPs
> Integrating out the latent function values $\boldsymbol{f}_X \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{K}_{X,X})$ under Gaussian likelihood $p(\boldsymbol{y} \mid \boldsymbol{f}_X) = \mathcal{N}(\boldsymbol{y} \mid \boldsymbol{f}_X, \sigma_y^2\boldsymbol{I}_N)$ gives the exact log marginal likelihood [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 18, Eq. (18.163), p. 718):
>
> $$
> \begin{aligned}
> \log p(\boldsymbol{y} \mid \boldsymbol{X}, \boldsymbol{\theta})
> &= \underbrace{-\frac{1}{2} \boldsymbol{y}^{\top} \boldsymbol{K}_y^{-1} \boldsymbol{y}}_{\text{data-fit term}}
>   \underbrace{-\frac{1}{2} \log\det \boldsymbol{K}_y}_{\text{complexity penalty}}
>   -\frac{N}{2}\log(2\pi), \\
> \frac{\partial }{\partial \theta_j}\log p(\boldsymbol{y} \mid \boldsymbol{X}, \boldsymbol{\theta})
> &= \frac{1}{2}\mathrm{Tr}\!\left((\boldsymbol{\alpha}\boldsymbol{\alpha}^{\top} - \boldsymbol{K}_y^{-1})\frac{\partial \boldsymbol{K}_y}{\partial \theta_j}\right),
> \end{aligned}
>
> $$
>
> with $\boldsymbol{\alpha} = \boldsymbol{K}_y^{-1} \boldsymbol{y}$. To overcome the $\mathcal{O}(N^3)$ bottleneck of $\boldsymbol{K}_y^{-1}$, **inducing-point sparse GPs** introduce $M \ll N$ pseudo-inputs $\boldsymbol{Z} \in \mathbb{R}^{M \times D}$ with latent values $\boldsymbol{u} = f(\boldsymbol{Z})$ and assume conditional independence $\boldsymbol{f}_* \perp\!\!\!\perp \boldsymbol{f}_X \mid \boldsymbol{u}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 18, Eqs. (18.116)–(18.129), pp. 707–709). Using the Nyström low-rank matrix $\boldsymbol{Q}_{A,B} = \boldsymbol{K}_{A,Z}\boldsymbol{K}_{Z,Z}^{-1}\boldsymbol{K}_{Z,B}$ and $\boldsymbol{\Sigma} = (\sigma_y^{-2}\boldsymbol{K}_{Z,X}\boldsymbol{K}_{X,Z} + \boldsymbol{K}_{Z,Z})^{-1}$:
> - **SOR / DIC:** $q(\boldsymbol{f}_X, \boldsymbol{f}_*)$ replaces $\boldsymbol{K}$ with $\boldsymbol{Q}$ everywhere, yielding $\mu_* = \sigma_y^{-2}\boldsymbol{K}_{*,Z}\boldsymbol{\Sigma}\boldsymbol{K}_{Z,X}\boldsymbol{y}$ and variance $\boldsymbol{K}_{*,Z}\boldsymbol{\Sigma}\boldsymbol{K}_{Z,*}$ (rank-$M$, underestimates uncertainty far from $\boldsymbol{Z}$).
> - **DTC:** Retains exact test prior $\boldsymbol{K}_{*,*}$, giving variance $\boldsymbol{K}_{*,*} - \boldsymbol{Q}_{*,*} + \boldsymbol{K}_{*,Z}\boldsymbol{\Sigma}\boldsymbol{K}_{Z,*}$.
> - **FITC:** Uses diagonal correction $\boldsymbol{K}_{X,X} \approx \boldsymbol{Q}_{X,X} + \mathrm{diag}(\boldsymbol{K}_{X,X} - \boldsymbol{Q}_{X,X})$ so $q(\boldsymbol{f}_X \mid \boldsymbol{u})$ factorizes across data points.
>
> All three reduce training from $\mathcal{O}(N^3)$ to $\mathcal{O}(NM^2 + M^3)$ and per-test prediction to $\mathcal{O}(M)$ mean / $\mathcal{O}(M^2)$ variance.

> [!tip] Automatic Occam's Razor in Marginal Likelihood
> Equation `(eq:pgmgp:lml)` automatically balances fit and complexity without cross-validation: a short lengthscale $\ell$ makes $\boldsymbol{K}_{X,X} \approx \sigma_f^2 \boldsymbol{I}_N$, achieving near-zero residual $\boldsymbol{y}^{\top}\boldsymbol{K}_y^{-1}\boldsymbol{y}$ at the cost of a large $\log\det\boldsymbol{K}_y$ penalty (high effective degrees of freedom); an overly long $\ell$ produces a rank-deficient, near-constant $\boldsymbol{K}_{X,X}$ with small $\log\det\boldsymbol{K}_y$ but poor data fit $\boldsymbol{y}^{\top}\boldsymbol{K}_y^{-1}\boldsymbol{y}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 18, p. 718).

## Algorithmic Implementation & Comparative Synthesis

> [!example] Algorithm: Tree Sum-Product Belief Propagation (Collect–Distribute)
> **Input:** Tree MRF $\mathcal{T} = (\mathcal{V}, \mathcal{E})$, node potentials $\{\psi_s(z_s)\}$, edge potentials $\{\psi_{st}(z_s, z_t)\}$, root $r \in \mathcal{V}$
> **Output:** Exact posterior node marginals $\{\mathrm{bel}_s(z_s)\}_{s \in \mathcal{V}}$
> Orient edges away from root $r$ to define $\mathrm{pa}(s)$ and $\mathrm{ch}(s)$
> **for each** node $s \in \mathcal{V} \setminus \{r\}$ in post-order (leaves to root):
>
>   $\mathrm{bel}_s(z_s) \gets \psi_s(z_s) \prod_{c \in \mathrm{ch}(s)} m_{c \to s}(z_s)$, normalize $\mathrm{bel}_s$
>   $t \gets \mathrm{pa}(s)$
>   $m_{s \to t}(z_t) \gets \sum_{z_s} \psi_{st}(z_s, z_t)  \mathrm{bel}_s(z_s)$
>
> $\mathrm{bel}_r(z_r) \gets \psi_r(z_r) \prod_{c \in \mathrm{ch}(r)} m_{c \to r}(z_r)$, normalize $\mathrm{bel}_r$
> **for each** node $t \in \mathcal{V} \setminus \{r\}$ in pre-order (root to leaves):
>
>   $s \gets \mathrm{pa}(t)$
>   $m_{s \to t}(z_t) \gets \sum_{z_s} \psi_{st}(z_s, z_t)  \frac{\mathrm{bel}_s(z_s)}{m_{t \to s}(z_s)}$
>   $\mathrm{bel}_t(z_t) \gets \mathrm{bel}_t(z_t)  m_{s \to t}(z_t)$, normalize $\mathrm{bel}_t$

> [!example] Algorithm: Numerically Stable Cholesky GP Regression & Marginal Likelihood
> **Input:** Training data $(\boldsymbol{X}, \boldsymbol{y})$, kernel $k(\cdot,\cdot;\boldsymbol{\theta})$, noise $\sigma_y^2$, jitter $\epsilon = 10^{-6}$, test inputs $\boldsymbol{X}_*$
> **Output:** Predictive mean $\boldsymbol{\mu}_*$, covariance $\boldsymbol{\Sigma}_*$, log marginal likelihood $\log p(\boldsymbol{y} \mid \boldsymbol{X}, \boldsymbol{\theta})$
> Compute $\boldsymbol{K}_y \gets k(\boldsymbol{X}, \boldsymbol{X}; \boldsymbol{\theta}) + (\sigma_y^2 + \epsilon)\boldsymbol{I}_N$
> Factor $\boldsymbol{L} \gets \mathrm{cholesky}(\boldsymbol{K}_y)$  *// Lower triangular $\boldsymbol{L}\boldsymbol{L}^{\top} = \boldsymbol{K}_y$, $\mathcal{O}(N^3)$*
> Solve $\boldsymbol{\alpha} \gets \boldsymbol{L}^{\top} \backslash (\boldsymbol{L} \backslash \boldsymbol{y})$  *// Forward/backward substitution, $\mathcal{O}(N^2)$*
> $\boldsymbol{\mu}_* \gets \boldsymbol{K}_{*,X} \boldsymbol{\alpha}$
> Solve $\boldsymbol{V} \gets \boldsymbol{L} \backslash \boldsymbol{K}_{X,*}$; \quad $\boldsymbol{\Sigma}_* \gets \boldsymbol{K}_{*,*} - \boldsymbol{V}^{\top} \boldsymbol{V}$
> $\log p(\boldsymbol{y} \mid \boldsymbol{X}, \boldsymbol{\theta}) \gets -\frac{1}{2}\boldsymbol{y}^{\top}\boldsymbol{\alpha} - \sum_{i=1}^N \log L_{ii} - \frac{N}{2}\log(2\pi)$

**Table:** Asymptotic time and space complexity across exact/approximate PGM inference and GP regression [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 9, 18).

| **Method** | **Train / Setup** | **Query / Pass** | **Space** | **Exactness & Scope** |
| — | — | — | — | — |
| Tree Sum-Product BP | — | $\mathcal{O}(\mid \mathcal{E}\mid  K^2)$ | $\mathcal{O}(\mid \mathcal{E}\mid  K)$ | Exact marginals on trees/polytrees |
| Junction Tree (JTA) | Triangulation | $\mathcal{O}(\mid \mathcal{C}\mid  K^{w^*+1})$ | $\mathcal{O}(\mid \mathcal{C}\mid  K^{w^*+1})$ | Exact on general graphs ($w^*$ = treewidth) |
| Loopy BP (Residual) | — | $\mathcal{O}(T_{\mathrm{it}}\mid \mathcal{E}\mid  K^2)$ | $\mathcal{O}(\mid \mathcal{E}\mid  K)$ | Approx. (Bethe stationary points) |
| Exact Cholesky GP | $\mathcal{O}(N^3)$ | $\mathcal{O}(N)$ / $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ | Exact Gaussian posterior mean / var |
| Sparse GP (SOR/DTC/FITC) | $\mathcal{O}(NM^2 + M^3)$ | $\mathcal{O}(M)$ / $\mathcal{O}(M^2)$ | $\mathcal{O}(NM)$ | $M \ll N$ inducing points $\boldsymbol{Z}$ |

> [!info] Cross-Book Synthesis: Structural Factorization, Inference & Non-Parametric Scaling
> Murphy (Adv. Ch. 4, 9) and Goodfellow et al. (Ch. 16) emphasize that graph topology governs both statistical expressivity and computational tractability: DAGs permit $\mathcal{O}(D)$ ancestral sampling without partition functions, whereas MRFs trade global normalization $Z$ for symmetric spatial potentials and superior conditional discrimination (CRFs). Bishop (DLFC Ch. 11) grounds d-separation in the generative fuel-gauge collider, demonstrating how explaining away necessitates co-parents in the Markov blanket. Finally, Murphy (Intro Sec. 17.2; Adv. Ch. 18) bridges finite-dimensional Bayesian linear models to non-parametric GPs: exact Cholesky inference (Alg. `alg:pgmgp:cholesky_gp`) provides calibrated $\pm 2\sigma$ uncertainty and automatic Occam's razor via $\log\det\boldsymbol{K}_y$, while inducing-point factorizations ($\boldsymbol{f}_* \perp\!\!\!\perp \boldsymbol{f}_X \mid \boldsymbol{u}$) apply graphical conditional independence directly to dense kernel matrices to recover $\mathcal{O}(NM^2)$ scalability.

## Visual Diagnostics & Geometric Structure

> [!note] Source Figure
> *Figure:* Bayes ball rules for d-separation in DAGs [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 4, Fig. 4.4, p. 150). Shaded nodes denote observed evidence $C$: (a)–(d) chains and forks block paths when observed; (e)–(f) colliders (v-structures) block paths when unobserved and unblock (explaining away) when observed. (`probabilisticml_kevinmurphy_advancedtopics_p184_fig1.png`)

> [!example] Visual Diagram (LaTeX / PDF)
> *Conversion from (a) a DAG v-structure ($X_1 \to X_3 \leftarrow X_2$) with observed collider $X_3$ (explaining away) to (b) a moralized MRF with dashed moral edge $X_1 - X_2$, and (c) an unambiguous bipartite factor graph separating unary priors $f_1, f_2$ from the $3$-way CPD factor $f_3$.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!note] Source Figure
> *Figure:* GP prior and posterior samples with SE kernel [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 18, Fig. 18.7, p. 694). (a) Prior draws $f \sim \mathcal{GP}(0, k_{\mathrm{SE}})$. (b) Noise-free observations ($\sigma_y=0$) collapse predictive variance to zero at data points. (c) Noisy observations ($\sigma_y > 0$) retain non-zero uncertainty bands ($\mu_* \pm 2\sigma_*$). (`probabilisticml_kevinmurphy_advancedtopics_p728_fig1.png`)

> [!example] Visual Diagram (LaTeX / PDF)
> *Left: Matérn-$1/2$ exhibits a sharp cusp at $r=0$ (rough $C^0$ paths), whereas Matérn-$3/2$ ($C^1$) and SE ($C^\infty$) are smooth at the origin. Right: GP posterior mean $\mu_*(x)$ with $\pm 2\sigma_*(x)$ credible band contracting near observed data and reverting toward prior uncertainty in extrapolation gaps.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!note] Source Figure
> *Figure:* Effect of GP hyperparameters $(\ell, \sigma_f, \sigma_y)$ on 1D SE regression [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 18, Fig. 18.15, p. 718). (a) $(\ell,\sigma_f,\sigma_y)=(1.0,1.0,0.1)$ captures signal structure with tight uncertainty. (b) $(\ell,\sigma_f,\sigma_y)=(3.0,1.16,0.89)$ corresponds to an alternative local optimum of the marginal likelihood that over-smooths the signal as observation noise. (`probabilisticml_kevinmurphy_advancedtopics_p752_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Loopy BP Non-Convergence, Oscillation & Overconfident Marginals
> On graphs with short, strongly coupled cycles (e.g., ferromagnetic Ising grids at low temperature), synchronous Loopy BP frequently oscillates or diverges because messages reinforce their own echoes around cycles. Even when convergent, Bethe marginals can be severely overconfident. **Remedy:** Use damped asynchronous updates $m^{(k+1)} \gets (1-\alpha)m^{(k)} + \alpha m^{\mathrm{new}}$ ($\alpha \in (0, 0.5]$), residual BP scheduling (Eq. `(eq:pgmgp:sum_product)`), or convexified BP / Junction Tree triangulation when treewidth permits [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 9, pp. 421–423).

> [!warning] Naive $\boldsymbol{K}_y^{-1}$ Inversion Instability vs. Jittered Cholesky
> For smooth kernels like SE ($C^\infty$) with closely spaced inputs or small noise $\sigma_y^2$, $\boldsymbol{K}_y$ is exponentially ill-conditioned ($\kappa(\boldsymbol{K}_y) > 10^{12}$). Explicit inversion $\boldsymbol{K}_y^{-1}$ causes catastrophic cancellation and negative predictive variances $\sigma_*^2 < 0$. **Remedy:** Always compute $\boldsymbol{L}\boldsymbol{L}^{\top} = \boldsymbol{K}_y + \epsilon \boldsymbol{I}_N$ via Cholesky with adaptive diagonal jitter ($\epsilon \approx 10^{-6}\sigma_f^2$) and evaluate $\log\det\boldsymbol{K}_y = 2\sum_{i=1}^N \log L_{ii}$ as in Algorithm `alg:pgmgp:cholesky_gp` [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 17.2.5, p. 578).

> [!warning] Cubic $\mathcal{O}(N^3)$ Bottleneck & Inconsistent Nyström Variances
> Exact GP inference scales as $\mathcal{O}(N^3)$ time and $\mathcal{O}(N^2)$ memory, becoming intractable for $N > 10^4$. Furthermore, substituting the Nyström approximation $\boldsymbol{Q}_{X,X}$ only in $\boldsymbol{K}_{X,X}$ while keeping exact $\boldsymbol{K}_{*,X}$ can produce negative predictive variances. **Remedy:** Use consistent inducing-point priors (DTC, FITC, or variational Titsias/SVGP) with $M \ll N$ inducing points, guaranteeing positive semidefinite covariances in $\mathcal{O}(NM^2 + M^3)$ time [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 18, pp. 707–709).

> [!warning] Marginal Likelihood Hyperparameter Local Optima
> The GP log marginal likelihood $\log p(\boldsymbol{y} \mid \boldsymbol{X}, \boldsymbol{\theta})$ is non-convex in $(\ell, \sigma_f, \sigma_y)$ and routinely exhibits multiple local optima: one explaining variation as short-lengthscale signal with low noise (Fig. `fig:pgmgp:gp_hparams`a), and another explaining it as long-lengthscale trend with high observation noise (Fig. `fig:pgmgp:gp_hparams`b). **Remedy:** Optimize $\log p(\boldsymbol{y} \mid \boldsymbol{X}, \boldsymbol{\theta})$ from multiple log-space restarts, place weakly informative hyperpriors on $(\ell, \sigma_y)$, or marginalize $\boldsymbol{\theta}$ via HMC/quadrature [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 18, pp. 718–720).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[Gaussian Processes]], [[Graphical Models (BN & MRF)]]
- **Prerequisites (Directed $\leftarrow$):** [[Mercer Kernels & RKHS]], [[Multivariate Gaussian & Schur Complement]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[MCMC & HMC Sampling]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Energy-Based Models (EBM)]], [[Mercer Kernels & RKHS]], [[Multivariate Gaussian & Schur Complement]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]]
