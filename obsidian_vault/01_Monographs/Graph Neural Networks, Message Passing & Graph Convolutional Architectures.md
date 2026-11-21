---
type: monograph
domain: "D03"
parent_topic: "[[PT 03 - Convolutional, Graph, Recurrent & State-Space Architectures]]"
sub_topic: "[[ST 03 - Convolutional Networks & Graph Neural Networks]]"
cheatsheet: "[[Cheatsheet - Graph Neural Networks, Message Passing & Graph Convolutional Architectures]]"
part_of: 
  - "[[ST 03 - Convolutional Networks & Graph Neural Networks]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
concepts: 
  - "[[Graph Neural Networks (MPNN)]]"
prerequisites: []
enables: []
related: 
  - "[[Self-Attention & Multi-Head Attention]]"
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-03
---

# Graph Neural Networks, Message Passing & Graph Convolutional Architectures

> [!abstract] Monograph Navigation
> - **Domain:** [[D03 - Neural Architectures, Representations & Regularization|Neural Architectures, Representations & Regularization]]
> - **Parent / Sub-Topic:** [[PT 03 - Convolutional, Graph, Recurrent & State-Space Architectures|Convolutional, Graph, Recurrent & State-Space Architectures]] → [[ST 03 - Convolutional Networks & Graph Neural Networks|Convolutional Networks & Graph Neural Networks]]
> - **Visual Cheatsheet:** [[Cheatsheet - Graph Neural Networks, Message Passing & Graph Convolutional Architectures]]
> - **Description:** Derives permutation equivariance/invariance on irregular graphs, spectral graph Laplacians vs. spatial message passing (GCN, GraphSAGE, GAT), neighborhood aggregation, and the over-smoothing bottleneck.

## Problem Formulation & Unified Notation

Unlike Euclidean grid data (images) or linear sequences (text), relational data lack a canonical coordinate system or ordering. Let $G = (\mathcal{V}, \mathcal{E})$ denote a graph with $|\mathcal{V}| = N$ nodes indexed arbitrarily by $i \in \{1,\dots,N\}$ and $|\mathcal{E}| = M$ edges $(i,j) \in \mathcal{E}$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, p. 410). Each node $i$ carries a feature vector $\boldsymbol{x}_i \in \mathbb{R}^D$, stacked row-wise into the node feature matrix $\boldsymbol{X} \in \mathbb{R}^{N \times D}$, while optional edge attributes are denoted $\boldsymbol{e}_{ij} \in \mathbb{R}^{D_e}$. The topology is encoded by the adjacency matrix $\boldsymbol{A} \in \{0,1\}^{N \times N}$ (or weighted affinity $\boldsymbol{A} \in \mathbb{R}_{\ge 0}^{N \times N}$), where $A_{ij} > 0$ iff $(i,j) \in \mathcal{E}$. The open neighborhood of node $i$ is $\mathcal{N}(i) = \{j \in \mathcal{V} : (i,j) \in \mathcal{E}\}$, its degree is $d_i = \sum_{j=1}^N A_{ij}$, and $\boldsymbol{D} = \mathrm{diag}(d_1,\dots,d_N)$ is the degree matrix. Adding self-loops yields $\tilde{\boldsymbol{A}} = \boldsymbol{A} + \boldsymbol{I}_N$ and $\tilde{\boldsymbol{D}} = \boldsymbol{D} + \boldsymbol{I}_N$ with $\tilde{d}_i = d_i + 1$.

Because node indexing is arbitrary, any relabeling by a permutation matrix $\boldsymbol{P} \in \{0,1\}^{N \times N}$ ($\boldsymbol{P}^{\top}\boldsymbol{P} = \boldsymbol{I}_N$) transforms the inputs as $(\boldsymbol{X}, \boldsymbol{A}) \mapsto (\boldsymbol{P}\boldsymbol{X}, \boldsymbol{P}\boldsymbol{A}\boldsymbol{P}^{\top})$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, p. 412), [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13, p. 249). Rather than learning over $N!$ permutations via data augmentation, graph neural networks (GNNs) enforce two structural inductive biases by construction:

$$
\begin{aligned}
f(\boldsymbol{P}\boldsymbol{X}, \boldsymbol{P}\boldsymbol{A}\boldsymbol{P}^{\top}) &= f(\boldsymbol{X}, \boldsymbol{A}) \quad &&\text{(Permutation Invariance: graph-level tasks)}, \\
  \boldsymbol{F}(\boldsymbol{P}\boldsymbol{X}, \boldsymbol{P}\boldsymbol{A}\boldsymbol{P}^{\top}) &= \boldsymbol{P}\boldsymbol{F}(\boldsymbol{X}, \boldsymbol{A}) \quad &&\text{(Permutation Equivariance: node/edge-level tasks)}.
\end{aligned}
$$

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Source textbooks—Bishop & Bishop [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13), Prince [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13), and Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 23)—adopt different matrix orientations and symbol conventions for node features, neighborhoods, and adjacency operators. Table `tab:gnn:notation` reconciles these into a unified row-major monograph standard.
>
> | **Concept** | **Bishop & Bishop** | **Prince** | **Murphy / Unified** |
> | --- | --- | --- | --- |
> | Node features | $\boldsymbol{X} \in \mathbb{R}^{N \times D}$ (rows) | $\boldsymbol{X} \in \mathbb{R}^{D \times N}$ (columns) | $\boldsymbol{X} \in \mathbb{R}^{N \times D}$ (row $i$ is $\boldsymbol{x}_i^{\top}$) |
> | Hidden states | $\boldsymbol{H}^{(l)} \in \mathbb{R}^{N \times d_l}$ | $\boldsymbol{H}_k \in \mathbb{R}^{d_k \times N}$ | $\boldsymbol{H}^{(\ell)} \in \mathbb{R}^{N \times d_\ell}$ (row $i$ is $(\boldsymbol{h}_i^{(\ell)})^{\top}$) |
> | Adjacency / Affinity | $\boldsymbol{A} \in \{0,1\}^{N \times N}$ | $\boldsymbol{A} \in \{0,1\}^{N \times N}$ | $\boldsymbol{W}$ (Murphy) / $\boldsymbol{A}$ (Unified); $\tilde{\boldsymbol{A}} = \boldsymbol{A} + \boldsymbol{I}_N$ |
> | Permutation action | $\boldsymbol{P}\boldsymbol{X},  \boldsymbol{P}\boldsymbol{A}\boldsymbol{P}^{\top}$ | $\boldsymbol{H}_k\boldsymbol{P},  \boldsymbol{P}^{\top}\boldsymbol{A}\boldsymbol{P}$ | $\boldsymbol{P}\boldsymbol{H}^{(\ell)},  \boldsymbol{P}\boldsymbol{A}\boldsymbol{P}^{\top}$ |
> | Neighborhood set | $\mathcal{N}(n)$ | $\mathrm{ne}[n]$ | $\mathrm{nbr}(v_i)$ (Murphy) / $\mathcal{N}(i)$ (Unified) |
> | Layer weights | $\boldsymbol{W}_{\mathrm{self}}, \boldsymbol{W}_{\mathrm{neigh}}$ | $\boldsymbol{\Omega}_k, \boldsymbol{\Psi}_k, \boldsymbol{\beta}_k$ | $\boldsymbol{\Theta}^{(\ell)}$ (Murphy) / $\boldsymbol{W}^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell+1}}$ |

## Core Mathematical Derivations & Theorems

### Spectral Graph Theory, Chebyshev Localization & Kipf–Welling GCN

On a regular grid, convolution is diagonalized by the Discrete Fourier Transform. On an undirected graph, the analogous Fourier basis is given by the eigendecomposition of the graph Laplacian [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13, p. 262), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 23, p. 763).

> [!important] Spectral Graph Laplacian & Graph Fourier Transform
> For an undirected graph with symmetric adjacency $\boldsymbol{A}$ and degree matrix $\boldsymbol{D}$, the unnormalized and symmetric normalized graph Laplacians are
>
> $$
> \boldsymbol{L} = \boldsymbol{D} - \boldsymbol{A}, \qquad \boldsymbol{L}_{\mathrm{sym}} = \boldsymbol{D}^{-1/2}\boldsymbol{L}\boldsymbol{D}^{-1/2} = \boldsymbol{I}_N - \boldsymbol{D}^{-1/2}\boldsymbol{A}\boldsymbol{D}^{-1/2}.
> $$
>
> For any signal $\boldsymbol{x} \in \mathbb{R}^N$, the quadratic form (Dirichlet energy) satisfies $\boldsymbol{x}^{\top}\boldsymbol{L}\boldsymbol{x} = \frac{1}{2}\sum_{i,j} A_{ij}(x_i - x_j)^2 \ge 0$, proving $\boldsymbol{L} \succeq 0$ and $\boldsymbol{L}_{\mathrm{sym}} \succeq 0$. Moreover, $\boldsymbol{L}_{\mathrm{sym}}$ admits an orthonormal eigendecomposition $\boldsymbol{L}_{\mathrm{sym}} = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top}$ with eigenvalues $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_N \le 2$. Defining the Graph Fourier Transform $\hat{\boldsymbol{x}} = \boldsymbol{U}^{\top}\boldsymbol{x}$ and its inverse $\boldsymbol{x} = \boldsymbol{U}\hat{\boldsymbol{x}}$, a spectral convolution with filter $g_\theta(\boldsymbol{\Lambda}) = \mathrm{diag}(\theta_1,\dots,\theta_N)$ is
>
> $$
> g_\theta \star_G \boldsymbol{x} = \boldsymbol{U}  g_\theta(\boldsymbol{\Lambda})  \boldsymbol{U}^{\top} \boldsymbol{x}.
> $$
>

Direct evaluation of `(eq:gnn:spectral_conv)` suffers from three bottlenecks: (i) $O(N^3)$ eigendecomposition of $\boldsymbol{L}_{\mathrm{sym}}$, (ii) $O(N^2)$ dense multiplication by $\boldsymbol{U}$, and (iii) non-localized filters that depend on the fixed eigenbasis $\boldsymbol{U}$ and cannot transfer across graphs [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 23, p. 763). ChebNet and the Kipf–Welling Graph Convolutional Network (GCN) resolve all three via polynomial approximation.

> [!important] Chebyshev Polynomial Localization to 1-Hop GCN
> Approximating $g_\theta(\boldsymbol{\Lambda})$ by a $K$-th order truncated expansion in Chebyshev polynomials $T_k(z)$ ($T_0(z)=1$, $T_1(z)=z$, $T_k(z)=2zT_{k-1}(z)-T_{k-2}(z)$) on the rescaled spectrum $\tilde{\boldsymbol{\Lambda}} = \frac{2}{\lambda_{\max}}\boldsymbol{\Lambda} - \boldsymbol{I}_N \in [-\boldsymbol{I}_N, \boldsymbol{I}_N]$ yields a $K$-localized spatial operator [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13, p. 262):
>
> $$
> g_\theta \star_G \boldsymbol{x} \approx \sum_{k=0}^K \theta_k T_k(\tilde{\boldsymbol{L}})\boldsymbol{x}, \qquad \tilde{\boldsymbol{L}} = \frac{2}{\lambda_{\max}}\boldsymbol{L}_{\mathrm{sym}} - \boldsymbol{I}_N.
> $$
>
> Setting $K=1$ and $\lambda_{\max} \approx 2$ gives $\tilde{\boldsymbol{L}} = \boldsymbol{L}_{\mathrm{sym}} - \boldsymbol{I}_N = -\boldsymbol{D}^{-1/2}\boldsymbol{A}\boldsymbol{D}^{-1/2}$, so
>
> $$
> g_\theta \star_G \boldsymbol{x} \approx \theta_0 \boldsymbol{x} - \theta_1 \boldsymbol{D}^{-1/2}\boldsymbol{A}\boldsymbol{D}^{-1/2}\boldsymbol{x}.
> $$
>
> Constraining $\theta = \theta_0 = -\theta_1$ to prevent overfitting reduces this to $\theta(\boldsymbol{I}_N + \boldsymbol{D}^{-1/2}\boldsymbol{A}\boldsymbol{D}^{-1/2})\boldsymbol{x}$. Because $\boldsymbol{I}_N + \boldsymbol{D}^{-1/2}\boldsymbol{A}\boldsymbol{D}^{-1/2}$ has eigenvalues in $[0, 2]$, repeated application causes numerical instability. Applying the *renormalization trick* $\boldsymbol{I}_N + \boldsymbol{D}^{-1/2}\boldsymbol{A}\boldsymbol{D}^{-1/2} \mapsto \tilde{\boldsymbol{D}}^{-1/2}\tilde{\boldsymbol{A}}\tilde{\boldsymbol{D}}^{-1/2}$ (with $\tilde{\boldsymbol{A}} = \boldsymbol{A} + \boldsymbol{I}_N$ and $\tilde{D}_{ii} = d_i + 1$) and generalizing to $d_\ell$ input channels and $d_{\ell+1}$ output channels yields the Kipf–Welling GCN layer [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, p. 416), [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13, p. 258):
>
> $$
> \boldsymbol{H}^{(\ell+1)} = \sigma\!\left(\hat{\boldsymbol{A}}\boldsymbol{H}^{(\ell)}\boldsymbol{W}^{(\ell)}\right), \qquad \hat{\boldsymbol{A}} := \tilde{\boldsymbol{D}}^{-1/2}\tilde{\boldsymbol{A}}\tilde{\boldsymbol{D}}^{-1/2},
> $$
>
> where $\boldsymbol{W}^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell+1}}$ is a shared linear projection and $\sigma(\cdot)$ is an element-wise nonlinearity.

> [!tip] Why Symmetric Kipf Normalization Down-Weights High-Degree Hubs
> In node-wise form, `(eq:gnn:gcn_layer)` computes $\boldsymbol{h}_i^{(\ell+1)} = \sigma\!\Big((\boldsymbol{W}^{(\ell)})^{\top} \sum_{j \in \mathcal{N}(i)\cup\{i\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} \boldsymbol{h}_j^{(\ell)}\Big)$. Unlike unnormalized summation $\sum_j \boldsymbol{h}_j^{(\ell)}$ (which scales linearly with degree) or row-stochastic mean $\tilde{\boldsymbol{D}}^{-1}\tilde{\boldsymbol{A}}$ (which normalizes only by $\tilde{d}_i$), symmetric normalization divides by $\sqrt{\tilde{d}_i \tilde{d}_j}$. Messages from high-degree hubs ($\tilde{d}_j \gg 1$) are attenuated because hubs broadcast to many neighbors and carry less node-specific information [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13, p. 258), while keeping $\hat{\boldsymbol{A}}$ symmetric with spectrum in $(-1, 1]$.

### Message Passing Neural Networks (MPNN), GAT & GraphSAGE

Spatial methods generalize GCNs by defining local neighborhood aggregation directly on $G$, decoupling the architecture from a fixed Laplacian [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, p. 415), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 23, p. 763).

> [!important] General MPNN Framework & Permutation Equivariance Proof
> A Message Passing Neural Network (MPNN) updates every node $i \in \mathcal{V}$ at layer $\ell$ via three stages [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, pp. 415–424), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 23, p. 763):
>
> $$
> \boldsymbol{m}_i^{(\ell+1)} = \bigoplus_{j \in \mathcal{N}(i)} \psi_{\mathrm{msg}}^{(\ell)}\!\left(\boldsymbol{h}_i^{(\ell)}, \boldsymbol{h}_j^{(\ell)}, \boldsymbol{e}_{ij}\right), \qquad \boldsymbol{h}_i^{(\ell+1)} = \psi_{\mathrm{upd}}^{(\ell)}\!\left(\boldsymbol{h}_i^{(\ell)}, \boldsymbol{m}_i^{(\ell+1)}\right),
> $$
>
> where $\bigoplus$ is any differentiable, permutation-invariant multiset operator (e.g., $\sum$, $\mathrm{mean}$, $\max$), and $\psi_{\mathrm{msg}}^{(\ell)}, \psi_{\mathrm{upd}}^{(\ell)}$ are shared learnable functions (typically MLPs or linear maps). For graph-level tasks, a permutation-invariant readout maps $\boldsymbol{H}^{(L)}$ to $\boldsymbol{h}_G = \mathrm{READOUT}(\{\!\!\{\boldsymbol{h}_i^{(L)} : i \in \mathcal{V}\}\!\!\})$.
>
> **Proof of Equivariance and Invariance:** Let $\pi$ be the bijection corresponding to permutation matrix $\boldsymbol{P}$ ($P_{ij} = \delta_{i,\pi(j)}$). Under relabeling, node $i$ moves to index $\pi(i)$ with neighborhood $\mathcal{N}'(\pi(i)) = \{\pi(j) : j \in \mathcal{N}(i)\}$. Because $\bigoplus$ is symmetric in its multiset arguments and $\psi_{\mathrm{msg}}^{(\ell)}, \psi_{\mathrm{upd}}^{(\ell)}$ share parameters across all nodes, the aggregated message at $\pi(i)$ is $\boldsymbol{m}_{\pi(i)}^{\prime(\ell+1)} = \boldsymbol{m}_i^{(\ell+1)}$, hence $\boldsymbol{h}_{\pi(i)}^{\prime(\ell+1)} = \boldsymbol{h}_i^{(\ell+1)}$, which in matrix form is $\boldsymbol{F}(\boldsymbol{P}\boldsymbol{H}^{(\ell)}, \boldsymbol{P}\boldsymbol{A}\boldsymbol{P}^{\top}) = \boldsymbol{P}\boldsymbol{F}(\boldsymbol{H}^{(\ell)}, \boldsymbol{A})$. By induction over $L$ layers, the node representations are permutation-equivariant; applying a symmetric $\mathrm{READOUT}$ yields $f(\boldsymbol{P}\boldsymbol{X}, \boldsymbol{P}\boldsymbol{A}\boldsymbol{P}^{\top}) = f(\boldsymbol{X}, \boldsymbol{A})$. $\square$

Two major spatial instantiations overcome the fixed isotropic edge weights of GCNs and the full-neighborhood dependency of spectral methods:
- **Graph Attention Networks (GAT)** [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, p. 421), [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13, p. 259): Instead of fixed degree weights $1/\sqrt{\tilde{d}_i\tilde{d}_j}$, GAT computes data-dependent anisotropic edge weights via masked self-attention over $j \in \mathcal{N}(i) \cup \{i\}$:
  
$$
e_{ij}^{(\ell)} = \mathrm{LeakyReLU}\!\left(\boldsymbol{a}^{\top} \left[\boldsymbol{W}\boldsymbol{h}_i^{(\ell)}  \|  \boldsymbol{W}\boldsymbol{h}_j^{(\ell)}\right]\right), \qquad \alpha_{ij}^{(\ell)} = \frac{\exp(e_{ij}^{(\ell)})}{\sum_{k \in \mathcal{N}(i)\cup\{i\}} \exp(e_{ik}^{(\ell)})},
$$

  updating $\boldsymbol{h}_i^{(\ell+1)} = \sigma\!\left(\sum_{j \in \mathcal{N}(i)\cup\{i\}} \alpha_{ij}^{(\ell)}\boldsymbol{W}\boldsymbol{h}_j^{(\ell)}\right)$ (or concatenating $K$ independent attention heads).
- **GraphSAGE**: For inductive mini-batch training on large graphs, GraphSAGE samples a fixed-size subset $\mathcal{N}_S(i) \subseteq \mathcal{N}(i)$ of cardinality $q$ at each layer and concatenates the self-embedding with the aggregated neighborhood ([[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 23, p. 764); [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13, p. 254)):
  
$$
\boldsymbol{h}_i^{(\ell+1)} = \sigma\!\left(\boldsymbol{W}_1^{(\ell)}\boldsymbol{h}_i^{(\ell)} + \boldsymbol{W}_2^{(\ell)} \mathrm{AGG}\!\left(\{\!\!\{\boldsymbol{h}_j^{(\ell)} : j \in \mathcal{N}_S(i)\}\!\!\}\right)\right),
$$

  followed by optional $\ell_2$-normalization $\boldsymbol{h}_i^{(\ell+1)} \leftarrow \boldsymbol{h}_i^{(\ell+1)} / \lVert \boldsymbol{h}_i^{(\ell+1)} \rVert_2$.

### Expressive Power: 1-Weisfeiler–Lehman (1-WL) Test & GIN Injectivity

 A fundamental question is which non-isomorphic graphs an MPNN can distinguish [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13, p. 262). The 1-Weisfeiler–Lehman (1-WL) color-refinement test [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, p. 416) iteratively updates discrete node colors via $c_i^{(\ell+1)} = \mathrm{HASH}\big(c_i^{(\ell)}, \{\!\!\{c_j^{(\ell)} : j \in \mathcal{N}(i)\}\!\!\}\big)$.

> [!important] 1-WL Expressivity Bound & Graph Isomorphism Network (GIN)
> **Theorem (Xu et al., 2019; Morris et al., 2019):** Any standard MPNN of the form `(eq:gnn:mpnn)` is at most as powerful as the 1-WL test in distinguishing non-isomorphic graphs: if 1-WL assigns identical color multisets to $G_1$ and $G_2$, then any MPNN produces identical graph representations $\boldsymbol{h}_{G_1} = \boldsymbol{h}_{G_2}$. An MPNN achieves exact 1-WL expressive power if and only if its aggregation, update, and graph-level readout maps are *injective* over countable multisets.
>
> By the Deep Sets universal representation theorem [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, p. 417), any injective multiset function over a countable universe can be written as $\mathrm{MLP}\!\left((1+\epsilon)\boldsymbol{h}_i^{(\ell)} + \sum_{j \in \mathcal{N}(i)} \boldsymbol{h}_j^{(\ell)}\right)$, yielding the Graph Isomorphism Network (GIN) update:
>
> $$
> \boldsymbol{h}_i^{(\ell+1)} = \mathrm{MLP}^{(\ell)}\!\left((1 + \epsilon^{(\ell)})\boldsymbol{h}_i^{(\ell)} + \sum_{j \in \mathcal{N}(i)} \boldsymbol{h}_j^{(\ell)}\right),
> $$
>
> where $\epsilon^{(\ell)}$ is a learnable scalar (or fixed to $0$). Crucially, *mean* and *max* aggregators are strictly less expressive than *sum* aggregation: mean aggregation fails to distinguish multisets with identical proportions but different cardinalities (e.g., $\{\!\!\{a, b\}\!\!\}$ vs. $\{\!\!\{a, a, b, b\}\!\!\}$), while max aggregation collapses multiplicities entirely ($\{\!\!\{a, b\}\!\!\}$ vs. $\{\!\!\{a, a, b\}\!\!\}$) [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, p. 416).

## Algorithmic Implementation & Comparative Synthesis

Full-batch GCNs require storing intermediate activations for all $N$ nodes across $L$ layers. Algorithm `alg:gnn:minibatch` details mini-batch neighbor-sampled MPNN training [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13, pp. 254–255), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 23, p. 764), constructing a bounded $L$-hop computation subgraph backward from batch $\mathcal{B}$ before forward message passing.

> [!example] Algorithm: Mini-Batch Neighbor-Sampled MPNN / GraphSAGE Forward Pass
> **Input:** Graph $G=(\mathcal{V},\mathcal{E})$, features $\boldsymbol{X} \in \mathbb{R}^{N\times D}$, target mini-batch $\mathcal{B} \subseteq \mathcal{V}$, depth $L$, fanouts $\{q_1,\dots,q_L\}$, weights $\{\boldsymbol{W}_1^{(\ell)}, \boldsymbol{W}_2^{(\ell)}\}_{\ell=0}^{L-1}$
> **Output:** Target node embeddings $\{\boldsymbol{h}_i^{(L)} : i \in \mathcal{B}\}$
> $\mathcal{B}^{(L)} \leftarrow \mathcal{B}$
> **for** $\ell = L, L-1, \dots, 1$:
>
>   $\mathcal{B}^{(\ell-1)} \leftarrow \mathcal{B}^{(\ell)}$
>   **for** $i \in \mathcal{B}^{(\ell)}$:
>
>     Sample $\mathcal{N}_S^{(\ell)}(i) \leftarrow \mathrm{Sample}(\mathcal{N}(i), q_\ell)$ without replacement
>     $\mathcal{B}^{(\ell-1)} \leftarrow \mathcal{B}^{(\ell-1)} \cup \mathcal{N}_S^{(\ell)}(i)$
>
> Initialize $\boldsymbol{h}_i^{(0)} \leftarrow \boldsymbol{x}_i$ for all $i \in \mathcal{B}^{(0)}$
> **for** $\ell = 0, 1, \dots, L-1$:
>
>   **for** $i \in \mathcal{B}^{(\ell+1)}$:
>
>     $\boldsymbol{m}_i^{(\ell+1)} \leftarrow \mathrm{AGG}\!\left(\{\!\!\{\boldsymbol{h}_j^{(\ell)} : j \in \mathcal{N}_S^{(\ell+1)}(i)\}\!\!\}\right)$  *// Sum, Mean, Max, or Attention*
>     $\boldsymbol{h}_i^{(\ell+1)} \leftarrow \sigma\!\left(\boldsymbol{W}_1^{(\ell)}\boldsymbol{h}_i^{(\ell)} + \boldsymbol{W}_2^{(\ell)}\boldsymbol{m}_i^{(\ell+1)}\right)$
>     $\boldsymbol{h}_i^{(\ell+1)} \leftarrow \boldsymbol{h}_i^{(\ell+1)} / \max(\lVert \boldsymbol{h}_i^{(\ell+1)} \rVert_2, \varepsilon)$
>
> **return** $\{\boldsymbol{h}_i^{(L)} : i \in \mathcal{B}\}$

Table `tab:gnn:complexity` summarizes the asymptotic complexity and structural trade-offs across canonical GNN families, assuming $L$ layers, hidden dimension $d$, average degree $\bar{d}$, and $K$-hop or $H$-head parameters.

**Table:** Complexity and expressive power comparison across graph neural network architectures ($N=|\mathcal{V}|$, $M=|\mathcal{E}|$, $B=|\mathcal{B}|$ batch size, $q$ sample fanout, $H$ attention heads).

| **Architecture** | **Time / Layer** | **Memory** | **Inductive?** | **Edge Weights** | **1-WL Expressivity** |
| --- | --- | --- | --- | --- | --- |
| Spectral CNN | $O(N^3 + N^2 d)$ | $O(N^2 + LNd)$ | No | Fixed eigenbasis | $\le$ 1-WL (fixed graph) |
| ChebNet ($K$-hop) | $O(K M d + N d^2)$ | $O(L N d + M)$ | Partial | Isotropic $T_k(\tilde{\boldsymbol{L}})$ | $\le$ 1-WL |
| GCN (Kipf–Welling) | $O(M d + N d^2)$ | $O(L N d + M)$ | Full-graph | Fixed $1/\sqrt{\tilde{d}_i\tilde{d}_j}$ | $<$ 1-WL (mean-like) |
| GraphSAGE ($q$-sample) | $O(B q^L d^2)$ | $O(B q^L d)$ | Yes | Uniform / learned | $<$ 1-WL (mean/max) |
| GAT ($H$ heads) | $O(H(M d + N d^2))$ | $O(L(Nd + HM))$ | Yes | Data-driven $\alpha_{ij}$ | $<$ 1-WL (normalized) |
| GIN (Sum + MLP) | $O(M d + N d^2)$ | $O(L N d + M)$ | Yes | Sum + $(1+\epsilon)\boldsymbol{I}$ | **Exact 1-WL** |

> [!info] Cross-Book Synthesis: Transductive vs. Inductive Regimes & Latent Topology
> Across the four synthesized texts, GNN training regimes and graph assumptions divide into three complementary axes:
> - **Transductive vs. Inductive Node Splits:** Bishop & Bishop [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, pp. 419–420) partition nodes into $\mathcal{V}_{\mathrm{train}}$ (labeled, in training loss and message passing), $\mathcal{V}_{\mathrm{trans}}$ (unlabeled, participate in message passing during training/inference), and $\mathcal{V}_{\mathrm{induct}}$ (unseen during training, added at inference). Prince [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13, pp. 254–256) shows how neighbor sampling and METIS-style graph partitioning convert massive transductive graphs into inductive mini-batches.
> - **Encoder–Decoder & Non-Euclidean Geometry:** Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 23, pp. 765–768) unifies GNNs under GraphEDM, extending message passing to unsupervised contrastive objectives (Deep Graph Infomax) and Hyperbolic GCNs (HGCN) that embed hierarchical trees in tangent spaces with lower distortion than Euclidean $\mathbb{R}^d$.
> - **Observed vs. Inferred Graph Topology:** Whereas Ch. 13 of Bishop and Prince assume $G$ is fixed and observed, Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Ch. 30, pp. 1043–1045) addresses settings where $\boldsymbol{A}$ is latent or unknown—inferring sparse undirected precision graphs via graphical lasso or directed causal DAGs from $\boldsymbol{X} \in \mathbb{R}^{N \times D}$ prior to downstream reasoning.

## Visual Diagnostics & Geometric Structure

Figure `fig:gnn:receptive_field` illustrates how stacking $L$ local message-passing layers expands each node's effective receptive field to its $L$-hop neighborhood, directly paralleling spatial receptive fields in CNNs.

> [!note] Source Figure
> *Figure:* Expansion of a node's receptive field across two GNN layers: the target node in hidden layer 2 aggregates from its 1-hop neighborhood in layer 1, which in turn aggregates from its 2-hop neighborhood in the input graph [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13, Fig. 13.9, p. 254). (`understandingdl_simonprince_p268_fig1.png`)

Figure `fig:gnn:gcn_vs_gat` contrasts isotropic adjacency weighting in GCNs, dense all-to-all Transformer self-attention, and topology-masked attention in GATs, while Figure `fig:gnn:graphsage` depicts GraphSAGE's three-stage inductive pipeline.

> [!note] Source Figure
> *Figure:* Structural comparison of (a) Graph Convolutional Networks with fixed adjacency weighting $\boldsymbol{A}+\boldsymbol{I}$, (b) dense dot-product self-attention, and (c) Graph Attention Networks masking attention scores by $\boldsymbol{A}+\boldsymbol{I}$ [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13, Fig. 13.12, p. 259). (`understandingdl_simonprince_p273_fig1.png`)

> [!note] Source Figure
> *Figure:* Inductive GraphSAGE pipeline: (1) uniform $k$-hop neighborhood sampling ($k=1,2$), (2) hierarchical feature aggregation, and (3) downstream context or label prediction [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 23, Fig. 23.5, p. 764). (`probabilisticml_kevinmurphy_introduction_p794_fig1.png`)

To visualize the algebraic difference between fixed-degree GCN aggregation and data-dependent GAT attention at the single-node level, Figure `fig:gnn:tikz_mpnn` diagrams the permutation-equivariant MPNN update pipeline.

> [!example] Visual Diagram (LaTeX / PDF)
> *Permutation-equivariant MPNN message-passing pipeline at node $i$. Symmetric aggregation $\bigoplus$ over $\mathcal{N}(i)$ ensures invariance to neighbor ordering, while weight sharing across nodes guarantees global equivariance.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Figure `fig:gnn:oversmoothing_plot` quantifies the over-smoothing phenomenon via the normalized Dirichlet energy ratio $E(\boldsymbol{H}^{(\ell)}) / E(\boldsymbol{H}^{(0)})$ across depth $L = 1 \dots 16$, contrasting exponential spectral collapse in a vanilla GCN against residual connections, PairNorm, and DropEdge.

> [!example] Visual Diagram (LaTeX / PDF)
> *Theoretical Dirichlet energy decay $E(\boldsymbol{H}^{(L)})/E(\boldsymbol{H}^{(0)})$ across GNN depth $L$ (log scale). Repeated low-pass filtering in Vanilla GCN contracts node features exponentially toward $\ker(\tilde{\boldsymbol{L}}_{\mathrm{sym}})$, whereas Residual connections, PairNorm, and DropEdge preserve feature discriminability.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Over-Smoothing: Exponential Collapse to $\ker(\tilde{\boldsymbol{L}}_{\mathrm{sym}})$
> The normalized adjacency $\hat{\boldsymbol{A}} = \tilde{\boldsymbol{D}}^{-1/2}\tilde{\boldsymbol{A}}\tilde{\boldsymbol{D}}^{-1/2} = \boldsymbol{I}_N - \tilde{\boldsymbol{L}}_{\mathrm{sym}}$ acts as a low-pass graph filter with eigenvalues $1 = \mu_1 > \mu_2 \ge \dots \ge \mu_N > -1$ on a connected non-bipartite graph. As depth $L \to \infty$, $\hat{\boldsymbol{A}}^L \to \boldsymbol{u}_1\boldsymbol{u}_1^{\top}$ where $\boldsymbol{u}_1 = \tilde{\boldsymbol{D}}^{1/2}\boldsymbol{1} / \lVert \tilde{\boldsymbol{D}}^{1/2}\boldsymbol{1} \rVert_2$ spans $\ker(\tilde{\boldsymbol{L}}_{\mathrm{sym}})$. Consequently, if the singular values of $\boldsymbol{W}^{(\ell)}$ do not counteract $\mu_2 < 1$, all node embeddings converge exponentially fast to a rank-1 subspace proportional to $\sqrt{\tilde{d}_i}$, destroying class separability [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, p. 422).\\
> **Remedies:** Add residual skip connections $\boldsymbol{h}_i^{(\ell+1)} = \mathrm{Update}(\cdot) + \boldsymbol{h}_i^{(\ell)}$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, p. 422), concatenate multi-scale layer outputs via Jumping Knowledge ($\boldsymbol{y}_i = f(\boldsymbol{h}_i^{(1)} \oplus \dots \oplus \boldsymbol{h}_i^{(L)})$) [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, p. 423), apply PairNorm to maintain constant pairwise feature distance, or mask random edges during training (DropEdge) [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, p. 423).

> [!warning] Over-Squashing at Tree-Like Bottlenecks
> In graphs with exponential neighborhood growth (e.g., trees or scale-free networks), the $L$-hop receptive field of node $i$ contains $O(d^L)$ nodes, yet all incoming information must be compressed into a fixed-dimensional vector $\boldsymbol{h}_i^{(L)} \in \mathbb{R}^d$. Across narrow bridges with small Cheeger constant or negative Ollivier–Ricci curvature, the Jacobian sensitivity $\lVert \partial \boldsymbol{h}_i^{(L)} / \partial \boldsymbol{x}_j \rVert$ decays exponentially with shortest-path distance $d_G(i,j)$, preventing long-range signal propagation even without over-smoothing.\\
> **Remedies:** Introduce a virtual super-node connected to all $N$ nodes [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 13, pp. 416–417), apply curvature-based graph rewiring, or hybridize with global self-attention (Graph Transformers).

> [!warning] 1-WL Expressivity Ceiling: $C_6$ vs. $2 \times C_3$
> Because every standard MPNN is bounded by 1-WL, it cannot distinguish any two $r$-regular graphs of the same size—most famously, a 6-cycle $C_6$ versus two disjoint triangles $2 \times C_3$ (both 2-regular on 6 nodes with uniform node features). In both graphs, every node sees an identical infinite computation tree of degree-2 neighbors, so any MPNN (including GIN) outputs identical node and graph embeddings and cannot count triangles or cycle lengths.\\
> **Remedies:** Augment node features with structural or positional encodings (Laplacian eigenvectors, random-walk landing probabilities) or adopt higher-order $k$-WL / subgraph GNNs.

> [!warning] Neighbor Explosion ($O(d^L)$ Memory) in Large-Graph Training
> Computing exact $L$-layer gradients for a mini-batch of $B$ nodes touches up to $O(B \bar{d}^L)$ nodes, which rapidly saturates GPU memory on dense or hub-heavy graphs (the graph expansion problem) [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13, p. 254).\\
> **Remedies:** Use fixed-fanout neighbor sampling (Algorithm `alg:gnn:minibatch`) with $q_\ell \ll \bar{d}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 23, p. 764), or partition the graph into dense clusters via METIS (Cluster-GCN) and sample clusters as mini-batches [[Source - Understanding Deep Learning|understandingdl_simonprince]] (Ch. 13, pp. 254–256).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (1):** [[Graph Neural Networks (MPNN)]]
- **Prerequisites (Directed $\leftarrow$):** None
- **Enables / Unlocks (Directed $\rightarrow$):** None
- **Related Concepts (Undirected $\leftrightarrow$):** [[Self-Attention & Multi-Head Attention]]
- **Synthesized Sources:** [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
