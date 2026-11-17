---
type: monograph
domain: "D01"
parent_topic: "[[PT 01 - Linear Algebra, Matrix Decompositions & Vector Calculus]]"
sub_topic: "[[ST 01 - Matrix Calculus, Jacobians & Automatic Differentiation]]"
cheatsheet: "[[Cheatsheet - Matrix Calculus, Computational Graphs & Automatic Differentiation]]"
part_of: 
  - "[[ST 01 - Matrix Calculus, Jacobians & Automatic Differentiation]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
concepts: 
  - "[[Automatic Differentiation (AD)]]"
  - "[[Matrix Calculus & Jacobians]]"
prerequisites: 
  - "[[Matrix Calculus & Jacobians]]"
  - "[[Vector Space & Subspaces]]"
enables: 
  - "[[Automatic Differentiation (AD)]]"
  - "[[MLP & Universal Approximation]]"
  - "[[Normalizing Flows]]"
  - "[[RNN, LSTM & GRU]]"
  - "[[Reparameterization Trick]]"
  - "[[Score Matching (DSM & SSM)]]"
related: 
  - "[[Automatic Differentiation (AD)]]"
  - "[[MLP & Universal Approximation]]"
  - "[[Matrix Calculus & Jacobians]]"
  - "[[Normalizing Flows]]"
  - "[[Score Matching (DSM & SSM)]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - monograph
  - domain-01
---

# Matrix Calculus, Computational Graphs & Automatic Differentiation

> [!abstract] Monograph Navigation
> - **Domain:** [[D01 - Mathematical Foundations for Machine Intelligence|Mathematical Foundations for Machine Intelligence]]
> - **Parent / Sub-Topic:** [[PT 01 - Linear Algebra, Matrix Decompositions & Vector Calculus|Linear Algebra, Matrix Decompositions & Vector Calculus]] → [[ST 01 - Matrix Calculus, Jacobians & Automatic Differentiation|Matrix Calculus, Jacobians & Automatic Differentiation]]
> - **Visual Cheatsheet:** [[Cheatsheet - Matrix Calculus, Computational Graphs & Automatic Differentiation]]
> - **Description:** Reconciles numerator vs. denominator layout conventions across books, derives Jacobian/Hessian chain rules, and contrasts forward-mode (JVP) vs. reverse-mode (VJP / backpropagation) AD.

## Problem Formulation & Unified Notation

Modern statistical learning and deep architectures optimize scalar objectives
$L : \Theta \to \mathbb{R}$ formed by deep compositions of vector-, matrix-, and
tensor-valued maps. Let $U \subseteq \mathbb{R}^n$ be open and let
$\boldsymbol{f} : U \to \mathbb{R}^m$ be differentiable at $\boldsymbol{x} \in U$. Coordinate-free
differential calculus defines the *Fr\'{e}chet derivative* (or total
differential) at $\boldsymbol{x}$ as the unique linear map
$D\boldsymbol{f}(\boldsymbol{x}) : \mathbb{R}^n \multimap \mathbb{R}^m$ satisfying

$$
\boldsymbol{f}(\boldsymbol{x} + \boldsymbol{h})
  = \boldsymbol{f}(\boldsymbol{x}) + D\boldsymbol{f}(\boldsymbol{x})[\boldsymbol{h}] + o(\lVert \boldsymbol{h} \rVert_2)
  \quad \text{as } \boldsymbol{h} \to \boldsymbol{0}.
$$

In functional AD notation
[[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.2.1, pp. 262–263),
this linear operator is written $\partial f(x) : \mathbb{R}^n \multimap \mathbb{R}^m$.
Fixing the standard bases $\{\boldsymbol{e}_j\}_{j=1}^n$ of $\mathbb{R}^n$ and
$\{\boldsymbol{e}_i\}_{i=1}^m$ of $\mathbb{R}^m$ yields the canonical matrix representation of
$D\boldsymbol{f}(\boldsymbol{x})$, the *Jacobian matrix*
$\boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x}) \in \mathbb{R}^{m \times n}$:

$$
\boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})
  \triangleq \frac{\partial \boldsymbol{f}(\boldsymbol{x})}{\partial \boldsymbol{x}^{\top}}
  = \begin{bmatrix}
      \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
      \vdots & \ddots & \vdots \\
      \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
    \end{bmatrix}
  = \begin{bmatrix}
      \nabla f_1(\boldsymbol{x})^{\top} \\
      \vdots \\
      \nabla f_m(\boldsymbol{x})^{\top}
    \end{bmatrix}
  \in \mathbb{R}^{m \times n}.
$$

For a scalar function $f : \mathbb{R}^n \to \mathbb{R}$ ($m=1$) or $f : \mathbb{R}^{m \times n} \to \mathbb{R}$,
optimization requires parameter updates $\boldsymbol{x}_{k+1} = \boldsymbol{x}_k - \eta \nabla_{\boldsymbol{x}} f(\boldsymbol{x}_k)$
and $\boldsymbol{X}_{k+1} = \boldsymbol{X}_k - \eta \nabla_{\boldsymbol{X}} f(\boldsymbol{X}_k)$ in the
same space as the primal variables. Accordingly, this monograph adopts a
**consistent shape-preserving hybrid convention**:
- **Jacobians follow numerator layout** ($\mathbb{R}^{m \times n}$, output
        rows $\times$ input columns), matching linear operator application
        $D\boldsymbol{f}(\boldsymbol{x})[\boldsymbol{v}] = \boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})\boldsymbol{v}$.
- **Scalar gradients preserve primal shape**:
        $\nabla_{\boldsymbol{x}} f(\boldsymbol{x}) = \boldsymbol{J}_f(\boldsymbol{x})^{\top} \in \mathbb{R}^n$ is a column
        vector and $(\nabla_{\boldsymbol{X}} f(\boldsymbol{X}))_{ij} = \frac{\partial f}{\partial X_{ij}}$ has the
        exact shape $\mathbb{R}^{m \times n}$ of $\boldsymbol{X}$.

> [!info] Cross-Book Notation & Pedagogical Reconciliation: Numerator vs. Denominator Layouts
> Matrix calculus texts split across three incompatible layout conventions for
> $\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}}$ and $\nabla_{\boldsymbol{x}} f$, causing factor-of-transpose
> bugs when formulas are mixed verbatim. Table `tab:matcalc:layouts`
> provides a unified Rosetta Stone across all six synthesized sources.
>
> 
>
>
> | **Source Text** | **Convention** | $\nabla_{\boldsymbol{x}} f$ | $\frac{\partial \boldsymbol{A}\boldsymbol{x}}{\partial \boldsymbol{x}}$ | $\boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})$ | **Chain Rule ($z = g(\boldsymbol{f}(\boldsymbol{x}))$)** |
> | — | — | — | — | — | — |
> | Deisenroth et al. | Pure Numerator | $\mathbb{R}^{1\times n}$ | $\boldsymbol{A}$ | $\mathbb{R}^{m\times n}$ | $\frac{\partial g}{\partial \boldsymbol{x}} = \frac{\partial g}{\partial \boldsymbol{f}}\frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}$ (no transposes) |
> | [2pt] Tiep Vu | Pure Denominator | $\mathbb{R}^{n\times 1}$ | $\boldsymbol{A}^{\top}$ | $\mathbb{R}^{n\times m}$ | $\nabla_{\boldsymbol{x}} g = (\nabla_{\boldsymbol{x}} \boldsymbol{f})(\nabla_{\boldsymbol{f}} g) = \boldsymbol{J}_{\boldsymbol{f}}^{\top} \nabla_{\boldsymbol{f}} g$ |
> | [2pt] Murphy (Intro) | Hybrid (Col / Num) | $\mathbb{R}^{n\times 1}$ | $\boldsymbol{A}$ | $\mathbb{R}^{m\times n}$ | $\nabla_{\boldsymbol{x}} g = \boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})^{\top} \nabla_{\boldsymbol{f}} g$ ($\nabla f = \boldsymbol{J}_f^{\top}$) |
> | [2pt] Goodfellow et al. | Hybrid (Col / Num) | $\mathbb{R}^{n\times 1}$ | $\boldsymbol{A}$ | $\mathbb{R}^{m\times n}$ | $\nabla_{\boldsymbol{x}} z = (\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}})^{\top} \nabla_{\boldsymbol{y}} z$ |
> | [2pt] Bishop & Bishop | Num. Jac / Adjoints | $\delta_j = \frac{\partial E_n}{\partial a_j}$ | $W_{ki}$ | $\mathbb{R}^{m\times n}$ | $\bar{v}_i = \sum_{j\in\mathrm{Ch}(i)} \bar{v}_j \frac{\partial v_j}{\partial v_i}$ |
> | [2pt] Murphy (Adv) | Functional Maps | $\partial f(x)^{\top}[1]$ | $v \mapsto \boldsymbol{A}v$ | $\mathbb{R}^n \multimap \mathbb{R}^m$ | $\partial(g \circ f)(x) = \partial g(f(x)) \circ \partial f(x)$ |
>
>
>
>
> 
> **Reconciliation Note on Denominator-Layout Chain Rules:** Under the denominator convention of [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 2, Eq. (2.8) & (2.11), pp. 32–33),
> $\nabla_{\boldsymbol{x}}\boldsymbol{h} \in \mathbb{R}^{k\times n}$ already stores the transpose of
> $\boldsymbol{J}_{\boldsymbol{h}} \in \mathbb{R}^{n\times k}$, yielding $\nabla_{\boldsymbol{x}}(\boldsymbol{A}\boldsymbol{x}) = \boldsymbol{A}^{\top}$.
> When differentiating $g(\boldsymbol{f}(\boldsymbol{x})) \in \mathbb{R}$, dimension matching requires
> $\nabla_{\boldsymbol{x}} g = (\nabla_{\boldsymbol{x}} \boldsymbol{f})(\nabla_{\boldsymbol{f}} g)$ (as applied in
> [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 2.4.4, p. 34) to obtain
> $2\boldsymbol{A}^{\top}(\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b})$); Eq. (2.10) therein prints an extra
> transpose $(\nabla_{\boldsymbol{X}} f)^{\top}$ that matches numerator-layout Jacobians
> (cf. [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Eq. (6.46), p. 207)).
> Always verifying outer dimensions $(n \times m)(m \times 1) = (n \times 1)$
> immediately catches layout-transpose mismatches.

## Core Mathematical Derivations & Theorems

### The Multivariate Chain Rule & Kronecker Flattening

> [!important] Theorem 1: Multivariate Chain Rule & Tensor-Matrix Isomorphism
> Let $\boldsymbol{f} : U \subseteq \mathbb{R}^n \to V \subseteq \mathbb{R}^p$ be differentiable at $\boldsymbol{x} \in U$ and $\boldsymbol{g} : V \to \mathbb{R}^m$ be differentiable at $\boldsymbol{y} = \boldsymbol{f}(\boldsymbol{x})$. Then the composition $\boldsymbol{h} = \boldsymbol{g} \circ \boldsymbol{f} : U \to \mathbb{R}^m$ is differentiable at $\boldsymbol{x}$ with Fr\'{e}chet derivative $D\boldsymbol{h}(\boldsymbol{x}) = D\boldsymbol{g}(\boldsymbol{y}) \circ D\boldsymbol{f}(\boldsymbol{x})$, i.e., in matrix Jacobian form [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Eq. (5.48), p. 148); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eq. (7.252), p. 272):
>
> $$
> \boldsymbol{J}_{\boldsymbol{g}\circ\boldsymbol{f}}(\boldsymbol{x})
>   = \boldsymbol{J}_{\boldsymbol{g}}(\boldsymbol{f}(\boldsymbol{x}))   \boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})
>   \in \mathbb{R}^{m \times n},
>   \qquad
>   \text{with entries }
>   \frac{\partial h_i}{\partial x_j} = \sum_{k=1}^{p} \frac{\partial g_i}{\partial y_k} \frac{\partial f_k}{\partial x_j}.
> $$
>
> **Proof Sketch.** Expanding $\boldsymbol{f}(\boldsymbol{x}+\boldsymbol{s}) = \boldsymbol{y} + \boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})\boldsymbol{s} + \boldsymbol{r}_1(\boldsymbol{s})$ with $\lVert \boldsymbol{r}_1(\boldsymbol{s}) \rVert = o(\lVert \boldsymbol{s} \rVert)$ and substituting $\boldsymbol{t} = \boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})\boldsymbol{s} + \boldsymbol{r}_1(\boldsymbol{s})$ into $\boldsymbol{g}(\boldsymbol{y}+\boldsymbol{t}) = \boldsymbol{g}(\boldsymbol{y}) + \boldsymbol{J}_{\boldsymbol{g}}(\boldsymbol{y})\boldsymbol{t} + \boldsymbol{r}_2(\boldsymbol{t})$ yields
>
> $$
> (\boldsymbol{g}\circ\boldsymbol{f})(\boldsymbol{x}+\boldsymbol{s}) - (\boldsymbol{g}\circ\boldsymbol{f})(\boldsymbol{x})
>   = \boldsymbol{J}_{\boldsymbol{g}}(\boldsymbol{y})\boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})\boldsymbol{s}
>   + \underbrace{\boldsymbol{J}_{\boldsymbol{g}}(\boldsymbol{y})\boldsymbol{r}_1(\boldsymbol{s}) + \boldsymbol{r}_2(\boldsymbol{t})}_{o(\lVert \boldsymbol{s} \rVert)},
> $$
>
> because $\lVert \boldsymbol{t} \rVert \le (\lVert \boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x}) \rVert_2 + o(1))\lVert \boldsymbol{s} \rVert$. Uniqueness of the Fr\'{e}chet linear approximation gives `(eq:matcalc:chain_rule)`. For scalar $g : \mathbb{R}^p \to \mathbb{R}$ ($m=1$), transposing `(eq:matcalc:chain_rule)` gives the gradient pull-back [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Eq. (6.46), p. 207):
>
> $$
> \nabla_{\boldsymbol{x}} (g \circ \boldsymbol{f})(\boldsymbol{x})
>   = \boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})^{\top} \nabla_{\boldsymbol{y}} g(\boldsymbol{y}) \in \mathbb{R}^n.
> $$
>
>
> **Matrix-Valued Maps via Vectorization:** For $\boldsymbol{F} : \mathbb{R}^{p \times q} \to \mathbb{R}^{m \times n}$, the raw derivative is a 4th-order tensor $\mathcal{J}_{ijkl} = \frac{\partial F_{ij}}{\partial X_{kl}}$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 5.4, p. 156). Using column-major vectorization $\mathrm{vec} : \mathbb{R}^{m \times n} \xrightarrow{\sim} \mathbb{R}^{mn}$ and the Kronecker identity $\mathrm{vec}(\boldsymbol{A}\boldsymbol{X}\boldsymbol{B}) = (\boldsymbol{B}^{\top} \otimes \boldsymbol{A})\mathrm{vec}(\boldsymbol{X})$, the linearized map $\boldsymbol{Y} = \boldsymbol{A}\boldsymbol{X}\boldsymbol{B}$ corresponds to the flat Jacobian $\boldsymbol{J}_{\mathrm{vec}(\boldsymbol{Y})}(\mathrm{vec}(\boldsymbol{X})) = \boldsymbol{B}^{\top} \otimes \boldsymbol{A} \in \mathbb{R}^{mn \times pq}$, allowing `(eq:matcalc:chain_rule)` to compose directly as matrix products.

### Differential Calculus & The Frobenius Trace Trick

Materializing $(mn) \times (pq)$ flattened Jacobians scales quadratically in matrix size and obscures algebraic structure. Working directly with first-order differentials $\mathrm{d}\boldsymbol{X}$ and the Frobenius inner product eliminates explicit tensor indices altogether.

> [!important] Theorem 2: Frobenius Identification (The Trace Trick) & Matrix Identities
> Equip $\mathbb{R}^{m \times n}$ with the Frobenius inner product $\langle \boldsymbol{A}, \boldsymbol{B} \rangle_F \triangleq \mathrm{Tr}(\boldsymbol{A}^{\top} \boldsymbol{B}) = \sum_{i,j} A_{ij} B_{ij}$. For any differentiable scalar function $f : \mathbb{R}^{m \times n} \to \mathbb{R}$, the total differential $\mathrm{d}f$ at $\boldsymbol{X}$ satisfies
>
> $$
> \mathrm{d}f(\boldsymbol{X})
>   = \sum_{i=1}^m \sum_{j=1}^n \frac{\partial f}{\partial X_{ij}} \mathrm{d}X_{ij}
>   = \mathrm{Tr}\!\big((\nabla_{\boldsymbol{X}} f(\boldsymbol{X}))^{\top}   \mathrm{d}\boldsymbol{X}\big)
>   = \langle \nabla_{\boldsymbol{X}} f(\boldsymbol{X}), \mathrm{d}\boldsymbol{X} \rangle_F.
> $$
>
> By the Riesz representation theorem on $(\mathbb{R}^{m\times n}, \langle \cdot, \cdot \rangle_F)$, if $\mathrm{d}f = \mathrm{Tr}(\boldsymbol{G}^{\top} \mathrm{d}\boldsymbol{X})$ holds for all perturbations $\mathrm{d}\boldsymbol{X}$ and some $\boldsymbol{G} \in \mathbb{R}^{m\times n}$ independent of $\mathrm{d}\boldsymbol{X}$, then uniquely $\nabla_{\boldsymbol{X}} f(\boldsymbol{X}) = \boldsymbol{G}$.
>
> **Proofs of Canonical Matrix Gradients:** Using `(eq:matcalc:trace_trick)`, we derive the unified identities across [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.8.7.3, p. 274), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 5.5, p. 158), and [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Table 2.1, p. 35):
> - **Trace Bilinear Form** $f(\boldsymbol{X}) = \mathrm{Tr}(\boldsymbol{A}\boldsymbol{X}\boldsymbol{B})$:
>         By linearity and cyclic invariance $\mathrm{Tr}(\boldsymbol{U}\boldsymbol{V}) = \mathrm{Tr}(\boldsymbol{V}\boldsymbol{U})$,
>
> $$
> \mathrm{d}f = \mathrm{Tr}(\boldsymbol{A} \mathrm{d}\boldsymbol{X} \boldsymbol{B})
>           = \mathrm{Tr}(\boldsymbol{B}\boldsymbol{A} \mathrm{d}\boldsymbol{X})
>           = \mathrm{Tr}\!\big((\boldsymbol{A}^{\top}\boldsymbol{B}^{\top})^{\top} \mathrm{d}\boldsymbol{X}\big)
>           \implies \nabla_{\boldsymbol{X}} \mathrm{Tr}(\boldsymbol{A}\boldsymbol{X}\boldsymbol{B}) = \boldsymbol{A}^{\top} \boldsymbol{B}^{\top}.
> $$
>
>         Specializing to rank-1 $\boldsymbol{A} = \boldsymbol{a}^{\top}$, $\boldsymbol{B} = \boldsymbol{b}$ gives $\nabla_{\boldsymbol{X}} (\boldsymbol{a}^{\top} \boldsymbol{X}\boldsymbol{b}) = \boldsymbol{a}\boldsymbol{b}^{\top}$.
> - **Matrix Quadratic Form** $f(\boldsymbol{X}) = \mathrm{Tr}(\boldsymbol{X}^{\top}\boldsymbol{A}\boldsymbol{X})$:
>         Using $\mathrm{d}(\boldsymbol{U}\boldsymbol{V}) = (\mathrm{d}\boldsymbol{U})\boldsymbol{V} + \boldsymbol{U}(\mathrm{d}\boldsymbol{V})$ and $\mathrm{Tr}(\boldsymbol{M}) = \mathrm{Tr}(\boldsymbol{M}^{\top})$,
>
> $$
> \begin{aligned}
> \mathrm{d}f &= \mathrm{Tr}((\mathrm{d}\boldsymbol{X}^{\top})\boldsymbol{A}\boldsymbol{X} + \boldsymbol{X}^{\top}\boldsymbol{A} \mathrm{d}\boldsymbol{X})
>           = \mathrm{Tr}(\boldsymbol{X}^{\top}\boldsymbol{A}^{\top} \mathrm{d}\boldsymbol{X}) + \mathrm{Tr}(\boldsymbol{X}^{\top}\boldsymbol{A} \mathrm{d}\boldsymbol{X})  \\
>           &= \mathrm{Tr}\!\big(((\boldsymbol{A}+\boldsymbol{A}^{\top})\boldsymbol{X})^{\top} \mathrm{d}\boldsymbol{X}\big)
>           \implies \nabla_{\boldsymbol{X}} \mathrm{Tr}(\boldsymbol{X}^{\top}\boldsymbol{A}\boldsymbol{X}) = (\boldsymbol{A} + \boldsymbol{A}^{\top})\boldsymbol{X}.
> \end{aligned}
> $$
> - **Matrix Multiplication VJP (`op.bprop`)** for $\boldsymbol{C} = \boldsymbol{A}\boldsymbol{B}$ inside scalar $L(\boldsymbol{C})$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 6.5.6, p. 216):
>         Let $\boldsymbol{G} = \nabla_{\boldsymbol{C}} L$. Using $\mathrm{d}\boldsymbol{C} = (\mathrm{d}\boldsymbol{A})\boldsymbol{B} + \boldsymbol{A}(\mathrm{d}\boldsymbol{B})$ inside `(eq:matcalc:trace_trick)`:
>
> $$
> \mathrm{d}L = \mathrm{Tr}(\boldsymbol{G}^{\top} \mathrm{d}\boldsymbol{C})
>           = \mathrm{Tr}(\boldsymbol{B}\boldsymbol{G}^{\top} \mathrm{d}\boldsymbol{A}) + \mathrm{Tr}(\boldsymbol{G}^{\top}\boldsymbol{A} \mathrm{d}\boldsymbol{B})
>           \implies
>           \nabla_{\boldsymbol{A}} L = \boldsymbol{G}\boldsymbol{B}^{\top},\quad
>           \nabla_{\boldsymbol{B}} L = \boldsymbol{A}^{\top}\boldsymbol{G}.
> $$
> - **Matrix Inverse & Log-Determinant** (for invertible $\boldsymbol{X} \in \mathbb{R}^{n\times n}$):
>         Differentiating $\boldsymbol{X}\boldsymbol{X}^{-1} = \boldsymbol{I}$ yields $(\mathrm{d}\boldsymbol{X})\boldsymbol{X}^{-1} + \boldsymbol{X} \mathrm{d}(\boldsymbol{X}^{-1}) = \boldsymbol{0} \implies \mathrm{d}(\boldsymbol{X}^{-1}) = -\boldsymbol{X}^{-1}(\mathrm{d}\boldsymbol{X})\boldsymbol{X}^{-1}$. Hence for $f(\boldsymbol{X}) = \mathrm{Tr}(\boldsymbol{X}^{-1} \boldsymbol{A})$,
>
> $$
> \begin{aligned}
>           \mathrm{d}f &= -\mathrm{Tr}(\boldsymbol{X}^{-1} (\mathrm{d}\boldsymbol{X}) \boldsymbol{X}^{-1} \boldsymbol{A})
>           = -\mathrm{Tr}(\boldsymbol{X}^{-1} \boldsymbol{A} \boldsymbol{X}^{-1} \mathrm{d}\boldsymbol{X}) \\
>           &\implies \nabla_{\boldsymbol{X}} \mathrm{Tr}(\boldsymbol{X}^{-1}\boldsymbol{A}) = -\boldsymbol{X}^{-\top}\boldsymbol{A}^{\top}\boldsymbol{X}^{-\top}.
>         \end{aligned}
> $$
>
>         For $\log\lvert \det(\boldsymbol{X}) \rvert$, Jacobi's formula gives $\mathrm{d}\det(\boldsymbol{X}) = \det(\boldsymbol{X})\mathrm{Tr}(\boldsymbol{X}^{-1} \mathrm{d}\boldsymbol{X})$, so
>
> $$
> \mathrm{d}\log\lvert \det(\boldsymbol{X}) \rvert = \mathrm{Tr}(\boldsymbol{X}^{-1} \mathrm{d}\boldsymbol{X})
>           \implies \nabla_{\boldsymbol{X}} \log\lvert \det(\boldsymbol{X}) \rvert = \boldsymbol{X}^{-\top}.
> $$

> [!tip] Why the Trace Trick Equals Coordinate-Free Adjoint Propagation
> The trace pairing $\mathrm{d}L = \langle \boldsymbol{G}, \mathrm{d}\boldsymbol{Y} \rangle_F = \langle \boldsymbol{G}, \mathcal{L}[\mathrm{d}\boldsymbol{X}] \rangle_F = \langle \mathcal{L}^*[\boldsymbol{G}], \mathrm{d}\boldsymbol{X} \rangle_F$ computes the Hilbert-space adjoint $\mathcal{L}^*$ of the linearized forward operator $\mathcal{L}(\cdot) = D\boldsymbol{Y}(\boldsymbol{X})[\cdot]$ directly. For $\boldsymbol{Y} = \boldsymbol{A}\boldsymbol{X}\boldsymbol{B}$, one never allocates the $(mn)\times(pq)$ Kronecker matrix $\boldsymbol{B}^{\top} \otimes \boldsymbol{A}$; cyclic permutation inside $\mathrm{Tr}(\cdot)$ immediately delivers the VJP $\mathcal{L}^*(\boldsymbol{G}) = \boldsymbol{A}^{\top}\boldsymbol{G}\boldsymbol{B}^{\top}$ in native matrix multiplications.

### Implicit Differentiation & Equilibrium Layers

Explicit layers compute $\boldsymbol{y} = \boldsymbol{f}(\boldsymbol{x})$, whereas *implicit layers*—including fixed-point networks, Neural ODEs, and differentiable convex optimization layers $\boldsymbol{y}^* \in \arg\min_{\boldsymbol{y}} f(\boldsymbol{x}, \boldsymbol{y})$ subject to $\boldsymbol{g}(\boldsymbol{x},\boldsymbol{y}) = \boldsymbol{0}$—define $\boldsymbol{y}^*(\boldsymbol{x})$ implicitly as the root of a stationarity residual $\boldsymbol{g}(\boldsymbol{x}, \boldsymbol{y}^*(\boldsymbol{x})) = \boldsymbol{0}$; see [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 16.2.10, pp. 640–641).

> [!important] Theorem 3: Implicit Function Theorem & Constant-Memory Adjoint VJP
> Let $\boldsymbol{g} : \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}^m$ be continuously differentiable ($C^1$) in a neighborhood of $(\boldsymbol{x}_0, \boldsymbol{y}_0)$ where $\boldsymbol{g}(\boldsymbol{x}_0, \boldsymbol{y}_0) = \boldsymbol{0}$. If the partial Jacobian $\boldsymbol{A} \triangleq \frac{\partial \boldsymbol{g}}{\partial \boldsymbol{y}^{\top}}(\boldsymbol{x}_0, \boldsymbol{y}_0) \in \mathbb{R}^{m\times m}$ is nonsingular, there exists a unique $C^1$ local solution map $\boldsymbol{y}^*(\boldsymbol{x})$ satisfying $\boldsymbol{g}(\boldsymbol{x}, \boldsymbol{y}^*(\boldsymbol{x})) = \boldsymbol{0}$. Differentiating $\boldsymbol{g}(\boldsymbol{x}, \boldsymbol{y}^*(\boldsymbol{x})) = \boldsymbol{0}$ via the chain rule yields
>
> $$
> \frac{\partial \boldsymbol{g}}{\partial \boldsymbol{x}^{\top}} + \frac{\partial \boldsymbol{g}}{\partial \boldsymbol{y}^{\top}} \boldsymbol{J}_{\boldsymbol{y}^*}(\boldsymbol{x}) = \boldsymbol{0}
>   \quad \Longrightarrow \quad
>   \boldsymbol{J}_{\boldsymbol{y}^*}(\boldsymbol{x})
>   = -\left(\frac{\partial \boldsymbol{g}}{\partial \boldsymbol{y}^{\top}}\right)^{-1} \frac{\partial \boldsymbol{g}}{\partial \boldsymbol{x}^{\top}} \in \mathbb{R}^{m \times n}.
> $$
>
> For a downstream scalar loss $L(\boldsymbol{y}^*(\boldsymbol{x}))$ with upstream gradient $\boldsymbol{u} = \nabla_{\boldsymbol{y}^*} L \in \mathbb{R}^m$, we compute the VJP without forming $\boldsymbol{J}_{\boldsymbol{y}^*}$ or inverting $\boldsymbol{A}$ explicitly by solving a single $m \times m$ adjoint linear system for $\boldsymbol{\lambda} \in \mathbb{R}^m$:
>
> $$
> \nabla_{\boldsymbol{x}} L
>   = \boldsymbol{J}_{\boldsymbol{y}^*}(\boldsymbol{x})^{\top} \boldsymbol{u}
>   = -\left(\frac{\partial \boldsymbol{g}}{\partial \boldsymbol{x}^{\top}}\right)^{\top} \boldsymbol{\lambda},
>   \qquad \text{where }
>   \left(\frac{\partial \boldsymbol{g}}{\partial \boldsymbol{y}^{\top}}\right)^{\top} \boldsymbol{\lambda} = \boldsymbol{u}.
> $$
>
> Notice that $\boldsymbol{\lambda}\mapsto \left(\frac{\partial \boldsymbol{g}}{\partial \boldsymbol{y}^{\top}}\right)^{\top}\boldsymbol{\lambda}$ and $\boldsymbol{\lambda} \mapsto \left(\frac{\partial \boldsymbol{g}}{\partial \boldsymbol{x}^{\top}}\right)^{\top} \boldsymbol{\lambda}$ are both single reverse-mode VJPs of $\boldsymbol{g}$ evaluated solely at $(\boldsymbol{x}, \boldsymbol{y}^*)$; thus Krylov solvers (GMRES/CG) or fixed-point iterations solve `(eq:matcalc:ift_vjp)` using $O(m+n)$ memory regardless of the number of forward solver steps $K$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 16.2.10, p. 641).

### Second-Order Derivatives & Pearlmutter's Hessian-Vector Product

For a twice continuously differentiable scalar map $f : \mathbb{R}^n \to \mathbb{R}$, the *Hessian matrix* $\nabla^2 f(\boldsymbol{x}) \in \mathbb{R}^{n\times n}$ is defined as the Jacobian of the gradient $\boldsymbol{J}_{\nabla f}(\boldsymbol{x})$ ([[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 5.7, p. 165); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.8.6, p. 272)), with entries $(\nabla^2 f)_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ that are symmetric by Schwarz's theorem. Because forming $\nabla^2 f$ at modern scale ($n \sim 10^6\text{–}10^{11}$) demands $O(n^2)$ memory and arithmetic ([[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 8.1.6, p. 243)), second-order Krylov solvers such as Conjugate Gradient, Lanczos, and Newton-CG ([[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 6.5.10, p. 224)) operate exclusively via matrix-free Hessian-vector products (HVPs) $\boldsymbol{H}\boldsymbol{v}$.

> [!important] Theorem 4: Exact $O(W)$ Hessian-Vector Products (Forward-over-Reverse)
> For any direction $\boldsymbol{v} \in \mathbb{R}^n$ independent of $\boldsymbol{x}$, the directional derivative of the gradient field equals the Hessian-vector product [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 8.2.2, p. 250); [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Eq. (6.59), p. 224):
>
> $$
> \nabla^2 f(\boldsymbol{x}) \boldsymbol{v}
>   = \boldsymbol{J}_{\nabla f}(\boldsymbol{x}) \boldsymbol{v}
>   = \left.\frac{\mathrm{d}}{\mathrm{d}\alpha} \nabla_{\boldsymbol{x}} f(\boldsymbol{x} + \alpha\boldsymbol{v})\right|_{\alpha=0}
>   = \nabla_{\boldsymbol{x}}\!\Big(\nabla_{\boldsymbol{x}} f(\boldsymbol{x})^{\top} \mathrm{stopgrad}(\boldsymbol{v})\Big).
> $$
> - **Forward-over-Reverse ($\mathcal{F} \circ \mathcal{R}$, Pearlmutter / $\mathcal{R}$-operator):** Construct the reverse-mode gradient function $\boldsymbol{g}(\boldsymbol{x}) = \nabla_{\boldsymbol{x}} f(\boldsymbol{x})$, then execute one forward-mode JVP pass $\partial \boldsymbol{g}(\boldsymbol{x})[\boldsymbol{v}]$ with tangent seed $\dot{\boldsymbol{x}} = \boldsymbol{v}$. This achieves exact $\boldsymbol{H}\boldsymbol{v}$ in $O(W)$ FLOPs ($2\text{–}3\times$ gradient cost) and $O(W)$ memory without nested graph allocation.
> - **Reverse-over-Forward ($\mathcal{R} \circ \mathcal{F}$):** Evaluate the directional derivative scalar $s(\boldsymbol{x}) = \partial f(\boldsymbol{x})[\boldsymbol{v}] = \nabla_{\boldsymbol{x}} f(\boldsymbol{x})^{\top} \boldsymbol{v}$ in one JVP pass, and then apply reverse mode to recover $\nabla_{\boldsymbol{x}} s(\boldsymbol{x}) = \boldsymbol{H}\boldsymbol{v}$ ([[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 8.2.2, p. 250)).
> - **Reverse-over-Reverse ($\mathcal{R} \circ \mathcal{R}$):** Differentiate the scalar dot product $\nabla_{\boldsymbol{x}} f(\boldsymbol{x})^{\top} \boldsymbol{v}$ via a second reverse tape; requires storing the first-order backward graph in memory unless symbol-to-symbol compilation is used [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 6.5.5 & 6.5.10, pp. 214, 224).

## Algorithmic Implementation & Comparative Synthesis

Automatic differentiation (AD) decomposes any differentiable program into a directed acyclic graph (DAG) $G = (V, E)$ of elementary primitives $f_t$, ordered topologically $t = 1, \dots, T$. Unlike numerical finite differences (truncated and unstable) and symbolic algebra (prone to exponential *expression swell* [[Source - Foundations|deep_learning_foundations_concepts]] (Eq. (8.42)–(8.48), pp. 245–246)), AD evaluates exact numerical derivatives up to machine precision by accumulating chain-rule contractions along $G$.

### Forward-Mode JVP vs. Reverse-Mode VJP Accumulation

Consider a chain $\boldsymbol{f} = \boldsymbol{f}_T \circ \cdots \circ \boldsymbol{f}_1$ with intermediate states $\boldsymbol{x}_t = \boldsymbol{f}_t(\boldsymbol{x}_{t-1})$. Exploiting associativity of matrix multiplication (as formulated across [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Eqs. (5.120)–(5.121), p. 161), [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Eq. (6.58), p. 223), and [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Eqs. (13.26)–(13.27), p. 439)), the chain product can be parenthesized from the right or the left:

$$
\begin{aligned}
\text{Forward-Mode JVP:}\quad
  \boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})\boldsymbol{v}
  &= \boldsymbol{J}_T \Big(\cdots \big(\boldsymbol{J}_2 (\boldsymbol{J}_1 \boldsymbol{v})\big)\cdots\Big),
   \\
  \text{Reverse-Mode VJP:}\quad
  \boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})^{\top} \boldsymbol{u}
  &= \boldsymbol{J}_1^{\top} \Big(\boldsymbol{J}_2^{\top} \big(\cdots (\boldsymbol{J}_T^{\top} \boldsymbol{u})\cdots\big)\Big).
\end{aligned}
$$

In **forward-mode AD**, every primal intermediate $\boldsymbol{x}_t$ is paired with a tangent variable $\dot{\boldsymbol{x}}_t = \sum_{q \in \mathrm{Pa}(t)} \partial_{q} \boldsymbol{f}_t(\boldsymbol{x}_{\mathrm{Pa}(t)})[\dot{\boldsymbol{x}}_q]$ (realized via dual numbers $x + \dot{x}\varepsilon$ with $\varepsilon^2 = 0$), propagating left-to-right in a single pass with zero activation retention [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 8.2.1, pp. 246–249); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Alg. 6.3, p. 269). Setting $\dot{\boldsymbol{x}} = \boldsymbol{e}_j$ computes column $j$ of $\boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})$, so materializing the full Jacobian costs $n_{\text{in}}$ forward passes.

In **reverse-mode AD**, a forward sweep evaluates and caches primal intermediates on an execution *tape* (unless checkpointed), and a backward sweep traverses $t = T, \dots, 1$ accumulating *adjoint* variables $\bar{\boldsymbol{x}}_q \triangleq (\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}_q^{\top}})^{\top} \boldsymbol{u}$ from children $c \in \mathrm{Ch}(q)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Alg. 6.4, p. 270):

$$
\bar{\boldsymbol{x}}_q = \sum_{c \in \mathrm{Ch}(q)} \big(\partial_q \boldsymbol{f}_c(\boldsymbol{x}_{\mathrm{Pa}(c)})\big)^{\top} [\bar{\boldsymbol{x}}_c].
$$

Fan-out (shared variables $x \mapsto \mathrm{dup}(x) = (x, x)$) has forward Jacobian $[\boldsymbol{I}; \boldsymbol{I}]$ whose transpose $[\boldsymbol{I}  \boldsymbol{I}]$ enforces summation across outgoing paths in `(eq:matcalc:adjoint_fanout)` [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. (6.36)–(6.40), p. 269). Setting $\boldsymbol{u} = \boldsymbol{e}_i$ yields row $i$ of $\boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})$ in one sweep, so computing $\nabla_{\boldsymbol{x}} L$ for scalar losses ($n_{\text{out}} = 1$) requires only $O(1)$ sweeps regardless of $n_{\text{in}}$. By the Baur–Strassen theorem, a single reverse sweep costs at most $c \le 5\text{–}6\times$ (typically $2\text{–}3\times$) the arithmetic operations of one forward evaluation [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 8.2.2, p. 250).

> [!example] Algorithm: Topological Reverse-Mode Tape AD (VJP with Fan-Out Accumulation)
> **Input:** DAG $G = (V, E)$ in topological order $1, \dots, T$; inputs $\boldsymbol{x} = (\boldsymbol{x}_1, \dots, \boldsymbol{x}_{n_{\text{in}}})$; output seed $\boldsymbol{u} \in \mathbb{R}^{n_{\text{out}}}$; optional checkpoint set $\mathcal{C} \subseteq V$.
> **Output:** Primal evaluation $\boldsymbol{y} = \boldsymbol{x}_T$ and parameter VJP $\{\bar{\boldsymbol{x}}_i = (\boldsymbol{J}_{\boldsymbol{f},  i}(\boldsymbol{x}))^{\top} \boldsymbol{u}\}_{i=1}^{n_{\text{in}}}$.
>
>  *// Phase 1: Forward primal pass and tape construction*
> Initialize $\mathcal{T} \leftarrow \emptyset$ (activation tape), adjoints $\bar{\boldsymbol{x}}_t \leftarrow \boldsymbol{0}$ for $t = 1, \dots, T$
> **for** $t \leftarrow n_{\text{in}} + 1$ to $T$:
>
>   Retrieve parent tuple $\boldsymbol{p}_t \leftarrow (\boldsymbol{x}_q : q \in \mathrm{Pa}(t))$
>   Compute primal $\boldsymbol{x}_t \leftarrow f_t(\boldsymbol{p}_t)$
>   **if** checkpointing enabled **and** $t \notin \mathcal{C}$:
>
>     Discard intermediate $\boldsymbol{x}_t$ from $\mathcal{T}$ (retain only inputs needed for recomputation)
>   **else**:
>
>     Push $(t, \boldsymbol{x}_t, \mathrm{Pa}(t))$ onto tape $\mathcal{T}$
>
>
>  *// Phase 2: Reverse topological adjoint sweep (VJP)*
> Set output adjoint $\bar{\boldsymbol{x}}_T \leftarrow \boldsymbol{u}$
> **for** $t \leftarrow T$ to $n_{\text{in}} + 1$:
>
>   **if** $\boldsymbol{x}_{\mathrm{Pa}(t)}$ was discarded from $\mathcal{T}$:
>
>     Recompute local segment from nearest prior checkpoint in $\mathcal{C}$
>
>   For each parent $q_k \in \mathrm{Pa}(t)$ ($k = 1, \dots, \lvert \mathrm{Pa}(t) \rvert$), pull back local VJP
>    $\bar{\boldsymbol{x}}_{q_k} \leftarrow \bar{\boldsymbol{x}}_{q_k} + \big(\partial_k f_t(\boldsymbol{x}_{\mathrm{Pa}(t)})\big)^{\top} [\bar{\boldsymbol{x}}_t]$  *// Accumulate over fan-out edges*
>    Free cached activation $\boldsymbol{x}_t$ once all parents of $t$ are processed
>
> **return** $\boldsymbol{x}_T$ and $\{\bar{\boldsymbol{x}}_1, \dots, \bar{\boldsymbol{x}}_{n_{\text{in}}}\}$

### Comparative Complexity & Paradigm Synthesis

Table `tab:matcalc:complexity` summarizes asymptotic FLOPs and memory requirements across all differentiation strategies for a program $\boldsymbol{f} : \mathbb{R}^{n_{\text{in}}} \to \mathbb{R}^{n_{\text{out}}}$ with $T$ primitive operations of average cost $c_{\text{op}}$.

**Table:** Asymptotic time, memory, and accuracy comparison across differentiation paradigms for $\boldsymbol{f} : \mathbb{R}^{n_{\text{in}}} \to \mathbb{R}^{n_{\text{out}}}$ ($T$ primitives, implicit state dimension $m$) [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 8.1.4, 8.2, pp. 239–250); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.2.2, 16.2.10, pp. 267–270, 641).

| **Method** | **Precision** | **Full $\boldsymbol{J}$ FLOPs** | **Scalar ($n_{\text{out}}\!=\!1$)** | **Peak Mem.** | **Primary Strength / Regime** |
| — | — | — | — | — | — |
| Forward Finite Diff. | $O(\epsilon)$ | $O(n_{\text{in}} T)$ | $O(n_{\text{in}} T)$ | $O(\text{width})$ | Black-box quick check only |
| Central Finite Diff. | $O(\epsilon^2)$ | $O(2 n_{\text{in}} T)$ | $O(2 n_{\text{in}} T)$ | $O(\text{width})$ | Unit-testing AD implementations |
| Symbolic Algebra | Exact | Exp. swell | Exp. swell | Exp. swell | Small closed-form proofs |
| Forward AD (JVP) | Exact | $O(n_{\text{in}} T)$ | $O(n_{\text{in}} T)$ | $O(\text{width})$ | $n_{\text{out}} \gg n_{\text{in}}$; directional/HVP |
| Reverse AD (VJP) | Exact | $O(n_{\text{out}} T)$ | $O(T)$ | $O(T)$ | $n_{\text{in}} \gg n_{\text{out}}$ (deep learning) |
| Checkpointed VJP | Exact | $O(n_{\text{out}} T \log T)$ | $O(T \log T)$ | $O(\log T)$ or $O(\sqrt{T})$ | Memory-bound deep/long graphs |
| Implicit Diff. (IFT) | Exact | $O(n_{\text{out}} m^2)$ | $O(m^2)$ | $O(m+n_{\text{in}})$ | Deep equilibrium / optimization layers |

> [!info] Cross-Book Synthesis: Static Symbol-to-Symbol Graphs vs. Dynamic Tracing
> Across software realizations, the six sources emphasize complementary architectural trade-offs:
> - **Symbol-to-Number vs. Symbol-to-Symbol:** Early libraries (Torch/Caffe) executed imperative numeric backward passes (symbol-to-number), whereas Theano and TensorFlow 1 lowered expressions into explicit static derivative subgraphs (symbol-to-symbol), enabling algebraic fusion and recursive higher-order differentiation ([[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 6.5.5, p. 214)).
> - **Dynamic Tape Tracing vs. Functional Transforms:** PyTorch eager mode constructs a dynamic tape just-in-time on each forward run, supporting arbitrary Python control flow (`if`/`while`; see [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 13.3.4, p. 446)). Conversely, JAX traces pure side-effect-free primitives (`jaxpr`) into composable `jvp`/`vjp` transforms that eliminate fan-out redundancy via `dup` and XLA fusion ([[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.2.2.3, pp. 270–271)).
> - **NP-Hard Optimal Accumulation:** Finding the minimum-FLOP elimination order of intermediate Jacobians (vertex, edge, or face elimination on a general DAG) is NP-complete (Naumann, 2008; discussed in [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 6.5.9, p. 222)). Reverse mode and forward mode sit at the two extreme linear-schedule poles, whereas mixed-mode Jacobian accumulation can outperform both on bottlenecked graphs.

## Visual Diagnostics & Geometric Structure

### Evaluation Traces & Computational Graph Topology

Figures `fig:matcalc:bishop_trace` and `fig:matcalc:murphy_mlp` illustrate two complementary computational graph abstractions from the literature: fine-grained scalar evaluation traces showing primal node dependencies and fan-out, and coarse-grained tensor DAGs coupling parameter weights simultaneously to a predictive loss and a weight-decay regularizer.

> [!note] Source Figure
> *Figure:* Primal evaluation trace DAG for $f(x_1, x_2) = x_1 x_2 + \exp(x_1 x_2) - \sin(x_2)$, decomposing the expression into elementary operations $v_1, \dots, v_7$ so shared subexpressions ($v_3 = v_1 v_2$) are computed once and fan-out edges accumulate during reverse mode [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 8, Fig. 8.4, p. 247). (`deep_learning_foundations_concepts_p264_fig1.png`)

> [!note] Source Figure
> *Figure:* Tensor-level computational graph of a single-hidden-layer MLP with weight decay: pre-activation $\boldsymbol{z} = \boldsymbol{W}^{(1)}\boldsymbol{x}$, activation $\boldsymbol{h} = \phi(\boldsymbol{z})$, output $\boldsymbol{o} = \boldsymbol{W}^{(2)}\boldsymbol{h}$, loss $L = \ell(\boldsymbol{o}, y)$, regularizer $s = \frac{\lambda}{2}(\lVert \boldsymbol{W}^{(1)} \rVert_F^2 + \lVert \boldsymbol{W}^{(2)} \rVert_F^2)$, and total objective $J = L + s$. Fan-out at $\boldsymbol{W}^{(1)}$ and $\boldsymbol{W}^{(2)}$ sums gradients from the data-loss and regularization branches [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 13.3.4, Fig. 13.13, p. 446). (`probabilisticml_kevinmurphy_introduction_p476_fig2.png`)

### Forward-Mode JVP vs. Reverse-Mode VJP Signal Flow

Figure `fig:matcalc:fwd_rev_dag` contrasts forward-mode tangent propagation and reverse-mode adjoint propagation side by side on the branching DAG of $f(x_1, x_2) = x_1 x_2 + \exp(x_1 x_2) - \sin(x_2)$, following the evaluation trace of [[Source - Foundations|deep_learning_foundations_concepts]] (Eqs. (8.49)–(8.76), pp. 247–250) to show how fan-out at $v_3$ splits forward tangents and sums backward adjoints.

> [!example] Visual Diagram (LaTeX / PDF)
> *Forward-mode JVP vs. reverse-mode VJP execution on the branching DAG of $f(x_1, x_2) = x_1 x_2 + \exp(x_1 x_2) - \sin(x_2)$. Fan-out edges exiting $v_3$ duplicate tangent perturbations forward and sum adjoint sensitivities backward [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Eqs. (6.39)–(6.40), p. 269); [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 8.2, pp. 247–250).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

### Asymptotic Compute & Activation-Memory Scaling Laws

Figure `fig:matcalc:complexity_scaling` plots the exact arithmetic and memory crossover regimes governing forward-mode AD, reverse-mode AD, finite differences, and checkpointing. In Panel (a), reverse-mode VJP dominates whenever $n_{\text{in}} \gg n_{\text{out}}$, achieving $O(1)\times$ primal cost for scalar losses ($n_{\text{out}} = 1$), whereas forward-mode JVP dominates when $n_{\text{out}} > n_{\text{in}}$. In Panel (b), uniform $\sqrt{L}$-segment and binomial Revolve checkpointing break the linear $O(L)$ activation-memory bottleneck of standard reverse-mode backpropagation.

> [!example] Visual Diagram (LaTeX / PDF)
> *Asymptotic compute and memory scaling across AD modes ($c = 2.5$ tape overhead factor). (a) Full Jacobian cost scaling with $n_{\text{in}}$: reverse mode runs in $O(n_{\text{out}})$ sweeps regardless of $n_{\text{in}}$, overtaking forward mode at $n_{\text{in}} > n_{\text{out}}$. (b) Peak activation storage vs. sequential depth $L$: $\sqrt{L}$-checkpointing and binomial Revolve trade one or $O(\log L)$ recomputations for sublinear memory [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 13.3.1, pp. 439–440); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.2.2.1, pp. 267–268).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Pitfall 1: Tape Memory Explosion vs. Recomputation (Checkpointing)
> **Failure Mode:** Reverse-mode VJP must retain primal intermediates $\boldsymbol{x}_t$ needed by local Jacobian adjoints $(\partial \boldsymbol{f}_t/\partial \boldsymbol{x}_q)^{\top}$ from the forward pass until the backward sweep reaches step $t$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 6.5.3 & 6.5.7, pp. 208, 221); [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.2.2.1, pp. 267–268). For an $L$-layer transformer or $T$-step RNN with batch size $B$ and width $d$, activation memory scales as $\Theta(L B d)$, rapidly exceeding accelerator HBM even when weights fit comfortably.\\
> **Remedy:** Partition the sequence into $k$ segments storing only $k$ boundary checkpoints and recomputing intra-segment activations during backprop, yielding peak memory $O(k + L/k)$. Choosing $k = \sqrt{L}$ minimizes memory to $O(\sqrt{L})$ at the cost of only $1$ extra forward pass ($+33\%$ total compute). Optimal Griewank–Walther (Revolve) binomial checkpointing attains $O(\log L)$ memory in $O(L \log L)$ FLOPs; reversible layers (RevNets) and implicit layers (Theorem 3) achieve $O(1)$ depth memory [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_advancedtopics]] (Sec. 6.2.2.1, 16.2.10, pp. 268, 641).

> [!warning] Pitfall 2: Non-Smooth Activations, Subdifferentials & Broken Chain Rules
> **Failure Mode:** Deep networks routinely apply piecewise-linear primitives such as $\mathrm{ReLU}(a) = \max(0, a)$ or $\lvert x \rvert$, which fail to be classically differentiable at $a = 0$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 8.1.4.1, pp. 282–283; Sec. 13.3.3.2, p. 443). At $a = 0$, the convex subdifferential of $\lvert x \rvert$ is $[-1, 1]$ and of $\mathrm{ReLU}(a)$ is $[0, 1]$; AD frameworks pick a deterministic branch convention (typically $\mathrm{ReLU}'(0) = 0$, i.e., Heaviside $H(a)$). Because the standard chain rule fails on Clarke subdifferentials at kinks, AD can return values outside the true subdifferential: for $f(x) = \mathrm{ReLU}(x) - \mathrm{ReLU}(-x) = x$, setting $\mathrm{ReLU}'(0)=0$ causes AD to output $0$ at $x=0$ instead of $1$! Moreover, negative pre-activations induce zero-gradient blocking ("dead ReLUs"), and discrete operators ($\arg\max$, hard thresholding) possess identically zero gradients a.e. [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 6.5.8, p. 221).\\
> **Remedy:** Substitute $C^\infty$ smooth surrogates ($\mathrm{softplus}(a) = \log(1 + e^a)$, $\mathrm{GELU}$, $\mathrm{SiLU}$) when computing Hessians, implicit derivatives, or sensitivity checks, and use straight-through or Gumbel-Softmax relaxations for discrete operations.

> [!warning] Pitfall 3: Finite-Difference Truncation vs. Floating-Point Cancellation
> **Failure Mode:** Numerical gradient checks verify AD implementations against finite differences ([[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Sec. 5.2.1, p. 149); [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Sec. 2.6, pp. 36–39)). Shrinking the step size $\epsilon$ reduces Taylor truncation error, but below a precision-dependent threshold, subtracting nearly equal values $f(\boldsymbol{x}+\epsilon\boldsymbol{e}_i) - f(\boldsymbol{x}-\epsilon\boldsymbol{e}_i)$ amplifies machine-epsilon round-off $\varepsilon_{\text{mach}}$ by $O(\varepsilon_{\text{mach}}/\epsilon)$ ([[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 7.8.1, p. 270); [[Source - Foundations|deep_learning_foundations_concepts]] (Sec. 8.1.4, Fig. 8.2, pp. 239–240)):
>
> $$
> \begin{aligned}
>   E_{\text{fwd}}(\epsilon) &\lesssim \frac{\lvert f" \rvert}{2}\epsilon + \frac{2\lvert f \rvert\varepsilon_{\text{mach}}}{\epsilon}
>    \implies  \epsilon^*_{\text{fwd}} \sim \varepsilon_{\text{mach}}^{1/2}, \\
>   E_{\text{cen}}(\epsilon) &\lesssim \frac{\lvert f"' \rvert}{6}\epsilon^2 + \frac{\lvert f \rvert\varepsilon_{\text{mach}}}{\epsilon}
>    \implies  \epsilon^*_{\text{cen}} \sim \varepsilon_{\text{mach}}^{1/3}.
> \end{aligned}
> $$
>
> In IEEE 754 `float64` ($\varepsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$), $\epsilon^*_{\text{cen}} \approx 6 \times 10^{-6}$; running central differences in `float32` ($\varepsilon_{\text{mach}} \approx 1.19 \times 10^{-7}$) with $\epsilon = 10^{-6}$ falls deep inside the cancellation regime ($\epsilon^*_{\text{cen}} \approx 5 \times 10^{-3}$), producing $O(1)$ relative errors even for bug-free AD.\\
> **Remedy:** Always execute gradient checks using **central differences** in `float64` with $\epsilon \in [10^{-6}, 10^{-5}]$ and relative discrepancy tolerance $\lVert \boldsymbol{g}_{\text{AD}} - \boldsymbol{g}_{\text{FD}} \rVert_2 / \max(\lVert \boldsymbol{g}_{\text{AD}} \rVert_2, \lVert \boldsymbol{g}_{\text{FD}} \rVert_2, 10^{-8}) < 10^{-6}$, or use complex-step differentiation $\mathrm{Im}(f(\boldsymbol{x} + \mathrm{i}\epsilon\boldsymbol{e}_i))/\epsilon$ which avoids subtractive cancellation down to $\epsilon = 10^{-15}$.

> [!warning] Pitfall 4: Higher-Order Graph Leaks, In-Place Tape Corruption & Spectral Blow-Up
> **Failure Mode:** Three subtle implementation bugs corrupt automatic derivatives in deep networks:
> - **Differentiating through HVP tangent vectors:** Evaluating $\nabla^2 f(\boldsymbol{x})\boldsymbol{v} = \nabla_{\boldsymbol{x}}((\nabla_{\boldsymbol{x}} f(\boldsymbol{x}))^{\top} \boldsymbol{v})$ when $\boldsymbol{v} = \boldsymbol{v}(\boldsymbol{x})$ depends on $\boldsymbol{x}$ (e.g., $\boldsymbol{v} = \nabla f(\boldsymbol{x})$) without detaching $\boldsymbol{v}$ erroneously adds $\boldsymbol{J}_{\boldsymbol{v}}(\boldsymbol{x})^{\top} \nabla f(\boldsymbol{x})$ [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 6.5.10, p. 224). Always wrap $\boldsymbol{v}$ in `stop\_gradient` or use forward-mode tangent seeds.
> - **In-place buffer mutation:** Overwriting a cached activation $\boldsymbol{x}_t$ in place on the forward pass silently invalidates local Jacobian rules $(\partial f_c/\partial \boldsymbol{x}_t)$ during the reverse tape sweep.
> - **Jacobian spectral instability:** Across $L$ layers with layer Jacobians $\boldsymbol{J}_l$, backward adjoints evolve via $\boldsymbol{g}_1 = (\prod_{l=1}^L \boldsymbol{J}_l)^{\top} \boldsymbol{g}_{L+1}$; if the spectral radius $\rho(\boldsymbol{J}_l) < 1$ or $> 1$, gradients vanish or explode exponentially with $L$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Sec. 13.4.2, p. 447), requiring residual connections ($\boldsymbol{J}_l = \boldsymbol{I} + \boldsymbol{J}_{\text{res}}$), normalization, and gradient norm clipping $\boldsymbol{g}' = \min(1, c/\lVert \boldsymbol{g} \rVert_2)\boldsymbol{g}$.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[Automatic Differentiation (AD)]], [[Matrix Calculus & Jacobians]]
- **Prerequisites (Directed $\leftarrow$):** [[Matrix Calculus & Jacobians]], [[Vector Space & Subspaces]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Automatic Differentiation (AD)]], [[MLP & Universal Approximation]], [[Normalizing Flows]], [[RNN, LSTM & GRU]], [[Reparameterization Trick]], [[Score Matching (DSM & SSM)]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Automatic Differentiation (AD)]], [[MLP & Universal Approximation]], [[Matrix Calculus & Jacobians]], [[Normalizing Flows]], [[Score Matching (DSM & SSM)]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
