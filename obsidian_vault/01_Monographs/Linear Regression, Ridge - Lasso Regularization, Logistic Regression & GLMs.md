---
type: monograph
domain: "D02"
parent_topic: "[[PT 02 - Linear Models, Generalization, Kernels & Ensembles]]"
sub_topic: "[[ST 02 - Linear, Regularized & Generalized Linear Models]]"
cheatsheet: "[[Cheatsheet - Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]"
part_of: 
  - "[[ST 02 - Linear, Regularized & Generalized Linear Models]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
concepts: 
  - "[[Generalized Linear Models (GLM)]]"
  - "[[Linear & Ridge Regression]]"
  - "[[Logistic Regression]]"
prerequisites: 
  - "[[Exponential Families]]"
  - "[[Maximum Likelihood & Fisher Info]]"
  - "[[Orthogonal Projection]]"
  - "[[Singular Value Decomposition (SVD)]]"
enables: []
related: 
  - "[[Bias-Variance Trade-off]]"
  - "[[Double Descent]]"
  - "[[Exponential Families]]"
  - "[[Generalized Linear Models (GLM)]]"
  - "[[Logistic Regression]]"
  - "[[Maximum Likelihood & Fisher Info]]"
  - "[[Moore-Penrose Pseudoinverse]]"
  - "[[Orthogonal Projection]]"
  - "[[Proximal & Subgradient Methods]]"
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Speech and Language Processing]]"
tags:
  - monograph
  - domain-02
---

# Linear Regression, Ridge/Lasso Regularization, Logistic Regression & GLMs

> [!abstract] Monograph Navigation
> - **Domain:** [[D02 - Classical, Kernel & Probabilistic Machine Learning|Classical, Kernel & Probabilistic Machine Learning]]
> - **Parent / Sub-Topic:** [[PT 02 - Linear Models, Generalization, Kernels & Ensembles|Linear Models, Generalization, Kernels & Ensembles]] → [[ST 02 - Linear, Regularized & Generalized Linear Models|Linear, Regularized & Generalized Linear Models]]
> - **Visual Cheatsheet:** [[Cheatsheet - Linear Regression, Ridge - Lasso Regularization, Logistic Regression & GLMs]]
> - **Description:** Unifies OLS orthogonal projection, Ridge L2 Tikhonov duality, Lasso L1 sparsity geometry, binary/multinomial logistic cross-entropy gradients, and Exponential-Family canonical link Generalized Linear Models.

## Problem Formulation & Unified Notation

Given a supervised dataset $\mathcal{D} = \{(\boldsymbol{x}_n, y_n)\}_{n=1}^N$ with inputs $\boldsymbol{x}_n \in \mathbb{R}^d$ and targets $y_n \in \mathcal{Y}$, a linear or generalized linear model (GLM) maps each input through a fixed feature map $\boldsymbol{\phi}: \mathbb{R}^d \to \mathbb{R}^D$ (often with $\phi_0(\boldsymbol{x}) = 1$ absorbing the bias) to form the design matrix $\boldsymbol{\Phi} = [\boldsymbol{\phi}(\boldsymbol{x}_1), \dots, \boldsymbol{\phi}(\boldsymbol{x}_N)]^{\top} \in \mathbb{R}^{N \times D}$ and linear predictor

$$
\boldsymbol{\eta} = \boldsymbol{\Phi}\boldsymbol{w} \in \mathbb{R}^N, \qquad \eta_n = \boldsymbol{w}^{\top} \boldsymbol{\phi}(\boldsymbol{x}_n),
$$

where $\boldsymbol{w} \in \mathbb{R}^D$ is the parameter vector. In ordinary least squares (OLS), $\mathcal{Y} = \mathbb{R}$ and the conditional mean equals $\eta_n$. In classification ($\mathcal{Y} = \{0,1\}$ or $\{1,\dots,C\}$) and count regression ($\mathcal{Y} = \mathbb{N}_0$), $\eta_n$ parameterizes an exponential-family conditional distribution $p(y_n \mid \eta_n, \phi)$ via a monotone link function $g(\mu_n) = \eta_n$ with $\mu_n = \mathbb{E}[y_n \mid \boldsymbol{x}_n, \boldsymbol{w}] = g^{-1}(\eta_n)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 12, PDF p. 445).

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Source textbooks adopt distinct conventions for design matrices, parameter vectors, targets, and link/activation functions. Table `tab:glm:notation` establishes a unified mapping used throughout this monograph, reconciling [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 10–12), [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 9), [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 7), [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 4), [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 7, 14–15), and [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 4–5).
>
>
>
>
> | **Concept** | **Unified** | **Source Variants** | **Reconciliation Notes** |
> | — | — | — | — |
> | Design matrix | $\boldsymbol{\Phi} \in \mathbb{R}^{N \times D}$ | $\boldsymbol{X}$ (Murphy, IPDS, Tiep Vu); $\boldsymbol{\Phi}$ (MML, Bishop) | Rows are feature vectors $\boldsymbol{\phi}(\boldsymbol{x}_n)^{\top}$; $D=d$ when $\boldsymbol{\phi}(\boldsymbol{x})=\boldsymbol{x}$. |
> | Weight vector | $\boldsymbol{w} \in \mathbb{R}^D$ | $\boldsymbol{w}$ (Murphy, Bishop, Tiep Vu); $\boldsymbol{\theta}$ (MML, IPDS, SLP) | Bias $w_0$ is absorbed via $\phi_0(\boldsymbol{x})=1$, but excluded from $\ell_1/\ell_2$ penalties. |
> | Target / observation | $y_n \in \mathcal{Y}$ | $y_n$ (Murphy, MML, IPDS, SLP); $t_n$ (Bishop) | Bishop reserves $y(\boldsymbol{x},\boldsymbol{w})$ for the model prediction $\hat{y}_n = \mu_n$. |
> | Dispersion / noise | $\phi$ (or $\sigma^2$) | $\sigma^2$ (Murphy, MML); $s$ (Bishop Ch. 5.4.6) | Gaussian $\phi = \sigma^2$; Bernoulli and Poisson have fixed $\phi = 1$. |
> | Link / inverse link | $g(\mu) = \eta$ | $\ell(\mu) = \eta$ (Murphy); $f(\eta)=\mu,  f^{-1}=\psi$ (Bishop) | Statistics uses link $g=f^{-1}$; ML uses activation $f = g^{-1}$. |

## Core Mathematical Derivations & Theorems

### OLS Orthogonal Projection & Gaussian MLE Equivalence

> [!important] Normal Equations, Orthogonal Projection & Gaussian MLE
> Assume i.i.d. observations $y_n = \boldsymbol{w}^{\top}\boldsymbol{\phi}(\boldsymbol{x}_n) + \epsilon_n$ with $\epsilon_n \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 4, PDF p. 134). The negative log-likelihood is
>
> $$
> -\log p(\boldsymbol{y} \mid \boldsymbol{\Phi}, \boldsymbol{w}, \sigma^2) = \frac{N}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\lVert \boldsymbol{y} - \boldsymbol{\Phi}\boldsymbol{w} \rVert_2^2.
> $$
>
> Because $\sigma^2 > 0$ is independent of $\boldsymbol{w}$, maximizing the Gaussian likelihood is equivalent to minimizing the residual sum of squares $\mathcal{L}_{\mathrm{OLS}}(\boldsymbol{w}) = \frac{1}{2}\lVert \boldsymbol{y} - \boldsymbol{\Phi}\boldsymbol{w} \rVert_2^2$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 7, PDF p. 97). Setting $\nabla_{\boldsymbol{w}}\mathcal{L}_{\mathrm{OLS}}(\boldsymbol{w}) = \boldsymbol{\Phi}^{\top}(\boldsymbol{\Phi}\boldsymbol{w} - \boldsymbol{y}) = \boldsymbol{0}$ yields the normal equations:
>
> $$
> (\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi})\hat{\boldsymbol{w}}_{\mathrm{MLE}} = \boldsymbol{\Phi}^{\top}\boldsymbol{y} \implies \hat{\boldsymbol{w}}_{\mathrm{MLE}} = (\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^{\top} \boldsymbol{y} \quad (\text{if }\mathrm{rank}(\boldsymbol{\Phi}) = D).
> $$
>
> The predicted target vector $\hat{\boldsymbol{y}} = \boldsymbol{\Phi}\hat{\boldsymbol{w}}_{\mathrm{MLE}} = \boldsymbol{P}_{\mathcal{S}}\boldsymbol{y}$ is the orthogonal projection of $\boldsymbol{y} \in \mathbb{R}^N$ onto the $D$-dimensional column space $\mathcal{S} = \mathrm{col}(\boldsymbol{\Phi}) \subset \mathbb{R}^N$, where
>
> $$
> \boldsymbol{P}_{\mathcal{S}} = \boldsymbol{\Phi}(\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^{\top}
> $$
>
> is symmetric ($\boldsymbol{P}_{\mathcal{S}}^{\top} = \boldsymbol{P}_{\mathcal{S}}$) and idempotent ($\boldsymbol{P}_{\mathcal{S}}^2 = \boldsymbol{P}_{\mathcal{S}}$), and the residual $\boldsymbol{r} = \boldsymbol{y} - \hat{\boldsymbol{y}} = (\boldsymbol{I}_N - \boldsymbol{P}_{\mathcal{S}})\boldsymbol{y}$ is orthogonal to $\mathcal{S}$, i.e., $\boldsymbol{\Phi}^{\top}\boldsymbol{r} = \boldsymbol{0}$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 9, PDF p. 320).

> [!tip] Subspace Geometry of Least Squares
> In sample space $\mathbb{R}^N$, each column $\boldsymbol{\varphi}_j \in \mathbb{R}^N$ of $\boldsymbol{\Phi}$ spans one coordinate axis of the hypothesis subspace $\mathcal{S} = \mathrm{span}\{\boldsymbol{\varphi}_1,\dots,\boldsymbol{\varphi}_D\}$. Any linear model prediction $\boldsymbol{\Phi}\boldsymbol{w}$ lies in $\mathcal{S}$. The Euclidean distance $\lVert \boldsymbol{y} - \boldsymbol{\Phi}\boldsymbol{w} \rVert_2$ is minimized precisely when the error vector $\boldsymbol{y} - \hat{\boldsymbol{y}}$ meets $\mathcal{S}$ at a right angle (Figure `fig:glm:ols_proj`). When the basis columns are orthonormal ($\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi} = \boldsymbol{I}_D$), features decouple completely: $\hat{\boldsymbol{y}} = \sum_{j=1}^D \boldsymbol{\varphi}_j(\boldsymbol{\varphi}_j^{\top}\boldsymbol{y})$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 9, PDF p. 320).

> [!note] Source Figure
> *Figure:* Orthogonal projection geometry of ordinary least squares in $\mathbb{R}^N$: the target vector $\mathbf{t}$ (denoted $\boldsymbol{y}$ in our unified notation) is projected orthogonally onto the subspace $\mathcal{S}$ spanned by basis columns $\boldsymbol{\varphi}_1, \boldsymbol{\varphi}_2$, yielding the minimum-norm residual $\mathbf{t} - \mathbf{y}$. Source: [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 4, Fig. 4.3, PDF p. 136). (`deep_learning_foundations_concepts_p136_fig1.png`)

### Bayesian Linear Regression & Ridge ($\ell_2$) Duality

> [!important] Conjugate Gaussian Posterior & Ridge MAP Equivalence
> Place a conjugate Gaussian prior $p(\boldsymbol{w}) = \mathcal{N}(\boldsymbol{w} \mid \boldsymbol{m}_0, \boldsymbol{S}_0)$ on the weights. By completing the square in the log-joint $\log p(\boldsymbol{y} \mid \boldsymbol{\Phi},\boldsymbol{w},\sigma^2) + \log p(\boldsymbol{w})$, the exact posterior is $p(\boldsymbol{w} \mid \mathcal{D}) = \mathcal{N}(\boldsymbol{w} \mid \boldsymbol{m}_N, \boldsymbol{S}_N)$ [[Source - Mathematics for Machine Learning|mathematics_machine_learning_orienly]] (Ch. 9, PDF p. 310), where
>
> $$
> \boldsymbol{S}_N^{-1} = \boldsymbol{S}_0^{-1} + \frac{1}{\sigma^2}\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}, \qquad
> \boldsymbol{m}_N = \boldsymbol{S}_N\!\left(\boldsymbol{S}_0^{-1}\boldsymbol{m}_0 + \frac{1}{\sigma^2}\boldsymbol{\Phi}^{\top}\boldsymbol{y}\right).
> $$
>
> Integrating out $\boldsymbol{w}$ yields the Gaussian posterior predictive distribution at a test input $\boldsymbol{\phi}_* = \boldsymbol{\phi}(\boldsymbol{x}_*)$:
>
> $$
> p(y_* \mid \boldsymbol{x}_*, \mathcal{D}) = \int \mathcal{N}(y_* \mid \boldsymbol{w}^{\top}\boldsymbol{\phi}_*, \sigma^2)\mathcal{N}(\boldsymbol{w} \mid \boldsymbol{m}_N, \boldsymbol{S}_N) \mathrm{d}\boldsymbol{w}
> = \mathcal{N}\!\left(y_*  \middle|  \boldsymbol{m}_N^{\top}\boldsymbol{\phi}_*,  \sigma^2 + \boldsymbol{\phi}_*^{\top}\boldsymbol{S}_N\boldsymbol{\phi}_*\right),
> $$
>
> where $\boldsymbol{\phi}_*^{\top}\boldsymbol{S}_N\boldsymbol{\phi}_*$ captures epistemic parameter uncertainty [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 11, PDF p. 437). Choosing an isotropic zero-mean prior $\boldsymbol{m}_0 = \boldsymbol{0}$, $\boldsymbol{S}_0 = \tau^2\boldsymbol{I}_D$ reduces the negative log-posterior to the Tikhonov/Ridge objective [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 11, PDF p. 412):
>
> $$
> \hat{\boldsymbol{w}}_{\mathrm{Ridge}} = \arg\min_{\boldsymbol{w}\in\mathbb{R}^D} \left\{\lVert \boldsymbol{y} - \boldsymbol{\Phi}\boldsymbol{w} \rVert_2^2 + \lambda\lVert \boldsymbol{w} \rVert_2^2\right\} = (\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi} + \lambda\boldsymbol{I}_D)^{-1}\boldsymbol{\Phi}^{\top}\boldsymbol{y}, \quad \lambda = \frac{\sigma^2}{\tau^2}.
> $$

Taking the eigendecomposition $\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi} = \boldsymbol{V}\boldsymbol{\Sigma}^2\boldsymbol{V}^{\top}$ (from the thin SVD $\boldsymbol{\Phi} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$ with singular values $\sigma_j \ge 0$), Ridge shifts every eigenvalue of $\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}$ from $\sigma_j^2$ to $\sigma_j^2 + \lambda > 0$, guaranteeing invertibility even when $D > N$ or $\boldsymbol{\Phi}$ is rank-deficient [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 7, PDF p. 458). On the training data, the Ridge fit acts as a smooth spectral filter [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 11, PDF p. 413):

$$
\hat{\boldsymbol{y}}_{\mathrm{Ridge}} = \boldsymbol{\Phi}\hat{\boldsymbol{w}}_{\mathrm{Ridge}} = \sum_{j=1}^{r} \frac{\sigma_j^2}{\sigma_j^2 + \lambda}\boldsymbol{u}_j(\boldsymbol{u}_j^{\top}\boldsymbol{y}), \qquad
\mathrm{dof}(\lambda) = \mathrm{Tr}\!\left(\boldsymbol{\Phi}(\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}+\lambda\boldsymbol{I}_D)^{-1}\boldsymbol{\Phi}^{\top}\right) = \sum_{j=1}^r \frac{\sigma_j^2}{\sigma_j^2 + \lambda}.
$$

Directions with small singular values $\sigma_j^2 \ll \lambda$ (high-variance noise directions) are heavily attenuated, trading a small bias for a large variance reduction.

### Lasso ($\ell_1$) Sparsity, Subdifferential Optimality & Soft-Thresholding

> [!important] Laplace Prior MAP, KKT Subdifferential & Soft-Thresholding
> Replacing the Gaussian prior with an i.i.d. Laplace prior $p(\boldsymbol{w} \mid \lambda) = \prod_{j=1}^D \frac{\lambda}{2}\exp(-\lambda\lvert w_j \rvert)$ yields the Lasso ($\ell_1$-regularized) convex optimization problem [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 11, PDF p. 415):
>
> $$
> \hat{\boldsymbol{w}}_{\mathrm{Lasso}} = \arg\min_{\boldsymbol{w}\in\mathbb{R}^D} \left\{\frac{1}{2}\lVert \boldsymbol{y} - \boldsymbol{\Phi}\boldsymbol{w} \rVert_2^2 + \lambda\lVert \boldsymbol{w} \rVert_1\right\}, \qquad \lVert \boldsymbol{w} \rVert_1 = \sum_{j=1}^D \lvert w_j \rvert.
> $$
>
> Because $\lvert w_j \rvert$ is non-differentiable at $w_j = 0$, first-order optimality requires $\boldsymbol{0} \in \boldsymbol{\Phi}^{\top}(\boldsymbol{\Phi}\boldsymbol{w} - \boldsymbol{y}) + \lambda\partial\lVert \boldsymbol{w} \rVert_1$, where the coordinate-wise subdifferential is $\partial\lvert w_j \rvert = \{\mathrm{sign}(w_j)\}$ for $w_j \neq 0$ and $[-1, 1]$ for $w_j = 0$. Consequently, for each coordinate $j$:
>
> $$
> \boldsymbol{\varphi}_j^{\top}(\boldsymbol{\Phi}\hat{\boldsymbol{w}} - \boldsymbol{y}) = -\lambda\mathrm{sign}(\hat{w}_j)    (\hat{w}_j \neq 0), \qquad
> \lvert \boldsymbol{\varphi}_j^{\top}(\boldsymbol{\Phi}\hat{\boldsymbol{w}} - \boldsymbol{y}) \rvert \le \lambda    (\hat{w}_j = 0).
> $$
>
> When columns of $\boldsymbol{\Phi}$ are orthonormal ($\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi} = \boldsymbol{I}_D$) with $c_j = \boldsymbol{\varphi}_j^{\top}\boldsymbol{y}$, the coordinates decouple and yield the exact closed-form soft-thresholding operator:
>
> $$
> \hat{w}_j^{\mathrm{Lasso}} = \mathcal{S}_\lambda(c_j) \triangleq \mathrm{sign}(c_j)\max(0, \lvert c_j \rvert - \lambda).
> $$

By Lagrangian duality, `(eq:glm:lasso_obj)` is equivalent to minimizing $\frac{1}{2}\lVert \boldsymbol{y}-\boldsymbol{\Phi}\boldsymbol{w} \rVert_2^2$ subject to $\lVert \boldsymbol{w} \rVert_1 \le B$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 11, PDF p. 416). As shown in Figure `fig:glm:l1_l2_geom`, the $\ell_1$ ball is a cross-polytope (diamond in 2D) whose sharp vertices lie on the coordinate axes; expanding elliptical MSE level sets generically intersect a vertex first, forcing $\hat{w}_j = 0$ exactly. By contrast, the smooth $\ell_2$ ball has no corners, shrinking coefficients toward zero without inducing exact sparsity [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE|introduction_probability_data_science]] (Ch. 7, PDF p. 466). Combining both penalties yields Elastic-Net, $\lambda_1\lVert \boldsymbol{w} \rVert_1 + \frac{\lambda_2}{2}\lVert \boldsymbol{w} \rVert_2^2$, which preserves sparsity while stabilizing correlated feature groups.

> [!note] Source Figure
> *Figure:* Constraint-set geometry of $\ell_1$ Lasso (left) versus $\ell_2$ Ridge (right) regularization. Elliptical least-squares contours centered at $\hat{\mathbf{w}}_{\mathrm{MLE}}$ first contact the $\ell_1$ diamond at a coordinate axis vertex ($w_1 = 0$), inducing exact sparsity, whereas the smooth $\ell_2$ disk yields a dense shrunk solution. Source: [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 11, Fig. 11.8, PDF p. 416). (`probabilisticml_kevinmurphy_introduction_p416_fig1.png`)

### Binary & Multinomial Logistic Regression, Hessian & IRLS

> [!important] Cross-Entropy Gradient, Convexity & IRLS Hessian
> For binary targets $y_n \in \{0,1\}$, modeling $p(y_n=1 \mid \boldsymbol{x}_n, \boldsymbol{w}) = \mu_n = \sigma(\eta_n) = (1 + e^{-\eta_n})^{-1}$ with $\eta_n = \boldsymbol{w}^{\top}\boldsymbol{\phi}_n$ gives the Bernoulli negative log-likelihood (binary cross-entropy) [[Source - Speech and Language Processing|speech_language_processing_aug26]] (Ch. 4, PDF p. 110):
>
> $$
> \mathrm{NLL}(\boldsymbol{w}) = -\sum_{n=1}^N \Big[y_n\log\mu_n + (1-y_n)\log(1-\mu_n)\Big].
> $$
>
> Using the sigmoid identity $\sigma'(\eta_n) = \mu_n(1-\mu_n)$, the gradient and Hessian with respect to $\boldsymbol{w}$ are [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 10, PDF p. 374):
>
> $$
> \nabla_{\boldsymbol{w}}\mathrm{NLL}(\boldsymbol{w}) = \boldsymbol{\Phi}^{\top}(\boldsymbol{\mu} - \boldsymbol{y}), \qquad
> \boldsymbol{H}(\boldsymbol{w}) = \nabla^2_{\boldsymbol{w}}\mathrm{NLL}(\boldsymbol{w}) = \boldsymbol{\Phi}^{\top}\boldsymbol{W}\boldsymbol{\Phi},
> $$
>
> where $\boldsymbol{W} = \mathrm{diag}(\mu_1(1-\mu_1), \dots, \mu_N(1-\mu_N))$. Since $\mu_n(1-\mu_n) > 0$, for any $\boldsymbol{v} \in \mathbb{R}^D$, $\boldsymbol{v}^{\top}\boldsymbol{H}(\boldsymbol{w})\boldsymbol{v} = \lVert \boldsymbol{W}^{1/2}\boldsymbol{\Phi}\boldsymbol{v} \rVert_2^2 \ge 0$, proving $\mathrm{NLL}(\boldsymbol{w})$ is convex (Figure `fig:glm:logreg_surface`). Applying Newton–Raphson $\boldsymbol{w}^{(t+1)} = \boldsymbol{w}^{(t)} - \boldsymbol{H}(\boldsymbol{w}^{(t)})^{-1} \nabla\mathrm{NLL}(\boldsymbol{w}^{(t)})$ yields Iteratively Reweighted Least Squares (IRLS):
>
> $$
> \boldsymbol{w}^{(t+1)} = (\boldsymbol{\Phi}^{\top}\boldsymbol{W}^{(t)}\boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^{\top}\boldsymbol{W}^{(t)}\boldsymbol{z}^{(t)}, \qquad
> \boldsymbol{z}^{(t)} = \boldsymbol{\Phi}\boldsymbol{w}^{(t)} + (\boldsymbol{W}^{(t)})^{-1}(\boldsymbol{y} - \boldsymbol{\mu}^{(t)}).
> $$
>
> For $C$-class multinomial logistic (softmax) regression with weights $\boldsymbol{W}_{\mathrm{cls}} = [\boldsymbol{w}_1,\dots,\boldsymbol{w}_C] \in \mathbb{R}^{D \times C}$, class probabilities are $\mu_{nc} = \frac{\exp(\boldsymbol{w}_c^{\top}\boldsymbol{\phi}_n)}{\sum_{k=1}^C\exp(\boldsymbol{w}_k^{\top}\boldsymbol{\phi}_n)}$ [[Source - Machine Learning co ban|machine_learning_co_ban_tiep_vu]] (Ch. 15, PDF p. 198). With one-hot targets $\boldsymbol{y}_n \in \{0,1\}^C$, the cross-entropy gradient for class $c$ is $\nabla_{\boldsymbol{w}_c}\mathrm{NLL} = \sum_{n=1}^N (\mu_{nc} - y_{nc})\boldsymbol{\phi}_n$, and the block Hessian is [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 10, PDF p. 385)
>
> $$
> \boldsymbol{H}(\boldsymbol{w}) = \sum_{n=1}^N \Big(\mathrm{diag}(\boldsymbol{\mu}_n) - \boldsymbol{\mu}_n\boldsymbol{\mu}_n^{\top}\Big) \otimes (\boldsymbol{\phi}_n\boldsymbol{\phi}_n^{\top}) \succeq \boldsymbol{0}.
> $$
>
> Note that unregularized softmax is shift-invariant ($\boldsymbol{w}_c \mapsto \boldsymbol{w}_c + \boldsymbol{c}$ leaves $\mu_{nc}$ unchanged); adding $\ell_2$ regularization $\frac{\lambda}{2}\sum_{c=1}^C\lVert \boldsymbol{w}_c \rVert_2^2$ enforces the unique sum-to-zero constraint $\sum_{c=1}^C \hat{w}_{cj} = 0$ for every feature $j$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 10, PDF p. 386).

> [!note] Source Figure
> *Figure:* Binary logistic regression NLL loss surface in 2D parameter space $(w, b)$: (a) convex contour plot and (b) bowl-shaped surface with a unique global minimum whenever the data are not linearly separable. Source: [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 10, Fig. 10.5, PDF p. 375). (`probabilisticml_kevinmurphy_introduction_p375_fig1.png`)

### Unified Canonical Exponential-Family GLMs

> [!important] Exponential Dispersion Family & Canonical Link Gradient
> A Generalized Linear Model assumes the target $y_n$ follows an exponential dispersion family distribution conditional on $\boldsymbol{x}_n$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 12, PDF p. 445; \bookcite{deep_learning_foundations_concepts}{Ch. 5, PDF p. 183}):
>
> $$
> p(y_n \mid \theta_n, \phi) = h(y_n, \phi)\exp\!\left(\frac{y_n\theta_n - A(\theta_n)}{\phi}\right),
> $$
>
> where $\theta_n$ is the natural (canonical) parameter, $A(\theta_n)$ is the log-partition (cumulant) function, and $\phi > 0$ is the dispersion parameter. Differentiating the normalization identity $\int p(y\mid\theta,\phi) \mathrm{d}y = 1$ twice with respect to $\theta$ gives the exact mean and variance:
>
> $$
> \mu_n = \mathbb{E}[y_n \mid \theta_n] = A'(\theta_n), \qquad \mathrm{Var}(y_n \mid \theta_n) = \phi A"(\theta_n) = \phi  V(\mu_n).
> $$
>
> A *canonical link function* chooses $g = (A')^{-1}$ so that the natural parameter equals the linear predictor directly: $\theta_n = g(\mu_n) = \eta_n = \boldsymbol{w}^{\top}\boldsymbol{\phi}_n$ [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 5, PDF p. 184). Under this canonical pairing, the log-likelihood $\ell(\boldsymbol{w}) = \sum_{n=1}^N \frac{y_n\eta_n - A(\eta_n)}{\phi} + \mathrm{const}$ has score and Fisher Information:
>
> $$
> \frac{\partial \ell_n}{\partial \eta_n} = \frac{y_n - A'(\eta_n)}{\phi} = \frac{y_n - \mu_n}{\phi}, \qquad
> \nabla_{\boldsymbol{w}}\mathrm{NLL}(\boldsymbol{w}) = \frac{1}{\phi}\boldsymbol{\Phi}^{\top}(\boldsymbol{\mu} - \boldsymbol{y}), \qquad
> \mathcal{I}(\boldsymbol{w}) = \frac{1}{\phi}\boldsymbol{\Phi}^{\top}\boldsymbol{W}\boldsymbol{\Phi},
> $$
>
> where $W_{nn} = A"(\eta_n) = \frac{\mathrm{d}\mu_n}{\mathrm{d}\eta_n}$. Thus, OLS, logistic regression, and Poisson regression share the identical residual-weighted gradient structure $\boldsymbol{\Phi}^{\top}(\boldsymbol{\mu} - \boldsymbol{y})$ and positive semi-definite Hessian $\boldsymbol{\Phi}^{\top}\boldsymbol{W}\boldsymbol{\Phi}$.

| **Distribution** | **Support $\mathcal{Y}$** | **Cumulant $A(\eta)$** | **$\phi$** | **Canonical Link $g(\mu)$** | **Mean $\mu = A'(\eta)$** |
| — | — | — | — | — | — |
| Gaussian $\mathcal{N}(\mu,\sigma^2)$ | $\mathbb{R}$ | $\frac{1}{2}\eta^2$ | $\sigma^2$ | $\mu$ (Identity) | $\eta$ |
| Bernoulli $\mathrm{Bern}(\mu)$ | $\{0,1\}$ | $\log(1+e^\eta)$ | $1$ | $\log\frac{\mu}{1-\mu}$ (Logit) | $\sigma(\eta)$ |
| Poisson $\mathrm{Poi}(\mu)$ | $\mathbb{N}_0$ | $e^\eta$ | $1$ | $\log\mu$ (Log) | $\exp(\eta)$ |

Non-canonical links decouple $\theta_n$ from $\eta_n$, adding a Jacobian factor $\frac{\mathrm{d}\theta_n}{\mathrm{d}\eta_n} = [A"(\theta_n) g'(\mu_n)]^{-1}$ to the score [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 5, PDF p. 184). For example, probit regression uses $\eta_n = \Phi_{\mathcal{N}}^{-1}(\mu_n)$ (inverse standard normal CDF). Because $\Phi_{\mathcal{N}}(\eta)$ has Gaussian tails $\mathcal{O}(e^{-\eta^2/2})$ rather than logistic tails $\mathcal{O}(e^{-\lvert \eta \rvert})$, probit regression is significantly more sensitive to mislabeled outliers far from the decision boundary [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 5, PDF p. 183).

## Algorithmic Implementation & Comparative Synthesis

We present two core production solvers: Algorithm `alg:glm:irls` (IRLS for any canonical GLM, which reduces to a single step for Gaussian OLS) and Algorithm `alg:glm:coord_lasso` (Coordinate Descent with soft-thresholding for Lasso, avoiding non-smooth subgradient oscillation).

> [!example] Algorithm: Iteratively Reweighted Least Squares (IRLS) for Canonical GLMs
> **Input:** Design matrix $\boldsymbol{\Phi} \in \mathbb{R}^{N \times D}$, targets $\boldsymbol{y} \in \mathcal{Y}^N$, ridge $\lambda \ge 0$, tolerance $\varepsilon > 0$
> **Output:** MAP/MLE parameter vector $\hat{\boldsymbol{w}} \in \mathbb{R}^D$
> Initialize $\boldsymbol{w}^{(0)} \leftarrow \boldsymbol{0}$
> **for** $t = 0, 1, \dots, T_{\max}$:
>
>   $\boldsymbol{\eta} \leftarrow \boldsymbol{\Phi}\boldsymbol{w}^{(t)}$  *// Linear predictor*
>   $\boldsymbol{\mu} \leftarrow A'(\boldsymbol{\eta})$,\quad $\boldsymbol{W} \leftarrow \mathrm{diag}\!\big(A"(\boldsymbol{\eta})\big)$  *// Mean and variance weights*
>   $\boldsymbol{z} \leftarrow \boldsymbol{\eta} + \boldsymbol{W}^{-1}(\boldsymbol{y} - \boldsymbol{\mu})$  *// Working response*
>   Solve $(\boldsymbol{\Phi}^{\top}\boldsymbol{W}\boldsymbol{\Phi} + \lambda\boldsymbol{I}_D)\boldsymbol{w}^{(t+1)} = \boldsymbol{\Phi}^{\top}\boldsymbol{W}\boldsymbol{z}$ via Cholesky / QR
>   **if** $\lVert \boldsymbol{w}^{(t+1)} - \boldsymbol{w}^{(t)} \rVert_2 < \varepsilon$:
> **return** $\boldsymbol{w}^{(t+1)}$
>
> **return** $\boldsymbol{w}^{(T_{\max})}$

> [!example] Algorithm: Coordinate Descent with Soft-Thresholding for Lasso ($\ell_1$)
> **Input:** Standardized $\boldsymbol{\Phi} \in \mathbb{R}^{N \times D}$ ($\lVert \boldsymbol{\varphi}_j \rVert_2^2 = N$), centered $\boldsymbol{y}$, penalty $\lambda > 0$, tol $\varepsilon$
> **Output:** Sparse parameter vector $\hat{\boldsymbol{w}} \in \mathbb{R}^D$
> Initialize $\boldsymbol{w} \leftarrow \boldsymbol{0}$,\quad $\boldsymbol{r} \leftarrow \boldsymbol{y} - \boldsymbol{\Phi}\boldsymbol{w}$
> **while** $\lVert \boldsymbol{w} - \boldsymbol{w}_{\mathrm{prev}} \rVert_\infty > \varepsilon$:
>
>   $\boldsymbol{w}_{\mathrm{prev}} \leftarrow \boldsymbol{w}$
>   **for** $j = 1, \dots, D$:
>
>     $\rho_j \leftarrow \boldsymbol{\varphi}_j^{\top}(\boldsymbol{r} + w_j\boldsymbol{\varphi}_j)$  *// Partial residual correlation*
>     $w_j^{\mathrm{new}} \leftarrow \frac{1}{N}\mathrm{sign}(\rho_j)\max(0,  \lvert \rho_j \rvert - \lambda)$  *// Soft-thresholding*
>     $\boldsymbol{r} \leftarrow \boldsymbol{r} - (w_j^{\mathrm{new}} - w_j)\boldsymbol{\varphi}_j$,\quad $w_j \leftarrow w_j^{\mathrm{new}}$
>
>
> **return** $\boldsymbol{w}$

| **Model / Solver** | **Time Complexity** | **Memory** | **Regime & Numerical Trade-off** |
| — | — | — | — |
| OLS / Ridge (QR) | $\mathcal{O}(ND^2)$ | $\mathcal{O}(ND)$ | Preferred for $N \ge D$; avoids squaring condition number $\kappa(\boldsymbol{\Phi})$. |
| Ridge (Dual SVD) | $\mathcal{O}(DN^2)$ | $\mathcal{O}(ND)$ | Optimal when $D \gg N$; operates in $N\times N$ Gram space. |
| Lasso (Coord. Desc.) | $\mathcal{O}(K_{\mathrm{sweep}} ND)$ | $\mathcal{O}(ND)$ | Exact sparsity via active-set updates; no matrix inversion. |
| GLM / Logistic (IRLS) | $\mathcal{O}(T_{\mathrm{IRLS}} ND^2)$ | $\mathcal{O}(ND + D^2)$ | Quadratic convergence; requires solving $D\times D$ system per step. |
| Logistic (L-BFGS / SGD) | $\mathcal{O}(K N D)$ / $\mathcal{O}(K B D)$ | $\mathcal{O}(mD)$ / $\mathcal{O}(BD)$ | Scales to massive $D$ and streaming $N$ where $D^2$ Hessian is intractable. |

> [!info] Cross-Book Synthesis: Generative vs. Discriminative Classifiers & Calibration
> Both [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 9, PDF p. 366) and [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 5, PDF pp. 172–178) prove that Linear Discriminant Analysis (LDA) with tied class covariances $\boldsymbol{\Sigma}_c = \boldsymbol{\Sigma}$ yields a posterior $p(y=c \mid \boldsymbol{x})$ of exact softmax form. However, logistic regression estimates $p(y \mid \boldsymbol{x})$ directly without modeling $p(\boldsymbol{x} \mid y)$, making it more robust to non-Gaussian features, whereas generative models can handle missing inputs and require fewer samples when model assumptions hold. For count/rate targets (e.g., insurance claims), [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 12, PDF pp. 450–451) demonstrates that Ridge linear regression can predict negative counts and miscalibrate low-frequency bins, whereas a Poisson GLM with canonical log link ($\mu = e^\eta$) minimizes Poisson deviance and maintains calibrated expected totals.

## Visual Diagnostics & Geometric Structure

Figure `fig:glm:tikz_pipeline` maps the canonical GLM architecture from input features through the exponential dispersion family to IRLS parameter updates. Figure `fig:glm:pgf_paths` provides exact analytical comparisons of regularization shrinkage paths under orthonormal design and binary classification tail behavior.

> [!example] Visual Diagram (LaTeX / PDF)
> *Canonical Generalized Linear Model (GLM) pipeline: choosing the canonical link $g=(A')^{-1}$ equates the natural parameter $\theta_n$ to $\eta_n$, collapsing the score to residual-weighted features and yielding the IRLS Fisher scoring update.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Exact analytical diagnostics: (a) under orthonormal design, Ridge shrinks proportionally, whereas Lasso and Elastic-Net apply soft-thresholding with a deadzone $[-\lambda_1, \lambda_1]$; (b) for misclassified outliers ($\eta \ll 0$), logistic NLL grows linearly while probit NLL grows quadratically [[Source - Foundations|deep_learning_foundations_concepts]] (Ch. 5, PDF p. 183).* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Complete & Quasi-Complete Separation in Logistic Regression
> If training classes are linearly separable ($\exists \boldsymbol{w}: (2y_n - 1)\boldsymbol{w}^{\top}\boldsymbol{\phi}_n > 0 \forall n$), scaling $\boldsymbol{w} \leftarrow \alpha\boldsymbol{w}$ with $\alpha \to \infty$ drives $\mathrm{NLL}(\alpha\boldsymbol{w}) \to 0$ strictly. Consequently, the MLE does not exist at any finite $\boldsymbol{w}$, causing unregularized Newton/IRLS steps to diverge ($\lVert \boldsymbol{w} \rVert_2 \to \infty$) and $\boldsymbol{W} \to \boldsymbol{0}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 10, PDF p. 378). **Remedy:** Always include $\ell_2$ regularization ($\lambda > 0$) or a Jeffreys prior, which makes the objective strongly convex and guarantees a unique finite MAP estimate.

> [!warning] Logit & Softmax Overflow/Underflow
> Evaluating $\log(1 + e^\eta)$ or $\mathrm{softmax}(\boldsymbol{\eta})_c = \frac{e^{\eta_c}}{\sum_k e^{\eta_k}}$ naively overflows IEEE-754 float64 for $\eta_c > 709.78$ and underflows to $0/0 = \mathrm{NaN}$ when all $\eta_c \ll -745$. **Remedy:** Apply the Log-Sum-Exp shift $m = \max_k \eta_k$:
>
> $$
> \log\sum_{k=1}^C e^{\eta_k} = m + \log\sum_{k=1}^C e^{\eta_k - m}, \qquad
> \mathrm{softplus}(\eta) = \max(\eta, 0) + \log\!\left(1 + e^{-\lvert \eta \rvert}\right).
> $$

> [!warning] Collinearity, Unscaled Features & Normal-Equation Conditioning
> Forming $(\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi})^{-1}$ explicitly squares the condition number, $\kappa(\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}) = \kappa(\boldsymbol{\Phi})^2$, losing double precision digits when features are collinear [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 11, PDF p. 412). Furthermore, Ridge and Lasso penalties $\lVert \boldsymbol{w} \rVert_2^2, \lVert \boldsymbol{w} \rVert_1$ are not scale-invariant: multiplying feature $j$ by $10^{-3}$ scales $w_j$ by $10^3$, disproportionately penalizing small-scale features. **Remedy:** Standardize features to zero mean and unit variance before regularization (keeping the intercept $w_0$ unpenalized), and solve OLS/Ridge via QR or SVD rather than normal equations.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[Generalized Linear Models (GLM)]], [[Linear & Ridge Regression]], [[Logistic Regression]]
- **Prerequisites (Directed $\leftarrow$):** [[Exponential Families]], [[Maximum Likelihood & Fisher Info]], [[Orthogonal Projection]], [[Singular Value Decomposition (SVD)]]
- **Enables / Unlocks (Directed $\rightarrow$):** None
- **Related Concepts (Undirected $\leftrightarrow$):** [[Bias-Variance Trade-off]], [[Double Descent]], [[Exponential Families]], [[Generalized Linear Models (GLM)]], [[Logistic Regression]], [[Maximum Likelihood & Fisher Info]], [[Moore-Penrose Pseudoinverse]], [[Orthogonal Projection]], [[Proximal & Subgradient Methods]]
- **Synthesized Sources:** [[Source - Foundations]], [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]], [[Source - Speech and Language Processing]]
