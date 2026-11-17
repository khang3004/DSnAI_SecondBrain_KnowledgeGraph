---
type: monograph
domain: "D02"
parent_topic: "[[PT 02 - Linear Models, Generalization, Kernels & Ensembles]]"
sub_topic: "[[ST 02 - Kernel Methods, Support Vector Machines & Tree Ensembles]]"
cheatsheet: "[[Cheatsheet - Decision Trees, Bagging, Random Forests & Gradient Boosting]]"
part_of: 
  - "[[ST 02 - Kernel Methods, Support Vector Machines & Tree Ensembles]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
concepts: 
  - "[[CART Decision Trees]]"
  - "[[Gradient Boosted Decision Trees (GBDT)]]"
  - "[[Random Forests & Bagging]]"
prerequisites: 
  - "[[Bias-Variance Trade-off]]"
  - "[[CART Decision Trees]]"
  - "[[Gradient Descent]]"
enables: 
  - "[[Gradient Boosted Decision Trees (GBDT)]]"
  - "[[Random Forests & Bagging]]"
related: 
  - "[[Bias-Variance Trade-off]]"
  - "[[CART Decision Trees]]"
  - "[[Gradient Boosted Decision Trees (GBDT)]]"
  - "[[Random Forests & Bagging]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Designing Machine Learning Systems]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - monograph
  - domain-02
---

# Decision Trees, Bagging, Random Forests & Gradient Boosting

> [!abstract] Monograph Navigation
> - **Domain:** [[D02 - Classical, Kernel & Probabilistic Machine Learning|Classical, Kernel & Probabilistic Machine Learning]]
> - **Parent / Sub-Topic:** [[PT 02 - Linear Models, Generalization, Kernels & Ensembles|Linear Models, Generalization, Kernels & Ensembles]] → [[ST 02 - Kernel Methods, Support Vector Machines & Tree Ensembles|Kernel Methods, Support Vector Machines & Tree Ensembles]]
> - **Visual Cheatsheet:** [[Cheatsheet - Decision Trees, Bagging, Random Forests & Gradient Boosting]]
> - **Description:** Covers greedy CART splitting (Gini/entropy/MSE), variance reduction via bootstrap aggregating and feature-subsampled Random Forests, AdaBoost exponential loss, and functional gradient boosting (XGBoost/LightGBM).

## Problem Formulation & Unified Notation

Given a supervised dataset $\mathcal{D} = \{(\boldsymbol{x}_i, y_i)\}_{i=1}^{N}$ with feature vectors $\boldsymbol{x}_i \in \mathcal{X} \subseteq \mathbb{R}^{D}$ (possibly mixing continuous and categorical coordinates) and targets $y_i \in \mathcal{Y}$, a *tree ensemble* constructs a predictor $f : \mathcal{X} \to \mathbb{R}$ (or $\mathbb{R}^{C}$ for $C$-class logits) as a weighted combination of $M$ base decision trees:

$$
f(\boldsymbol{x}; \Theta) = \sum_{m=1}^{M} \beta_m F_m(\boldsymbol{x}; \theta_m), \qquad
  F_m(\boldsymbol{x}; \theta_m) = \sum_{j=1}^{J_m} w_{jm}  \mathbb{I}(\boldsymbol{x} \in R_{jm}),
$$

where $\{R_{1m}, \dots, R_{J_m m}\}$ forms a disjoint axis-aligned partition of $\mathcal{X}$ induced by tree $m$ with $J_m$ leaves, and $w_{jm}$ is the leaf prediction weight [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, pp. 603–611). Letting $q_m : \mathcal{X} \to \{1, \dots, J_m\}$ map $\boldsymbol{x}$ to its leaf index, we write $F_m(\boldsymbol{x}) = w_{q_m(\boldsymbol{x}), m}$.

Because tree ensembles unify parallel variance-reduction methods (Bagging, Random Forests) and sequential bias-reduction methods (AdaBoost, Gradient Boosted Decision Trees), notation varies across statistical, deep learning, and ML systems texts. Table `tab:trees:notation` establishes a unified convention.

> [!info] Cross-Book Notation & Pedagogical Reconciliation
> Notation reconciled across Murphy [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18), Goodfellow et al. [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 7.11), and Huyen [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Sec. 6.1):
>
>
>
>
>
> | Concept | Unified | Murphy (Ch. 18) | Goodfellow (7.11) | Huyen (Sec. 6.1) |
> | — | — | — | — | — |
> | Ensemble size | $M$ (or $B$) | $M$ | $k$ | $n$ base learners |
> | Base tree / learner | $F_m(\boldsymbol{x})$ | $F_m(\boldsymbol{x}; \theta_m)$ | $i$-th model | Base learner |
> | Leaf count / regions | $J_m$, $R_{jm}$ | $J$, $R_j$ | — | — |
> | Error var. / cov. | $\sigma^2$, $\rho\sigma^2$ | $\mathrm{Var}$, bias-var. | $v$, $c$ | Uncorr. vs. corr. |
> | Sample weights | $\omega_{i,m}$ | $\omega_{i,m}$ | — | Reweighted samples |
> | Shrinkage / step | $\nu$, $\beta_m$ | $\nu$, $\beta_m$ | — | Learning rate |

## Core Mathematical Derivations & Theorems

### CART Greedy Axis-Aligned Partitioning & Pruning

Finding the globally optimal binary tree partition minimizing empirical risk $\sum_{i=1}^{N} \ell(y_i, f(\boldsymbol{x}_i; \theta))$ is NP-complete [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, p. 605). Classification and Regression Trees (CART) instead grow a binary tree greedily top-down. At internal node $i$ receiving sample subset $\mathcal{D}_i \subseteq \mathcal{D}$, each candidate split $(d, t)$ on feature $d \in \{1,\dots,D\}$ and threshold $t \in \mathcal{T}_d$ partitions $\mathcal{D}_i$ into left and right children:

$$
\mathcal{D}_{i}^{L}(d, t) = \{(\boldsymbol{x}_n, y_n) \in \mathcal{D}_i : x_{n,d} \le t\}, \qquad
  \mathcal{D}_{i}^{R}(d, t) = \{(\boldsymbol{x}_n, y_n) \in \mathcal{D}_i : x_{n,d} > t\}.
$$

> [!important] CART Split Criterion & Minimal Cost-Complexity Pruning
> At node $i$, CART selects $(d_i^*, t_i^*)$ minimizing the size-weighted child impurity [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, pp. 605–606):
>
> $$
> (d_i^*, t_i^*) = \arg\min_{d \in \{1,\dots,D\},  t \in \mathcal{T}_d} \left[
>     \frac{|\mathcal{D}_i^L(d, t)|}{|\mathcal{D}_i|}  c(\mathcal{D}_i^L(d, t))
>     + \frac{|\mathcal{D}_i^R(d, t)|}{|\mathcal{D}_i|}  c(\mathcal{D}_i^R(d, t))
>   \right].
> $$
>
> For regression, $c(\mathcal{D}_i) = \frac{1}{|\mathcal{D}_i|}\sum_{n \in \mathcal{D}_i}(y_n - \bar{y}_i)^2$ with $\bar{y}_i = \frac{1}{|\mathcal{D}_i|}\sum_{n \in \mathcal{D}_i} y_n$. For $C$-class classification with empirical class frequencies $\hat{\pi}_{ic} = \frac{1}{|\mathcal{D}_i|}\sum_{n \in \mathcal{D}_i}\mathbb{I}(y_n = c)$, standard impurity functions are:
>
> $$
> \begin{aligned}
> G_i &= \sum_{c=1}^{C} \hat{\pi}_{ic}(1 - \hat{\pi}_{ic}) = 1 - \sum_{c=1}^{C} \hat{\pi}_{ic}^2 \quad \text{(Gini index)},  \\
>   H_i &= -\sum_{c=1}^{C} \hat{\pi}_{ic} \log \hat{\pi}_{ic} \quad \text{(Shannon entropy / deviance)}.
> \end{aligned}
> $$
>
> To prevent overfitting from unpruned deep trees, minimal cost-complexity pruning grows a maximal tree $T_0$ and selects the subtree $T \preceq T_0$ minimizing
>
> $$
> R_{\alpha}(T) = R(T) + \alpha |T|,
> $$
>
> where $R(T) = \sum_{j=1}^{|T|} \frac{|\mathcal{D}_j|}{N} c(\mathcal{D}_j)$ is total leaf impurity, $|T|$ is the number of terminal leaves, and $\alpha \ge 0$ is tuned via cross-validation.

> [!tip] Why Single Trees Are Unstable Yet Invariant
> Because splits depend only on threshold orderings $x_{n,d} \le t$, CART is strictly invariant to monotone transformations of any feature, handles mixed continuous/categorical inputs without scaling, and performs embedded feature selection [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, p. 607). However, hierarchical greedy splitting creates high variance: perturbing even a single training point near the root can alter the top split and cascade into a completely different partition.

### Bagging Variance Reduction, Correlation Floor & Random Forests

Bootstrap Aggregating (Bagging) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, p. 609) turns tree instability into a virtue by drawing $B$ bootstrap datasets $\mathcal{D}^{(1)}, \dots, \mathcal{D}^{(B)}$ of size $N$ with replacement, fitting a tree $F_b$ on each, and averaging $\bar{F}_B(\boldsymbol{x}) = \frac{1}{B}\sum_{b=1}^{B} F_b(\boldsymbol{x})$ (or taking majority vote in classification).

> [!important] Ensemble Variance Decomposition & Out-of-Bag Probability
> Suppose each base predictor $F_b(\boldsymbol{x})$ incurs a zero-mean error $\epsilon_b$ on test point $\boldsymbol{x}$ with identical variance $\mathbb{E}[\epsilon_b^2] = \sigma^2$ and pairwise covariance $\mathbb{E}[\epsilon_a \epsilon_b] = \rho\sigma^2$ for $a \neq b$ (where $\rho \in [0, 1]$ is the pairwise error correlation; cf. $v = \sigma^2$, $c = \rho\sigma^2$, $k = B$ in [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 7.11, Eq. 7.51)). The expected squared error of the ensemble average $\frac{1}{B}\sum_{b=1}^{B}\epsilon_b$ is:
>
> $$
> \begin{aligned}
> \mathbb{E}\!\left[\left(\frac{1}{B}\sum_{b=1}^{B}\epsilon_b\right)^2\right]
>   &= \frac{1}{B^2}\left(\sum_{b=1}^{B} \mathbb{E}[\epsilon_b^2] + \sum_{a \neq b} \mathbb{E}[\epsilon_a \epsilon_b]\right)  \\
>   &= \frac{1}{B^2}\left(B\sigma^2 + B(B-1)\rho\sigma^2\right)
>   = \rho\sigma^2 + \frac{1 - \rho}{B}\sigma^2.
> \end{aligned}
> $$
>
> As $B \to \infty$, the second term vanishes and error converges to the irreducible correlation floor $\rho\sigma^2$. Moreover, in each bootstrap sample of size $N$, the probability that training example $i$ is never drawn is
>
> $$
> \mathbb{P}(i \notin \mathcal{D}^{(b)}) = \left(1 - \frac{1}{N}\right)^{N} \xrightarrow{N \to \infty} e^{-1} \approx 0.368,
> $$
>
> leaving $\approx 36.8\%$ *out-of-bag* (OOB) instances per tree for internal cross-validation without a separate holdout set [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, p. 609).

**Random Forests** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, p. 610) break the $\rho\sigma^2$ ceiling by injecting a second source of randomness: at every internal node $i$, the split search in Eq. `(eq:trees:cart_objective)` is restricted to a uniformly sampled feature subset $\mathcal{S}_i \subset \{1, \dots, D\}$ of size $m$ (typically $m = \lfloor \sqrt{D} \rfloor$ for classification and $m = \lfloor D/3 \rfloor$ for regression). When a few dominant predictors exist, standard bagged trees repeatedly select those predictors near the root, keeping $\rho$ high; Random Forest feature subsampling forces diverse split paths, lowering $\rho$ substantially at a modest increase in single-tree bias.

### AdaBoost: Exponential Loss & Stagewise Additive Modeling

Rather than averaging independent trees to cut variance, *boosting* reduces bias by fitting weak learners sequentially. In forward stagewise additive modeling, iteration $m$ solves

$$
(\beta_m, F_m) = \arg\min_{\beta,  F} \sum_{i=1}^{N} \ell\!\left(y_i,  f_{m-1}(\boldsymbol{x}_i) + \beta F(\boldsymbol{x}_i)\right),
  \qquad f_m(\boldsymbol{x}) = f_{m-1}(\boldsymbol{x}) + \beta_m F_m(\boldsymbol{x}),
$$

freezing previously added learners $F_1, \dots, F_{m-1}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, p. 612).

> [!important] Derivation of AdaBoost.M1 from Exponential Loss
> For binary targets $\tilde{y}_i \in \{-1, +1\}$ and base classifiers $F_m(\boldsymbol{x}) \in \{-1, +1\}$, AdaBoost minimizes the exponential margin loss $\ell(\tilde{y}, f) = \exp(-\tilde{y} f)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, pp. 614–615). Defining sample weights $\omega_{i,m} \triangleq \exp(-\tilde{y}_i f_{m-1}(\boldsymbol{x}_i))$, the step-$m$ objective is:
>
> $$
> \begin{aligned}
> \mathcal{L}_m(\beta, F) &= \sum_{i=1}^{N} \omega_{i,m} \exp\!\left(-\beta \tilde{y}_i F(\boldsymbol{x}_i)\right)  \\
>   &= e^{-\beta}\sum_{\tilde{y}_i = F(\boldsymbol{x}_i)} \omega_{i,m} + e^{\beta}\sum_{\tilde{y}_i \neq F(\boldsymbol{x}_i)} \omega_{i,m}  \\
>   &= (e^{\beta} - e^{-\beta}) \sum_{i=1}^{N} \omega_{i,m} \mathbb{I}(\tilde{y}_i \neq F(\boldsymbol{x}_i)) + e^{-\beta}\sum_{i=1}^{N}\omega_{i,m}.
> \end{aligned}
> $$
>
> For any $\beta > 0$, the optimal weak learner minimizes weighted misclassification error:
>
> $$
> F_m = \arg\min_{F} \sum_{i=1}^{N} \omega_{i,m} \mathbb{I}(\tilde{y}_i \neq F(\boldsymbol{x}_i)).
> $$
>
> Letting $\mathrm{err}_m = \frac{\sum_{i=1}^{N} \omega_{i,m}\mathbb{I}(\tilde{y}_i \neq F_m(\boldsymbol{x}_i))}{\sum_{i=1}^{N} \omega_{i,m}}$ and setting $\frac{\partial \mathcal{L}_m}{\partial \beta} = 0$ yields the closed-form step weight and multiplicative sample reweighting:
>
> $$
> \beta_m = \frac{1}{2}\log\!\left(\frac{1 - \mathrm{err}_m}{\mathrm{err}_m}\right), \qquad
>   \omega_{i,m+1} = \omega_{i,m}\exp\!\left(2\beta_m \mathbb{I}(\tilde{y}_i \neq F_m(\boldsymbol{x}_i))\right) \cdot e^{-\beta_m}.
> $$
>
> Dropping the common factor $e^{-\beta_m}$ leaves misclassified examples scaled by $e^{2\beta_m} = \frac{1-\mathrm{err}_m}{\mathrm{err}_m} > 1$ whenever $\mathrm{err}_m < 0.5$.

### Functional Gradient Boosting & Second-Order XGBoost/LightGBM

To optimize arbitrary differentiable losses $\ell(y_i, f(\boldsymbol{x}_i))$, Friedman's Gradient Boosting Machine (GBM) treats the evaluation vector $\boldsymbol{f} = (f(\boldsymbol{x}_1), \dots, f(\boldsymbol{x}_N))^{\top} \in \mathbb{R}^N$ as the parameter being updated via functional steepest descent [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, pp. 616–618). At iteration $m$, the negative gradient (pseudo-residual) is

$$
r_{im} = -g_{im} = -\left.\frac{\partial \ell(y_i, f(\boldsymbol{x}_i))}{\partial f(\boldsymbol{x}_i)}\right|_{f = f_{m-1}},
$$

and a regression tree $F_m$ is fit by least squares to $\{(\boldsymbol{x}_i, r_{im})\}_{i=1}^{N}$, followed by a shrunk update $f_m(\boldsymbol{x}) = f_{m-1}(\boldsymbol{x}) + \nu F_m(\boldsymbol{x})$ with learning rate $0 < \nu \le 1$.

> [!important] Second-Order Taylor Objective, Optimal Leaf Weights & Split Gain
> Modern implementations (XGBoost, LightGBM) augment the functional objective with a tree complexity penalty $\Omega(F) = \gamma J + \frac{1}{2}\lambda \sum_{j=1}^{J} w_j^2$ and expand $\ell$ to second order around $f_{m-1}$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, pp. 619–620):
>
> $$
> \tilde{\mathcal{L}}^{(m)}(q, \boldsymbol{w}) = \sum_{i=1}^{N} \left[g_{im} w_{q(\boldsymbol{x}_i)} + \frac{1}{2} h_{im} w_{q(\boldsymbol{x}_i)}^2\right] + \gamma J + \frac{1}{2}\lambda \sum_{j=1}^{J} w_j^2,
> $$
>
> where $g_{im} = \partial_{f}\ell(y_i, f_{m-1}(\boldsymbol{x}_i))$ and $h_{im} = \partial_{f}^2\ell(y_i, f_{m-1}(\boldsymbol{x}_i)) \ge 0$. Grouping samples by leaf index $I_j = \{i : q(\boldsymbol{x}_i) = j\}$ and defining $G_{jm} = \sum_{i \in I_j} g_{im}$ and $H_{jm} = \sum_{i \in I_j} h_{im}$ decouples Eq. `(eq:trees:xgb_taylor)` into $J$ independent quadratics:
>
> $$
> \tilde{\mathcal{L}}^{(m)}(q, \boldsymbol{w}) = \sum_{j=1}^{J} \left[G_{jm} w_j + \frac{1}{2}(H_{jm} + \lambda) w_j^2\right] + \gamma J.
> $$
>
> Setting $\frac{\partial \tilde{\mathcal{L}}^{(m)}}{\partial w_j} = 0$ gives the exact Newton leaf weight and optimal tree structure score:
>
> $$
> w_j^* = -\frac{G_{jm}}{H_{jm} + \lambda}, \qquad
>   \tilde{\mathcal{L}}^{(m)}(q, \boldsymbol{w}^*) = -\frac{1}{2}\sum_{j=1}^{J}\frac{G_{jm}^2}{H_{jm} + \lambda} + \gamma J.
> $$
>
> Evaluating a candidate split $I = I_L \cup I_R$ yields the exact loss reduction (split gain):
>
> $$
> \mathrm{Gain}(I_L, I_R) = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma,
> $$
>
> which automatically enforces $L_2$ leaf shrinkage ($\lambda$) and minimum-gain pre-pruning ($\gamma$).

> [!tip] Why Second-Order Gradients Accelerate Boosting
> First-order GBM fits all leaf regions with uniform MSE weighting, requiring a separate line search or Newton step per leaf afterward. Using $h_{im}$ inside the split criterion acts as diagonal Newton preconditioning: regions with small curvature $H_{jm}$ (e.g., confident log-loss predictions where $\pi_i(1-\pi_i) \approx 0$) are regularized strongly by $\lambda$, preventing explosive leaf steps without manual step-size tuning.

## Algorithmic Implementation & Comparative Synthesis

To scale Eq. `(eq:trees:xgb_gain)` to millions of rows, XGBoost and LightGBM replace exact feature sorting with quantile/histogram binning ($K_{\mathrm{bin}} \approx 256$ bins per feature), reducing split-finding from $\mathcal{O}(N D \log N)$ to $\mathcal{O}(K_{\mathrm{bin}} D)$ per node once $(g_{im}, h_{im})$ are accumulated into histograms.

> [!example] Algorithm: Second-Order Histogram Gradient Boosted Decision Tree (XGBoost / LightGBM)
> **Input:** Dataset $\mathcal{D} = \{(\boldsymbol{x}_i, y_i)\}_{i=1}^{N}$, loss $\ell$, iterations $M$, shrinkage $\nu \in (0,1]$, depth $d_{\max}$, regularization $(\gamma, \lambda)$, subsample rates $(r_{\mathrm{row}}, r_{\mathrm{col}})$
> **Output:** Additive tree ensemble $f_M(\boldsymbol{x})$
> Quantile-bin continuous features into discrete bin indices $b_{i,d} \in \{1, \dots, K_{\mathrm{bin}}\}$
> Initialize $f_0(\boldsymbol{x}) \leftarrow \arg\min_{c \in \mathbb{R}} \sum_{i=1}^{N} \ell(y_i, c)$
> **for** $m = 1, \dots, M$:
>
>   Sample row subset $\mathcal{I}_m \subseteq \{1,\dots,N\}$ (rate $r_{\mathrm{row}}$) and feature subset $\mathcal{S}_m \subseteq \{1,\dots,D\}$ (rate $r_{\mathrm{col}}$)
>   Compute gradients $g_{im} \leftarrow \partial_f \ell(y_i, f_{m-1}(\boldsymbol{x}_i))$ and Hessians $h_{im} \leftarrow \partial_f^2 \ell(y_i, f_{m-1}(\boldsymbol{x}_i))$ for $i \in \mathcal{I}_m$
>   Grow tree $F_m$ up to depth $d_{\max}$ by recursively splitting active leaf $I$:
>   **for** each feature $d \in \mathcal{S}_m$:
>
>     Accumulate bin sums $G_{d,k} \leftarrow \sum_{i \in I : b_{i,d}=k} g_{im}$ and $H_{d,k} \leftarrow \sum_{i \in I : b_{i,d}=k} h_{im}$
>     Scan prefix sums $(G_L, H_L)$ across bins $k=1,\dots,K_{\mathrm{bin}}-1$ to evaluate $\mathrm{Gain}$ via Eq. `(eq:trees:xgb_gain)`
>
>   **if** $\max_{d,k} \mathrm{Gain}(d, k) > 0$:
>
>     Split $I$ into $(I_L, I_R)$ at $(d^*, k^*)$
>
>   **else**:
>
>     Set leaf weight $w_I^* \leftarrow -\frac{\sum_{i \in I} g_{im}}{\sum_{i \in I} h_{im} + \lambda}$
>
>   Update ensemble: $f_m(\boldsymbol{x}) \leftarrow f_{m-1}(\boldsymbol{x}) + \nu F_m(\boldsymbol{x})$
>
> **return** $f_M(\boldsymbol{x})$

**Table:** Computational and statistical comparison across tree-based paradigms.

| Method | Train Complexity | Parallelism | Primary Effect | Key Hyperparameters |
| — | — | — | — | — |
| Single CART | $\mathcal{O}(D N \log N)$ | Per-feature | Low bias (deep) | $d_{\max}$, min leaf size, $\alpha$ |
| Bagged Trees | $\mathcal{O}(B D N \log N)$ | Embarrassing ($B$) | Variance $\downarrow$ | $B$, bootstrap size |
| Random Forest | $\mathcal{O}(B m N \log N)$ | Embarrassing ($B$) | Variance $\downarrow$, $\rho \downarrow$ | $B$, $m \approx \sqrt{D}$, $d_{\max}$ |
| AdaBoost.M1 | $\mathcal{O}(M D N \log N)$ | Sequential in $M$ | Bias $\downarrow$ (margin) | $M$, stump/tree depth |
| Hist-GBDT | $\mathcal{O}(M (N D_{\mathrm{sub}} + 2^{d} K_{\mathrm{bin}} D_{\mathrm{sub}}))$ | Feature/bin-level | Bias $\downarrow$, reg. | $M$, $\nu$, $d_{\max}$, $\lambda$, $\gamma$ |

> [!info] Cross-Book Synthesis: Statistical Theory vs. Deep Learning vs. Production Systems
> - **Statistical & Probabilistic View** [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, pp. 609–620): Ensembling is *not* Bayes Model Averaging (BMA). BMA weights $\mathbb{P}(m \mid \mathcal{D})$ collapse onto the single MAP model as $N \to \infty$, whereas ensemble weights expand the hypothesis class so diverse trees jointly approximate smooth, non-axis-aligned decision boundaries.
> - **Deep Learning Regularization View:** While bagging is a universal variance-reduction principle [[Source - Deep Learning|deep_learning_goodfellow_bengio_courville_2017]] (Sec. 7.11, pp. 256–257), deep neural networks rarely benefit from bootstrap resampling because seeing only $63.2\%$ of unique examples degrades representation learning; instead, neural ensembles rely on random initialization and minibatch stochasticity for diversity.
> - **Production ML Systems View** [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Sec. 6.1, pp. 150–161): Gradient-boosted trees (XGBoost, LightGBM) dominate structured tabular workloads and strict-latency CPU serving, outperforming deep networks on small-to-medium tabular data and handling class imbalance effectively. However, large multi-model ensembles increase deployment complexity and are reserved for high-ROI tasks (e.g., ad click-through rate prediction) where marginal AUC gains translate directly to revenue.

## Visual Diagnostics & Geometric Structure

Figure `fig:trees:arch` contrasts the geometric partition of a single CART tree with the parallel variance-reduction topology of Bagging/Random Forests and the sequential functional residual descent of Gradient Boosting. Figure `fig:trees:curves` quantifies the two governing theoretical curves: the $\rho\sigma^2$ correlation floor in Eq. `(eq:trees:bagging_var)` and the binary node impurity profiles in Eqs. `(eq:trees:gini)`–`(eq:trees:entropy)`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Geometric and architectural duality of tree ensembles: (a) recursive axis-aligned splits partition $\mathbb{R}^2$ into constant-weight rectangles; (b) Bagging/Random Forests average parallel decorrelated trees to reduce variance, whereas Gradient Boosting sequentially fits trees to negative gradients and Hessians to reduce bias.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!example] Visual Diagram (LaTeX / PDF)
> *Quantitative ensemble and splitting laws: (a) ensemble variance $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ plateaus at $\rho\sigma^2$, explaining why Random Forest feature subsampling outperforms pure Bagging; (b) Gini and scaled Shannon entropy are strictly concave and differentiable on $(0,1)$, favoring purer child splits more aggressively than misclassification error.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

Empirical comparisons from [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18) corroborate these theoretical curves: Figure `fig:trees:rf_boundary` illustrates how averaging 50 trees smooths the jagged axis-aligned overfitting of a single CART model, Figure `fig:trees:spam_err` demonstrates the test-error hierarchy $\text{Boosting} < \text{Random Forest} < \text{Bagging}$ on the 57-feature spam dataset, and Figure `fig:trees:boost_resid` visualizes sequential residual fitting in 1D regression.

> [!note] Source Figure
> *Figure:* Decision boundaries in 2D feature space: (a) a single unpruned CART tree overfits noisy pockets ($\text{acc}=0.86$); (b–c) bagging 10 and 50 trees stabilizes the boundary ($\text{acc}=0.91$); (d) a 50-tree Random Forest ($\text{acc}=0.90$) [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, Fig. 18.4, p. 610). (`probabilisticml_kevinmurphy_introduction_p640_fig1.png`)

> [!note] Source Figure
> *Figure:* Test error vs. number of trees on the 4,601-sample email spam dataset ($D=57$): Random Forests substantially outperform Bagging by decorrelating trees across irrelevant features, while Gradient Boosting achieves the lowest test error via sequential bias reduction [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, Fig. 18.5, p. 611). (`probabilisticml_kevinmurphy_introduction_p641_fig1.png`)

> [!note] Source Figure
> *Figure:* Stagewise residual fitting with depth-2 regression trees on a 1D quadratic curve: left panels show each weak learner $F_m(x_1)$ fitted to the current residuals $y - f_{m-1}(x_1)$; right panels show the cumulative strong learner $f_m(x_1) = \sum_{k=1}^{m} F_k(x_1)$ [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, Fig. 18.6, p. 613). (`probabilisticml_kevinmurphy_introduction_p643_fig1.png`)

## Numerical Pitfalls, Edge Cases & Failure Modes

> [!warning] Axis-Aligned Staircase Extrapolation Failure
> Because every leaf region $R_{jm}$ predicts a constant $w_{jm}$, tree ensembles *cannot extrapolate* linear or monotonic trends beyond the minimum and maximum feature values observed in $\mathcal{D}$. Under covariate shift (e.g., forecasting prices or temporal growth outside historical bounds), predictions clamp flatly at boundary leaf values. **Remedy:** Detrend targets with a linear/GAM baseline prior to tree fitting, or use linear-tree hybrids when extrapolation is required.

> [!warning] High-Cardinality Categorical Split Overfitting & Target-Encoding Leakage
> Allowing arbitrary subset splits on a categorical feature with $K$ levels searches $2^{K-1}-1$ partitions, severely biasing greedy Gini/gain selection toward high-cardinality IDs. Conversely, naive out-of-fold-free target encoding leaks $y_i$ into feature values. **Remedy:** Use LightGBM's Fisher-sorted histogram categorical partitioning [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, p. 619), or compute smoothed target statistics strictly inside cross-validation folds / ordered boosting permutations.

> [!warning] Boosting Overfitting Without Shrinkage $\nu$ & Early Stopping
> Unlike Random Forests—where increasing $B$ never overfits in expectation (Eq. `(eq:trees:bagging_var)`)—unregularized boosting ($M \to \infty$ with $\nu = 1$) drives training loss to zero and memorizes label noise, especially under exponential loss which assigns $\exp(-\tilde{y}_i f(\boldsymbol{x}_i))$ penalty to outliers [[Source - Probabilistic Machine Learning|probabilisticml_kevinmurphy_introduction]] (Ch. 18, pp. 614, 617). **Remedy:** Set small shrinkage $\nu \in [0.01, 0.1]$, enable row/column subsampling ($r_{\mathrm{row}}, r_{\mathrm{col}} < 1$), and halt at the minimum validation loss via early stopping.

> [!warning] Probabilistic Calibration & OOB Shift Sensitivity
> Unconstrained Random Forests push leaf probabilities toward $0$ or $1$ or suffer bias from class-reweighting heuristics, while OOB error estimates assume i.i.d. sampling and fail under temporal or group-structured data [[Source - Designing Machine Learning Systems|chip_huyen_designing_machine_learning_systems]] (Sec. 6.1, pp. 155, 158). **Remedy:** Apply post-hoc Platt scaling or isotonic regression on a held-out calibration split, and replace OOB with time-series or group-aware cross-validation.

---

## Knowledge Graph Connections

- **Core Atomic Concepts (3):** [[CART Decision Trees]], [[Gradient Boosted Decision Trees (GBDT)]], [[Random Forests & Bagging]]
- **Prerequisites (Directed $\leftarrow$):** [[Bias-Variance Trade-off]], [[CART Decision Trees]], [[Gradient Descent]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Gradient Boosted Decision Trees (GBDT)]], [[Random Forests & Bagging]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Bias-Variance Trade-off]], [[CART Decision Trees]], [[Gradient Boosted Decision Trees (GBDT)]], [[Random Forests & Bagging]]
- **Synthesized Sources:** [[Source - Deep Learning]], [[Source - Designing Machine Learning Systems]], [[Source - Probabilistic Machine Learning]]
