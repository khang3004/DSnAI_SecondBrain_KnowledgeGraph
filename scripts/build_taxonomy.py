#!/usr/bin/env python3
"""Hierarchical taxonomy builder and validator for the AI/ML Knowledge Monograph.

Constructs a 4-level knowledge taxonomy (Domain -> Parent Topic -> Sub-Topic ->
Leaf Monograph) grounded in ``./data/book_index.json``, validates every book
slug and page range against the indexed PDFs, verifies full library coverage,
mirrors the directory skeleton inside ``./monographs/``, and writes
``./data/taxonomy_tree.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


@dataclass(slots=True)
class SourceMapping:
    """Maps a source textbook chapter/section and PDF page range to a leaf."""

    book_slug: str
    chapter_or_section: str
    page_start: int
    page_end: int
    synthesis_role: str


@dataclass(slots=True)
class LeafMonograph:
    """Represents a single compilable leaf monograph topic."""

    id: str
    title: str
    tex_path: str
    description: str
    key_visuals: list[str]
    sources: list[SourceMapping]


@dataclass(slots=True)
class SubTopicNode:
    """Third level of the taxonomy grouping cohesive leaf monographs."""

    id: str
    title: str
    dir_path: str
    leaves: list[LeafMonograph] = field(default_factory=list)


@dataclass(slots=True)
class ParentTopicNode:
    """Second level of the taxonomy representing a major discipline pillar."""

    id: str
    title: str
    dir_path: str
    sub_topics: list[SubTopicNode] = field(default_factory=list)


@dataclass(slots=True)
class DomainNode:
    """Top level of the taxonomy."""

    id: str
    title: str
    dir_path: str
    parent_topics: list[ParentTopicNode] = field(default_factory=list)


def src(
    book_slug: str,
    chapter_or_section: str,
    page_start: int,
    page_end: int,
    synthesis_role: str,
) -> SourceMapping:
    """Shorthand constructor for ``SourceMapping`` entries."""
    return SourceMapping(
        book_slug=book_slug,
        chapter_or_section=chapter_or_section,
        page_start=page_start,
        page_end=page_end,
        synthesis_role=synthesis_role,
    )


def build_taxonomy_spec(monographs_root: str = "monographs") -> list[DomainNode]:
    """Define the cross-book synthesized 4-level taxonomy across all 34 sources."""

    def leaf(
        sub_dir: str,
        leaf_id: str,
        title: str,
        description: str,
        key_visuals: list[str],
        sources: list[SourceMapping],
    ) -> LeafMonograph:
        return LeafMonograph(
            id=leaf_id,
            title=title,
            tex_path=f"{sub_dir}/{leaf_id}.tex",
            description=description,
            key_visuals=key_visuals,
            sources=sources,
        )

    # -------------------------------------------------------------------------
    # Domain 1: Mathematical Foundations for Machine Intelligence
    # -------------------------------------------------------------------------
    d1 = DomainNode(
        id="01_mathematical_foundations",
        title="Mathematical Foundations for Machine Intelligence",
        dir_path=f"{monographs_root}/01_mathematical_foundations",
    )
    d1_p1 = ParentTopicNode(
        id="01_linear_algebra_and_calculus",
        title="Linear Algebra, Matrix Decompositions & Vector Calculus",
        dir_path=f"{d1.dir_path}/01_linear_algebra_and_calculus",
    )
    d1_p1_s1 = SubTopicNode(
        id="01_vector_spaces_and_decompositions",
        title="Vector Spaces, Spectral Theory & Matrix Factorizations",
        dir_path=f"{d1_p1.dir_path}/01_vector_spaces_and_decompositions",
    )
    d1_p1_s1.leaves = [
        leaf(
            d1_p1_s1.dir_path,
            "vector_spaces_norms_and_projections",
            "Vector Spaces, Norms, Orthogonality & Least-Squares Projections",
            "Synthesizes vector subspaces, inner products, Lp/Frobenius norms, orthogonal projections, Gram-Schmidt, and Moore-Penrose pseudoinverses for overdetermined systems.",
            [
                "Extracted figure: four fundamental subspaces / orthogonal projection geometry",
                "TikZ diagram: orthogonal projection onto column space Col(A) with residual e = b - Ax",
                "PGFPlots curve: Lp unit balls (p = 1, 2, infinity) and condition-number geometry",
            ],
            [
                src("linear_algebra", "Ch. 4 Vector Spaces & Ch. 6 Orthogonality and Least Squares", 208, 257, "Rigorous four-subspace geometry and basis theory"),
                src("linear_algebra", "Ch. 6 Orthogonality, Gram-Schmidt & Least-Squares", 348, 394, "Projection matrices, QR factorization, and normal equations"),
                src("mathematics_machine_learning_orienly", "Ch. 2 Linear Algebra & Ch. 3 Analytic Geometry", 23, 101, "Compact ML-oriented treatment of inner products and orthogonal projections"),
                src("probabilisticml_kevinmurphy_introduction", "Sec. 7.1-7.3 & 7.7 Matrix Ops and Linear Systems", 259, 282, "Conditioning, numerical stability, and Cholesky/QR solvers"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Ch. 2 Linear Algebra (Norms & Pseudoinverse)", 46, 67, "Tensor notation, operator norms, and pseudoinverse"),
                src("machine_learning_co_ban_tiep_vu", "Ch. 1 Linear Algebra Review", 24, 41, "Concise matrix norms and subspace intuition"),
            ],
        ),
        leaf(
            d1_p1_s1.dir_path,
            "eigendecomposition_svd_and_low_rank",
            "Spectral Theory, Singular Value Decomposition & Low-Rank Approximations",
            "Unifies eigendecomposition of symmetric matrices, spectral theorem, full/thin SVD, Eckart-Young-Mirsky best rank-k approximation, and data-compression interpretations.",
            [
                "Extracted figure: SVD action mapping unit circle to hyper-ellipse / principal axes",
                "TikZ diagram: thin vs. full SVD matrix block dimensions (U, Sigma, V^T)",
                "PGFPlots chart: singular-value decay spectrum and relative Frobenius reconstruction error",
            ],
            [
                src("linear_algebra", "Ch. 5 Eigenvalues/Eigenvectors & Ch. 7 Symmetric Matrices and SVD", 284, 313, "Spectral diagonalization and geometric eigen-action"),
                src("linear_algebra", "Ch. 7 Symmetric Matrices, Quadratic Forms & SVD", 412, 450, "Positive definiteness, Rayleigh quotients, and SVD"),
                src("mathematics_machine_learning_orienly", "Ch. 4 Matrix Decompositions", 104, 140, "Unifying phylogeny of Cholesky, EVD, and SVD + Eckart-Young approximation"),
                src("probabilisticml_kevinmurphy_introduction", "Sec. 7.4-7.6 EVD & SVD", 283, 295, "Spectral properties of covariance matrices and truncated SVD"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Sec. 2.7-2.8 EVD & SVD", 57, 61, "Principal directions and pseudoinverse via SVD"),
                src("machine_learning_co_ban_tiep_vu", "Ch. 17 Singular Value Decomposition", 258, 265, "Worked low-rank approximation examples"),
            ],
        ),
    ]
    d1_p1_s2 = SubTopicNode(
        id="02_matrix_calculus_and_autodiff",
        title="Matrix Calculus, Jacobians & Automatic Differentiation",
        dir_path=f"{d1_p1.dir_path}/02_matrix_calculus_and_autodiff",
    )
    d1_p1_s2.leaves = [
        leaf(
            d1_p1_s2.dir_path,
            "matrix_calculus_and_automatic_differentiation",
            "Matrix Calculus, Computational Graphs & Automatic Differentiation",
            "Reconciles numerator vs. denominator layout conventions across books, derives Jacobian/Hessian chain rules, and contrasts forward-mode (JVP) vs. reverse-mode (VJP / backpropagation) AD.",
            [
                "Extracted figure: computational graph forward/backward pass diagram",
                "TikZ flowchart: forward-mode JVP vs. reverse-mode VJP computational complexity graph",
                "TikZ matrix table: numerator vs. denominator layout convention Rosetta Stone",
            ],
            [
                src("mathematics_machine_learning_orienly", "Ch. 5 Vector Calculus & Automatic Differentiation", 145, 175, "Rigorous Jacobian/matrix derivatives and backpropagation graph"),
                src("probabilisticml_kevinmurphy_introduction", "Sec. 7.8 Matrix Calculus & Sec. 13.3 Backpropagation", 299, 303, "Trace-trick matrix identities and gradient derivations"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Sec. 6.2 Automatic Differentiation", 295, 304, "Forward-mode vs. reverse-mode AD, VJPs/JVPs, and checkpointing"),
                src("deep_learning_foundations_concepts", "Ch. 8 Backpropagation & Automatic Differentiation", 250, 266, "Jacobian chaining and reverse-mode complexity analysis"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Sec. 6.5 Back-Propagation and Other Differentiation Algorithms", 219, 229, "Symbol-to-symbol vs. symbol-to-number computational graphs"),
                src("machine_learning_co_ban_tiep_vu", "Ch. 2 Matrix Calculus", 42, 51, "Denominator-layout matrix derivative lookup derivations"),
            ],
        ),
    ]
    d1_p1.sub_topics = [d1_p1_s1, d1_p1_s2]

    d1_p2 = ParentTopicNode(
        id="02_probability_information_and_transport",
        title="Probability Theory, Information Theory & Optimal Transport",
        dir_path=f"{d1.dir_path}/02_probability_information_and_transport",
    )
    d1_p2_s1 = SubTopicNode(
        id="01_probability_and_distributions",
        title="Probability Spaces, Gaussians & Statistical Estimation",
        dir_path=f"{d1_p2.dir_path}/01_probability_and_distributions",
    )
    d1_p2_s1.leaves = [
        leaf(
            d1_p2_s1.dir_path,
            "probability_spaces_gaussians_and_exponential_families",
            "Probability Spaces, Multivariate Gaussians & Exponential Families",
            "Bridges Kolmogorov axioms, change-of-variables Jacobian densities, conditioning/marginalization of multivariate Gaussians (Schur complement), and canonical exponential-family sufficient statistics.",
            [
                "Extracted figure: multivariate Gaussian covariance contours & conditional slicing",
                "TikZ diagram: exponential family natural parameter <-> mean parameter duality",
                "PGFPlots chart: 1D/2D Gaussian densities and Student-t heavy-tail robustness",
            ],
            [
                src("introduction_probability_data_science", "Ch. 2-5 Probability Spaces, RVs & Multivariate Gaussians", 59, 318, "Measure-theoretic intuition, CDF/PDF transforms, and Gaussian algebra"),
                src("deep_learning_foundations_concepts", "Ch. 2-3 Probabilities & Exponential Family", 43, 123, "Sum/product rules, Gaussian conditioning, and exponential family"),
                src("mathematics_machine_learning_orienly", "Ch. 6 Probability, Conjugacy & Change of Variables", 178, 226, "Density transformation theorem and conjugate priors"),
                src("probabilisticml_kevinmurphy_introduction", "Ch. 2-3 Univariate & Multivariate Probability", 63, 136, "Schur complement Gaussian conditioning and precision matrices"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 2 Probability", 39, 96, "Measure-theoretic foundations, copulas, and exponential families"),
            ],
        ),
        leaf(
            d1_p2_s1.dir_path,
            "asymptotics_concentration_and_estimation",
            "Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE/MAP)",
            "Synthesizes Markov/Chebyshev/Hoeffding/Chernoff concentration bounds, LLN and CLT asymptotics, Fisher information, Cramér-Rao lower bound, and MLE vs. MAP vs. Bayesian decision theory.",
            [
                "Extracted figure: concentration bound tail decay / MLE vs. MAP posterior shrinkage",
                "PGFPlots chart: tail decay comparison (Markov vs. Chebyshev vs. Hoeffding)",
                "TikZ table: frequentist MLE vs. Bayesian MAP vs. full posterior predictive",
            ],
            [
                src("introduction_probability_data_science", "Ch. 6 Sample Statistics & Ch. 8 Estimation", 335, 396, "MGFs, probability inequalities, LLN, and CLT"),
                src("introduction_probability_data_science", "Ch. 8 MLE, MAP & MMSE Estimation", 481, 549, "Consistency, Fisher information, and Bayesian MMSE"),
                src("probabilisticml_kevinmurphy_introduction", "Ch. 4 Statistics & Ch. 5 Decision Theory", 137, 236, "MLE, MAP, empirical Bayes, and risk minimization"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 3 Statistics", 97, 176, "Asymptotic normality, Cramér-Rao, and hypothesis testing"),
                src("machine_learning_co_ban_tiep_vu", "Ch. 3-4 Probability & MLE/MAP", 52, 74, "Direct MLE/MAP derivations for regression and classification"),
            ],
        ),
    ]
    d1_p2_s2 = SubTopicNode(
        id="02_information_theory_and_optimal_transport",
        title="Information Theory, Divergences & Optimal Transport",
        dir_path=f"{d1_p2.dir_path}/02_information_theory_and_optimal_transport",
    )
    d1_p2_s2.leaves = [
        leaf(
            d1_p2_s2.dir_path,
            "entropy_kl_divergence_and_information_theory",
            "Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck",
            "Derives entropy, cross-entropy, forward vs. reverse KL asymmetry (mode-covering vs. mode-seeking), mutual information, data-processing inequality, Jensen-Shannon divergence, and the Information Bottleneck.",
            [
                "Extracted figure: forward KL (zero-avoiding) vs. reverse KL (zero-forcing) Gaussian fits",
                "TikZ Venn diagram: H(X), H(Y), H(X,Y), H(X|Y), and mutual information I(X;Y)",
                "PGFPlots plot: binary entropy curve and f-divergence generator functions",
            ],
            [
                src("probabilisticml_kevinmurphy_introduction", "Ch. 6 Information Theory", 237, 258, "Entropy, cross-entropy, and forward vs. reverse KL intuition"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 5 Information Theory", 253, 294, "f-divergences, Bregman divergences, and Information Bottleneck"),
                src("deep_learning_foundations_concepts", "Sec. 2.5 Information Theory", 66, 73, "Differential entropy and KL convexity"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Sec. 3.13 Information Theory", 74, 75, "Cross-entropy loss equivalence to KL minimization"),
                src("understandingdl_simonprince", "Appendix C.5 Distances Between Distributions", 473, 475, "Concise KL and JS divergence reference"),
            ],
        ),
        leaf(
            d1_p2_s2.dir_path,
            "optimal_transport_wasserstein_and_sinkhorn",
            "Monge-Kantorovich Optimal Transport, Wasserstein Metrics & Sinkhorn Scaling",
            "Unifies Monge maps, Kantorovich LP relaxation, dual Kantorovich-Rubinstein potential form, entropic regularization, Sinkhorn-Knopp matrix scaling complexity, and minibatch/sliced OT.",
            [
                "Extracted figure: Monge non-existence vs. Kantorovich coupling matrix / Sinkhorn plan",
                "TikZ diagram: Monge map vs. Kantorovich transport plan vs. Entropic Sinkhorn coupling",
                "PGFPlots curve: Wasserstein vs. JS divergence continuity on disjoint 1D Diracs",
            ],
            [
                src("optimal_transport", "Foundations, Monge, Kantorovich, Entropic OT & Sinkhorn", 1, 40, "Core OT formulations, Sinkhorn algorithm, and complexity bounds"),
                src("optimal_transport", "Minibatch OT, Partial OT & Sliced OT", 59, 97, "Scalable OT approximations and curse-of-dimensionality mitigations"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Sec. 6.8 Optimal Transport", 348, 355, "Wasserstein distance, Kantorovich duality, and Sinkhorn iterations"),
                src("theprinciple_diffusion_models", "Sec. B.6 Diffusion Models and Optimal Transport", 196, 228, "Benamou-Brenier dynamic OT and Schrödinger bridges"),
            ],
        ),
    ]
    d1_p2.sub_topics = [d1_p2_s1, d1_p2_s2]

    d1_p3 = ParentTopicNode(
        id="03_numerical_and_convex_optimization",
        title="Convex, Smooth & Stochastic Optimization",
        dir_path=f"{d1.dir_path}/03_numerical_and_convex_optimization",
    )
    d1_p3_s1 = SubTopicNode(
        id="01_convexity_and_gradient_descent",
        title="Convex Duality & Deterministic Gradient Descent Theory",
        dir_path=f"{d1_p3.dir_path}/01_convexity_and_gradient_descent",
    )
    d1_p3_s1.leaves = [
        leaf(
            d1_p3_s1.dir_path,
            "convexity_duality_and_lagrangians",
            "Convex Sets, Fenchel/Lagrange Duality & KKT Conditions",
            "Presents convex sets/functions, first/second-order convexity characterizations, Lagrangian duality, weak vs. strong duality (Slater's condition), and KKT complementary slackness.",
            [
                "Extracted figure: supporting hyperplane / duality gap / KKT active-constraint geometry",
                "TikZ geometric diagram: feasible region, objective gradient, and constraint normal cone",
                "TikZ comparison table: Primal vs. Dual formulation and Slater qualification",
            ],
            [
                src("machine_learning_co_ban_tiep_vu", "Part VII Convex Optimization & Appendix Lagrange Multipliers", 293, 338, "Convex sets, Jensen's inequality, strong duality, and KKT"),
                src("machine_learning_co_ban_tiep_vu", "Lagrange Multipliers", 391, 394, "Step-by-step KKT worked examples"),
                src("mathematics_machine_learning_orienly", "Sec. 7.2-7.3 Constrained & Convex Optimization", 239, 251, "Lagrangian dual derivations for ML"),
                src("probabilisticml_kevinmurphy_introduction", "Sec. 8.5 Constrained Optimization", 332, 337, "Projected gradient, Lagrange multipliers, and KKT summary"),
                src("deep_learning_foundations_concepts", "Appendix C Lagrange Multipliers", 628, 631, "Geometric normal-vector interpretation"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Sec. 4.4 Constrained Optimization", 108, 110, "KKT generalized Lagrangian formulation"),
            ],
        ),
        leaf(
            d1_p3_s1.dir_path,
            "gradient_descent_convergence_and_pl_landscape",
            "Gradient Descent Dynamics: Strong Convexity, Smoothness, PL Condition & Saddle Escape",
            "Derives tight convergence rates O(1/T) for L-smooth convex functions, linear rate (1 - mu/L)^T under strong convexity or the Polyak-Lojasiewicz (PL) condition, local strong convexity, and perturbed saddle-point escape.",
            [
                "Extracted figure: ill-conditioned quadratic ravine zig-zagging vs. condition number",
                "PGFPlots convergence comparison: O(1/T) vs. linear O(rho^T) vs. divergence when eta > 2/L",
                "TikZ hierarchy diagram: Strong Convexity => PL Condition => Quadratic Growth",
            ],
            [
                src("gradient_descent_unconstrained", "Full Lecture Deck: Smoothness, Strong Convexity, PL & Saddle Escape", 1, 92, "Rigorous non-asymptotic proofs for GLM, PL condition, and perturbed GD"),
                src("stochastic_gradient_decent_algorithm_explain", "Ch. 2 Smoothness/Convexity/PL & Ch. 3 Gradient Descent", 5, 18, "Unified lemma proofs for convex, strongly convex, and PL rates"),
                src("probabilisticml_kevinmurphy_introduction", "Sec. 8.1-8.3 First/Second-Order Optimization", 305, 321, "Line search, conditioning, and Newton step comparison"),
                src("mathematics_machine_learning_orienly", "Sec. 7.1 Gradient Descent", 233, 238, "Steepest descent geometry and step-size selection"),
                src("deep_learning_foundations_concepts", "Ch. 7 Gradient Descent", 226, 246, "Error surfaces, saddle points, and normalization effects"),
            ],
        ),
    ]
    d1_p3_s2 = SubTopicNode(
        id="02_stochastic_adaptive_and_proximal_methods",
        title="Stochastic, Adaptive, Proximal & Second-Order Optimizers",
        dir_path=f"{d1_p3.dir_path}/02_stochastic_adaptive_and_proximal_methods",
    )
    d1_p3_s2.leaves = [
        leaf(
            d1_p3_s2.dir_path,
            "stochastic_gradient_descent_and_variance_bounds",
            "Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods",
            "Derives finite-sum SGD convergence to an O(eta * sigma^2) noise ball, diminishing step-size schedules, expected smoothness for minibatching, subgradient methods, and proximal operator splitting (ISTA/LASSO).",
            [
                "Extracted figure: SGD noisy trajectory around minimum vs. deterministic GD",
                "PGFPlots plot: constant step-size noise floor vs. 1/sqrt(T) decaying step-size convergence",
                "TikZ diagram: proximal gradient splitting (smooth step + nonsmooth shrinkage)",
            ],
            [
                src("stochastic_gradient_decent_algorithm_explain", "Ch. 4-6 Sum of Functions, SGD & Minibatch SGD", 18, 36, "Expected smoothness and tight variance-controlled SGD theorems"),
                src("stochastic_gradient_decent_algorithm_explain", "Ch. 8-11 Nonsmooth Subgradients & Proximal GD", 39, 65, "Subdifferential calculus, proximal operators, and stochastic proximal bounds"),
                src("probabilisticml_kevinmurphy_introduction", "Sec. 8.4 SGD & Sec. 8.6 Proximal Gradient", 322, 341, "Practical SGD schedules, Polyak averaging, and L1 soft-thresholding"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Sec. 5.9 SGD & Sec. 8.1-8.3 Basic Algorithms", 166, 169, "Empirical risk vs. expected risk gradient variance"),
                src("understandingdl_simonprince", "Sec. 6.2 Stochastic Gradient Descent", 97, 99, "Minibatch stochastic trajectory intuition"),
            ],
        ),
        leaf(
            d1_p3_s2.dir_path,
            "momentum_adaptive_optimizers_and_second_order",
            "Momentum, Nesterov, Adam/AdamW, LR Schedules & Second-Order / Natural Gradient",
            "Reconciles Polyak heavy-ball vs. Nesterov accelerated gradient, derives AdaGrad/RMSProp/Adam/AdamW bias correction and decoupled weight decay, cosine warmup schedules, and Fisher/natural-gradient & L-BFGS curvature methods.",
            [
                "Extracted figure: Adam vs. SGD with momentum path on saddle/ravine landscapes",
                "PGFPlots curves: warmup + cosine decay schedule and Nesterov vs. Heavy-Ball oscillation",
                "TikZ comparison table: SGD, Momentum, NAG, RMSProp, Adam, AdamW, L-BFGS, Natural Gradient",
            ],
            [
                src("deep_learning_goodfellow_bengio_courville_2017", "Sec. 8.3-8.7 Momentum, Adaptive LR & Second-Order Methods", 290, 344, "Momentum, NAG, AdaGrad, RMSProp, Adam, Newton, Conjugate Gradient, BFGS"),
                src("understandingdl_simonprince", "Sec. 6.3-6.5 Momentum, Adam & Hyperparameters", 100, 109, "Bias-corrected moments and clear trajectory diagrams"),
                src("stochastic_gradient_decent_algorithm_explain", "Ch. 7 Stochastic Momentum", 36, 39, "Unified algebraic parameterization of stochastic momentum"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Sec. 6.3-6.4 Stochastic Optimization & Natural Gradient", 305, 320, "AdamW decoupled weight decay, Fisher information matrix, and K-FAC"),
                src("build_large_language_model_from_scratch", "Appendix D Warmup, Cosine Decay & Gradient Clipping", 754, 775, "LLM training stabilization recipes"),
                src("machine_learning_co_ban_tiep_vu", "Ch. 7 Gradient Descent Variants", 152, 167, "Momentum and NAG intuition on contour plots"),
            ],
        ),
    ]
    d1_p3.sub_topics = [d1_p3_s1, d1_p3_s2]
    d1.parent_topics = [d1_p1, d1_p2, d1_p3]

    # -------------------------------------------------------------------------
    # Domain 2: Classical, Kernel & Probabilistic Machine Learning
    # -------------------------------------------------------------------------
    d2 = DomainNode(
        id="02_classical_and_probabilistic_ml",
        title="Classical, Kernel & Probabilistic Machine Learning",
        dir_path=f"{monographs_root}/02_classical_and_probabilistic_ml",
    )
    d2_p1 = ParentTopicNode(
        id="01_supervised_learning_and_generalization",
        title="Linear Models, Generalization, Kernels & Ensembles",
        dir_path=f"{d2.dir_path}/01_supervised_learning_and_generalization",
    )
    d2_p1_s1 = SubTopicNode(
        id="01_linear_and_generalized_linear_models",
        title="Linear, Regularized & Generalized Linear Models",
        dir_path=f"{d2_p1.dir_path}/01_linear_and_generalized_linear_models",
    )
    d2_p1_s1.leaves = [
        leaf(
            d2_p1_s1.dir_path,
            "linear_logistic_regression_and_glms",
            "Linear Regression, Ridge/Lasso Regularization, Logistic Regression & GLMs",
            "Unifies OLS orthogonal projection, Ridge L2 Tikhonov duality, Lasso L1 sparsity geometry, binary/multinomial logistic cross-entropy gradients, and Exponential-Family canonical link Generalized Linear Models.",
            [
                "Extracted figure: L1 diamond vs. L2 circle constraint geometry with MSE ellipses",
                "TikZ diagram: GLM canonical link mapping (Linear predictor -> Link -> Exponential family)",
                "PGFPlots plot: regularization coefficient shrinkage paths (Ridge vs. Lasso)",
            ],
            [
                src("probabilisticml_kevinmurphy_introduction", "Ch. 9-12 LDA, Logistic, Linear Regression & GLMs", 353, 452, "Complete probabilistic unification of linear and generalized linear models"),
                src("mathematics_machine_learning_orienly", "Ch. 9 Linear Regression", 295, 320, "Geometric orthogonal-projection vs. Bayesian linear regression"),
                src("introduction_probability_data_science", "Ch. 7 Regression & Regularization", 405, 472, "Statistical OLS, Ridge, Lasso, and orthogonal projections"),
                src("speech_language_processing_aug26", "Ch. 4 Logistic Regression", 100, 131, "Multinomial softmax regression and cross-entropy gradient derivation"),
                src("machine_learning_co_ban_tiep_vu", "Ch. 5 Linear Regression & Ch. 9-10 Logistic/Softmax", 95, 102, "Worked linear least-squares derivations"),
                src("machine_learning_co_ban_tiep_vu", "Ch. 9-10 Logistic & Softmax Regression", 177, 204, "Decision boundary and softmax cross-entropy formulations"),
                src("deep_learning_foundations_concepts", "Ch. 4-5 Single-Layer Regression & Classification", 130, 184, "Generative vs. discriminative linear classifiers"),
            ],
        ),
        leaf(
            d2_p1_s1.dir_path,
            "bias_variance_capacity_and_double_descent",
            "Bias-Variance Decomposition, Overfitting, Generalization & Double Descent",
            "Derives the exact squared-error bias-variance-noise decomposition, links effective model capacity and VC/Rademacher intuition to classical U-shaped curves, and explains modern overparameterized double descent and benign overfitting.",
            [
                "Extracted figure: classic U-shaped curve vs. modern double-descent interpolation threshold",
                "PGFPlots plot: bias^2, variance, and test risk across capacity including interpolation peak",
                "TikZ conceptual diagram: underfitting, classical sweet spot, interpolation regime, overparameterized regime",
            ],
            [
                src("deep_learning_goodfellow_bengio_courville_2017", "Sec. 5.2-5.4 Capacity, Overfitting, Bias & Variance", 125, 154, "Formal bias-variance derivation and capacity-generalization fundamentals"),
                src("understandingdl_simonprince", "Ch. 8 Measuring Performance & Ch. 20 Why Does DL Work?", 132, 151, "Explicit double descent curves and lottery-ticket/flat-minima intuitions"),
                src("understandingdl_simonprince", "Ch. 20 Why Does Deep Learning Work?", 415, 433, "Overparameterization, implicit bias, and generalization geometry"),
                src("introduction_probability_data_science", "Sec. 7.3-7.4 Overfitting & Bias-Variance Trade-Off", 434, 455, "Rigorous statistical expectation proof of bias-variance"),
                src("deep_learning_foundations_concepts", "Sec. 4.3 The Bias-Variance Trade-Off", 142, 146, "Clean polynomial regression bias-variance breakdown"),
                src("machine_learning_co_ban_tiep_vu", "Ch. 6 Overfitting", 103, 110, "Cross-validation and regularization remedies"),
            ],
        ),
    ]
    d2_p1_s2 = SubTopicNode(
        id="02_kernels_svms_and_ensembles",
        title="Kernel Methods, Support Vector Machines & Tree Ensembles",
        dir_path=f"{d2_p1.dir_path}/02_kernels_svms_and_ensembles",
    )
    d2_p1_s2.leaves = [
        leaf(
            d2_p1_s2.dir_path,
            "support_vector_machines_and_mercer_kernels",
            "Support Vector Machines, Margin Duality & Mercer Kernel Methods",
            "Derives maximum-margin hard/soft SVMs via hinge loss and Lagrangian duality, characterizes support vectors via KKT complementary slackness, proves the Representer Theorem, and connects RKHS Mercer kernels.",
            [
                "Extracted figure: maximum-margin hyperplane with slack variables xi_i and support vectors",
                "TikZ geometric plot: primal margin width 2/||w|| vs. kernel lift phi(x) to linearly separable RKHS",
                "PGFPlots plot: hinge loss vs. logistic log-loss vs. 0-1 loss",
            ],
            [
                src("mathematics_machine_learning_orienly", "Ch. 12 Classification with Support Vector Machines", 376, 400, "Geometric margin derivation, primal/dual equivalence, and kernels"),
                src("probabilisticml_kevinmurphy_introduction", "Sec. 17.1 & 17.3 Mercer Kernels & SVMs", 597, 626, "Mercer's theorem, RKHS, RBF kernels, and hinge-loss connection"),
                src("machine_learning_co_ban_tiep_vu", "Part VIII Hard/Soft-Margin & Kernel SVM", 339, 390, "Complete algebraic dual derivations and multi-class SVM extensions"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Sec. 5.7.2 Support Vector Machines", 155, 165, "Kernel trick computational scaling vs. neural feature learning"),
            ],
        ),
        leaf(
            d2_p1_s2.dir_path,
            "trees_bagging_random_forests_and_boosting",
            "Decision Trees, Bagging, Random Forests & Gradient Boosting",
            "Covers greedy CART splitting (Gini/entropy/MSE), variance reduction via bootstrap aggregating and feature-subsampled Random Forests, AdaBoost exponential loss, and functional gradient boosting (XGBoost/LightGBM).",
            [
                "Extracted figure: axis-aligned decision tree partition vs. ensemble decision boundary",
                "TikZ flowchart: bagging parallel variance reduction vs. boosting sequential residual fitting",
                "PGFPlots comparison: error decomposition across tree depth and boosting iterations",
            ],
            [
                src("probabilisticml_kevinmurphy_introduction", "Ch. 18 Trees, Forests, Bagging, and Boosting", 633, 654, "CART, Random Forests, AdaBoost, Gradient Boosting, and SHAP"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Sec. 7.11 Bagging and Other Ensemble Methods", 271, 272, "Theoretical ensemble variance reduction proof"),
                src("chip_huyen_designing_machine_learning_systems", "Sec. 6.1 Model Development and Offline Evaluation", 170, 197, "Tabular ensemble dominance, calibration, and production trade-offs"),
            ],
        ),
    ]
    d2_p1.sub_topics = [d2_p1_s1, d2_p1_s2]

    d2_p2 = ParentTopicNode(
        id="02_unsupervised_and_bayesian_inference",
        title="Latent Variables, Graphical Models, VI & MCMC Sampling",
        dir_path=f"{d2.dir_path}/02_unsupervised_and_bayesian_inference",
    )
    d2_p2_s1 = SubTopicNode(
        id="01_dimensionality_reduction_and_clustering",
        title="PCA, Factor Analysis, Mixture Models & EM",
        dir_path=f"{d2_p2.dir_path}/01_dimensionality_reduction_and_clustering",
    )
    d2_p2_s1.leaves = [
        leaf(
            d2_p2_s1.dir_path,
            "pca_gmm_and_expectation_maximization",
            "PCA, Gaussian Mixture Models & the Expectation-Maximization Algorithm",
            "Unifies maximum-variance and minimum-reconstruction PCA, probabilistic PCA/Factor Analysis, K-means as hard-EM, Gaussian Mixture Models, and the general E-step/M-step monotonic ELBO proof.",
            [
                "Extracted figure: PCA projection vs. GMM soft responsibilities / EM log-likelihood ascent",
                "TikZ diagram: E-step (tighten ELBO at theta_t) and M-step (maximize lower bound)",
                "PGFPlots curve: monotonic EM lower-bound climb toward log p(X|theta)",
            ],
            [
                src("mathematics_machine_learning_orienly", "Ch. 10 PCA & Ch. 11 GMM/EM", 323, 373, "Parallel variance/projection PCA proofs and full GMM EM equations"),
                src("deep_learning_foundations_concepts", "Ch. 15-16 Discrete & Continuous Latent Variables, EM & PCA", 474, 540, "K-means, GMMs, probabilistic PCA, and ELBO derivation of EM"),
                src("probabilisticml_kevinmurphy_introduction", "Ch. 20.1-20.2 PCA/FA & Ch. 21 Clustering", 687, 708, "SVD-based PCA and factor analysis identifiability"),
                src("probabilisticml_kevinmurphy_introduction", "Ch. 21 Clustering & Mixture Models", 745, 766, "K-means, GMMs, and spectral clustering"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 28 Latent Factor Models", 963, 1012, "Mixture models, factor analysis, LDA, and ICA"),
                src("machine_learning_co_ban_tiep_vu", "Ch. 13 K-means & Ch. 18-19 PCA/LDA", 266, 292, "PCA vs. Fisher LDA projection geometry"),
            ],
        ),
    ]
    d2_p2_s2 = SubTopicNode(
        id="02_bayesian_inference_and_sampling",
        title="Graphical Models, Gaussian Processes, Variational Inference & MCMC",
        dir_path=f"{d2_p2.dir_path}/02_bayesian_inference_and_sampling",
    )
    d2_p2_s2.leaves = [
        leaf(
            d2_p2_s2.dir_path,
            "graphical_models_and_gaussian_processes",
            "Probabilistic Graphical Models, Message Passing & Gaussian Processes",
            "Covers directed DAGs (d-separation) vs. undirected MRFs (Hammersley-Clifford), belief propagation/sum-product message passing, and non-parametric Gaussian Process regression, marginal likelihood, and sparse inducing-point scaling.",
            [
                "Extracted figure: d-separation Bayes-ball patterns & GP posterior predictive uncertainty bands",
                "TikZ graph: Bayesian network vs. factor graph message-passing update equations",
                "PGFPlots curve: GP prior draws and posterior mean +/- 2 sigma credible intervals",
            ],
            [
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 4 Graphical Models & Ch. 9 Message Passing", 177, 252, "DAGs, MRFs, conditional independence, and factor graphs"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 9 Message Passing & Ch. 18 Gaussian Processes", 715, 768, "GP kernels, closed-form inference, and Nyström/inducing-point sparse GPs"),
                src("deep_learning_foundations_concepts", "Ch. 11 Structured Distributions", 341, 368, "D-separation, Markov blankets, and factor graphs"),
                src("probabilisticml_kevinmurphy_introduction", "Sec. 17.2 Gaussian Processes", 604, 614, "Introductory GP posterior prediction"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Ch. 16 Structured Probabilistic Models", 573, 604, "Partition functions, energy factorizations, and deep graphical models"),
            ],
        ),
        leaf(
            d2_p2_s2.dir_path,
            "variational_inference_and_elbo",
            "Variational Inference, Evidence Lower Bound (ELBO) & Expectation Propagation",
            "Derives log p(x) = ELBO(q) + KL(q || p), mean-field coordinate ascent (CAVI), black-box score-function (REINFORCE) vs. pathwise reparameterized gradients, tighter IWAE bounds, and Expectation Propagation.",
            [
                "Extracted figure: mean-field factorized posterior vs. correlated true Gaussian posterior",
                "TikZ equation decomposition: log p(x) = ELBO + KL(q||p) bar/balance visualization",
                "TikZ comparison table: CAVI vs. BBVI (Score Function) vs. Pathwise Reparameterization vs. EP",
            ],
            [
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 10 Variational Inference", 473, 516, "CAVI, black-box VI, pathwise gradients, IWAE, and Expectation Propagation"),
                src("deep_learning_foundations_concepts", "Sec. 15.4 & 16.3 Evidence Lower Bound", 500, 535, "Clean ELBO decomposition for discrete and continuous latents"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Ch. 19 Approximate Inference", 646, 668, "Calculus of variations mean-field derivation and learned inference"),
                src("understandingdl_simonprince", "Sec. 17.4-17.6 ELBO & Variational Approximation", 347, 351, "Concise geometric ELBO optimization view"),
            ],
        ),
        leaf(
            d2_p2_s2.dir_path,
            "monte_carlo_mcmc_and_hmc_sampling",
            "Monte Carlo Integration, Metropolis-Hastings, Gibbs, HMC & Langevin Sampling",
            "Synthesizes importance sampling and variance reduction (control variates), ergodic Markov chains and detailed balance, Metropolis-Hastings, Gibbs sampling, Hamiltonian Monte Carlo (leapfrog symplectic dynamics), and ULA/SGLD Langevin sampling.",
            [
                "Extracted figure: random-walk MH vs. HMC Hamiltonian phase-space trajectory",
                "TikZ flowchart: proposal-accept/reject loop in Metropolis-Hastings vs. HMC leapfrog integrator",
                "PGFPlots trace: random-walk autocorrelation vs. HMC mixing efficiency in high dimensions",
            ],
            [
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 11-13 Monte Carlo, MCMC, HMC & SMC", 517, 606, "Comprehensive MCMC, HMC, NUTS, SGMCMC, and sequential Monte Carlo"),
                src("deep_learning_foundations_concepts", "Ch. 14 Sampling, MCMC & Langevin Sampling", 444, 470, "Rejection/importance sampling, detailed balance, and Langevin SDEs"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Ch. 17 Monte Carlo Methods", 605, 619, "Mode-mixing bottlenecks and tempered transitions"),
                src("diffusionmodels_stanleychan", "Sec. 3.1 Langevin Dynamics", 30, 32, "Score-driven Langevin sampling intuition"),
            ],
        ),
    ]
    d2_p2.sub_topics = [d2_p2_s1, d2_p2_s2]
    d2.parent_topics = [d2_p1, d2_p2]

    # -------------------------------------------------------------------------
    # Domain 3: Neural Architectures, Representations & Regularization
    # -------------------------------------------------------------------------
    d3 = DomainNode(
        id="03_deep_learning_architectures",
        title="Neural Architectures, Representations & Regularization",
        dir_path=f"{monographs_root}/03_deep_learning_architectures",
    )
    d3_p1 = ParentTopicNode(
        id="01_feedforward_normalization_and_kan",
        title="MLPs, Kolmogorov-Arnold Networks, Residuals & Regularization",
        dir_path=f"{d3.dir_path}/01_feedforward_normalization_and_kan",
    )
    d3_p1_s1 = SubTopicNode(
        id="01_feedforward_and_training_dynamics",
        title="Feedforward Networks, Normalization, Residuals & KANs",
        dir_path=f"{d3_p1.dir_path}/01_feedforward_and_training_dynamics",
    )
    d3_p1_s1.leaves = [
        leaf(
            d3_p1_s1.dir_path,
            "mlp_universal_approximation_and_kan",
            "Feedforward Networks, Universal Approximation & Kolmogorov-Arnold Networks (KAN)",
            "Connects piecewise-linear ReLU polyhedral regions and the Cybenko/Hornik Universal Approximation Theorem with Kolmogorov-Arnold superposition and learnable B-spline edge activations in KANs.",
            [
                "Extracted figure: piecewise linear regions of shallow/deep ReLU networks & KAN spline edges",
                "TikZ comparison architecture: MLP (fixed node activations, linear edges) vs. KAN (learnable spline edges, node summation)",
                "PGFPlots curves: cubic B-spline basis functions and ReLU/GELU/SwiGLU activations",
            ],
            [
                src("understandingdl_simonprince", "Ch. 3-4 Shallow & Deep Neural Networks", 39, 69, "Piecewise-linear region counting and exponential depth efficiency"),
                src("deep_learning_foundations_concepts", "Ch. 6 Deep Neural Networks", 189, 221, "Basis-function limits, universal approximation, and error functions"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Ch. 6 Deep Feedforward Networks", 183, 224, "XOR, hidden units, depth-width trade-offs, and architecture design"),
                src("math_behind_kan", "Full Monograph: Universal Approximation vs. KA Theorem & B-Splines", 1, 9, "Kolmogorov-Arnold theorem, Cox-de Boor B-spline recursion, and KAN layers"),
                src("probabilisticml_kevinmurphy_introduction", "Sec. 13.2 & 13.6 MLPs & Feedforward Variants", 456, 467, "Matrix parameterization and activation families"),
            ],
        ),
        leaf(
            d3_p1_s1.dir_path,
            "normalization_residuals_initialization_and_regularization",
            "Initialization, Batch/LayerNorm, Residual Connections, Weight Decay & Dropout",
            "Synthesizes Xavier/He variance-preserving initialization, BatchNorm vs. LayerNorm vs. RMSNorm gradient conditioning, ResNet highway gradient flow and ensemble-of-paths view, L2 weight decay, and inverted dropout.",
            [
                "Extracted figure: shattered gradients vs. ResNet skip-connection gradient covariance",
                "TikZ diagram: Pre-LN vs. Post-LN residual block & BatchNorm vs. LayerNorm tensor axes",
                "PGFPlots chart: activation variance propagation across 50 layers (poor init vs. He init + Norm)",
            ],
            [
                src("understandingdl_simonprince", "Sec. 7.4 Init, Ch. 9 Regularization & Ch. 11 Residual Networks", 152, 220, "He/Xavier initialization, implicit/explicit regularization, BatchNorm, and ResNets"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Ch. 7 Regularization for Deep Learning", 243, 288, "Norm penalties, noise robustness, early stopping, and dropout"),
                src("deep_learning_foundations_concepts", "Sec. 7.4 Normalization & Ch. 9 Regularization/Residuals", 241, 297, "Covariate shift, LayerNorm/BatchNorm, and residual inductive bias"),
                src("build_large_language_model_from_scratch", "Sec. 4.2 & 4.4 LayerNorm & Shortcut Connections", 257, 288, "Exact Pre-LN Transformer LayerNorm and skip-connection gradient behavior"),
            ],
        ),
    ]
    d3_p1.sub_topics = [d3_p1_s1]

    d3_p2 = ParentTopicNode(
        id="02_spatial_graph_and_sequence_models",
        title="Convolutional, Graph, Recurrent & State-Space Architectures",
        dir_path=f"{d3.dir_path}/02_spatial_graph_and_sequence_models",
    )
    d3_p2_s1 = SubTopicNode(
        id="01_convolutional_and_graph_networks",
        title="Convolutional Networks & Graph Neural Networks",
        dir_path=f"{d3_p2.dir_path}/01_convolutional_and_graph_networks",
    )
    d3_p2_s1.leaves = [
        leaf(
            d3_p2_s1.dir_path,
            "convolutional_networks_equivariance_and_vision",
            "Convolutional Neural Networks, Translation Equivariance & Vision Backbones",
            "Formalizes discrete/1D/2D convolution vs. cross-correlation, sparse Toeplitz weight sharing, translation equivariance vs. pooling invariance, receptive field growth, dilated/depthwise-separable convolutions, and U-Net/ResNet vision designs.",
            [
                "Extracted figure: sparse connectivity / receptive field / U-Net encoder-decoder skip topology",
                "TikZ diagram: Toeplitz matrix equivalence and equivariance commutative square",
                "TikZ comparison table: Standard Conv vs. Depthwise-Separable vs. Dilated vs. Transposed Conv",
            ],
            [
                src("deep_learning_goodfellow_bengio_courville_2017", "Ch. 9 Convolutional Networks", 345, 387, "Sparse interactions, parameter sharing, equivariance, and infinitely strong priors"),
                src("deep_learning_foundations_concepts", "Ch. 10 Convolutional Networks", 303, 337, "2D filters, pooling, object detection, and U-Net segmentation"),
                src("understandingdl_simonprince", "Ch. 10 Convolutional Networks", 175, 199, "1D/2D convolutions, downsampling/upsampling, and invariance"),
                src("probabilisticml_kevinmurphy_introduction", "Ch. 14 Neural Networks for Images", 497, 532, "ResNet, DenseNet, depthwise separable convs, and vision tasks"),
            ],
        ),
        leaf(
            d3_p2_s1.dir_path,
            "graph_neural_networks_and_message_passing",
            "Graph Neural Networks, Message Passing & Graph Convolutional Architectures",
            "Derives permutation equivariance/invariance on irregular graphs, spectral graph Laplacians vs. spatial message passing (GCN, GraphSAGE, GAT), neighborhood aggregation, and the over-smoothing bottleneck.",
            [
                "Extracted figure: graph message aggregation and 2-hop receptive field neighborhood",
                "TikZ diagram: permutation-equivariant MPNN update & readout pipeline",
                "TikZ comparison matrix: GCN vs. GraphSAGE vs. GAT aggregation & normalization",
            ],
            [
                src("deep_learning_foundations_concepts", "Ch. 13 Graph Neural Networks", 423, 440, "Permutation equivariance, adjacency operators, and general MPNNs"),
                src("understandingdl_simonprince", "Ch. 13 Graph Neural Networks", 254, 281, "Graph representations, inductive vs. transductive GCNs, and edge graphs"),
                src("probabilisticml_kevinmurphy_introduction", "Ch. 23 Graph Embeddings", 783, 802, "Random-walk graph embeddings and GNN connections"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 30 Graph Learning", 1077, 1080, "Latent variable graph inference"),
            ],
        ),
    ]
    d3_p2_s2 = SubTopicNode(
        id="02_sequence_models_and_state_spaces",
        title="Recurrent Networks & Selective State-Space Models (Mamba)",
        dir_path=f"{d3_p2.dir_path}/02_sequence_models_and_state_spaces",
    )
    d3_p2_s2.leaves = [
        leaf(
            d3_p2_s2.dir_path,
            "rnns_lstms_and_vanishing_gradients",
            "Recurrent Neural Networks, BPTT, LSTMs/GRUs & Long-Range Gradient Flow",
            "Derives unfolded recurrent computation graphs, Backpropagation Through Time (BPTT), spectral-radius causes of vanishing/exploding gradients, gradient clipping, and constant-error-carousel gating in LSTMs and GRUs.",
            [
                "Extracted figure: unfolded RNN computational graph & LSTM cell gating structure",
                "TikZ diagram: LSTM forget/input/output gate equations and additive cell-state highway",
                "PGFPlots chart: eigenvalue-power decay lambda^T across sequence length T",
            ],
            [
                src("deep_learning_goodfellow_bengio_courville_2017", "Ch. 10 Sequence Modeling: Recurrent and Recursive Nets", 388, 435, "Unfolding, BPTT, vanishing/exploding gradient analysis, and LSTM/GRU"),
                src("speech_language_processing_aug26", "Ch. 13 RNNs and LSTMs", 310, 335, "Autoregressive RNN language modeling, stacked/bidirectional RNNs, and seq2seq"),
                src("probabilisticml_kevinmurphy_introduction", "Sec. 15.2 Recurrent Neural Networks", 533, 545, "Elman networks, BPTT, and sequence classification/generation"),
            ],
        ),
        leaf(
            d3_p2_s2.dir_path,
            "state_space_models_kalman_and_mamba",
            "Classical State-Space Models, Structured SSMs & Selective Scan (Mamba)",
            "Bridges classical linear-Gaussian Kalman filtering/smoothing, continuous-to-discrete ZOH discretization of LTI state-space models (S4), convolutional-vs-recurrent duality, input-dependent selective SSMs (Mamba/S6), and hardware-aware parallel scan.",
            [
                "Extracted figure: Mamba selective SSM block / recurrent vs. convolutional dual view",
                "TikZ diagram: continuous (A,B,C) -> ZOH discrete -> Parallel Scan / Recurrent Step",
                "TikZ comparison matrix: Transformer (O(N^2) train, O(N) KV) vs. LTI S4 vs. Selective Mamba (O(N) train, O(1) state)",
            ],
            [
                src("mamba", "Full Monograph: SSMs, Convolutional Dual, Selection & Hardware Scan", 1, 15, "Discretization, S4 vs. Mamba selective scan, and complexity comparison"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 8 Gaussian Filtering & Ch. 29 State-Space Models", 393, 434, "Kalman filter/RTS smoother recursions and LDS foundation"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 29 State-Space Models & Deep SSMs", 1013, 1076, "HMMs, linear dynamical systems, switching SSMs, and deep SSMs"),
                src("foundations_large_language_models", "Sec. 2.3 Long Sequence Modeling (SSMs & Linear Attention)", 73, 100, "S4, HiPPO initialization, Mamba, and sub-quadratic sequence architectures"),
            ],
        ),
    ]
    d3_p2.sub_topics = [d3_p2_s1, d3_p2_s2]
    d3.parent_topics = [d3_p1, d3_p2]

    # -------------------------------------------------------------------------
    # Domain 4: Deep Generative Modeling & Optimal Transport
    # -------------------------------------------------------------------------
    d4 = DomainNode(
        id="04_generative_models_and_optimal_transport",
        title="Deep Generative Modeling: VAEs, Flows, EBMs, GANs & Diffusion",
        dir_path=f"{monographs_root}/04_generative_models_and_optimal_transport",
    )
    d4_p1 = ParentTopicNode(
        id="01_latent_implicit_and_invertible_models",
        title="Autoencoders, VAEs, Normalizing Flows, EBMs & GANs",
        dir_path=f"{d4.dir_path}/01_latent_implicit_and_invertible_models",
    )
    d4_p1_s1 = SubTopicNode(
        id="01_autoencoders_vaes_and_flows",
        title="Autoencoders, Variational Autoencoders & Normalizing Flows",
        dir_path=f"{d4_p1.dir_path}/01_autoencoders_vaes_and_flows",
    )
    d4_p1_s1.leaves = [
        leaf(
            d4_p1_s1.dir_path,
            "autoencoders_and_variational_autoencoders",
            "Autoencoders, Variational Autoencoders (VAEs), Reparameterization & VQ-VAE",
            "Synthesizes undercomplete/denoising/contractive autoencoders, amortized variational inference via the VAE ELBO, pathwise reparameterization trick, posterior collapse mitigations (beta-VAE/free bits), and discrete VQ-VAE codebooks.",
            [
                "Extracted figure: VAE encoder-decoder computational graph with reparameterization z = mu + sigma * eps",
                "TikZ diagram: deterministic AE vs. VAE reparameterized bottleneck vs. VQ-VAE codebook",
                "PGFPlots chart: rate-distortion (KL vs. reconstruction) trade-off frontier",
            ],
            [
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 21 Variational Autoencoders", 825, 854, "Reparameterization, beta-VAE, posterior collapse, hierarchical VAEs, and VQ-VAE"),
                src("deep_learning_foundations_concepts", "Ch. 19 Autoencoders", 574, 590, "PCA equivalence of linear AEs and amortized VAE ELBO"),
                src("understandingdl_simonprince", "Ch. 17 Variational Autoencoders", 340, 361, "Latent manifold interpolation and ELBO geometry"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Ch. 14 Autoencoders", 517, 540, "Denoising and contractive score interpretations"),
                src("diffusionmodels_stanleychan", "Sec. 1 The Basics: Variational Auto-Encoder (VAE)", 2, 9, "Clean bridge from 1-step VAE ELBO to hierarchical diffusion ELBO"),
                src("theprinciple_diffusion_models", "Sec. B.1.1 Variational Autoencoder", 38, 47, "Variational bridge into DDPMs"),
            ],
        ),
        leaf(
            d4_p1_s1.dir_path,
            "normalizing_flows_and_continuous_flows",
            "Change-of-Variables, Coupling/Autoregressive Normalizing Flows & Neural ODEs",
            "Derives exact log-likelihood training via change-of-variables Jacobian determinants, RealNVP/Glow affine coupling, MAF/IAF triangular Jacobians, continuous normalizing flows (Neural ODEs), and Hutchinson trace estimators.",
            [
                "Extracted figure: invertible diffeomorphism warping base Gaussian to multimodal target",
                "TikZ diagram: affine coupling layer split (x_1, x_2) showing triangular Jacobian matrix",
                "TikZ taxonomy tree: exact-likelihood flows (Coupling, Autoregressive, Continuous/ODE)",
            ],
            [
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 23 Normalizing Flows", 863, 882, "Coupling, autoregressive (MAF/IAF), residual, and continuous flows"),
                src("deep_learning_foundations_concepts", "Ch. 18 Normalizing Flows", 559, 573, "Change-of-variables derivation, Glow, and instantaneous change of variables"),
                src("understandingdl_simonprince", "Ch. 16 Normalizing Flows", 317, 339, "1D-to-ND intuition and multi-scale flow architectures"),
                src("theprinciple_diffusion_models", "Sec. B.4.1 & Appendix Density Evolution", 134, 140, "Neural ODEs and continuity-equation bridge to flow matching"),
                src("theprinciple_diffusion_models", "Appendix Density Evolution: Change of Variable to Fokker-Planck", 403, 416, "Rigorous derivation of instantaneous change of variables"),
            ],
        ),
    ]
    d4_p1_s2 = SubTopicNode(
        id="02_ebms_and_adversarial_models",
        title="Energy-Based Models, Score Matching & Adversarial Networks",
        dir_path=f"{d4_p1.dir_path}/02_ebms_and_adversarial_models",
    )
    d4_p1_s2.leaves = [
        leaf(
            d4_p1_s2.dir_path,
            "energy_based_models_and_score_matching",
            "Energy-Based Models, Partition Functions, Contrastive Divergence & Score Matching",
            "Addresses intractable normalizing constants Z(theta) via MLE positive/negative phases, Contrastive Divergence (CD-k), Noise Contrastive Estimation (NCE), Hyvärinen score matching, sliced score matching, and Vincent's denoising score matching equivalence.",
            [
                "Extracted figure: energy landscape pushing down data modes and raising contrastive samples",
                "TikZ vector-field diagram: Stein score grad_x log p(x) pointing toward high-density modes",
                "TikZ equation block: Hyvärinen integration-by-parts -> Vincent DSM equivalence",
            ],
            [
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 24 Energy-Based Models", 883, 900, "EBM MLE gradient, Langevin MCMC, score matching, and NCE"),
                src("deep_learning_goodfellow_bengio_courville_2017", "Ch. 18 Partition Function & Ch. 20.1-20.4 RBMs/DBMs", 620, 645, "SML/CD, denoising score matching, and AIS partition estimation"),
                src("theprinciple_diffusion_models", "Sec. B.2 Score-Based Perspective: From EBMs to NCSN", 61, 90, "Rigorous derivations of implicit, sliced, and denoising score matching"),
                src("diffusionmodels_stanleychan", "Sec. 3.2-3.3 Stein Score & Score Matching", 33, 38, "Tweedie's formula and DSM algebraic identity"),
            ],
        ),
        leaf(
            d4_p1_s2.dir_path,
            "generative_adversarial_networks_and_wasserstein_gans",
            "GAN Game Dynamics, Mode Collapse, f-GANs & Wasserstein/OT-GANs",
            "Derives the minimax JS-divergence theorem, explains vanishing gradients on disjoint low-dimensional manifolds, introduces f-GAN variational divergence minimization, Wasserstein-1 Kantorovich-Rubinstein duality (WGAN/WGAN-GP), and minibatch/sliced OT-GANs.",
            [
                "Extracted figure: JS saturation vs. smooth WGAN critic gradient on disjoint manifolds",
                "TikZ architecture: Generator/Discriminator adversarial game vs. WGAN 1-Lipschitz critic",
                "PGFPlots chart: JS vs. W1 loss landscape across support separation parameter theta",
            ],
            [
                src("probabilisticml_kevinmurphy_advancedtopics", "Ch. 26 Generative Adversarial Networks", 927, 958, "f-GANs, WGAN, spectral norm, StyleGAN, and mode-collapse diagnostics"),
                src("understandingdl_simonprince", "Ch. 15 Generative Adversarial Networks", 289, 316, "Minimax optimal discriminator derivation and WGAN stability"),
                src("deep_learning_foundations_concepts", "Ch. 17 Generative Adversarial Networks", 546, 558, "Adversarial training objective and discriminator equilibrium"),
                src("optimal_transport", "Deep Generative Models via OT: WGAN, mOT-GAN & Sliced OT-GAN", 40, 96, "Continuity theorem, Kantorovich duality, misspecified minibatch matching, and sliced OT"),
            ],
        ),
    ]
    d4_p1.sub_topics = [d4_p1_s1, d4_p1_s2]

    d4_p2 = ParentTopicNode(
        id="02_diffusion_sdes_and_flow_matching",
        title="Diffusion Models, Score SDEs, Flow Matching & Fast Samplers",
        dir_path=f"{d4.dir_path}/02_diffusion_sdes_and_flow_matching",
    )
    d4_p2_s1 = SubTopicNode(
        id="01_foundations_of_diffusion_and_sdes",
        title="DDPM, NCSN, Score SDEs & Flow Matching",
        dir_path=f"{d4_p2.dir_path}/01_foundations_of_diffusion_and_sdes",
    )
    d4_p2_s1.leaves = [
        leaf(
            d4_p2_s1.dir_path,
            "ddpm_and_score_based_generative_models",
            "Denoising Diffusion Probabilistic Models (DDPM) & Noise-Conditional Score Networks",
            "Derives the forward Gaussian Markov chain, closed-form marginal q(x_t|x_0), reverse posterior q(x_{t-1}|x_t,x_0), variational ELBO decomposition, epsilon-prediction vs. x_0-prediction vs. v-prediction parameterization, and Tweedie equivalence with multi-scale NCSN.",
            [
                "Extracted figure: forward noising and reverse denoising Markov chain / DDPM ELBO terms",
                "TikZ diagram: equivalence triangle linking x_0-pred, eps-pred, and score s_theta(x_t, t)",
                "PGFPlots curve: linear vs. cosine noise schedule alpha_bar_t and SNR(t)",
            ],
            [
                src("diffusionmodels_stanleychan", "Sec. 2 DDPM & Sec. 3 SMLD", 10, 38, "Step-by-step DDPM ELBO derivation, Tweedie's formula, and NCSN bridge"),
                src("theprinciple_diffusion_models", "Sec. B.1.2 DDPM, B.2 NCSN & B.5 Unified Lens", 48, 90, "Rigorous variational + score derivations and noise-level conditioning"),
                src("theprinciple_diffusion_models", "Sec. B.5 A Unified and Systematic Lens on Diffusion Models", 171, 195, "Parameterization equivalence (x_0, eps, v, score) and EDM roadmap"),
                src("understandingdl_simonprince", "Ch. 18 Diffusion Models", 362, 386, "Clear discrete-time encoder/decoder derivation and reparameterized loss"),
                src("deep_learning_foundations_concepts", "Ch. 20.1-20.3 Diffusion Models", 591, 608, "Forward kernel, reverse Gaussian posterior, and score matching"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Sec. 25.1-25.3 Diffusion Models", 901, 910, "DDPM and SGM unified formulation"),
            ],
        ),
        leaf(
            d4_p2_s1.dir_path,
            "score_sdes_fokker_planck_and_flow_matching",
            "Score SDEs, Itô/Girsanov/Fokker-Planck Equations & Flow Matching",
            "Unifies continuous-time VE/VP/sub-VP SDEs, Anderson's reverse-time SDE, deterministic Probability Flow ODE via Fokker-Planck, Girsanov likelihood weighting, conditional Flow Matching / Rectified Flow velocity fields, and dynamic Benamou-Brenier OT connections.",
            [
                "Extracted figure: forward SDE trajectories vs. reverse SDE and deterministic PF-ODE paths",
                "TikZ commutative diagram: Discrete DDPM/NCSN <-> Continuous Score SDE <-> PF-ODE / Flow Matching",
                "PGFPlots schematic: curved DDPM ODE trajectory vs. straightened Rectified Flow path",
            ],
            [
                src("theprinciple_diffusion_models", "Sec. B.3 Score SDE & Sec. B.4 Flow Matching", 91, 170, "Score SDE framework, VP/VE instantiations, and affine Flow Matching"),
                src("theprinciple_diffusion_models", "Appendices: ODE/SDE, Fokker-Planck & Itô-Girsanov Calculus", 387, 430, "Complete proofs of continuity equation, Fokker-Planck, Itô's lemma, and Girsanov"),
                src("diffusionmodels_stanleychan", "Sec. 4 Stochastic Differential Equation (SDE)", 39, 48, "Explicit continuous limits of DDPM and SMLD"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Sec. 25.4 Continuous-Time Diffusion Models", 911, 915, "Probability flow ODE and continuous likelihood computation"),
            ],
        ),
    ]
    d4_p2_s2 = SubTopicNode(
        id="02_controllable_sampling_and_distillation",
        title="Guidance, Fast ODE Solvers & Consistency Distillation",
        dir_path=f"{d4_p2.dir_path}/02_controllable_sampling_and_distillation",
    )
    d4_p2_s2.leaves = [
        leaf(
            d4_p2_s2.dir_path,
            "diffusion_guidance_solvers_and_consistency_distillation",
            "Classifier/Classifier-Free Guidance, DDIM/DPM-Solver ODEs & Consistency Models",
            "Covers Bayes-rule classifier guidance vs. Classifier-Free Guidance (CFG), non-Markovian DDIM, exponential integrators (DEIS, DPM-Solver/DPM-Solver++/v3), progressive distillation, Consistency Models (CD/CTM), Mean Flow, and RL/DPO diffusion alignment.",
            [
                "Extracted figure: CFG guidance scale quality-diversity trade-off & 1-step consistency mapping",
                "TikZ taxonomy tree: Fast sampling (Training-free ODE solvers vs. Distillation vs. Consistency/Flow maps)",
                "PGFPlots curve: FID vs. NFE (Number of Function Evaluations) comparing DDPM, DDIM, DPM-Solver, and CM",
            ],
            [
                src("theprinciple_diffusion_models", "Part C Sampling: Guidance & Fast Solvers (DDIM/DPM-Solver)", 229, 326, "CFG, training-free guidance, RL/DPO alignment, DDIM, DEIS, DPM-Solver++"),
                src("theprinciple_diffusion_models", "Part D Learning Fast Diffusion-Based Generators", 327, 385, "Progressive distillation, discrete/continuous consistency models, CTM, and Mean Flow"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Sec. 25.5-25.7 Speeding Up, Conditional & Discrete Diffusion", 916, 926, "DDIM acceleration, classifier-free guidance, and discrete diffusion"),
                src("deep_learning_foundations_concepts", "Sec. 20.4 Guided Diffusion", 609, 612, "Bayes decomposition for conditional score guidance"),
            ],
        ),
    ]
    d4_p2.sub_topics = [d4_p2_s1, d4_p2_s2]
    d4.parent_topics = [d4_p1, d4_p2]

    # -------------------------------------------------------------------------
    # Domain 5: Transformers, Language Modeling & LLM Engineering
    # -------------------------------------------------------------------------
    d5 = DomainNode(
        id="05_transformers_and_llms",
        title="Transformers, Language Modeling & LLM Engineering",
        dir_path=f"{monographs_root}/05_transformers_and_llms",
    )
    d5_p1 = ParentTopicNode(
        id="01_tokenization_attention_and_transformers",
        title="Tokenization, Embeddings, Attention & Transformer Architectures",
        dir_path=f"{d5.dir_path}/01_tokenization_attention_and_transformers",
    )
    d5_p1_s1 = SubTopicNode(
        id="01_tokenization_and_embeddings",
        title="Subword Tokenization & Vector Semantics",
        dir_path=f"{d5_p1.dir_path}/01_tokenization_and_embeddings",
    )
    d5_p1_s1.leaves = [
        leaf(
            d5_p1_s1.dir_path,
            "subword_tokenization_bpe_and_embeddings",
            "Words, Byte-Pair Encoding (BPE), Distributional Embeddings & Positional Encodings",
            "Synthesizes Unicode/subword tokenization, the BPE merge algorithm, sparse PPMI vs. dense skip-gram/CBOW Word2Vec embeddings, sliding-window causal dataset construction, and sinusoidal vs. learned vs. Rotary Position Embeddings (RoPE).",
            [
                "Extracted figure: BPE subword merge steps & Word2Vec skip-gram / RoPE rotation",
                "TikZ diagram: text -> UTF-8 bytes -> BPE merges -> token IDs -> embedding + RoPE",
                "TikZ comparison table: Absolute Sinusoidal vs. Learned vs. Relative vs. RoPE",
            ],
            [
                src("speech_language_processing_aug26", "Ch. 2 Words/Tokens (BPE) & Ch. 5 Embeddings (Word2Vec)", 41, 75, "Formal BPE algorithm, Unicode handling, and minimum edit distance"),
                src("speech_language_processing_aug26", "Ch. 5 Embeddings & Ch. 7.4 Position Embeddings", 135, 158, "Distributional hypothesis, cosine similarity, and skip-gram negative sampling"),
                src("build_large_language_model_from_scratch", "Ch. 2 Working with Text Data", 64, 139, "End-to-end BPE tokenization, sliding-window sampling, and positional embeddings"),
                src("probabilisticml_kevinmurphy_introduction", "Sec. 20.5 Word Embeddings", 735, 741, "Matrix factorization perspective on Word2Vec and GloVe"),
            ],
        ),
    ]
    d5_p1_s2 = SubTopicNode(
        id="02_attention_and_transformer_blocks",
        title="Attention Mechanisms & Transformer Architectures",
        dir_path=f"{d5_p1.dir_path}/02_attention_and_transformer_blocks",
    )
    d5_p1_s2.leaves = [
        leaf(
            d5_p1_s2.dir_path,
            "self_attention_multihead_and_transformer_architecture",
            "Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention & GPT/BERT Blocks",
            "Derives scaled dot-product attention (1/sqrt(d_k) variance proof), causal lower-triangular masking, Multi-Head / Grouped-Query Attention (MHA/GQA), Pre-LN residual blocks with SwiGLU/GELU MLPs, and encoder (BERT) vs. decoder (GPT) vs. encoder-decoder designs.",
            [
                "Extracted figure: QKV attention matrix workflow & full Transformer block wiring",
                "TikZ architecture diagram: Pre-LN GPT Transformer block with causal MHA + RMSNorm + SwiGLU",
                "TikZ tensor-shape trace: [B, T, d_model] -> [B, h, T, d_k] -> Attention -> [B, T, d_model]",
            ],
            [
                src("build_large_language_model_from_scratch", "Ch. 3 Coding Attention & Ch. 4 GPT Architecture", 140, 321, "Step-by-step QKV self-attention, causal masking, MHA, and GPT block"),
                src("deep_learning_foundations_concepts", "Ch. 12 Transformers", 373, 418, "Mathematical attention formulation, BERT/GPT variants, and multimodal ViT"),
                src("understandingdl_simonprince", "Ch. 12 Transformers", 221, 253, "Clean QKV matrix equations, BERT/GPT-3/ViT comparisons, and long-sequence attention"),
                src("speech_language_processing_aug26", "Ch. 7 Transformers & Ch. 9 Masked LMs", 185, 216, "Matrix-parallel attention, LM head tying, and transformer mechanics"),
                src("probabilisticml_kevinmurphy_introduction", "Sec. 15.4-15.6 Attention & Transformers", 548, 566, "Cross-attention, sparse/linear attention variants, and FlashAttention notes"),
            ],
        ),
    ]
    d5_p1.sub_topics = [d5_p1_s1, d5_p1_s2]

    d5_p2 = ParentTopicNode(
        id="02_pretraining_adaptation_and_inference",
        title="LLM Pretraining, Adaptation, Prompting & Efficient Inference",
        dir_path=f"{d5.dir_path}/02_pretraining_adaptation_and_inference",
    )
    d5_p2_s1 = SubTopicNode(
        id="01_pretraining_and_parameter_efficient_finetuning",
        title="Pretraining, Scaling Laws & Parameter-Efficient Fine-Tuning",
        dir_path=f"{d5_p2.dir_path}/01_pretraining_and_parameter_efficient_finetuning",
    )
    d5_p2_s1.leaves = [
        leaf(
            d5_p2_s1.dir_path,
            "llm_pretraining_scaling_and_peft_lora",
            "Self-Supervised Pretraining, Scaling Laws, Instruction Fine-Tuning & LoRA/QLoRA",
            "Covers causal LM cross-entropy and perplexity, MLM/denoising objectives, Kaplan vs. Chinchilla compute-optimal scaling laws, distributed parallelism (DP/TP/PP/ZeRO), SFT chat templates, and low-rank adaptation (LoRA/QLoRA).",
            [
                "Extracted figure: Chinchilla compute-optimal frontier & LoRA low-rank A*B adapter bypass",
                "TikZ diagram: W_0 + (alpha/r) * B A LoRA weight injection and QLoRA NF4 quantization flow",
                "PGFPlots curve: Chinchilla power-law loss scaling L(N, D) under fixed FLOPs budget",
            ],
            [
                src("foundations_large_language_models", "Ch. 1 Pre-training & Ch. 2 Generative LLMs / Training at Scale", 8, 72, "Pretraining objectives, scaling laws, distributed training, and data curation"),
                src("build_large_language_model_from_scratch", "Ch. 5 Pretraining, Ch. 6-7 Fine-Tuning & Appendix E LoRA", 322, 417, "Cross-entropy, perplexity, and end-to-end pretraining loop"),
                src("build_large_language_model_from_scratch", "Ch. 7 Instruction Fine-Tuning & Appendix E LoRA", 497, 602, "Alpaca-style masking, batch collation, and evaluation"),
                src("build_large_language_model_from_scratch", "Appendix E Parameter-Efficient Fine-Tuning with LoRA", 776, 835, "Exact LoRA layer parameterization and rank trade-offs"),
                src("llm_engineer_s_handbook", "Ch. 5 Supervised Fine-Tuning (SFT, LoRA, QLoRA)", 206, 250, "Data curation/decontamination, packing, LoRA/QLoRA, and gradient checkpointing"),
                src("speech_language_processing_aug26", "Ch. 3 Perplexity & Ch. 8.1-8.2 Instruction Tuning/PEFT", 76, 95, "Cross-entropy to perplexity information-theoretic bridge"),
                src("reinforcement_learning_from_human_feedback", "Ch. 4 Instruction Fine-Tuning", 33, 38, "Chat templates and loss masking for post-training SFT"),
            ],
        ),
    ]
    d5_p2_s2 = SubTopicNode(
        id="02_prompting_decoding_and_inference_scaling",
        title="Prompting, Decoding, KV Caching & Test-Time Compute",
        dir_path=f"{d5_p2.dir_path}/02_prompting_decoding_and_inference_scaling",
    )
    d5_p2_s2.leaves = [
        leaf(
            d5_p2_s2.dir_path,
            "prompting_decoding_kv_cache_and_inference_scaling",
            "Prompt Engineering, Decoding Strategies, KV Caching & Test-Time Scaling",
            "Synthesizes in-context learning, Chain-of-Thought/Tree-of-Thought prompting, greedy/beam/temperature/top-k/top-p sampling, prefill vs. decode arithmetic intensity, KV-cache memory math & PagedAttention, speculative decoding, quantization, and inference-time search/scaling.",
            [
                "Extracted figure: prefill compute-bound vs. decode memory-bandwidth-bound & KV cache layout",
                "TikZ diagram: KV-cache prefill/decode loop and speculative draft-verify pipeline",
                "PGFPlots curve: temperature/top-p entropy modulation and test-time compute scaling",
            ],
            [
                src("foundations_large_language_models", "Ch. 3 Prompting & Ch. 5 Inference", 103, 161, "ICL, CoT, ToT, soft prompting, and prompt optimization"),
                src("foundations_large_language_models", "Ch. 5 Inference: Prefilling, KV Cache & Test-Time Scaling", 210, 252, "KV caching, FlashAttention, speculative decoding, quantization, and CoT scaling"),
                src("build_large_language_model_from_scratch", "Sec. 5.3 Decoding Strategies to Control Randomness", 378, 396, "Temperature scaling and top-k/top-p logits filtering"),
                src("speech_language_processing_aug26", "Sec. 7.6 Decoding, Sec. 12.4 Beam Search & Ch. 10 Interpretability", 204, 208, "Autoregressive sampling algorithms"),
                src("speech_language_processing_aug26", "Ch. 10 Interpretability (Logit Lens, SAEs & Induction Heads)", 251, 259, "Mechanistic view of in-context learning"),
                src("reinforcement_learning_from_human_feedback", "Ch. 7 Reasoning and Inference-Time Scaling", 101, 112, "RLVR-driven reasoning and test-time compute allocation"),
            ],
        ),
    ]
    d5_p2.sub_topics = [d5_p2_s1, d5_p2_s2]
    d5.parent_topics = [d5_p1, d5_p2]

    # -------------------------------------------------------------------------
    # Domain 6: Reinforcement Learning, Control & LLM Alignment
    # -------------------------------------------------------------------------
    d6 = DomainNode(
        id="06_reinforcement_learning_and_alignment",
        title="Reinforcement Learning, Control & LLM Alignment",
        dir_path=f"{monographs_root}/06_reinforcement_learning_and_alignment",
    )
    d6_p1 = ParentTopicNode(
        id="01_tabular_and_value_based_rl",
        title="Bandits, Markov Decision Processes & Value-Based RL",
        dir_path=f"{d6.dir_path}/01_tabular_and_value_based_rl",
    )
    d6_p1_s1 = SubTopicNode(
        id="01_bandits_mdps_and_dynamic_programming",
        title="Multi-Armed Bandits, MDPs & Dynamic Programming",
        dir_path=f"{d6_p1.dir_path}/01_bandits_mdps_and_dynamic_programming",
    )
    d6_p1_s1.leaves = [
        leaf(
            d6_p1_s1.dir_path,
            "multi_armed_bandits_and_exploration",
            "Multi-Armed & Contextual Bandits, UCB, Thompson Sampling & Regret Bounds",
            "Covers exploration-exploitation trade-offs, cumulative regret, epsilon-greedy and optimistic initialization, Hoeffding-derived UCB1, Bayesian Thompson Sampling, LinUCB contextual bandits, and gradient bandit REINFORCE updates.",
            [
                "Extracted figure: 10-armed bandit testbed / UCB vs. epsilon-greedy regret curves",
                "TikZ diagram: Thompson Sampling Beta-Bernoulli posterior sampling & update loop",
                "PGFPlots plot: logarithmic O(log T) UCB regret vs. linear greedy regret",
            ],
            [
                src("reinforcement_learning_introduction_richard_s_su", "Ch. 2 Multi-Armed Bandits", 47, 68, "Incremental estimation, UCB, gradient bandits, and associative search"),
                src("reinforement_learning_kevin_murphy", "Sec. 1.2.5 Contextual Bandits & Sec. 7.1-7.2 Regret/Exploration", 173, 179, "Bayesian optimal Gittins/POMDP view, Thompson sampling, and UCB regret"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Sec. 34.4 Contextual Bandits", 1154, 1161, "LinUCB, Thompson sampling, and inverse propensity weighting"),
            ],
        ),
        leaf(
            d6_p1_s1.dir_path,
            "markov_decision_processes_and_dynamic_programming",
            "MDPs, Bellman Optimality Equations, Policy/Value Iteration & GPI",
            "Formalizes finite MDPs and POMDPs, discounted returns, Bellman expectation and optimality backup operators as gamma-contractions (Banach fixed point), iterative policy evaluation, policy iteration, value iteration, and Generalized Policy Iteration.",
            [
                "Extracted figure: agent-environment loop & backup diagrams for V^pi, Q^pi, V*, Q*",
                "TikZ diagram: Generalized Policy Iteration (GPI) converging evaluation/improvement loop",
                "TikZ backup trees: Bellman expectation vs. Bellman optimality (max) lookahead",
            ],
            [
                src("reinforcement_learning_introduction_richard_s_su", "Ch. 3 Finite MDPs & Ch. 4 Dynamic Programming", 69, 112, "Foundational MDP notation, Bellman equations, and Policy/Value Iteration"),
                src("reinforement_learning_kevin_murphy", "Sec. 1.1-1.2 MDPs/POMDPs & Sec. 2.1-2.2 Value Functions/DP", 13, 34, "Unified POMDP/MDP taxonomy and matrix Bellman operators"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Sec. 34.5-34.6 MDPs & Planning", 1162, 1170, "Contraction proofs and linear-programming / DP formulations"),
                src("understandingdl_simonprince", "Sec. 19.1-19.3 MDPs & Tabular RL", 387, 398, "Clean visual MDP transition graphs and DP backups"),
            ],
        ),
    ]
    d6_p1_s2 = SubTopicNode(
        id="02_monte_carlo_td_and_value_approximation",
        title="Monte Carlo, Temporal-Difference, DQN & Function Approximation",
        dir_path=f"{d6_p1.dir_path}/02_monte_carlo_td_and_value_approximation",
    )
    d6_p1_s2.leaves = [
        leaf(
            d6_p1_s2.dir_path,
            "mc_td_q_learning_and_deadly_triad",
            "Monte Carlo, SARSA, Q-Learning, DQN, Eligibility Traces & the Deadly Triad",
            "Contrasts MC high-variance unbiased returns against biased low-variance TD(0) bootstrapping, n-step and TD(lambda) eligibility traces, on-policy SARSA vs. off-policy Q-learning & Double Q-learning maximization bias, DQN stabilization (replay + target nets), and Baird's Deadly Triad.",
            [
                "Extracted figure: TD vs. MC vs. DP backup spectrum & Baird's counterexample divergence",
                "TikZ 2D grid: backup depth (TD(0) -> n-step -> MC) vs. sample/full expectation (DP)",
                "TikZ flowchart: DQN experience replay + frozen target network training loop",
            ],
            [
                src("reinforcement_learning_introduction_richard_s_su", "Ch. 5-7 MC, TD & n-step Bootstrapping", 113, 180, "Importance sampling MC, SARSA, Q-learning, Expected Sarsa, and Double Learning"),
                src("reinforcement_learning_introduction_richard_s_su", "Ch. 9-12 Approximation, Deadly Triad & Eligibility Traces", 219, 342, "Semi-gradient TD, Deadly Triad proof, Emphatic/Gradient-TD, and TD(lambda)"),
                src("reinforement_learning_kevin_murphy", "Sec. 2.3-2.5 MC, TD, SARSA, Q-Learning & DQN", 35, 50, "DQN extensions (Double, Dueling, PER, Rainbow) and hindsight relabeling"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Sec. 35.2 Value-Based RL", 1184, 1189, "Fitted Q-iteration and deep Q-learning"),
                src("understandingdl_simonprince", "Sec. 19.4 Fitted Q-Learning", 399, 401, "Deep Q-network stabilization summary"),
            ],
        ),
    ]
    d6_p1.sub_topics = [d6_p1_s1, d6_p1_s2]

    d6_p2 = ParentTopicNode(
        id="02_policy_gradients_mbrl_and_alignment",
        title="Policy Gradients, Model-Based RL, MARL & LLM Alignment",
        dir_path=f"{d6.dir_path}/02_policy_gradients_mbrl_and_alignment",
    )
    d6_p2_s1 = SubTopicNode(
        id="01_policy_gradients_and_model_based_rl",
        title="Policy Gradients, Actor-Critic, Model-Based RL & MARL",
        dir_path=f"{d6_p2.dir_path}/01_policy_gradients_and_model_based_rl",
    )
    d6_p2_s1.leaves = [
        leaf(
            d6_p2_s1.dir_path,
            "policy_gradient_actor_critic_trpo_and_ppo",
            "Policy Gradient Theorem, REINFORCE, GAE, Actor-Critic, TRPO/PPO & RL-as-Inference",
            "Proves the Policy Gradient Theorem, derives baseline variance reduction, Generalized Advantage Estimation (GAE(gamma, lambda)), Natural Policy Gradient / TRPO monotonic improvement lower bound, PPO clipped surrogate, SAC maximum-entropy RL, and Control-as-Inference.",
            [
                "Extracted figure: PPO clipped surrogate ratio curves (A > 0 vs. A < 0) & Actor-Critic architecture",
                "PGFPlots chart: PPO clipped objective L^CLIP vs. likelihood ratio r_t(theta)",
                "TikZ diagram: Actor-Critic + GAE advantage loop and TRPO -> PPO -> GRPO evolution",
            ],
            [
                src("reinforcement_learning_introduction_richard_s_su", "Ch. 13 Policy Gradient Methods", 343, 358, "Full proof of the Policy Gradient Theorem, REINFORCE, and Actor-Critic"),
                src("reinforement_learning_kevin_murphy", "Ch. 3 Policy-Based RL & RL as Inference", 51, 82, "REINFORCE, GAE, DDPG/TD3, TRPO, PPO, VMPO, SAC, and variational RL-as-inference"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Sec. 35.3 & 35.6 Policy-Based RL & Control as Inference", 1190, 1218, "Monotonic improvement bounds, natural gradient, and KL control"),
                src("understandingdl_simonprince", "Sec. 19.5-19.6 Policy Gradients & Actor-Critic", 402, 407, "Intuitive score-function gradient and variance reduction"),
                src("reinforcement_learning_from_human_feedback", "Sec. 6.2 Policy Gradient Algorithms for LLMs", 58, 78, "REINFORCE, RLOO, PPO, and GRPO adapted for autoregressive token generation"),
            ],
        ),
        leaf(
            d6_p2_s1.dir_path,
            "model_based_rl_mcts_world_models_and_marl",
            "Dyna, MCTS, World Models, Offline RL & Multi-Agent Reinforcement Learning",
            "Synthesizes Dyna planning/acting integration, Monte Carlo Tree Search (PUCT/AlphaZero), latent world models (Dreamer/MuZero) and compounding model error, successor representations, Offline RL conservatism (CQL/Decision Transformer), and Multi-Agent RL equilibria (Nash, CTDE, CFR).",
            [
                "Extracted figure: Dyna architecture / MCTS 4-phase loop (Select, Expand, Evaluate, Backup)",
                "TikZ diagram: MCTS PUCT tree-search cycle & CTDE multi-agent centralized-critic graph",
                "TikZ comparison table: Model-Free vs. Dyna vs. Online MCTS vs. Latent World Model",
            ],
            [
                src("reinforcement_learning_introduction_richard_s_su", "Ch. 8 Planning and Learning with Tabular Methods", 181, 213, "Dyna-Q, prioritized sweeping, trajectory sampling, and MCTS"),
                src("reinforement_learning_kevin_murphy", "Ch. 4 Model-Based RL & Ch. 5 Multi-Agent RL", 83, 146, "MCTS, MPC, Dreamer/MuZero world models, successor features, Nash/CTDE/CFR"),
                src("reinforement_learning_kevin_murphy", "Sec. 7.5-7.7 HRL, Imitation Learning & Offline RL", 183, 196, "Options, inverse RL, and offline pessimism"),
                src("probabilisticml_kevinmurphy_advancedtopics", "Sec. 35.4-35.5 Model-Based & Off-Policy RL", 1198, 1210, "Probabilistic dynamics models and off-policy importance corrections"),
            ],
        ),
    ]
    d6_p2_s2 = SubTopicNode(
        id="02_llm_alignment_rlhf_and_dpo",
        title="Post-Training Alignment: RLHF, Reward Modeling, DPO & RLVR",
        dir_path=f"{d6_p2.dir_path}/02_llm_alignment_rlhf_and_dpo",
    )
    d6_p2_s2.leaves = [
        leaf(
            d6_p2_s2.dir_path,
            "rlhf_reward_modeling_dpo_and_rlvr",
            "Bradley-Terry Reward Models, KL-Regularized RLHF, Direct Preference Optimization (DPO) & RLVR",
            "Derives the Bradley-Terry preference likelihood, outcome vs. process reward models (ORM/PRM), KL-constrained RL objective and closed-form optimal Gibbs policy pi*(y|x), the exact DPO change-of-variables derivation, GRPO/RLVR reasoning training, Constitutional AI, and Goodhart's Law reward over-optimization.",
            [
                "Extracted figure: SFT -> Reward Model -> PPO/GRPO loop vs. DPO closed-form shortcut",
                "TikZ pipeline: 3-stage RLHF vs. Direct Alignment (DPO/IPO/KTO) vs. Verifiable RLVR",
                "PGFPlots curve: Goodhart's Law divergence (proxy RM score climbs while gold score drops)",
            ],
            [
                src("reinforcement_learning_from_human_feedback", "Ch. 5-8 Reward Modeling, RL, RLVR & Direct Alignment (DPO)", 39, 124, "Bradley-Terry RM, ORM/PRM, PPO/GRPO, RLVR, and exact DPO derivation"),
                src("reinforcement_learning_from_human_feedback", "Ch. 11-14 Synthetic/AI Feedback & KL Regularization", 151, 192, "Constitutional AI, Goodhart over-optimization scaling, and KL penalties"),
                src("foundations_large_language_models", "Ch. 4 Alignment", 162, 209, "Instruction alignment, RLHF PPO math, and DPO/SimPO/KTO derivations"),
                src("reinforement_learning_kevin_murphy", "Ch. 6 LLMs and RL", 147, 172, "Single-turn vs. multi-turn RLFT, assistance games, and token-level loss"),
                src("speech_language_processing_aug26", "Sec. 8.3-8.4 Preference Alignment", 223, 230, "Clear Bradley-Terry and DPO gradient analysis"),
                src("theprinciple_diffusion_models", "Sec. C.1.5 RL to DPO for Diffusion Alignment", 248, 257, "Transfer of RLHF/DPO to diffusion reward alignment"),
            ],
        ),
    ]
    d6_p2.sub_topics = [d6_p2_s1, d6_p2_s2]
    d6.parent_topics = [d6_p1, d6_p2]

    # -------------------------------------------------------------------------
    # Domain 7: Agentic AI Architectures, RAG, Protocols & Production Systems
    # -------------------------------------------------------------------------
    d7 = DomainNode(
        id="07_agentic_ai_and_ml_systems",
        title="Agentic AI Architectures, Protocols & Production ML Systems",
        dir_path=f"{monographs_root}/07_agentic_ai_and_ml_systems",
    )
    d7_p1 = ParentTopicNode(
        id="01_agent_architectures_memory_and_protocols",
        title="Agentic Reasoning, Memory, RAG, MCP/A2A & Workflows",
        dir_path=f"{d7.dir_path}/01_agent_architectures_memory_and_protocols",
    )
    d7_p1_s1 = SubTopicNode(
        id="01_agent_reasoning_and_design_patterns",
        title="Agent Taxonomies, ReAct, Planning & Multi-Agent Patterns",
        dir_path=f"{d7_p1.dir_path}/01_agent_reasoning_and_design_patterns",
    )
    d7_p1_s1.leaves = [
        leaf(
            d7_p1_s1.dir_path,
            "agentic_architectures_react_and_planning",
            "Agent Taxonomy, ReAct Synergy, Reflection, Planning & Multi-Agent Orchestration",
            "Synthesizes agent autonomy taxonomies (Levels 0-4 / workflows vs. autonomous agents), the ReAct interleaved Thought-Action-Observation trajectory loop, Reflection/Tool-Use/Planning/Multi-Agent design patterns, manager vs. decentralized handoff topologies, and failure-mode taxonomies.",
            [
                "Extracted figure: ReAct Thought-Action-Observation trajectory vs. CoT / Act-only",
                "TikZ state-machine diagram: Orchestration loop (Think -> Act -> Observe -> Reflect/Halt)",
                "TikZ taxonomy matrix: 5 Levels of Agentic Systems & Multi-Agent Orchestration topologies",
            ],
            [
                src("react_agent", "Full Paper: ReAct Synergizing Reasoning and Acting", 1, 33, "Original ReAct formulation, HotpotQA/FEVER/ALFWorld results, and error taxonomy"),
                src("introduction_agents", "Core Agent Architecture & Taxonomy (Levels 0-4)", 6, 46, "Model/Tools/Orchestration triad, enterprise scaling, and self-evolving agents"),
                src("practical_guide_building_agents", "Agent Design Foundations & Guardrails", 4, 31, "Workflow-vs-agent criteria, single vs. multi-agent handoffs, and exit criteria"),
                src("ai_agents_2025_illustrated_guidebook", "Building Blocks, 5 Design Patterns & 5 Levels", 6, 34, "Role-playing, reflection, tool-use, planning, and multi-agent patterns"),
                src("speech_language_processing_aug26", "Sec. 1.7 & Sec. 11 Agents", 32, 32, "NLP integration perspective on tool-augmented LLM agents"),
            ],
        ),
    ]
    d7_p1_s2 = SubTopicNode(
        id="02_rag_context_memory_and_mcp",
        title="RAG, Context Engineering, Memory & Tool/Agent Protocols",
        dir_path=f"{d7_p1.dir_path}/02_rag_context_memory_and_mcp",
    )
    d7_p1_s2.leaves = [
        leaf(
            d7_p1_s2.dir_path,
            "rag_vector_databases_and_context_memory",
            "Retrieval-Augmented Generation, Vector Indexing, Context Engineering & Agent Memory",
            "Unifies sparse BM25 vs. dense bi-encoder retrieval, HNSW/IVF vector DB indexing, pre/post-retrieval reranking (query expansion, RRF, cross-encoders), dynamic Context Engineering assembly, short-term Session compaction, and long-term declarative vs. procedural Memory consolidation.",
            [
                "Extracted figure: RAG ingestion/retrieval pipeline & Context/Memory lifecycle",
                "TikZ architecture diagram: Two-stage Retrieval (BM25 + Dense HNSW -> RRF -> Cross-Encoder Rerank)",
                "TikZ taxonomy table: Session State vs. Episodic/Semantic/Procedural Memory consolidation",
            ],
            [
                src("llm_engineer_s_handbook", "Ch. 4 RAG Feature Pipeline & Vector Databases", 128, 205, "Chunking, bi-encoders, HNSW indexing, pre/post-retrieval, and feature stores"),
                src("context_engineering_sessions_memory", "Full Monograph: Context Engineering, Sessions & Memory", 6, 70, "Dynamic context assembly, session compaction, memory extraction, and retrieval"),
                src("speech_language_processing_aug26", "Ch. 11 Retrieval-Based Models (BM25, Dense IR & RAG)", 260, 282, "Formal BM25 TF-IDF scoring, bi-encoder inner product, and IR evaluation metrics"),
                src("ai_agents_2025_illustrated_guidebook", "Sec. #6 Memory & #1 Agentic RAG", 36, 41, "Agentic retrieval routing and human-like memory patterns"),
            ],
        ),
        leaf(
            d7_p1_s2.dir_path,
            "tool_use_mcp_a2a_and_workflow_automation",
            "Function Calling, Model Context Protocol (MCP), A2A Interoperability & Workflow Automation",
            "Resolves the N x M tool-integration bottleneck via MCP Host-Client-Server JSON-RPC architecture and Tools/Resources/Prompts primitives, integrates Agent2Agent (A2A) AgentCard discovery, catalogs MCP security threats (tool poisoning, confused deputy), and bridges deterministic node-graph workflows (n8n).",
            [
                "Extracted figure: N x M integration problem vs. MCP Hub / A2A interoperability diagram",
                "TikZ protocol sequence: Host <-> MCP Client <-> MCP Server (JSON-RPC initialize, list_tools, call_tool) & A2A",
                "TikZ security matrix: Prompt Injection, Tool Poisoning, Rug Pulls, and Mitigation Controls",
            ],
            [
                src("agent_tools_interoperability_mcp", "Full Whitepaper: Tool Calling, MCP Architecture & Security", 7, 51, "Tool-design best practices, JSON-RPC transports, primitives, and security mitigations"),
                src("mcp_2025_editionthe_illustrated_guidebook", "MCP Overview, Architecture & Primitives", 5, 19, "Illustrated Host/Client/Server mechanics and Tools/Resources/Prompts"),
                src("prototype_production", "Sec. A2A - Reusability and Standardization", 27, 35, "A2A vs. MCP complementarity and enterprise tool/agent registries"),
                src("reinforcement_learning_from_human_feedback", "Ch. 12 Tool Use, Function Calling & MCP", 167, 175, "Training-time tool-call interleaving and multi-step reasoning"),
                src("ultimate_n8n_starter_kit_2025", "Modules 1-5 Workflow Nodes, Error Handling & AI Agents", 1, 36, "Deterministic trigger/transform DAGs coupled with AI agent nodes"),
            ],
        ),
    ]
    d7_p1.sub_topics = [d7_p1_s1, d7_p1_s2]

    d7_p2 = ParentTopicNode(
        id="02_production_ml_systems_and_agent_ops",
        title="Production ML Systems, Evaluation, Observability & AgentOps",
        dir_path=f"{d7.dir_path}/02_production_ml_systems_and_agent_ops",
    )
    d7_p2_s1 = SubTopicNode(
        id="01_ml_systems_data_and_drift",
        title="ML Systems Design, Data Pipelines & Distribution Shifts",
        dir_path=f"{d7_p2.dir_path}/01_ml_systems_data_and_drift",
    )
    d7_p2_s1.leaves = [
        leaf(
            d7_p2_s1.dir_path,
            "ml_systems_design_data_pipelines_and_drift",
            "Iterative ML Systems Design, Feature Engineering, Leakage, Distribution Shift & Continual Learning",
            "Synthesizes iterative ML systems architecture, the FTI (Feature-Training-Inference) pipeline separation, sampling/labeling/class imbalance, data leakage diagnostics, model compression & batch-vs-online serving, covariate/label/concept shift mathematics, and continual retraining.",
            [
                "Extracted figure: FTI pipeline separation & Covariate vs. Label vs. Concept shift P(X,Y)",
                "TikZ diagram: Feature / Training / Inference decoupled pipelines with CDC feature-store sync",
                "TikZ taxonomy table: Covariate Shift P(X), Label Shift P(Y), Concept Drift P(Y|X) & Detection",
            ],
            [
                src("chip_huyen_designing_machine_learning_systems", "Ch. 1-5 Systems Design, Data & Feature Engineering", 21, 168, "ML framing, data models, sampling/imbalance, feature engineering, and data leakage"),
                src("chip_huyen_designing_machine_learning_systems", "Ch. 7-10 Deployment, Distribution Shifts & Continual Learning", 211, 350, "Online/batch prediction, compression, shift taxonomy, and MLOps infrastructure"),
                src("llm_engineer_s_handbook", "Ch. 1-3 FTI Architecture, Tooling & Data Engineering", 30, 127, "Decoupled FTI pipelines, ZenML artifact tracking, and NoSQL data warehouses"),
            ],
        ),
    ]
    d7_p2_s2 = SubTopicNode(
        id="02_agent_quality_evaluation_and_agentops",
        title="Evaluation, Observability, Guardrails & AgentOps",
        dir_path=f"{d7_p2.dir_path}/02_agent_quality_evaluation_and_agentops",
    )
    d7_p2_s2.leaves = [
        leaf(
            d7_p2_s2.dir_path,
            "agent_quality_observability_and_production_ops",
            "Outside-In Agent Evaluation, LLM-as-a-Judge, Observability, Guardrails & AgentOps CI/CD",
            "Covers non-deterministic agent evaluation hierarchies (end-to-end outcome vs. trajectory/process), deterministic metrics vs. LLM-as-a-Judge & human-in-the-loop calibration, OpenTelemetry logging/tracing/metrics pillars, defense-in-depth guardrails, canary/shadow rollouts, and the Agent Quality Flywheel.",
            [
                "Extracted figure: Outside-In evaluation hierarchy & Agent Quality Flywheel",
                "TikZ architecture: CI/CD Evaluation Gate -> Canary/Shadow Rollout -> Tracing/Metrics -> Regression Dataset",
                "TikZ comparison table: Offline Perturbation/Slicing vs. LLM-as-a-Judge vs. Online A/B & Canary",
            ],
            [
                src("agent_quality", "Full Whitepaper: Quality Pillars, Outside-In Eval & Observability", 6, 48, "Process vs. outcome judging, evaluators, logging/tracing/metrics, and quality flywheel"),
                src("prototype_production", "Journey to Production, Operations & AgentOps Lifecycle", 6, 26, "CI/CD quality gates, canary/blue-green rollouts, and production feedback loops"),
                src("prototype_production", "Putting It All Together: The AgentOps Lifecycle", 36, 39, "End-to-end operational lifecycle synthesis"),
                src("practical_guide_building_agents", "Guardrails & Human Intervention", 24, 31, "Layered classifier, relevance, safety, PII, and tool-risk guardrails"),
                src("reinforcement_learning_from_human_feedback", "Ch. 15 Evaluation & Appendix Practical Issues", 193, 202, "Evaluation variance, prompt sensitivity, and benchmark contamination"),
                src("chip_huyen_designing_machine_learning_systems", "Sec. 6.2 Offline Evaluation & Ch. 11 Responsible AI", 198, 207, "Slicing, perturbation tests, and baseline benchmarking"),
                src("chip_huyen_designing_machine_learning_systems", "Ch. 11 Responsible AI", 359, 372, "Fairness, model cards, and production accountability"),
            ],
        ),
    ]
    d7_p2.sub_topics = [d7_p2_s1, d7_p2_s2]
    d7.parent_topics = [d7_p1, d7_p2]

    return [d1, d2, d3, d4, d5, d6, d7]


def validate_and_build(index_path: Path, output_path: Path, create_dirs: bool = True) -> dict[str, Any]:
    """Validate taxonomy mappings against ``book_index.json`` and emit ``taxonomy_tree.json``."""
    if not index_path.exists():
        raise FileNotFoundError(f"Book index not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as handle:
        book_index = json.load(handle)

    books_by_slug: dict[str, dict[str, Any]] = {
        book["book_slug"]: book for book in book_index.get("books", [])
    }
    unique_slugs = {
        book["book_slug"]
        for book in book_index.get("books", [])
        if not book.get("is_duplicate", False)
    }

    domains = build_taxonomy_spec()
    referenced_slugs: set[str] = set()
    errors: list[str] = []

    parent_count = 0
    sub_count = 0
    leaf_count = 0
    mapping_count = 0

    for dom in domains:
        if create_dirs:
            Path(dom.dir_path).mkdir(parents=True, exist_ok=True)
        for parent in dom.parent_topics:
            parent_count += 1
            if create_dirs:
                Path(parent.dir_path).mkdir(parents=True, exist_ok=True)
            for sub in parent.sub_topics:
                sub_count += 1
                if create_dirs:
                    Path(sub.dir_path).mkdir(parents=True, exist_ok=True)
                for lf in sub.leaves:
                    leaf_count += 1
                    if not lf.sources:
                        errors.append(f"Leaf {lf.id} has no source mappings")
                    for sm in lf.sources:
                        mapping_count += 1
                        book = books_by_slug.get(sm.book_slug)
                        if book is None:
                            errors.append(f"Leaf {lf.id}: unknown book_slug '{sm.book_slug}'")
                            continue
                        referenced_slugs.add(sm.book_slug)
                        max_p = int(book["page_count"])
                        if not (1 <= sm.page_start <= sm.page_end <= max_p):
                            errors.append(
                                f"Leaf {lf.id}: invalid page range {sm.page_start}-{sm.page_end} "
                                f"for '{sm.book_slug}' (1..{max_p})"
                            )

    unmapped_books = sorted(unique_slugs - referenced_slugs)
    if unmapped_books:
        errors.append(f"Unmapped unique books in taxonomy: {unmapped_books}")

    if errors:
        raise ValueError("Taxonomy validation failed:\n  - " + "\n  - ".join(errors))

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "book_index_path": index_path.as_posix(),
        "summary": {
            "domains_count": len(domains),
            "parent_topics_count": parent_count,
            "sub_topics_count": sub_count,
            "leaf_monographs_count": leaf_count,
            "source_books_total": len(unique_slugs),
            "source_books_mapped": len(referenced_slugs),
            "unmapped_books": unmapped_books,
            "total_source_mappings": mapping_count,
        },
        "domains": [asdict(dom) for dom in domains],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the taxonomy builder."""
    parser = argparse.ArgumentParser(
        description="Build and validate data/taxonomy_tree.json and monographs/ directories."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/book_index.json"),
        help="Input book index JSON (default: data/book_index.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/taxonomy_tree.json"),
        help="Output taxonomy tree JSON (default: data/taxonomy_tree.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    try:
        payload = validate_and_build(args.index, args.output, create_dirs=True)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    s = payload["summary"]
    print(
        f"Built taxonomy: {s['domains_count']} domains -> {s['parent_topics_count']} parent topics -> "
        f"{s['sub_topics_count']} sub-topics -> {s['leaf_monographs_count']} leaf monographs | "
        f"books mapped: {s['source_books_mapped']}/{s['source_books_total']} "
        f"({s['total_source_mappings']} source mappings) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
