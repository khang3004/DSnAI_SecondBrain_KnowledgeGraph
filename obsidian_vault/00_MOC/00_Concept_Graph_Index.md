---
type: concept_index
part_of: "[[00_Home_MOC]]"
tags:
  - moc
  - index
---

# Concept Knowledge Graph Index

Total atomic nodes: **143** across 7 domains.

## Theorems (19)

- [[Cauchy-Schwarz Inequality]] *(D01)* — Bounds inner products by the product of induced norms; tight iff vectors are collinear.
- [[Eigendecomposition & Spectral Theorem]] *(D01)* — Every real symmetric matrix admits an orthonormal eigenbasis with real eigenvalues; SPD matrices have strictly positive spectrum.
- [[Singular Value Decomposition (SVD)]] *(D01)* — Universal orthogonal factorization of any rectangular matrix into left/right singular bases and nonnegative singular values.
- [[Low-Rank Approximation (Eckart-Young)]] *(D01)* — Truncated SVD gives the exact best rank-k approximation in both spectral and Frobenius norms.
- [[Law of Large Numbers & CLT]] *(D01)* — Sample averages converge almost surely to population means (LLN) with sqrt(N)-scaled Gaussian fluctuations (CLT).
- [[Concentration Inequalities]] *(D01)* — Non-asymptotic tail bounds (Markov, Chebyshev, Hoeffding, Bernstein) giving exponential decay for bounded/sub-Gaussian variables.
- [[Mutual Information & DPI]] *(D01)* — Measures shared information I(X;Y) = D_KL(p(x,y) || p(x)p(y)); cannot increase along Markov chains (Data Processing Inequality).
- [[Kantorovich Duality]] *(D01)* — Dual formulation of Optimal Transport over c-concave potentials; reduces W_1 to maximization over 1-Lipschitz critics.
- [[Lagrange Duality & KKT]] *(D01)* — Weak duality always holds (d* <= p*); Slater's condition guarantees strong duality and necessary/sufficient KKT optimality conditions.
- [[Bias-Variance Trade-off]] *(D02)* — Expected squared error decomposes into irreducible noise, squared approximation bias, and estimation variance.
- [[Mercer Kernels & RKHS]] *(D02)* — Positive-definite kernels induce implicit inner products in Reproducing Kernel Hilbert Spaces; Representer Theorem guarantees finite-sum solutions.
- [[MLP & Universal Approximation]] *(D03)* — Feedforward ReLU networks partition input space into exponentially many linear regions O((D/d)^d K) with depth K.
- [[Score-Based SDE]] *(D04)* — Continuous-time stochastic differential equation (VP/VE-SDE) whose time-reversal uses the learned score via Anderson's theorem.
- [[Fokker-Planck Equation]] *(D04)* — Deterministic PDE governing density evolution of an Ito SDE; rewriting diffusion as a transport term yields the Probability Flow ODE.
- [[Chinchilla Scaling Laws]] *(D05)* — Compute-optimal pretraining allocates tokens D proportional to parameters N (D approx 20 N) along C approx 6ND.
- [[Bellman Equations]] *(D06)* — Recursive fixed-point equations for V^pi and Q^*; gamma-contraction in L_infty norm guarantees unique value fixed points.
- [[Policy Gradient Theorem]] *(D06)* — Score-function gradient of expected return avoids differentiating environment dynamics via log-derivative trick and baseline subtraction.
- [[Image Filtering & 2D Fourier Transform]] *(D03)* — LSI image filtering is diagonalized by 2D complex exponentials via the Convolution Theorem; Nyquist sampling requires low-pass anti-aliasing.
- [[Epipolar Geometry & Stereo]] *(D03)* — Two-view coplanarity constrains matches to 1D epipolar lines via Essential E / Fundamental F matrices and yields depth Z = fB/d.

## Objects (20)

- [[Vector Space & Subspaces]] *(D01)* — Linear space closed under addition and scalar multiplication; every matrix defines four fundamental subspaces.
- [[Inner Product & Norms]] *(D01)* — Bilinear SPD form inducing geometry (angles, orthogonality) and topological norm equivalence on finite-dimensional spaces.
- [[Orthogonal Projection]] *(D01)* — Idempotent symmetric linear operator mapping vectors to their closest Euclidean approximation in a subspace.
- [[Moore-Penrose Pseudoinverse]] *(D01)* — Minimum-norm least-squares inverse defined for arbitrary rectangular or rank-deficient matrices via SVD.
- [[Matrix Calculus & Jacobians]] *(D01)* — Differential and trace-duality rules for vector/matrix-valued functions, Hessians, and Kronecker-vec identities.
- [[Multivariate Gaussian & Schur Complement]] *(D01)* — Maximum-entropy distribution under fixed mean/covariance; closed under marginalization (in Sigma) and conditioning (in Precision Lambda).
- [[Exponential Families]] *(D01)* — Parametric family whose log-partition function A(eta) acts as cumulant generator and induces Legendre-dual mean parameters.
- [[Convexity & Subgradients]] *(D01)* — Jensen's inequality, first/second-order conditions, L-smoothness, and strong convexity defining global optimality guarantees.
- [[Bregman & Legendre Duality]] *(D01)* — Fenchel conjugate f*(y) and Bregman divergence D_psi(x,y) unify mirror descent, exponential families, and Sinkhorn projections.
- [[Polyak-Lojasiewicz (PL) Condition]] *(D01)* — Non-convex gradient-dominance condition guaranteeing linear convergence of GD without requiring convexity or unique minima.
- [[Generalized Linear Models (GLM)]] *(D02)* — Unifies OLS, Logistic, Softmax, and Poisson regression via exponential-family response and canonical link g(mu)=eta.
- [[Gaussian Mixture Models (GMM)]] *(D02)* — Latent-variable density model combining K Gaussian components; universal smooth density approximator trained via EM.
- [[Graphical Models (BN & MRF)]] *(D02)* — Factorizes high-dimensional distributions via DAGs (Bayesian Networks) or undirected cliques (Markov Random Fields / Hammersley-Clifford).
- [[Gaussian Processes]] *(D02)* — Nonparametric Bayesian prior over functions whose finite-dimensional marginals are multivariate Gaussians governed by a kernel.
- [[Energy-Based Models (EBM)]] *(D04)* — Unnormalized Gibbs densities p(x) = exp(-E(x))/Z; circumvents intractable partition function Z via contrastive divergence or score matching.
- [[Probability Flow ODE]] *(D04)* — Deterministic ODE sharing the exact marginal densities p_t(x) of the reverse SDE while enabling exact likelihood and fast ODE solvers.
- [[Markov Decision Process (MDP)]] *(D06)* — Tuple (S, A, P, R, gamma) formalizing sequential decision making under Markovian stochastic transitions.
- [[Bradley-Terry Reward Model]] *(D06)* — Pairwise preference likelihood modeling P(y_w > y_l | x) = sigmoid(r(x,y_w) - r(x,y_l)) for scalar reward training.
- [[Pinhole & Thin-Lens Optics]] *(D03)* — Perspective 3D-to-2D projection x=fX/Z, Gaussian thin-lens law 1/z_o + 1/z_i = 1/f, Lambertian shading, and linear vs. gamma sRGB spaces.
- [[Camera Calibration & Homography]] *(D03)* — Homogeneous projection P = K[R|t] and 8-DoF planar/rotational homography H estimated via Hartley-normalized DLT + RANSAC.

## Algorithms (56)

- [[QR Factorization]] *(D01)* — Decomposes a matrix into an orthonormal basis Q and upper-triangular R via Householder reflections or MGS without squaring condition number.
- [[Automatic Differentiation (AD)]] *(D01)* — Exact machine-precision derivatives via computational-graph chain rule in Forward (JVP) or Reverse (VJP / backprop) mode.
- [[Maximum Likelihood & Fisher Info]] *(D01)* — Minimizes forward KL to empirical data; asymptotically efficient achieving the Cramer-Rao lower bound governed by Fisher information.
- [[Information Bottleneck]] *(D01)* — Compresses input X into representation T keeping maximal information about target Y while discarding irrelevant nuisance bits.
- [[Sinkhorn Algorithm]] *(D01)* — Entropic regularization turns OT into strictly convex matrix scaling solvable in O(n^2) GPU-friendly alternating projections.
- [[Gradient Descent]] *(D01)* — Steepest Euclidean descent with step eta <= 1/L; achieves O(1/T) on convex smooth and linear O(kappa log 1/eps) under SC or PL.
- [[Stochastic Gradient Descent (SGD)]] *(D01)* — Unbiased noisy gradient updates converge to an O(eta sigma^2) noise ball under constant step size or O(1/t) under Robbins-Monro decay.
- [[SVRG & Variance Reduction]] *(D01)* — Control-variate anchor snapshots eliminate asymptotic gradient variance, recovering linear convergence on finite sums.
- [[Proximal & Subgradient Methods]] *(D01)* — Splits smooth + nonsmooth composite objectives via proximal operator (e.g., soft-thresholding for L1 sparsity) instead of slow subgradients.
- [[Momentum & Nesterov NAG]] *(D01)* — Velocity accumulation accelerates convergence across ill-conditioned valleys from O(kappa) to O(sqrt(kappa)) and O(1/T^2) convex.
- [[AdamW]] *(D01)* — Adaptive per-coordinate RMS scaling with bias-corrected moments and decoupled weight decay; standard optimizer for Transformers.
- [[Natural Gradient & K-FAC]] *(D01)* — Steepest descent in KL-metric via inverse Fisher Information Matrix (or Kronecker-factored block approximation), invariant to parameterization.
- [[Linear & Ridge Regression]] *(D02)* — Closed-form OLS projection and L2-regularized Ridge shrinking small singular directions to control variance.
- [[Logistic Regression]] *(D02)* — Linear classifier modeling log-odds via sigmoid/softmax link; strictly convex cross-entropy loss trained by Newton/IRLS or GD.
- [[Support Vector Machines (SVM)]] *(D02)* — Maximum-margin hyperplane classifier solved via dual quadratic program over sparse support vectors.
- [[CART Decision Trees]] *(D02)* — Greedy recursive axis-aligned partitioning minimizing Gini impurity, entropy, or MSE within leaf regions.
- [[Random Forests & Bagging]] *(D02)* — Parallel ensemble of decorrelated bootstrap trees reducing prediction variance without increasing bias.
- [[Gradient Boosted Decision Trees (GBDT)]] *(D02)* — Sequential functional gradient descent in tree space (XGBoost/LightGBM) using second-order Newton leaf weights to slash bias.
- [[Principal Component Analysis (PCA)]] *(D02)* — Orthogonal linear projection maximizing retained variance and minimizing reconstruction MSE simultaneously via top-k eigenvectors/SVD.
- [[EM Algorithm]] *(D02)* — Alternating coordinate ascent on the ELBO: E-step sets exact posterior q(z)=p(z|x,theta); M-step maximizes expected complete log-likelihood.
- [[Variational Inference & ELBO]] *(D02)* — Converts intractable posterior inference into optimization by maximizing the Evidence Lower Bound (minimizing Reverse KL to posterior).
- [[Reparameterization Trick]] *(D02)* — Expresses random sample z = g(theta, eps) with parameter-free base noise eps to obtain low-variance pathwise gradients.
- [[MCMC & HMC Sampling]] *(D02)* — Ergodic Markov chains (Metropolis-Hastings, Gibbs) and symplectic Hamiltonian dynamics (HMC/NUTS) for asymptotically exact posterior sampling.
- [[He & Xavier Initialization]] *(D03)* — Variance-preserving weight scaling preventing exponential 2^l explosion or collapse across deep forward/backward passes.
- [[Weight Decay & Dropout]] *(D03)* — Explicit L2/decoupled shrinkage and stochastic sub-network masking regularizing capacity and calibration.
- [[Kalman Filter]] *(D03)* — Exact closed-form Bayesian filtering and smoothing for linear-Gaussian state-space dynamics via predict-update recursion.
- [[Score Matching (DSM & SSM)]] *(D04)* — Learns the Stein score grad_x log p(x) without Z_theta via denoising perturbation (DSM) or Hutchinson trace slicing (SSM).
- [[Flow Matching & Rectified Flow]] *(D04)* — Simulation-free regression of continuous velocity fields along straight OT-like interpolation paths x_t = (1-t)x_0 + t x_1.
- [[Classifier-Free Guidance (CFG)]] *(D04)* — Extrapolates conditional vs. unconditional score predictions to trade diversity for sharp prompt alignment without an external classifier.
- [[BPE & Subword Tokenization]] *(D05)* — Greedy frequency-based byte/character merge vocabulary balancing sequence length against open-vocabulary generalization.
- [[FlashAttention]] *(D05)* — IO-aware tiled SRAM online-softmax algorithm computing exact attention in O(N) memory without materializing the N x N matrix in HBM.
- [[LoRA & QLoRA]] *(D05)* — Parameter-efficient adaptation via frozen 4-bit NF4 base weights plus trainable low-rank bypasses (alpha/r) B A.
- [[KV Cache & PagedAttention]] *(D05)* — Caches past K,V tensors to avoid O(L^2) recomputation per token; PagedAttention eliminates fragmentation via OS-style virtual memory blocks.
- [[Speculative Decoding]] *(D05)* — Fast draft model proposes gamma tokens verified in parallel by target model with exact rejection sampling preserving target distribution.
- [[Multi-Armed Bandits & UCB]] *(D06)* — Stateless exploration-exploitation trade-off; Upper Confidence Bound (UCB) achieves logarithmic O(log T) regret via optimism under uncertainty.
- [[Thompson Sampling]] *(D06)* — Probability-matching Bayesian bandit selecting arms proportional to posterior probability of being optimal.
- [[Value & Policy Iteration]] *(D06)* — Dynamic programming sweeps alternating Bellman evaluation and greedy improvement when transition model P is known.
- [[MC & TD Learning]] *(D06)* — Model-free value estimation: unbiased high-variance Monte Carlo returns vs. biased low-variance bootstrapped TD(0) targets.
- [[Q-Learning & DQN]] *(D06)* — Off-policy TD control learning Q*; stabilized at scale via experience replay, target networks, and Double-Q.
- [[Generalized Advantage Estimation (GAE)]] *(D06)* — Exponentially weighted sum of k-step TD residuals trading bias (lambda -> 0) against variance (lambda -> 1).
- [[TRPO]] *(D06)* — Monotonic policy improvement under a hard KL trust-region constraint solved with Fisher-vector products and conjugate gradients.
- [[PPO (Proximal Policy Optimization)]] *(D06)* — First-order trust-region proxy clipping probability ratios r_t(theta) within [1-eps, 1+eps]; workhorse of RL and RLHF.
- [[GRPO]] *(D06)* — Group Relative Policy Optimization drops the critic network and normalizes rewards across G sampled completions per prompt.
- [[Model-Based RL (Dyna & MBPO)]] *(D06)* — Learns dynamics model to generate short k-step branched rollouts (MBPO), bounding compounding model error.
- [[MCTS & MuZero]] *(D06)* — Decision-time planning via PUCT tree search over learned latent value, policy, and reward dynamics without pixel reconstruction.
- [[Offline RL (CQL)]] *(D06)* — Conservative Q-Learning penalizes OOD action values outside the behavior dataset support to prevent extrapolation overestimation.
- [[RLHF (KL-Regularized)]] *(D06)* — Maximizes learned reward while penalizing KL divergence from reference SFT policy to prevent reward hacking.
- [[DPO (Direct Preference Optimization)]] *(D06)* — Reparameterizes the closed-form Gibbs optimum of KL-RLHF into a direct binary cross-entropy loss on policy log-ratios without an explicit RM.
- [[RLVR (Verifiable Rewards)]] *(D06)* — Trains reasoning policies via GRPO against deterministic programmatic verifiers (math/code unit tests) instead of learned reward models.
- [[Sparse vs Dense Retrieval & RRF]] *(D07)* — Combines lexical BM25 exact matching with Bi-Encoder semantic embeddings and ColBERT late interaction via Reciprocal Rank Fusion.
- [[HNSW & IVF-PQ Indexing]] *(D07)* — Approximate nearest-neighbor vector search trading RAM vs. recall: hierarchical small-world graphs (HNSW) vs. product-quantized Voronoi cells.
- [[Constrained JSON Decoding]] *(D07)* — Grammar/FSM token logit masking guaranteeing tool-call outputs strictly conform to JSON Schema.
- [[Optical Flow (Lucas-Kanade & Horn-Schunck)]] *(D03)* — Estimates dense 2D pixel velocity from brightness constancy I_x u + I_y v + I_t = 0 via local window least squares or global smoothness.
- [[Collaborative Filtering (User-Item)]] *(D07)* — Predicts user preferences from neighborhood cosine/Pearson similarity over mean-centered user-item interaction vectors.
- [[Matrix Factorization (ALS & SGD)]] *(D07)* — Low-rank biased completion Y approx mu + b_u + b_i + P Q^T solved via SGD or closed-form Alternating Least Squares (including implicit WALS).
- [[Bayesian Personalized Ranking (BPR)]] *(D07)* — Pairwise implicit-feedback ranking loss maximizing log-sigmoid preference margin between observed positives and sampled negatives.

## Architectures (34)

- [[Kolmogorov-Arnold Networks (KAN)]] *(D03)* — Replaces fixed node activations with learnable univariate B-spline functions on graph edges.
- [[BatchNorm, LayerNorm & RMSNorm]] *(D03)* — Stabilizes activation scales across layers: BatchNorm over (N,H,W), LayerNorm over feature dim D, RMSNorm without mean-centering.
- [[Residual Connections]] *(D03)* — Identity skip paths x + F(x) preserve gradient norms across hundreds of layers and yield implicit ensemble behavior.
- [[CNN & Equivariance]] *(D03)* — Weight-shared local convolutions enforce translation equivariance f(T_g x) = T_g f(x) and hierarchical receptive fields.
- [[Vision Transformer (ViT)]] *(D03)* — Tokenizes images into patches and applies global self-attention with lower inductive bias and higher data-scaling capacity than CNNs.
- [[Graph Neural Networks (MPNN)]] *(D03)* — Permutation-equivariant message passing over graph neighborhoods; expressivity bounded by the 1-WL isomorphism test.
- [[RNN, LSTM & GRU]] *(D03)* — Fixed-state recurrent sequence models using additive cell-state highways (LSTM/GRU) to mitigate vanishing gradients during BPTT.
- [[State Space Models (S4 & Mamba)]] *(D03)* — Discretized continuous linear/selective SSMs achieving O(L log L) or O(LN) parallel training and O(1) constant-memory autoregressive inference.
- [[Autoencoders]] *(D04)* — Bottleneck encoder-decoder learning compressed latent manifolds; regularized via sparsity, denoising, or discrete VQ codebooks.
- [[Variational Autoencoder (VAE)]] *(D04)* — Amortized variational generative model optimizing the ELBO via encoder inference network and reparameterized latent sampling.
- [[Normalizing Flows]] *(D04)* — Invertible diffeomorphic compositions enabling exact likelihood evaluation via change-of-variables and triangular/coupling Jacobians.
- [[GAN & Mode Collapse]] *(D04)* — Implicit minimax game between generator and discriminator; minimizes JS divergence at optimum but suffers vanishing gradients and mode collapse.
- [[Wasserstein GAN (WGAN)]] *(D04)* — Replaces JS discriminator with a 1-Lipschitz Kantorovich critic via gradient penalty or spectral norm for stable Wasserstein-1 training.
- [[DDPM]] *(D04)* — Discrete-time Markovian noising and learned reverse Gaussian denoising; equivalent to multi-scale denoising score matching.
- [[Consistency Models]] *(D04)* — Maps any point on a PF-ODE trajectory directly to its clean origin x_0 in 1-4 steps via self-consistency distillation.
- [[RoPE & Positional Embeddings]] *(D05)* — Rotary Position Embedding encodes relative token displacement m-n directly into Q-K inner products via 2D rotation blocks.
- [[Self-Attention & Multi-Head Attention]] *(D05)* — Content-addressed bilinear routing over Q, K, V scaled by 1/sqrt(d_k) across parallel representation subspaces.
- [[GQA & MQA]] *(D05)* — Grouped-Query and Multi-Query Attention share K/V heads across Q-head groups to slash autoregressive KV cache bandwidth.
- [[Transformer Architecture]] *(D05)* — Pre-LN residual backbone interleaving Multi-Head Attention (mixing tokens) and SwiGLU FFN (mixing channels).
- [[ReAct & Tool-Use Loop]] *(D07)* — Interleaves chain-of-thought reasoning traces with grounded external tool actions and environment observations.
- [[Agent Planning & Multi-Agent Patterns]] *(D07)* — Structured decomposition (Plan-and-Execute, Tree-of-Thoughts, Reflexion) and coordinator/worker delegation with isolated contexts.
- [[Context Memory & Compaction]] *(D07)* — Manages finite context budgets across system prompts, compacted session traces, and consolidated declarative/procedural long-term memory.
- [[Model Context Protocol (MCP)]] *(D07)* — Vertical O(M+N) JSON-RPC 2.0 protocol connecting Host/Client agents to standardized Tools, Resources, and Prompts.
- [[Agent-to-Agent Protocol (A2A)]] *(D07)* — Horizontal peer interoperability protocol using AgentCards and stateful task lifecycles for cross-agent delegation.
- [[Workflow DAGs]] *(D07)* — Deterministic orchestration graphs compounding step reliability around probabilistic LLM nodes via retries and fallbacks.
- [[FTI Architecture & Leakage Prevention]] *(D07)* — Decouples Feature, Training, and Inference pipelines with point-in-time as-of joins and split-invariant transforms.
- [[Gaussian & Laplacian Pyramids]] *(D03)* — Multiresolution bandpass decomposition storing downsampled Gaussian levels and Laplacian residuals with 4/3 memory overhead.
- [[Radiance Fields (NeRF & 3DGS)]] *(D03)* — Continuous neural volume density + view-dependent color rendered via differentiable ray-marching, or explicit 3D Gaussian splats.
- [[Object Detection (R-CNN, SSD & DETR)]] *(D03)* — Two-stage R-CNN (RPN + RoIAlign) vs. one-stage SSD/YOLO anchor grids vs. anchor-free DETR transformer bipartite Hungarian matching.
- [[Semantic & Instance Segmentation (FCN, U-Net & SAM)]] *(D03)* — Dense pixel-level classification using transposed convolutions or skip-connected encoder-decoders (FCN, U-Net) and promptable SAM.
- [[CLIP & Vision-Language Alignment]] *(D03)* — Dual image-text encoders trained with symmetric InfoNCE contrastive loss enabling zero-shot classification and open-vocabulary vision.
- [[AutoRec]] *(D07)* — Masked nonlinear autoencoder reconstructing observed entries of sparse user or item rating vectors.
- [[Factorization Machines & DeepFM]] *(D07)* — Computes all degree-2 sparse feature interactions in linear O(kd) time via shared embeddings, augmented with a parallel deep MLP in DeepFM.
- [[Two-Tower Retrieval & Multi-Stage Ranking]] *(D07)* — Decoupled user/item encoders with log-Q corrected in-batch softmax enabling O(log N) ANN candidate retrieval before DeepFM scoring.

## Concepts (4)

- [[Double Descent]] *(D02)* — Test risk peaks at the interpolation threshold (P approx N) and descends again in the overparameterized regime via implicit minimum-norm bias.
- [[Chain-of-Thought & Test-Time Compute]] *(D05)* — Allocates extra inference tokens/verifiers (CoT, self-consistency, MCTS) to trade latency for reasoning accuracy.
- [[Deadly Triad]] *(D06)* — Combining (1) function approximation, (2) bootstrapping, and (3) off-policy sampling can cause unbounded value divergence.
- [[Distribution Shift & Drift]] *(D07)* — Decomposes joint shift into Covariate P(X), Prior P(Y), and Concept P(Y|X) drift detected via PSI, KS, or MMD.

## Metrics (10)

- [[Condition Number]] *(D01)* — Ratio of largest to smallest singular value (or eigenvalue); governs numerical error amplification and GD convergence speed.
- [[Mahalanobis Distance]] *(D01)* — Precision-weighted quadratic distance invariant under invertible affine transforms; equals Euclidean distance after whitening.
- [[Shannon Entropy]] *(D01)* — Expected surprisal measuring uncertainty of a random variable and the fundamental lower bound on lossless source coding.
- [[KL Divergence]] *(D01)* — Asymmetric information divergence: Forward KL (M-projection, mode-covering) vs. Reverse KL (I-projection, mode-seeking).
- [[f-Divergence]] *(D01)* — Convex-generator family of probability divergences satisfying the Data Processing Inequality and Pinsker-type bounds.
- [[Wasserstein Distance]] *(D01)* — Optimal transport metric lifting ground geometry to distributions; remains smooth and informative even on disjoint supports.
- [[Rademacher Complexity & VC Dimension]] *(D02)* — Distribution-dependent and combinatorial capacity measures yielding uniform convergence generalization bounds.
- [[AgentOps & pass^k Reliability]] *(D07)* — Distinguishes demo pass@k from production SLA consistency pass^k = p_step^(kH) under OpenTelemetry tracing and guardrails.
- [[IoU, Anchor Boxes & NMS]] *(D03)* — Jaccard bounding-box overlap, scale/aspect anchor priors, Focal Loss for class imbalance, and greedy Non-Maximum Suppression.
- [[NDCG, MAP & Calibration]] *(D07)* — Position-discounted ranking quality (DCG/IDCG) and IPS propensity debiasing for evaluation under presentation bias.
