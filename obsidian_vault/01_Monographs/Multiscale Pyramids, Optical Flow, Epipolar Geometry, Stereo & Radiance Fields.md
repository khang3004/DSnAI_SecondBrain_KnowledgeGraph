---
type: monograph
domain: "D03"
parent_topic: "[[PT 03 - Computer Vision, Image Formation & 3D Scene Understanding]]"
sub_topic: "[[ST 03 - Image Formation, Fourier Filtering, Pyramids & 3D Geometry]]"
cheatsheet: "[[Cheatsheet - Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]"
part_of: 
  - "[[ST 03 - Image Formation, Fourier Filtering, Pyramids & 3D Geometry]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
concepts: 
  - "[[Camera Calibration & Homography]]"
  - "[[Epipolar Geometry & Stereo]]"
  - "[[Gaussian & Laplacian Pyramids]]"
  - "[[Optical Flow (Lucas-Kanade & Horn-Schunck)]]"
  - "[[Radiance Fields (NeRF & 3DGS)]]"
prerequisites: 
  - "[[Camera Calibration & Homography]]"
  - "[[Epipolar Geometry & Stereo]]"
  - "[[Gaussian & Laplacian Pyramids]]"
  - "[[Image Filtering & 2D Fourier Transform]]"
  - "[[Pinhole & Thin-Lens Optics]]"
  - "[[Singular Value Decomposition (SVD)]]"
enables: 
  - "[[Epipolar Geometry & Stereo]]"
  - "[[Object Detection (R-CNN, SSD & DETR)]]"
  - "[[Optical Flow (Lucas-Kanade & Horn-Schunck)]]"
  - "[[Radiance Fields (NeRF & 3DGS)]]"
related: 
  - "[[Camera Calibration & Homography]]"
  - "[[Epipolar Geometry & Stereo]]"
  - "[[Gaussian & Laplacian Pyramids]]"
  - "[[Image Filtering & 2D Fourier Transform]]"
  - "[[MLP & Universal Approximation]]"
  - "[[Object Detection (R-CNN, SSD & DETR)]]"
  - "[[Optical Flow (Lucas-Kanade & Horn-Schunck)]]"
  - "[[Pinhole & Thin-Lens Optics]]"
  - "[[Radiance Fields (NeRF & 3DGS)]]"
sources: 
  - "[[Source - Antonio Torralba, Phillip Isola, and William Freeman]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - monograph
  - domain-03
---

# Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields

> [!abstract] Monograph Navigation
> - **Domain:** [[D03 - Neural Architectures, Representations & Regularization|Neural Architectures, Representations & Regularization]]
> - **Parent / Sub-Topic:** [[PT 03 - Computer Vision, Image Formation & 3D Scene Understanding|Computer Vision, Image Formation & 3D Scene Understanding]] → [[ST 03 - Image Formation, Fourier Filtering, Pyramids & 3D Geometry|Image Formation, Fourier Filtering, Pyramids & 3D Geometry]]
> - **Visual Cheatsheet:** [[Cheatsheet - Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
> - **Description:** Gaussian/Laplacian pyramids, steerable filter banks, Lucas-Kanade & Horn-Schunck optical flow, homogeneous camera matrices, homographies, epipolar constraints (F/E matrices), stereo disparity, SfM, and Neural Radiance Fields (NeRF).

## Problem Formulation & Unified Notation

Recovering 3D structure and motion from 2D images requires analyzing visual signals across spatial scales, tracking pixel displacement across time, and inverting projective multi-view geometry [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 23, 39–40, 48). We unify multiscale image pyramids, differential optical flow $(u, v)^{\top}$, homogeneous camera projection $\boldsymbol{P} = \boldsymbol{K}[\boldsymbol{R} \mid \boldsymbol{t}]$, epipolar constraints $(\boldsymbol{x}')^{\top} \boldsymbol{F} \boldsymbol{x} = 0$, and continuous volumetric rendering (NeRF) [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 39–45, pp. 777–935).

> [!info] Cross-Book Notation: Pyramids, Camera Matrices & Epipolar Geometry
> | **Concept** | **MIT Vision Book** | **Unified Monograph** |
> | — | — | — |
> | Homogeneous 3D/2D Point | $\tilde{\boldsymbol{X}} = [X,Y,Z,1]^{\top}$, $\tilde{\boldsymbol{x}} = [x,y,1]^{\top}$ | $\tilde{\boldsymbol{X}} \in \mathbb{R}^4$, $\tilde{\boldsymbol{x}} \in \mathbb{R}^3$ |
> | Intrinsics & Extrinsics | $\boldsymbol{K}$ (3x3), $[\boldsymbol{R} \mid \boldsymbol{t}]$ (3x4) | $\boldsymbol{P} = \boldsymbol{K}[\boldsymbol{R} \mid \boldsymbol{t}]$ |
> | Epipolar Matrices | Essential $\boldsymbol{E}$, Fundamental $\boldsymbol{F}$ | $\boldsymbol{E} = [\boldsymbol{t}]_\times \boldsymbol{R}$, $\boldsymbol{F} = (\boldsymbol{K}')^{-\top}\boldsymbol{E}\boldsymbol{K}^{-1}$ |
> | Optical Flow | $(u, v) = (\dot{x}, \dot{y})$ | $\boldsymbol{u} = (u, v)^{\top}$ via $I_x u + I_y v + I_t = 0$ |

## Core Mathematical Derivations

### Multiscale Image Pyramids & Optical Flow

> [!important] Gaussian/Laplacian Pyramids & Brightness Constancy Optical Flow
> A *Gaussian pyramid* repeatedly applies anti-aliasing smoothing and $\downarrow_2$ subsampling: $G_0 = I$, $G_{\ell+1} = \downarrow_2(g_\sigma * G_\ell)$ [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 23, pp. 466–478). The *Laplacian pyramid* stores bandpass residuals $L_\ell = G_\ell - \uparrow_2(G_{\ell+1})$, enabling exact reconstruction ($I = \sum_\ell \uparrow_2^\ell L_\ell$) and seamless spline blending with total memory overhead $\sum_{\ell=0}^\infty 4^{-\ell} = \frac{4}{3}$ [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 23, pp. 475–485).
>
> For video motion, the *brightness constancy assumption* $I(x+u, y+v, t+1) \approx I(x,y,t)$ yields the linearized *optical flow constraint equation* [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 48, pp. 961–970):
>
> $$
> I_x u + I_y v + I_t = \nabla I^{\top} \boldsymbol{u} + I_t = 0.
>
> $$
>
> Because one scalar equation cannot determine two unknowns (the *aperture problem*, recovering only motion normal to edges), *Lucas–Kanade* assumes constant flow in a local window $W$ and solves the $2\times 2$ structure-tensor system $(\sum_W \nabla I \nabla I^{\top})\boldsymbol{u} = -\sum_W \nabla I  I_t$, while *Horn–Schunck* minimizes a global smoothness functional $\int [(\nabla I^{\top}\boldsymbol{u}+I_t)^2 + \alpha(\lVert \nabla u \rVert_2^2 + \lVert \nabla v \rVert_2^2)] \mathrm{d}\boldsymbol{x}$ [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 48, pp. 964–972).

### Projective Geometry, Homographies, Stereo & Neural Radiance Fields

> [!important] Camera Projection, Epipolar Constraint, Stereo Disparity & NeRF Rendering
> A calibrated perspective camera maps 3D world points $\tilde{\boldsymbol{X}}$ to homogeneous pixels $\tilde{\boldsymbol{x}} \sim \boldsymbol{K}[\boldsymbol{R} \mid \boldsymbol{t}]\tilde{\boldsymbol{X}}$, where $\boldsymbol{K}$ encodes focal lengths $(f_x, f_y)$ and principal point $(c_x, c_y)$ [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 39, pp. 777–795). For planar scenes or pure camera rotation, two views are related by a $3\times 3$ *homography* $\tilde{\boldsymbol{x}}' \sim \boldsymbol{H}\tilde{\boldsymbol{x}}$ (8 DoF, estimated via DLT + RANSAC) [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 41, pp. 827–843).
>
> For general 3D scenes observed from two views with relative pose $(\boldsymbol{R}, \boldsymbol{t})$, coplanarity of the baseline $\boldsymbol{t}$ and viewing rays gives the *epipolar constraint* [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 44, pp. 889–910):
>
> $$
> (\hat{\boldsymbol{x}}')^{\top} \boldsymbol{E} \hat{\boldsymbol{x}} = 0 \quad (\boldsymbol{E} = [\boldsymbol{t}]_\times \boldsymbol{R}), \qquad (\tilde{\boldsymbol{x}}')^{\top} \boldsymbol{F} \tilde{\boldsymbol{x}} = 0 \quad (\boldsymbol{F} = (\boldsymbol{K}')^{-\top}\boldsymbol{E}\boldsymbol{K}^{-1}).
>
> $$
>
> In rectified parallel stereo with baseline $B$ and focal length $f$, horizontal disparity $d = x_L - x_R$ yields depth via triangulation: $Z = \frac{f B}{d}$ [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 40, pp. 804–820). Moving from discrete point clouds to continuous 3D representations, *Neural Radiance Fields (NeRF)* parameterize volume density $\sigma(\boldsymbol{x})$ and view-dependent color $\boldsymbol{c}(\boldsymbol{x}, \boldsymbol{d})$ with an MLP, rendering pixel color along ray $\boldsymbol{r}(t)=\boldsymbol{o}+t\boldsymbol{d}$ via differentiable quadrature [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 45, pp. 921–935):
>
> $$
> \hat{\boldsymbol{C}}(\boldsymbol{r}) = \int_{t_n}^{t_f} T(t) \sigma(\boldsymbol{r}(t)) \boldsymbol{c}(\boldsymbol{r}(t),\boldsymbol{d}) \mathrm{d}t, \qquad T(t) = \exp\!\left(-\int_{t_n}^{t} \sigma(\boldsymbol{r}(s)) \mathrm{d}s\right).
>
> $$

> [!tip] Why Epipolar Geometry Collapses 2D Search to 1D and Why Pyramids Solve Large Motion
> In uncalibrated stereo, matching a pixel in image 1 would require searching all $H\times W$ pixels in image 2. The fundamental matrix $\boldsymbol{F}$ restricts the match to the 1D *epipolar line* $\boldsymbol{l}' = \boldsymbol{F}\tilde{\boldsymbol{x}}$ [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 44, pp. 892–898). Similarly, Taylor-based optical flow fails for displacements larger than  1 pixel; coarse-to-fine Gaussian pyramids shrink large motions into sub-pixel steps at coarse levels before refining at fine levels [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 48, pp. 968–971).

## Visual Diagnostics & 3D Geometry

> [!example] Visual Diagram (LaTeX / PDF)
> *Two-view epipolar geometry: camera centers $\boldsymbol{O}_L, \boldsymbol{O}_R$ and 3D point $\boldsymbol{X}$ span the epipolar plane, constraining correspondences $\tilde{\boldsymbol{x}} \leftrightarrow \tilde{\boldsymbol{x}}'$ to 1D epipolar lines.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!warning] Aperture Problem, Ill-Conditioned Triangulation & Spectral Bias in NeRFs
> Lucas–Kanade fails inside textureless regions or along straight edges where $\boldsymbol{J} = \sum_W \nabla I\nabla I^{\top}$ is rank-deficient ($\lambda_{\min} \approx 0$) [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 48, pp. 965–967). In stereo, depth error scales quadratically with distance ($\Delta Z \propto Z^2 / (fB)$), degrading rapidly when baseline $B$ is small [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 40, pp. 808–812). Finally, standard MLPs in NeRF suffer from low-frequency spectral bias unless coordinates $\boldsymbol{x}$ are lifted via sinusoidal positional encoding (Fourier features) [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 45, pp. 928–932).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (5):** [[Camera Calibration & Homography]], [[Epipolar Geometry & Stereo]], [[Gaussian & Laplacian Pyramids]], [[Optical Flow (Lucas-Kanade & Horn-Schunck)]], [[Radiance Fields (NeRF & 3DGS)]]
- **Prerequisites (Directed $\leftarrow$):** [[Camera Calibration & Homography]], [[Epipolar Geometry & Stereo]], [[Gaussian & Laplacian Pyramids]], [[Image Filtering & 2D Fourier Transform]], [[Pinhole & Thin-Lens Optics]], [[Singular Value Decomposition (SVD)]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Epipolar Geometry & Stereo]], [[Object Detection (R-CNN, SSD & DETR)]], [[Optical Flow (Lucas-Kanade & Horn-Schunck)]], [[Radiance Fields (NeRF & 3DGS)]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Camera Calibration & Homography]], [[Epipolar Geometry & Stereo]], [[Gaussian & Laplacian Pyramids]], [[Image Filtering & 2D Fourier Transform]], [[MLP & Universal Approximation]], [[Object Detection (R-CNN, SSD & DETR)]], [[Optical Flow (Lucas-Kanade & Horn-Schunck)]], [[Pinhole & Thin-Lens Optics]], [[Radiance Fields (NeRF & 3DGS)]]
- **Synthesized Sources:** [[Source - Antonio Torralba, Phillip Isola, and William Freeman]], [[Source - Understanding Deep Learning]]
