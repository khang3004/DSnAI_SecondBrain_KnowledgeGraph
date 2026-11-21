---
type: monograph
domain: "D03"
parent_topic: "[[PT 03 - Computer Vision, Image Formation & 3D Scene Understanding]]"
sub_topic: "[[ST 03 - Image Formation, Fourier Filtering, Pyramids & 3D Geometry]]"
cheatsheet: "[[Cheatsheet - Image Formation, Lenses, Color Spaces & Fourier Image Filtering]]"
part_of: 
  - "[[ST 03 - Image Formation, Fourier Filtering, Pyramids & 3D Geometry]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
concepts: 
  - "[[Image Filtering & 2D Fourier Transform]]"
  - "[[Pinhole & Thin-Lens Optics]]"
prerequisites: 
  - "[[CNN & Equivariance]]"
enables: 
  - "[[Camera Calibration & Homography]]"
  - "[[Gaussian & Laplacian Pyramids]]"
  - "[[Optical Flow (Lucas-Kanade & Horn-Schunck)]]"
related: 
  - "[[Camera Calibration & Homography]]"
  - "[[Gaussian & Laplacian Pyramids]]"
  - "[[Image Filtering & 2D Fourier Transform]]"
  - "[[Pinhole & Thin-Lens Optics]]"
sources: 
  - "[[Source - Antonio Torralba, Phillip Isola, and William Freeman]]"
  - "[[Source - Dive into Deep Learning]]"
tags:
  - monograph
  - domain-03
---

# Image Formation, Lenses, Color Spaces & Fourier Image Filtering

> [!abstract] Monograph Navigation
> - **Domain:** [[D03 - Neural Architectures, Representations & Regularization|Neural Architectures, Representations & Regularization]]
> - **Parent / Sub-Topic:** [[PT 03 - Computer Vision, Image Formation & 3D Scene Understanding|Computer Vision, Image Formation & 3D Scene Understanding]] → [[ST 03 - Image Formation, Fourier Filtering, Pyramids & 3D Geometry|Image Formation, Fourier Filtering, Pyramids & 3D Geometry]]
> - **Visual Cheatsheet:** [[Cheatsheet - Image Formation, Lenses, Color Spaces & Fourier Image Filtering]]
> - **Description:** Pinhole and thin-lens optics, radiometry/BRDF, linear camera pipelines, trichromatic color spaces, 2D convolution, Gaussian/Laplacian derivatives, and 2D Fourier spectrum & Nyquist sampling.

## Problem Formulation & Unified Notation

Computer vision inverts the physical *forward image formation* process: a 3D scene illuminated by light sources reflects radiance onto a sensor through optics, producing a discrete 2D array $I[n_y, n_x, c]$ [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 5, pp. 99–113). Classical and neural vision pipelines analyze and transform this signal via linear shift-invariant (LSI) filtering, spectral frequency decomposition, and sampling theory [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 15–16, pp. 267–328), [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 7, pp. 273–285).

> [!info] Cross-Book Notation: Optics, Convolution vs. Cross-Correlation
>
> | **Concept** | **MIT Vision Book** | **D2L (Zhang et al.)** | **Unified Monograph** |
> | --- | --- | --- | --- |
> | 2D Image | $f[n,m]$ or $\ell(x,y)$ | $\boldsymbol{X} \in \mathbb{R}^{n_h \times n_w}$ | $I[y,x]$ or $I(\boldsymbol{x})$ |
> | Linear Filtering | Convolution $f \circ h$ vs. correlation | Cross-correlation $\boldsymbol{X} \star \boldsymbol{K}$ | $(I * K)[y,x]$ (flipped), $(I \star K)$ (corr.) |
> | 2D DFT Frequency | $(u, v)$ or $(\omega_x, \omega_y)$ | Spatial kernel size $k_h \times k_w$ | $\hat{I}(u,v) = \mathcal{F}\{I\}(u,v)$ |

## Core Mathematical Derivations

### Pinhole Optics, Thin-Lens Equation & Color Spaces

> [!important] Perspective Projection, Thin-Lens Law & Radiometry
> Under an ideal pinhole at distance $f$ (focal length) from the image plane, a 3D point $(X, Y, Z)^{\top}$ projects to $(x, y)^{\top}$ by similar triangles [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 5, pp. 103–108):
>
> $$
> x = f \frac{X}{Z}, \qquad y = f \frac{Y}{Z}.
> $$
>
> To gather finite photon flux without blurring, a thin convex lens of focal length $f$ focuses rays from object depth $z_o$ onto sensor distance $z_i$ via the *Gaussian thin-lens equation* [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 6, pp. 114–126):
>
> $$
> \frac{1}{z_o} + \frac{1}{z_i} = \frac{1}{f}.
> $$
>
> Out-of-focus points produce a circle of confusion of diameter $c \propto \frac{A}{z_o}|z_o - z_{\mathrm{focus}}|$ where $A = f / N$ is aperture diameter ($N$: f-number). For a Lambertian surface with albedo $\rho(\boldsymbol{x})$ and normal $\boldsymbol{n}$ under directional light $\boldsymbol{l}$, reflected radiance is $L_o = \rho(\boldsymbol{x})\max(0, \boldsymbol{n}^{\top}\boldsymbol{l}) E_0$ [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 5, pp. 100–102). Trichromatic sensors integrate spectral power $S(\lambda)$ against cone/filter curves $c_k(\lambda)$, giving linear camera responses $r_k = \int S(\lambda) c_k(\lambda) \mathrm{d}\lambda$ before nonlinear gamma encoding ($I_{\mathrm{sRGB}} \approx I_{\mathrm{lin}}^{1/2.2}$) [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 7–8, pp. 147–178).

### Linear Shift-Invariant Filtering, 2D Fourier Analysis & Nyquist Sampling

> [!important] 2D Convolution Theorem, Derivative Filters & Nyquist-Shannon Limit
> Any Linear Shift-Invariant (LSI) operator $\mathcal{L}$ on images is uniquely determined by its impulse response $h$: $(I * h)[y,x] = \sum_{i,j} I[y-i, x-j]  h[i,j]$ [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 15, pp. 267–285). Because complex exponentials $e^{i 2\pi(ux+vy)}$ are eigenfunctions of all LSI operators, the 2D Discrete Fourier Transform diagonalizes convolution into pointwise multiplication [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 16, pp. 292–325):
>
> $$
> \hat{I}[u,v] = \sum_{y=0}^{H-1}\sum_{x=0}^{W-1} I[y,x] e^{-i 2\pi\left(\frac{ux}{W} + \frac{vy}{H}\right)}, \qquad \mathcal{F}\{I * h\} = \hat{I} \odot \hat{h}.
> $$
>
> Gaussian smoothing $G_\sigma(x,y) = \frac{1}{2\pi\sigma^2}e^{-(x^2+y^2)/(2\sigma^2)}$ has a Gaussian spectrum $\hat{G}_\sigma(u,v) = e^{-2\pi^2\sigma^2(u^2+v^2)}$, acting as an isotropic low-pass filter [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 17, pp. 331–348). Image gradients $\nabla I = (\partial_x I, \partial_y I)^{\top}$ and edges (zero-crossings of Laplacian $\Delta I = \partial_{xx} I + \partial_{yy} I$) are regularized via Derivative-of-Gaussian ($\nabla G_\sigma * I$) and Sobel kernels [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 18, pp. 352–380). By the *Nyquist–Shannon sampling theorem*, sampling with spacing $\Delta x$ aliases frequencies $|u| > \frac{1}{2\Delta x}$; anti-aliasing low-pass prefiltering is therefore mandatory before downsampling [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 20, pp. 409–428).

> [!tip] Why CNNs Compute Cross-Correlation and Why Depthwise Separability Works
> Deep learning libraries implement $(I \star K)$ without flipping the kernel because learnable weights absorb the flip [[Source - Dive into Deep Learning|dive_into_deep_learning_d2l]] (Ch. 7, pp. 274–276). Moreover, a 2D Gaussian filter (and many edge kernels) is rank-1 separable: $G_\sigma(x,y) = g_\sigma(x)g_\sigma(y)$, reducing $O(k^2 HW)$ filtering cost to $O(2k HW)$ [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 15, pp. 282–285).

## Visual Diagnostics & Geometry

> [!example] Visual Diagram (LaTeX / PDF)
> *Left: perspective pinhole geometry $y = fY/Z$. Right: low-pass Gaussian spectrum attenuating frequencies above the Nyquist limit to prevent aliasing.* — see rendered vector figure in `master.pdf` / `atlas.pdf`.

> [!warning] Gamma Nonlinearity & Unfiltered Strided Downsampling
> Applying linear blur, interpolation, or physically-based shading in gamma-compressed sRGB space introduces dark halos and hue shifts; always linearize ($I_{\mathrm{lin}} = I_{\mathrm{sRGB}}^{2.2}$) before physical operations [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 7, pp. 152–156). Similarly, strided pooling or `stride=2` convolutions without anti-aliasing low-pass filters fold high-frequency edges into spurious low-frequency Moiré patterns, breaking shift invariance [[Source - Antonio Torralba, Phillip Isola, and William Freeman|foundations_computer_vision_mit]] (Ch. 20, pp. 412–425).

---

## Knowledge Graph Connections

- **Core Atomic Concepts (2):** [[Image Filtering & 2D Fourier Transform]], [[Pinhole & Thin-Lens Optics]]
- **Prerequisites (Directed $\leftarrow$):** [[CNN & Equivariance]]
- **Enables / Unlocks (Directed $\rightarrow$):** [[Camera Calibration & Homography]], [[Gaussian & Laplacian Pyramids]], [[Optical Flow (Lucas-Kanade & Horn-Schunck)]]
- **Related Concepts (Undirected $\leftrightarrow$):** [[Camera Calibration & Homography]], [[Gaussian & Laplacian Pyramids]], [[Image Filtering & 2D Fourier Transform]], [[Pinhole & Thin-Lens Optics]]
- **Synthesized Sources:** [[Source - Antonio Torralba, Phillip Isola, and William Freeman]], [[Source - Dive into Deep Learning]]
