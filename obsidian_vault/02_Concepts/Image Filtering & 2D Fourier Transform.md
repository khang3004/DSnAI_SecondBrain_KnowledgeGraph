---
type: theorem
domain: "D03"
aliases: 
  - "2D Fourier Transform"
  - "Nyquist Sampling"
part_of: 
  - "[[Image Formation, Lenses, Color Spaces & Fourier Image Filtering]]"
  - "[[Cheatsheet - Image Formation, Lenses, Color Spaces & Fourier Image Filtering]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[CNN & Equivariance]]"
enables: 
  - "[[Gaussian & Laplacian Pyramids]]"
  - "[[Optical Flow (Lucas-Kanade & Horn-Schunck)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Gaussian & Laplacian Pyramids]]"
  - "[[Pinhole & Thin-Lens Optics]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Antonio Torralba, Phillip Isola, and William Freeman]]"
  - "[[Source - Dive into Deep Learning]]"
tags:
  - theorem
  - domain-03
---

# Image Filtering & 2D Fourier Transform

> [!abstract] Definition (Theorem)
> LSI image filtering is diagonalized by 2D complex exponentials via the Convolution Theorem; Nyquist sampling requires low-pass anti-aliasing.

$$
\mathcal{F}\{I * h\} = \hat{I} \odot \hat{h}, \qquad |u|_{\max} \le \frac{1}{2\Delta x}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[CNN & Equivariance]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Gaussian & Laplacian Pyramids]], [[Optical Flow (Lucas-Kanade & Horn-Schunck)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Gaussian & Laplacian Pyramids]], [[Pinhole & Thin-Lens Optics]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Image Formation, Lenses, Color Spaces & Fourier Image Filtering]]
- **Visual Cheatsheet:** [[Cheatsheet - Image Formation, Lenses, Color Spaces & Fourier Image Filtering]]
- **Referenced Across Monographs:** [[Image Formation, Lenses, Color Spaces & Fourier Image Filtering]]
- **Primary Textbooks:** [[Source - Antonio Torralba, Phillip Isola, and William Freeman]], [[Source - Dive into Deep Learning]]
