---
type: algorithm
domain: "D03"
aliases: 
  - "Optical Flow"
  - "Lucas-Kanade"
part_of: 
  - "[[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]"
  - "[[Cheatsheet - Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[Gaussian & Laplacian Pyramids]]"
  - "[[Image Filtering & 2D Fourier Transform]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Epipolar Geometry & Stereo]]"
  - "[[Gaussian & Laplacian Pyramids]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Antonio Torralba, Phillip Isola, and William Freeman]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - algorithm
  - domain-03
---

# Optical Flow (Lucas-Kanade & Horn-Schunck)

> [!abstract] Definition (Algorithm)
> Estimates dense 2D pixel velocity from brightness constancy I_x u + I_y v + I_t = 0 via local window least squares or global smoothness.

$$
\nabla I^\top \boldsymbol{u} + I_t = 0, \qquad \Bigl(\sum_W \nabla I \nabla I^\top\Bigr)\boldsymbol{u} = -\sum_W \nabla I\, I_t
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Gaussian & Laplacian Pyramids]], [[Image Filtering & 2D Fourier Transform]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Epipolar Geometry & Stereo]], [[Gaussian & Laplacian Pyramids]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Visual Cheatsheet:** [[Cheatsheet - Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Referenced Across Monographs:** [[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Primary Textbooks:** [[Source - Antonio Torralba, Phillip Isola, and William Freeman]], [[Source - Understanding Deep Learning]]
