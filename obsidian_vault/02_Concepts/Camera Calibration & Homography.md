---
type: object
domain: "D03"
aliases: 
  - "Homography"
  - "Camera Matrix"
part_of: 
  - "[[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]"
  - "[[Cheatsheet - Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[Pinhole & Thin-Lens Optics]]"
  - "[[Singular Value Decomposition (SVD)]]"
enables: 
  - "[[Epipolar Geometry & Stereo]]"
  - "[[Radiance Fields (NeRF & 3DGS)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Epipolar Geometry & Stereo]]"
  - "[[Pinhole & Thin-Lens Optics]]"
  - "[[Radiance Fields (NeRF & 3DGS)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Antonio Torralba, Phillip Isola, and William Freeman]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - object
  - domain-03
---

# Camera Calibration & Homography

> [!abstract] Definition (Object)
> Homogeneous projection P = K[R|t] and 8-DoF planar/rotational homography H estimated via Hartley-normalized DLT + RANSAC.

$$
\tilde{\boldsymbol{x}} \sim \boldsymbol{K}[\boldsymbol{R}\mid\boldsymbol{t}]\tilde{\boldsymbol{X}}, \qquad \tilde{\boldsymbol{x}}' \sim \boldsymbol{H}\tilde{\boldsymbol{x}}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Pinhole & Thin-Lens Optics]], [[Singular Value Decomposition (SVD)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Epipolar Geometry & Stereo]], [[Radiance Fields (NeRF & 3DGS)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Epipolar Geometry & Stereo]], [[Pinhole & Thin-Lens Optics]], [[Radiance Fields (NeRF & 3DGS)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Visual Cheatsheet:** [[Cheatsheet - Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Referenced Across Monographs:** [[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Primary Textbooks:** [[Source - Antonio Torralba, Phillip Isola, and William Freeman]], [[Source - Understanding Deep Learning]]
