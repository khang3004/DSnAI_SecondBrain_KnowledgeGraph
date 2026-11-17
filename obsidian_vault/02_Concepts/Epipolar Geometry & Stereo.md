---
type: theorem
domain: "D03"
aliases: 
  - "Epipolar Geometry"
  - "Fundamental Matrix"
  - "Stereo Vision"
part_of: 
  - "[[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]"
  - "[[Cheatsheet - Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[Camera Calibration & Homography]]"
enables: 
  - "[[Radiance Fields (NeRF & 3DGS)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Camera Calibration & Homography]]"
  - "[[Optical Flow (Lucas-Kanade & Horn-Schunck)]]"
  - "[[Radiance Fields (NeRF & 3DGS)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Antonio Torralba, Phillip Isola, and William Freeman]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - theorem
  - domain-03
---

# Epipolar Geometry & Stereo

> [!abstract] Definition (Theorem)
> Two-view coplanarity constrains matches to 1D epipolar lines via Essential E / Fundamental F matrices and yields depth Z = fB/d.

$$
\tilde{\boldsymbol{x}}'^\top \boldsymbol{F}\tilde{\boldsymbol{x}} = 0, \qquad Z = \frac{fB}{x_L - x_R}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Camera Calibration & Homography]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Radiance Fields (NeRF & 3DGS)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Camera Calibration & Homography]], [[Optical Flow (Lucas-Kanade & Horn-Schunck)]], [[Radiance Fields (NeRF & 3DGS)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Visual Cheatsheet:** [[Cheatsheet - Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Referenced Across Monographs:** [[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Primary Textbooks:** [[Source - Antonio Torralba, Phillip Isola, and William Freeman]], [[Source - Understanding Deep Learning]]
