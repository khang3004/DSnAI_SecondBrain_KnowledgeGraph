---
type: architecture
domain: "D03"
aliases: 
  - "NeRF"
  - "3D Gaussian Splatting"
part_of: 
  - "[[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]"
  - "[[Cheatsheet - Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[Camera Calibration & Homography]]"
  - "[[Epipolar Geometry & Stereo]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Camera Calibration & Homography]]"
  - "[[Epipolar Geometry & Stereo]]"
  - "[[MLP & Universal Approximation]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Antonio Torralba, Phillip Isola, and William Freeman]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - architecture
  - domain-03
---

# Radiance Fields (NeRF & 3DGS)

> [!abstract] Definition (Architecture)
> Continuous neural volume density + view-dependent color rendered via differentiable ray-marching, or explicit 3D Gaussian splats.

$$
\hat{\boldsymbol{C}}(\boldsymbol{r}) = \int_{t_n}^{t_f} T(t)\sigma(\boldsymbol{r}(t))\boldsymbol{c}(\boldsymbol{r}(t),\boldsymbol{d})\,\mathrm{d}t
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Camera Calibration & Homography]], [[Epipolar Geometry & Stereo]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Camera Calibration & Homography]], [[Epipolar Geometry & Stereo]], [[MLP & Universal Approximation]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Visual Cheatsheet:** [[Cheatsheet - Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Referenced Across Monographs:** [[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Primary Textbooks:** [[Source - Antonio Torralba, Phillip Isola, and William Freeman]], [[Source - Understanding Deep Learning]]
