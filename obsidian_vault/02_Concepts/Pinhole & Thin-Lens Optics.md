---
type: object
domain: "D03"
aliases: 
  - "Perspective Projection"
  - "Thin-Lens Equation"
part_of: 
  - "[[Image Formation, Lenses, Color Spaces & Fourier Image Filtering]]"
  - "[[Cheatsheet - Image Formation, Lenses, Color Spaces & Fourier Image Filtering]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: []
enables: 
  - "[[Camera Calibration & Homography]]"
generalizes: []
special_case_of: []
related: 
  - "[[Camera Calibration & Homography]]"
  - "[[Image Filtering & 2D Fourier Transform]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Antonio Torralba, Phillip Isola, and William Freeman]]"
  - "[[Source - Dive into Deep Learning]]"
tags:
  - object
  - domain-03
---

# Pinhole & Thin-Lens Optics

> [!abstract] Definition (Object)
> Perspective 3D-to-2D projection x=fX/Z, Gaussian thin-lens law 1/z_o + 1/z_i = 1/f, Lambertian shading, and linear vs. gamma sRGB spaces.

$$
x = f\frac{X}{Z}, \quad y = f\frac{Y}{Z}, \qquad \frac{1}{z_o} + \frac{1}{z_i} = \frac{1}{f}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Camera Calibration & Homography]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Camera Calibration & Homography]], [[Image Filtering & 2D Fourier Transform]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Image Formation, Lenses, Color Spaces & Fourier Image Filtering]]
- **Visual Cheatsheet:** [[Cheatsheet - Image Formation, Lenses, Color Spaces & Fourier Image Filtering]]
- **Referenced Across Monographs:** [[Image Formation, Lenses, Color Spaces & Fourier Image Filtering]]
- **Primary Textbooks:** [[Source - Antonio Torralba, Phillip Isola, and William Freeman]], [[Source - Dive into Deep Learning]]
