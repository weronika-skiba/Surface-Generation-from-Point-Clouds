# Surface-Generation-from-Point-Clouds

This project implements two methods for constructing an implicit surface representation from a 3D point cloud by defining a **signed distance function (SDF)** \( f(x, y, z) \). The surface is extracted at the zero level set \( f(x, y, z) = 0 \) using the **Marching Cubes** algorithm.

## Overview

Given a point cloud \( P = \{p_1, p_2, \dots, p_n\} \), the goal is to define an implicit function \( f(x, y, z) \) that measures the signed distance to the surface approximated by the points. Two methods are provided:

### Method A: Geometric Tangent Plane Approximation

- For each query point \((x, y, z)\), find the nearest point \( p_i \in P \).
- Estimate the surface normal at \( p_i \) (e.g., using PCA on neighboring points).
- Compute the signed distance from \((x, y, z)\) to the tangent plane at \( p_i \).
- This yields a fast, local, and interpretable approximation of the SDF.

### Method B: Neural Network Approximation

- Train a neural network to learn the signed distance function \( f(x, y, z) \) from the point cloud.
- The network takes 3D coordinates as input and outputs the signed distance.
- Training data includes samples near the surface with corresponding signed distances.
- This approach can model complex shapes and generalize smoothly.

## Surface Extraction

Once the signed distance function is defined, the surface is extracted by evaluating \( f(x, y, z) \) on a 3D grid and applying the **Marching Cubes** algorithm to extract the isosurface at zero level.

