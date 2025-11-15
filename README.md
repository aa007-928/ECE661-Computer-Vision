# ECE 661 – Computer Vision
This repository contains my solutions to homeworks completed for ECE661: Computer Vision, a graduate-level course focused on geometric vision, feature detection, image segmentation, 3D reconstruction, and modern deep learning techniques.
Each homework builds progressively toward a full understanding of classical and modern computer vision pipelines.
---

## HW1 — Projective Geometry Fundamentals
- Homogeneous coordinates, points at infinity, and conic geometry  
- Line/conic intersections using cross products  
- Tangent & polar line computation  
- Ray–triangle intersection (geometry-based hit detection)

## HW2 — Homographies & Planar Image Mapping
- Manual 3×3 homography estimation from point correspondences  
- Image warping and ROI projection across viewpoints  
- Composition of chained homographies  
- Comparison of affine vs full projective transforms  
- Parameterized tilt/rotation homographies

## HW3 — Metric Rectification
- Removing projective + affine distortions from images  
- Vanishing-line based rectification (two-step method)  
- Dual-conic one-step rectification approach  
- Restoring true geometry from distorted photos

## HW4 — Interest Point Detection & Matching
- Custom Harris Corner Detector (multi-scale)  
- SSD/NCC matching from scratch  
- SIFT/SURF feature matching (OpenCV)  
- SuperPoint + SuperGlue for deep keypoint detection & robust matching

## HW5 — Robust Homography & Panorama Stitching
- Automatic feature matching pipeline  
- RANSAC for outlier rejection  
- SVD-based homography estimation + LM refinement  
- Multi-image panoramic stitching

## HW6 — Image Segmentation & Contours
- Otsu thresholding on RGB channels  
- Texture-based segmentation via sliding-window variance  
- Iterative refinement for improved foreground extraction  
- Custom contour extraction from binary masks

## HW7 — Texture Descriptors & Image Classification
- LBP descriptors on Hue channel  
- Gram matrix descriptors via VGG & ResNet features  
- SVM-based weather image classification  
- Channel-normalization (AdaIN-style) texture descriptor

## HW8 — Camera Calibration (Zhang’s Method)
- Canny + Hough-based corner detection  
- Intrinsic & extrinsic calibration via homographies  
- Reprojection-error evaluation with LM refinement  
- 3D visualization of calibrated camera poses  
- Radial distortion parameter estimation

## HW9 — Stereo Vision & 3D Reconstruction
- Fundamental matrix estimation (8-point algorithm)  
- Stereo rectification and correspondence search  
- Dense stereo using Census Transform  
- Projective 3D reconstruction from stereo pair  
- Depth-map–based automatic dense correspondences

## HW10 — Dimensionality Reduction & Object Detection
- PCA & LDA embeddings + nearest-neighbor classification  
- Autoencoder latent-space representations (3, 8, 16 dims)  
- UMAP visualization of feature clusters  
- Cascaded AdaBoost classifier for object detection

---
