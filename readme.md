# CVML: Camera, Perception & Medical Image Processing

A 20-hour introductory course covering the fundamentals of how cameras form images, how
machines perceive scenes (detection, segmentation, tracking), and how these ideas carry over
to medical imaging problems such as segmentation and registration.

## Audience & Prerequisites

- Comfortable with Python and NumPy
- Linear algebra (matrices, SVD, least squares) and basic probability
- No prior computer vision or deep learning experience required

## Learning Outcomes

By the end of the course, students will be able to:

1. Explain the pinhole camera model, intrinsic/extrinsic parameters, and lens distortion.
2. Calibrate a camera and use the result for undistortion, pose estimation, and stereo depth.
3. Build classic and learned pipelines for feature matching, detection, segmentation, and tracking.
4. Load, preprocess, and visualize medical images (DICOM / NIfTI).
5. Perform rigid, affine, and deformable registration of medical images and evaluate the result.

## Schedule (10 sessions × 2 hours)

| # | Session | Topics | Lab |
|---|---------|--------|-----|
| 1 | Images & Image Processing | Pixels, color spaces, histograms, convolution, filtering, edges | Filters and edge detection with OpenCV |
| 2 | Camera Model | Pinhole model, homogeneous coordinates, intrinsics/extrinsics, projection, lens distortion | Project 3D points into a virtual camera |
| 3 | Camera Calibration | Zhang's method, checkerboards/ChArUco, reprojection error, undistortion, PnP pose | Calibrate your own webcam/phone camera |
| 4 | Features & Multi-View Geometry | Keypoints (SIFT/ORB), matching, homography, RANSAC, epipolar geometry, stereo, triangulation | Panorama stitching; stereo depth map |
| 5 | Deep Learning for Vision | CNNs, training loop, transfer learning, data augmentation, evaluation | Fine-tune a classifier in PyTorch |
| 6 | Object Detection | Sliding windows → two-stage (Faster R-CNN) → one-stage (YOLO) → DETR; IoU, NMS, mAP | Run and evaluate a pretrained detector |
| 7 | Segmentation | Semantic / instance / panoptic, FCN, U-Net, Mask R-CNN, foundation models (SAM); Dice, IoU | Train a small U-Net; prompt SAM |
| 8 | Tracking | Optical flow (Lucas–Kanade), Kalman filter, Hungarian matching, SORT / ByteTrack, MOT metrics | Multi-object tracking on a video |
| 9 | Medical Imaging Basics | Modalities (X-ray, CT, MRI, US), DICOM/NIfTI, spacing & orientation, intensity normalization, medical segmentation (nnU-Net / MONAI) | Load and segment a CT/MRI volume |
| 10 | Medical Image Registration | Transforms (rigid, affine, B-spline, deformable), similarity metrics (SSD, NCC, mutual information), optimization, learning-based registration (VoxelMorph); evaluation (TRE, Dice) | Register two scans with SimpleITK; final project presentations |

## Tools

- Python 3.10+, NumPy, Matplotlib, Jupyter
- OpenCV — image processing, calibration, features, stereo
- PyTorch / torchvision — deep learning
- Ultralytics YOLO, Segment Anything — detection & segmentation
- SimpleITK, nibabel, pydicom, MONAI — medical imaging

## Assessment

- Weekly labs — 50%
- Final mini-project (perception or medical imaging track) — 40%
- Participation — 10%

## Repository Layout (planned)

```
cvml/
├── lectures/   # slides and lecture notes per session
├── labs/       # Jupyter notebooks for hands-on exercises
├── data/       # small sample datasets (or download scripts)
└── projects/   # final project descriptions and starter code
```

## References

- R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed.
- R. Hartley & A. Zisserman, *Multiple View Geometry in Computer Vision*
- S. Prince, *Understanding Deep Learning*
- Sotiras, Davatzikos & Paragios, "Deformable Medical Image Registration: A Survey," IEEE TMI 2013
