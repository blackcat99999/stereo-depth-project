# Object Distance Estimation Using Stereo Vision and YOLOv8

## 1. Introduction
This project implements an integrated computer vision pipeline that detects objects in a scene and accurately estimates their physical distance from a camera setup. By fusing traditional stereo vision techniques for depth estimation with deep learning for object recognition, the system provides a reliable 3D understanding of the environment.

## 2. Methodology
The system utilizes a pair of rectified grayscale stereo images to reconstruct 3D geometry.

### Core Pipeline
1.  **Disparity Computation:** Uses the **Stereo Semi-Global Block Matching (StereoSGBM)** algorithm to find pixel shifts between left and right images.
2.  **Post-Processing:** Applies **Bilateral Filtering** to the disparity map to reduce noise while preserving sharp object edges.
3.  **Depth Mapping:** Translates disparity into real-world depth through mathematical triangulation using camera calibration parameters.
4.  **Object Detection:** Employs **YOLOv8** on the left image to detect vehicles and pedestrians.
5.  **Distance Extraction:** Maps the center coordinates of YOLO bounding boxes to the depth map and extracts the median distance.

### Mathematical Model
The depth $Z$ of a given pixel is calculated using the following stereo vision formula:

$$Z = \frac{f \cdot B}{d}$$

Where:
* $f$ is the focal length (707.09 pixels).
* $B$ is the baseline distance (0.537 meters).
* $d$ is the corresponding disparity.

To handle noise, the system extracts the **median depth** of a small pixel grid ($11 \times 11$) around the detected object's center.

## 3. Core Implementation Highlights
The implementation utilizes OpenCV for stereo processing and the Ultralytics library for YOLOv8 inference.

### StereoSGBM Configuration
```python
stereo = cv2.StereoSGBM_create(
    minDisparity = 0,
    numDisparities = 16*8,
    blockSize = 5,
    P1 = 8*3*5*2,
    P2 = 32*3*5*2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
) [cite: 47-57]
