# Object Distance Estimation Using Stereo Vision and YOLOv8

## 1. Introduction
[cite_start]This project implements an integrated computer vision pipeline that detects objects in a scene and accurately estimates their physical distance from a camera setup[cite: 17]. [cite_start]By fusing traditional stereo vision techniques for depth estimation with deep learning for object recognition, the system provides a reliable 3D understanding of the environment[cite: 18].

## 2. Methodology
[cite_start]The system utilizes a pair of rectified grayscale stereo images to reconstruct 3D geometry[cite: 20, 29].

### Core Pipeline
1.  [cite_start]**Disparity Computation:** Uses the **Stereo Semi-Global Block Matching (StereoSGBM)** algorithm to find pixel shifts between left and right images[cite: 21, 31].
2.  [cite_start]**Post-Processing:** Applies **Bilateral Filtering** to the disparity map to reduce noise while preserving sharp object edges[cite: 32].
3.  [cite_start]**Depth Mapping:** Translates disparity into real-world depth through mathematical triangulation using camera calibration parameters[cite: 22, 33].
4.  [cite_start]**Object Detection:** Employs **YOLOv8** on the left image to detect vehicles and pedestrians[cite: 23, 34].
5.  [cite_start]**Distance Extraction:** Maps the center coordinates of YOLO bounding boxes to the depth map and extracts the median distance[cite: 24, 35].

### Mathematical Model
The depth $Z$ of a given pixel is calculated using the following stereo vision formula:

$$Z = \frac{f \cdot B}{d}$$

Where:
* [cite_start]$f$ is the focal length (707.09 pixels)[cite: 38, 94].
* [cite_start]$B$ is the baseline distance (0.537 meters)[cite: 38, 94].
* [cite_start]$d$ is the corresponding disparity[cite: 39].

[cite_start]To handle noise, the system extracts the **median depth** of a small pixel grid ($11 \times 11$) around the detected object's center[cite: 43].

## 3. Core Implementation Highlights
[cite_start]The implementation utilizes OpenCV for stereo processing and the Ultralytics library for YOLOv8 inference[cite: 47, 74].

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
