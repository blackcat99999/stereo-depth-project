import cv2
import numpy as np
import glob

# ==============================
# 1. Checkerboard Configuration
# ==============================

CHECKERBOARD = (9, 6)   # Inner corners (columns, rows)
square_size = 1.0       # Real-world square size (set in meters or cm)

# Prepare object points (0,0,0), (1,0,0), ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],
                       0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_left = []
imgpoints_right = []

# ==============================
# 2. Load Calibration Images
# ==============================

images_left = sorted(glob.glob('calibration/left_*.jpg'))
images_right = sorted(glob.glob('calibration/right_*.jpg'))

if len(images_left) == 0 or len(images_right) == 0:
    print("No calibration images found.")
    exit()

print("Found", len(images_left), "image pairs.")

# ==============================
# 3. Detect Checkerboard Corners
# ==============================

for left_path, right_path in zip(images_left, images_right):

    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if retL and retR:

        # Refine corners
        cornersL = cv2.cornerSubPix(
            grayL, cornersL, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        cornersR = cv2.cornerSubPix(
            grayR, cornersR, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

        print("Corners detected in pair:", left_path)

print("Total valid pairs:", len(objpoints))

if len(objpoints) < 5:
    print("Not enough valid pairs. Capture more images.")
    exit()

# ==============================
# 4. Calibrate Each Camera
# ==============================

retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, imgpoints_left, grayL.shape[::-1], None, None)

retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints, imgpoints_right, grayR.shape[::-1], None, None)

print("\nLeft Camera Matrix:\n", mtxL)
print("\nRight Camera Matrix:\n", mtxR)

# ==============================
# 5. Stereo Calibration
# ==============================

criteria = (cv2.TERM_CRITERIA_MAX_ITER +
            cv2.TERM_CRITERIA_EPS, 100, 1e-5)

flags = cv2.CALIB_FIX_INTRINSIC

ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    mtxL,
    distL,
    mtxR,
    distR,
    grayL.shape[::-1],
    criteria=criteria,
    flags=flags
)

print("\nStereo Calibration RMS error:", ret)

# ==============================
# 6. Compute Baseline
# ==============================

baseline = np.linalg.norm(T)
print("\nBaseline (in square_size units):", baseline)

# ==============================
# 7. Save Calibration Results
# ==============================

np.savez("stereo_calibration.npz",
         CM1=CM1, dist1=dist1,
         CM2=CM2, dist2=dist2,
         R=R, T=T, E=E, F=F)

print("\nCalibration saved as stereo_calibration.npz")
print("Calibration Complete.")