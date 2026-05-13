import cv2
import numpy as np
import time
from ultralytics import YOLO

# -----------------------------
# Start timer for FPS
# -----------------------------
start_time = time.time()

# -----------------------------
# 1️⃣ Load Images
# -----------------------------
left_img = cv2.imread("dataset/left.png")
right_img = cv2.imread("dataset/right.png")

if left_img is None or right_img is None:
    print("Error: Could not load images.")
    exit()

gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 2️⃣ Compute Disparity
# -----------------------------
print("Computing disparity using StereoSGBM...")

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*8,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

# ⭐ Improvement 1 — Bilateral filtering (better than median blur)
disparity = cv2.bilateralFilter(disparity, 9, 75, 75)

disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp_vis = np.uint8(disp_vis)

cv2.imshow("Disparity Map", disp_vis)

# -----------------------------
# 3️⃣ Read Calibration
# -----------------------------
with open("dataset/calib.txt", "r") as f:
    lines = f.readlines()

for line in lines:
    if line.startswith("P0:"):
        P0 = np.array(line.split()[1:], dtype=np.float32).reshape(3,4)
    if line.startswith("P1:"):
        P1 = np.array(line.split()[1:], dtype=np.float32).reshape(3,4)

focal_length = P0[0,0]
baseline = abs(P1[0,3] - P0[0,3]) / focal_length

print("Focal Length:", focal_length)
print("Baseline:", baseline)

# -----------------------------
# 4️⃣ Compute Depth
# -----------------------------
depth = np.zeros(disparity.shape, np.float32)

valid = disparity > 1
depth[valid] = (focal_length * baseline) / disparity[valid]

# ⭐ Improvement 2 — Remove unrealistic depth
depth[(depth < 1) | (depth > 80)] = 0

np.save("output/depth_raw.npy", depth)

# -----------------------------
# 5️⃣ Depth Visualization
# -----------------------------
depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_norm = np.uint8(depth_norm)

depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

display_depth = depth_color.copy()

# -----------------------------
# ⭐ Mouse Callback Function
# -----------------------------
def mouse_callback(event, x, y, flags, param):
    global display_depth

    if event == cv2.EVENT_LBUTTONDOWN:

        # ⭐ Improvement 3 — median depth region
        region = depth[max(0,y-5):y+6, max(0,x-5):x+6]

        valid = region[region > 0]

        if len(valid) > 0:
            distance = np.median(valid)
        else:
            print("No valid depth at this point.")
            return

        print(f"Distance at ({x},{y}) = {distance:.2f} meters")

        display_depth = depth_color.copy()

        cv2.circle(display_depth, (x, y), 5, (255,255,255), -1)

        cv2.putText(
            display_depth,
            f"{distance:.2f} m",
            (x+10, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.imshow("Depth Map - Click to Measure", display_depth)

cv2.namedWindow("Depth Map - Click to Measure")
cv2.setMouseCallback("Depth Map - Click to Measure", mouse_callback)

cv2.imshow("Depth Map - Click to Measure", display_depth)

# -----------------------------
# 6️⃣ Load YOLOv8
# -----------------------------
print("Loading YOLOv8 model...")

model = YOLO("yolov8n.pt")

results = model(left_img)

boxes = results[0].boxes.xyxy.cpu().numpy()
classes = results[0].boxes.cls.cpu().numpy()
scores = results[0].boxes.conf.cpu().numpy()

names = model.names

output_img = left_img.copy()

# -----------------------------
# 7️⃣ Object Distance Estimation
# -----------------------------
for i, box in enumerate(boxes):

    confidence = scores[i]

    if confidence < 0.15:
        continue

    x1, y1, x2, y2 = map(int, box)

    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)

    # median region sampling
    region = depth[max(0,cy-5):cy+6, max(0,cx-5):cx+6]

    valid_depth = region[region > 0]

    if len(valid_depth) > 0:
        distance = np.median(valid_depth)
    else:
        distance = 0

    label = names[int(classes[i])]

    text = f"{label} {confidence:.2f} : {distance:.2f} m"

    cv2.rectangle(output_img,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.circle(output_img,(cx,cy),5,(0,0,255),-1)

    # ⭐ Bonus visualization line
    cv2.line(
        output_img,
        (left_img.shape[1]//2, left_img.shape[0]),
        (cx, cy),
        (255,255,0),
        1
    )

    cv2.putText(
        output_img,
        text,
        (x1,y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )

# -----------------------------
# 8️⃣ FPS Calculation
# -----------------------------
end_time = time.time()
fps = 1/(end_time-start_time)

cv2.putText(
    output_img,
    f"FPS: {fps:.2f}",
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0,255,255),
    2
)

# -----------------------------
# 9️⃣ Save Output
# -----------------------------
cv2.imwrite("output/final_result.png", output_img)

# -----------------------------
# 🔟 Show Result
# -----------------------------
cv2.imshow("Object Distance Detection", output_img)

print("Result saved to output/final_result.png")
print("Click anywhere on the depth map to measure distance.")
print("Press any key to exit.")

cv2.waitKey(0)
cv2.destroyAllWindows()