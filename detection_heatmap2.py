import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

frames_per_row = 5       
columns_per_side = 10     

video_path = "mere.MP4"
model = YOLO(r"C:\Users\andre\Desktop\code\runs\detect\train16\weights\best.pt")
cap = cv2.VideoCapture(video_path)

frame_rate = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / frame_rate

left_limit = int(0.10 * width)
right_limit = int(0.90 * width)

heatmap_left = []
heatmap_right = []

current_row_left = np.zeros(columns_per_side)
current_row_right = np.zeros(columns_per_side)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))
    results = model(frame)

    left_counts = np.zeros(columns_per_side)
    right_counts = np.zeros(columns_per_side)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if conf < 0.4:
            continue

        center_x = (x1 + x2) // 2

        if center_x < left_limit:
            col_index = int(center_x / (left_limit / columns_per_side))
            if col_index < columns_per_side:
                left_counts[col_index] += 1
        elif center_x > right_limit:
            col_index = int((center_x - right_limit) / ((width - right_limit) / columns_per_side))
            if col_index < columns_per_side:
                right_counts[col_index] += 1

    current_row_left += left_counts
    current_row_right += right_counts
    frame_count += 1

    if frame_count % frames_per_row == 0:
        heatmap_left.append(current_row_left)
        heatmap_right.append(current_row_right)
        current_row_left = np.zeros(columns_per_side)
        current_row_right = np.zeros(columns_per_side)

cap.release()

heatmap_left = np.array(heatmap_left)
heatmap_right = np.array(heatmap_right)

heatmap = np.hstack((heatmap_left, heatmap_right))

plt.figure(figsize=(10, 8))
plt.imshow(heatmap, cmap='hot', interpolation='nearest', origin='lower', aspect='auto')

num_rows = heatmap.shape[0]
interval_sec = frames_per_row / frame_rate
tick_interval = 5  

yticks = np.arange(0, num_rows, int(tick_interval / interval_sec))
yticklabels = [f"{int(i * interval_sec)}s" for i in yticks]

plt.yticks(yticks, yticklabels)
plt.xlabel("Pozitie pe latime (stanga si dreapta)")
plt.ylabel("Timp")
plt.title("Heatmap densitate mere pe margini")
plt.colorbar(label="Numar detetii")
plt.tight_layout()
plt.savefig("heatmap_densitate_mere.png")
plt.show()
