# import cv2, os, numpy as np

# SRC = "data/clean_frames"
# DST = "data/stream"
# os.makedirs(DST, exist_ok=True)

# # Make a simple high-contrast patch if you don't have patch.png
# if not os.path.exists("patch.png"):
#     p = np.zeros((60,60,3), np.uint8)
#     p[:30,:,:] = (255,255,255)
#     cv2.imwrite("patch.png", p)

# patch = cv2.imread("patch.png")

# for name in sorted(os.listdir(SRC)):
#     if not name.lower().endswith((".png",".jpg",".jpeg",".bmp")): 
#         continue
#     img = cv2.imread(os.path.join(SRC, name))
#     if img is None: 
#         continue
#     h, w = img.shape[:2]
#     # Paste patch at fixed location so it persists across frames
#     y1, x1 = max(10, h//20), max(10, w//20)
#     ph, pw = patch.shape[:2]
#     y2, x2 = min(h, y1+ph), min(w, x1+pw)
#     img[y1:y2, x1:x2] = cv2.resize(patch, (x2-x1, y2-y1))
#     cv2.imwrite(os.path.join(DST, name), img)

# print("Patched stream written to", DST)

import os, cv2, numpy as np

"""
Create a simple high-contrast patch and paste it onto frames.
Input:  data/clean_frames
Output: data/stream
"""

SRC = "data/clean_frames"
DST = "data/stream"
os.makedirs(DST, exist_ok=True)

if not os.path.exists("patch.png"):
    p = np.zeros((60, 60, 3), np.uint8)
    p[:30, :, :] = (255, 255, 255)
    cv2.imwrite("patch.png", p)

patch = cv2.imread("patch.png")

for name in sorted(os.listdir(SRC)):
    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        continue
    img = cv2.imread(os.path.join(SRC, name))
    if img is None:
        continue
    h, w = img.shape[:2]
    y1, x1 = max(10, h // 20), max(10, w // 20)
    ph, pw = patch.shape[:2]
    y2, x2 = min(h, y1 + ph), min(w, x1 + pw)
    img[y1:y2, x1:x2] = cv2.resize(patch, (x2 - x1, y2 - y1))
    cv2.imwrite(os.path.join(DST, name), img)

print("Patched stream written to", DST)
