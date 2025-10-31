"""
compare_seg_videos.py
---------------------
Combine clean, attacked, and defended segmentation results side-by-side
for qualitative comparison.
"""

import os, cv2
import numpy as np
from natsort import natsorted

def stack_frames(clean_dir, attack_dir, defense_dir, out_path, fps=20):
    files = [f for f in os.listdir(clean_dir) if f.lower().endswith((".png",".jpg"))]
    files = natsorted(files)
    if not files:
        raise RuntimeError("No frames found in clean_dir")

    first = cv2.imread(os.path.join(clean_dir, files[0]))
    h, w = first.shape[:2]
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # writer = cv2.VideoWriter(out_path, fourcc, fps, (w*3, h))  # 3 panels horizontally
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # common for .avi
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w*3, h))

    print(f"Writing {len(files)} frames to {out_path}")

    for f in files:
        c1 = cv2.imread(os.path.join(clean_dir, f))
        c2 = cv2.imread(os.path.join(attack_dir, f))
        c3 = cv2.imread(os.path.join(defense_dir, f))
        if c1 is None or c2 is None or c3 is None:
            continue
        vis = np.hstack([c1, c2, c3])
        # Add labels
        cv2.putText(vis, "Clean", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(vis, "Attacked", (w+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(vis, "Defended", (2*w+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        writer.write(vis)

    writer.release()
    print(f"[DONE] Video saved to {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--attack_dir", required=True)
    ap.add_argument("--defense_dir", required=True)
    ap.add_argument("--out", default="comparison.mp4")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()
    stack_frames(args.clean_dir, args.attack_dir, args.defense_dir, args.out, fps=args.fps)
