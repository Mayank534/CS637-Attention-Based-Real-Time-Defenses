"""
frames_to_video.py
------------------

Combine sequentially named image frames (e.g. frame_00001.png, frame_00002.png, …)
into a video file using OpenCV.

Usage examples:
  # Convert visualization frames into a video (default 30 FPS)
  python frames_to_video.py --frames_dir out_vis --out out_vis_video.mp4

  # Specify FPS or codec
  python frames_to_video.py --frames_dir out_vis_video --out result.avi --fps 24 --codec XVID
"""

import os
import cv2
import argparse
from natsort import natsorted  # safer sorting of frame_1, frame_2, …

def make_video_from_frames(frames_dir: str, out_path: str, fps: int = 30, codec: str = "mp4v"):
    # Collect frames (sorted naturally)
    frame_names = [f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    if not frame_names:
        raise RuntimeError(f"No frames found in {frames_dir}")
    frame_names = natsorted(frame_names)

    # Read first frame to get size
    first_frame = cv2.imread(os.path.join(frames_dir, frame_names[0]))
    if first_frame is None:
        raise RuntimeError(f"Cannot read {frame_names[0]}")
    height, width = first_frame.shape[:2]

    # Setup video writer
    # fourcc = cv2.VideoWriter_fourcc(*codec)
    # writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # common for .avi
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))


    print(f"[INFO] Writing {len(frame_names)} frames from '{frames_dir}' to '{out_path}' "
          f"({width}x{height}, {fps} FPS, codec={codec})")

    for name in frame_names:
        path = os.path.join(frames_dir, name)
        frame = cv2.imread(path)
        if frame is None:
            print(f"[WARN] Skipping unreadable frame {name}")
            continue
        writer.write(frame)

    writer.release()
    print(f"[DONE] Video saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rebuild a video from frames.")
    ap.add_argument("--frames_dir", type=str, required=True, help="Input frames directory")
    ap.add_argument("--out", type=str, required=True, help="Output video file path (e.g. out.mp4)")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second")
    ap.add_argument("--codec", type=str, default="mp4v", help="FourCC codec (e.g. mp4v, XVID, MJPG)")
    args = ap.parse_args()

    make_video_from_frames(args.frames_dir, args.out, fps=args.fps, codec=args.codec)
