import cv2, os, argparse

def extract(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:05d}.png"), frame)
        i += 1
    cap.release()
    print(f"Extracted {i} frames to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    extract(args.video, args.out_dir)
