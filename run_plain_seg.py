"""
run_plain_seg.py
----------------
Runs plain segmentation (no defense) on frames, both clean and patched,
to visualize the effect of the adversarial patch.
Works with both torchvision segmentation models (DeepLabV3, FCN, etc.)
and your custom MinimalBiSeNet.
"""

import os, cv2, torch
import numpy as np
from bisenet_model_loader import load_bisenet_and_preprocess

def run_plain_seg(frames_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model, preprocess_fn = load_bisenet_and_preprocess()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for name in sorted(os.listdir(frames_dir)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        img = cv2.imread(os.path.join(frames_dir, name))
        if img is None:
            continue

        inp, meta = preprocess_fn(img)
        inp = inp.to(device)

        with torch.no_grad():
            out = model(inp)

        # Handle output type (dict or tensor)
        if isinstance(out, dict):
            logits = out["out"][0].detach().cpu().numpy()  # torchvision models
        elif torch.is_tensor(out):
            logits = out[0].detach().cpu().numpy()         # your custom model
        else:
            raise TypeError(f"Unexpected output type: {type(out)}")

        pred = logits.argmax(0).astype(np.uint8)
        # Map to colors for visualization
        color = cv2.applyColorMap((pred * 255 // max(1, pred.max())).astype(np.uint8), cv2.COLORMAP_JET)
        color = cv2.resize(color, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(out_dir, name), color)
        print(f"Saved {name}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    run_plain_seg(args.frames_dir, args.out)
