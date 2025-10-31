# """
# Combined attention-based defense implementation supporting both calibration and defense modes.

# Usage:
#     # Calibration mode:
#     python proj.py --frames_dir data/clean_frames --out baseline.pt

#     # Defense mode:
#     python proj.py --frames_dir data/stream --baseline baseline.pt --save_vis out_vis
# """

# from __future__ import annotations
# import os
# import time
# import argparse
# import torch
# import torch.nn.functional as F
# import cv2
# import numpy as np
# from dataclasses import dataclass
# from typing import Dict, List, Tuple
# from bisenet_model_loader import load_bisenet_and_preprocess, get_shallow_module, load_model
# from collections import deque
# import math

# @dataclass
# class ChannelBaseline:
#     """Keeps per-channel baseline statistics (mu, var) for shallow activations.
#     Use `calibrate_offline.py` once to compute stable stats, then load them here.
#     """
#     mu: torch.Tensor
#     var: torch.Tensor

#     @staticmethod
#     def zeros(C: int, device: str = "cpu") -> "ChannelBaseline":
#         mu = torch.zeros(C, device=device)
#         var = torch.ones(C, device=device)
#         return ChannelBaseline(mu=mu, var=var)

#     @torch.no_grad()
#     def z_scores(self, A: torch.Tensor) -> torch.Tensor:
#         """A: (C, H, W) on same device as mu/var."""
#         s = A.abs().mean(dim=(1, 2))  # (C,)
#         sigma = (self.var + 1e-6).sqrt()
#         return (s - self.mu) / sigma

# # =============================
# # File: tracker.py
# # =============================
# # from __future__ import annotations
# from dataclasses import dataclass
# from typing import List, Dict

# @dataclass
# class Track:
#     tid: int
#     bbox: List[int]  # [x1,y1,x2,y2]
#     hits: int = 1
#     missed: int = 0


# def _iou(a, b):
#     xa = max(a[0], b[0]); ya = max(a[1], b[1])
#     xb = min(a[2], b[2]); yb = min(a[3], b[3])
#     inter = max(0, xb - xa) * max(0, yb - ya)
#     areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
#     areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
#     denom = areaA + areaB - inter + 1e-8
#     return inter / denom

# class SimpleIoUTracker:
#     def __init__(self, iou_thresh: float = 0.3, max_missed: int = 5, confirm_hits: int = 3):
#         self.iou_thresh = iou_thresh
#         self.max_missed = max_missed
#         self.confirm_hits = confirm_hits
#         self.tracks: List[Track] = []
#         self._next_id = 1

#     def update(self, detections: List[List[int]]) -> List[Dict]:
#         assigned = set()
#         # Greedy match
#         for t in self.tracks:
#             best_iou, best_idx = 0.0, -1
#             for i, det in enumerate(detections):
#                 if i in assigned: 
#                     continue
#                 val = _iou(t.bbox, det)
#                 if val > best_iou:
#                     best_iou, best_idx = val, i
#             if best_iou >= self.iou_thresh and best_idx >= 0:
#                 t.bbox = detections[best_idx]
#                 t.hits += 1
#                 t.missed = 0
#                 assigned.add(best_idx)
#             else:
#                 t.missed += 1

#         # Add new tracks
#         for i, det in enumerate(detections):
#             if i not in assigned:
#                 self.tracks.append(Track(self._next_id, det))
#                 self._next_id += 1

#         # Prune
#         self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

#         out = []
#         for t in self.tracks:
#             out.append({
#                 "id": t.tid,
#                 "bbox": t.bbox,
#                 "confirmed": t.hits >= self.confirm_hits,
#                 "age": t.hits,
#             })
#         return out

# # =============================
# # File: realtime_pipeline.py
# # =============================
# # from __future__ import annotations
# import time
# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
# from typing import Dict, Tuple, List

# # ChannelBaseline and SimpleIoUTracker are defined above

# class RealTimeAttentionDefense:
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         shallow_module: torch.nn.Module,
#         baseline: ChannelBaseline,
#         tracker: SimpleIoUTracker,
#         z_thresh: float = 3.0,
#         heat_percentile: float = 98.0,
#         min_area: int = 200,  # pixels in input space
#         mask_mode: str = "feature",  # "feature" or "input"
#         device: str = "cuda" if torch.cuda.is_available() else "cpu",
#     ):
#         self.model = model.to(device).eval()
#         self.baseline = baseline
#         self.tracker = tracker
#         self.z_thresh = z_thresh
#         self.heat_percentile = heat_percentile
#         self.min_area = min_area
#         self.mask_mode = mask_mode
#         self.device = device

#         self._feat = None
#         shallow_module.register_forward_hook(self._save_shallow)
#                 # new ACAT configs
#         self.tau = 2.0                   # τ in Eq. (3)
#         self.sigma: torch.Tensor = None  # adversarial trace σ (on device)
#         self.xi = None                   # adaptive threshold ξ
#         self.reinit_count = 0            # how many times we re-initialized σ
#         self.update_period = 1           # update σ every N frames (set>1 to update less often)
#         self.frame_idx = 0               # global frame counter
#         self.min_update_area = 10        # min pixels in feature space to consider a valid mask (helps reset)
#         self.noise_filter_kernel = (5,5) # gaussian kernel size for heatmap smoothing


#     def _save_shallow(self, m, x, y):
#         # y: (B,C,H,W)
#         self._feat = y

#     @torch.no_grad()
#     def _channels_to_heat(self, A: torch.Tensor) -> np.ndarray:
#         # A: (C,H,W) on device
#         z = self.baseline.z_scores(A)  # (C,)
#         sel = z > self.z_thresh
#         if sel.sum().item() == 0:
#             # fallback: take top-1 channel to keep pipeline alive
#             top = torch.topk(z, k=1).indices
#             sel = torch.zeros_like(z, dtype=torch.bool)
#             sel[top] = True
#         H = A[sel].abs().sum(dim=0)  # (H,W)
#         H = (H - H.min()) / (H.max() - H.min() + 1e-8)
#         return H.detach().float().cpu().numpy()

#     def _heat_to_boxes(self, heat: np.ndarray, input_hw: Tuple[int,int]) -> List[List[int]]:
#         h_in, w_in = input_hw
#         heat_resized = cv2.resize(heat, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
#         thr = np.percentile(heat_resized, self.heat_percentile)
#         binm = (heat_resized >= thr).astype(np.uint8) * 255
#         binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
#         contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         boxes = []
#         for c in contours:
#             x,y,w,h = cv2.boundingRect(c)
#             if w*h >= self.min_area:
#                 boxes.append([x,y,x+w,y+h])
#         # keep top-3 by area
#         boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:3]
#         return boxes

#     def _apply_input_mask(self, img: np.ndarray, boxes: List[List[int]]) -> np.ndarray:
#         out = img.copy()
#         for (x1,y1,x2,y2) in boxes:
#             roi = out[y1:y2, x1:x2]
#             if roi.size == 0: 
#                 continue
#             out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (15,15), 0)
#         return out

#     def _apply_feature_mask(self, feat: torch.Tensor, boxes: List[List[int]], in_hw: Tuple[int,int]) -> torch.Tensor:
#         # feat: (B,C,Hf,Wf), B==1 assumed; map input boxes to feature coords
#         B,C,Hf,Wf = feat.shape
#         h_in, w_in = in_hw
#         xs = lambda x: int(np.clip(x * (Wf / float(w_in)), 0, Wf-1))
#         ys = lambda y: int(np.clip(y * (Hf / float(h_in)), 0, Hf-1))
#         for (x1,y1,x2,y2) in boxes:
#             fx1, fy1, fx2, fy2 = xs(x1), ys(y1), xs(x2), ys(y2)
#             feat[:, :, fy1:fy2+1, fx1:fx2+1] = 0.0
#         return feat

#     @torch.no_grad()
#     def process_frame(self, img_bgr: np.ndarray, preprocess_fn) -> Dict:
#         # 1) preprocess → tensor
#         t0 = time.time()
#         inp, meta = preprocess_fn(img_bgr)  # (1,3,H,W) tensor on device, plus any resize meta
#         H_in, W_in = meta["input_hw"]
#         inp = inp.to(self.device)
#         t1 = time.time()

#         # 2) forward to capture shallow feat
#         _ = self.model(inp)
#         assert self._feat is not None, "Shallow hook didn't capture features"
#         A = self._feat[0]  # (C,Hf,Wf)
#         t2 = time.time()

#         # 3) channel attention → heatmap → candidate boxes
#         # heat = self._channels_to_heat(A)
#         # boxes = self._heat_to_boxes(heat, (H_in, W_in))
#                 # 3) channel attention → heatmap → candidate boxes (ACAT behavior)
#         self.frame_idx += 1

#         if self.sigma is None:
#             # no trace yet: compute heat via channel z-scores (existing behavior),
#             # and do not update sigma until a confirmed detection is found.
#             heat = self._channels_to_heat(A)
#             boxes = self._heat_to_boxes(heat, (H_in, W_in))
#         else:
#             # compute heat using sigma
#             heat = self._compute_heat_with_sigma(A)
#             boxes = self._heat_to_boxes(heat, (H_in, W_in))

#         # optionally, only update sigma every update_period frames
#         do_update = (self.frame_idx % self.update_period == 0)

#         t3 = time.time()

#         # 4) tracker update & confirmation
#         tracks = self.tracker.update(boxes)
#         confirmed_boxes = [t["bbox"] for t in tracks if t["confirmed"]]
#         t4 = time.time()

#         # 5) mitigation (optional until confirmed)
#         defended_inp = inp
#         if len(confirmed_boxes) > 0:
#             if self.mask_mode == "feature":
#                 defended_feat = self._apply_feature_mask(self._feat.clone(), confirmed_boxes, (H_in, W_in))
#                 # Patch: replace stored feat with defended_feat before another forward
#                 self._feat = defended_feat
#                 # Re-run forward from just after shallow layer would be ideal; as a
#                 # simple compat path, run full forward again (extra cost but ok for demo)
#                 out = self.model(inp)  # defended via modified features cached by hook
#             else:
#                 masked_bgr = self._apply_input_mask(img_bgr, confirmed_boxes)
#                 inp2, _ = preprocess_fn(masked_bgr)
#                 out = self.model(inp2.to(self.device))
#         else:
#             out = self.model(inp)
#         t5 = time.time()
#                 # -------------------------
#         # Update adversarial trace
#         # -------------------------
#         if len(confirmed_boxes) > 0:
#             # compute adaptive threshold on resized heatmap (pass heat resized to input space)
#             # Our heat variable is feature-res heat; resize for thresholding
#             H_resized_for_xi = cv2.resize((heat*255).astype(np.uint8), (W_in, H_in), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
#             # compute xi (store)
#             self.xi = self._compute_adaptive_xi(heat, confirmed_boxes, (H_in, W_in))
#             # update sigma using Eq. (4) if allowed by update_period
#             if do_update:
#                 self._update_sigma(A, confirmed_boxes, (H_in, W_in))
#         else:
#             # if no confirmed boxes and we have an active trace, optionally decay sigma,
#             # or reset after prolonged misses (simple decay here)
#             if self.sigma is not None:
#                 # simple decay to forget old traces slowly
#                 self.sigma = (0.95 * self.sigma).clamp(0.0, 1.0)
#                 # reset if trace values are too small
#                 if torch.max(self.sigma).item() < 0.05:
#                     self.sigma = None
#                     self.reinit_count += 1

#         return {
#             "output": out,
#             "heatmap": heat,
#             "boxes": boxes,
#             "tracks": tracks,
#             "confirmed_boxes": confirmed_boxes,
#             "timing_ms": {
#                 "preprocess": (t1-t0)*1000,
#                 "forward_shallow": (t2-t1)*1000,
#                 "attention_heatmap": (t3-t2)*1000,
#                 "tracking": (t4-t3)*1000,
#                 "mitigation_forward": (t5-t4)*1000,
#                 "total": (t5-t0)*1000,
#             }
#         }
#     @torch.no_grad()
#     def _compute_heat_with_sigma(self, A: torch.Tensor) -> np.ndarray:
#         """Compute heat H = sum_c (sigma[c]^tau * A_c.abs().sum over channels selected)"""
#         # A: (C,Hf,Wf), sigma on device (C,)
#         if self.sigma is None:
#             # fallback to z-score channel selection (existing behavior)
#             return self._channels_to_heat(A)

#         # sigma^tau (device)
#         s_pow = self.sigma.clamp(min=0.0) ** self.tau  # (C,)
#         # multiply per-channel and sum: (C,H,W) * (C,1,1) -> (H,W)
#         weighted = (A.abs() * s_pow.view(-1,1,1)).sum(dim=0)
#         # normalize heat to [0,1]
#         wmin, wmax = weighted.min(), weighted.max()
#         H = (weighted - wmin) / ( (wmax - wmin) + 1e-8 )
#         # smooth with small gaussian (use cv2 on cpu)
#         H_np = H.detach().cpu().numpy()
#         H_np = cv2.GaussianBlur((H_np*255).astype(np.uint8), self.noise_filter_kernel, 0).astype(np.float32)/255.0
#         return H_np

#     @torch.no_grad()
#     def _update_sigma(self, A: torch.Tensor, mask_boxes: List[List[int]], in_hw: Tuple[int,int]):
#         """
#         Update sigma using Eq. (4):
#         sigma_next = N( mean(A outside mask) - mean(A inside mask) )
#         A: (C,Hf,Wf) (on device)
#         mask_boxes in input coords -> convert to feature coords
#         """
#         B,C,Hf,Wf = 1, *A.shape  # A is (C,Hf,Wf)
#         # Build binary feature-space mask
#         mask_feat = torch.zeros((Hf, Wf), dtype=torch.bool, device=self.device)
#         h_in, w_in = in_hw
#         def to_fx(x): return int(np.clip(x * (Wf / float(w_in)), 0, Wf-1))
#         def to_fy(y): return int(np.clip(y * (Hf / float(h_in)), 0, Hf-1))
#         for (x1,y1,x2,y2) in mask_boxes:
#             fx1, fy1, fx2, fy2 = to_fx(x1), to_fy(y1), to_fx(x2), to_fy(y2)
#             if fx2<=fx1 or fy2<=fy1:
#                 continue
#             mask_feat[fy1:fy2+1, fx1:fx2+1] = True

#         # avoid empty masks
#         inside_cnt = int(mask_feat.sum().item())
#         outside_cnt = int((~mask_feat).sum().item())
#         if inside_cnt < self.min_update_area:
#             # mask too small → consider re-init (reset)
#             self.reinit_count += 1
#             self.sigma = None
#             return

#         # compute per-channel mean inside and outside
#         # A: (C,Hf,Wf)
#         A_flat = A.view(C, -1)  # (C, Hf*Wf)
#         mask_flat = mask_feat.view(-1)  # (Hf*Wf,)
#         inside_vals = A_flat[:, mask_flat].abs().mean(dim=1)   # (C,)
#         outside_vals = A_flat[:, ~mask_flat].abs().mean(dim=1) # (C,)

#         delta = (outside_vals - inside_vals)  # favor channels that are larger outside (positive), as paper defines
#         # Normalize N(·) to [0,1] via ReLU + min-max
#         delta_relu = torch.relu(delta)
#         if torch.max(delta_relu) - torch.min(delta_relu) > 1e-8:
#             delta_norm = (delta_relu - delta_relu.min()) / (delta_relu.max() - delta_relu.min())
#         else:
#             delta_norm = torch.zeros_like(delta_relu)
#         # update sigma (simple running average to add stability)
#         if self.sigma is None:
#             self.sigma = delta_norm.to(self.device)
#         else:
#             self.sigma = (0.6 * self.sigma + 0.4 * delta_norm.to(self.device)).clamp(0.0, 1.0)
            
#     def _compute_adaptive_xi(self, heat: np.ndarray, mask_boxes: List[List[int]], in_hw: Tuple[int,int]) -> float:
#         """
#         Compute ξ_k+1 = max(H_bar) + psi(H_bar)
#         H_bar = H \odot D(M_bar)
#         D = dilate expansion of complement mask; implement by expanding each box and zero-out region in mask
#         """
#         H = heat.copy()  # (Hf,Wf) or (Hin,Win) depending on caller
#         h_in, w_in = in_hw
#         # Work in input resolution: assume heat already resized to input when calling (do that before)
#         H_resized = cv2.resize((H*255).astype(np.uint8), (w_in, h_in), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0

#         # build expanded mask of predicted adversarial area (complement)
#         mask = np.zeros_like(H_resized, dtype=np.uint8)
#         # expand by a kernel (e.g., 5 px)
#         pad = 5
#         for (x1,y1,x2,y2) in mask_boxes:
#             # clip
#             xa, ya, xb, yb = max(0,x1-pad), max(0,y1-pad), min(w_in,x2+pad), min(h_in,y2+pad)
#             mask[ya:yb, xa:xb] = 1
#         # complementary area values
#         H_bar_vals = H_resized[mask==0]
#         if H_bar_vals.size == 0:
#             return float(H_resized.max() + 0.01)
#         max_hbar = float(H_bar_vals.max())
#         v = 70.0
#         pct_v = np.percentile(H_bar_vals, v)
#         psi = float(pct_v - float(H_bar_vals.mean()))
#         xi = max_hbar + max(0.0, psi)
#         return xi


# # Helper functions for both modes
# def load_model_and_hook():
#     """Load model and setup feature hook for calibration."""
#     model, preprocess_fn = load_bisenet_and_preprocess()
#     shallow = get_shallow_module(model)
#     feats = {}
#     def hook(m, x, y):
#         feats['A'] = y.detach()
#     shallow.register_forward_hook(hook)
#     return model.eval(), preprocess_fn, feats

# def draw_boxes(img, boxes, color=(0,255,0)):
#     """Draw bounding boxes on image copy."""
#     out = img.copy()
#     for x1,y1,x2,y2 in boxes:
#         cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
#     return out

# # Main entry points for each mode
# def defend_main():
#     """Run defense mode with visualization."""
#     ap = argparse.ArgumentParser(description="Run attention defense on video/images")
#     src = ap.add_mutually_exclusive_group(required=True)
#     src.add_argument('--video', type=str, help="Input video file")
#     src.add_argument('--frames_dir', type=str, help="Input frames directory")
#     ap.add_argument('--baseline', type=str, required=True, help="Baseline stats file from calibration")
#     ap.add_argument('--mask_mode', type=str, default='feature', choices=['feature','input'],
#                    help="Defense masking mode: feature-level or input-level")
#     ap.add_argument('--z', type=float, default=3.0, help="Z-score threshold")
#     ap.add_argument('--percentile', type=float, default=98.0, help="Heat percentile threshold")
#     ap.add_argument('--min_area', type=int, default=200, help="Minimum box area in pixels")
#     ap.add_argument('--save_vis', type=str, help="Output directory for visualization frames")
#     args = ap.parse_args()

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # Load model and preprocessing
#     model, preprocess_fn = load_bisenet_and_preprocess()
#     shallow = get_shallow_module(model)

#     # Load baseline stats
#     ckpt = torch.load(args.baseline, map_location=device)
#     baseline = ChannelBaseline(mu=ckpt['mu'].to(device), var=ckpt['var'].to(device))

#     # Setup tracker and defense
#     tracker = SimpleIoUTracker(iou_thresh=0.3, max_missed=5, confirm_hits=3)
#     defense = RealTimeAttentionDefense(
#         model=model, shallow_module=shallow, baseline=baseline, tracker=tracker,
#         z_thresh=args.z, heat_percentile=args.percentile, min_area=args.min_area,
#         mask_mode=args.mask_mode, device=device,
#     )

#     if args.save_vis:
#         os.makedirs(args.save_vis, exist_ok=True)

#     # Load frames
#     frames = []
#     if args.video:
#         cap = cv2.VideoCapture(args.video)
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frames.append(frame)
#         cap.release()
#     else:
#         for name in sorted(os.listdir(args.frames_dir)):
#             if name.lower().endswith((".png",".jpg",".jpeg",".bmp")):
#                 frames.append(cv2.imread(os.path.join(args.frames_dir, name)))

#     # Process stream
#     total_ms = []
#     for i, bgr in enumerate(frames):
#         res = defense.process_frame(bgr, preprocess_fn)
#         heat = (res['heatmap']*255).astype(np.uint8)
#         heat_vis = cv2.applyColorMap(cv2.resize(heat,(bgr.shape[1], bgr.shape[0])), cv2.COLORMAP_JET)
#         vis = draw_boxes(bgr, res['boxes'], (0,255,0))
#         vis = draw_boxes(vis, res['confirmed_boxes'], (0,0,255))
#         blend = cv2.addWeighted(vis, 0.7, heat_vis, 0.3, 0)

#         ms = res['timing_ms']['total']
#         total_ms.append(ms)
#         cv2.putText(blend, f"{ms:.1f} ms ({1000.0/max(ms,1e-3):.1f} FPS)".replace('inf','∞'),
#                     (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

#         if args.save_vis:
#             cv2.imwrite(os.path.join(args.save_vis, f"frame_{i:05d}.png"), blend)
#         else:
#             cv2.imshow('defense', blend)
#             if cv2.waitKey(1) == 27:  # ESC to quit
#                 break

#     if not args.save_vis:
#         cv2.destroyAllWindows()

#     if total_ms:
#         mean_ms = float(np.mean(total_ms))
#         print(f"Mean latency: {mean_ms:.2f} ms | {1000.0/max(mean_ms,1e-3):.2f} FPS")

# def calibrate_main():
#     """Run calibration mode to compute baseline statistics."""
#     ap = argparse.ArgumentParser(description="Compute baseline channel statistics from clean frames")
#     ap.add_argument('--frames_dir', type=str, required=True, help="Directory with clean training frames")
#     ap.add_argument('--out', type=str, default='baseline.pt', help="Output path for baseline stats")
#     args = ap.parse_args()

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model, preprocess_fn, feats = load_model_and_hook()
#     model.to(device).eval()

#     mu_acc, var_acc, n = None, None, 0

#     for name in sorted(os.listdir(args.frames_dir)):
#         if not name.lower().endswith((".png",".jpg",".jpeg",".bmp")):
#             continue
#         img = cv2.imread(os.path.join(args.frames_dir, name))
#         if img is None:
#             continue
#         inp, _ = preprocess_fn(img)
#         _ = model(inp.to(device))
#         A = feats['A']  # (1,C,H,W)
#         ch_means = A.abs().mean(dim=(0,2,3)).cpu()  # (C,)
#         ch_vars  = A.abs().var(dim=(0,2,3)).cpu()
#         if mu_acc is None:
#             mu_acc, var_acc = ch_means, ch_vars
#         else:
#             mu_acc = mu_acc + ch_means
#             var_acc = var_acc + ch_vars
#         n += 1

#     mu = mu_acc / max(n,1)
#     var = var_acc / max(n,1)
#     torch.save({"mu": mu, "var": var}, args.out)
#     print(f"Saved baseline with C={mu.numel()} to {args.out} from {n} frames")

# # def calibrate_main():
# #     """Run calibration mode to generate baseline statistics."""
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument('--frames_dir', type=str, required=True)
# #     ap.add_argument('--out', type=str, default='baseline.pt')
# #     args = ap.parse_args()

# #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
# #     model, preprocess_fn, feats = load_model_and_hook()
# #     model.to(device).eval()

# #     mu_acc, var_acc, n = None, None, 0

# #     for name in sorted(os.listdir(args.frames_dir)):
# #         if not name.lower().endswith((".png",".jpg",".jpeg",".bmp")):
# #             continue
# #         img = cv2.imread(os.path.join(args.frames_dir, name))
# #         if img is None: 
# #             continue
# #         inp, _ = preprocess_fn(img)
# #         _ = model(inp.to(device))
# #         A = feats['A']  # (1,C,H,W)
# #         ch_means = A.abs().mean(dim=(0,2,3)).cpu()  # (C,)
# #         ch_vars  = A.abs().var(dim=(0,2,3)).cpu()
# #         if mu_acc is None:
# #             mu_acc = ch_means
# #             var_acc = ch_vars
# #         else:
# #             mu_acc = mu_acc + ch_means
# #             var_acc = var_acc + ch_vars
# #         n += 1

# #     mu = mu_acc / max(n,1)
# #     var = var_acc / max(n,1)
# #     torch.save({"mu": mu, "var": var}, args.out)
# #     print(f"Saved baseline with C={mu.numel()} to {args.out} from {n} frames")

# if __name__ == '__main__':
#     # Simple command parsing to choose mode
#     import sys
#     if len(sys.argv) > 1 and '--baseline' in sys.argv:
#         defend_main()  # Running defense mode with visualization
#     else:
#         calibrate_main()  # Running calibration mode

# # =============================
# # Defense implementation
# # ============================

# # def defend_main():
# #     """Run defense mode with visualization."""
# #     ap = argparse.ArgumentParser()
# #     src = ap.add_mutually_exclusive_group(required=True)
# #     src.add_argument('--video', type=str)
# #     src.add_argument('--frames_dir', type=str)
# #     ap.add_argument('--baseline', type=str, required=True)
# #     ap.add_argument('--mask_mode', type=str, default='feature', choices=['feature','input'])
# #     ap.add_argument('--z', type=float, default=3.0)
# #     ap.add_argument('--percentile', type=float, default=98.0)
# #     ap.add_argument('--min_area', type=int, default=200)
# #     ap.add_argument('--save_vis', type=str, default=None)
# #     args = ap.parse_args()

# #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
# #     # Load model and preprocessing
# #     model, preprocess_fn = load_bisenet_and_preprocess()
# #     shallow = get_shallow_module(model)

# #     # Load baseline stats
# #     ckpt = torch.load(args.baseline, map_location=device)
# #     baseline = ChannelBaseline(mu=ckpt['mu'].to(device), var=ckpt['var'].to(device))

# #     # Tracker and defense
# #     tracker = SimpleIoUTracker(iou_thresh=0.3, max_missed=5, confirm_hits=3)
# #     defense = RealTimeAttentionDefense(
# #         model=model, shallow_module=shallow, baseline=baseline, tracker=tracker,
# #         z_thresh=args.z, heat_percentile=args.percentile, min_area=args.min_area,
# #         mask_mode=args.mask_mode, device=device,
# #     )

# #     if args.save_vis is not None:
# #         os.makedirs(args.save_vis, exist_ok=True)

# #     # Open source
# #     frames = []
# #     if args.video:
# #         cap = cv2.VideoCapture(args.video)
# #         while True:
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break
# #             frames.append(frame)
# #         cap.release()
# #     else:
# #         for name in sorted(os.listdir(args.frames_dir)):
# #             if name.lower().endswith((".png",".jpg",".jpeg",".bmp")):
# #                 frames.append(cv2.imread(os.path.join(args.frames_dir, name)))

# #     # Process stream
# #     total_ms = []
# #     for i, bgr in enumerate(frames):
# #         res = defense.process_frame(bgr, preprocess_fn)
# #         heat = (res['heatmap']*255).astype(np.uint8)
# #         heat_vis = cv2.applyColorMap(cv2.resize(heat,(bgr.shape[1], bgr.shape[0])), cv2.COLORMAP_JET)
# #         vis = draw_boxes(bgr, res['boxes'], (0,255,0))
# #         vis = draw_boxes(vis, res['confirmed_boxes'], (0,0,255))
# #         blend = cv2.addWeighted(vis, 0.7, heat_vis, 0.3, 0)

# #         ms = res['timing_ms']['total']
# #         total_ms.append(ms)
# #         cv2.putText(blend, f"{ms:.1f} ms ({1000.0/max(ms,1e-3):.1f} FPS)".replace('inf','∞'), (10,25),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

# #         if args.save_vis:
# #             cv2.imwrite(os.path.join(args.save_vis, f"frame_{i:05d}.png"), blend)
# #         else:
# #             cv2.imshow('defense', blend)
# #             if cv2.waitKey(1) == 27:  # ESC to quit
# #                 break

# #     if not args.save_vis:
# #         cv2.destroyAllWindows()

# #     if total_ms:
# #         mean_ms = float(np.mean(total_ms))
# #         print(f"Mean latency: {mean_ms:.2f} ms | {1000.0/max(mean_ms,1e-3):.2f} FPS")
# # """Run attention-based multi-frame defense on a video or image folder.

# # Usage examples:
# #   python defend_video_bisenet.py --video input.mp4 --baseline baseline.pt --mask_mode feature
# #   python defend_video_bisenet.py --frames_dir data/stream --baseline baseline.pt --save_vis out_vis/
# # """
# # import os
# # import argparse
# # import cv2
# # import numpy as np
# # import torch

# # # All classes are defined above in this file

# # # ---- glue helpers to reuse your existing demo preprocessing & model ----
# # # Implement these thin wrappers once for your repo. Keep them here for clarity.

# # def load_bisenet_and_preprocess():
# #     """Return (model, preprocess_fn) matching zmask_demo BiSeNet setup.
# #     You can copy the transforms from the demo notebook.
# #     preprocess_fn(img_bgr) -> (tensor[1,3,H,W], {"input_hw": (H,W)})
# #     """
# #     import torchvision.transforms as T
# #     from torchvision.transforms.functional import to_tensor
# #     # Replace with actual BiSeNet loader used in the repo
# #     # Example skeleton (user must adapt weights path):
# #     from bisenet_model_loader import load_model  # you create this small helper file
# #     model = load_model()  # returns nn.Module ready for eval

# #     # Preprocessing matching the training of the provided weights
# #     mean = [0.3257, 0.3690, 0.3223]; std = [0.2112, 0.2148, 0.2115]
# #     def preprocess_fn(img_bgr):
# #         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# #         H, W = img_rgb.shape[:2]
# #         # if the demo resizes, mirror it here; else keep native
# #         img_t = to_tensor(img_rgb)  # (3,H,W) in [0,1]
# #         img_t = T.Normalize(mean=mean, std=std)(img_t)
# #         return img_t.unsqueeze(0), {"input_hw": (H, W)}

# #     return model, preprocess_fn


# # def get_shallow_module(model: torch.nn.Module) -> torch.nn.Module:
# #     """Return the shallow module to hook (e.g., model.layer1 or first conv).
# #     Must match your backbone structure. For BiSeNet, pick the first stage."""
# #     # Example placeholder; change to actual module name
# #     return list(model.children())[0]

# # # ----------------------------------------------------------------------

# # def draw_boxes(img, boxes, color=(0,255,0)):
# #     out = img.copy()
# #     for x1,y1,x2,y2 in boxes:
# #         cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
# #     return out

# # @torch.no_grad()
# # def main():
# #     ap = argparse.ArgumentParser()
# #     src = ap.add_mutually_exclusive_group(required=True)
# #     src.add_argument('--video', type=str)
# #     src.add_argument('--frames_dir', type=str)
# #     ap.add_argument('--baseline', type=str, required=True)
# #     ap.add_argument('--mask_mode', type=str, default='feature', choices=['feature','input'])
# #     ap.add_argument('--z', type=float, default=3.0)
# #     ap.add_argument('--percentile', type=float, default=98.0)
# #     ap.add_argument('--min_area', type=int, default=200)
# #     ap.add_argument('--save_vis', type=str, default=None)
# #     args = ap.parse_args()

# #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
# #     # Load model and preprocessing
# #     model, preprocess_fn = load_bisenet_and_preprocess()
# #     shallow = get_shallow_module(model)

# #     # Load baseline stats
# #     ckpt = torch.load(args.baseline, map_location=device)
# #     baseline = ChannelBaseline(mu=ckpt['mu'].to(device), var=ckpt['var'].to(device))

# #     # Tracker and defense
# #     tracker = SimpleIoUTracker(iou_thresh=0.3, max_missed=5, confirm_hits=3)
# #     defense = RealTimeAttentionDefense(
# #         model=model, shallow_module=shallow, baseline=baseline, tracker=tracker,
# #         z_thresh=args.z, heat_percentile=args.percentile, min_area=args.min_area,
# #         mask_mode=args.mask_mode, device=device,
# #     )

# #     if args.save_vis is not None:
# #         os.makedirs(args.save_vis, exist_ok=True)

# #     # Open source
# #     frames = []
# #     if args.video:
# #         cap = cv2.VideoCapture(args.video)
# #         while True:
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break
# #             frames.append(frame)
# #         cap.release()
# #     else:
# #         for name in sorted(os.listdir(args.frames_dir)):
# #             if name.lower().endswith((".png",".jpg",".jpeg",".bmp")):
# #                 frames.append(cv2.imread(os.path.join(args.frames_dir, name)))

# #     # Process stream
# #     total_ms = []
# #     for i, bgr in enumerate(frames):
# #         res = defense.process_frame(bgr, preprocess_fn)
# #         heat = (res['heatmap']*255).astype(np.uint8)
# #         heat_vis = cv2.applyColorMap(cv2.resize(heat,(bgr.shape[1], bgr.shape[0])), cv2.COLORMAP_JET)
# #         vis = draw_boxes(bgr, res['boxes'], (0,255,0))
# #         vis = draw_boxes(vis, res['confirmed_boxes'], (0,0,255))
# #         blend = cv2.addWeighted(vis, 0.7, heat_vis, 0.3, 0)

# #         ms = res['timing_ms']['total']
# #         total_ms.append(ms)
# #         cv2.putText(blend, f"{ms:.1f} ms ({1000.0/max(ms,1e-3):.1f} FPS)".replace('inf','∞'), (10,25),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

# #         if args.save_vis:
# #             cv2.imwrite(os.path.join(args.save_vis, f"frame_{i:05d}.png"), blend)
# #         else:
# #             cv2.imshow('defense', blend)
# #             if cv2.waitKey(1) == 27:  # ESC to quit
# #                 break

# #     if not args.save_vis:
# #         cv2.destroyAllWindows()

# #     if total_ms:
# #         mean_ms = float(np.mean(total_ms))
# #         print(f"Mean latency: {mean_ms:.2f} ms | {1000.0/max(mean_ms,1e-3):.2f} FPS")

# # if __name__ == '__main__':
# #     main()

"""
ACAT-style multi-frame defense (Faithful to paper’s core ideas):

 - Calibration mode: compute per-channel shallow activation stats on clean frames
 - Defense mode: run on frames folder OR on a video file
 - Adversarial trace σ with Eq.(4)-style update
 - Heatmap via Eq.(3) with τ
 - Adaptive threshold ξ using Eq.(5)-like margining on complement region
 - Gaussian noise filtering of heatmap
 - Reset logic when mask too small or σ decays
 - Single-pass-friendly model usage: reuse shallow feature & run tail only

USAGE:
  # 1) Calibrate baseline from clean frames
  python proj.py --frames_dir data/clean_frames --out baseline.pt

  # 2) Create a patched stream
  python patch.py

  # 3a) Defend on frames directory
  python proj.py --frames_dir data/stream --baseline baseline.pt --mask_mode feature --save_vis out_vis

  # 3b) Defend on a single video
  python proj.py --video videos/drive1.mp4 --baseline baseline.pt --mask_mode feature --save_vis out_vis_video
"""

from __future__ import annotations

import os
import cv2
import math
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import deque

from bisenet_model_loader import (
    load_bisenet_and_preprocess,
    get_shallow_module,
)

# =========================
# Calibration data structure
# =========================

@dataclass
class ChannelBaseline:
    mu: torch.Tensor  # (C,)
    var: torch.Tensor  # (C,)

    @staticmethod
    def zeros(C: int, device: str = "cpu") -> "ChannelBaseline":
        mu = torch.zeros(C, device=device)
        var = torch.ones(C, device=device)
        return ChannelBaseline(mu=mu, var=var)

    @torch.no_grad()
    def z_scores(self, A: torch.Tensor) -> torch.Tensor:
        """
        A: (C, H, W), returns z of per-channel |A| mean vs baseline.
        """
        s = A.abs().mean(dim=(1, 2))  # (C,)
        sigma = (self.var + 1e-6).sqrt()
        return (s - self.mu) / sigma


# ==========
# Simple IoU tracker
# ==========

@dataclass
class Track:
    tid: int
    bbox: List[int]  # [x1,y1,x2,y2]
    hits: int = 1
    missed: int = 0

def _iou(a, b):
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    denom = areaA + areaB - inter + 1e-8
    return inter / denom

class SimpleIoUTracker:
    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 5, confirm_hits: int = 3):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.confirm_hits = confirm_hits
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: List[List[int]]) -> List[Dict]:
        assigned = set()
        # Greedy match
        for t in self.tracks:
            best_iou, best_idx = 0.0, -1
            for i, det in enumerate(detections):
                if i in assigned:
                    continue
                val = _iou(t.bbox, det)
                if val > best_iou:
                    best_iou, best_idx = val, i
            if best_iou >= self.iou_thresh and best_idx >= 0:
                t.bbox = detections[best_idx]
                t.hits += 1
                t.missed = 0
                assigned.add(best_idx)
            else:
                t.missed += 1

        # Add new tracks
        for i, det in enumerate(detections):
            if i not in assigned:
                self.tracks.append(Track(self._next_id, det))
                self._next_id += 1

        # Prune
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        out = []
        for t in self.tracks:
            out.append({
                "id": t.tid,
                "bbox": t.bbox,
                "confirmed": t.hits >= self.confirm_hits,
                "age": t.hits,
            })
        return out


# =========================
# Real-time ACAT defense
# =========================

class RealTimeAttentionDefense:
    def __init__(
        self,
        model: torch.nn.Module,
        shallow_module: torch.nn.Module,
        baseline: ChannelBaseline,
        tracker: SimpleIoUTracker,
        z_thresh: float = 3.0,
        heat_percentile: float = 98.0,
        min_area: int = 200,  # in input pixels
        mask_mode: str = "feature",  # "feature" or "input"
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device).eval()
        self.baseline = baseline
        self.tracker = tracker
        self.z_thresh = z_thresh
        self.heat_percentile = heat_percentile
        self.min_area = min_area
        self.mask_mode = mask_mode
        self.device = device

        self._feat = None
        shallow_module.register_forward_hook(self._save_shallow)

        # ACAT params/state
        self.tau = 2.0
        self.sigma: torch.Tensor = None   # (C,)
        self.xi = None                    # adaptive threshold (not mandatory for boxes path)
        self.reinit_count = 0
        self.update_period = 1
        self.frame_idx = 0
        self.min_update_area = 10
        self.noise_filter_kernel = (5, 5)

    def _save_shallow(self, m, x, y):
        # y: (B,C,Hf,Wf)
        self._feat = y

    # ---------- Heat via z-score selection (fallback when sigma is None) ----------
    @torch.no_grad()
    def _channels_to_heat(self, A: torch.Tensor) -> np.ndarray:
        z = self.baseline.z_scores(A)  # (C,)
        sel = z > self.z_thresh
        if sel.sum().item() == 0:
            # fallback: top-1
            top = torch.topk(z, k=1).indices
            sel = torch.zeros_like(z, dtype=torch.bool)
            sel[top] = True
        H = A[sel].abs().sum(dim=0)
        H = (H - H.min()) / (H.max() - H.min() + 1e-8)
        H_np = H.detach().float().cpu().numpy()
        H_np = cv2.GaussianBlur((H_np * 255).astype(np.uint8), self.noise_filter_kernel, 0).astype(np.float32) / 255.0
        return H_np

    # ---------- Heat via Eq.(3) using sigma ----------
    @torch.no_grad()
    def _compute_heat_with_sigma(self, A: torch.Tensor) -> np.ndarray:
        if self.sigma is None:
            return self._channels_to_heat(A)
        s_pow = self.sigma.clamp(min=0.0) ** self.tau
        weighted = (A.abs() * s_pow.view(-1, 1, 1)).sum(dim=0)
        H = (weighted - weighted.min()) / (weighted.max() - weighted.min() + 1e-8)
        H_np = H.detach().float().cpu().numpy()
        H_np = cv2.GaussianBlur((H_np * 255).astype(np.uint8), self.noise_filter_kernel, 0).astype(np.float32) / 255.0
        return H_np

    # ---------- Heatmap to boxes (input space) ----------
    def _heat_to_boxes(self, heat: np.ndarray, input_hw: Tuple[int, int]) -> List[List[int]]:
        h_in, w_in = input_hw
        heat_resized = cv2.resize(heat, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        thr = np.percentile(heat_resized, self.heat_percentile)
        binm = (heat_resized >= thr).astype(np.uint8) * 255
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= self.min_area:
                boxes.append([x, y, x + w, y + h])
        boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:3]
        return boxes

    # ---------- Adaptive threshold (Eq. 5) (not mandatory for boxes, useful if you binarize H directly) ----------
    def _compute_adaptive_xi(self, heat: np.ndarray, mask_boxes: List[List[int]], in_hw: Tuple[int, int]) -> float:
        H = heat.copy()
        h_in, w_in = in_hw
        H_resized = cv2.resize((H * 255).astype(np.uint8), (w_in, h_in), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        # Build expanded mask (pad 5px)
        mask = np.zeros_like(H_resized, dtype=np.uint8)
        pad = 5
        for (x1, y1, x2, y2) in mask_boxes:
            xa, ya, xb, yb = max(0, x1 - pad), max(0, y1 - pad), min(w_in, x2 + pad), min(h_in, y2 + pad)
            mask[ya:yb, xa:xb] = 1
        H_bar_vals = H_resized[mask == 0]
        if H_bar_vals.size == 0:
            return float(H_resized.max() + 0.01)
        max_hbar = float(H_bar_vals.max())
        v = 70.0
        psi = float(np.percentile(H_bar_vals, v) - float(H_bar_vals.mean()))
        xi = max_hbar + max(0.0, psi)
        return xi

    # ---------- Sigma update (Eq. 4) ----------
    @torch.no_grad()
    def _update_sigma(self, A: torch.Tensor, mask_boxes: List[List[int]], in_hw: Tuple[int, int]):
        C, Hf, Wf = A.shape
        mask_feat = torch.zeros((Hf, Wf), dtype=torch.bool, device=self.device)
        h_in, w_in = in_hw

        def to_fx(x): return int(np.clip(x * (Wf / float(w_in)), 0, Wf - 1))
        def to_fy(y): return int(np.clip(y * (Hf / float(h_in)), 0, Hf - 1))

        for (x1, y1, x2, y2) in mask_boxes:
            fx1, fy1, fx2, fy2 = to_fx(x1), to_fy(y1), to_fx(x2), to_fy(y2)
            if fx2 <= fx1 or fy2 <= fy1:
                continue
            mask_feat[fy1:fy2+1, fx1:fx2+1] = True

        inside_cnt = int(mask_feat.sum().item())
        if inside_cnt < self.min_update_area:
            # tiny mask -> reset
            self.reinit_count += 1
            self.sigma = None
            return

        A_flat = A.view(C, -1)  # (C, Hf*Wf)
        mask_flat = mask_feat.view(-1)
        inside_vals = A_flat[:, mask_flat].abs().mean(dim=1)   # (C,)
        outside_vals = A_flat[:, ~mask_flat].abs().mean(dim=1) # (C,)

        delta = (outside_vals - inside_vals)  # Eq. (4) terms
        delta_relu = torch.relu(delta)
        if torch.max(delta_relu) - torch.min(delta_relu) > 1e-8:
            delta_norm = (delta_relu - delta_relu.min()) / (delta_relu.max() - delta_relu.min())
        else:
            delta_norm = torch.zeros_like(delta_relu)

        if self.sigma is None:
            self.sigma = delta_norm.to(self.device)
        else:
            self.sigma = (0.6 * self.sigma + 0.4 * delta_norm.to(self.device)).clamp(0.0, 1.0)

    # ---------- Mask application ----------
    def _apply_input_mask(self, img: np.ndarray, boxes: List[List[int]]) -> np.ndarray:
        out = img.copy()
        for (x1, y1, x2, y2) in boxes:
            roi = out[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (15, 15), 0)
        return out

    def _apply_feature_mask(self, feat: torch.Tensor, boxes: List[List[int]], in_hw: Tuple[int, int]) -> torch.Tensor:
        # feat: (B,C,Hf,Wf), B==1 assumed
        B, C, Hf, Wf = feat.shape
        h_in, w_in = in_hw

        def xs(x): return int(np.clip(x * (Wf / float(w_in)), 0, Wf - 1))
        def ys(y): return int(np.clip(y * (Hf / float(h_in)), 0, Hf - 1))

        for (x1, y1, x2, y2) in boxes:
            fx1, fy1, fx2, fy2 = xs(x1), ys(y1), xs(x2), ys(y2)
            feat[:, :, fy1:fy2+1, fx1:fx2+1] = 0.0
        return feat

    # ---------- Main per-frame processing ----------
    @torch.no_grad()
    def process_frame(self, img_bgr: np.ndarray, preprocess_fn, resume_from_shallow: bool = True) -> Dict:
        t0 = time.time()
        inp, meta = preprocess_fn(img_bgr)   # (1,3,H,W)
        H_in, W_in = meta["input_hw"]
        inp = inp.to(self.device)
        t1 = time.time()

        # forward up to shallow (hook captures _feat)
        _ = self.model(inp)
        assert self._feat is not None, "Shallow hook did not capture features"
        A = self._feat[0]  # (C,Hf,Wf)
        t2 = time.time()

        # heatmap with sigma if present (Eq. 3), else z-score fallback
        self.frame_idx += 1
        if self.sigma is None:
            heat = self._channels_to_heat(A)
        else:
            heat = self._compute_heat_with_sigma(A)

        boxes = self._heat_to_boxes(heat, (H_in, W_in))
        t3 = time.time()

        tracks = self.tracker.update(boxes)
        confirmed_boxes = [t["bbox"] for t in tracks if t["confirmed"]]
        t4 = time.time()

        # Initialize sigma the first time we have a confirmed region
        if self.sigma is None and len(confirmed_boxes) > 0:
            C, Hf, Wf = A.shape
            mask_feat = torch.zeros((Hf, Wf), dtype=torch.bool, device=self.device)
            for (x1, y1, x2, y2) in confirmed_boxes:
                fx1 = int(np.clip(x1 * (Wf / float(W_in)), 0, Wf - 1))
                fy1 = int(np.clip(y1 * (Hf / float(H_in)), 0, Hf - 1))
                fx2 = int(np.clip(x2 * (Wf / float(W_in)), 0, Wf - 1))
                fy2 = int(np.clip(y2 * (Hf / float(H_in)), 0, Hf - 1))
                if fx2 > fx1 and fy2 > fy1:
                    mask_feat[fy1:fy2+1, fx1:fx2+1] = True
            if mask_feat.sum() > 0:
                inside_vals = A.abs().view(C, -1)[:, mask_feat.view(-1)].mean(dim=1)
                s_init = (inside_vals - inside_vals.min())
                if torch.max(s_init) - torch.min(s_init) > 1e-8:
                    s_init = (s_init - s_init.min()) / (s_init.max() - s_init.min())
                else:
                    s_init = torch.zeros_like(inside_vals)
                self.sigma = s_init.to(self.device)

        # Mitigation + “single-pass” tail forward
        if len(confirmed_boxes) > 0:
            if self.mask_mode == "feature":
                defended_feat = self._apply_feature_mask(self._feat.clone(), confirmed_boxes, (H_in, W_in))
                if hasattr(self.model, "forward_from_shallow") and resume_from_shallow:
                    # run only the tail (single-pass friendly)
                    out = self.model.forward_from_shallow(defended_feat)
                else:
                    # fallback: full forward again
                    out = self.model(inp)
            else:
                masked_bgr = self._apply_input_mask(img_bgr, confirmed_boxes)
                inp2, _ = preprocess_fn(masked_bgr)
                out = self.model(inp2.to(self.device))
        else:
            out = self.model(inp)
        t5 = time.time()

        # Adaptive ξ (not strictly required for bbox flow)
        if len(confirmed_boxes) > 0:
            self.xi = self._compute_adaptive_xi(heat, confirmed_boxes, (H_in, W_in))

        # Periodic σ update (Eq. 4)
        if len(confirmed_boxes) > 0 and (self.frame_idx % self.update_period == 0):
            self._update_sigma(A, confirmed_boxes, (H_in, W_in))
        elif len(confirmed_boxes) == 0 and self.sigma is not None:
            # gentle decay; full reset if it collapses
            self.sigma = (0.95 * self.sigma).clamp(0.0, 1.0)
            if torch.max(self.sigma).item() < 0.05:
                self.sigma = None
                self.reinit_count += 1

        return {
            "output": out,
            "heatmap": heat,
            "boxes": boxes,
            "tracks": tracks,
            "confirmed_boxes": confirmed_boxes,
            "timing_ms": {
                "preprocess": (t1 - t0) * 1000.0,
                "forward_shallow": (t2 - t1) * 1000.0,
                "attention_heatmap": (t3 - t2) * 1000.0,
                "tracking": (t4 - t3) * 1000.0,
                "mitigation_forward": (t5 - t4) * 1000.0,
                "total": (t5 - t0) * 1000.0,
            }
        }


# ==============
# Helpers / glue
# ==============

def draw_boxes(img, boxes, color=(0, 255, 0)):
    out = img.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out

def load_model_and_hook():
    model, preprocess_fn = load_bisenet_and_preprocess()
    shallow = get_shallow_module(model)
    feats = {}
    def hook(m, x, y):
        feats['A'] = y.detach()
    shallow.register_forward_hook(hook)
    return model.eval(), preprocess_fn, feats


# ==============
# CLI Entrypoints
# ==============

def defend_main():
    ap = argparse.ArgumentParser(description="Run ACAT-style defense on frames or a video")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', type=str, help="Input video file")
    src.add_argument('--frames_dir', type=str, help="Input frames directory")
    ap.add_argument('--baseline', type=str, required=True, help="Baseline stats from calibration")
    ap.add_argument('--mask_mode', type=str, default='feature', choices=['feature','input'])
    ap.add_argument('--z', type=float, default=3.0, help="z-score channel threshold (fallback)")
    ap.add_argument('--percentile', type=float, default=98.0, help="heat percentile for boxes")
    ap.add_argument('--min_area', type=int, default=200, help="min area for a box (input pixels)")
    ap.add_argument('--tau', type=float, default=2.0, help="sigma exponent τ (Eq. 3)")
    ap.add_argument('--update_period', type=int, default=1, help="update σ every N frames (>=1)")
    ap.add_argument('--noise_kernel', type=int, default=5, help="gaussian kernel for heat smoothing")
    ap.add_argument('--save_vis', type=str, help="output directory for visualization frames")
    ap.add_argument('--log_csv', type=str, default=None, help="optional CSV log path")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess_fn = load_bisenet_and_preprocess()
    shallow = get_shallow_module(model)

    # Load baseline
    ckpt = torch.load(args.baseline, map_location=device)
    baseline = ChannelBaseline(mu=ckpt['mu'].to(device), var=ckpt['var'].to(device))

    tracker = SimpleIoUTracker(iou_thresh=0.3, max_missed=5, confirm_hits=3)
    defense = RealTimeAttentionDefense(
        model=model, shallow_module=shallow, baseline=baseline, tracker=tracker,
        z_thresh=args.z, heat_percentile=args.percentile, min_area=args.min_area,
        mask_mode=args.mask_mode, device=device,
    )
    defense.tau = args.tau
    defense.update_period = max(1, args.update_period)
    defense.noise_filter_kernel = (args.noise_kernel, args.noise_kernel)

    if args.save_vis:
        os.makedirs(args.save_vis, exist_ok=True)

    # Collect frames
    frames = []
    if args.video:
        cap = cv2.VideoCapture(args.video)
        ok, frm = cap.read()
        while ok:
            frames.append(frm)
            ok, frm = cap.read()
        cap.release()
    else:
        for name in sorted(os.listdir(args.frames_dir)):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                frames.append(cv2.imread(os.path.join(args.frames_dir, name)))

    # Optional CSV logging
    writer = None
    if args.log_csv:
        import csv
        writer = csv.writer(open(args.log_csv, "w", newline=""))
        writer.writerow(["frame", "ms_total", "n_boxes", "n_confirmed",
                         "defending", "reinit_count", "sigma_max", "sigma_mean"])

    total_ms = []
    for i, bgr in enumerate(frames):
        res = defense.process_frame(bgr, preprocess_fn, resume_from_shallow=True)
        heat = (res['heatmap'] * 255).astype(np.uint8)
        heat_vis = cv2.applyColorMap(cv2.resize(heat, (bgr.shape[1], bgr.shape[0])), cv2.COLORMAP_JET)

        vis = draw_boxes(bgr, res['boxes'], (0, 255, 0))
        vis = draw_boxes(vis, res['confirmed_boxes'], (0, 0, 255))
        blend = cv2.addWeighted(vis, 0.7, heat_vis, 0.3, 0)

        ms = res['timing_ms']['total']
        total_ms.append(ms)
        fps = 1000.0 / max(ms, 1e-3)
        cv2.putText(blend, f"{ms:.1f} ms ({fps:.1f} FPS)  reinit={defense.reinit_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if args.save_vis:
            cv2.imwrite(os.path.join(args.save_vis, f"frame_{i:05d}.png"), blend)
        else:
            cv2.imshow('defense', blend)
            if cv2.waitKey(1) == 27:
                break

        if writer is not None:
            if defense.sigma is not None:
                sigma_max = float(torch.max(defense.sigma).item())
                sigma_mean = float(torch.mean(defense.sigma).item())
            else:
                sigma_max, sigma_mean = 0.0, 0.0
            writer.writerow([i, float(ms), len(res['boxes']), len(res['confirmed_boxes']),
                             1 if len(res['confirmed_boxes']) > 0 else 0,
                             defense.reinit_count, sigma_max, sigma_mean])

    if not args.save_vis:
        cv2.destroyAllWindows()

    if total_ms:
        mean_ms = float(np.mean(total_ms))
        print(f"Mean latency: {mean_ms:.2f} ms | {1000.0/max(mean_ms,1e-3):.2f} FPS")

def calibrate_main():
    ap = argparse.ArgumentParser(description="Calibrate shallow-layer channel stats on clean frames")
    ap.add_argument('--frames_dir', type=str, required=True)
    ap.add_argument('--out', type=str, default='baseline.pt')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess_fn, feats = load_model_and_hook()
    model.to(device).eval()

    mu_acc, var_acc, n = None, None, 0

    for name in sorted(os.listdir(args.frames_dir)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        img = cv2.imread(os.path.join(args.frames_dir, name))
        if img is None:
            continue
        inp, _ = preprocess_fn(img)
        _ = model(inp.to(device))
        A = feats['A']  # (1,C,H,W)
        ch_means = A.abs().mean(dim=(0, 2, 3)).cpu()
        ch_vars  = A.abs().var(dim=(0, 2, 3)).cpu()
        if mu_acc is None:
            mu_acc, var_acc = ch_means, ch_vars
        else:
            mu_acc += ch_means
            var_acc += ch_vars
        n += 1

    if n == 0:
        raise RuntimeError("No frames found for calibration.")
    mu = mu_acc / n
    var = var_acc / n
    torch.save({"mu": mu, "var": var}, args.out)
    print(f"Saved baseline with C={mu.numel()} to {args.out} from {n} frames")


if __name__ == '__main__':
    import sys
    # If '--baseline' appears → defense mode. Else → calibration.
    if '--baseline' in sys.argv or '--video' in sys.argv:
        defend_main()
    else:
        calibrate_main()
