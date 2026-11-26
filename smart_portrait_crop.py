"""
Smart Portrait Crop - ComfyUI custom node
Turns a landscape video (IMAGE batch) into a 9:16 portrait video that follows the action.

Install: put this file in  ComfyUI/custom_nodes/  (or in a folder there) and restart ComfyUI.
Needs only torch + numpy (both already part of ComfyUI).

How it picks the "good part" of each frame
  1. Builds a per-frame saliency map from motion (frame difference) + detail (edges) + a mild center prior.
     Optionally you can feed a MASK batch (from SAM / YOLO / Florence2 etc.) and it will follow that instead.
  2. Takes the weighted center of that map along the free axis (x for landscape sources).
  3. Detects scene cuts and restarts the virtual camera there (no slow pan across a cut).
  4. Smooths the camera path: median filter -> dead-zone hold (camera stays still until the subject
     leaves the safe zone) -> zero-lag forward/backward smoothing.
  5. Crops the 9:16 window with sub-pixel precision and scales to the output size.
"""
import numpy as np
import torch
import torch.nn.functional as F

# state carried between chunks when used with VHS Meta Batch Manager
_STATE = {"cam": None, "thumb": None, "hist": None, "std": None, "off": 0}


_YOLO_CACHE = {}

_CLASS_CHOICES = ["(any)", "person", "airplane", "person + airplane", "face"] + [
    c for c in ("bicycle car motorcycle bus train truck boat traffic_light fire_hydrant stop_sign parking_meter bench "
                "bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase "
                "frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
                "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli "
                "carrot hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse "
                "remote keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors "
                "teddy_bear hair_drier toothbrush").split()]
_CLASS_CHOICES = [c.replace("_", " ") if c not in ("(any)",) else c for c in _CLASS_CHOICES]


def _resolve_model(name):
    import os
    cands = [name]
    try:
        import folder_paths
        md = folder_paths.models_dir
        for sub in ("ultralytics/bbox", "ultralytics/segm", "ultralytics", ""):
            cands.append(os.path.join(md, *[p for p in sub.split("/") if p], name))
    except Exception:
        pass
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"YOLO model '{name}' not found. Looked in: {cands}")


def _face_positions(images, model_name, conf, select, track_x, target_class=""):
    """Per-frame face centre along the free axis in source pixels (NaN where no face found)."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("face_yolo mode needs the 'ultralytics' package: " + str(e))
    path = _resolve_model(model_name)
    if path not in _YOLO_CACHE:
        _YOLO_CACHE[path] = YOLO(path)
    model = _YOLO_CACHE[path]
    tc = "" if target_class.strip() == "(any)" else target_class
    wanted = {w.strip().lower() for w in tc.replace("+", ",").replace(";", ",").split(",") if w.strip()}
    class_ids = None
    if wanted:
        class_ids = [int(k) for k, v in model.names.items() if str(v).lower() in wanted]
        if not class_ids:
            raise ValueError(f"target_class {sorted(wanted)} not in this model's classes: {list(model.names.values())[:40]}")
    B = images.shape[0]
    out = np.full(B, np.nan, dtype=np.float64)
    for s in range(0, B, 16):
        e = min(B, s + 16)
        arr = (images[s:e, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()[..., ::-1]  # RGB -> BGR for ultralytics
        res = model.predict([np.ascontiguousarray(a) for a in arr], conf=conf, verbose=False, imgsz=640, classes=class_ids)
        for i, r in enumerate(res):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            lo_i, hi_i = (0, 2) if track_x else (1, 3)
            if select == "all_faces":
                c = 0.5 * (xyxy[:, lo_i].min() + xyxy[:, hi_i].max())
            else:
                if select == "most_confident":
                    k = int(np.argmax(cf))
                else:  # largest
                    k = int(np.argmax((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])))
                c = 0.5 * (xyxy[k, lo_i] + xyxy[k, hi_i])
            out[s + i] = c
    return out


def _fill_nan(x, start):
    """Carry the last known position through frames with no detection."""
    ok = ~np.isnan(x)
    if not ok.any():
        return None
    y = x.copy()
    last = start if start is not None else x[ok][0]
    for i in range(len(y)):
        if np.isnan(y[i]):
            y[i] = last
        else:
            last = y[i]
    return y


def _parse_keyframes(text):
    """'0:0.3, 120:0.5, 300:0.7' -> sorted [(frame, x_fraction)]. x: 0=left edge, 0.5=center, 1=right edge."""
    pts = []
    for tok in text.replace("\n", ",").replace(";", ",").split(","):
        tok = tok.strip()
        if ":" in tok:
            try:
                f, x = tok.split(":", 1)
                pts.append((float(f), min(1.0, max(0.0, float(x)))))
            except ValueError:
                pass
    pts.sort()
    return pts or [(0.0, 0.5)]


def _gray_thumb(imgs, thumb_w):
    """imgs: (B,H,W,3) float -> (B,h,w) grayscale thumbnail."""
    x = imgs[..., :3].permute(0, 3, 1, 2)
    h = max(8, int(round(x.shape[2] * thumb_w / x.shape[3])))
    x = F.interpolate(x, size=(h, thumb_w), mode="area")
    g = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    return g


def _blur(t, k=5):
    t = t.unsqueeze(1)
    t = F.avg_pool2d(t, k, stride=1, padding=k // 2)
    return t.squeeze(1)


def _hist(g, bins=32):
    out = []
    for f in g:
        h = torch.histc(f, bins=bins, min=0.0, max=1.0)
        out.append(h / (h.sum() + 1e-6))
    return torch.stack(out)


def _median(x, k):
    if k <= 1 or len(x) < 3:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def _ema(x, a):
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]
    return y


def _smooth_segment(f, deadzone, smoothing, start=None):
    f = _median(f, 5)
    c = f[0] if start is None else start
    held = np.empty_like(f)
    for i, v in enumerate(f):
        if v > c + deadzone:
            c = v - deadzone
        elif v < c - deadzone:
            c = v + deadzone
        held[i] = c
    a = max(0.02, 1.0 - 0.97 * smoothing)
    fwd = _ema(held, a)
    both = _ema(fwd[::-1], a)[::-1]  # zero-phase: no lag
    return both


class SmartPortraitCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "out_width": ("INT", {"default": 1080, "min": 64, "max": 4320, "step": 2}),
                "out_height": ("INT", {"default": 1920, "min": 64, "max": 7680, "step": 2}),
                "fit_mode": (["crop_to_fill", "blurred_background"],),
                "tracking": (["aircraft", "manual", "yolo_detect", "face_yolo", "auto_saliency", "use_mask"],),
                "target_class": (_CLASS_CHOICES,
                                 {"tooltip": "yolo_detect: what to follow. '(any)' = every class the model knows. Must exist in the model (a face model only knows 'face')."}),
                "camera_min_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                           "tooltip": "Guard rail: the camera center never goes left of this (0 = left edge, 1 = right edge). Works in every mode."}),
                "camera_max_x": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                           "tooltip": "Guard rail: the camera center never goes right of this."}),
                "yolo_model": ("STRING", {"default": "face_yolov8n.pt",
                                          "tooltip": "face_yolo mode: model file name (searched in ComfyUI/models/ultralytics/bbox etc.) or a full path."}),
                "detect_conf": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.95, "step": 0.05}),
                "face_select": (["largest", "most_confident", "all_faces"],),
                "smoothing": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "deadzone": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.5, "step": 0.01,
                                       "tooltip": "Fraction of the crop width the subject can drift before the camera moves."}),
                "motion_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "detail_weight": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 4.0, "step": 0.05}),
                "center_bias": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05}),
                "hold_from_frame": ("INT", {"default": -1, "min": -1, "max": 10000000,
                                            "tooltip": "From this frame on, freeze the camera (e.g. once the boom has connected). -1 = off. 30 fps: frame 300 = 10 s."}),
                "hold_x": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                     "tooltip": "Where to freeze the camera, 0 (left) to 1 (right). -1 = freeze exactly where the camera was at hold_from_frame."}),
                "x_offset": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01,
                                       "tooltip": "Shift the final camera left (-) or right (+), as a fraction of the video width. Works in every mode."}),
                "camera_keyframes": ("STRING", {"default": "0:0.5", "multiline": True,
                                                "tooltip": "Manual mode: 'frame:x' pairs, x from 0 (left edge) to 1 (right edge). Example: 0:0.35, 240:0.5, 600:0.65"}),
                "ignore_static": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                                            "tooltip": "Aircraft mode: suppress parts of the frame that never change (window frame, pod interior, cockpit)."}),
                "scene_cut_sensitivity": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0, "step": 0.01,
                                                    "tooltip": "Histogram distance that counts as a cut. Lower = more cuts detected."}),
            },
            "optional": {
                "masks": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "camera_path_info")
    FUNCTION = "run"
    CATEGORY = "video/reframe"

    def run(self, images, out_width, out_height, fit_mode, tracking, smoothing, deadzone,
            motion_weight, detail_weight, center_bias, ignore_static, scene_cut_sensitivity,
            x_offset=0.0, camera_keyframes="0:0.5", hold_from_frame=-1, hold_x=-1.0,
            yolo_model="face_yolov8n.pt", detect_conf=0.25, face_select="largest", target_class="(any)", camera_min_x=0.0, camera_max_x=1.0, masks=None):
        B, H, W, _ = images.shape
        aspect = out_width / out_height
        track_x = (W / H) > aspect  # landscape-ish source -> slide window left/right
        if track_x:
            ch, cw = float(H), H * aspect
            free, size_free = W, cw
        else:
            cw, ch = float(W), W / aspect
            free, size_free = H, ch

        # ---------- analysis on thumbnails ----------
        thumb_w = 192
        g = _gray_thumb(images, thumb_w).cpu()
        th, tw = g.shape[1], g.shape[2]
        prev = _STATE["thumb"]
        prev_ok = prev is not None and prev.shape == g[0].shape
        g_prev = torch.cat([(prev if prev_ok else g[0]).unsqueeze(0), g[:-1]], 0)

        hist = _hist(g)
        hist_prev = torch.cat([(_STATE["hist"] if prev_ok and _STATE["hist"] is not None else hist[0]).unsqueeze(0),
                               hist[:-1]], 0)
        cut_score = 0.5 * (hist - hist_prev).abs().sum(1)
        cuts = (cut_score > scene_cut_sensitivity).numpy()
        # first frame of this chunk: continue the previous chunk's camera unless it is a cut
        continues = prev_ok and not bool(cuts[0])
        cuts[0] = not continues

        motion = _blur((g - g_prev).abs())
        gx = (g[:, :, 1:] - g[:, :, :-1]).abs()
        gy = (g[:, 1:, :] - g[:, :-1, :]).abs()
        detail = _blur(F.pad(gx, (0, 1)) + F.pad(gy, (0, 0, 0, 1)), 7)
        motion = motion / (motion.flatten(1).max(1).values.view(-1, 1, 1) + 1e-6)
        detail = detail / (detail.flatten(1).max(1).values.view(-1, 1, 1) + 1e-6)
        score = motion_weight * motion + detail_weight * detail

        axis_len = tw if track_x else th
        pos = torch.linspace(0, 1, axis_len)
        prior = center_bias * torch.exp(-((pos - 0.5) ** 2) / (2 * 0.22 ** 2))
        prior = prior.view(1, 1, -1) if track_x else prior.view(1, -1, 1)

        aircraft = tracking == "aircraft"
        if aircraft:
            # Aircraft stand out from smooth sky: local contrast against a heavy blur of the frame.
            # Independent of camera panning (the plane may be static in frame while the sky moves).
            bgp = F.avg_pool2d(g.unsqueeze(1), 51, stride=1, padding=25, count_include_pad=False).squeeze(1)
            contrast = _blur((g - bgp).abs(), 3)
            contrast = contrast / (contrast.flatten(1).max(1).values.view(-1, 1, 1) + 1e-6)
            if ignore_static > 0 and B > 1:
                # Fixed camera (e.g. boom pod): window frame / interior never change, so their per-pixel
                # variation over time is ~0. Down-weight them; the receiver aircraft and boom do vary.
                sd = g.std(0)
                if _STATE["std"] is not None and _STATE["std"].shape == sd.shape:
                    sd = torch.maximum(sd, _STATE["std"] * 0.98)
                _STATE["std"] = sd
                live = (sd / (sd.max() + 1e-6)).clamp(0, 1) ** 0.5
                contrast = contrast * ((1 - ignore_static) + ignore_static * live).unsqueeze(0)
            score = contrast ** 2 + 0.3 * detail_weight * detail + 0.25 * motion_weight * motion + 1e-4
        else:
            score = score + prior * score.flatten(1).mean(1).view(-1, 1, 1) * 2 + 1e-4

        if tracking == "use_mask" and masks is not None:
            m = masks if masks.dim() == 3 else masks.squeeze(1)
            if m.shape[0] != B:  # single mask for all frames, or mismatch -> repeat/trim
                m = m[:1].expand(B, -1, -1) if m.shape[0] == 1 else m[:B]
            m = F.interpolate(m.unsqueeze(1).float().cpu(), size=(th, tw), mode="area").squeeze(1)
            has = (m.flatten(1).sum(1) > 1e-3).view(-1, 1, 1)
            score = torch.where(has, m + 1e-4, score)

        prof = score.sum(1) if track_x else score.sum(2)  # (B, axis_len)
        prof = prof ** 2  # sharpen so the strongest blob wins over diffuse noise
        if aircraft and not (tracking == "use_mask" and masks is not None):
            # Stick to one object: after the first frame of a shot, only look near where the plane was,
            # then take the centroid of the strongest peak (ignores clouds / ground clutter elsewhere).
            last = _STATE["cam"] / free if (not cuts[0] and _STATE["cam"] is not None) else None
            cl = []
            for i in range(B):
                p = prof[i]
                if cuts[i]:
                    last = None
                if last is not None:
                    p = p * (0.1 + torch.exp(-((pos - last) ** 2) / (2 * 0.18 ** 2)))
                pk = int(torch.argmax(p))
                win = max(2, int(0.30 * axis_len))  # wide: the receiver is huge, spans much of the frame
                lo_i, hi_i = max(0, pk - win), min(axis_len, pk + win + 1)
                seg = p[lo_i:hi_i]
                seg = seg * (seg > 0.2 * seg.max()).float()  # centroid of the whole object, not just its strongest edge
                c = float((seg * pos[lo_i:hi_i]).sum() / (seg.sum() + 1e-9))
                cl.append(c)
                last = c
            centers = torch.tensor(cl)
        else:
            centers = (prof * pos.view(1, -1)).sum(1) / (prof.sum(1) + 1e-9)
        raw = (centers.numpy() * free).astype(np.float64)

        face_info = ""
        if tracking in ("face_yolo", "yolo_detect"):
            fp = _face_positions(images, yolo_model, detect_conf, face_select, track_x, target_class)
            found = int((~np.isnan(fp)).sum())
            start = _STATE["cam"] if (continues and _STATE["cam"] is not None) else None
            filled = _fill_nan(fp, start)
            if filled is not None:
                raw = filled  # faces drive the camera; frames with no face keep the last position
            face_info = f" | detections in {found}/{B} frames" + ("" if found else " (none: fell back to saliency)")

        # global frame index (stays correct across Meta Batch chunks; restarts on a new video)
        base = _STATE["off"] if continues else 0
        if not continues:
            _STATE["hold"] = None
        manual = tracking == "manual"
        if manual:
            kp = _parse_keyframes(camera_keyframes)
            raw = np.interp(base + np.arange(B), [k[0] for k in kp], [k[1] for k in kp]) * free
        raw = raw + x_offset * free

        # ---------- smooth per scene ----------
        cam = np.empty(B, dtype=np.float64)
        if manual:
            cam[:] = _ema(_ema(raw, 0.3)[::-1], 0.3)[::-1]  # keyframes are already smooth; just round corners
        else:
            starts = [i for i in range(B) if cuts[i]] or [0]
            if starts[0] != 0:
                starts = [0] + starts
            starts.append(B)
            for a, b in zip(starts[:-1], starts[1:]):
                init = _STATE["cam"] if (a == 0 and continues and _STATE["cam"] is not None) else None
                cam[a:b] = _smooth_segment(raw[a:b], deadzone * size_free, smoothing, init)
        lo, hi = size_free / 2, free - size_free / 2
        cam = np.clip(cam, lo, hi)

        # ---------- optional hold: freeze the camera from a given frame on ----------
        if hold_from_frame >= 0:
            idx = base + np.arange(B)
            if hold_x >= 0:
                target = float(np.clip(hold_x * free, lo, hi))
            else:
                if _STATE.get("hold") is None and idx[-1] >= hold_from_frame:
                    _STATE["hold"] = float(cam[max(0, hold_from_frame - base)])
                target = _STATE.get("hold")
            if target is not None:
                w = np.clip((idx - hold_from_frame) / 15.0, 0.0, 1.0)  # 0.5 s ease so it never snaps
                cam = cam * (1 - w) + target * w

        # ---------- guard rail: keep the camera inside a chosen range ----------
        g_lo, g_hi = sorted((camera_min_x * free, camera_max_x * free))
        cam = np.clip(cam, max(lo, g_lo), max(max(lo, g_lo), min(hi, g_hi)))

        _STATE["cam"], _STATE["thumb"], _STATE["hist"] = float(cam[-1]), g[-1].clone(), hist[-1].clone()
        _STATE["off"] = base + B

        # ---------- render ----------
        out = torch.empty((B, out_height, out_width, 3), dtype=torch.float32)
        dev = images.device
        chunk = 16
        for s in range(0, B, chunk):
            e = min(B, s + chunk)
            x = images[s:e, ..., :3].permute(0, 3, 1, 2).to(dev).float()
            n = e - s
            c = torch.tensor(cam[s:e], dtype=torch.float32, device=dev)
            theta = torch.zeros((n, 2, 3), device=dev)
            if track_x:
                theta[:, 0, 0] = cw / W
                theta[:, 1, 1] = 1.0
                theta[:, 0, 2] = 2 * c / W - 1
            else:
                theta[:, 0, 0] = 1.0
                theta[:, 1, 1] = ch / H
                theta[:, 1, 2] = 2 * c / H - 1

            if fit_mode == "blurred_background":
                bw, bh = max(8, out_width // 8), max(8, out_height // 8)
                grid = F.affine_grid(theta, (n, 3, bh, bw), align_corners=False)
                bg = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
                bg = F.interpolate(bg, size=(out_height, out_width), mode="bilinear", align_corners=False)
                bg = bg * 0.55  # dim so the foreground pops
                # whole source frame, fitted inside the output and centred (works for both directions:
                # landscape -> portrait gets bars top/bottom, portrait -> landscape gets bars left/right)
                sc = min(out_width / W, out_height / H)
                fw, fh = min(out_width, int(round(W * sc))), min(out_height, int(round(H * sc)))
                fg = F.interpolate(x, size=(fh, fw), mode="bicubic", align_corners=False)
                top, left = (out_height - fh) // 2, (out_width - fw) // 2
                bg[:, :, top:top + fh, left:left + fw] = fg
                res = bg
            else:
                grid = F.affine_grid(theta, (n, 3, out_height, out_width), align_corners=False)
                res = F.grid_sample(x, grid, mode="bicubic", padding_mode="border", align_corners=False)
            out[s:e] = res.clamp(0, 1).permute(0, 2, 3, 1).cpu()

        info = (f"source {W}x{H} -> {out_width}x{out_height} | frames {B} | cuts {int(cuts.sum())}{face_info} | "
                f"camera range {cam.min():.0f}-{cam.max():.0f}px of {free}px | "
                f"path (frame:x 0-1) " + ", ".join(f"{base + i}:{(cam[i] - lo) / max(1e-6, hi - lo):.2f}"
                                                   for i in range(0, B, max(1, B // 10))))
        return (out, info)


NODE_CLASS_MAPPINGS = {"SmartPortraitCrop": SmartPortraitCrop}
NODE_DISPLAY_NAME_MAPPINGS = {"SmartPortraitCrop": "Smart Portrait Crop (9:16)"}
