from __future__ import annotations
import cv2, numpy as np

def normalize_eye_patch(gray: np.ndarray) -> np.ndarray:
    # CLAHE + mild gamma
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    g = clahe.apply(gray)
    g = np.clip((g/255.0) ** 0.9 * 255.0, 0, 255).astype(np.uint8)
    return g

def remove_glints(gray: np.ndarray) -> np.ndarray:
    # remove small bright spots (glints) by inpainting mask
    thr = min(255, int(np.percentile(gray, 98)))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    if mask.sum() == 0: return gray
    return cv2.inpaint(gray, mask, 2, cv2.INPAINT_TELEA)

def estimate_pupil_center(eye_roi_bgr: np.ndarray, eye_mask: np.ndarray|None=None):
    """Return (cx, cy, conf[0..1]) in ROI coords."""
    gray = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = normalize_eye_patch(gray)
    gray = remove_glints(gray)

    if eye_mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=eye_mask)

    # Adaptive threshold (look for dark pupil)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 21, 5)
    thr = cv2.medianBlur(thr, 3)
    # Morphology to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 8:  # too tiny
        return None, 0.0
    (x,y), (MA,ma), angle = cv2.fitEllipse(cnt)
    # conf: darker mean + larger area => higher
    mean_int = float(np.mean(gray[thr>0])) if (thr>0).any() else 255.0
    conf = float(np.clip((area/ (gray.shape[0]*gray.shape[1]))*2.0 + (1.0 - mean_int/255.0), 0, 1))
    return (float(x), float(y)), conf
