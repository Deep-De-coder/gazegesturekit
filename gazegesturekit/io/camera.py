from __future__ import annotations
import cv2, time
from typing import Iterator, Dict, Any

def frames(camera: int|str=0, width: int=1280, height: int=720) -> Iterator[Dict[str,Any]]:
    cap = cv2.VideoCapture(camera)
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            yield {"image": frame, "meta": {"ts": time.time()}}
    finally:
        cap.release()
