from __future__ import annotations
import mediapipe as mp
import numpy as np
import cv2

class HandLandmarks:
    def __init__(self, max_hands=2):
        self.hands = mp.solutions.hands.Hands(max_num_hands=max_hands, model_complexity=0)
    def __call__(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks: return []
        out=[]
        for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            pts = np.array([(p.x,p.y,p.z) for p in lm.landmark], dtype=float)
            out.append({"pts":pts, "handedness": handed.classification[0].label.lower()})
        return out
