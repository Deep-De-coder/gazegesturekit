from __future__ import annotations
import time
from collections import deque
from typing import Deque, Dict, Any, Optional, List

class History:
    def __init__(self, maxlen:int=30):
        self.events: Deque[Dict[str,Any]] = deque(maxlen=maxlen)
    def add(self, ev: Dict[str,Any]):
        self.events.append(ev)
    def last(self, typ:str) -> Optional[Dict[str,Any]]:
        for e in reversed(self.events):
            if e.get("type")==typ: return e
        return None

class FusionSM:
    """
    Simple state machine with priorities:
    zoom > drag > select/click > scroll > hover > cancel
    """
    PRIORITY = {"zoom":5,"drag":4,"double_click":4,"select":3,"click":3,"scroll":2,"hover_start":1,"hover_end":1,"cancel":0}

    def __init__(self, window_ms:int=300):
        self.window_ms=window_ms
        self.hist = History()
        self._hovering=False
        self._last_click_t=0.0

    def _double_click(self, t:float) -> bool:
        if (t - self._last_click_t) < 0.35:
            self._last_click_t = 0.0
            return True
        self._last_click_t = t
        return False

    def fuse(self, gaze: Dict[str,Any], hand: Dict[str,Any]) -> List[Dict[str,Any]]:
        """
        Input: latest gaze & hand dicts
        Output: list of prioritized events
        """
        out=[]
        t=time.time()
        # Hover logic
        if gaze and gaze.get("fixation_ms",0) > 400 and not self._hovering:
            out.append({"type":"hover_start"})
            self._hovering=True
        if self._hovering and gaze and gaze.get("fixation_ms",0) < 100:
            out.append({"type":"hover_end"}); self._hovering=False

        # Select/Click via pinch or blink+fixation
        if hand.get("gesture") in ("pinch","point","thumbs_up"):
            evt={"type":"select"}
            if self._double_click(t): evt={"type":"double_click"}
            out.append(evt)

        # Cancel via palm or fist
        if hand.get("gesture") in ("palm","fist"):
            out.append({"type":"cancel"})

        # Scroll via high gaze dx while pointing
        if abs(gaze.get("dx",0))>300 and hand.get("gesture")=="point":
            out.append({"type":"scroll","direction":"auto"})

        # Prioritize
        out.sort(key=lambda e: self.PRIORITY.get(e["type"],0), reverse=True)
        # Attach context
        for e in out:
            e["ts"]=t; e["gaze"]=gaze; e["hand"]=hand
            self.hist.add(e)
        return out
