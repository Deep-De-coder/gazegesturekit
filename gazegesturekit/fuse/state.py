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
        self._dragging=False
        self._drag_start_pos=None
        self._drag_start_time=None
        self._last_gaze_pos=None
        self._drag_threshold=50  # pixels to move before drag starts

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
        
        # Get current gaze position
        current_gaze_pos = None
        if gaze:
            current_gaze_pos = (gaze.get("x", 0), gaze.get("y", 0))
        
        # Drag detection: pinch held + gaze movement
        is_pinch_held = hand.get("gesture") == "pinch"
        
        if is_pinch_held and current_gaze_pos:
            if not self._dragging:
                # Check if we should start dragging
                if self._drag_start_pos is None:
                    # First pinch - initialize drag start
                    self._drag_start_pos = current_gaze_pos
                    self._drag_start_time = t
                else:
                    # Calculate distance from drag start
                    dx = current_gaze_pos[0] - self._drag_start_pos[0]
                    dy = current_gaze_pos[1] - self._drag_start_pos[1]
                    dist = (dx*dx + dy*dy) ** 0.5
                    
                    # Also check if gaze is moving (velocity threshold)
                    gaze_dx = abs(gaze.get("dx", 0))
                    gaze_dy = abs(gaze.get("dy", 0))
                    is_moving = (gaze_dx > 50) or (gaze_dy > 50)
                    
                    # Start drag if moved enough distance or moving fast
                    if dist > self._drag_threshold or (is_moving and dist > 20):
                        self._dragging = True
                        out.append({
                            "type":"drag", 
                            "extra": {"action":"start", "start_pos":self._drag_start_pos}
                        })
            else:
                # Already dragging - emit drag event
                if self._last_gaze_pos:
                    dx = current_gaze_pos[0] - self._last_gaze_pos[0]
                    dy = current_gaze_pos[1] - self._last_gaze_pos[1]
                    if abs(dx) > 5 or abs(dy) > 5:  # Only emit if moved significantly
                        out.append({
                            "type":"drag", 
                            "extra": {"action":"move", "delta":(dx, dy)}
                        })
        else:
            # Not pinching - end drag if was dragging
            if self._dragging:
                out.append({
                    "type":"drag", 
                    "extra": {"action":"end"}
                })
                self._dragging = False
            self._drag_start_pos = None
            self._drag_start_time = None
        
        # Update last gaze position
        if current_gaze_pos:
            self._last_gaze_pos = current_gaze_pos
        
        # Hover logic
        if gaze and gaze.get("fixation_ms",0) > 400 and not self._hovering:
            out.append({"type":"hover_start"})
            self._hovering=True
        if self._hovering and gaze and gaze.get("fixation_ms",0) < 100:
            out.append({"type":"hover_end"}); self._hovering=False

        # Select/Click via pinch or point/thumbs_up (but not if dragging)
        if hand.get("gesture") in ("pinch","point","thumbs_up") and not self._dragging:
            # Only emit select if we haven't started dragging
            if not is_pinch_held or (self._drag_start_pos is None or 
                ((current_gaze_pos and self._drag_start_pos) and 
                 ((current_gaze_pos[0] - self._drag_start_pos[0])**2 + 
                  (current_gaze_pos[1] - self._drag_start_pos[1])**2) ** 0.5 < self._drag_threshold)):
                evt={"type":"select"}
                if self._double_click(t): evt={"type":"double_click"}
                out.append(evt)

        # Cancel via palm or fist
        if hand.get("gesture") in ("palm","fist"):
            out.append({"type":"cancel"})
            # Also cancel drag
            if self._dragging:
                out.append({"type":"drag", "action":"end"})
                self._dragging = False
                self._drag_start_pos = None

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
