from __future__ import annotations
import yaml, json, time
from typing import Dict, Any, Optional

class RuleEngine:
    def __init__(self, cfg: Dict[str,Any]):
        self.window_ms = int(cfg.get("window_ms", 300))
        self.min_fix = int(cfg.get("min_fixation_ms", 250))
        self.rules = cfg.get("rules", [])
        self.last_hand = None
        self.last_gaze = None

    def update(self, gaze: Optional[Dict[str,Any]], hand: Optional[Dict[str,Any]]):
        t = time.time()*1000
        if gaze: self.last_gaze = (t, gaze)
        if hand: self.last_hand = (t, hand)
        evs=[]
        if self.last_gaze and self.last_hand:
            tg,g = self.last_gaze; th,h = self.last_hand
            if abs(tg-th) <= self.window_ms:
                ctx = {
                    "gaze": {"on_screen": True, "fixation_ms": g.get("fixation_ms",0), "dx": g.get("dx",0), "dy": g.get("dy",0)},
                    "hand": {"gesture": h.get("gesture")},
                    "eye":  {"blink": "single" if g.get("blink") else None}
                }
                for r in self.rules:
                    if self._match(r["when"], ctx):
                        ev = {"ts": t/1000.0, **r["do"]}
                        evs.append(ev)
        return evs

    def _match(self, expr: str, ctx: Dict[str,Any]) -> bool:
        # very small safe eval
        g=ctx["gaze"]; h=ctx["hand"]; e=ctx["eye"]
        try:
            return bool(eval(expr, {"__builtins__": {}}, {"gaze":g, "hand":h, "eye":e}))
        except Exception:
            return False

def load_rules(path:str) -> RuleEngine:
    with open(path,"r") as f: cfg=yaml.safe_load(f)
    return RuleEngine(cfg)
