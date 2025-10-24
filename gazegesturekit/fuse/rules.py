from __future__ import annotations
import yaml
from typing import Dict, Any, Optional, List
from .state import FusionSM

class RuleEngine:
    """
    Backwards-compatible: loads YAML to configure window and still exposes update().
    Internally uses FusionSM for priority/conflict resolution.
    """
    def __init__(self, cfg: Dict[str,Any]):
        self.sm = FusionSM(window_ms=int(cfg.get("window_ms", 300)))
        self.cfg = cfg

    def update(self, gaze: Optional[Dict[str,Any]], hand: Optional[Dict[str,Any]]) -> List[Dict[str,Any]]:
        gaze = gaze or {}
        hand = hand or {}
        return self.sm.fuse(gaze, hand)

def load_rules(path:str) -> RuleEngine:
    with open(path,"r") as f: cfg=yaml.safe_load(f)
    return RuleEngine(cfg)
