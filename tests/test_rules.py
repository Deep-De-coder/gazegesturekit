from gazegesturekit.fuse.rules import RuleEngine
def test_rule_match():
    cfg = {"window_ms":300,"rules":[{"when":"gaze.on_screen and hand.gesture=='pinch'","do":{"type":"select"}}]}
    eng = RuleEngine(cfg)
    ev = eng.update({"fixation_ms":300,"dx":0,"dy":0}, {"gesture":"pinch"})
    assert isinstance(ev, list)
