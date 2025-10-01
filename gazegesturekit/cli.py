from __future__ import annotations
import typer, json, asyncio, cv2, sys
from rich import print
from pathlib import Path
from typing import Optional
from .io.camera import frames
from .calibrate.wizard import Calibrator
from .eye.gaze import GazeEstimator
from .hand.landmarks import HandLandmarks
from .hand.gestures import classify
from .fuse.rules import load_rules, RuleEngine
from .runtime.events import Event, Gaze, Hand, ws_broadcast
from .demos.mouse import move_and_click

app = typer.Typer(add_completion=False, help="GazeGestureKit CLI (ggk)")

@app.command()
def calibrate(points:int=typer.Option(5), camera:Optional[int]=0, width:int=1280, height:int=720, save:str=".ggk_calibration.json"):
    """
    Show invisible targets and record raw gaze to learn a polynomial mapping to screen coords.
    """
    est = GazeEstimator(screen=(width,height))
    def get_raw():
        for f in frames(camera, width, height):
            res = est(f["image"])
            if res is None: continue
            gx,gy = res["screen_xy"]
            # Back to 0..1 approximation for calibration; normalize
            return (gx/width, gy/height)
    params = Calibrator(points=points, save_path=save).run(get_raw_gaze=get_raw, screen_size=(width,height))
    print("[green]Saved calibration[/green]", save, "=>", params.keys())

@app.command()
def demo(action: str = typer.Argument("mouse"), camera:Optional[int]=0, width:int=1280, height:int=720, calibration:str=".ggk_calibration.json"):
    """
    Live demo: move cursor with gaze; pinch = click.
    """
    cal = None
    if Path(calibration).exists():
        import json; cal = json.loads(Path(calibration).read_text())
    est = GazeEstimator(calibration=cal, screen=(width,height))
    hands = HandLandmarks()
    for f in frames(camera, width, height):
        res = est(f["image"])
        if res is None: continue
        x,y = res["screen_xy"]
        hs = hands(f["image"]); hg=None
        if hs:
            hg = classify(hs[0])
        # draw debug preview
        dbg = f["image"].copy()
        cv2 = __import__("cv2")
        cv2.circle(dbg,(x,y),6,(0,255,0),-1)
        cv2.imshow("GazeGestureKit", dbg)
        cv2.waitKey(1)
        act = None
        if hg and hg["gesture"]=="pinch": act="click"
        move_and_click(x,y, act)

@app.command()
def run(rules: str = typer.Option("examples/rules.yaml"), ws: Optional[str]=typer.Option(None), camera:Optional[int]=0, width:int=1280, height:int=720, calibration:str=".ggk_calibration.json"):
    """
    Run fusion engine and print JSONL events; optionally broadcast over WebSocket.
    """
    import json, time
    cal = None
    if Path(calibration).exists():
        cal = json.loads(Path(calibration).read_text())
    est = GazeEstimator(calibration=cal, screen=(width,height))
    hands = HandLandmarks()
    eng: RuleEngine = load_rules(rules)

    queue: "asyncio.Queue[str]" = asyncio.Queue()

    async def producer():
        for f in frames(camera, width, height):
            res = est(f["image"])
            hs = hands(f["image"])
            hand = classify(hs[0]) if hs else {"gesture":None,"conf":0.0,"handedness":None}
            events=[]
            if res:
                gaze = {
                    "x": res["screen_xy"][0], "y": res["screen_xy"][1], "conf": res["conf"],
                    "fixation_ms": res["fixation_ms"], "dx": res["dx"], "dy": res["dy"]
                }
                events = eng.update(gaze=gaze, hand=hand)
            for ev in events:
                e = Event(type=ev["type"]).model_copy(update={"gaze":gaze, "hand":hand})
                line = e.model_dump_json()
                print(line)
                if ws: await queue.put(line)
            # press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    async def main():
        if ws:
            host, port = ("0.0.0.0", 8765)
            bcast = asyncio.create_task(ws_broadcast(queue, host, port))
            prod  = asyncio.create_task(producer())
            await asyncio.gather(prod, bcast)
        else:
            await producer()

    asyncio.run(main())

if __name__ == "__main__":
    app()
