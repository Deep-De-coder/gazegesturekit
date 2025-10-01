from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any
import json, asyncio, websockets, time

class Gaze(BaseModel):
    x:int; y:int; conf:float=0.0; fixation_ms:int=0; dx:float=0.0; dy:float=0.0

class Hand(BaseModel):
    gesture: Optional[str]=None
    handedness: Optional[str]=None
    conf: float=0.0

class Event(BaseModel):
    ts: float = Field(default_factory=lambda: time.time())
    type: Literal["select","click","drag","scroll","cancel","debug"]
    gaze: Optional[Gaze]=None
    hand: Optional[Hand]=None
    extra: Dict[str,Any] = {}

async def ws_broadcast(queue: "asyncio.Queue[str]", host="0.0.0.0", port=8765):
    clients=set()
    async def handler(websocket):
        clients.add(websocket)
        try:
            while True:
                msg = await queue.get()
                await asyncio.gather(*[c.send(msg) for c in list(clients)])
        except Exception:
            pass
        finally:
            clients.remove(websocket)
    async with websockets.serve(handler, host, port):
        await asyncio.Future()
