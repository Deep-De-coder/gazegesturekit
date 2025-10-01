from __future__ import annotations
import pyautogui, time

def move_and_click(x:int, y:int, action:str|None=None):
    try:
        pyautogui.moveTo(x, y, duration=0.0)
        if action in ("select","click"): pyautogui.click()
        elif action=="scroll":
            pyautogui.scroll(-500)
    except Exception:
        pass
