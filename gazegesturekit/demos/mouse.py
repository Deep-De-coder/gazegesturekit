from __future__ import annotations
import pyautogui, time

class DragState:
    """Track drag state across calls"""
    def __init__(self):
        self.is_dragging = False
        self.start_pos = None
        self.last_pos = None

_drag_state = DragState()

def move_and_click(x:int, y:int, action:str|None=None, event_extra:dict|None=None):
    """
    Move cursor and perform action.
    event_extra can contain drag event information.
    """
    global _drag_state
    
    try:
        # Handle drag events if provided
        if event_extra and "action" in event_extra:
            drag_action = event_extra.get("action")
            
            if drag_action == "start":
                # Start drag: press mouse button and move to start position
                start_pos = event_extra.get("start_pos", (x, y))
                pyautogui.moveTo(start_pos[0], start_pos[1], duration=0.0)
                pyautogui.mouseDown()
                _drag_state.is_dragging = True
                _drag_state.start_pos = start_pos
                _drag_state.last_pos = start_pos
                
            elif drag_action == "move" and _drag_state.is_dragging:
                # Continue drag: move mouse with button held
                delta = event_extra.get("delta", (0, 0))
                if _drag_state.last_pos:
                    new_x = _drag_state.last_pos[0] + delta[0]
                    new_y = _drag_state.last_pos[1] + delta[1]
                    pyautogui.moveTo(new_x, new_y, duration=0.0)
                    _drag_state.last_pos = (new_x, new_y)
                else:
                    pyautogui.moveTo(x, y, duration=0.0)
                    _drag_state.last_pos = (x, y)
                    
            elif drag_action == "end" and _drag_state.is_dragging:
                # End drag: release mouse button
                pyautogui.mouseUp()
                _drag_state.is_dragging = False
                _drag_state.start_pos = None
                _drag_state.last_pos = None
                
            # If handling drag, don't process normal click
            return
        
        # Normal cursor movement
        if not _drag_state.is_dragging:
            pyautogui.moveTo(x, y, duration=0.0)
        
        # Handle non-drag actions
        if action in ("select","click"):
            if not _drag_state.is_dragging:
                pyautogui.click()
        elif action=="scroll":
            pyautogui.scroll(-500)
    except Exception as e:
        # Silently handle errors to avoid disrupting the main loop
        pass

def reset_drag_state():
    """Reset drag state (useful for cleanup)"""
    global _drag_state
    if _drag_state.is_dragging:
        try:
            pyautogui.mouseUp()
        except:
            pass
    _drag_state.is_dragging = False
    _drag_state.start_pos = None
    _drag_state.last_pos = None
