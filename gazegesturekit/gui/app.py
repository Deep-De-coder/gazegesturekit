from __future__ import annotations
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QCheckBox, QComboBox, QTextEdit, QMessageBox,
    QFrame, QSplitter
)
from PySide6.QtCore import QThread, Signal, QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import qdarkstyle
import pyautogui

from ..eye.gaze import GazeEstimator
from ..hand.landmarks import HandLandmarks
from ..hand.gestures import classify
from ..calibrate.wizard import Calibrator
from ..fuse.rules import load_rules
from ..runtime.events import Event
import pyautogui


class GazeWorker(QThread):
    """Worker thread for gaze tracking and event processing."""
    
    frame_ready = Signal(np.ndarray, dict, object)  # frame, gaze, event
    error_occurred = Signal(str)
    verify_status = Signal(dict)  # verification status updates
    confirm = Signal(int, int, int, int)  # frame_w, frame_h, gaze_x, gaze_y
    
    def __init__(self, camera_index: int, calibration_path: str = ".ggk_calibration.json"):
        super().__init__()
        self.camera_index = camera_index
        self.calibration_path = calibration_path
        self.running = False
        self.cap = None
        self.estimator = None
        self.hands = None
        self.rule_engine = None
        self.calibration = None
        self.last_verify_status = {}
        
    def load_calibration(self):
        """Load calibration data if available."""
        if Path(self.calibration_path).exists():
            try:
                with open(self.calibration_path, 'r') as f:
                    self.calibration = json.load(f)
            except Exception as e:
                print(f"Failed to load calibration: {e}")
                self.calibration = None
    
    def start_capture(self):
        """Start the gaze tracking process."""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera {self.camera_index}")
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Load calibration
            self.load_calibration()
            
            # Initialize components
            self.estimator = GazeEstimator(
                calibration=self.calibration,
                screen=(1280, 720)
            )
            self.hands = HandLandmarks()
            
            # Load rules
            try:
                self.rule_engine = load_rules("examples/rules.yaml")
            except:
                # Fallback if rules file doesn't exist
                self.rule_engine = None
            
            self.running = True
            self.start()
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to start capture: {str(e)}")
    
    def stop_capture(self):
        """Stop the gaze tracking process."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.wait()
    
    def run(self):
        """Main worker loop."""
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                gaze_result = self.estimator(frame)
                hand_results = self.hands(frame)
                hand_data = classify(hand_results[0]) if hand_results else {"gesture": None, "conf": 0.0, "handedness": None}
                
                # Check for pinch confirmation
                if gaze_result and hand_data.get("gesture") == "pinch" and hand_data.get("conf", 0.0) >= 0.6:
                    h, w = frame.shape[:2]
                    gx, gy = gaze_result.get("screen_xy", (0, 0))
                    self.confirm.emit(w, h, int(gx), int(gy))
                
                # Update verification status
                verify_status = {
                    "face_detected": gaze_result is not None,
                    "gaze_detected": gaze_result is not None and "screen_xy" in gaze_result,
                    "gaze_confidence": gaze_result.get("conf", 0.0) if gaze_result else 0.0,
                    "gaze_position": gaze_result.get("screen_xy", (0, 0)) if gaze_result else None,
                    "hands_detected": len(hand_results) > 0 if hand_results else False,
                    "num_hands": len(hand_results) if hand_results else 0,
                    "gesture_detected": hand_data.get("gesture") is not None,
                    "gesture": hand_data.get("gesture"),
                    "gesture_confidence": hand_data.get("conf", 0.0),
                    "handedness": hand_data.get("handedness"),
                    "pupils_detected": gaze_result is not None  # If gaze detected, pupils were found
                }
                self.last_verify_status = verify_status
                self.verify_status.emit(verify_status)
                
                # Generate events
                event = None
                if gaze_result and self.rule_engine:
                    gaze_dict = {
                        "x": gaze_result["screen_xy"][0],
                        "y": gaze_result["screen_xy"][1],
                        "conf": gaze_result["conf"],
                        "fixation_ms": gaze_result["fixation_ms"],
                        "dx": gaze_result["dx"],
                        "dy": gaze_result["dy"]
                    }
                    events = self.rule_engine.update(gaze=gaze_dict, hand=hand_data)
                    if events:
                        event = events[0]  # Take first event
                        # Ensure extra field exists (should already be set by FusionSM)
                        if "extra" not in event:
                            event["extra"] = {}
                
                # Emit frame and data
                self.frame_ready.emit(frame, gaze_result or {}, event)
                
                # Target ~30 FPS
                time.sleep(1/30)
                
            except Exception as e:
                self.error_occurred.emit(f"Processing error: {str(e)}")
                time.sleep(0.1)


class GazePreview(QLabel):
    """Custom label for displaying camera feed with gaze overlay."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("No camera feed")
        self.gaze_point = None
        self.frame = None
        self.confirm_pt = None
        self.confirm_until = 0.0
    
    def update_frame(self, frame: np.ndarray, gaze: dict, confirm_pt=None, confirm_until=0.0):
        """Update the preview with new frame and gaze point."""
        self.frame = frame.copy()
        if gaze and "screen_xy" in gaze:
            self.gaze_point = gaze["screen_xy"]
        else:
            self.gaze_point = None
        self.confirm_pt = confirm_pt
        self.confirm_until = confirm_until
        self.update()
    
    def paintEvent(self, event):
        """Custom paint event to draw gaze point."""
        super().paintEvent(event)
        
        if self.frame is None:
            return
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create QImage
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        painter = QPainter(scaled_pixmap)
        
        # Draw pinch confirmation marker (green) if active
        if time.time() < self.confirm_until and self.confirm_pt:
            # Scale confirm point from frame coordinates to preview coordinates
            scale_x = scaled_pixmap.width() / w
            scale_y = scaled_pixmap.height() / h
            px = int(self.confirm_pt[0] * scale_x)
            py = int(self.confirm_pt[1] * scale_y)
            
            # Draw bright green circle/crosshair
            painter.setPen(QPen(QColor(0, 255, 0), 3))
            painter.setBrush(QColor(0, 255, 0, 180))
            radius = 12
            painter.drawEllipse(px - radius, py - radius, radius * 2, radius * 2)
            
            # Draw crosshair inside
            cross_size = 8
            painter.drawLine(px - cross_size, py, px + cross_size, py)
            painter.drawLine(px, py - cross_size, px, py + cross_size)
        
        # Draw gaze point if available (red crosshair)
        if self.gaze_point:
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            painter.setBrush(QColor(255, 0, 0, 100))
            
            # Scale gaze point to preview coordinates
            scale_x = scaled_pixmap.width() / w
            scale_y = scaled_pixmap.height() / h
            x = int(self.gaze_point[0] * scale_x)
            y = int(self.gaze_point[1] * scale_y)
            
            # Draw crosshair
            size = 20
            painter.drawLine(x - size, y, x + size, y)
            painter.drawLine(x, y - size, x, y + size)
            painter.drawEllipse(x - 5, y - 5, 10, 10)
        
        painter.end()
        
        # Display the pixmap
        self.setPixmap(scaled_pixmap)


class GazeGestureGUI(QMainWindow):
    """Main GUI window for GazeGestureKit."""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.mouse_control = False
        self.calibration_path = ".ggk_calibration.json"
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.session_start_time = datetime.now()
        self.verifying = False
        self.verify_timer = None
        self._confirm_until = 0.0
        self._confirm_pt = None
        self._screen_wh = (1280, 720)  # Default, will be updated from calibration
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("GazeGestureKit - Touch-Free Control")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Preview and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Preview
        self.preview = GazePreview()
        left_layout.addWidget(self.preview)
        
        # Controls
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.StyledPanel)
        controls_layout = QVBoxLayout(controls_frame)
        
        # Start/Stop buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        controls_layout.addLayout(button_layout)
        
        # Calibrate button
        self.calibrate_btn = QPushButton("🎯 Calibrate")
        self.calibrate_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        controls_layout.addWidget(self.calibrate_btn)
        
        # Verify button
        self.verify_btn = QPushButton("✓ Verify Detection")
        self.verify_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        controls_layout.addWidget(self.verify_btn)
        
        # Mouse control checkbox
        self.mouse_control_cb = QCheckBox("🖱️ Mouse Control")
        self.mouse_control_cb.setStyleSheet("font-weight: bold; padding: 5px;")
        controls_layout.addWidget(self.mouse_control_cb)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("background-color: #333; color: #4CAF50; padding: 5px; border-radius: 3px; font-weight: bold;")
        controls_layout.addWidget(self.status_label)
        
        # Camera selector
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Camera {i}" for i in range(4)])
        camera_layout.addWidget(self.camera_combo)
        controls_layout.addLayout(camera_layout)
        
        left_layout.addWidget(controls_frame)
        
        # Right panel - Event log
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        log_label = QLabel("📋 Event Log:")
        log_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        right_layout.addWidget(log_label)
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: 'Consolas', monospace; font-size: 10px;")
        right_layout.addWidget(self.event_log)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([800, 400])
        
    def setup_connections(self):
        """Setup signal connections."""
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.calibrate_btn.clicked.connect(self.calibrate)
        self.verify_btn.clicked.connect(self.toggle_verification)
        self.mouse_control_cb.toggled.connect(self.toggle_mouse_control)
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
    
    def start_tracking(self):
        """Start gaze tracking."""
        camera_index = self.camera_combo.currentIndex()
        
        # Stop existing worker
        if self.worker:
            self.worker.stop_capture()
            self.worker = None
        
        # Reset session start time for this tracking session
        self.session_start_time = datetime.now()
        # Clear event log for new session
        self.event_log.clear()
        
        # Create new worker
        self.worker = GazeWorker(camera_index, self.calibration_path)
        self.worker.frame_ready.connect(self.update_preview)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.verify_status.connect(self.update_verify_status)
        self.worker.confirm.connect(self._on_confirm)
        
        # Update screen dimensions from calibration if available
        self._update_screen_dimensions()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.calibrate_btn.setEnabled(False)
        self.status_label.setText("Status: 🟢 Tracking")
        self.status_label.setStyleSheet("background-color: #333; color: #4CAF50; padding: 5px; border-radius: 3px; font-weight: bold;")
        
        # Start worker
        self.worker.start_capture()
        
        self.log_event("Started gaze tracking")
    
    def stop_tracking(self):
        """Stop gaze tracking."""
        if self.worker:
            self.worker.stop_capture()
            self.worker = None
        
        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.calibrate_btn.setEnabled(True)
        self.status_label.setText("Status: ⚪ Stopped")
        self.status_label.setStyleSheet("background-color: #333; color: #9e9e9e; padding: 5px; border-radius: 3px; font-weight: bold;")
        
        self.log_event("Stopped gaze tracking")
        # Save log file after stopping
        self.save_log_file()
    
    def calibrate(self):
        """Run calibration process."""
        if self.worker and self.worker.isRunning():
            # Pause worker
            self.worker.stop_capture()
            self.worker = None
        
        try:
            # Run calibration
            camera_index = self.camera_combo.currentIndex()
            
            # Initialize GazeEstimator for feature extraction (without calibration)
            from ..eye.gaze import GazeEstimator
            estimator = GazeEstimator(calibration=None, screen=(1280, 720))
            
            # Initialize camera
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise Exception(f"Failed to open camera {camera_index}")
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            def get_feature():
                """Extract raw gaze feature from current frame."""
                ret, frame = cap.read()
                if not ret:
                    return None
                
                # Extract raw feature using GazeEstimator
                feat = estimator.get_raw_feature(frame)
                return feat
            
            # Get screen size (use primary screen for now)
            screen_w, screen_h = pyautogui.size()
            
            calibrator = Calibrator(points=5, save_path=self.calibration_path, mapping_type="tps")
            
            # Show calibration targets and collect samples
            self.log_event("Starting calibration - please look at the targets")
            
            # Collect calibration data with visual targets
            targets = calibrator._targets(screen_w, screen_h)
            samples_collected = 0
            
            for target_idx, (tx, ty) in enumerate(targets):
                self.log_event(f"Target {target_idx + 1}/{len(targets)}: ({tx}, {ty})")
                
                # Show target on screen
                try:
                    # Draw a target on screen using pyautogui
                    # Note: This is a simple approach - could be enhanced with a proper overlay
                    pass  # Targets will be shown by calibrator or we can enhance this
                except:
                    pass
                
                # Collect samples for this target
                for sample_idx in range(20):
                    raw = get_feature()
                    if raw is not None:
                        fx, fy = raw
                        calibrator.samples.append((fx, fy, tx, ty))
                        samples_collected += 1
                    time.sleep(0.02)
                    
                    # Process GUI events to keep it responsive
                    QApplication.processEvents()
            
            # Fit the calibration model
            mapping = calibrator.fit()
            params = {
                "screen": {"id": calibrator.screen_id, "w": screen_w, "h": screen_h}, 
                "mapping": mapping
            }
            
            # Save calibration
            cfg = {}
            if Path(self.calibration_path).exists():
                try:
                    cfg = json.loads(Path(self.calibration_path).read_text())
                except:
                    cfg = {}
            cfg[calibrator.screen_id] = params
            Path(self.calibration_path).write_text(json.dumps(cfg, indent=2))
            
            self.log_event(f"Calibration complete - collected {samples_collected} samples")
            cap.release()
            
            self.log_event("Calibration completed successfully")
            
            # Show success message
            QMessageBox.information(self, "Calibration", "Calibration completed successfully!")
            
        except Exception as e:
            self.log_event(f"Calibration failed: {str(e)}")
            QMessageBox.critical(self, "Calibration Error", f"Calibration failed: {str(e)}")
            if 'cap' in locals():
                cap.release()
    
    def toggle_mouse_control(self, enabled: bool):
        """Toggle mouse control on/off."""
        self.mouse_control = enabled
        status_text = "🟢 Enabled" if enabled else "⚪ Disabled"
        self.log_event(f"Mouse control {status_text.lower()}")
    
    def change_camera(self, index: int):
        """Change camera selection."""
        self.log_event(f"Switched to camera {index}")
    
    def _update_screen_dimensions(self):
        """Update screen dimensions from calibration or use default."""
        try:
            if Path(self.calibration_path).exists():
                cal_data = json.loads(Path(self.calibration_path).read_text())
                # Try to get screen dimensions from calibration
                if isinstance(cal_data, dict) and "primary" in cal_data:
                    screen_info = cal_data["primary"].get("screen", {})
                    w = screen_info.get("w", 1280)
                    h = screen_info.get("h", 720)
                    self._screen_wh = (w, h)
                else:
                    # Fallback to system screen size
                    screen_w, screen_h = pyautogui.size()
                    self._screen_wh = (screen_w, screen_h)
            else:
                # No calibration, use system screen size
                screen_w, screen_h = pyautogui.size()
                self._screen_wh = (screen_w, screen_h)
        except:
            # Fallback to default
            self._screen_wh = (1280, 720)
    
    def _on_confirm(self, frame_w: int, frame_h: int, gx: int, gy: int):
        """Handle pinch confirmation signal - map screen gaze to frame coordinates."""
        w, h = self._screen_wh
        px = int(gx / max(1, w) * frame_w)
        py = int(gy / max(1, h) * frame_h)
        self._confirm_pt = (px, py)
        self._confirm_until = time.time() + 0.5
        self.log_event(f"SELECT ({gx},{gy})")
    
    def toggle_verification(self):
        """Toggle verification mode to test gaze and gesture detection."""
        if not self.verifying:
            # Start verification
            if not self.worker or not self.worker.isRunning():
                QMessageBox.warning(self, "Verification", 
                    "Please start tracking first to verify detection.")
                return
            
            self.verifying = True
            self.verify_btn.setText("⏹ Stop Verification")
            self.verify_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
            self.status_label.setText("Status: 🔍 Verifying")
            self.status_label.setStyleSheet("background-color: #333; color: #FF9800; padding: 5px; border-radius: 3px; font-weight: bold;")
            self.log_event("=== VERIFICATION MODE STARTED ===")
            self.log_event("Looking for: Face, Pupils, Hands, Gestures...")
            
            # Start verification timer
            self.verify_timer = QTimer()
            self.verify_timer.timeout.connect(self.verify_status_report)
            self.verify_timer.start(2000)  # Report every 2 seconds
            
        else:
            # Stop verification
            self.verifying = False
            self.verify_btn.setText("✓ Verify Detection")
            self.verify_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
            if self.worker and self.worker.isRunning():
                self.status_label.setText("Status: 🟢 Tracking")
                self.status_label.setStyleSheet("background-color: #333; color: #4CAF50; padding: 5px; border-radius: 3px; font-weight: bold;")
            else:
                self.status_label.setText("Status: ⚪ Stopped")
                self.status_label.setStyleSheet("background-color: #333; color: #9e9e9e; padding: 5px; border-radius: 3px; font-weight: bold;")
            if self.verify_timer:
                self.verify_timer.stop()
                self.verify_timer = None
            self.log_event("=== VERIFICATION MODE STOPPED ===")
    
    def verify_detection(self, frame: np.ndarray, gaze: dict, event: Optional[dict]):
        """Verify gaze and gesture detection in real-time."""
        # This will be called during update_preview when verifying
        pass
    
    def update_verify_status(self, status: dict):
        """Update verification status display."""
        if not self.verifying:
            return
        
        # Update status display (could add a status widget)
        # For now, log periodic updates via timer
        pass
    
    def verify_status_report(self):
        """Periodic status report during verification."""
        if not self.worker or not self.verifying:
            return
        
        status = self.worker.last_verify_status
        if not status:
            return
        
        # Create status report
        report = []
        report.append("--- Detection Status ---")
        
        # Face & Gaze
        if status.get("face_detected"):
            report.append("✅ Face: Detected")
            if status.get("gaze_detected"):
                conf = status.get("gaze_confidence", 0.0)
                x, y = status.get("gaze_position", (0, 0))
                report.append(f"   ✅ Gaze: Position ({x}, {y}), Confidence: {conf:.2f}")
                if status.get("pupils_detected"):
                    report.append("   ✅ Pupils: Detected")
                else:
                    report.append("   ❌ Pupils: Not detected")
            else:
                report.append("   ❌ Gaze: Not detected")
        else:
            report.append("❌ Face: Not detected")
            report.append("   ❌ Gaze: N/A")
            report.append("   ❌ Pupils: N/A")
        
        # Hands & Gestures
        num_hands = status.get("num_hands", 0)
        if num_hands > 0:
            report.append(f"✅ Hands: {num_hands} detected")
            if status.get("gesture_detected"):
                gesture = status.get("gesture")
                conf = status.get("gesture_confidence", 0.0)
                handedness = status.get("handedness", "unknown")
                report.append(f"   ✅ Gesture: {gesture} ({handedness}), Confidence: {conf:.2f}")
            else:
                report.append("   ⚠️ Gesture: None detected (show hand clearly)")
        else:
            report.append("❌ Hands: Not detected (show hand to camera)")
            report.append("   ❌ Gesture: N/A")
        
        report.append("---")
        
        # Log the report
        self.log_event("\n".join(report))
    
    def update_preview(self, frame: np.ndarray, gaze: dict, event: Optional[dict]):
        """Update preview with new frame and gaze data."""
        # Pass confirm point and timestamp to preview
        confirm_pt = self._confirm_pt if time.time() < self._confirm_until else None
        confirm_until = self._confirm_until if time.time() < self._confirm_until else 0.0
        self.preview.update_frame(frame, gaze, confirm_pt, confirm_until)
        
        # Handle verification mode
        if self.verifying:
            self.verify_detection(frame, gaze, event)
        
        # Handle mouse control
        if self.mouse_control and event and gaze:
            self.handle_mouse_control(event, gaze)
        
        # Log events
        if event:
            self.log_event(f"Event: {event['type']}")
    
    def handle_mouse_control(self, event: dict, gaze: dict):
        """Handle mouse control based on events."""
        try:
            from ..demos.mouse import move_and_click
            
            x, y = gaze["screen_xy"]
            event_type = event.get("type")
            event_extra = event.get("extra", {})
            
            if event_type == "drag":
                # Handle drag events
                move_and_click(x, y, action=None, event_extra=event_extra)
            elif event_type in ["select", "click", "double_click"]:
                action = "click" if event_type in ["click", "double_click"] else None
                move_and_click(x, y, action=action)
                
                if event_type == "double_click":
                    pyautogui.click()  # Second click for double-click
            else:
                # Just move cursor for other events
                move_and_click(x, y, action=None)
                    
        except Exception as e:
            self.log_event(f"Mouse control error: {str(e)}")
    
    def handle_error(self, error_msg: str):
        """Handle worker errors."""
        self.log_event(f"Error: {error_msg}")
        
        # Show error dialog for critical errors
        if "camera" in error_msg.lower():
            QMessageBox.warning(self, "Camera Error", error_msg)
            self.stop_tracking()
    
    def log_event(self, message: str):
        """Log an event to the event log."""
        timestamp = time.strftime("%H:%M:%S")
        self.event_log.append(f"[{timestamp}] {message}")
        
        # Keep only last 200 lines
        lines = self.event_log.toPlainText().split('\n')
        if len(lines) > 200:
            self.event_log.setPlainText('\n'.join(lines[-200:]))
    
    def save_log_file(self):
        """Save current log to a file with timestamp."""
        try:
            log_content = self.event_log.toPlainText()
            if not log_content.strip():
                return  # Don't save empty logs
            
            # Create filename with timestamp
            timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
            log_filename = self.log_dir / f"ggk_session_{timestamp}.log"
            
            # Write log file
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write(f"GazeGestureKit Session Log\n")
                f.write(f"Session started: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n\n")
                f.write(log_content)
            
            print(f"Log saved to: {log_filename}")
        except Exception as e:
            print(f"Failed to save log file: {e}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.worker:
            self.worker.stop_capture()
        # Save log file before closing
        self.save_log_file()
        event.accept()


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    
    # Apply dark theme
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyside6())
    
    # Create and show main window
    window = GazeGestureGUI()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
