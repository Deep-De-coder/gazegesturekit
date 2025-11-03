from __future__ import annotations
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

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
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("No camera feed")
        self.gaze_point = None
        self.frame = None
    
    def update_frame(self, frame: np.ndarray, gaze: dict):
        """Update the preview with new frame and gaze point."""
        self.frame = frame.copy()
        if gaze and "screen_xy" in gaze:
            self.gaze_point = gaze["screen_xy"]
        else:
            self.gaze_point = None
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
        
        # Draw gaze point if available
        if self.gaze_point:
            painter = QPainter(scaled_pixmap)
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
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("GazeGestureKit GUI")
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
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        controls_layout.addLayout(button_layout)
        
        # Calibrate button
        self.calibrate_btn = QPushButton("Calibrate")
        controls_layout.addWidget(self.calibrate_btn)
        
        # Mouse control checkbox
        self.mouse_control_cb = QCheckBox("Mouse Control")
        controls_layout.addWidget(self.mouse_control_cb)
        
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
        
        right_layout.addWidget(QLabel("Event Log:"))
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
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
        self.mouse_control_cb.toggled.connect(self.toggle_mouse_control)
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
    
    def start_tracking(self):
        """Start gaze tracking."""
        camera_index = self.camera_combo.currentIndex()
        
        # Stop existing worker
        if self.worker:
            self.worker.stop_capture()
            self.worker = None
        
        # Create new worker
        self.worker = GazeWorker(camera_index, self.calibration_path)
        self.worker.frame_ready.connect(self.update_preview)
        self.worker.error_occurred.connect(self.handle_error)
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.calibrate_btn.setEnabled(False)
        
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
        
        self.log_event("Stopped gaze tracking")
    
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
        self.log_event(f"Mouse control {'enabled' if enabled else 'disabled'}")
    
    def change_camera(self, index: int):
        """Change camera selection."""
        self.log_event(f"Switched to camera {index}")
    
    def update_preview(self, frame: np.ndarray, gaze: dict, event: Optional[dict]):
        """Update preview with new frame and gaze data."""
        self.preview.update_frame(frame, gaze)
        
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
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.worker:
            self.worker.stop_capture()
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
