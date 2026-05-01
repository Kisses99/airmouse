import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Optimize PyAutoGUI
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False # Removed as requested

class AirMouse:
    def __init__(self):
        # MediaPipe Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Screen Setup
        self.screen_w, self.screen_h = pyautogui.size()
        
        # ACTIVE REGION (for easier movement)
        self.active_rect = (0.2, 0.8) 
        
        # Smoothing Variables
        self.smooth_factor = 2 
        self.prev_x, self.prev_y = self.screen_w // 2, self.screen_h // 2
        self.curr_x, self.curr_y = self.prev_x, self.prev_y
        
        # Hover Click Variables
        self.hover_start_time = None
        self.hover_duration = 0.8  # seconds
        self.hover_threshold = 40  # pixels
        self.last_stable_pos = (0, 0)
        
        # Drag State
        self.is_dragging_left = False
        self.is_dragging_right = False
        
        # Scroll State
        self.last_scroll_y = None
        
    def get_distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def is_fist(self, hand_landmarks):
        # Check if all 4 fingers are below their MCP joints
        tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        mcps = [
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP
        ]
        
        closed_count = 0
        for tip, mcp in zip(tips, mcps):
            if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y:
                closed_count += 1
        return closed_count >= 3 # Most fingers closed

    def update(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        h, w, _ = frame.shape
        gesture_data = {
            "click": False,
            "hover_progress": 0.0,
            "dragging_left": self.is_dragging_left,
            "dragging_right": self.is_dragging_right,
            "scrolling": False
        }
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Landmarks
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            
            # 1. GESTURE DETECTION
            index_up = index_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y
            middle_up = middle_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
            ring_up = ring_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP].y
            
            # A. SCROLL GESTURE (Index + Middle up, Ring down)
            if index_up and middle_up and not ring_up and not self.is_dragging_left and not self.is_dragging_right:
                gesture_data["scrolling"] = True
                curr_scroll_y = (index_tip.y + middle_tip.y) / 2
                if self.last_scroll_y is not None:
                    dy = (self.last_scroll_y - curr_scroll_y) * 1000
                    if abs(dy) > 10:
                        pyautogui.scroll(int(dy))
                self.last_scroll_y = curr_scroll_y
            else:
                self.last_scroll_y = None

                # B. MOVE CURSOR
                target_x = np.interp(index_tip.x, [self.active_rect[0], self.active_rect[1]], [0, self.screen_w])
                target_y = np.interp(index_tip.y, [self.active_rect[0], self.active_rect[1]], [0, self.screen_h])
                
                target_x = np.clip(target_x, 0, self.screen_w)
                target_y = np.clip(target_y, 0, self.screen_h)

                self.curr_x = self.prev_x + (target_x - self.prev_x) / self.smooth_factor
                self.curr_y = self.prev_y + (target_y - self.prev_y) / self.smooth_factor
                
                pyautogui.moveTo(int(self.curr_x), int(self.curr_y))
                self.prev_x, self.prev_y = self.curr_x, self.curr_y
                
                # C. LEFT DRAG (Pinch Index + Thumb)
                pinch_dist = self.get_distance(index_tip, thumb_tip)
                if pinch_dist < 0.04 and not self.is_dragging_right:
                    if not self.is_dragging_left:
                        pyautogui.mouseDown(button='left')
                        self.is_dragging_left = True
                else:
                    if self.is_dragging_left:
                        pyautogui.mouseUp(button='left')
                        self.is_dragging_left = False

                # D. RIGHT DRAG (Fist)
                if self.is_fist(hand_landmarks) and not self.is_dragging_left:
                    if not self.is_dragging_right:
                        pyautogui.mouseDown(button='right')
                        self.is_dragging_right = True
                else:
                    if self.is_dragging_right:
                        pyautogui.mouseUp(button='right')
                        self.is_dragging_right = False
                
                gesture_data["dragging_left"] = self.is_dragging_left
                gesture_data["dragging_right"] = self.is_dragging_right

                # E. HOVER LEFT-CLICK (Only if not dragging/scrolling)
                if not self.is_dragging_left and not self.is_dragging_right:
                    dist_from_last = np.sqrt((self.curr_x - self.last_stable_pos[0])**2 + (self.curr_y - self.last_stable_pos[1])**2)
                    if dist_from_last < self.hover_threshold:
                        if self.hover_start_time is None:
                            self.hover_start_time = time.time()
                        
                        elapsed = time.time() - self.hover_start_time
                        gesture_data["hover_progress"] = min(elapsed / self.hover_duration, 1.0)
                        
                        if elapsed > self.hover_duration:
                            pyautogui.click()
                            gesture_data["click"] = True
                            self.hover_start_time = time.time() + 1.0 
                    else:
                        self.hover_start_time = time.time()
                        self.last_stable_pos = (self.curr_x, self.curr_y)

                    # Draw Hover Progress
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    cv2.circle(frame, (ix, iy), 20, (255, 255, 255), 1)
                    if gesture_data["hover_progress"] > 0:
                        angle = int(gesture_data["hover_progress"] * 360)
                        cv2.ellipse(frame, (ix, iy), (20, 20), -90, 0, angle, (0, 255, 255), 3)

            # Visual Feedback HUD
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            if self.is_dragging_left:
                cv2.circle(frame, (ix, iy), 30, (0, 255, 0), 3)
                cv2.putText(frame, "LEFT DRAG", (ix + 20, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif self.is_dragging_right:
                cv2.circle(frame, (ix, iy), 30, (0, 0, 255), 3)
                cv2.putText(frame, "RIGHT DRAG", (ix + 20, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif gesture_data["scrolling"]:
                cv2.circle(frame, (ix, iy), 30, (255, 0, 255), 3)
                cv2.putText(frame, "SCROLLING", (ix + 20, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        return frame, gesture_data

        return frame, gesture_data

from pygrabber.dshow_graph import FilterGraph

def get_camera_names():
    graph = FilterGraph()
    return graph.get_input_devices()

def main():
    print("\nLumina Air Mouse - Setup")
    print("-------------------------")
    
    # 1. Camera Discovery
    cameras = get_camera_names()
    if not cameras:
        print("No cameras found! Please ensure your webcam is connected.")
        return
    
    print("Available Cameras:")
    for i, name in enumerate(cameras):
        print(f"[{i}] {name}")
    
    # 2. Camera Selection
    if len(cameras) == 1:
        choice = 0
        print(f"\nOnly one camera found. Using: {cameras[0]}")
    else:
        try:
            val = input(f"\nSelect camera index [0-{len(cameras)-1}] (default 0): ").strip()
            choice = int(val) if val else 0
            if choice < 0 or choice >= len(cameras):
                print("Invalid index. Using default 0.")
                choice = 0
        except (EOFError, RuntimeError):
            # This happens when running as a windowed application (no console)
            choice = 0
            print(f"No console input available. Defaulting to: {cameras[choice]}")
        except ValueError:
            print("Invalid input. Using default 0.")
            choice = 0
            
    print(f"Connecting to: {cameras[choice]}...")
    
    # 3. Camera Initialization
    # Try DirectShow first as it's most compatible with Windows names
    cap = cv2.VideoCapture(choice, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback to default
        cap = cv2.VideoCapture(choice)
        
    if not cap.isOpened():
        print(f"Error: Could not open camera {cameras[choice]}.")
        return

    # Standard settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    air_mouse = AirMouse()
    
    print("\nAir Mouse Active.")
    print("-----------------")
    print("Move Index Finger to move cursor | Hover to Click")
    print("Hand to Corner for Failsafe | ESC or Ctrl+C to Stop\n")
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to capture frame.")
                break
            
            frame = cv2.flip(frame, 1)
            frame, gestures = air_mouse.update(frame)
            
            if gestures is None: # Failsafe
                break
            
            # HUD
            cv2.putText(frame, f"Cam: {cameras[choice]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Lumina Air Mouse - Tracking Feed', frame)
            
            if cv2.waitKey(1) & 0xFF == 27: break
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Cleaning up resources.")
        if cap:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
