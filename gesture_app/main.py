import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import time
import math
from collections import deque
import winsound

# Configuration
COOLDOWN = 1.5  
STABILIZATION_FRAMES = 8 
SMOOTHING = 5  
FRAME_REDUCTION = 100 # Margin for the virtual trackpad

# State tracking
last_action_time = 0
last_action_name = ""
gesture_history = deque(maxlen=STABILIZATION_FRAMES)

# Mouse tracking
pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()
ploc_x, ploc_y = 0, 0
target_history = deque(maxlen=4)

def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def count_fingers(hand_landmarks):
    fingers = []
    
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    pinky_mcp = hand_landmarks[17]
    fingers.append(get_distance(thumb_tip, pinky_mcp) > get_distance(thumb_ip, pinky_mcp))

    tip_ids = [8, 12, 16, 20]
    for id in tip_ids:
        if hand_landmarks[id].y < hand_landmarks[id - 2].y:
            fingers.append(True)
        else:
            fingers.append(False)
            
    return fingers

def recognize_gesture(fingers):
    up_count = sum(fingers)
    
    if up_count == 5:
        return "OPEN_PALM"
    elif up_count == 0:
        return "FIST"
    elif fingers == [False, True, True, False, False]:
        return "PEACE"
    elif fingers == [False, True, False, False, False]:
        return "INDEX_UP"
    elif fingers == [True, False, False, False, False]:
        return "THUMB_UP"
    
    return "UNKNOWN"

def execute_action(gesture):
    global last_action_time, last_action_name
    
    if time.time() - last_action_time < COOLDOWN:
        return False
        
    action_triggered = False
    
    if gesture == "OPEN_PALM":
        last_action_name = "MAXIMIZED"
        pyautogui.hotkey('win', 'up')
        action_triggered = True
    elif gesture == "FIST":
        last_action_name = "MINIMIZED"
        pyautogui.hotkey('win', 'down')
        action_triggered = True
    elif gesture == "PEACE":
        last_action_name = "CLOSED App"
        pyautogui.hotkey('alt', 'f4')
        action_triggered = True
    elif gesture == "THUMB_UP":
        last_action_name = "DOUBLE CLICKED"
        pyautogui.doubleClick(_pause=False)
        action_triggered = True

    if action_triggered:
        last_action_time = time.time()
        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
        print(f"Action Executed: {last_action_name}")
        
    return action_triggered

def draw_landmarks(img, hand_landmarks):
    h, w, _ = img.shape
    for idx, lm in enumerate(hand_landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (200, 200, 200), cv2.FILLED)
        
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), 
        (0, 5), (5, 6), (6, 7), (7, 8), 
        (5, 9), (9, 10), (10, 11), (11, 12), 
        (9, 13), (13, 14), (14, 15), (15, 16), 
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) 
    ]
    for connection in connections:
        point1 = hand_landmarks[connection[0]]
        point2 = hand_landmarks[connection[1]]
        cx1, cy1 = int(point1.x * w), int(point1.y * h)
        cx2, cy2 = int(point2.x * w), int(point2.y * h)
        cv2.line(img, (cx1, cy1), (cx2, cy2), (255, 180, 50), 2)

def draw_overlay_text(img, text, pos, font_scale=0.7, color=(255, 255, 255), thickness=2, bg_color=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x-5, y-text_size[1]-10), (x+text_size[0]+5, y+5), bg_color, cv2.FILLED)
    cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

def main():
    global gesture_history, last_action_time, ploc_x, ploc_y, target_history
    
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = 0

    while True:
        success, img = cap.read()
        if not success:
            continue
            
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # Draw virtual trackpad
        cv2.rectangle(img, (FRAME_REDUCTION, FRAME_REDUCTION), 
                     (w - FRAME_REDUCTION, h - FRAME_REDUCTION), (255, 0, 255), 2)
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        detection_result = detector.detect(mp_image)
        current_gesture = "UNKNOWN"
        stabilized_gesture = "UNKNOWN"
        time_since_last = time.time() - last_action_time

        if detection_result.hand_landmarks:
            for hand_landmarks_list in detection_result.hand_landmarks:
                draw_landmarks(img, hand_landmarks_list)
                
                fingers = count_fingers(hand_landmarks_list)
                current_gesture = recognize_gesture(fingers)
                
                if current_gesture == "INDEX_UP":
                    index_tip = hand_landmarks_list[8]
                    
                    # Convert to pixel coordinates
                    idx_px = index_tip.x * w
                    idx_py = index_tip.y * h
                    
                    # Constrain within the Active Trackpad boundary
                    box_w = w - 2 * FRAME_REDUCTION
                    box_h = h - 2 * FRAME_REDUCTION
                    mx = max(FRAME_REDUCTION, min(idx_px, w - FRAME_REDUCTION))
                    my = max(FRAME_REDUCTION, min(idx_py, h - FRAME_REDUCTION))
                    
                    # Map to full screen coordinates
                    target_x = ((mx - FRAME_REDUCTION) / box_w) * SCREEN_W
                    target_y = ((my - FRAME_REDUCTION) / box_h) * SCREEN_H
                    
                    # Smooth raw targets slightly with moving average
                    target_history.append((target_x, target_y))
                    avg_target_x = sum([t[0] for t in target_history]) / len(target_history)
                    avg_target_y = sum([t[1] for t in target_history]) / len(target_history)
                    
                    # Deadzone + Dynamic Smoothing Optimization
                    dx = avg_target_x - ploc_x
                    dy = avg_target_y - ploc_y
                    dist = math.hypot(dx, dy)
                    
                    # Smaller deadzone for smoother micro-movements
                    if dist > 1.5:
                        # Dynamic smoothing: high for slow/small movements, low for fast/large movements
                        dynamic_smoothing = SMOOTHING
                        if dist < 30:
                            dynamic_smoothing = SMOOTHING * 1.5  # Filter out fine jitter
                        elif dist > 150:
                            dynamic_smoothing = SMOOTHING * 0.4  # Snap fast on large movements
                            
                        cloc_x = ploc_x + dx / dynamic_smoothing
                        cloc_y = ploc_y + dy / dynamic_smoothing
                        
                        try:
                            pyautogui.moveTo(int(cloc_x), int(cloc_y), _pause=False)
                            ploc_x, ploc_y = cloc_x, cloc_y
                        except pyautogui.FailSafeException:
                            pass
                else:
                    target_history.clear()

                gesture_history.append(current_gesture)
        else:
            gesture_history.clear()
            target_history.clear()

        # Execute Action 
        if len(gesture_history) == STABILIZATION_FRAMES and len(set(gesture_history)) == 1:
            stabilized_gesture = gesture_history[0]
            if stabilized_gesture not in ["UNKNOWN", "INDEX_UP"]:
                execute_action(stabilized_gesture)

        # -- UX Rendering --
        status_text = "READY" if time_since_last >= COOLDOWN else f"COOLDOWN ({COOLDOWN - time_since_last:.1f}s)"
        status_color = (0, 255, 0) if time_since_last >= COOLDOWN else (0, 0, 255)
        draw_overlay_text(img, f"Status: {status_text}", (15, 35), color=status_color)
        draw_overlay_text(img, f"Gesture: {current_gesture}", (15, 75))
        draw_overlay_text(img, f"FPS: {int(fps)}", (15, 115), font_scale=0.5, thickness=1)

        if time_since_last < 2.0:
            alpha = max(0.0, 1.0 - (time_since_last/2.0))
            if alpha > 0:
                text = f"ACTION: {last_action_name}"
                font = cv2.FONT_HERSHEY_DUPLEX
                scale = 1.2
                thick = 3
                t_size, _ = cv2.getTextSize(text, font, scale, thick)
                tx, ty = (w - t_size[0]) // 2, (h // 2)
                
                overlay = img.copy()
                cv2.rectangle(overlay, (tx-20, ty-t_size[1]-20), (tx+t_size[0]+20, ty+20), (50,200,50), cv2.FILLED)
                cv2.putText(overlay, text, (tx, ty), font, scale, (255,255,255), thick, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        inst = "Palm=Max|Fist=Min|Peace=Close|Index=Move|Thumb=Click"
        draw_overlay_text(img, inst, (15, h - 20), font_scale=0.5, thickness=1)

        cv2.imshow("Hand Gesture Control - Pro UX", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
