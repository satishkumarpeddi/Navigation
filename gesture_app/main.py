import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import time
import math

# Cooldown to prevent spamming actions
last_action_time = 0
COOLDOWN = 2.0  # seconds

def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def count_fingers(hand_landmarks):
    """
    Returns a list of 5 booleans representing which fingers are up.
    Order: Thumb, Index, Middle, Ring, Pinky
    hand_landmarks is a list of NormalizedLandmark objects with x, y, z properties.
    """
    fingers = []
    
    # Thumb: Calculate distance from tip (4) to pinky mcp (17) vs ip (3) to pinky mcp
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    pinky_mcp = hand_landmarks[17]
    dist_tip = get_distance(thumb_tip, pinky_mcp)
    dist_ip = get_distance(thumb_ip, pinky_mcp)
    fingers.append(dist_tip > dist_ip)

    # 4 Fingers
    tip_ids = [8, 12, 16, 20]
    for id in tip_ids:
        # If tip is higher (y is smaller) than the PIP joint, finger is up
        if hand_landmarks[id].y < hand_landmarks[id - 2].y:
            fingers.append(True)
        else:
            fingers.append(False)
            
    return fingers

def recognize_gesture(fingers):
    """
    fingers: [Thumb, Index, Middle, Ring, Pinky] booleans
    Returns gesture name
    """
    up_count = sum(fingers)
    
    if up_count == 5:
        return "OPEN_PALM"
    elif up_count == 0:
        return "FIST"
    elif fingers == [False, True, True, False, False]:
        return "PEACE" # Index and Middle up
    elif fingers == [True, True, False, False, False]:
        return "L_SIGN" # Thumb and Index up
    
    return "UNKNOWN"

def execute_action(gesture):
    global last_action_time
    if time.time() - last_action_time < COOLDOWN:
        return # Still in cooldown
        
    if gesture == "OPEN_PALM":
        print("Action: MAXIMIZE Window")
        pyautogui.hotkey('win', 'up')
        last_action_time = time.time()
        
    elif gesture == "FIST":
        print("Action: MINIMIZE Window")
        pyautogui.hotkey('win', 'down')
        last_action_time = time.time()
        
    elif gesture == "PEACE":
        print("Action: CLOSE Window")
        pyautogui.hotkey('alt', 'f4')
        last_action_time = time.time()
        
    elif gesture == "L_SIGN":
        print("Action: OPEN FILE EXPLORER")
        pyautogui.hotkey('win', 'e')
        last_action_time = time.time()

def draw_landmarks(img, hand_landmarks):
    h, w, _ = img.shape
    # Draw points
    for idx, lm in enumerate(hand_landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        
    # Connections (simplified list of pairs)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8), # Index
        (5, 9), (9, 10), (10, 11), (11, 12), # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
    ]
    for connection in connections:
        point1 = hand_landmarks[connection[0]]
        point2 = hand_landmarks[connection[1]]
        cx1, cy1 = int(point1.x * w), int(point1.y * h)
        cx2, cy2 = int(point2.x * w), int(point2.y * h)
        cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)

def main():
    # Setup MediaPipe Hand Landmarker Task
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Gesture Control.")
    print("Gestures:")
    print(" - Open Palm -> Maximize Window")
    print(" - Fist -> Minimize Window")
    print(" - Peace Sign (Index + Middle) -> Close Window")
    print(" - L-Sign (Thumb + Index) -> Open File Explorer")
    print("Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success:
            continue
            
        # Flip image horizontally for a mirrored view
        img = cv2.flip(img, 1)
        
        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        detection_result = detector.detect(mp_image)
        
        gesture_name = "UNKNOWN"
        
        if detection_result.hand_landmarks:
            for hand_landmarks_list in detection_result.hand_landmarks:
                draw_landmarks(img, hand_landmarks_list)
                
                fingers = count_fingers(hand_landmarks_list)
                gesture_name = recognize_gesture(fingers)
                
                if gesture_name != "UNKNOWN":
                    execute_action(gesture_name)
                    
        # Display Gesture on Screen
        cv2.putText(img, f"Gesture: {gesture_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        status = "READY" if (time.time() - last_action_time >= COOLDOWN) else "COOLDOWN"
        cv2.putText(img, f"Status: {status}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255) if status == "COOLDOWN" else (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Hand Gesture System Control", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
