import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Cooldown to prevent spamming actions
last_action_time = 0
COOLDOWN = 2.0  # seconds

def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def count_fingers(hand_landmarks):
    """
    Returns a list of 5 booleans representing which fingers are up.
    Order: Thumb, Index, Middle, Ring, Pinky
    """
    fingers = []
    
    # Thumb: compare tip x to ip x (works for right hand mostly, simplified for both here)
    # A more robust way is to check if thumb tip is further from the wrist than the thumb MCP
    # Hand orientation matters, but for simplicity we rely on y and x heuristics.
    
    # For thumb, checking distance to index MCP vs thumb MCP can work
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(True) # Left hand mostly or flipped right
    elif hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(True)
    # Actually just a simple y/x heuristic is hard for thumb. Let's use x relative to pinky.
    # We will just evaluate the other 4 robustly based on y.
    
    # Let's write a better thumb state:
    # If the x-distance between thumb tip and pinky base is greater than thumb ip and pinky base
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    pinky_mcp = hand_landmarks.landmark[17]
    dist_tip = get_distance(thumb_tip, pinky_mcp)
    dist_ip = get_distance(thumb_ip, pinky_mcp)
    fingers.append(dist_tip > dist_ip)

    # 4 Fingers
    tip_ids = [8, 12, 16, 20]
    for id in tip_ids:
        # If tip is higher (y is smaller) than the PIP joint, finger is up
        if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 2].y:
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

def main():
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
        results = hands.process(img_rgb)
        
        gesture_name = "UNKNOWN"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                fingers = count_fingers(hand_landmarks)
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
