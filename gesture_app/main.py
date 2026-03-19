import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import time
import math
from collections import deque
import winsound

# ── Configuration ──────────────────────────────────────────
COOLDOWN            = 1.5   # seconds between right-hand actions
STABILIZATION_FRAMES = 8    # frames gesture must be stable before triggering
SMOOTHING           = 5     # base cursor smoothing factor
FRAME_REDUCTION     = 100   # virtual trackpad margin (pixels)

# ── Gesture Definitions ─────────────────────────────────────
# fingers list order: [Thumb, Index, Middle, Ring, Pinky]
GESTURES = {
    "OPEN_PALM":    [True,  True,  True,  True,  True ],
    "FIST":         [False, False, False, False, False],
    "PEACE":        [False, True,  True,  False, False],   # Index + Middle → cursor move
    "INDEX_UP":     [False, True,  False, False, False],
    "THUMB_UP":     [True,  False, False, False, False],   # Right: click/select app
    "FOUR_FINGERS": [False, True,  True,  True,  True ],   # Right: drag/reposition window
}

# ── Right-Hand Action Mapping ───────────────────────────────
# Maps gesture name → (display_name, action_fn)
def _do_click():
    pyautogui.click(_pause=False)

def _do_drag_start():
    pyautogui.mouseDown(button='left')

def _do_drag_end():
    pyautogui.mouseUp(button='left')

RIGHT_HAND_ACTIONS = {
    "THUMB_UP": ("CLICKED", _do_click),
}

# ── State Tracking ──────────────────────────────────────────
last_action_time  = 0
last_action_name  = ""
gesture_history   = deque(maxlen=STABILIZATION_FRAMES)

# ── Mouse / Drag State ──────────────────────────────────────
pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()
ploc_x, ploc_y     = 0, 0
target_history      = deque(maxlen=4)
is_dragging         = False   # right-hand window drag
cursor_locked       = False   # both-palm lock mode

# ── Helpers ─────────────────────────────────────────────────
def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def count_fingers(hand_landmarks):
    """Returns [Thumb, Index, Middle, Ring, Pinky] booleans."""
    fingers = []
    thumb_tip   = hand_landmarks[4]
    thumb_ip    = hand_landmarks[3]
    pinky_mcp   = hand_landmarks[17]
    fingers.append(get_distance(thumb_tip, pinky_mcp) > get_distance(thumb_ip, pinky_mcp))
    for tip_id in [8, 12, 16, 20]:
        fingers.append(hand_landmarks[tip_id].y < hand_landmarks[tip_id - 2].y)
    return fingers

def recognize_gesture(fingers):
    """Match finger state to a gesture name via GESTURES dict."""
    for name, pattern in GESTURES.items():
        if fingers == pattern:
            return name
    return "UNKNOWN"

def move_cursor(track_point, w, h):
    """Smooth cursor movement towards track_point with dynamic smoothing."""
    global ploc_x, ploc_y
    idx_px = track_point.x * w
    idx_py = track_point.y * h
    box_w = w - 2 * FRAME_REDUCTION
    box_h = h - 2 * FRAME_REDUCTION
    mx = max(FRAME_REDUCTION, min(idx_px, w - FRAME_REDUCTION))
    my = max(FRAME_REDUCTION, min(idx_py, h - FRAME_REDUCTION))
    target_x = ((mx - FRAME_REDUCTION) / box_w) * SCREEN_W
    target_y = ((my - FRAME_REDUCTION) / box_h) * SCREEN_H
    target_history.append((target_x, target_y))
    avg_x = sum(t[0] for t in target_history) / len(target_history)
    avg_y = sum(t[1] for t in target_history) / len(target_history)
    dx, dy = avg_x - ploc_x, avg_y - ploc_y
    dist = math.hypot(dx, dy)
    if dist > 1.5:
        dyn = SMOOTHING
        if dist < 30:
            dyn = SMOOTHING * 1.5
        elif dist > 150:
            dyn = SMOOTHING * 0.4
        cloc_x = ploc_x + dx / dyn
        cloc_y = ploc_y + dy / dyn
        try:
            pyautogui.moveTo(int(cloc_x), int(cloc_y), _pause=False)
            ploc_x, ploc_y = cloc_x, cloc_y
        except pyautogui.FailSafeException:
            pass

def execute_right_action(gesture):
    """Fire a stabilized right-hand action with cooldown."""
    global last_action_time, last_action_name
    if time.time() - last_action_time < COOLDOWN:
        return
    if gesture in RIGHT_HAND_ACTIONS:
        name, fn = RIGHT_HAND_ACTIONS[gesture]
        fn()
        last_action_name = name
        last_action_time = time.time()
        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
        print(f"Action: {name}")

def draw_landmarks(img, hand_landmarks, is_left=True):
    h, w, _ = img.shape
    node_color = (255, 255, 255)
    line_color = (255, 180, 50) if is_left else (50, 255, 50)
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, node_color, cv2.FILLED)
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(0,17),(17,18),(18,19),(19,20)
    ]
    for a, b in connections:
        p1, p2 = hand_landmarks[a], hand_landmarks[b]
        cv2.line(img,
                 (int(p1.x*w), int(p1.y*h)),
                 (int(p2.x*w), int(p2.y*h)),
                 line_color, 2)

def draw_overlay_text(img, text, pos, font_scale=0.7, color=(255,255,255), thickness=2, bg_color=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x-5, y-text_size[1]-10), (x+text_size[0]+5, y+5), bg_color, cv2.FILLED)
    cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

# ── Main Loop ───────────────────────────────────────────────
def main():
    global gesture_history, last_action_time, ploc_x, ploc_y
    global target_history, is_dragging, cursor_locked

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector     = vision.HandLandmarker.create_from_options(options)

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

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result    = detector.detect(mp_image)

        time_since_last = time.time() - last_action_time

        left_gesture  = "UNKNOWN"
        right_gesture = "UNKNOWN"
        left_landmarks  = None
        right_landmarks = None

        # ── Classify each hand ──────────────────────────────
        if result.hand_landmarks:
            for idx, lm_list in enumerate(result.hand_landmarks):
                label = result.handedness[idx][0].category_name
                # Mirrored feed: MediaPipe "Right" = physical Left
                is_physical_left = (label == "Right")
                draw_landmarks(img, lm_list, is_left=is_physical_left)
                fingers = count_fingers(lm_list)
                gesture = recognize_gesture(fingers)
                if is_physical_left:
                    left_gesture   = gesture
                    left_landmarks = lm_list
                else:
                    right_gesture   = gesture
                    right_landmarks = lm_list

        # ── Dual-hand signals (checked before individual hand logic) ──
        both_open_palm = (left_gesture == "OPEN_PALM" and right_gesture == "OPEN_PALM")
        both_fist      = (left_gesture == "FIST"      and right_gesture == "FIST")

        if both_open_palm and not cursor_locked:
            # Enter cursor lock mode
            cursor_locked = True
            if is_dragging:
                pyautogui.mouseUp(button='left')
                is_dragging = False
            gesture_history.clear()
            last_action_name = "CURSOR LOCKED"
            last_action_time = time.time()
            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)

        elif both_fist and cursor_locked:
            # Exit cursor lock mode
            cursor_locked = False
            last_action_name = "CURSOR UNLOCKED"
            last_action_time = time.time()
            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)

        # ── Left Hand: cursor movement (PEACE = Index + Middle) ──
        if left_landmarks and left_gesture == "PEACE":
            # Use midpoint of index tip (8) and middle tip (12) as tracking point
            lm = left_landmarks
            mid_x = (lm[8].x + lm[12].x) / 2
            mid_y = (lm[8].y + lm[12].y) / 2

            class _MP:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

            move_cursor(_MP(mid_x, mid_y), w, h)
        else:
            target_history.clear()

        # ── Right Hand: actions (disabled in cursor lock mode) ────
        if not cursor_locked and right_landmarks:
            if right_gesture == "FOUR_FINGERS":
                # Drag window — hold left mouse btn and follow hand palm (wrist = 0)
                if not is_dragging:
                    pyautogui.mouseDown(button='left')
                    is_dragging = True
                move_cursor(right_landmarks[9], w, h)   # use palm center (MCP 9)

            else:
                if is_dragging:
                    pyautogui.mouseUp(button='left')
                    is_dragging = False

                # Stabilised action
                gesture_history.append(right_gesture)
                if (len(gesture_history) == STABILIZATION_FRAMES
                        and len(set(gesture_history)) == 1):
                    stabilized = gesture_history[0]
                    if stabilized not in ("UNKNOWN", "OPEN_PALM", "FIST", "FOUR_FINGERS"):
                        execute_right_action(stabilized)
        else:
            if is_dragging:
                pyautogui.mouseUp(button='left')
                is_dragging = False
            if not cursor_locked:
                gesture_history.clear()

        # ── HUD ────────────────────────────────────────────────────
        if cursor_locked:
            lock_label = "CURSOR LOCKED"
            draw_overlay_text(img, lock_label, (15, 35), color=(0, 255, 255))
        else:
            status_text  = "READY" if time_since_last >= COOLDOWN else f"COOLDOWN ({COOLDOWN - time_since_last:.1f}s)"
            status_color = (0, 255, 0) if time_since_last >= COOLDOWN else (0, 0, 255)
            draw_overlay_text(img, f"Status: {status_text}", (15, 35), color=status_color)

        draw_overlay_text(img, f"L: {left_gesture}",  (15, 75))
        draw_overlay_text(img, f"R: {right_gesture}", (15, 110))
        draw_overlay_text(img, f"FPS: {int(fps)}",    (15, 145), font_scale=0.5, thickness=1)

        # Action flash
        if time_since_last < 2.0:
            alpha = max(0.0, 1.0 - (time_since_last / 2.0))
            if alpha > 0:
                text  = f"ACTION: {last_action_name}"
                font  = cv2.FONT_HERSHEY_DUPLEX
                scale = 1.2
                thick = 3
                t_size, _ = cv2.getTextSize(text, font, scale, thick)
                tx, ty = (w - t_size[0]) // 2, h // 2
                overlay = img.copy()
                cv2.rectangle(overlay, (tx-20, ty-t_size[1]-20), (tx+t_size[0]+20, ty+20), (50,200,50), cv2.FILLED)
                cv2.putText(overlay, text, (tx, ty), font, scale, (255,255,255), thick, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        inst = "L-Peace=Move | R-Thumb=Click | R-4Fingers=DragWindow | BothPalm=Lock | BothFist=Unlock"
        draw_overlay_text(img, inst, (15, h - 20), font_scale=0.4, thickness=1)

        cv2.imshow("Hand Gesture Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if is_dragging:
        pyautogui.mouseUp(button='left')
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
