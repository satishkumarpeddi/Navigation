import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import time
import math
import numpy as np
from collections import deque
import winsound

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════
SMOOTHING           = 0.18   # EMA alpha: lower = smoother but laggier
FRAME_REDUCTION     = 100    # virtual trackpad margin (px)
PINCH_THRESHOLD     = 0.07   # normalised thumb-index distance
MID_PINCH_THRESHOLD = 0.07   # normalised thumb-middle distance
DOUBLE_CLICK_GAP    = 0.35   # seconds between two pinches = double-click
SCROLL_SENSITIVITY  = 30     # px of hand movement per scroll tick
CONFIDENCE_FRAMES   = 6      # frames a gesture must hold before accepted
FIST_STABILIZE      = 6      # frames fist must be held to pause
PROCESS_EVERY_N     = 2      # only run ML every N frames (FPS boost)

# Iron Man palette
IM_CYAN   = (255, 220, 0)    # BGR
IM_GOLD   = (0, 200, 255)
IM_RED    = (30,  30, 230)
IM_WHITE  = (230, 230, 230)
IM_DARK   = (10,  10,  10)
IM_GREEN  = (80, 255, 80)

# ══════════════════════════════════════════════════════════════
#  GESTURE DEFINITIONS   [Thumb, Index, Middle, Ring, Pinky]
# ══════════════════════════════════════════════════════════════
GESTURES = {
    "OPEN_PALM": [True,  True,  True,  True,  True ],
    "FIST":      [False, False, False, False, False],
    "PEACE":     [False, True,  True,  False, False],
    "INDEX_UP":  [False, True,  False, False, False],
    "THUMB_UP":  [True,  False, False, False, False],
}

# ══════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ══════════════════════════════════════════════════════════════
pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()

# Smooth cursor — EMA
ema_x, ema_y     = 0.0, 0.0
cursor_initialised = False

# Gesture confidence filter
gesture_vote_buf = deque(maxlen=CONFIDENCE_FRAMES)
confirmed_gesture = "NONE"

# Drag / click state
is_dragging      = False
pinch_active     = False
last_pinch_time  = 0.0

# Pause
paused           = False
fist_frame_count = 0

# Scroll
scroll_ref_y     = None

# Display
last_action_name = ""
last_action_time = 0.0

# FPS optimisation
frame_counter    = 0
cached_result    = None

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def count_fingers(lm):
    fingers = [get_distance(lm[4], lm[17]) > get_distance(lm[3], lm[17])]
    for tip in [8, 12, 16, 20]:
        fingers.append(lm[tip].y < lm[tip - 2].y)
    return fingers

def recognize_gesture(fingers):
    for name, pattern in GESTURES.items():
        if fingers == pattern:
            return name
    return "UNKNOWN"

def confident_gesture(raw):
    """Return a gesture only once it has been stable for CONFIDENCE_FRAMES."""
    global confirmed_gesture
    gesture_vote_buf.append(raw)
    if len(gesture_vote_buf) == CONFIDENCE_FRAMES and len(set(gesture_vote_buf)) == 1:
        confirmed_gesture = gesture_vote_buf[0]
    return confirmed_gesture

def is_thumb_index_pinch(lm):
    return get_distance(lm[4], lm[8]) < PINCH_THRESHOLD

def is_thumb_middle_pinch(lm):
    return get_distance(lm[4], lm[12]) < MID_PINCH_THRESHOLD

class _Pt:
    def __init__(self, x, y): self.x = x; self.y = y

def move_cursor_ema(px_norm, py_norm, w, h):
    """EMA smoothed cursor movement."""
    global ema_x, ema_y, cursor_initialised
    bw = w - 2 * FRAME_REDUCTION
    bh = h - 2 * FRAME_REDUCTION
    px = px_norm * w
    py = py_norm * h
    mx = max(FRAME_REDUCTION, min(px, w - FRAME_REDUCTION))
    my = max(FRAME_REDUCTION, min(py, h - FRAME_REDUCTION))
    tx = ((mx - FRAME_REDUCTION) / bw) * SCREEN_W
    ty = ((my - FRAME_REDUCTION) / bh) * SCREEN_H

    if not cursor_initialised:
        ema_x, ema_y = tx, ty
        cursor_initialised = True
    else:
        # Adaptive alpha: move faster for big deltas
        dist = math.hypot(tx - ema_x, ty - ema_y)
        alpha = min(1.0, SMOOTHING + dist / 500)
        ema_x = ema_x + alpha * (tx - ema_x)
        ema_y = ema_y + alpha * (ty - ema_y)

    try:
        pyautogui.moveTo(int(ema_x), int(ema_y), _pause=False)
    except pyautogui.FailSafeException:
        pass

def flash_action(name):
    global last_action_name, last_action_time
    last_action_name = name
    last_action_time = time.time()
    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
    print(f"[Action] {name}")

# ── Drawing helpers ────────────────────────────────────────────
def draw_landmarks(img, lm, active=True):
    h, w, _ = img.shape
    node_col = IM_CYAN if active else (80, 80, 80)
    line_col = IM_GOLD if active else (50, 50, 50)
    connections = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
        (13,17),(0,17),(17,18),(18,19),(19,20)
    ]
    for a, b in connections:
        cv2.line(img,
                 (int(lm[a].x*w), int(lm[a].y*h)),
                 (int(lm[b].x*w), int(lm[b].y*h)),
                 line_col, 2, cv2.LINE_AA)
    for p in lm:
        cv2.circle(img, (int(p.x*w), int(p.y*h)), 5, node_col, cv2.FILLED)

def draw_scanlines(img, gap=6, alpha=0.06):
    """Subtle Iron Man scanline effect."""
    overlay = np.zeros_like(img, dtype=np.uint8)
    for y in range(0, img.shape[0], gap):
        cv2.line(overlay, (0, y), (img.shape[1], y), (0, 255, 255), 1)
    cv2.addWeighted(overlay, alpha, img, 1.0, 0, img)

def draw_corner_brackets(img, color=IM_CYAN, size=22, thick=2):
    h, w, _ = img.shape
    pts = [(0,0),(w,0),(0,h),(w,h)]
    dirs= [(1,1),(-1,1),(1,-1),(-1,-1)]
    for (cx,cy),(dx,dy) in zip(pts,dirs):
        cv2.line(img,(cx,cy),(cx+dx*size,cy),color,thick,cv2.LINE_AA)
        cv2.line(img,(cx,cy),(cx,cy+dy*size),color,thick,cv2.LINE_AA)

def draw_hud_panel(img, x, y, width, lines, title=None):
    """Glassmorphism-style dark panel."""
    line_h = 26
    panel_h = line_h * len(lines) + 20 + (22 if title else 0)
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+width, y+panel_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x, y), (x+width, y+panel_h), IM_CYAN, 1, cv2.LINE_AA)
    # top accent bar
    cv2.rectangle(img, (x, y), (x+width, y+3), IM_GOLD, cv2.FILLED)
    ty = y + 16
    if title:
        cv2.putText(img, title, (x+8, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    IM_GOLD, 1, cv2.LINE_AA)
        ty += 22
    for txt, col in lines:
        cv2.putText(img, txt, (x+8, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    col, 1, cv2.LINE_AA)
        ty += line_h

def draw_gesture_confidence_bar(img, x, y, width, vote_buf):
    """Shows how many frames the current gesture has been stable."""
    filled = len(vote_buf)
    total  = CONFIDENCE_FRAMES
    bar_h  = 6
    # Background
    cv2.rectangle(img, (x, y), (x+width, y+bar_h), (40,40,40), cv2.FILLED)
    # Fill
    fill_w = int((filled / total) * width)
    color  = IM_GREEN if filled == total else IM_GOLD
    cv2.rectangle(img, (x, y), (x+fill_w, y+bar_h), color, cv2.FILLED)
    cv2.rectangle(img, (x, y), (x+width, y+bar_h), IM_CYAN, 1)

def draw_action_flash(img, w, h, now):
    t = now - last_action_time
    if t >= 2.0:
        return
    alpha = max(0.0, 1.0 - t / 2.0)
    text  = last_action_name.upper()
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 1.2, 2
    sz, _ = cv2.getTextSize(text, font, scale, thick)
    tx, ty = (w - sz[0]) // 2, h // 2
    ov = img.copy()
    cv2.rectangle(ov, (tx-20, ty-sz[1]-16), (tx+sz[0]+20, ty+16), (0,30,0), cv2.FILLED)
    cv2.rectangle(ov, (tx-20, ty-sz[1]-16), (tx+sz[0]+20, ty+16), IM_GREEN, 1)
    cv2.putText(ov, text, (tx, ty), font, scale, IM_GREEN, thick, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    global cursor_initialised, ema_x, ema_y
    global is_dragging, paused, fist_frame_count
    global pinch_active, last_pinch_time, scroll_ref_y
    global frame_counter, cached_result
    global confirmed_gesture

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector     = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    # FPS optimisation: prefer 60 fps from camera
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = 0
    fps_display = 0

    while True:
        ok, img = cap.read()
        if not ok:
            continue

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        now = time.time()

        # ── FPS optimisation: skip ML every PROCESS_EVERY_N frames ──
        frame_counter += 1
        if frame_counter % PROCESS_EVERY_N == 0:
            img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img      = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            cached_result = detector.detect(mp_img)

        result = cached_result
        lm     = None
        raw_g  = "NONE"

        if result and result.hand_landmarks:
            lm      = result.hand_landmarks[0]
            fingers = count_fingers(lm)
            raw_g   = recognize_gesture(fingers)

        gesture = confident_gesture(raw_g)

        # ── Draw hand skeleton ───────────────────────────────────
        if lm:
            draw_landmarks(img, lm, active=not paused)

        # ── PAUSE / RESUME ───────────────────────────────────────
        if gesture == "FIST":
            fist_frame_count += 1
            if fist_frame_count >= FIST_STABILIZE and not paused:
                paused = True
                if is_dragging:
                    pyautogui.mouseUp(button='left')
                    is_dragging = False
                flash_action("PAUSED")
        else:
            if paused and gesture == "OPEN_PALM":
                paused = False
                fist_frame_count = 0
                confirmed_gesture = "NONE"
                flash_action("RESUMED")
            elif not paused:
                fist_frame_count = 0

        # ── CONTROL LOGIC (skipped when paused) ─────────────────
        if not paused and lm:
            ti_pinch = is_thumb_index_pinch(lm)
            tm_pinch = is_thumb_middle_pinch(lm)

            # 1. CURSOR — Open Palm
            if gesture == "OPEN_PALM":
                scroll_ref_y = None
                move_cursor_ema(lm[9].x, lm[9].y, w, h)

            # 2. SCROLL — Peace ✌️
            elif gesture == "PEACE":
                tip_y = (lm[8].y + lm[12].y) / 2
                if scroll_ref_y is None:
                    scroll_ref_y = tip_y
                else:
                    delta = tip_y - scroll_ref_y
                    ticks = int(delta * SCREEN_H / SCROLL_SENSITIVITY)
                    if ticks != 0:
                        pyautogui.scroll(-ticks, _pause=False)
                        scroll_ref_y = tip_y

            # 3. RIGHT CLICK — thumb + middle
            elif tm_pinch and not ti_pinch:
                scroll_ref_y = None
                if not pinch_active:
                    pyautogui.click(button='right', _pause=False)
                    pinch_active = True
                    flash_action("RIGHT CLICK")

            # 4. LEFT CLICK / DOUBLE-CLICK / DRAG — thumb + index
            elif ti_pinch:
                scroll_ref_y = None
                if not pinch_active:
                    elapsed = now - last_pinch_time
                    if elapsed < DOUBLE_CLICK_GAP and not is_dragging:
                        pyautogui.doubleClick(_pause=False)
                        flash_action("DOUBLE CLICK")
                        last_pinch_time = 0
                    else:
                        last_pinch_time = now
                    pinch_active = True

                # Hold pinch → drag
                if now - last_pinch_time > DOUBLE_CLICK_GAP:
                    if not is_dragging:
                        pyautogui.mouseDown(button='left')
                        is_dragging = True
                    mid_x = (lm[4].x + lm[8].x) / 2
                    mid_y = (lm[4].y + lm[8].y) / 2
                    move_cursor_ema(mid_x, mid_y, w, h)

            # 5. No special gesture
            else:
                if is_dragging:
                    pyautogui.mouseUp(button='left')
                    is_dragging = False
                    flash_action("DRAG RELEASED")

                if pinch_active and not ti_pinch and not tm_pinch:
                    if not is_dragging and (now - last_pinch_time) < DOUBLE_CLICK_GAP:
                        pyautogui.click(_pause=False)
                        flash_action("LEFT CLICK")

                pinch_active = False
                scroll_ref_y = None

        # ══════════════════════════════════════════════════════════
        #  IRON MAN GUI OVERLAY
        # ══════════════════════════════════════════════════════════
        # Background scanlines
        draw_scanlines(img)
        # Corner brackets
        draw_corner_brackets(img)

        # FPS calc
        fps_display = 1 / (now - prev_time) if prev_time else fps_display
        prev_time   = now

        # Left panel — status
        state_col  = IM_RED if paused else IM_GREEN
        state_txt  = "PAUSED" if paused else ("DRAG" if is_dragging else "ACTIVE")
        draw_hud_panel(img, 12, 12, 230, [
            (f"STATUS : {state_txt}",               state_col),
            (f"GESTURE: {gesture}",                  IM_WHITE),
            (f"CONFIRM: {'|||' * len(gesture_vote_buf):8}", IM_GOLD),
            (f"FPS    : {int(fps_display)}",         IM_CYAN),
        ], title="◈ HAND CTRL v2")

        # Confidence bar just below panel
        draw_gesture_confidence_bar(img, 12, 138, 230, gesture_vote_buf)

        # Right panel — controls reference
        ctrl_lines = [
            ("Palm     → Move cursor",   IM_WHITE),
            ("Peace ✌  → Scroll",        IM_WHITE),
            ("T+I Pinch→ Click / Drag",  IM_WHITE),
            ("T+M Pinch→ Right-click",   IM_WHITE),
            ("Fist     → Pause",         IM_RED  ),
        ]
        draw_hud_panel(img, w - 232, 12, 220, ctrl_lines, title="◈ CONTROLS")

        # Centre action flash
        draw_action_flash(img, w, h, now)

        # Bottom bar
        if paused:
            bar_txt = "[ SYSTEM PAUSED ]  Open Palm to resume"
            bar_col = IM_RED
        elif is_dragging:
            bar_txt = "[ DRAGGING ]  Release pinch to drop"
            bar_col = IM_GOLD
        else:
            bar_txt = "◈ Hand Gesture Control — Iron Man Edition"
            bar_col = IM_CYAN
        ov = img.copy()
        cv2.rectangle(ov, (0, h-28), (w, h), IM_DARK, cv2.FILLED)
        cv2.addWeighted(ov, 0.7, img, 0.3, 0, img)
        cv2.putText(img, bar_txt, (12, h-9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, bar_col, 1, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Control — Iron Man Edition", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if is_dragging:
        pyautogui.mouseUp(button='left')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
