import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import time
import math
from collections import deque
import winsound

# ══════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════
SMOOTHING           = 5      # cursor smoothing factor
FRAME_REDUCTION     = 100    # virtual trackpad margin (px)
PINCH_THRESHOLD     = 0.07   # normalised thumb-index distance for pinch
MID_PINCH_THRESHOLD = 0.07   # normalised thumb-middle distance for right-click pinch
DOUBLE_CLICK_GAP    = 0.35   # max seconds between two pinches = double-click
SCROLL_SENSITIVITY  = 30     # pixels of hand movement per scroll tick
FIST_STABILIZE      = 6      # frames fist must be held to enter pause mode

# ══════════════════════════════════════════════════════
#  GESTURE DEFINITIONS   [Thumb, Index, Middle, Ring, Pinky]
# ══════════════════════════════════════════════════════
GESTURES = {
    "OPEN_PALM":  [True,  True,  True,  True,  True ],
    "FIST":       [False, False, False, False, False],
    "PEACE":      [False, True,  True,  False, False],  # scroll / cursor (prev scheme)
    "INDEX_UP":   [False, True,  False, False, False],
    "THUMB_UP":   [True,  False, False, False, False],
}

# ══════════════════════════════════════════════════════
#  GLOBAL STATE
# ══════════════════════════════════════════════════════
pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()

# Cursor
ploc_x, ploc_y  = 0, 0
target_history  = deque(maxlen=4)

# Drag
is_dragging     = False

# Pause (fist)
paused          = False
fist_frame_count = 0

# Click timing (for double-click detection)
last_pinch_time      = 0.0
pinch_active         = False   # True while thumb+index are currently together
click_cooldown       = 0.3     # seconds to ignore new left-pinch after a click

# Scroll
scroll_ref_y    = None         # y position when PEACE entered

# Last action display
last_action_name = ""
last_action_time = 0.0

# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════
def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def count_fingers(lm):
    """[Thumb, Index, Middle, Ring, Pinky]"""
    fingers = []
    fingers.append(
        get_distance(lm[4], lm[17]) > get_distance(lm[3], lm[17])
    )
    for tip in [8, 12, 16, 20]:
        fingers.append(lm[tip].y < lm[tip - 2].y)
    return fingers

def recognize_gesture(fingers):
    for name, pattern in GESTURES.items():
        if fingers == pattern:
            return name
    return "UNKNOWN"

class _Pt:
    """Lightweight point wrapper."""
    def __init__(self, x, y): self.x = x; self.y = y

def is_thumb_index_pinch(lm):
    return get_distance(lm[4], lm[8]) < PINCH_THRESHOLD

def is_thumb_middle_pinch(lm):
    return get_distance(lm[4], lm[12]) < MID_PINCH_THRESHOLD

def move_cursor(pt, w, h):
    """Smoothly move cursor to normalised hand point (pt.x, pt.y)."""
    global ploc_x, ploc_y
    px = pt.x * w
    py = pt.y * h
    bw = w - 2 * FRAME_REDUCTION
    bh = h - 2 * FRAME_REDUCTION
    mx = max(FRAME_REDUCTION, min(px, w - FRAME_REDUCTION))
    my = max(FRAME_REDUCTION, min(py, h - FRAME_REDUCTION))
    tx = ((mx - FRAME_REDUCTION) / bw) * SCREEN_W
    ty = ((my - FRAME_REDUCTION) / bh) * SCREEN_H
    target_history.append((tx, ty))
    ax = sum(t[0] for t in target_history) / len(target_history)
    ay = sum(t[1] for t in target_history) / len(target_history)
    dx, dy = ax - ploc_x, ay - ploc_y
    dist = math.hypot(dx, dy)
    if dist > 1.5:
        s = SMOOTHING * (1.5 if dist < 30 else (0.4 if dist > 150 else 1.0))
        cloc_x = ploc_x + dx / s
        cloc_y = ploc_y + dy / s
        try:
            pyautogui.moveTo(int(cloc_x), int(cloc_y), _pause=False)
            ploc_x, ploc_y = cloc_x, cloc_y
        except pyautogui.FailSafeException:
            pass

def flash_action(name):
    global last_action_name, last_action_time
    last_action_name = name
    last_action_time = time.time()
    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
    print(f"[Action] {name}")

def draw_landmarks(img, lm, color=(255, 180, 50)):
    h, w, _ = img.shape
    for p in lm:
        cv2.circle(img, (int(p.x*w), int(p.y*h)), 5, (255,255,255), cv2.FILLED)
    for a, b in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                 (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                 (13,17),(0,17),(17,18),(18,19),(19,20)]:
        cv2.line(img,
                 (int(lm[a].x*w), int(lm[a].y*h)),
                 (int(lm[b].x*w), int(lm[b].y*h)),
                 color, 2)

def draw_overlay_text(img, text, pos, font_scale=0.65,
                      color=(255,255,255), thickness=2, bg=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    sz, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y  = pos
    cv2.rectangle(img, (x-4, y-sz[1]-8), (x+sz[0]+4, y+4), bg, cv2.FILLED)
    cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    global ploc_x, ploc_y, target_history
    global is_dragging, paused, fist_frame_count
    global pinch_active, last_pinch_time
    global scroll_ref_y
    global last_action_name, last_action_time

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector     = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = 0

    while True:
        ok, img = cap.read()
        if not ok:
            continue

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        now      = time.time()
        fps      = 1 / (now - prev_time) if prev_time else 0
        prev_time = now

        img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result   = detector.detect(mp_img)

        lm       = None
        gesture  = "NONE"

        if result.hand_landmarks:
            lm      = result.hand_landmarks[0]
            fingers = count_fingers(lm)
            gesture = recognize_gesture(fingers)
            lm_color = (80, 80, 255) if paused else (255, 180, 50)
            draw_landmarks(img, lm, color=lm_color)

        # ── PAUSE / RESUME via FIST ────────────────────────────────
        if gesture == "FIST":
            fist_frame_count += 1
            if fist_frame_count >= FIST_STABILIZE and not paused:
                paused = True
                if is_dragging:
                    pyautogui.mouseUp(button='left')
                    is_dragging = False
                target_history.clear()
                flash_action("PAUSED")
        else:
            if paused and gesture == "OPEN_PALM":
                paused = False
                fist_frame_count = 0
                flash_action("RESUMED")
            elif not paused:
                fist_frame_count = 0

        if paused or lm is None:
            # Only render HUD, skip all control logic
            _render_hud(img, h, w, fps, gesture, now)
            cv2.imshow("Hand Gesture Control", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ══════════════════════════════════════════
        #  GESTURE DISPATCH
        # ══════════════════════════════════════════

        ti_pinch  = is_thumb_index_pinch(lm)   # thumb + index
        tm_pinch  = is_thumb_middle_pinch(lm)   # thumb + middle

        # ── 1. CURSOR MOVEMENT — Open Palm ────────────────────────
        if gesture == "OPEN_PALM":
            scroll_ref_y = None   # exit scroll mode if we were in it
            # Use mid-palm (landmark 9) as tracking point
            move_cursor(lm[9], w, h)

        # ── 2. SCROLL — PEACE (index + middle up) ─────────────────
        elif gesture == "PEACE":
            tip_y = (lm[8].y + lm[12].y) / 2   # avg of index+middle tip
            if scroll_ref_y is None:
                scroll_ref_y = tip_y
            else:
                delta = tip_y - scroll_ref_y
                ticks = int(delta * SCREEN_H / SCROLL_SENSITIVITY)
                if ticks != 0:
                    pyautogui.scroll(-ticks, _pause=False)
                    scroll_ref_y = tip_y   # reset ref after scroll
            target_history.clear()

        # ── 3. RIGHT CLICK — thumb + middle pinch ─────────────────
        elif tm_pinch and not ti_pinch:
            scroll_ref_y = None
            target_history.clear()
            if not pinch_active:
                pyautogui.click(button='right', _pause=False)
                pinch_active = True
                flash_action("RIGHT CLICK")
        
        # ── 4. LEFT CLICK / DOUBLE-CLICK / DRAG — thumb+index pinch
        elif ti_pinch:
            scroll_ref_y = None
            if not pinch_active:
                # New pinch started
                elapsed = now - last_pinch_time
                if elapsed < DOUBLE_CLICK_GAP and not is_dragging:
                    # Second pinch within window → double click
                    pyautogui.doubleClick(_pause=False)
                    flash_action("DOUBLE CLICK")
                    last_pinch_time = 0   # reset so triple doesn't trigger
                else:
                    last_pinch_time = now
                pinch_active = True

            # While pinch is held → drag
            if now - last_pinch_time > DOUBLE_CLICK_GAP:
                if not is_dragging:
                    pyautogui.mouseDown(button='left')
                    is_dragging = True
                # Track midpoint of thumb+index while dragging
                mid_x = (lm[4].x + lm[8].x) / 2
                mid_y = (lm[4].y + lm[8].y) / 2
                move_cursor(_Pt(mid_x, mid_y), w, h)

        # ── No pinch / no special gesture ─────────────────────────
        else:
            # Release left drag if pinch lifted
            if is_dragging:
                pyautogui.mouseUp(button='left')
                is_dragging = False
                flash_action("DRAG RELEASED")

            # Fire single left click on pinch release
            if pinch_active and not ti_pinch and not tm_pinch:
                if not is_dragging and (now - last_pinch_time) < DOUBLE_CLICK_GAP:
                    pyautogui.click(_pause=False)
                    flash_action("LEFT CLICK")

            pinch_active = False
            scroll_ref_y = None

            if gesture not in ("OPEN_PALM", "PEACE"):
                target_history.clear()

        _render_hud(img, h, w, fps, gesture, now)
        cv2.imshow("Hand Gesture Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if is_dragging:
        pyautogui.mouseUp(button='left')
    cap.release()
    cv2.destroyAllWindows()


def _render_hud(img, h, w, fps, gesture, now):
    global last_action_name, last_action_time, paused

    time_since = now - last_action_time

    if paused:
        draw_overlay_text(img, "⏸  PAUSED  (Open Palm to resume)", (15, 35),
                          color=(80, 80, 255), font_scale=0.65)
    else:
        draw_overlay_text(img, f"Gesture: {gesture}", (15, 35))

    draw_overlay_text(img, f"FPS: {int(fps)}", (15, 70), font_scale=0.5, thickness=1)

    # Action flash in centre
    if time_since < 2.0:
        alpha = max(0.0, 1.0 - time_since / 2.0)
        if alpha > 0:
            text  = f"  {last_action_name}  "
            font  = cv2.FONT_HERSHEY_DUPLEX
            scale, thick = 1.1, 2
            sz, _ = cv2.getTextSize(text, font, scale, thick)
            tx, ty = (w - sz[0]) // 2, h // 2
            ov = img.copy()
            cv2.rectangle(ov, (tx-15, ty-sz[1]-15), (tx+sz[0]+15, ty+15), (30,180,30), cv2.FILLED)
            cv2.putText(ov, text, (tx, ty), font, scale, (255,255,255), thick, cv2.LINE_AA)
            cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

    # Bottom hint bar
    hint = "Palm=Move | Peace=Scroll | Pinch(T+I)=Click/Drag | Pinch(T+M)=RightClick | Fist=Pause"
    draw_overlay_text(img, hint, (15, h-18), font_scale=0.38, thickness=1)


if __name__ == "__main__":
    main()
