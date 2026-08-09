import cv2
import mediapipe as mp
import pyautogui
from screeninfo import get_monitors

# -------------------- SCREEN SIZE --------------------
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# -------------------- MEDIAPIPE --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)

# -------------------- VARIABLES --------------------
prev_x, prev_y = 0, 0
smoothening = 6
scroll_reference = None

# -------------------- FINGER DETECTION --------------------
def fingers_up(lm):
    fingers = []

    # Thumb
    if lm[4].x < lm[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index, Middle, Ring, Pinky
    for tip in [8, 12, 16, 20]:
        if lm[tip].y < lm[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# -------------------- MAIN LOOP --------------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            finger_status = fingers_up(lm)
            total_fingers = finger_status.count(1)

            # Index finger position
            x1 = int(lm[8].x * screen_width)
            y1 = int(lm[8].y * screen_height)

            # ---------------- MOVE CURSOR (V SIGN) ----------------
            if finger_status[1] == 1 and finger_status[2] == 1 and total_fingers == 2:
                cur_x = prev_x + (x1 - prev_x) / smoothening
                cur_y = prev_y + (y1 - prev_y) / smoothening
                pyautogui.moveTo(cur_x, cur_y)
                prev_x, prev_y = cur_x, cur_y
                cv2.putText(img, "MOVE", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

            # ---------------- LEFT CLICK ----------------
            elif finger_status[1] == 1 and total_fingers == 1:
                pyautogui.click()
                cv2.putText(img, "LEFT CLICK", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # ---------------- RIGHT CLICK ----------------
            elif finger_status[2] == 1 and total_fingers == 1:
                pyautogui.rightClick()
                cv2.putText(img, "RIGHT CLICK", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            # ---------------- SCROLL (3 FINGERS) ----------------
            elif total_fingers == 3:
                current_y = lm[8].y

                if scroll_reference is None:
                    scroll_reference = current_y

                diff = current_y - scroll_reference

                if diff < -0.02:
                    pyautogui.scroll(60)
                    scroll_reference = current_y
                    cv2.putText(img, "SCROLL UP", (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

                elif diff > 0.02:
                    pyautogui.scroll(-60)
                    scroll_reference = current_y
                    cv2.putText(img, "SCROLL DOWN", (10, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            else:
                scroll_reference = None

            mp_draw.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("AI Gesture Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------- CLEANUP --------------------
cap.release()
cv2.destroyAllWindows()
