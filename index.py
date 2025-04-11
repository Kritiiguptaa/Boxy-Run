import cv2 as cv
import mediapipe as mp
import util
import pyautogui
import time

# Screen size
screen_width, screen_height = pyautogui.size()

# MediaPipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
draw = mp.solutions.drawing_utils

# Track previous direction
prev_direction = None
jumped = False
frame_count = 0

def detect_gesture(frame, landmark_list, raw_landmarks, processed):
    global jumped, prev_direction

    if len(landmark_list) >= 21:
        index_up = raw_landmarks[8].y < raw_landmarks[6].y
        middle_up = raw_landmarks[12].y < raw_landmarks[10].y
        ring_down = raw_landmarks[16].y > raw_landmarks[14].y
        pinky_down = raw_landmarks[20].y > raw_landmarks[18].y

        two_fingers_up = index_up and middle_up and ring_down and pinky_down
        one_finger_up = index_up and not middle_up and ring_down and pinky_down

        tips = [raw_landmarks[8], raw_landmarks[12], raw_landmarks[16], raw_landmarks[20]]
        pips = [raw_landmarks[6], raw_landmarks[10], raw_landmarks[14], raw_landmarks[18]]
        fist = all(tip.y > pip.y for tip, pip in zip(tips, pips))

        if two_fingers_up and prev_direction != 'right':
            pyautogui.press('right')
            print("Right")
            prev_direction = 'right'

        elif one_finger_up and prev_direction != 'left':
            pyautogui.press('left')
            print("Left")
            prev_direction = 'left'

        elif not one_finger_up and not two_fingers_up:
            prev_direction = None

        # wrist = raw_landmarks[0]
        # x = wrist.x
        # # print(f"Wrist X: {x:.2f}")
        # if x < 0.5 and prev_direction != 'left':
        #     pyautogui.press('left')
        #     print("← Moved LEFT by hand position")
        #     prev_direction = 'left'

        # elif x > 0.6 and prev_direction != 'right':
        #     pyautogui.press('right')
        #     print("→ Moved RIGHT by hand position")
        #     prev_direction = 'right'

        # elif 0.5 <= x <= 0.6:
        #     prev_direction = None


        if fist:
            if not jumped:
                pyautogui.press('up')
                print("Jump")
                jumped = True
        else:
            jumped = False

def main():
    global frame_count
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.flip(frame, 1)
            frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            raw_landmarks = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))
                    raw_landmarks.append(lm)

                detect_gesture(frame, landmark_list, raw_landmarks, processed)

            frame_count += 1
            cv.imshow('Hand Control', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)

    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()
