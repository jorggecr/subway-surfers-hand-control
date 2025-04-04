import cv2 as cv
import mediapipe as mp
import math
import pyautogui

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Key states
left_arrow_pressed = False
right_arrow_pressed = False
down_pressed = False
up_pressed = False 

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Can't receive video. Exiting ...")
    exit()

def press_key(key):
    pyautogui.keyDown(key)

def release_key(key):
    pyautogui.keyUp(key)

with mp_hands.Hands(min_detection_confidence=0.8,
                    min_tracking_confidence=0.8,
                    max_num_hands=2) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break    

        frame = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Gesture control variables 
        left_arrow_active = False
        right_arrow_active = False
        down_active = False
        up_active = False  

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[idx].classification[0].label if results.multi_handedness else "Unknown"
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Finger coordinates 
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

                h, w, _ = frame.shape
                x_index, y_index = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x_middle, y_middle = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
                x_ring, y_ring = int(ring_finger_tip.x * w), int(ring_finger_tip.y * h)

                # Distances for gesture detection  
                thumb_index_distance = math.hypot(x_index - x_thumb, y_index - y_thumb)
                middle_ring_distance = math.hypot(x_middle - x_ring, y_middle - y_ring)

                # Gesture detection 
                if hand_label == "Right" and thumb_index_distance < 50:
                    right_arrow_active = True  # Move to the right 
                if hand_label == "Left" and thumb_index_distance < 50:
                    left_arrow_active = True  # Move to the left  
                if hand_label == "Right" and middle_ring_distance < 30:
                    down_active = True  # Crouch 
                if hand_label == "Left" and middle_ring_distance < 30:
                    up_active = True  # Jump 

        # Key control 
        if left_arrow_active and not left_arrow_pressed:
            press_key('left')
            print("press left")
            left_arrow_pressed = True
        elif not left_arrow_active and left_arrow_pressed:
            release_key('left')
            print("release left")
            left_arrow_pressed = False

        if right_arrow_active and not right_arrow_pressed:
            press_key('right')
            print("press right")
            right_arrow_pressed = True
        elif not right_arrow_active and right_arrow_pressed:
            release_key('right')
            print("release right")
            right_arrow_pressed = False

        if down_active and not down_pressed:
            press_key('down')
            print("press down")
            down_pressed = True
        elif not down_active and down_pressed:
            release_key('down')
            print("release down")
            down_pressed = False

        if up_active and not up_pressed:
            press_key('up')
            print("press up")
            up_pressed = True
        elif not up_active and up_pressed:
            release_key('up')
            print("release up")
            up_pressed = False

        # Set up the window with the camera  
        frame = cv.resize(frame, (800, 600))
        cv.imshow("Subway Surfers Controller", frame)

        # Exit with key 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
