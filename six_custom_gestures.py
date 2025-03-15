import cv2
import mediapipe as mp
import math

from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

import pyautogui

# Initialize Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define a function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.hypot(point2[0]-point1[0], point2[1]-point1[1])

# FOUND SOLUTION FOR HAND ORIENTATION:
    # The program reads hand gestured based on relative x and y coordinates on the screen, not just relative distances
    # between hand landmarks. Therefore, for a 'Thumbs_Up'gesture, the y-position of Thumb-Tip is higher than the
    # y-position of the Thumb-base. In a 'Thumbs_Down' gesture, this is the exact opposite; the y-position of th Thumb-Tip is 
    # lower than the y-position of Thumb-base.
def recognize(hand_landmarks):
    # We will recognize 5 different gestures according to the hand-landmarks of the user (and a 6th gesture that has no key-inputs).
    # Acquire key landmarks from thumb and index finger
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    

    # Calculate distances
    thumb_dist = calculate_distance(
        (thumb_tip.x, thumb_tip.y), 
        (thumb_mcp.x, thumb_mcp.y)
    )
    index_dist = calculate_distance(
        (index_tip.x, index_tip.y), 
        (index_mcp.x, index_mcp.y)
    )
    thumb_to_index_dist = calculate_distance(
        (thumb_ip.x, thumb_ip.y),
        (index_mcp.x, index_mcp.y)
    )

    # Returns the appropriate string
    # 1. Jump
    if thumb_to_index_dist < 0.05 and index_dist > 0.1:
        return "Jump"
    if thumb_to_index_dist < 0.05 and index_dist < 0.1:
        return "None"
    # 2. Jump-Left
    elif index_dist > 0.1 and thumb_tip.x < thumb_mcp.x:
        return "Jump-Left"
    # 3. Jump-Right
    elif index_dist > 0.1 and thumb_tip.x > thumb_mcp.x:
        return "Jump-Right"
    # 4. Move-Left
    elif index_dist < 0.1 and thumb_tip.x < thumb_mcp.x:
        return "Move-Left"
    # 5. Move-Right
    elif index_dist < 0.1 and thumb_tip.x > thumb_mcp.x:
        return "Move-Right"
    else:
        return "None"

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally and convert the BGR image to RGB.
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)

            # Draw the hand annotations on the image.
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Recognizes the gesture of the user. There are 6 possible gestures:
                    #   1. Jump
                    #   2. Move-Right   3. Jump-Right
                    #   4. Move-Left    5. Jump-Left
                    #   6. None

                    gesture = recognize(hand_landmarks)

                    # # Press appropriate keys with pyautugui based on recognized gesture.
                    if gesture == "Jump":
                        pyautogui.press("up")
                    elif gesture == "Jump-Left":
                        pyautogui.press(["up", "left"])
                    elif gesture == "Jump-Right":
                        pyautogui.press(["up", "right"])
                    elif gesture == "Move-Right":
                        pyautogui.press("right")
                    elif gesture == "Move-Left":
                        pyautogui.press("left")

                    # Display gesture near hand location
                    cv2.putText(image, gesture, 
                                (int(hand_landmarks.landmark[0].x * image.shape[1]), 
                                 int(hand_landmarks.landmark[0].y * image.shape[0]) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Display the resulting image
            cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
