import cv2
import mediapipe as mp
import math

from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

import pyautogui
import re

# Initialize Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
handedness_model_path = "hand_landmarker.task"

handedness_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=handedness_model_path),
    num_hands=1
)
handedness_recognizer = HandLandmarker.create_from_options(handedness_options)

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

    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    

    # Calculate distances
    thumb_dist = calculate_distance(
        (thumb_tip.x, thumb_tip.y), 
        (thumb_mcp.x, thumb_mcp.y)
    )
    index_dist = calculate_distance(
        (index_tip.x, index_tip.y), 
        (index_mcp.x, index_mcp.y)
    )
    pinky_dist = calculate_distance(
        (pinky_tip.x, pinky_tip.y),
        (pinky_mcp.x, pinky_mcp.y)
    )
    thumb_to_index_dist = calculate_distance(
        (thumb_ip.x, thumb_ip.y),
        (index_mcp.x, index_mcp.y)
    )

    # Returns the appropriate string
    # 1. Jump
    if thumb_to_index_dist < 0.05 and index_dist > 0.1:
        return "Jump"
    # 2. None
    if thumb_to_index_dist < 0.05 and index_dist < 0.1:
        # 3. Restart
        if pinky_dist > 0.1:
            return "Restart"
    
        return "None1"
    # 3. Jump-Left
    elif index_dist > 0.1 and thumb_tip.x < thumb_mcp.x:
        return "Jump-Left"
    # 4. Jump-Right
    elif index_dist > 0.1 and thumb_tip.x > thumb_mcp.x:
        return "Jump-Right"
    # 5. Move-Left
    elif index_dist < 0.1 and thumb_tip.x < thumb_mcp.x:
        return "Move-Left"
    # 6. Move-Right
    elif index_dist < 0.1 and thumb_tip.x > thumb_mcp.x:
        return "Move-Right"
    else:
        return "None2"

def getCoord(landmark_string):
    xCoord = []
    yCoord = []
    landmark_string = re.split(r'\n', landmark_string)
    for line in landmark_string:
        line = line.strip()
        if len(line) > 3:
            if line[0] == 'x':
                xCoord.append(float(line[2:]))
            elif line[0] == 'y':
                yCoord.append(float(line[2:]))
    
    return min(xCoord), max(xCoord), min(yCoord), max(yCoord)


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    left_count = 5
    right_count = 5
    w = False
    a = False
    d = False
    up = False
    left = False
    right = False

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

            
            
            if left_count > 0:
                left_count -= 1
                if left_count == 0:
                    if w:
                        pyautogui.keyUp("w")
                        w = False
                    if a:
                        pyautogui.keyUp("a")
                        a = False
                    if d:
                        pyautogui.keyUp("d")
                        d = False
            if right_count > 0:
                right_count -= 1
                if right_count == 0:
                    if up:
                        pyautogui.keyUp("up")
                        up = False
                    if left:
                        pyautogui.keyUp("left")
                        left = False
                    if right:
                        pyautogui.keyUp("right")
                        right = False

            # Flip the image horizontally and convert the BGR image to RGB.
            old_image = image
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            

            # Draw the hand annotations on the image.
            image_rgb.flags.writeable = True
            image2 = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image2, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    landmarks_string = str(hand_landmarks)
                    minX, maxX, minY, maxY = getCoord(landmarks_string)
                    minX = max(0, int(minX * image.shape[1]) - 100)
                    maxX = min(image.shape[1], int(maxX * image.shape[1]) + 100)
                    minY = max(0, int(minY * image.shape[0]) - 100)
                    maxY = min(image.shape[0], int(maxY * image.shape[0]) + 100)

                    cropped_img = image[minY:maxY, minX:maxX]
                    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                    mp_cropped = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_rgb)
                    #cv2.imshow('Cropped', cropped_img)

                    handedness_result = handedness_recognizer.detect(mp_cropped)
                    gesture = recognize(hand_landmarks)


                    if handedness_result.handedness:
                        perceived_handedness = handedness_result.handedness[0][0].category_name

                        # due to flipped video, handedness is also flipped
                        # it is less computationally intensive to simply switch the result rather than reflip the video
                        real_handedness = "left"
                        if (perceived_handedness == "Left"):
                            real_handedness = "right"

                        if (real_handedness == "left"):
                            left_count = 5
                            # Recognizes the gesture of the user. There are 6 possible gestures:
                            #   1. Jump
                            #   2. Move-Right   3. Jump-Right
                            #   4. Move-Left    5. Jump-Left
                            #   6. None

                            # # Press appropriate keys with pyautugui based on recognized gesture.
                            if gesture == "Jump":
                                if a:
                                    pyautogui.keyUp("a")
                                    a = False
                                if d:
                                    pyautogui.keyUp("d")
                                    d = False

                                if not w:
                                    pyautogui.keyDown("w")
                                    w = True
                            elif gesture == "Jump-Left":
                                if d:
                                    pyautogui.keyUp("d")
                                    d = False

                                if not w:
                                    pyautogui.keyDown("w")
                                    w = True
                                if not a:
                                    pyautogui.keyDown("a")
                                    a = True
                            elif gesture == "Jump-Right":
                                if a:
                                    pyautogui.keyUp("a")
                                    a = False
                                
                                if not w:
                                    pyautogui.keyDown("w")
                                    w = True
                                if not d:
                                    pyautogui.keyDown("d")
                                    d = True
                            elif gesture == "Move-Right":
                                if w:
                                    pyautogui.keyUp("w")
                                    w = False
                                if a:
                                    pyautogui.keyUp("a")
                                    a = False

                                if not d:
                                    pyautogui.keyDown("d")
                                    d = True
                            elif gesture == "Move-Left":
                                if w:
                                    pyautogui.keyUp("w")
                                    w = False
                                if d:
                                    pyautogui.keyUp("d")
                                    d = False

                                if not a:
                                    pyautogui.keyDown("a")
                                    a = True
                            elif gesture == "Restart":
                                if w:
                                    pyautogui.keyUp("w")
                                    w = False
                                if a:
                                    pyautogui.keyUp("a")
                                    a = False
                                if d:
                                    pyautogui.keyUp("d")
                                    d = False
                                pyautogui.press("r")
                            else:
                                if w:
                                    pyautogui.keyUp("w")
                                    w = False
                                if a:
                                    pyautogui.keyUp("a")
                                    a = False
                                if d:
                                    pyautogui.keyUp("d")
                                    d = False

                        else: #real_handedness == right
                            right_count = 5
                            if gesture == "Jump":
                                if left:
                                    pyautogui.keyUp("left")
                                    left = False
                                if right:
                                    pyautogui.keyUp("right")
                                    right = False

                                if not up:
                                    pyautogui.keyDown("up")
                                    up = True
                            elif gesture == "Jump-Left":
                                if right:
                                    pyautogui.keyUp("right")
                                    right = False

                                if not up:
                                    pyautogui.keyDown("up")
                                    up = True
                                if not left:
                                    pyautogui.keyDown("left")
                                    left = True
                            elif gesture == "Jump-Right":
                                if left:
                                    pyautogui.keyUp("left")
                                    left = False
                                
                                if not up:
                                    pyautogui.keyDown("up")
                                    up = True
                                if not right:
                                    pyautogui.keyDown("right")
                                    right = True
                            elif gesture == "Move-Right":
                                if up:
                                    pyautogui.keyUp("up")
                                    up = False
                                if left:
                                    pyautogui.keyUp("left")
                                    left = False

                                if not right:
                                    pyautogui.keyDown("right")
                                    right = True
                            elif gesture == "Move-Left":
                                if up:
                                    pyautogui.keyUp("up")
                                    up = False
                                pyautogui.keyUp("right")

                                if not left:
                                    pyautogui.keyDown("left")
                                    left = True
                            elif gesture == "Restart":
                                if up:
                                    pyautogui.keyUp("up")
                                    up = False
                                if left:
                                    pyautogui.keyUp("left")
                                    left = False
                                if right:
                                    pyautogui.keyUp("right")
                                    right = False
                                pyautogui.press("r")
                            else:
                                if up:
                                    pyautogui.keyUp("up")
                                    up = False
                                if left:
                                    pyautogui.keyUp("left")
                                    left = False
                                if right:
                                    pyautogui.keyUp("right")
                                    right = False
                    

                    # Display gesture near hand location
                    cv2.putText(image2, gesture, 
                                (int(hand_landmarks.landmark[0].x * image2.shape[1]), 
                                 int(hand_landmarks.landmark[0].y * image2.shape[0]) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Display the resulting image
            cv2.imshow('Gesture Recognition', image2)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
