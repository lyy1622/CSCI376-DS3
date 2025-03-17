import cv2
import mediapipe as mp

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

import pyautogui

# Path to the gesture recognition model
model_path = "gesture_recognizer.task"  # Update this to the correct path where the model is saved, if not in current directory
handedness_model_path = "hand_landmarker.task"

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=2
)
handedness_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=handedness_model_path),
    num_hands=2
)
gesture_recognizer = GestureRecognizer.create_from_options(options)
handedness_recognizer = HandLandmarker.create_from_options(handedness_options)

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally and convert the BGR image to RGB.
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a Mediapipe Image object for the gesture recognizer
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Perform gesture recognition on the image
        result = gesture_recognizer.recognize(mp_image)
        handedness_result = handedness_recognizer.detect(mp_image)

        # Draw the gesture recognition results on the image
        if result.gestures:
            #print(result.gestures)
            recognized_gesture_1 = result.gestures[0][0].category_name
            confidence_1 = result.gestures[0][0].score

            print(handedness_result)

            if handedness_result.handedness:
                perceived_handedness = handedness_result.handedness[0][0].category_name
                if perceived_handedness == "Left":
                    cv2.putText(image, f"Gesture: {recognized_gesture_1} ({confidence_1:.2f})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    # Display recognized gesture and confidence 
                    cv2.putText(image, f"Handedness: {perceived_handedness}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif perceived_handedness == "Right":
                    cv2.putText(image, f"Gesture: {recognized_gesture_1} ({confidence_1:.2f})", 
                            (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    # Display recognized gesture and confidence 
                    cv2.putText(image, f"Handedness: {perceived_handedness}", 
                                (1000, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            

            if len(result.gestures) > 1:
                recognized_gesture_2 = result.gestures[1][0].category_name
                confidence_2 = result.gestures[1][0].score

                

                if handedness_result.handedness:
                    perceived_handedness = handedness_result.handedness[1][0].category_name
                    if perceived_handedness == "Left":
                        cv2.putText(image, f"Gesture: {recognized_gesture_2} ({confidence_2:.2f})", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        # Display recognized gesture and confidence 
                        cv2.putText(image, f"Handedness: {perceived_handedness}", 
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    elif perceived_handedness == "Right":
                        cv2.putText(image, f"Gesture: {recognized_gesture_2} ({confidence_1:.2f})", 
                                (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        # Display recognized gesture and confidence 
                        cv2.putText(image, f"Handedness: {perceived_handedness}", 
                                    (1000, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Example of pressing keys with pyautogui based on recognized gesture
            '''
            if recognized_gesture == "Thumb_Up":
                pyautogui.press("w")
            elif recognized_gesture == "Thumb_Down":
                pyautogui.press("s")
            elif recognized_gesture == "Open_Palm":
                pyautogui.press("a")
            elif recognized_gesture == "Closed_Fist":
                pyautogui.press("d")
            elif recognized_gesture == "Victory":
                pyautogui.press("space")
            '''

            # Display recognized gesture and confidence 
            
            

        # Display the resulting image (can comment this out for better performance later on)
        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
