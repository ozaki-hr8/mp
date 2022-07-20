import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic =  mp.solutions.holistic

cap = cv2.VideoCapture(4)

counter = 0
show_of_hands = None
hand_gesture= "Down"

class Person:
    def __init__(self, hand_gesture):
        self.hand_gesture = hand_gesture

def calculate_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle =360-angle
    return angle

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        try:
            pose = results.pose_landmarks.landmark
            right_shoulder = [pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                            pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [pose[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,
                        pose[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [pose[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,
                        pose[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
            left_shoulder = [pose[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                            pose[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [pose[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                        pose[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [pose[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                        pose[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]

            right_angle = calculate_angle(
                right_shoulder, right_elbow, right_wrist)
            left_angle = calculate_angle(
                left_shoulder, left_elbow, left_wrist)

            if right_angle > 100 and left_angle > 100:
                hand_gesture = "Down"
            elif right_angle < 55 and left_angle > 100 and hand_gesture == "Down":
                hand_gesture = "Right Hand Up"
            elif left_angle < 55 and right_angle > 100 and hand_gesture == "Down":
                hand_gesture = "Left Hand Up"
            elif right_angle < 55 and left_angle < 55 and hand_gesture == "Down":
                hand_gesture = "Both Hands Up"
            else:
                pass
        except:
            pass
        image = cv2.copyMakeBorder(image, 130, 0, 0, 0, cv2.BORDER_CONSTANT, value=[245, 117, 16])
        # Display Probability
        cv2.putText(image, 'Gesture'
                    , (15,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(image, hand_gesture
                    , (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow('Gesture', image)
    
        person1 = Person(hand_gesture)
        print(person1.hand_gesture)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
