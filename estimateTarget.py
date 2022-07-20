import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose =  mp.solutions.pose

cap = cv2.VideoCapture(4)

right_index=None
left_index=None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()  

        image =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        copy_image = image.copy()

        hval = image.shape[0]
        wval = image.shape[1]

        try:
            landmarks = results.pose_landmarks.landmark

            right_index = (round(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x*wval),round(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y*hval))
            left_index = (round(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x*wval),round(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y*hval))
        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        cv2.circle(copy_image,right_index,120,(0,255,0),-1)
        cv2.circle(copy_image,left_index,120,(0,255,0),-1)
        mat_img = cv2.addWeighted(copy_image, 0.4, image, 0.6, 0)
        cv2.imshow('Mediapipe' ,mat_img)
        

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
