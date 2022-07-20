import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import pickle
import pandas as pd

with open('pkl/expression.pkl', 'rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_holistic =  mp.solutions.holistic

cap = cv2.VideoCapture(4)

# # Display FPS
# tm = cv2.TickMeter()
# tm.start()

# count = 0
# max_count = 10
# fps = 0

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        # #  Display FPS
        # if count == max_count:
        #     tm.stop()
        #     fps = max_count / tm.getTimeSec()
        #     tm.reset()
        #     tm.start()
        #     count = 0
        # cv2.putText(frame, 'FPS: {:.2f}'.format(fps),(180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        normalImage=image.copy()

        mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                                            )
        mp_drawing.draw_landmarks(image,results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                mp_drawing.DrawingSpec(color=(80,0,0),thickness=1,circle_radius=1),
                                mp_drawing.DrawingSpec(color=(0,236,255),thickness=1,circle_radius=1)
                                )
        # Export coordinates
        body_language_class = "None"
        body_language_prob = (0,0,0)
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            # Concate rows
            row = pose_row+face_row

            # Make Detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, body_language_prob)

        except:
            pass

        # Get status box
        image = cv2.copyMakeBorder(image, 130, 0, 0, 0, cv2.BORDER_CONSTANT, value=[245, 117, 16])
        # Display Class
        cv2.putText(image, 'Expression'
                    , (155,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(image, body_language_class.split(' ')[0]
                    , (150,90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

        # Display Probability
        cv2.putText(image, 'Probability'
                    , (15,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                    , (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow('Expression', image)
        cv2.imshow('image',normalImage)
        # # Display FPS
        # count += 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
