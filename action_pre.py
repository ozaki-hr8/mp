import cv2
import mediapipe as mp
import numpy as np
import csv
import os


mp_drawing = mp.solutions.drawing_utils
mp_holistic =  mp.solutions.holistic

cap = cv2.VideoCapture(0)

class_name ="Walking"

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # mp_drawing.draw_landmarks(image,results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        #                         mp_drawing.DrawingSpec(color=(80,0,0),thickness=2,circle_radius=1),
        #                         mp_drawing.DrawingSpec(color=(80,256,121),thickness=2,circle_radius=1)
        #                         )
        # mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #                         mp_drawing.DrawingSpec(color=(80,22,10),thickness=2,circle_radius=4),
        #                         mp_drawing.DrawingSpec(color=(80,44,121),thickness=2,circle_radius=2)
        #                         )
        # mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #                         mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),
        #                         mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2)
        #                         )
        mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                                )

        try:
            pose =results.pose_landmarks.landmark
            pose_row =list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in pose]).flatten())

            # face =results.face_landmarks.landmark
            # face_row =list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in face]).flatten())

            row = pose_row
            row.insert(0,class_name)

            with open('action_training.csv',mode='a' ,newline='') as f:
                csv_writer =csv.writer(f, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)

        except:
            pass

        cv2.imshow('Mediapipe' ,image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            # num_coords = len(results.pose_landmarks.landmark)
            #
            # landmarks = ['class']
            # for val in range(1, num_coords+1):
            #     landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
            #
            # with open('action.csv',mode='w' ,newline='') as f:
            #     csv_writer =csv.writer(f, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            #     csv_writer.writerow(landmarks)

            break

cap.release()
cv2.destroyAllWindows()
