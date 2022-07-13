import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
#install pose estimation models
mp_pose =  mp.solutions.pose

cap = cv2.VideoCapture(0)

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
im1 = cv2.imread("palette.png")
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

            angle = calculate_angle(shoulder, elbow, wrist)

            cv2.putText(image, str(round(angle)),
                           tuple(np.multiply(elbow, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA
                                )
            if angle >100:
                show_of_hands = "Down"
                hand_gesture = "Down"
            if angle <35 and show_of_hands == "Down":
                show_of_hands = "Up"
                hand_gesture = "Up"
                counter+=1
                # print(counter)
                # print(landmarks)
        except:
            pass

        # cv2.rectangle(image,(0,0),(200,73),(245,117.16),-1)
        # cv2.putText(image,'REPS',(15,12),
        #             cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        # cv2.putText(image,str(counter),
        #             (10,60),
        #             cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
        # cv2.putText(image,'Gesture',(65,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        # cv2.putText(image,show_of_hands,
        #             (10,60),
        #             cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        cv2.circle(copy_image,right_index,120,(0,255,0),-1)
        cv2.circle(copy_image,left_index,120,(0,255,0),-1)
        # cv2.circle(im1,right_index,10,(0,255,0),-1)

        print(wval,hval)
        mat_img = cv2.addWeighted(copy_image, 0.4, image, 0.6, 0)
        cv2.imshow('Mediapipe' ,mat_img)
        # cv2.imshow('drawing' ,im1)
        person1 = Person(hand_gesture)
        print(right_index)

        

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
