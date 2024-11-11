## jupyter nbconvert --to script motiondetection.ipynb 


#!/usr/bin/env python
# coding: utf-8

# 

# 

# ## import all the libaries--

# In[2]:


import cv2
import mediapipe as mp
import numpy as np


# ## Initailize the mediapipe-pose and drawing utilites

# In[2]:


mp_pose=mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils


# ## Calculate the angles using numpy that person is standing or sitting
# 
# 

# In[3]:


def Calc_angle(a,b,c):

    #Covert the ppoint to the vector
    a=np.array(a)             # a=[x,y]
    b=np.array(b)
    c=np.array(c)
    #creating vector ba & bc
    ba=a-b
    bc=c-b

    #creating dot vector and the magnitubde of ba and bc
    
    dot_product=np.dot(ba,bc)
    mag_ba=np.linalg.norm(ba)
    mag_bc=np.linalg.norm(bc)


    #using formula
    #accross---->return the inverse cosine and return the value in radian
    #degrees---->convert the radian value to the degree


    angle=np.arccos(dot_product/(mag_ba * mag_bc))
    angle=np.degrees(angle)
    return angle
    


# ## Create a function to detect the person is sitting or standing

# In[4]:


def detect_person(landmarks):

    # we give landmarks as the input 

    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]                            ## just like a=[x,y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
    
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]



     #calculate the angle using the calculating function defined above 
    left_knee_angle = Calc_angle(left_hip, left_knee, left_ankle)                  # since left knee is the vertex
    right_knee_angle = Calc_angle(right_hip, right_knee, right_ankle)


    standing = left_hip[1] < left_knee[1] and right_hip[1] < right_knee[1]


    if left_knee_angle > 160 and right_knee_angle > 160 and standing:
        return "Standing"
    else:
        return "Sitting"


# ## Create a function to detect the person if standing so walking or not
# 

# In[5]:


def detect_walking(landmarks,previos_landmark):
    left_foot=[landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
    right_foot=[landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

    prev_left_foot=[previos_landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
    prev_right_foot=[previos_landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

    left_movement=np.linalg.norm(np.array(left_foot) - np.array(prev_left_foot))
    right_movement=np.linalg.norm(np.array(right_foot) - np.array(prev_right_foot))

    movement_threshold = 0.5

    if left_movement > movement_threshold or right_movement > movement_threshold:
        return "Walking"
    else:
        return "Standing Still"



# ## Create a function which will combiine all the function and tell weather it is standing ,walking or sitting.

# In[6]:


def detect_all(landmarks,previos_landmark=None):
     posture = detect_person(landmarks)

    # If the person is standing, check for walking
     if posture == "Standing" and previos_landmark is not None:
        movement = detect_walking(landmarks, previos_landmark)
        return movement
     else:
        return posture


# ## Load it to the pickle file to use in the in the fronted file

# In[ ]:




# ## use cv2 with mediapipe ##
# 
# 
# use cv2 to open the camera or any external camera you want to use ,and mediapipe will tell and create the landmark of the object such as left_knee nose etc.
# nose->landmark-->0
# total 33 landmark

# In[33]:


mp_pose=mp.solutions.pose
video_path = '2791669-uhd_2160_3840_25fps.mp4'
cap = cv2.VideoCapture(video_path)


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret,frame=cap.read()


        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False

        result=pose.process(image)

        image.flags.writeable=True
        image=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)


        if result.pose_landmarks:
              posture=detect_all(result.pose_landmarks.landmark,previos_landmark=None)
              previos_landmark=result.pose_landmarks.landmark
              cv2.putText(image, posture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)




        mp_drawing.draw_landmarks(image,
                                 result.pose_landmarks,
                                 mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(255,255,0),thickness=2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=2))
        
        
    
        cv2.imshow('Detection',image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

