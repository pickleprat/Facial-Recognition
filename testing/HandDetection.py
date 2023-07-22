import mediapipe as mp 
import cv2 as cv 
import numpy as np 
import os 

try:
    os.mkdir('VIDEOS')
except Exception as E: 
    pass 

# Capturing the video 
cam = cv.VideoCapture(0)
width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (width, height)
fourcc = cv.VideoWriter_fourcc(*'mp4v')

writer = cv.VideoWriter('VIDEOS/HandDetectionVideo.mp4', fourcc, 30, size)

hand = mp.solutions.hands
hand_processor = mp.solutions.hands.Hands()
hand_visualise = mp.solutions.drawing_utils 

while True: 
    _, image = cam.read()
    rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    hand_cordinates = hand_processor.process(rgb_img)

    if hand_cordinates.multi_hand_landmarks:
        for landmark in hand_cordinates.multi_hand_landmarks: 
            hand_visualise.draw_landmarks(image, landmark, hand.HAND_CONNECTIONS)

    if cam.isOpened():
        
        writer.write(image)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break 

    cv.imshow('image', image)
    
cam.release()
cv.destroyAllWindows()

        
