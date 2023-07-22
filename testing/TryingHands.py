import tensorflow as tf 
import pandas as pd 
import numpy as np 
import cv2 as cv 
import mediapipe as mp 
import warnings as warn 
import os 
import matplotlib.pyplot as plt 
warn.simplefilter("ignore")


# Instantiating dependencies 
palm_utils = mp.solutions.hands 
palm_artist = mp.solutions.drawing_utils 
hand_locator = palm_utils.Hands()
video = cv.VideoCapture(0)

while True: 
    _, img = video.read()
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    height, width = gray_img.shape 
    landmarks = hand_locator.process(img)

    if landmarks.multi_hand_landmarks:
        hands_json = {}
        lmk_x, lmk_y = [], []
        for lmk in landmarks.multi_hand_landmarks:
            palm_artist.draw_landmarks(img, lmk, palm_utils.HAND_CONNECTIONS)
            for landmark in lmk.landmark:
                lmk_x.append(landmark.x)
                lmk_y.append(landmark.y)
        
        min_x, max_x, min_y, max_y = int(np.floor(np.min(lmk_x)*width)) - 18, int(np.ceil(np.max(lmk_x)*width)) + 18, int(np.floor(np.min(lmk_y)*height)) - 18, int(np.ceil(np.max(lmk_y)*height)) + 18 

        
    cv.imshow('ASL', img)
    if cv.waitKey(1) & 0xFF == ord('q'): break 
    
cv.destroyAllWindows()
video.release()

