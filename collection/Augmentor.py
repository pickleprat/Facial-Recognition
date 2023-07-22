import pandas as pd 
import cv2 as cv
import numpy as np 

import os 
import googlesearch 
import pickle 
import requests 
import os 
import openai 

from matplotlib import pyplot as plt 
from bs4 import BeautifulSoup 
from tqdm import tqdm 


class ImageAugmentor():
    
    def __init__(self, link = 'https://www.gettyimages.in/photos/pewdiepie'):
        self.skipper = 0 
        self.link = link 
        self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.page = BeautifulSoup(
            requests.get(
                link, 
                headers = {
                    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299', 
                }
            ).content 
        ) 

        self.__images = {
            'img_array':[], 
            'target':[],
        }
        
        self.img_links = []
        for img_tag in self.page.find_all('img'):

            try:
                
                if img_tag['src'].startswith('https'): 
                    self.img_links.append(img_tag['src'])

            except Exception as E:
                pass 
        

    def buffer_to_img(self, link):
        buffer = np.frombuffer(requests.get(link).content, np.uint8)
        img = cv.cvtColor(cv.imdecode(buffer, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)  
        return img 
    
    def array_to_buffer(self, array):
        _, buffer = cv.imencode('.png', array)
        img_buffer = buffer.tobytes()
        return img_buffer 
    
    def get_images(self):

        for img_link in tqdm(self.img_links):

            #Getting the image at the link 
            img = self.buffer_to_img(img_link)
            img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

            #Recognising faces in the image 
            faces = self.face_cascade.detectMultiScale(img_gray, 1.3, 5)
            try:
                if len(faces[0]) != 0:
                    x, y, w, h = faces[0]
                else:

                    img = cv.resize(img, (1024, 1024))
                    img_gray = cv.resize(img_gray, (1024, 1024))

                    faces = self.face_cascade.detectMultiScale(img_gray, 1.3, 5)
                    x, y, w, h = faces[0]

            except Exception as E:
                self.skipper += 1 
                continue 
                

            face_img = img[y:y+h, x:x+w]

            image = cv.resize(
                face_img, 
                (1024, 1024)
            ) 

            self.__images['img_array'].append(
                image
            )

            self.__images['target'].append('TARGET_OBJECT')

            response = openai.Image.create_variation(
               image = self.array_to_buffer(image), 
               n = 1, 
               size = "1024x1024", 
            )

            variant_url = response['data'][0]['url']
            variant_image = self.buffer_to_img(variant_url)
            self.__images['img_array'].append(
                variant_image
            )

            self.__images['target'].append('NOT_TARGET')

        return self.__images 
    
