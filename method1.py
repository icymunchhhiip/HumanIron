import picamera
import RPi.GPIO as GPIO
import time
from PIL import Image
import requests
import os
import pygame
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from pycocotools.coco import COCO

pygame.mixer.init()
bang = pygame.mixer.Sound("please_correct.wav")
quitm = pygame.mixer.Sound("quit.wav")
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN, GPIO.PUD_UP)


APP_KEY = '61438e2034d5616b9ecaf5ab8ccf7bf7'

session = requests.Session()
session.headers.update({'Authorization': 'KakaoAK ' + APP_KEY})
 

ORIGIN_PATH = '/home/pi/HumanIron/origin.jpg'
NEW_PATH = '/home/pi/HumanIron/new.jpg'

def inference(filename):
    with open(filename, 'rb') as f:
        response = session.post('https://cv-api.kakaobrain.com/pose', files=[('file', f)])
        return response.json()

def visualori(filename, annotations, threshold=0.2):
    for annotation in annotations:
        keypoints = np.asarray(annotation['keypoints']).reshape(-1, 3)
        low_confidence = keypoints[:, -1] < threshold
        keypoints[low_confidence, :] = [0, 0, 0]
        annotation['keypoints'] = keypoints.reshape(-1).tolist()

    # COCO API를 활용한 시각화
    plt.cla()
    plt.clf()
    plt.imshow(mpimg.imread(filename))
    plt.axis('off')

    coco = COCO()
    coco.dataset = {}
    coco.dataset = {
        "categories": [
            {
                "supercategory": "person",
                "id": 1,
                "name": "person",
                "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                              "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
                              "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
                "skeleton": [[1, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 7], [6, 8], [6, 12], [7, 9],
                             [7, 13], [8, 10], [9, 11], [12, 13], [14, 12], [15, 13], [16, 14], [17, 15]]
            }
        ]
    }
    coco.createIndex()
    coco.showAnns(annotations)
    plt.savefig('test.jpg')
    
    img = cv2.imread('test.jpg', cv2.IMREAD_UNCHANGED)
    cv2.waitKey(1)
    return img

def addpic(img, img2):
    newimg = np.hstack((img, img2))   
    cv2.imshow('Result', newimg)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()


with picamera.PiCamera() as camera:
    camera.start_preview()
    frame = 1
    GPIO.wait_for_edge(17, GPIO.FALLING)
    
    camera.capture(ORIGIN_PATH)
    orgimg = Image.open(ORIGIN_PATH)
    orgimg.save(ORIGIN_PATH, quality=86)           
    
    origin = inference(ORIGIN_PATH)
    img = visualori(ORIGIN_PATH, origin)

    while True:
        start = time.time()
        camera.capture(NEW_PATH)
        newimg = Image.open(NEW_PATH)
        newimg.save(NEW_PATH, quality=86)
        newdata = inference(NEW_PATH)
        img2 = visualori(NEW_PATH, newdata)

        try:
            #캡쳐 이미자
            # 파일로 이미지 입력시
            oleft_shoulder = origin[0]["keypoints"][6]
            oright_shoulder = origin[0]["keypoints"][7]
            onose =origin[0]["keypoints"][0]
            #t분에 한번씩 이미지 저장
            sleft_shoulder = newdata[0]["keypoints"][6]
            sright_shoulder = newdata[0]["keypoints"][7]
            snose =newdata[0]["keypoints"][0]
    
            lsd = oleft_shoulder - sleft_shoulder
            rsd = oright_shoulder - sright_shoulder
            nd = onose - snose

            if abs(lsd)>30 or abs(rsd)>30 or abs(nd)>30:
                print(" alarm")
                bang.play()
                addpic(img, img2)

            else:
                print(abs(lsd))
                print(abs(rsd))
                print(abs(nd))  
                camera.stop_preview()
                time.sleep(
                    int(5) - (time.time() - start)
                )
        except:
            quitm.play()
            break
        
