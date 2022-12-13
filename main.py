import os
import numpy as np
import cv2
import time

import mediapipe as mp
running = True
xp,yp = 0,0
fingers = []
canvas = np.zeros((480,640,3),np.uint8)
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while running:
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        h,x,y = image.shape
        lmList = []
        xl=[]
        yl=[]


        # x,y = results.lm
        for id,Lm in enumerate(hand_landmarks.landmark):
            cx,cy = int(x*Lm.x),int(h*Lm.y)
            xl.append(cx)
            yl.append(cy)
            lmList.append([id,cx,cy])
            cv2.circle(image,(cx,cy),10,(255,0,0),cv2.FILLED)
        mins = (min(xl)-20,min(yl)-20)
        maxs = (max(xl)+20,max(yl)+20)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS)
        if len(lmList) > 1:
          x1,y1 = lmList[8][1],lmList[8][2]
          if y1 < lmList[6][2] and lmList[12][2] > lmList[10][2]:
            if xp == 0 and yp==0:
              xp,yp = x1,y1
            # cv2.line(image,(xp,yp),(x1,y1),(255,0,0),10)
            cv2.line(canvas,(xp,yp),(x1,y1),(255,255,255),20)


          # if(lmList[8][2] < lmList[6][2] and lmList[20][2] < lmList[18][2] and lmList[12][2] > lmList[10][2] and lmList[16][2] > lmList[14][2]):
          #   print("Yo")
          xp,yp = x1,y1
        # imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # _,imageInv = cv2.threshold(imageGray,50,255,cv2.THRESH_BINARY_INV)
        # imageInv = cv2.cvtColor(imageInv,cv2.COLOR_GRAY2BGR)

        # image = imageInv
        # image = cv2.bitwise_and(image,imageInv2)
      
      
        # print(imageInv2.shape)

          if lmList[12][2] < lmList[10][2] and lmList[8][2] < lmList[6][2]:
            cv2.line(canvas,(xp,yp),(x1,y1),(0,0,0),80)

            
        cv2.rectangle(image,mins,maxs,(255,0,0),4)
      # image = cv2.addWeighted(image,0.5,canvas,0.5,0)
        

            
    image = cv2.bitwise_or(image,canvas)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Frame', cv2.flip(image, 1))
    # cv2.imshow('canvas', cv2.flip(canvas,1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()