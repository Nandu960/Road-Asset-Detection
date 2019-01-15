import cv2
import numpy as np

cap=cv2.VideoCapture('ped_video.mp4')

human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

while True:

    ret, frame = cap.read()
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    human = human_cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in human:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,220), 3)

        cv2.imshow('video' , frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
