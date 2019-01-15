import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options = {
    'model': 'path to cfg file',
    'load': 'path to weight file',
    'threshold': 'threshold depending upon ur model accuracy',
##    'gpu': 0.0
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
##out = cv2.VideoWriter('output.avi', -1, 20.0, (1920,1080))
capture = cv2.VideoCapture("video file path")
print("start")
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

##def imb(frame):
##    results = tfnet.return_predict(frame)
##    print(results)
##    for color, result in zip(colors, results):
##        tl = (result['topleft']['x'], result['topleft']['y'])
##        br = (result['bottomright']['x'], result['bottomright']['y'])
##        label = result['label']
##        confidence = result['confidence']
##        text = '{}: {:.0f}%'.format(label, confidence * 100)
##        frame = cv2.rectangle(frame, tl, br, color, 5)
##        frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
##        cv2.imshow('frame', frame)
##        cv2.imwrite('8output.png',frame)
##
##im=cv2.imread("8.png")
##imb(im)


while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
##        print(results)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        if results!=[]:
##            print(results)
            cv2.imshow('frame', frame)
##            out.write(frame)
##        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
##out.release()
cv2.destroyAllWindows()
