import cv2
import numpy as np
from deepface import DeepFace

# creating instance of Haar-Cascade model for feature recognition
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# video stream capture (0 - first web cam)
vid_capture = cv2.VideoCapture(0)

while True:
    # capture video frame
    return_value, frame = vid_capture.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray_frame, 1.1, 4)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    frame_analysis = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection = False)
    cv2.putText(frame, text = frame_analysis[0]['dominant_emotion'], org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 240, 0), thickness = 2)

    #d isplay video frame in new window
    cv2.imshow('Frame', frame)
    
    # wait 1 milisecond before refreshing and reading new frame
    key = cv2.waitKey(1)
    # if 'Esc' key is pressed, break loop and quit video frame
    if key == 27:
        break

# live video stream released and all windows closed
vid_capture.release()
cv2.destroyAllWindows()
