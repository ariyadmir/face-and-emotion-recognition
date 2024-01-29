import cv2
import matplotlib.pyplot as plt
import face_recognition
import numpy as np
from simple_facerec import SimpleFacerec
import os

# create instance for image encoding and face detection
face_rec = SimpleFacerec()

# get images for training
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder_path = os.path.join(script_dir, 'Images')

# encode images with SimpleFacerec
encodings = face_rec.load_encoding_images(image_folder_path)

print(f'encodings: {encodings}')

# video stream capture (0 - first web cam)
vid_capture = cv2.VideoCapture(0)

while True:
    # capture video frame
    return_value, frame = vid_capture.read()

    if not return_value:
        print("Error reading frame from the video capture.")
        break
    
    # detect if face captured on frame is a known face
    face_locs, face_names = face_rec.detect_known_faces(frame)
    print("Detected Faces:", face_names)

    for face_loc, face_name in zip(face_locs, face_names):
        # get the coordinates for each face in current frame (top, right, bottom, left)
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        # draw box over face and assign name on face
        cv2.rectangle(frame, pt1=(x1,y1), pt2=(x2,y2), color = (0,240,0), thickness = 4)
        cv2.putText(frame, text = face_name, org = (x1,y1-10), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0,240,0), thickness=2)
        

    # display video frame in new window
    cv2.imshow('Frame', frame)
    
    # wait 1 milisecond before refreshing and reading new frame
    key = cv2.waitKey(1)
    # if 'Esc' key is pressed, break loop and quit video frame
    if key == 27:
        break

# live video stream released and all windows closed
vid_capture.release()
cv2.destroyAllWindows()
