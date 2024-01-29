# Face and Emotion Recognition

This repository contains two Python scripts for face recognition and emotion recognition using different approaches.

## Face Recognition

### Dependencies
- OpenCV: An open-source computer vision library used for image and video processing.
- face_recognition: A library for face recognition that uses dlib and deep learning.

### Usage

1. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

2. Prepare a folder named 'Images' in the same directory as the script.

3. Place images of known faces in the 'Images' folder for training.

4. Run the script:

    ```bash
    python FaceRecognition.py
    ```

5. The script will capture video from the webcam and recognize known faces.

## Emotion Recognition

### Dependencies
- OpenCV: An open-source computer vision library used for image and video processing.
- DeepFace: A deep learning facial analysis library that provides pre-trained models for facial emotion detection.

### Usage

1. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

2. Download the Haar cascade XML file (haarcascade_frontalface_default.xml) for face detection from the OpenCV GitHub repository.
  
4. Run the script:

    ```bash
    python EmotionRecognition.py
    ```

5. The script will capture video from the webcam, detect faces, and display the dominant emotion on each face.

## Notes

- Both scripts use live video stream capture (0 - the first webcam). The video source can be modified as needed.

- Ensure the necessary dependencies are installed before running the scripts.

- The face recognition script requires a 'Images' folder with images of known faces for training.

- The emotion recognition script uses Haar-Cascade for face detection and DeepFace for emotion analysis.

- Press 'Esc' key to exit the video stream in both scripts.

## Acknowledgements

- OpenCV: https://github.com/opencv/opencv
- face_recognition: https://github.com/ageitgey/face_recognition
- DeepFace: https://github.com/serengil/deepface

