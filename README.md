# Detector

## Progress
- [x] face detector
- [x] hand detector
- [] gesture detector
- [] gaze detector

## Requirement

I strongly recommend using a CONDA environment to deploy this project to avoid conflicts between Python-dependent bags.

### Face detector
1. OpenCV : `pip install opencv-python`
2. dlib : `pip install dlib`

### Hand detector
1. OpenCV : `pip install opencv-python`
2. mediapipe : `pip install mediapipe`

## Usage
### Only face detector
`python face_detector.py`

### Only hand detector
`python hands_detector.py`

### Face and hand detector
`python detector.py`

### Gesture detector
You can run the below command to call your local camera to test the detector
```
python gesture_detector.py
```
By now, I just achieved the detection of whether the finger is straight or not, by detecting whether the four key points of each finger are collinear.

<p align="center">
  <img src="https://google.github.io/mediapipe/images/mobile/hand_landmarks.png" width="500">
</p>

### Gaze detector
TBA
## Reference
- [__`Face detector by opencv`__](https://www.datacamp.com/community/tutorials/face-detection-python-opencv)
- [__`Hands detector by opencv`__](https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/)