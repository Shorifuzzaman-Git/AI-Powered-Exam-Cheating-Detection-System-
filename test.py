import cv2
import mediapipe as mp
from ultralytics import YOLO
import ultralytics

print("CV",cv2.__version__)
print("MP",mp.__version__)
print("YOLO",YOLO._version)
print("Ultra",ultralytics.__version__)