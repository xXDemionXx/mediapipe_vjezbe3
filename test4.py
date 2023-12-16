# Age and Gender Detection

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# add image for analysis

img1 = cv2.imread('one_face.jpg')

# Get results

result = DeepFace.analyze(img1, actions=['age', 'gender', 'race', 'emotion'])
print(result[0]['age'])
print(result[0]['dominant_gender'])
print(result[0]['dominant_race'])
print(result[0]['dominant_emotion'])

