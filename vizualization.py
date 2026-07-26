#from tensorflow.keras.models import load_model
import tensorflow
from tensorflow.keras import models
import numpy as np
import cv2
from skimage import exposure
from pathlib import Path

BASE_DIR = Path('../')
IMAGE_SIZE = (32, 32)

model = tensorflow.keras.models.load_model("../best_model.keras")

print("NET LOADED")
#model.summary()

TARGET_NAMES = ["Clear", "Cloudy", "Fog"]

cap = cv2.VideoCapture(0)

FolderPath = BASE_DIR / '20251130_C'
ImPath = list(FolderPath.glob('*png'))
Image = cv2.imread(str(ImPath[1625]))

data = cv2.resize(Image, IMAGE_SIZE)
data = np.array(data, dtype = 'float32')/255.0
data = np.expand_dims(data, axis=0)
pred = model.predict(data)

print (pred)


cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Window', 500,500)


cv2.imshow('Window',Image)
cv2.waitKey(0)
cv2.destroyAllWindows
