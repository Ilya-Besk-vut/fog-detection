from pathlib import Path
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import random

# --- 1. Конфигурация данных ---
# Задаем категории и папку, где они лежат
CATEGORIES = {
    '20251112_A':'fog',
    '20251112_B':'fog',
    '20251130_A':'fog',
    '20251130_B':'fog',
    '20250915_Cebin_mlha': 'fog',
    '20250916_Cebin_oblacno': 'cloudy',
    '20251106_Cebin_jasno': 'clear'
}

BASE_DIR = Path('../')
MAX_IMAGES_PER_CLASS = 4000
IMAGE_SIZE = (32, 32)

data = []
labels = []

samples = []

for FolderName, LabelName in CATEGORIES.items():
    FolderPath = BASE_DIR / FolderName
    CurrentImagePath = list(FolderPath.glob('*.png'))[:MAX_IMAGES_PER_CLASS:10]
    for path in CurrentImagePath:
        samples.append((path, LabelName))
    
random.seed(42)
random.shuffle(samples)

for path, label in samples:
    
    image = cv2.imread(str(path))
    image = cv2.resize(image, IMAGE_SIZE)
    data.append(image)

    labels.append(label)

data = np.array(data, dtype="float32") / 255.0
 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
    
print(len(data))

print(len(labels))

np.savez('my_dataset.npz', features=data, targets=labels)
print('data saved to my_dataset.npz')