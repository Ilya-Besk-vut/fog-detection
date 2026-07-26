import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from model_conf import build_model

# --- 1. Гиперпараметры и пути ---
NUM_EPOCHS = 30
INIT_LR = 1e-3
BS = 40
data = np.load('my_dataset.npz')['features']
labels = np.load('my_dataset.npz')['targets']
BASE_DIR = Path('../')

TARGET_NAMES = ["Clear", "Cloudy", "Fog"]

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.15, random_state=42
)

# --- Инициализация и компиляция модели ---
model = build_model(input_shape=(32, 32, 3), num_classes=len(TARGET_NAMES))

opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

# --- 4. Аугментация и обучение ---
aug = ImageDataGenerator(
    rotation_range=5,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.10,
    fill_mode="nearest"
)

checkpointer = ModelCheckpoint(
    filepath=BASE_DIR / 'best_model.keras', 
    verbose=1, 
    save_best_only=True
)

print("[INFO] training network...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=NUM_EPOCHS,
    shuffle=True,
    callbacks=[checkpointer]
)

# --- 5. Оценка результатов ---
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(
    testY.argmax(axis=1),
    predictions.argmax(axis=1), 
    target_names=TARGET_NAMES
))