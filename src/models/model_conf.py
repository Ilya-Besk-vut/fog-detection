from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Activation,
    BatchNormalization, Flatten, Dropout, Dense
)

def build_model(input_shape=(32, 32, 3), num_classes=3):
    """
    Создает и возвращает модель CNN для классификации дорожных знаков.
    """
    model = Sequential()
    chanDim = -1

    # Блок 1: CONV => RELU => BN => POOL
    model.add(Conv2D(8, (5, 5), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Блок 2: (CONV => RELU => BN) * 2 => POOL
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Блок 3: (CONV => RELU => BN) * 2 => POOL
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Полносвязный слой 1 (FC => RELU => BN => DROPOUT)
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Полносвязный слой 2 (второй Flatten убран!)
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Выходной классификатор Softmax
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    return model