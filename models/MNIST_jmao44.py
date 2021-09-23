from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import MaxPooling2D, Conv2D
from keras.layers.normalization import BatchNormalization
import os

def MNIST_jmao44(use_softmax=True, rel_path='./'):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(10))

    if use_softmax:
        model.add(Activation('softmax'))

    model.load_weights(
        os.path.join('%smodels/weights' % rel_path, "MNIST_jmao44.keras_weights.h5")
    )
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.name = 'MNIST_jmao44'
    return model