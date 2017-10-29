import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras import optimizers

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def fully_conv_net(learing_rate=0.001):
    model = Sequential(layers=[
        Conv2D(8, (7,7), strides=(1,1), padding='same', activation='relu', input_shape=(None, None, 3)),
        Conv2D(8, (3,3), strides=(1,1), padding='same', activation='relu'),
        Conv2D(3, (1,1), strides=(1,1), padding='same', activation='relu'),
    ])

    # model.compile(optimizer=optimizers.Adam(), loss='mse', metrics=[])
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=[])
    #categorical_crossentropy
    return model

class Segmenter(object):
    def __init__(self, learing_rate=0.001):
        self.model = fully_conv_net(learing_rate=learing_rate)

    def train(self, features, labels):
        self.model.fit(features, labels, batch_size=1, epochs=15)

    def infer(self, features):
        return

    def save_model(self, path):
        return

    def load_model(self, path):
        return

if __name__ == '__main__':
    print('Running segmenter.py')
    segmenter = Segmenter()