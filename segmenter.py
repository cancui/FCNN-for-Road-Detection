import keras
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras import optimizers

import numpy as np
import scipy as sp

import time

def fully_conv_net(learing_rate=0.001):
    
    # model = Sequential(layers=[
    #     Conv2D(8, (7,7), strides=(1,1), padding='same', activation='relu', input_shape=(None, None, 3)),
    #     Conv2D(8, (5,5), strides=(1,1), padding='same', activation='relu'),
    #     Conv2D(2, (3,3), strides=(1,1), padding='same', activation='relu'),
    # ])

    model = Sequential(layers=[
        Conv2D(16, (7,7), strides=(2,2), padding='same', activation='relu', input_shape=(None, None, 3)),
        # MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (7,7), strides=(2,2), padding='same', activation='relu'),
        # MaxPooling2D(pool_size=(3, 3)),
        Conv2D(64, (5,5), strides=(1,1), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(128, (5,5), strides=(1,1), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),

        # Conv2D(2, (1,1), strides=(1,1), padding='same', activation='relu'),
        Conv2DTranspose(2, (9*4*2, 9*4*2), strides=(9*4, 9*4), padding='same')
    ])

    # model.compile(optimizer=optimizers.Adam(), loss='mse', metrics=[])
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=[])

    return model

class Segmenter(object):
    def __init__(self, learing_rate=0.001):
        self.model = fully_conv_net(learing_rate=learing_rate)

    def train(self, features, labels):
        self.model.fit(features, labels, batch_size=1, epochs=5)

        save_path = 'saved_models/'+ time.strftime("%m-%dth_%H:%M:%S") +'.h5'
        self.save_model(save_path)

    def infer(self, features):
        results = self.model.predict(features)
        
        binarized = np.zeros(results.shape)
        binarized[np.where(results[:,:,:,0] >= results[:,:,:,1])] = np.asarray([255, 0])
        binarized[np.where(results[:,:,:,0] < results[:,:,:,1])] = np.asarray([0, 255])

        print('Inference created output with shape {}'.format(binarized.shape))

        return binarized.astype('uint8')

    def save_model(self, path):
        self.model.save(path)
        print('Saved model to {}'.format(path))

    def load_model(self, path):
        self.model = load_model(path)
        print('Loaded model at {}'.format(path))

if __name__ == '__main__':
    print('Running segmenter.py')
    segmenter = Segmenter()