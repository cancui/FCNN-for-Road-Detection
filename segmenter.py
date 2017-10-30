import numpy as np
import scipy as sp
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dropout, Add, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras import optimizers

import time

import data
import metrics as met

def fully_conv_InceptionV3():
    input_shape = (None, None, 3)
    input_image = Input(shape=input_shape)

    # 21,802,784 parameters on its own 
    inception = InceptionV3(include_top=True, weights='imagenet', input_tensor=input_image, input_shape=input_shape)

    # 20,861,480 parameters on its own
    # xception = Xception(include_top=False, weights='imagenet', input_tensor=input_image, input_shape=input_shape)
    # print(xception.summary())

    for layer in inception.layers:
        layer.trainable = False

    # plot_model(inception, 'inception_v3.png')
    print(inception.summary())

def fully_conv_VGG16():
    input_shape = (None, None, 3)
    input_image = Input(shape=input_shape)

    # 14,714,688 parameters on its own
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_image, input_shape=input_shape)
    
    for layer in vgg16.layers:
        layer.trainable = False

    pool3 = vgg16.layers[10]
    pool4 = vgg16.layers[14]
    pool5 = vgg16.layers[18]

    skip3 = BatchNormalization()( Conv2D(2, (1,1), strides=(1, 1))(pool3.output) )
    skip4 = BatchNormalization()( Conv2D(2, (1,1), strides=(1, 1))(pool4.output) )
    skip5 = BatchNormalization()( Conv2D(2, (1,1), strides=(1, 1))(pool5.output) )
    
    upsampled_2x = Conv2DTranspose(2, kernel_size=(4,4), strides=(2, 2), padding='same')(skip5)
    upsampled_2x = BatchNormalization()(upsampled_2x)
    pre_fcn16 = Add()([upsampled_2x, skip4])

    upsampled_4x = Conv2DTranspose(2, kernel_size=(4,4), strides=(2, 2), padding='same')(pre_fcn16)
    upsampled_4x = BatchNormalization()(upsampled_4x)
    pre_fcn8 = Add()([upsampled_4x, skip3])

    fcn8 = Conv2DTranspose(2, kernel_size=(16,16), strides=(8, 8), padding='same')(pre_fcn8)

    model = Model(inputs=[vgg16.input], outputs=[fcn8])
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss='binary_crossentropy')

    return model

def fully_conv_net():
    
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
    def __init__(self):
        self.model = fully_conv_VGG16()

    def train(self, features, labels):
        self.model.fit(features, labels, batch_size=1, epochs=1)

        save_path = 'saved_models/'+ time.strftime("%m-%dth_%H:%M:%S") +'.h5'
        self.save_model(save_path)

    def infer(self, features):
        results = self.model.predict(features)
        
        binarized = np.zeros(results.shape)
        binarized[np.where(results[:,:,:,0] >= results[:,:,:,1])] = np.asarray([1, 0])
        binarized[np.where(results[:,:,:,0] < results[:,:,:,1])] = np.asarray([0, 1])

        print('Inference created output with shape {}'.format(binarized.shape))

        return binarized.astype('uint8')

    def test_model(self, test_features, ground_truth):
        results = self.infer(test_features)
        results = data.preprocess_for_metrics(results)
        ground_truth = data.preprocess_for_metrics(ground_truth)

        length = ground_truth.shape[0]
        p_acc, m_acc, m_IU, fw_IU = 0, 0, 0, 0
        for i in range(length):
            p_acc += met.pixel_accuracy(results[i], ground_truth[i])
            m_acc += met.mean_accuracy(results[i], ground_truth[i])
            m_IU +=  met.mean_IU(results[i], ground_truth[i])
            fw_IU += met.frequency_weighted_IU(results[i], ground_truth[i])
        p_acc, m_acc, m_IU, fw_IU = p_acc/length, m_acc/length, m_IU/length, fw_IU/length

        return p_acc, m_acc, m_IU, fw_IU

    def save_model(self, path):
        self.model.save(path)
        print('Saved model to {}'.format(path))

    def load_model(self, path):
        self.model = load_model(path)
        print('Loaded model at {}'.format(path))

if __name__ == '__main__':
    print('Running segmenter.py')
    # segmenter = Segmenter()

    # plot_model(fully_conv_VGG16(), 'vgg16_mod.png')

    fully_conv_InceptionV3()