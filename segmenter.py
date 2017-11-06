import numpy as np
import scipy as sp
import keras
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dropout, Add, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import time

import data
import metrics as met

def fully_conv_VGG16(metrics=None):
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
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=metrics)

    return model

class Segmenter(object):
    def __init__(self, save_name=''):
        self.metrics = [keras.metrics.binary_accuracy]
        self.model = fully_conv_VGG16(self.metrics)
        self.save_name = save_name

        reduce_lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, verbose=1)
        model_checkpoint = ModelCheckpoint('saved_models/' + self.save_name + '_weights.epoch{epoch:02d}-acc{val_loss:.4f}.h5', monitor='val_loss', verbose=1, save_best_only=True)

        self.callbacks = [reduce_lr_plateau, early_stop, model_checkpoint]

    def train(self, features, labels, val_features=None, val_labels=None):
        if val_features is None or val_labels is None: val_data = None
        else: val_data = (val_features, val_labels)

        self.model.fit(features, labels, batch_size=1, epochs=40, validation_data=val_data, callbacks=self.callbacks)

        if self.save_name: save_path = 'saved_models/'+ self.save_name
        else: save_path = 'saved_models/'+ time.strftime("%m-%dth_%H:%M:%S") +'.h5'
        self.save_model(save_path)

    def infer(self, features):
        results = self.model.predict(features, batch_size=1)
        
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
