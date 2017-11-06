import pickle
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import scipy as sp
import random, time
from keras.models import load_model

import data
from segmenter import Segmenter
import metrics as met
from testing import *
# from visualize import visualise_results

if __name__ == '__main__':
    train_images, train_labels = data.read_train_images_and_labels()

    # kfold_testing(Segmenter, train_images, train_labels, splits=5)
    single_testing(Segmenter, train_images, train_labels)