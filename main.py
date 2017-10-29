import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt

import data
from segmenter import Segmenter
from visualize import visualise_results

if __name__ == '__main__':
    train_images, train_labels = data.read_train_images_and_labels()
    visualise_results(train_labels, train_images, randomise=False)
    exit()
    segmenter = Segmenter(learing_rate=0.001)

    segmenter.train(train_images, train_labels)

    # segmenter.train(np.asarray([train_images[0]]), np.asarray([train_labels[0]]))
    # segmenter.save_model(path='model1.h5')

    # test_images = data.read_test_images()
    # results = segmenter.infer(test_images)

