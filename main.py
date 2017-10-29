import numpy as np
import scipy as sp
import random

import data
from segmenter import Segmenter
# from visualize import visualise_results

def test_model(model, test_images):
    from visualize import visualise_results
    results = model.infer(test_images)
    visualise_results(results, test_images, randomise=False, num=5)

if __name__ == '__main__':
    train_images, train_labels = data.read_train_images_and_labels()
    # visualise_results(train_labels, train_images, randomise=False)
    # exit()

    segmenter = Segmenter(learing_rate=0.001)

    # segmenter.train(train_images, train_labels)

    segmenter.load_model('saved_models/net4.h5')
    test_model(segmenter, train_images[:3])

    # test_images = data.read_test_images()
    # results = segmenter.infer(test_images)

