import numpy as np
import scipy as sp
import random

import data
from segmenter import Segmenter
# from visualize import visualise_results

def test_model(model, test_images, randomise=True):
    from visualize import visualise_results
    results = model.infer(test_images)
    visualise_results(results, test_images, randomise=randomise, num=5)

if __name__ == '__main__':
    train_images, train_labels = data.read_train_images_and_labels()

    # from visualize import visualise_results
    # visualise_results(train_labels, train_images, randomise=False, num=1)
    # exit()

    segmenter = Segmenter()

    segmenter.train(train_images, train_labels)

    # segmenter.load_model('saved_models/vgg3.h5')
    # test_images = data.read_test_images()
    # for i in range(test_images.shape[0]//5):
    #     test_model(segmenter, test_images[int(5*i):int(5*(i+1))])
