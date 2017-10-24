import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import data
from segmenter import Segmenter

def visualise_results(images, originals, randomise=True):
    if randomise:
        c = list(zip(images, originals))
        random.shuffle(c)
        images, originals = zip(*c)

    for image, original in zip(images, originals):
        f, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(original)
        ax[1].imshow(image)
        plt.show()

if __name__ == '__main__':
    train_images, train_labels = data.read_train_images_and_labels()

    segmenter = Segmenter(learing_rate=0.001)

    segmenter.train(np.asarray([train_images[0]]), np.asarray([train_labels[0]]))
    # segmenter.save_model(path='model1.h5')

    # test_images = data.read_test_images()
    # results = segmenter.infer(test_images)

