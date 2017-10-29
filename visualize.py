import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt

import data
from segmenter import Segmenter

def highlight_image(base_image, highlight, intensity=5.0):
    area = base_image[np.where(highlight[:,:,1] == 255)]
    area[:,0] = area[:,0] / intensity
    area[:,2] = area[:,2] / intensity
    area = area.astype(int)
    
    base_image[np.where(highlight[:,:,1] == 255)] = area
    return base_image.astype('uint8')

def visualise_results(results, originals, randomise=True, num=None):
    if randomise:
        c = list(zip(results, originals))
        random.shuffle(c)
        results, originals = zip(*c)

    for i, (result, original) in enumerate(zip(results, originals)):
        result = data.label_to_image(result)

        f, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].imshow(original)
        ax[1].imshow(highlight_image(original, result))
        ax[2].imshow(result)
        plt.show()

        if num and i == num:
            return