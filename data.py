import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt
from scipy.ndimage import imread
import glob

# def read_train_images_and_labels():
#     images, labels = [], []

#     for image_path in glob.glob("road_data/training/image_2/*.png"):
#         image = imread(image_path)
#         images.append(image)

#     for image_path in glob.glob("road_data/training/gt_image_2/*.png"):
#         if 'lane' in image_path:
#             continue

#         image = imread(image_path)
#         labels.append(image)

#     assert(len(images) == len(labels))
#     print('Read {} training images'.format(len(images)))

#     return images, labels

def get_label_path(image_path):
    path_parts = image_path.split('_')
    del(path_parts[1])
    label_path = '{}_data/training/gt_image_{}_road_{}'.format(*path_parts)
    return label_path
    # print(image_path)
    # print(label_path)

def read_train_images_and_labels():
    images, labels = [], []
    label_paths = []

    for image_path in glob.glob("road_data/training/image_2/*.png"):
        image = imread(image_path)
        images.append(image)

        label_path = get_label_path(image_path)
        # print(image_path)
        # print(label_path)
        labels.append(imread(label_path))

    # for image_path in label_paths:
    #     image = imread(image_path)
    #     labels.append(image)

    assert(len(images) == len(labels))
    print('Read {} training images'.format(len(images)))

    return images, labels

def read_test_images():
    images = [imread(image_path) for image_path in glob.glob("road_data/testing/image_2/*.png")]
    print('Read {} testing images'.format(len(images)))
    return images

if __name__ == '__main__':
    print('Running data.py')
    read_train_images_and_labels()
    read_test_images()