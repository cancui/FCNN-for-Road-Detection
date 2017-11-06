import numpy as np 
import scipy as sp 
from scipy.ndimage import imread
import glob, os.path, random

''' Convert label image into a matrix of 2-member onehot vectors. Red is not road, green is road '''
def image_to_label(image):
    image[:,:,0] = 1
    image[np.where(image[:,:,2] == 255)] = np.asarray([0, 1, 0])
    image = np.delete(image, 2, axis=2)
    return image

''' Convert matrix of onehot vectors to image that can be displayed. Red is not road, green is road '''
def label_to_image(label):
    added = np.full((*label.shape[0:2], 1), 0)
    image = np.concatenate((label, added), axis=2)
    image *= 255
    return image.astype('uint8')

''' Trim an image or label to a specific size '''
def trimmer(image, max=False):
    if max:
        image = image[-370:,-1226:,:] #crop for largest possible
    else:
        # image = image[-288:,-1152:,:] #9*32, 9*128
        image = image[-320:,-1216:,:] #64*5, 64*19
    return image

''' Convert predictions/labels from one hot to indices '''
def preprocess_for_metrics(predicted):
    return predicted[:,:,:,0]

def find_min(images):
    smallest_height = 10000
    smallest_width = 10000

    for image in images:
        shape = image.shape
        if shape[0] < smallest_height:
            smallest_height = shape[0]
        if shape[1] < smallest_width:
            smallest_width = shape[1]

    print('Smallest height:{} smallest width:{}'.format(smallest_height, smallest_width))

def get_label_path(image_path):
    if os.path.isfile('visualize.py'): 
        path_parts = image_path.split('_')
        del(path_parts[1])
        label_path = '{}_data/training/gt_image_{}_road_{}'.format(*path_parts)
    else:
        path_parts = image_path.split('_')
        del(path_parts[0])
        label_path = 'gt_image_{}_road_{}'.format(*path_parts)

    return label_path

def read_train_images_and_labels():
    images, labels = [], []
    label_paths = []

    if os.path.isfile('visualize.py'): path = "road_data/training/image_2/"
    else: path = "image_2/"

    for image_path in glob.glob(path+"*.png"):
        image = imread(image_path)
        image = trimmer(image)
        images.append(image)

        label_path = get_label_path(image_path)
        label = imread(label_path)

        label = image_to_label(label)
        label = trimmer(label)
        labels.append(label)

    assert(len(images) == len(labels))
    print('Read {} training images'.format(len(images)))

    images, labels = np.asarray(images), np.asarray(labels)
    print(images.shape, labels.shape)

    return images, labels

def read_test_images(randomise=True):
    images = [trimmer(imread(image_path), max=False) for image_path in glob.glob("road_data/testing/image_2/*.png")]

    if randomise: random.shuffle(images)

    print('Read {} testing images'.format(len(images)))
    images = np.asarray(images)
    return images

if __name__ == '__main__':
    print('Running data.py')
    read_train_images_and_labels()
    read_test_images()