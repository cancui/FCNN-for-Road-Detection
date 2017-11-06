import pickle
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import scipy as sp
import random, time
from keras.models import load_model

import data
from segmenter import Segmenter
import metrics as met
# from visualize import visualise_results

def test_model(model, test_images, randomise=True):
    from visualize import visualise_results
    results = model.infer(test_images)
    visualise_results(results, test_images, randomise=randomise, num=5)

def kfold_testing(segmenter_function, all_images, all_labels, splits=5):
    kfold = KFold(n_splits=splits, shuffle=True)
    
    p_acc_sum, m_acc_sum, m_IU_sum, fw_IU_sum = 0, 0, 0, 0
    save_string = ''

    for i, (train_index, test_index) in enumerate(kfold.split(all_images)):
        print('FOLD {} ######################'.format(i+1))
        train_images, test_images = all_images[train_index], all_images[test_index]
        train_labels, test_labels = all_labels[train_index], all_labels[test_index]

        segmenter = segmenter_function()
        segmenter.train(train_images, train_labels)
        p_acc, m_acc, m_IU, fw_IU = segmenter.test_model(test_images, test_labels)

        fold_string = 'For this fold:    pixel_acc:{}, mean_acc:{}, mean_IU:{}, freq_IU:{}\n'.format(p_acc, m_acc, m_IU, fw_IU)
        print(fold_string)
        save_string += fold_string

        p_acc_sum += p_acc
        m_acc_sum += m_acc
        m_IU_sum += m_IU
        fw_IU_sum += fw_IU

    p_acc_sum, m_acc_sum, m_IU_sum, fw_IU_sum = p_acc_sum/splits, m_acc_sum/splits, m_IU_sum/splits, fw_IU_sum/splits
    
    overall_string = 'Overall:    pixel_acc:{}, mean_acc:{}, mean_IU:{}, freq_IU:{}\n'.format(p_acc_sum, m_acc_sum, m_IU_sum, fw_IU_sum)
    print(overall_string)
    save_string += overall_string

    with open('results_' + time.strftime("%m-%dth_%H:%M:%S") + '.txt', 'w') as f:
        f.write(save_string)

    return p_acc_sum, m_acc_sum, m_IU_sum, fw_IU_sum

def kfold_data_creation(all_images, all_labels, splits=5):
    kfold = KFold(n_splits=splits, shuffle=True)

    for i, (train_index, test_index) in enumerate(kfold.split(all_images)):
        print('FOLD {} ######################'.format(i+1))
        train_images, test_images = all_images[train_index], all_images[test_index]
        train_labels, test_labels = all_labels[train_index], all_labels[test_index]

        fold_data = {
            'train_images' : train_images,
            'train_labels' : train_labels,
            'test_images' : test_images,
            'test_labels' : test_labels
        }

        with open('fold{}_data.p'.format(i+1), 'wb') as f:
            pickle.dump(fold_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickled_fold(num):
    path = 'fold{}_data.p'.format(num)
    with open(path, 'rb') as f:
        fold_data = pickle.load(f)

    return fold_data['train_images'], fold_data['train_labels'], fold_data['test_images'], fold_data['test_labels']

def single_testing(segmenter_function, all_images, all_labels):
    train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2, shuffle=True)

    segmenter = segmenter_function()
    segmenter.train(train_images, train_labels)
    p_acc, m_acc, m_IU, fw_IU = segmenter.test_model(test_images, test_labels)
    result_string = 'For this fold:    pixel_acc:{}, mean_acc:{}, mean_IU:{}, freq_IU:{}\n'.format(p_acc, m_acc, m_IU, fw_IU)
    print(result_string)

    with open('results_' + time.strftime("%m-%dth_%H:%M:%S") + '.txt', 'w') as f:
        f.write(result_string)

    return p_acc, m_acc, m_IU, fw_IU

def test_folds(num=5):
    p_acc_sum, m_acc_sum, m_IU_sum, fw_IU_sum = 0, 0, 0, 0
    save_string = ''

    for i in range(num):
        segmenter = Segmenter()
        segmenter.model = load_model('saved_models/fold{}.h5'.format(i+1))
        train_images, train_labels, test_images, test_labels = load_pickled_fold(i+1)

        p_acc, m_acc, m_IU, fw_IU = segmenter.test_model(test_images, test_labels)
        result_string = 'For this fold:    pixel_acc:{}, mean_acc:{}, mean_IU:{}, freq_IU:{}\n'.format(p_acc, m_acc, m_IU, fw_IU)
        print(result_string)
        save_string += result_string

        p_acc_sum += p_acc
        m_acc_sum += m_acc
        m_IU_sum += m_IU
        fw_IU_sum += fw_IU

    p_acc_sum, m_acc_sum, m_IU_sum, fw_IU_sum = p_acc_sum/num, m_acc_sum/num, m_IU_sum/num, fw_IU_sum/num
    
    overall_string = 'Overall:    pixel_acc:{}, mean_acc:{}, mean_IU:{}, freq_IU:{}\n'.format(p_acc_sum, m_acc_sum, m_IU_sum, fw_IU_sum)
    print(overall_string)
    save_string += overall_string

    with open('results_' + time.strftime("%m-%dth_%H:%M:%S") + '.txt', 'w') as f:
        f.write(save_string)

    return p_acc_sum, m_acc_sum, m_IU_sum, fw_IU_sum

if __name__ == '__main__':
    # train_images, train_labels = data.read_train_images_and_labels()

    # kfold_data_creation(train_images, train_labels, splits=5)

    # kfold_testing(Segmenter, train_images, train_labels, splits=5)
    # single_testing(Segmenter, train_images, train_labels)
    # exit()

    # from visualize import visualise_results
    # visualise_results(train_labels, train_images, randomise=False, num=1)
    # exit()



    # train_images, train_labels, test_images, test_labels = load_pickled_fold(4)
    # segmenter = Segmenter(save_name='fold4.h5')
    # segmenter.train(train_images, train_labels, test_images, test_labels)

    # p_acc, m_acc, m_IU, fw_IU = segmenter.test_model(test_images, test_labels)
    # result_string = 'For this fold:    pixel_acc:{}, mean_acc:{}, mean_IU:{}, freq_IU:{}\n'.format(p_acc, m_acc, m_IU, fw_IU)
    # print(result_string)

    # with open('results_' + time.strftime("%m-%dth_%H:%M:%S") + '.txt', 'w') as f:
    #     f.write(result_string)



    test_folds()


    # segmenter.load_model('saved_models/vgg4.h5')

    # print(segmenter.test_model(train_images[:3], train_labels[:3]))
    # exit()

    # test_images = data.read_test_images()
    # for i in range(test_images.shape[0]//5):
    #     test_model(segmenter, test_images[int(5*i):int(5*(i+1))])