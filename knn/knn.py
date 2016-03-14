# -*- coding:utf-8 -*-

"""
@version: ??
@author: lynch
@contact: 
@site: https://github.com/lynch25
@software: PyCharm
@file: knn.py
@time: 2016/3/14 15:05
"""

from numpy import *
import operator
import os


def img2vector(filename):
    return_vector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vector[0, 32*i+j] = int(line_str[j])
    return return_vector


def handwriting_class_test():
    hw_labels = []
    training_file_list = os.listdir('trainingDigits')
    m_train = len(training_file_list)
    training_matrix = zeros((m_train, 1024))
    for i in range(m_train):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_matrix[i, :] = img2vector('trainingDigits/%s' % file_name_str)

    test_file_list = os.listdir('testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('testDigits/%s' % file_name_str)
        classifier_result = classify0(vector_under_test, training_matrix, hw_labels, 3)
        print "the classifier come back with: %d, the real answer is: %d" % (classifier_result, class_num_str)
        if classifier_result != class_num_str:
            error_count += 1.0
    print "\n the total number of errors is: %d" % error_count
    print "\n the total error rate is: %f" % (error_count/float(m_test))


def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_matrix = tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_matrix = diff_matrix**2
    sq_distances = sq_diff_matrix.sum(axis=1)
    distances = sq_distances**0.5
    sorted_distances = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distances[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

