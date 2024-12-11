#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/11 19:34
# @Author  : Helenology
# @Site    : 
# @File    : ds.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
import sys
sys.path.append("../CD/")
sys.path.append("../../")
from cd import CD
from utils import K_tf


class DS:
    def __init__(self, p, q, train_list):
        """Initialization."""
        self.p = p                    # image height
        self.q = q                    # image width
        self.N = len(train_list)      # the number of train images
        self.train_list = train_list  # a list of train images in numpy.array
        self.CD_model = CD(p, q, train_list)  # a model for classical density estimation

    def new_location_filter(self, bandwidth, width=3, middle_point=(1, 1)):
        location_weight = np.zeros([width, width, 2])
        # weight by row
        for i in range(width):
            location_weight[i, 0:width, 0] = abs(i - middle_point[0]) / self.p
        # weight by column
        for j in range(width):
            location_weight[0:width, j, 1] = abs(j - middle_point[1]) / self.q
        # convert to tensor
        location_weight = tf.convert_to_tensor(location_weight, dtype=tf.float32)

        # product of row weight and column weight
        location_filter = K_tf(location_weight[:, :, 0], bandwidth) * K_tf(location_weight[:, :, 1], bandwidth)

        # normalize the sum to 1
        location_filter /= tf.reduce_sum(location_filter)

        location_filter = tf.reshape(location_filter, [width, width, 1, 1])
        return location_filter

    def DS_estimation(self, test_img, bandwidth, location_weight):
        size0 = location_weight.shape[0]  # filter height; for example = 3
        size1 = location_weight.shape[1]  # filter width; for example = 3
        constant_img = tf.zeros([self.p, self.q], dtype=tf.float32)
        DS_est = tf.zeros([self.p, self.q], dtype=tf.float32)
        n0 = int(self.p / size0)  # the number of non-overlapping filters
        n1 = int(self.q / size1)  # the number of non-overlapping filters
        for k in range(size0):
            for t in range(size1):

                for i in range(n0):
                    x = i * size0
                    for j in range(n1):
                        y = j * size1
                        constant_img[x:(x + size0), y:(y + size1)] = test_img[x + k, y + t]
                constant_CD_est = self.CD_model.CD_estimation(constant_img, bandwidth)
                constant_CD_est = tf.reshape(constant_CD_est, (1, self.p, self.q, 1))
                constant_DS_est = tf.nn.depthwise_conv2d(constant_CD_est, location_weight,
                                                         strides=[1, size0, size1, 1], padding='SAME')




