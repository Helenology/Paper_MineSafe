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
from tensorflow.linalg import LinearOperatorKronecker, LinearOperatorFullMatrix
import matplotlib.pyplot as plt


class DS:
    def __init__(self, p, q, train_list):
        """Initialization."""
        self.p = p                    # image height
        self.q = q                    # image width
        self.N = len(train_list)      # the number of train images
        self.train_list = train_list  # a list of train images in numpy.array
        self.CD_model = CD(p, q, train_list)  # a model for classical density estimation

    def new_location_filter(self, bandwidth, size=3, middle_point=(1, 1)):
        location_weight = np.zeros([size, size, 2])
        # weight by row
        for i in range(size):
            location_weight[i, 0:size, 0] = abs(i - middle_point[0]) / self.p
        # weight by column
        for j in range(size):
            location_weight[0:size, j, 1] = abs(j - middle_point[1]) / self.q
        # convert to tensor
        location_weight = tf.convert_to_tensor(location_weight, dtype=tf.float32)
        # product of row weight and column weight
        location_filter = K_tf(location_weight[:, :, 0], bandwidth) * K_tf(location_weight[:, :, 1], bandwidth)
        # normalize the sum to 1
        location_filter /= tf.reduce_sum(location_filter)
        location_filter = tf.reshape(location_filter, [size, size, 1, 1])
        return location_filter

    def new_location_indicator(self, size, middle_point=(1, 1)):
        k = middle_point[0]
        t = middle_point[1]
        location_indicator = np.zeros((size, size))
        location_indicator[k, t] = 1
        return location_indicator

    def DS_estimation(self, test_img, bandwidth, size):
        batch_size = size * size
        n0 = int(self.p / size)                                   # the number of non-overlapping filters
        n1 = int(self.q / size)                                   # the number of non-overlapping filters
        DS_est = tf.zeros([self.p, self.q], dtype=tf.float32)     # the DS estimator
        sub_imgs = np.zeros((batch_size, n0, n1))                 # [batch_size, n0, n1]
        location_ones = tf.ones((size, size), dtype=tf.float32)   # [size, size]
        location_inds = np.zeros((batch_size, size, size))        # [batch_size, size, size]
        # obtain sub-images
        for k in range(size):
            for t in range(size):
                i = k * size + t
                location_inds[i] = self.new_location_indicator(size, (k, t))
                location_ind = tf.reshape(tf.constant(location_inds[i], dtype=tf.float32), (size, size, 1, 1))
                sub_img = tf.nn.depthwise_conv2d(tf.reshape(test_img, (1, self.p, self.q, 1)), location_ind,
                                                 strides=[1, size, size, 1], padding='SAME')
                sub_imgs[i] = sub_img.numpy().reshape((n0, n1))
        # Kronecker Product: upsampling sub-images
        location_inds = tf.constant(location_inds, dtype=tf.float32)    # [batch_size, size, size]
        sub_imgs = tf.constant(sub_imgs, dtype=tf.float32)              # [batch_size, n0, n1]
        operator_1 = LinearOperatorFullMatrix(sub_imgs)                 # [batch_size, n0, n1]
        operator_2 = LinearOperatorFullMatrix(location_ones)            # [size, size]
        operator = LinearOperatorKronecker([operator_1, operator_2])    # [batch_size, p, q]
        constant_img = operator.to_dense()                              # [batch_size, p, q]
        constant_img = tf.cast(constant_img, dtype=tf.float32)          # [batch_size, p, q]
        # CD Estimation
        constant_CD_est = self.CD_model.CD_estimation(constant_img, bandwidth)
        constant_CD_est = tf.reshape(constant_CD_est, (batch_size, self.p, self.q, 1))  # [batch_size, p, q]

        # Double Smoothing
        for k in range(size):
            for t in range(size):
                location_weight = self.new_location_filter(bandwidth, size, (k, t))  # [size, size, 1, 1]
                CD_est_kt = tf.reshape(constant_CD_est[k * size + t], (1, self.p, self.q, 1))
                DS_est_kt = tf.nn.depthwise_conv2d(CD_est_kt, location_weight,
                                                   strides=[1, size, size, 1], padding='SAME')
                DS_est_kt = tf.squeeze(DS_est_kt)                                    # [n0, n1]
                # Kronecker Product
                operator_3 = LinearOperatorFullMatrix(DS_est_kt)                     # [n0, n1]
                operator_4 = LinearOperatorFullMatrix(location_inds[k * size + t])   # [size, size]
                operator = LinearOperatorKronecker([operator_3, operator_4])         #
                DS_est += operator.to_dense()                                        # [p, q]
        return DS_est

