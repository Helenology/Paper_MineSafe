#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/11 11:03
# @Author  : Helenology
# @Site    : 
# @File    : cd.py
# @Software: PyCharm


import numpy as np
import tensorflow as tf
import sys
sys.path.append("../../")
from utils import load_and_preprocess_image


class CD:
    def __init__(self, p, q, train_list):
        """Initialization."""
        self.p = p                    # image height
        self.q = q                    # image width
        self.N = len(train_list)      # the number of train images
        self.train_list = train_list  # a list of train images in numpy.array

    def CD_estimation(self, test_img, bandwidth):
        """
        Kernel Density Estimation or Classical Density (CD) Estimation.
        :param test_img: new test image(s)
        :param bandwidth: the $h$ in the kernel smoothing term $K(Xi/h - x/h)$
        :return:
        """
        batch_size = 1
        if len(test_img.shape) == 3:
            batch_size = test_img.shape[0]
        CD_test = tf.zeros((batch_size, self.p, self.q), dtype=tf.float32)
        for i in range(self.N):  # $\sum_{i} 1 / (Nh) K { (Xi(s) - x) / h }$
            # read in a train image $Xi(s)$
            try:
                train_img = tf.constant(np.load(self.train_list[i]))
            except:
                train_img = load_and_preprocess_image(self.train_list[i], (1, self.p, self.q))
            # $1 / (Nh) K { (Xi(s) - x) / h }$
            tmp_tensor = 1 / tf.sqrt(2 * np.pi) * tf.exp(-(train_img - test_img) ** 2 / (2 * bandwidth ** 2))
            tmp_tensor = tmp_tensor / (self.N * bandwidth)
            CD_test += tmp_tensor
        return CD_test
