#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/30 11:10
# @Author  : Helenology
# @Site    :
# @File    : gpa.py.py
# @Software: PyCharm


import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import time
import sys
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from tensorflow.keras.layers import AveragePooling2D
import cv2
import re
import pandas as pd

sys.path.append("./models/GPA")
from useful_functions import *


class GPA:
    def __init__(self, G, p, q, train_list, second_smooth=True):
        # basic parameters
        self.G = G
        self.p = p
        self.q = q
        self.train_list = train_list
        self.N0 = len(train_list)

        # compute optimal bandwidth
        self.alpha = np.log(p * q) / np.log(self.N0)
        self.bandwidth, self.bandwidth_star = compute_optimal_bandwidths(self.N0, self.alpha)

        # generate grid points {x_g^*}
        rng = np.random.default_rng(seed=0)
        tick_list = rng.random(size=G)  # tick list
        self.tick_tensor = tf.concat([tf.ones([1, p, q]) * tick for tick in tick_list], axis=0)

        # compute GPA matrix
        self.gpa_matrix, self.train_time = self.compute_GPA_matrix(second_smooth=second_smooth)

    def compute_GPA_matrix(self, second_smooth=True):
        gpa_matrix = tf.zeros([self.G, self.p, self.q])
        t1 = time.time()
        for i in range(self.N0):
            test_img = load_and_preprocess_image(self.train_list[i], (self.p, self.q))
            tmp_tensor = (1 / (self.N0 * self.bandwidth)) * (1 / tf.sqrt(2 * np.pi)) * tf.exp(
                -(test_img - self.tick_tensor) ** 2 / (2 * self.bandwidth ** 2))
            gpa_matrix += tmp_tensor
        t2 = time.time()

        if second_smooth:  # second layer smoothing
            filter_size = 3
            location_filter = get_location_filter(self.p, self.q, self.bandwidth, filter_size)
            gpa_matrix2 = tf.reshape(gpa_matrix, gpa_matrix.shape + (1,))
            gpa_matrix2 = tf.nn.depthwise_conv2d(gpa_matrix2, location_filter, strides=[1, 1, 1, 1], padding='SAME')
            gpa_matrix2 = tf.squeeze(gpa_matrix2)
            t2 = time.time()
            train_time = t2 - t1
            #             print(f"--Compute second-smoothed GPA matrix time: {train_time:.4f} seconds.")
            return gpa_matrix2, train_time

        train_time = t2 - t1
        #         print(f"--Compute GPA matrix time: {train_time:.4f} seconds.")
        return gpa_matrix, train_time

    def compute_density(self, raw_img):
        Omega2_star = K_tf(self.tick_tensor - raw_img, self.bandwidth_star)
        Omega1_star = Omega2_star * self.gpa_matrix
        Omega1_star = tf.reduce_sum(Omega1_star, axis=0)
        Omega2_star = tf.reduce_sum(Omega2_star, axis=0)
        GPA_density = Omega1_star / Omega2_star
        #     plt.imshow(GPA_density)
        #     plt.show()
        return GPA_density

    def obtain_mask(self, raw_img, blur_len=4, mask_thres=0.15, area_thres=150):
        # GPA density estimator
        t1 = time.time()
        GPA_density = self.compute_density(raw_img)

        # obtain mask
        avg_blur_2d = AveragePooling2D(pool_size=(blur_len, blur_len), strides=1, padding='same')
        GPA_density2 = tf.reshape(GPA_density, GPA_density.shape + (1, 1,))
        GPA_density2 = avg_blur_2d(GPA_density2)
        GPA_density2 = tf.squeeze(GPA_density2)
        mask = (GPA_density2.numpy() < mask_thres) * 1.0
        #     plt.imshow(mask)
        #     plt.show()

        # adjust mask
        mask_uint8 = mask.astype(np.uint8)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
        # stats = np.insert(stats, 5, stats[:, 2] * stats[:, 3], axis=1)
        stats = stats[np.argsort(-stats[:, 4])]  # 按照area排序
        stats = stats[1:, ]  # 去掉背景
        #     print(stats)
        for k in range(stats.shape[0]):
            area = stats[k, 4]
            if area < area_thres:
                #             print(f"delete the {k}th area")
                x1 = stats[k, 0]
                y1 = stats[k, 1]
                x2 = x1 + stats[k, 2]
                y2 = y1 + stats[k, 3]
                mask[y1:y2, x1:x2] = 0
        #     plt.imshow(mask)
        #     plt.show()
        t2 = time.time()
        gpa_time = t2 - t1
        return mask, gpa_time