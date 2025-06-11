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
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from tensorflow.keras.layers import AveragePooling2D
import cv2

sys.path.append("../../")
from useful_functions import *
from utils import load_and_preprocess_image, get_location_filter, K_tf


class GPA:
    def __init__(self, G, p, q, train_list, second_smooth=True, gpa_matrix=None, grid_point=None):
        """
        Initialization.
        :param G: the number of grid points
        :param p: image height
        :param q: image width
        :param train_list: path list of randomly selected images
        :param second_smooth: (1) True: smooth the GPA matrix over pixel locations $s$ or DS estimator;
                              (2) False: the original KDE or CD estimator;
        :param gpa_matrix: (1) None: GPA matrix will be computed based on the grid points;
                           (2) the pre-computed GPA matrix
        :param grid_point: (1) None: grid_point will be randomly generated;
                           (2) the pre-specified tick numbers
        """
        # basic parameters
        self.G = G  # the number of grid points
        self.p = p  # image height
        self.q = q  # image width
        self.train_list = train_list  # path list of randomly selected images
        self.N0 = len(train_list)  # the number of randomly selected images

        # compute optimal bandwidth
        self.alpha = np.log(p * q) / np.log(self.N0)
        self.bandwidth, self.bandwidth_star = compute_optimal_bandwidths(self.N0, self.alpha)

        # grid points {x_g^*: 1 <= g <= G}
        rng = np.random.default_rng(seed=0)  # random number generator
        if grid_point is None:
            tick_list = rng.random(size=G)  # grid points
            self.grid_point = tf.concat([tf.ones([1, p, q]) * tick for tick in tick_list], axis=0)
        else:
            self.grid_point = grid_point
        # compute GPA matrix
        if gpa_matrix is None:
            self.gpa_matrix, self.train_time = self.compute_GPA_matrix(second_smooth=second_smooth)
            self.gpa_matrix /= tf.reduce_max(self.gpa_matrix)
        else:
            self.gpa_matrix, self.train_time = gpa_matrix, None

    def compute_GPA_matrix(self, second_smooth=True):
        """
        Compute the GPA matrix.
        :param second_smooth: (1) True: smooth the GPA matrix over pixel locations $s$ or the DS estimator;
                              (2) False: the original KDE or CD estimator;
        :return:
        """
        # First, compute the CD estimator.
        gpa_matrix = tf.zeros([self.G, self.p, self.q])
        t1 = time.time()
        for i in range(self.N0):
            img = load_and_preprocess_image(self.train_list[i], (self.p, self.q))
            tmp_tensor = (1 / (self.N0 * self.bandwidth)) * (1 / tf.sqrt(2 * np.pi)) * tf.exp(
                -(img - self.grid_point) ** 2 / (2 * self.bandwidth ** 2))
            gpa_matrix += tmp_tensor
        t2 = time.time()

        # Second, smooth the GPA matrix over pixel locations $s$ or the DS estimator if needed.
        if second_smooth:
            filter_size = 3
            location_filter = get_location_filter(self.p, self.q, self.bandwidth, filter_size)
            gpa_matrix2 = tf.reshape(gpa_matrix, gpa_matrix.shape + (1,))
            gpa_matrix2 = tf.nn.depthwise_conv2d(gpa_matrix2, location_filter, strides=[1, 1, 1, 1], padding='SAME')
            gpa_matrix2 = tf.squeeze(gpa_matrix2)
            t2 = time.time()
            train_time = t2 - t1
            return gpa_matrix2, train_time

        train_time = t2 - t1
        return gpa_matrix, train_time

    def compute_density(self, new_img):
        """
        Compute the density of a new image.
        :param new_img: a new image.
        :return:
        """
        Omega2_star = K_tf(self.grid_point - new_img, self.bandwidth_star)
        Omega1_star = Omega2_star * self.gpa_matrix
        Omega1_star = tf.reduce_sum(Omega1_star, axis=0)
        Omega2_star = tf.reduce_sum(Omega2_star, axis=0)
        GPA_density = Omega1_star / Omega2_star
        return GPA_density

    def obtain_mask(self, GPA_density, density_thres=1.75, blur_len=4,
                    blur_thres=0.15, area_thres=150, return_box=False, debug=False):
        if debug:
            plt.imshow(GPA_density)
            plt.title(f"[0] GPA_density")
            plt.colorbar()
            plt.show()

        # obtain mask
        if density_thres is not None:
            tmp_tensor = tf.cast(GPA_density <= density_thres, tf.float32) * GPA_density + tf.cast(
                GPA_density > density_thres, tf.float32)
        else:
            tmp_tensor = GPA_density
        if debug:
            plt.imshow(tmp_tensor)
            plt.title(f"[1] tmp_tensor - density_thres={density_thres}")
            plt.show()
        tmp_tensor = tf.reshape(tmp_tensor, (1, self.p, self.q, 1))
        avg_blur_2d = AveragePooling2D(pool_size=(blur_len, blur_len), strides=1, padding='same')
        blur_tensor = tf.squeeze(avg_blur_2d(tmp_tensor))
        if debug:
            plt.imshow(blur_tensor)
            plt.title(f"[2] blur_tensor - avgblur={blur_len}")
            plt.colorbar()
            plt.show()
        mask = (blur_tensor.numpy() < blur_thres) * 1.0
        if debug:
            plt.imshow(mask)
            plt.title(f"[3] mask - blur_thres={blur_thres}")
            plt.show()

        # adjust mask
        mask_uint8 = mask.astype(np.uint8)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
        # stats = np.insert(stats, 5, stats[:, 2] * stats[:, 3], axis=1)
        stats = stats[np.argsort(-stats[:, 4])]  # 按照area排序
        stats = stats[1:, ]  # 去掉背景
        if return_box is False:  # 不返回bounding box；返回segmentation结果
            for k in range(stats.shape[0]):
                area = stats[k, 4]
                if area < area_thres:
                    #             print(f"delete the {k}th area")
                    x1 = stats[k, 0]
                    y1 = stats[k, 1]
                    x2 = x1 + stats[k, 2]
                    y2 = y1 + stats[k, 3]
                    mask[y1:y2, x1:x2] = 0
            return mask
        else:  # 返回bounding box
            if debug:
                print(stats)
            stats = stats[stats[:, 4] > area_thres, ]
            try:
                return stats[0, ]
            except:
                return None
