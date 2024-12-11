#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/3 10:24
# @Author  : Helenology
# @Site    : 
# @File    : simu_auxiliary.py
# @Software: PyCharm

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pickle
import os
import shutil
import sys
import time
sys.path.append("../models/GPA")
from useful_functions import *


def generate_random_simulate_image(mean, sigma):
    """Generate a random simulate image."""
    tfd = tfp.distributions
    dist = tfd.TruncatedNormal(loc=mean, scale=sigma, low=[0.], high=[1.])
    simulate_img = dist.sample(1)
    return simulate_img


def compute_true_density(tick_list, mean, sigma, IsSave=False):
    # Truncated Normal true density
    tfd = tfp.distributions
    dist = tfd.TruncatedNormal(loc=mean, scale=sigma, low=[0.], high=[1.])
    f_true = tf.concat([dist.prob(tick) for tick in tick_list], axis=0)
    print("True density shape:", f_true.shape)
    if IsSave:
        with open("./f_true.pkl", 'wb') as f:
            pickle.dump(f_true, f)
            print("save true density at ./f_true.pkl")
    return f_true


# optimal bandwidth
def compute_optimal_bandwidths(N, M, sigma):
    bandwidth = np.pi ** (-1/7) * sigma ** (5 / 7) * (N * M) ** (- 1 / 7)
    print("Optimal bandwidth from Rule of Thumb:", bandwidth)
    bandwidth_star = bandwidth**2 * 5
    print("Optimal bandwidth* from GPA estimation:", bandwidth_star)
    return bandwidth, bandwidth_star


def generate_simulate_data(path, N, mean, sigma):
    if os.path.exists(path):  # 文件夹存在就删除文件夹
        shutil.rmtree(path)   # 递归删除
    os.mkdir(path)            # 创建空文件夹

    # generate new images
    for i in range(N):
        train_img = generate_random_simulate_image(mean, sigma)
        np.save(path + f"/train_img_{i}.npy", train_img.numpy())


def compute_CD_matrix(path, N, G, p, q, bandwidth, tick_tensor):
    CD_tensor = tf.zeros((G, p, q), dtype=tf.float32)
    for i in range(N):
        simulate_img = np.load(path + f"/simulate_img_{i}.npy")
        simulate_img = tf.constant(simulate_img)
        # compute classic nonparametric density estimator
        tmp_tensor = 1 / tf.sqrt(2 * np.pi) * tf.exp(-(simulate_img - tick_tensor) ** 2 / (2 * bandwidth ** 2))
        tmp_tensor = tmp_tensor / (N * bandwidth)
        CD_tensor += tmp_tensor
    print(f"-[CD] Successfully compute classical density with N={N} at " + path)
    return tf.squeeze(CD_tensor)


def compute_location_weight(p, q, h, truncate_width=5):
    # the locations of the pixels
    location_weight0 = np.zeros([truncate_width, truncate_width, 2])
    for i in range(truncate_width):
        location_weight0[i, 0:truncate_width, 0] = abs(i-truncate_width//2) / p
    for j in range(truncate_width):
        location_weight0[0:truncate_width, j, 1] = abs(j-truncate_width//2) / q

    # location weight from (0, 0)
    location_weight = K_np(location_weight0[:, :, 0], h) * K_np(location_weight0[:, :, 1], h)
    # # show the location weights in heatmap
    # sns.heatmap(location_weight)
    # plt.show()
    return location_weight


def compute_DS_matrix(CD_est, location_weight):
    CD_est = tf.squeeze(CD_est)
    CD_est = tf.reshape(CD_est, [1, *CD_est.shape, 1])
    Omega1 = tf.nn.depthwise_conv2d(CD_est, location_weight, strides=[1, 1, 1, 1], padding='SAME')
    Omega2 = tf.reduce_sum(location_weight)
    DS_est = Omega1 / Omega2
    return DS_est


# def test_CD(p, q, test_img, bandwidth, path):
#     if os.path.exists(path) is False:
#         print("No simulating images stored!")
#         return None, None

#     # 得到每一张模拟图片的路径
#     train_list = os.listdir(path)
#     train_list = [path + '/' + item for item in train_list]
#     N = len(train_list)
#     CD_est = tf.zeros((1, p, q), dtype=tf.float32)

#     for i in range(N):
#         simulate_img = tf.constant(np.load(train_list[i]))
#         tmp_tensor = 1 / tf.sqrt(2 * np.pi) * tf.exp(-(simulate_img - test_img) ** 2 / (2 * bandwidth ** 2))
#         tmp_tensor = tmp_tensor / (N * bandwidth)
#         CD_est += tmp_tensor

#     return CD_est