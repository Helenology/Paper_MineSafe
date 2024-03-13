#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 14:40
# @Author  : Helenology
# @Site    :
# @File    : sssr.py
# @Software: PyCharm


import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append("./SLIC-master/python_interface/")
sys.path.append("./models/sssr/local_binary_patterns-master/")
from SLICdemo import *
from lbp import lbp_calculated_pixel


class SSSR:
    def __init__(self, X, middle_name, rp, s, lmbda, gamma1, gamma2):
        # --------------------------------------------------------------
        # Input
        # --------------------------------------------------------------
        # Frames
        self.X = X                   # (k, p, q)
        self.k = X.shape[0]          # the number of frames
        self.p = X.shape[1]
        self.q = X.shape[2]

        # Superpixels:
        #   - self.labels: segments of the superpixel label
        #   - self.s: the number of superpixels
        self.labels, self.s = self.get_superpixel_labels(s, middle_name)

        self.rp = rp                 # the number of feature representations
        self.l = self.rp * self.s    # the col dimension of the feature matrix D according to Section III's part A
        self.lmbda = lmbda           #
        self.gamma1 = gamma1         #
        self.gamma2 = gamma2         #

        # Initialization
        self.D = self.compute_feature()  # feature matrix with shape (l, k)
        self.B = np.zeros(())            # low-rank background
        self.F = np.zeros(())            # sparse foreground
        self.S = 0
        self.H = 0
        self.Y1 = np.zeros(())  # Lagrangian multiplier
        self.Y2 = np.zeros(())  # Lagrangian multiplier
        self.Y3 = np.zeros(())  # Lagrangian multiplier
        self.mu = 0.1                    # mu > 0 controls the penalty for violating the linear constraints
        self.mu_max = 1e10
        self.rho = 1.1
        self.epsilon2 = 1e-4
        self.m = 0
        self.LS = np.zeros((self.l, self.l))
        self.LT = np.zeros((self.k, self.k))

    def get_superpixel_labels(self, s, imgname):
        """
        Get superpixels segmentation labels and the number of superpixels.
        :param imgname: the image path of the middle frame
        :return: superpixels segmentation labels and the number of superpixels
        """
        # --------------------------------------------------------------
        # Create shared library
        # --------------------------------------------------------------
        if os.path.exists("./models/sssr/SLIC-master/python_interface/libslic.so"):
            print("Compiled library exists")
        else:
            subprocess.call(
                "gcc -c -fPIC ./models/sssr/SLIC-master/python_interface/slicpython.c -o ./models/sssr/SLIC-master/python_interface/slic.o",
                shell=True)
            subprocess.call(
                "gcc -shared ./models/sssr/SLIC-master/python_interface/slic.o -o ./models/sssr/SLIC-master/python_interface/libslic.so",
                shell=True)
            print("library compiled")

        numsuperpixels = s
        compactness = 20.0
        doRGBtoLAB = True  # only works if it is a three channel image
        labels, numlabels = segment(imgname, numsuperpixels, compactness, doRGBtoLAB)
        return labels, numlabels

    def get_diff_mat(self):
        m = self.p
        n = self.q
        Dh = np.zeros((n, n))
        Dv = np.zeros((m, m))
        for j in range(n - 1):
            Dh[j, j] = -1
            Dh[j + 1, j] = 1
        for i in range(m - 1):
            Dv[i, i] = -1
            Dv[i, i + 1] = 1
        return Dh, Dv

    def compute_feature(self):
        labels = self.labels
        D = np.zeros((self.l, self.k))
        Dh, Dv = self.get_diff_mat()

        for frame_idx in range(self.k):
            frame = self.X[frame_idx]          # shape (p, q)
            img_lbp = self.compute_LBP(frame)  # LBP feature with shape (p, q)
            hor_grad = np.dot(frame, Dh)       # horizontal gradient feature with shape (p, q)
            ver_grad = np.dot(Dv, frame)       # vertical gradient feature with shape (p, q)

            # Feature Construction
            d = []
            for label_idx in range(self.s):
                # intensity feature
                d.append(frame[labels == label_idx].mean())
                # LBP feature (local binary patterns)
                d.append(img_lbp[labels == label_idx].mean())
                # horizon gradient feature
                d.append(hor_grad[labels == label_idx].mean())
                # vertical gradient feature
                d.append(ver_grad[labels == label_idx].mean())
            # record the full feature vector for this frame into feature matrix D
            D[:, frame_idx] = d
        return D

    def compute_LBP(self, img_gray):
        height = img_gray.shape[0]
        width = img_gray.shape[1]
        # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_lbp = np.zeros((height, width), np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
        return img_lbp

    def fit(self):
        pass
