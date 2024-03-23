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
        self.B = np.zeros_like(self.D)   # low-rank background
        self.F = np.zeros_like(self.D)   # sparse foreground
        self.S = np.zeros_like(self.D)   #
        self.H = np.zeros_like(self.D)   #
        self.Y1 = np.zeros_like(self.D)  # Lagrangian multiplier
        self.Y2 = np.zeros_like(self.D)  # Lagrangian multiplier
        self.Y3 = np.zeros_like(self.D)  # Lagrangian multiplier
        self.mu = 0.1                    # mu > 0 controls the penalty for violating the linear constraints
        self.mu_max = 1e10
        self.rho = 1.1
        self.epsilon2 = 1e-4
        self.m = 0
        self.LS = np.zeros((self.l, self.l))
        self.LT = np.zeros((self.k, self.k))

        # metrics for convergence criteria
        self.p1 = self.nuclear_norm(self.B)  # (15)
        self.p2 = self.l1_norm(self.F)       # (15)
        self.p3 = self.gamma1 * np.trace(self.quadric_form(self.S, self.LS))  # (15)
        self.p4 = self.gamma2 * np.trace(self.quadric_form(self.H, self.LT))  # (15)

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

    def wT(self, i, j):
        weight = np.exp()
    def compute_LS(self):
        pass

    def compute_LT(self):
        pass

    def SVSO(self, Sigma, tau):
        """
        singular value shrinkage operator.
        :param Sigma:
        :param tau:
        :return:
        """
        Sigma -= tau
        Sigma[Sigma < 0] = 0
        return Sigma

    def update_B(self):
        tau = 1 / self.mu                         # (9)
        ZB = self.D - self.F + self.Y1 / self.mu  # (9)
        U, Sigma, Vh = np.linalg.svd(ZB, full_matrices=False)        # Singular Value Decomposition of ZB
        B = np.dot(U, np.dot(np.diag(self.SVSO(Sigma, tau)), Vh))  # (10)
        return B

    def update_H(self):
        H = (self.Y2 + self.mu * self.F) * (2 * self.gamma2 * self.LS + self.mu)**(-1)  # (11)
        return H

    def update_S(self):
        S = np.transpose(self.Y3 + self.mu * self.F) * (2 * self.gamma1 * self.LS + self.mu)**(-1)  # (12)
        return S

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def update_F(self):
        ZF = (self.D - self.B + self.H + self.S + (self.Y1 + self.Y2 + self.Y3) / self.mu) / 2  # (13)
        F = self.shrink(self.D - self.B + ZF / self.mu, self.lmbda / self.mu)                   # (14)
        return F

    def nuclear_norm(self, A):
        """Nuclear norm of input matrix"""
        return np.sum(np.linalg.svd(A)[1])

    def l1_norm(self, A):
        """l1 norm of input matrix"""
        return np.sum(np.abs(A))

    def p_diff(self, p1, p2):
        return (p1 - p2)**2 / p1**2

    def convergence_criteria(self):
        p1 = self.nuclear_norm(self.B)                                        # (15)
        p2 = self.l1_norm(self.F)                                             # (15)
        p3 = self.gamma1 * np.trace(np.transpose(self.S) @ self.LS @ self.S)  # (15)
        p4 = self.gamma2 * np.trace(self.H @ self.LT @ np.transpose(self.H))  # (15)
        # check the convergence criteria in (15)
        flag1 = self.p_diff(self.p1, p1) <= self.epsilon2                     # (15)
        flag2 = self.p_diff(self.p2, p2) <= self.epsilon2                     # (15)
        flag3 = self.p_diff(self.p3, p3) <= self.epsilon2                     # (15)
        flag4 = self.p_diff(self.p4, p4) <= self.epsilon2                     # (15)
        final_flag = (flag1) & (flag2) & (flag3) & (flag4)                    # (15)
        # update p-metrics
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        return final_flag

    def fit(self, max_iter=100):
        """
        Algorithm 1: Proposed B-SSSR Algorithm for MOD
        :param max_iter:
        :return:
        """
        for iter in range(max_iter):
            # Step 1. Update B
            self.B = self.update_B()  # (9)-(10)
            # Step 2. Compute H, S, and F
            self.H = self.update_H()  # (11)
            self.S = self.update_S()
            self.F = self.update_F()  # (14)
            # Step 3.1 Compute Y1
            self.Y1 = self.Y1 + self.mu * (self.X - self.B - self.F)
            # Step 3.2 Compute Y2
            self.Y2 = self.Y2 + self.mu * (self.F - self.H)
            # Step 4. Compute Y3
            self.Y3 = self.Y3 + self.mu * (self.F - self.S)
            # Self 5. Update mu
            self.mu = min(self.rho * self.mu, self.mu_max)
            # Step 6. Check convergence according to (15)
            flag = self.convergence_criteria()
            print(f"Iter {iter}: p1:{self.p1:.4f}; p2:{self.p2:.4f}; p3:{self.p3:.4f}; p4:{self.p4:.4f}")
            if flag:  # if converged, then end iteration and output
                break
            return self.B, self.F

