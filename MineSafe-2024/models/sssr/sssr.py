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

from pyflann import *
from numpy import *
from numpy.random import *
from numpy.linalg import norm

class SSSR:
    def __init__(self, X, middle_name, s, epsilon1=10):
        """
        The SSSR model of paper ``MOVING OBJECT DETECTION IN COMPLEX SCENE USING SPATIOTEMPORAL STRUCTURED-SPARSE RPCA"
        :param X: A batch of frames.
        :param middle_name: The middle frame's name, which is used to compute superpixels.
        :param s: An initial number of superpixels.
        :param epsilon1: The number of the nearest neighbors.
        """
        # --------------------------------------------------------------
        # Input
        # --------------------------------------------------------------
        self.X = X                # (k, p, q)
        self.k = X.shape[0]       # the number of frames
        self.p = X.shape[1]       # image height / width
        self.q = X.shape[2]       # image width / height
        self.rp = 4               # the number of feature representations
        self.epsilon1 = epsilon1  # the number of nearest neighbors

        # --------------------------------------------------------------
        # Superpixels
        #   - self.labels: segments of the superpixel label
        #   - self.s: the number of superpixels, which is an update of the input s
        # --------------------------------------------------------------
        self.labels, self.s = self.get_superpixel_labels(s, middle_name)

        # --------------------------------------------------------------
        # Algorithm 1 Setup
        # --------------------------------------------------------------
        # Part IV. Experimental Evaluation Settings
        self.l = self.rp * self.s             # the col dimension of the feature matrix D
        self.lmbda = 1 / max(self.l, self.k)  # parameter used in the RPCA model and in (2)
        self.gamma1 = 0.08                    # parameter introduced in (2)
        self.gamma2 = 0.08                    # parameter introduced in (2)
        self.sigma = 50                       # !not specified in the article!
        # Initialization
        self.D = self.compute_feature()       # feature matrix with shape (l, k) with D = B + F
        self.B = np.zeros_like(self.D)        # low-rank background
        self.F = np.zeros_like(self.D)        # sparse foreground
        self.S = np.zeros_like(self.D)        # variable introduced in (3)
        self.H = np.zeros_like(self.D)        # variable introduced in (3)
        self.Y1 = np.zeros_like(self.D)       # Lagrangian multiplier introduced in (8)
        self.Y2 = np.zeros_like(self.D)       # Lagrangian multiplier introduced in (8)
        self.Y3 = np.zeros_like(self.D)       # Lagrangian multiplier introduced in (8)
        self.mu = 0.1                         # mu > 0 controls the penalty for violating the linear constraints in (8)
        self.mu_max = 1e10                    # the max value of mu set by Algorithm 1
        self.rho = 1.1                        # set by Algorithm 1
        self.m = 0                            # what is this?? set by Algorithm 1
        self.LS = self.compute_LS()           # Laplacian matrix computed from the spatial graph introduced in (2)
        self.LT = self.compute_LT()           # Laplacian matrix computed from the temporal graph introduced in (2)
        # metrics for convergence criteria
        self.epsilon2 = 1e-4                  # the tolerance factor set by Algorithm 1
        self.p1 = self.nuclear_norm(self.B)                                        # defined under (15)
        self.p2 = self.l1_norm(self.F)                                             # defined under (15)
        self.p3 = self.gamma1 * np.trace(np.transpose(self.S) @ self.LS @ self.S)  # defined under (15)
        self.p4 = self.gamma2 * np.trace(self.H @ self.LT @ np.transpose(self.H))  # defined under (15)

    def get_superpixel_labels(self, s, imgname):
        """ This function is utilized directly from https://github.com/achanta/SLIC/tree/master
        Get superpixels segmentation labels and the number of superpixels.
        :param s: An initial number of superpixels.
        :param imgname: The frame's name used to compute superpixels.
        :return: superpixels segmentation labels and the actual number of superpixels
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
        """DX is the image gradient. D is the horizontal/vertial gradient operator."""
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
        labels = self.labels                   # superpixels segmentation labels
        D = np.zeros((self.l, self.k))         # feature matrix D
        Dh, Dv = self.get_diff_mat()           # horizontal and vertical gradient operators

        for frame_idx in range(self.k):
            frame = self.X[frame_idx]          # frame with shape (p, q)
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
        """Compute the local binary patterns (LBP) from https://github.com/arsho/local_binary_patterns"""
        # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_lbp = np.zeros((self.p, self.q), np.uint8)
        for i in range(0, self.p):
            for j in range(0, self.q):
                img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
        return img_lbp

    def wT(self, M, i, j):
        """Compute the pairwise similarity for temporal window; see (4)."""
        di = M[:, i]  # the ith column
        dj = M[:, j]  # the jth column
        weight = np.exp(- norm(di - dj)**2 / (2 * self.sigma**2))  # pairwise similarity
        return weight

    def get_edge_mat(self, M):
        """
        Compute the nearest neighbors of each row of M from http://www.cs.ubc.ca/research/flann/
        :param M: the matrix in concern with shape (n, p) with n nodes
        :return: Each row contains the index of the nearest neighbors (NN). The number of the NN is self.epsilon1.
        """
        flann = FLANN()
        result, dists = flann.nn(M, M, num_neighbors=self.epsilon1, algorithm="kmeans")
        return result

    def compute_LT(self):
        """Compute LT defined in (6)"""
        weight_mat = np.zeros((self.k, self.k))             # weight matrix $W_T$
        edge_mat = self.get_edge_mat(np.transpose(self.D))  # degree matrix containing edges of the neighbors
        for i in range(self.k):
            for j in edge_mat[i]:  # if there is no edge between di and dj, then w(i, j) = 0 under (4)
                weight_mat[i, j] = self.wT(self.D, i, j)
        LT = weight_mat * (-1)                              # temporal graph defined in (6)
        for i in range(self.k):
            row_sum = np.sum(weight_mat[i, :])              # see (6)
            LT[i, i] += row_sum                             # see (6)
        return LT

    def check_Fa(self, i, j):
        """Check if i and j come from the same feature"""
        flag = False
        if (i % self.rp) == (j % self.s):
            flag = True
        return flag

    def check_Sb(self, i, j):
        """Check if i and j come from the same superpixel"""
        flag = False
        if (i // self.s) == (j // self.s):
            flag = True
        return flag

    def compute_LS(self):
        """Compute LS relied on (7)"""
        weight_mat = np.zeros((self.l, self.l))  # weight matrix $W_S$
        edge_mat = self.get_edge_mat(self.D)     # degree matrix containing edges of the neighbors
        LS = None
        # the calculation of LS can be found in (7)
        for i in range(self.l):
            for j in edge_mat[i]:
                Fa_flag = self.check_Fa(i, j)
                Sb_flag = self.check_Sb(i, j)
                if Fa_flag and not Sb_flag:
                    weight_mat[i, j] = self.wT(np.transpose(self.D), i, j)
                if not Fa_flag and Sb_flag:
                    weight_mat[i, j] = 1
        LS = weight_mat * (-1)                              # spatial graph similar to (6)
        for i in range(self.l):
            col_sum = np.sum(weight_mat[:, i])              # similar to (6)
            LS[i, i] += col_sum                             # similar to (6)
        return LS

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
        tau = 1 / self.mu  # (9)
        ZB = self.D - self.F + self.Y1 / self.mu  # (9)
        U, Sigma, Vh = np.linalg.svd(ZB, full_matrices=False)  # Singular Value Decomposition of ZB
        B = np.dot(U, np.dot(np.diag(self.SVSO(Sigma, tau)), Vh))  # (10)
        return B

    def update_H(self):
        H = (self.Y2 + self.mu * self.F) * (2 * self.gamma2 * self.LS + self.mu) ** (-1)  # (11)
        return H

    def update_S(self):
        S = np.transpose(self.Y3 + self.mu * self.F) * (2 * self.gamma1 * self.LS + self.mu) ** (-1)  # (12)
        return S

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def update_F(self):
        ZF = (self.D - self.B + self.H + self.S + (self.Y1 + self.Y2 + self.Y3) / self.mu) / 2  # (13)
        F = self.shrink(self.D - self.B + ZF / self.mu, self.lmbda / self.mu)  # (14)
        return F

    def nuclear_norm(self, A):
        """Nuclear norm of input matrix"""
        return np.sum(np.linalg.svd(A)[1])

    def l1_norm(self, A):
        """l1 norm of input matrix"""
        return np.sum(np.abs(A))

    def p_diff(self, p1, p2):
        return (p1 - p2) ** 2 / p1 ** 2

    def convergence_criteria(self):
        p1 = self.nuclear_norm(self.B)  # (15)
        p2 = self.l1_norm(self.F)  # (15)
        p3 = self.gamma1 * np.trace(np.transpose(self.S) @ self.LS @ self.S)  # (15)
        p4 = self.gamma2 * np.trace(self.H @ self.LT @ np.transpose(self.H))  # (15)
        # check the convergence criteria in (15)
        flag1 = self.p_diff(self.p1, p1) <= self.epsilon2  # (15)
        flag2 = self.p_diff(self.p2, p2) <= self.epsilon2  # (15)
        flag3 = self.p_diff(self.p3, p3) <= self.epsilon2  # (15)
        flag4 = self.p_diff(self.p4, p4) <= self.epsilon2  # (15)
        final_flag = (flag1) & (flag2) & (flag3) & (flag4)  # (15)
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

