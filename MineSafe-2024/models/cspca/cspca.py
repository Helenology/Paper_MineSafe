import numpy as np
import matplotlib.pyplot as plt


class TV:
    def __init__(self, y, Phi, m, n):
        self.y = y
        self.Phi = Phi
        self.m = m
        self.n = n
        self.x_est = None

    def get_diff_mat(self):
        m = self.m
        n = self.n
        Dh = np.zeros((n, n))
        Dv = np.zeros((m, m))
        for j in range(n - 1):
            Dh[j, j] = -1
            Dh[j + 1, j] = 1
        for i in range(m - 1):
            Dv[i, i] = -1
            Dv[i, i + 1] = 1
        return Dh, Dv

    def tv_2d(self, x):
        m = self.m
        n = self.n
        X = x.reshape(m, n)
        Dh, Dv = self.get_diff_mat()
        X_Dh = np.dot(X, Dh)
        Dv_X = np.dot(Dv, X)
        tv = np.sum(np.abs(X_Dh)) + np.sum(np.abs(Dv_X))
        G = self.get_subgradient(Dh, Dv, X_Dh, Dv_X)
        return tv, G

    def sign(self, X):
        sign_mat = np.zeros_like(X)
        sign_mat[X >= 0] = 1
        sign_mat[X < 0] = -1
        return sign_mat

    def get_subgradient(self, Dh, Dv, X_Dh, Dv_X):
        G = np.transpose(Dv) @ self.sign(Dv_X) + self.sign(X_Dh) @ np.transpose(Dh)
        return G

    def fit(self, mu=1, step_size=5e-3, max_iter=100, tol=1e-4):
        X_est = x_t.reshape(self.m, self.n) * 0.9
        x_est = X_est.ravel()
        self.x_est = x_est * 1.0
        err = np.Inf
        loss_list = []

        for i in range(max_iter):
            tv, G = self.tv_2d(x_est)
            loss1 = mu * tv
            delta = self.y - self.Phi @ x_est
            loss2 = 0.5 * np.linalg.norm(delta) ** 2
            loss = loss1 + loss2
            loss_list.append(loss)

            tv_update = mu * G
            l2_update = -(np.transpose(self.Phi) @ delta).reshape(self.m, self.n)
            X_est -= step_size * (tv_update + l2_update)

            x_est = X_est.ravel()
            err = np.linalg.norm(self.x_est - x_est)
            print(f"Iter [{i}]: loss(tv):{loss1:.2f} + loss(l2):{loss2:.2f} = loss:{loss:.2f} \t error:{err:.4f}")
            self.x_est = x_est * 1.0
            if err < tol or err > 1e8:
                break