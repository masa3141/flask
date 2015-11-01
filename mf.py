#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Matrix Factorization
Trained by SGD(Stochastic Gradient Discent)

You can run following codes.
mf = MF(R, inds, K, alpha, lam)
mf.train(epochs=100)
mf.predict()
"""

import numpy as np
import random


class MF():
    """
    MF()
    """

    def __init__(self, R, inds, K, alpha, lam):
        """ init paramaters """
        self.R = R  # true value
        self.inds = inds  # index of true value
        self.K = K  # dimention of latent factor
        self.alpha = alpha  # training rate of SGD
        self.lam = lam  # regularized parameter of matrix factorization
        N = len(R)  # number of user
        M = len(R[0])  # number of item
        self.p = np.random.random([N, K])  # latent factor of user
        self.q = np.random.random([M, K])  # latent factor of item

    def train(self, epochs=10):
        for epoch in range(epochs):
            import time
            time.sleep(1)
            print "epoch=", epoch
            print "rmse=", self.rmse()
            R_ = self.predict()
            random.shuffle(self.inds)
            for ind in self.inds:
                u, i = ind
                err = self.R[u, i] - R_[u, i]
                self.p[u, :] += self.alpha*(2*err*self.q[i, :] - 2*self.lam*self.p[u, :])
                self.q[i, :] += self.alpha*(2*err*self.p[u, :] - 2*self.lam*self.q[i, :])

    def predict(self):
        return np.dot(self.p, self.q.T)

    def rmse(self):
        err = 0.0
        R_ = self.predict()
        for ind in self.inds:
            u, i = ind
            err += (self.R[u, i] - R_[u, i]) ** 2
        err = np.sqrt(err/len(self.inds))
        return err

if __name__ == "__main__":
    R = np.array([[4, 5, 0, 0],
                 [4, 0, 4, 1],
                 [0, 4, 0, 2],
                 [0, 0, 5, 2]])
    inds = [(0, 0), (0, 1), (1, 0), (1, 2), (1, 3), (2, 1), (2, 3), (3, 2), (3, 3)]
    K = 2
    alpha = 0.01
    lam = 0.1

    mf = MF(R, inds, K, alpha, lam)
    mf.train(epochs=100)
    print mf.predict()
    print mf.rmse()
