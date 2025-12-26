import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Unable to import matplotlib. RobustPCA.plot_fit() will not work.')


class RobustPCA:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu is not None:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D.flatten(), ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda is not None:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum(np.abs(M) - tau, 0)

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        n_iter = 0
        err = np.inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol is not None:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        #this loop implements the principal component pursuit (PCP) algorithm
        #located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while err > _tol and n_iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)                            #this line implements step 3
            Sk = self.shrink(
                self.D - Lk + self.mu_inv * Yk, self.lmbda * self.mu_inv)               #this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)                                      #this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            n_iter += 1
            if (n_iter % iter_print) == 0 or n_iter == 1 or n_iter > max_iter or err <= _tol:
                print(f'iteration: {n_iter}, error: {err}')

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size is not None:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print(f'ymin: {ymin}, ymax: {ymax}')

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for i in range(numplots):
            plt.subplot(nrows, ncols, i + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[i, :] + self.S[i, :], 'r')
            plt.plot(self.L[i, :], 'b')
            if not axis_on:
                plt.axis('off')
