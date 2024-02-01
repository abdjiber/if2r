import numpy as np
import pandas as pd
from numba import jit
import utils


class IFDA:

    def __init__(self,
                 alpha,
                 lr,
                 idx_sensitive_cols,
                 metric="euclidean",
                 delta=1,
                 q=1,
                 tol=1e-7,
                 maxiter=100,
                 seed=42,
                 init_theta="zeros"):
        """Implements the Individual Fairness algorithm introduced in Mukherjee et al. 2022.
        
        Args:
            alpha: float
                Training regularization strenght.
            lr: float
                The learning rate.
            idx_sensitive_cols: list(int)
                Indexes of sentive attributes.
            metric: string, default="euclidean"
                The pairwise similarity measure. Value should be one of np.pdist.
            delta: float, default=1
                The temperature parameter used to compute the Graph Laplacian matrix.
            q: float, default=1
                The quantile used to compute the proposed fairness metric in the main paper.
            init_theta: string, default="zeros"
                Specifies how theta should be initialized. Values should be "zeros" or "random".
            tol: float, default=1e-7
                The stopping criteria.
            maxiter: int, default=150
                The maximum number of iterations for convergence.
            seed: int, default=42
                The random seed used in all experiments.
        """
        self.alpha = alpha
        self.lr = lr
        self.metric = metric
        self.idx_sensitive_cols = idx_sensitive_cols
        self.delta = delta
        self.q = q
        self.tol = tol
        self.maxiter = maxiter
        self.seed = seed
        self.init_theta = init_theta

    @jit
    def get_grad_theta_S(self, X, S):
        grad = np.dot(X, np.diag(S * (1 - S)))
        return grad

    @jit
    def get_grad_R(self, X, S, L):
        grad_theta_S = self.get_grad_theta_S(X, S)
        grad_R = np.dot(grad_theta_S, np.dot(L, S))
        return grad_R

    @jit
    def get_grad_l(self, X, y, S):
        grad_l = np.dot(X, np.diag(S - y)).sum(axis=1) / len(y)
        return grad_l

    @jit
    def get_grad_total(self, X, y, S, L):
        grad_R = self.get_grad_R(X, S, L)
        grad_l = self.get_grad_l(X[:, :self.train_size], y[:self.train_size],
                                 S[:self.train_size])
        grad_total = grad_l + self.alpha * grad_R
        return grad_total

    @jit
    def sigmoid(self, theta, X):
        probs = 1. / (1. + np.exp(-np.dot(X.T, theta)))
        return probs

    @jit
    def get_loss_l(self, y, S, epsillon=1e-7):
        tmp = y * np.log(S + epsillon) + (1 - y) * np.log(1 - S + epsillon)
        loss_l = -np.mean(tmp)
        return loss_l

    @jit
    def get_loss_r(self, S, L):
        return np.dot(S.T, np.dot(L, S))

    @jit
    def get_loss_tr(self, y, S, L):
        loss_l = self.get_loss_l(y[:self.train_size], S[:self.train_size])
        loss_r = self.get_loss_r(S, L)
        loss = loss_l + self.alpha * loss_r
        return loss, loss_l, loss_r

    def fit(self, Xs, ys, Xt, yt):
        np.random.seed(self.seed)
        n, m = Xs.shape
        self.train_size = n
        X_st = pd.concat((Xs, Xt))
        y_st = np.append(ys, yt)

        if self.init_theta == "zeros":
            theta_new = np.zeros((m + 1, ))
        else:
            theta_new = np.random.randn(m + 1)
        scores = {
            "fpr_s": [],
            "fnr_s": [],
            "auc_s": [],
            "pc_s": [],
            "fg_s": [],
            "fpr_t": [],
            "fnr_t": [],
            "auc_t": [],
            "pc_t": [],
            "fg_t": [],
            "loss": [],
        }
        Xs, Ks, _ = utils.get_params_X(Xs,
                                       metric=self.metric,
                                       delta=self.delta)
        Xt, Kt, _ = utils.get_params_X(Xt,
                                       metric=self.metric,
                                       delta=self.delta)
        tau_s = utils.get_tau(Ks, q=self.q)
        tau_t = utils.get_tau(Kt, q=self.q)

        X_st, K_st, L_st = utils.get_params_X(X_st,
                                              metric=self.metric,
                                              delta=self.delta)

        S_st_new = self.sigmoid(theta_new, X_st)
        S_s_new = S_st_new[:self.train_size]
        S_t_new = S_st_new[self.train_size:]

        loss_new, loss_l, loss_r = self.get_loss_tr(y_st, S_st_new, L_st)
        loss_old = np.inf

        i = 1
        while np.abs(loss_old - loss_new) > self.tol and i < self.maxiter:
            S_st_old = S_st_new
            loss_old = loss_new
            theta_old = theta_new

            grad_total = self.get_grad_total(X_st, y_st, S_st_old, L_st)
            theta_new = theta_old - self.lr * grad_total

            S_st_new = self.sigmoid(theta_new, X_st)
            S_s_new = S_st_new[:self.train_size]
            S_t_new = S_st_new[self.train_size:]
            loss_new, loss_l, loss_r = self.get_loss_tr(y_st, S_st_new, L_st)
            scores = utils.add_scores(self, theta_new, scores, Xs, ys.values,
                                      S_s_new, Ks, tau_s, Xt, yt.values,
                                      S_t_new, Kt, tau_t, loss_new,
                                      self.idx_sensitive_cols)
            i += 1

        for score in scores:
            scores[score] = np.array(scores[score])

        self.S_s_new = S_s_new
        self.S_t_new = S_t_new
        self.theta_new = theta_new
        self.scores = scores
        self.loss_l = loss_l
        self.loss_r = loss_r
        return self
