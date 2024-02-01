import numpy as np
from numba import jit
import utils
from sklearn.metrics import roc_auc_score


class IF2R:

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
        """Implements the variant of Mukherjee et al. 2022.
        
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
        self.is_trained = False

    @jit
    def get_grad_theta_S(self, X, S):
        grad = np.dot(X, np.diag(S * (1 - S)))
        return grad

    @jit
    def get_grad_r(self, X, S, L):
        grad_theta_S = self.get_grad_theta_S(X, S)
        grad_R = np.dot(grad_theta_S, np.dot(L, S))
        return grad_R

    @jit
    def get_grad_l(self, X, y, S, W):
        grad_l = np.dot(X, np.diag(W * (S - y))).sum(axis=1) / len(y)
        return grad_l

    @jit
    def get_grad_tr(self, X, y, S, L, W):
        grad_r = self.get_grad_r(X, S, L)
        grad_l = self.get_grad_l(X, y, S, W)
        grad_total = grad_l + self.alpha * grad_r
        return grad_total

    @jit
    def get_grad_inference(self, Xt, St, Lt, theta_t, beta):
        grad_r = self.get_grad_r(Xt, St, Lt)
        diff = theta_t - self.theta_s
        grad_infer = grad_r + beta * diff
        return grad_infer

    @jit
    def sigmoid(self, theta, X):
        probs = 1. / (1. + np.exp(-np.dot(X.T, theta)))
        return probs

    @jit
    def get_loss_l(self, y, S, W, epsillon=1e-7):
        tmp = y * np.log(S + epsillon) + (1 - y) * np.log(1 - S + epsillon)
        loss_l = -np.mean(W * tmp)
        return loss_l

    @jit
    def get_loss_inference(self, S, L, theta_t, beta):
        loss_r = self.get_loss_r(S, L)
        loss_infer = loss_r
        return loss_infer

    @jit
    def get_loss_r(self, S, L):
        return np.dot(S.T, np.dot(L, S))

    @jit
    def get_loss_tr(self, y, S, W, L):
        loss_l = self.get_loss_l(y, S, W)
        loss_r = self.get_loss_r(S, L)
        loss = loss_l + self.alpha * loss_r
        return loss, loss_l, loss_r

    def fit(self, Xs, ys, Xt, yt, W):
        np.random.seed(self.seed)
        if self.init_theta == "zeros":
            theta_new = np.zeros((Xs.shape[1] + 1, ))
        else:
            theta_new = np.random.randn(Xs.shape[1] + 1)

        Xs, Ks, Ls = utils.get_params_X(Xs,
                                        metric=self.metric,
                                        delta=self.delta)
        Xt, Kt, Lt = utils.get_params_X(Xt,
                                        metric=self.metric,
                                        delta=self.delta)
        tau_s = utils.get_tau(Ks, q=self.q)
        tau_t = utils.get_tau(Kt, q=self.q)

        scores = {
            "fpr_s": [],
            "fnr_s": [],
            "auc_s": [],
            "pc_s": [],
            "fg_s": [],
            "fpr_t": [],  # Tracking the target data performance at training
            "fnr_t": [],
            "auc_t": [],
            "pc_t": [],
            "fg_t": [],
            "loss": [],
        }

        S_s_new = self.sigmoid(theta_new, Xs)
        S_t_new = self.sigmoid(theta_new, Xt)
        loss_new, loss_l, loss_r = self.get_loss_tr(ys, S_s_new, W, Ls)
        loss_old = np.inf

        i = 1
        while np.abs(loss_old - loss_new) > self.tol and i < self.maxiter:
            S_s_old = S_s_new
            loss_old = loss_new
            theta_old = theta_new

            grad_total = self.get_grad_tr(Xs, ys, S_s_old, Ls, W)
            theta_new = theta_old - self.lr * grad_total

            S_s_new = self.sigmoid(theta_new, Xs)
            S_t_new = self.sigmoid(theta_new, Xt)
            loss_new, loss_l, loss_r = self.get_loss_tr(ys, S_s_new, W, Ls)
            scores = utils.add_scores(self, theta_new, scores, Xs, ys.values,
                                      S_s_new, Ks, tau_s, Xt, yt.values,
                                      S_t_new, Kt, tau_t, loss_new,
                                      self.idx_sensitive_cols)
            i += 1
        for score in scores:
            scores[score] = np.array(scores[score])

        self.S_s = S_s_new
        self.theta_s = theta_new
        self.scores_s = scores
        self.is_trained = True
        self.loss_l = loss_l
        self.loss_r = loss_r
        return self

    def predict_proba(self, Xt, yt, lr, maxiter, tol):
        Xt, Kt, Lt = utils.get_params_X(Xt,
                                        metric=self.metric,
                                        delta=self.delta)
        tau_t = utils.get_tau(Kt, q=self.q)
        if self.is_trained:
            scores = {
                "fpr_t": [],
                "fnr_t": [],
                "auc_t": [],
                "pc_t": [],
                "fg_t": [],
                "loss": [],
            }

            theta_t_new = self.theta_s.copy()
            S_t_new = self.sigmoid(theta_t_new, Xt)
            loss_new = self.get_loss_r(S_t_new, Lt)
            loss_old = np.inf

            i = 1
            while np.abs(loss_old - loss_new) > tol and i < maxiter:
                S_t_old = S_t_new
                loss_old = loss_new
                theta_t_old = theta_t_new

                theta_t_new = theta_t_old - lr * self.get_grad_r(
                    Xt, S_t_old, Lt)

                S_t_new = self.sigmoid(theta_t_new, Xt)
                loss_new = self.get_loss_r(S_t_new, Lt)
                y_t_pred = utils.get_hard_preds(S_t_new)
                tn_t, fp_t, fn_t, tp_t = utils.confusion_matrix(
                    yt, y_t_pred).ravel()

                fn_t_rate = fn_t / (fn_t + tp_t)
                fp_t_rate = fp_t / (fp_t + tn_t)

                fc_t = utils.get_fairness_score(yt.values, y_t_pred, Kt, tau_t)
                pc_t = utils.pred_consistency(
                    self,
                    Xt,
                    theta_t_new,
                    idx_sensitive_cols=self.idx_sensitive_cols)

                scores["pc_t"].append(pc_t)
                scores["fg_t"].append(fc_t)

                scores["auc_t"].append(roc_auc_score(yt, S_t_new))
                scores["fnr_t"].append(fn_t_rate)
                scores["fpr_t"].append(fp_t_rate)
                scores["loss"].append(loss_new)
                i += 1

            for score in scores:
                scores[score] = np.array(scores[score])

            self.S_t = S_t_new
            self.theta_t = theta_t_new
            self.scores_t = scores
            #self.loss_l = loss_l
            #self.loss_r = loss_r
        return self
