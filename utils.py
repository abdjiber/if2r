import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from numba import jit
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from aif360 import datasets

import warnings
import pickle

warnings.filterwarnings(action="ignore")


def scale(X):
    cols = X.columns
    Scaler = MinMaxScaler()
    scaled = Scaler.fit_transform(X)
    scaled = pd.DataFrame(scaled, columns=cols)
    return scaled


def get_dist(X, metric="euclidean"):
    return pairwise_distances(X, metric=metric, n_jobs=-1)


def get_graph_laplacian(X, metric, delta):
    dist = get_dist(X, metric)
    K = np.exp(-delta * dist)
    D = np.diag(K.sum(axis=1))
    return dist, K, D - K


def add_intercepts(X):
    n = X.shape[0]
    slopes = np.ones(n)
    X_new = np.hstack((np.atleast_2d(slopes).T, X))
    return X_new.T


def get_params_X(X, metric="euclidean", delta=2):
    _, K, L = get_graph_laplacian(X, metric, delta)
    X_new = add_intercepts(X)
    return X_new, K, L


def get_tau(K, q=.2):
    trui_idx = np.triu_indices(K.shape[0])
    kk = K.copy()
    kk = kk[trui_idx].flatten()
    return np.quantile(kk, q)


def get_fairness_score(y, y_pred, K, tau=.2):
    trui_idx = np.triu_indices(K.shape[0])
    KK = K.copy()
    KK[trui_idx] = np.inf
    idx = np.argwhere(KK < tau)
    n = idx.shape[0]
    if n == 0:
        return 0
    else:
        idx_0, idx_1 = idx[:, 0], idx[:, 1]
        score = ((y[idx_0] != y[idx_1]) &
                 (y_pred[idx_0] == y_pred[idx_1])).sum()
        score /= n
        return score


def add_scores(model, theta, scores, Xs, ys, S_s_new, Ks, tau_s, Xt, yt,
               S_t_new, Kt, tau_t, loss_new, idx_sensitive_cols):
    y_s_pred = get_hard_preds(S_s_new)
    y_t_pred = get_hard_preds(S_t_new)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(ys, y_s_pred).ravel()
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(yt, y_t_pred).ravel()

    fn_s_rate = fn_s / (fn_s + tp_s)
    fp_s_rate = fp_s / (fp_s + tn_s)
    fn_t_rate = fn_t / (fn_t + tp_t)
    fp_t_rate = fp_t / (fp_t + tn_t)

    fc_s = get_fairness_score(ys, y_s_pred, Ks, tau_s)
    fc_t = get_fairness_score(yt, y_t_pred, Kt, tau_t)

    pc_s = pred_consistency(model,
                            Xs,
                            theta,
                            idx_sensitive_cols=idx_sensitive_cols)
    pc_t = pred_consistency(model,
                            Xt,
                            theta,
                            idx_sensitive_cols=idx_sensitive_cols)

    scores["pc_s"].append(pc_s)
    scores["pc_t"].append(pc_t)

    scores["fg_s"].append(fc_s)
    scores["fg_t"].append(fc_t)

    scores["auc_s"].append(roc_auc_score(ys, S_s_new))
    scores["fnr_s"].append(fn_s_rate)
    scores["fpr_s"].append(fp_s_rate)

    scores["auc_t"].append(roc_auc_score(yt, S_t_new))
    scores["fnr_t"].append(fn_t_rate)
    scores["fpr_t"].append(fp_t_rate)
    scores["loss"].append(loss_new)
    return scores


def swap_values(X, idx_col):
    _X = X.copy()
    vals = _X[idx_col + 1].astype(int)
    _X[idx_col + 1][vals == 0] = 1
    _X[idx_col + 1][vals == 1] = 0
    return _X


def pred_consistency(model, X, theta, idx_sensitive_cols):
    X_default = X.copy()
    X_swaped = X.copy()
    for idx_col in idx_sensitive_cols:
        X_swaped = swap_values(X_swaped, idx_col)
    true_preds = model.sigmoid(theta, X_default)
    s_preds = model.sigmoid(theta, X_swaped)
    pc = np.sum(get_hard_preds(true_preds) == get_hard_preds(s_preds)) / len(
        true_preds)
    return pc


def get_hard_preds(preds):
    return np.where(preds < .5, 0, 1)


def get_params(xtr, xte, ytr, yte):
    x_tr_te = pd.concat((xtr, xte))
    y_tr_te = np.append(ytr, yte)

    X_tr, K_tr, L_tr = get_params_X(xtr)
    X_te, K_te, L_te = get_params_X(xte)
    X_tr_te, K_tr_te, L_tr_te = get_params_X(x_tr_te)
    res = {}
    res["tr"] = (X_tr, K_tr, L_tr)
    res["te"] = (X_te, K_te, L_te)
    res["tr_te"] = (X_tr_te, K_tr_te, L_tr_te)
    res["y"] = (ytr, yte, y_tr_te)
    return res


def plot_res(scores, save=False):
    fig = plt.figure(figsize=(12, 8))
    scores_labs = [
        "AUC", "FNR", "FPR", "PC", "FG", "FGN", "AUC-PC", "AUC-FGN", "Loss"
    ]
    n = len(scores["auc_s"])
    x = np.arange(1, n + 1)
    for i, lab in enumerate(scores_labs):
        ax = fig.add_subplot(3, 3, i + 1)
        if lab == "Loss":
            ax = sns.lineplot(x=x, y=scores[lab.lower()], ax=ax)
            ax.set_ylabel(f"{lab}")
            ax.set_xlabel("Interations")

        elif "-" in lab:
            lab1, lab2 = lab.split('-')
            ax = sns.scatterplot(x=scores[lab2.lower() + "_s"],
                                 y=scores[lab1.lower() + "_s"],
                                 label="Source")
            ax = sns.scatterplot(x=scores[lab2.lower() + "_t"],
                                 y=scores[lab1.lower() + "_t"],
                                 label="Target")
            ax.set_xlabel(lab2)
            ax.set_ylabel(lab1)
        else:
            ax = sns.lineplot(x=x,
                              y=scores[lab.lower() + "_s"],
                              ax=ax,
                              label="Source")
            ax = sns.lineplot(x=x,
                              y=scores[lab.lower() + "_t"],
                              ax=ax,
                              label="Target")
            ax.set_ylabel(f"{lab}")
            ax.set_xlabel("Interations")
    fig.tight_layout()


def plot_res(scores, save=False):
    fig = plt.figure(figsize=(12, 8))
    scores_labs = [
        "AUC", "FNR", "FPR", "PC", "FG", "FGN", "AUC-PC", "AUC-FGN", "Loss"
    ]
    n = len(scores["auc_t"])
    x = np.arange(1, n + 1)
    for i, lab in enumerate(scores_labs):
        ax = fig.add_subplot(3, 3, i + 1)
        if lab == "Loss":
            ax = sns.lineplot(x=x, y=scores[lab.lower()], ax=ax)
            ax.set_ylabel(f"{lab}")
            ax.set_xlabel("Interations")

        elif "-" in lab:
            lab1, lab2 = lab.split('-')
            ax = sns.scatterplot(x=scores[lab2.lower() + "_s"],
                                 y=scores[lab1.lower() + "_s"],
                                 label="Source")
            ax = sns.scatterplot(x=scores[lab2.lower() + "_t"],
                                 y=scores[lab1.lower() + "_t"],
                                 label="Target")
            ax.set_xlabel(lab2)
            ax.set_ylabel(lab1)
        else:
            ax = sns.lineplot(x=x,
                              y=scores[lab.lower() + "_s"],
                              ax=ax,
                              label="Source")
            ax = sns.lineplot(x=x,
                              y=scores[lab.lower() + "_t"],
                              ax=ax,
                              label="Target")
            ax.set_ylabel(f"{lab}")
            ax.set_xlabel("Interations")
    fig.tight_layout()


def plot_res2(scores, title=None, save=False):
    fig = plt.figure(figsize=(12, 8))
    scores_labs = [
        "AUC", "FNR", "FPR", "PC", "FG", "FGN", "AUC-PC", "AUC-FGN", "Loss"
    ]
    n = len(scores["IFDA"]["auc_t"])
    x = np.arange(1, n + 1)
    for i, lab in enumerate(scores_labs):
        ax = fig.add_subplot(3, 3, i + 1)
        for method in scores:
            if lab == "Loss":
                ax = sns.lineplot(x=x,
                                  y=scores[method][lab.lower()],
                                  ax=ax,
                                  label=method)
                ax.set_ylabel(f"{lab}")
                ax.set_xlabel("Interations")

            elif "-" in lab:
                lab1, lab2 = lab.split('-')
                ax = sns.scatterplot(x=scores[method][lab2.lower() + "_t"],
                                     y=scores[method][lab1.lower() + "_t"],
                                     label=method)
                #ax = sns.scatterplot(x=scores[method][lab2.lower() + "_t"], y=scores[method][lab1.lower() + "_t"], label=method + "-I")
                ax.set_xlabel(lab2)
                ax.set_ylabel(lab1)
            else:
                ax = sns.lineplot(x=x,
                                  y=scores[method][lab.lower() + "_t"],
                                  ax=ax,
                                  label=method)
                #ax = sns.lineplot(x=x, y=scores[method][lab.lower() + "_t"], ax=ax, label=method + "-I")
                ax.set_ylabel(f"{lab}")
                ax.set_xlabel("Interations")
    fig.tight_layout()
    if save and title:
        fig.savefig(title + ".png", dpi=100, bbox_inches="tight")


def save_res(res, filename):
    dbfile = open(filename + ".pickle", 'ab')
    pickle.dump(res, dbfile)
    dbfile.close()


def get_score_summary(scores):
    scores_labs = ["AUC", "FNR", "FPR", "PC", "FG", "FGN"]
    for lab in scores_labs:
        print(lab, "Mean: ", np.round(scores[lab.lower() + "_t"].mean(), 3),
              "STD: ", np.round(scores[lab.lower() + "_t"].std(), 3))


def get_inverse_propensity_weights(xtr, xte, seed=42):
    X = pd.concat((xtr, xte))
    n = xtr.shape[0]
    n_rows = X.shape[0]
    y = np.zeros(
        n_rows)  # Enconding the train and target data respectively to 0 and 1
    y[n:] = 1

    LR = LogisticRegression(random_state=seed)
    w = LR.fit(X, y).predict_proba(X)[:, 1]
    return 1. / w[:n]


def get_auc_fp_fn_scores(ytr, ytr_pred, yte, yte_pred):
    ytr_pred_h = get_hard_preds(ytr_pred)
    yte_pred_h = get_hard_preds(yte_pred)

    tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(ytr, ytr_pred_h).ravel()
    tn_te, fp_te, fn_te, tp_te = confusion_matrix(yte, yte_pred_h).ravel()

    fn_tr_r = fn_tr / (fn_tr + tp_tr)
    fp_tr_r = fp_tr / (fp_tr + tn_tr)
    fn_te_r = fn_te / (fn_te + tp_te)
    fp_te_r = fp_te / (fp_te + tn_te)

    auc_tr, auc_te = roc_auc_score(ytr, ytr_pred), roc_auc_score(yte, yte_pred)
    scores = np.round(auc_tr,
                      2), np.round(auc_te, 2), np.round(fn_tr_r, 2), np.round(
                          fn_te_r, 2), np.round(fp_tr_r,
                                                2), np.round(fp_te_r, 2)
    return scores


def get_pc(true_labs, pred_labs):
    pc = np.sum(get_hard_preds(true_labs) == get_hard_preds(pred_labs)) / len(
        true_labs)
    return np.round(pc, 2)


def get_lr_scores(xtr, ytr, xte, yte, sensitive_cols, seed=42, maxiter=200):
    LR = LogisticRegression(random_state=seed, penalty=None, max_iter=maxiter)
    LR.fit(xtr, ytr)
    ytr_pred = LR.predict_proba(xtr)[:, 1]
    yte_pred = LR.predict_proba(xte)[:, 1]
    auc_tr, auc_te, fn_tr_r, fn_te_r, fp_tr_r, fp_te_r = get_auc_fp_fn_scores(
        ytr, ytr_pred, yte, yte_pred)
    print("Scores with swaps")
    print("FNR TR", fn_tr_r, "FPR TR", fp_tr_r)
    print("FNR TE", fn_te_r, "FPR TE", fp_te_r)
    print("AUC TR", auc_tr)
    print("AUC TE", auc_te)

    for col in sensitive_cols:
        x_te_swaped = xte.copy()
        x_te_swaped[col] = x_te_swaped[col].map({0: 1, 1: 0})
        yte_pred_s = LR.predict_proba(x_te_swaped)[:, 1]
        print(f"PC {col}", get_pc(yte_pred, get_hard_preds(yte_pred_s)))

    x_te_swaped_all = xte.copy()
    x_te_swaped_all[sensitive_cols] = x_te_swaped_all[sensitive_cols].replace({
        0:
        1,
        1:
        0
    })
    yte_pred_s_all = LR.predict_proba(x_te_swaped_all)[:, 1]
    print(f"PC {sensitive_cols}",
          get_pc(yte_pred, get_hard_preds(yte_pred_s_all)))
