import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import utils
import ifda, if2r


def get_scores(X_tr_cr,
               y_tr_cr,
               X_te_cr,
               y_te_cr,
               Wcr,
               sensitive_atts,
               init_theta="zeros"):
    for att, idx_cols in sensitive_atts.items():
        # Running an IFDA model
        ifda_cr = ifda.IFDA(alpha=alpha,
                            lr=lr,
                            idx_sensitive_cols=idx_cols,
                            metric=metric,
                            delta=delta,
                            q=q,
                            maxiter=maxiter,
                            seed=seed,
                            init_theta=init_theta)
        ifda_cr.fit(X_tr_cr, y_tr_cr, X_te_cr, y_te_cr)

        ifda_cr_0 = ifda.IFDA(alpha=0,
                              lr=lr,
                              idx_sensitive_cols=idx_cols,
                              metric=metric,
                              delta=delta,
                              q=q,
                              maxiter=maxiter,
                              seed=seed,
                              init_theta=init_theta)
        ifda_cr_0.fit(X_tr_cr, y_tr_cr, X_te_cr, y_te_cr)

        # Computing the normalize FS scores and plotting all scores
        ifda_cr.scores["fgn_s"] = np.array(ifda_cr.scores["fg_s"]) - np.array(
            ifda_cr_0.scores["fg_s"])[-1]
        ifda_cr.scores["fgn_t"] = np.array(ifda_cr.scores["fg_t"]) - np.array(
            ifda_cr_0.scores["fg_t"])[-1]
        #utils.plot_res(ifda_cr.scores, save=False)

        # Running an IF2R model
        if2r_cr = if2r.IF2R(alpha=alpha,
                            lr=lr,
                            idx_sensitive_cols=idx_cols,
                            metric=metric,
                            delta=delta,
                            q=q,
                            maxiter=maxiter,
                            seed=seed,
                            init_theta=init_theta)
        if2r_cr.fit(X_tr_cr, y_tr_cr, X_te_cr, y_te_cr, Wcr)

        if2r_cr_0 = if2r.IF2R(alpha=0,
                              lr=lr,
                              idx_sensitive_cols=idx_cols,
                              metric=metric,
                              delta=delta,
                              q=q,
                              maxiter=maxiter,
                              seed=seed,
                              init_theta=init_theta)
        if2r_cr_0.fit(X_tr_cr, y_tr_cr, X_te_cr, y_te_cr, Wcr)
        if2r_cr.predict_proba(X_te_cr,
                              y_te_cr,
                              lr=lr,
                              maxiter=maxiter,
                              tol=1e-10)

        # Computing the normalize FS scores and plotting all scores
        if2r_cr.scores_s["fgn_s"] = np.array(
            if2r_cr.scores_s["fg_s"]) - np.array(
                if2r_cr_0.scores_s["fg_s"])[-1]
        if2r_cr.scores_s["fgn_t"] = np.array(
            if2r_cr.scores_s["fg_t"]) - np.array(
                if2r_cr_0.scores_s["fg_t"])[-1]
        if2r_cr.scores_t["fgn_t"] = np.array(
            if2r_cr.scores_t["fg_t"]) - np.array(
                if2r_cr_0.scores_s["fg_t"])[-1]
        scores = {
            "IFDA": ifda_cr.scores,
            "IF2R-TR": if2r_cr.scores_s,
            "IF2R-IN": if2r_cr.scores_t
        }
        print(f"Summary IFDA {att}")
        utils.get_score_summary(ifda_cr.scores)

        print(f"Summary IF2R {att}")
        utils.get_score_summary(if2r_cr.scores_t)
        print("\n")
        utils.plot_res2(scores, title=att, save=True)


seed = 42
np.random.seed(seed)

credit = pd.read_csv("../Dataset/credit.csv")
y_cr = credit.credit
y_cr = y_cr.map({1: 0, 2: 1})
X_cr = credit.drop("credit", axis=1)
X_cr = utils.scale(X_cr)

# IID setting
X_tr_cr, X_te_cr, y_tr_cr, y_te_cr = train_test_split(X_cr,
                                                      y_cr,
                                                      test_size=0.3,
                                                      random_state=42)
res_cr = utils.get_params(X_tr_cr, X_te_cr, y_tr_cr, y_te_cr)

# CS setting
age_cr = X_cr.age
X_cr_s = X_cr.drop("age", axis=1)
X_cr_shift_tr, y_cr_shift_tr = X_cr_s[age_cr == 0], y_cr[age_cr == 0]
X_cr_shift_te, y_cr_shift_te = X_cr_s[age_cr == 1], y_cr[age_cr == 1]
res_cr_shift = utils.get_params(X_cr_shift_tr, X_cr_shift_te, y_cr_shift_tr,
                                y_cr_shift_te)

# Getting sensitive attributes column indexes
idx_col_cr_age = X_cr.columns.get_loc("age")
idx_col_cr_sex = X_cr.columns.get_loc("sex")

idx_col_cr_shift_sex = X_cr_shift_tr.columns.get_loc("sex")

# Getting scores from a Logistic Regression model
utils.get_lr_scores(X_tr_cr,
                    y_tr_cr,
                    X_te_cr,
                    y_te_cr,
                    sensitive_cols=["sex", "age"])

# Setting some of IF2R and IFDA hyperparams
lr = 1e-1
metric = "euclidean"
delta = 5
maxiter = 200
Wcr = np.ones(X_tr_cr.shape[0])
Wcr_shift = utils.get_inverse_propensity_weights(X_cr_shift_tr,
                                                 X_cr_shift_te,
                                                 seed=seed)
alpha = 10
q = 1
sensitive_atts_ID = {
    "sex": [idx_col_cr_sex],
    "age": [idx_col_cr_age],
    "sex-age": [idx_col_cr_sex, idx_col_cr_age]
}
sensitive_atts_CS = {"sex": [idx_col_cr_shift_sex]}

if __name__ == "__main__":
    # Get scores under the IID setting
    get_scores(X_tr_cr, y_tr_cr, X_te_cr, y_te_cr, Wcr, sensitive_atts_ID)

    # Get scores under the CS setting
    get_scores(X_cr_shift_tr, y_cr_shift_tr, X_cr_shift_te, y_cr_shift_te,
               Wcr_shift, sensitive_atts_CS)
