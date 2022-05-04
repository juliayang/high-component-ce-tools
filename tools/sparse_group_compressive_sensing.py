import numpy as np
from sklearn.linear_model import *
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import copy

__author__ = "Julia Yang"
__date__ = '2021/05/18'

# Note that the directions here are very sparse as the code was not written for
# distribution. Please ask J. Y. if you need to modify or clarify anything.

def make_groups(sw, ignore = 1):
    groups = []
    # ignore tells us to ignore the point terms as they have been regressed on already
    group_id = 1  # start by grouping the pairs together

    # groups all bit combos together
    for orbit_size in sw.cluster_subspace.orbits_by_size:
        if orbit_size == ignore: continue
        for orbit in sw.cluster_subspace.orbits_by_size[orbit_size]:
            group_id += 1
            for bc in orbit.bit_combos:
                groups.append(group_id)
    return groups

def zero_coefficients(beta, cutoff = 1E-5):
    to_prune = []
    new_beta = []
    for i_coef in range(len(beta)):
        if np.abs(beta[i_coef]) <= cutoff:
            to_prune.append(i_coef)
            new_beta.append(0.0)
        else:
            new_beta.append(beta[i_coef])
    return to_prune

class SparseGroupLasso(object):
    def __init__(self, groups, lambd, coef_ = None, alpha = None, intercept = None):
        self.groups = groups
        self.lambd = lambd  # amount of L1 to mix in
        self.coef_ = coef_
        self.alpha = alpha
        self.intercept = intercept

    def set_params(self, alpha):
        self.alpha = alpha

    def do_soft_thresholding_operator(self, z, alphaXlambd):
        return (np.sign(z) * (np.abs(z) - alphaXlambd))

    def fit(self, X, Y, ):
        # same thing as evaluate
        assert X.shape[1] == len(self.groups)
        self.X = X
        self.Y = Y
        lossFunction = 0
        intercept = 0.
        all_coef = []
        #alpha = self.alpha[0]
        alpha = self.alpha
        cutoff = alpha * (1 - self.lambd)
        Yprime = self.Y
        for iGroup in range(min(self.groups), max(self.groups) + 1):
            l = np.where(self.groups == iGroup)[0]
            S = self.do_soft_thresholding_operator(
                np.dot(self.X[:, l].T, Yprime) / len(Yprime),
                self.lambd * alpha)
            if np.linalg.norm(S, ord=2) < cutoff * len(l) ** 0.5:  # SGL paper cutoff
                coef = np.zeros(len(l))
                l1 = 0  # l1 zeroed out the
                l2 = 0  # zero out the group
            else:
                # do the fitting with group k
                fitElastic = ElasticNet(fit_intercept=False, alpha=alpha, selection='random',
                                        l1_ratio=self.lambd).fit(self.X[:, l],
                                                                 Yprime)
                l1 = alpha * (self.lambd) * \
                     np.linalg.norm(fitElastic.coef_, ord=1)
                l2 = alpha * (1 - self.lambd) * \
                     np.linalg.norm(fitElastic.coef_, ord=2) * np.sqrt(len(l))
                coef = fitElastic.coef_
            # determine if coefficients should be 0, step 2 in 3.2 of SGL paper

            lossFunction += (l1 + l2)
            all_coef.extend(coef)
            coef = coef.reshape(-1, 1)  # added this 2021/04/29
            Yprime -= np.dot(self.X[:, l], coef)  # have to subtract all other group fits
            # intercept += fit.intercept_
        self.coef_ = np.array(all_coef)
        self.intercept = intercept

        return self


    def predict(self, X):
        print ('coef', len(np.where(np.abs(self.coef_)> 1E-5)[0]))
        return np.dot(X, self.coef_.T)

    def evaluate(self, X, Y):
        assert X.shape[1] == len(self.groups)
        self.X = X
        self.Y = Y
        lossFunction = 0
        intercept = 0.
        all_coef = []
        # alpha = self.alpha[0]
        alpha = self.alpha
        cutoff = alpha * (1 - self.lambd)
        Yprime = self.Y
        for iGroup in range(min(self.groups), max(self.groups) + 1):
            l = np.where(self.groups == iGroup)[0]
            S = self.do_soft_thresholding_operator(
                np.dot(self.X[:, l].T, Yprime) / len(Yprime),
                self.lambd * alpha)
            if np.linalg.norm(S, ord=2) < cutoff * len(l) ** 0.5:  # SGL paper cutoff
                coef = np.zeros(len(l))
                l1 = 0  # l1 zeroed out the
                l2 = 0  # zero out the group
            else:
                # do the fitting with group k
                fitElastic = ElasticNet(fit_intercept=False, alpha=alpha, selection='random',
                                        l1_ratio=self.lambd).fit(self.X[:, l],
                                                                 Yprime)
                l1 = alpha * (self.lambd) * \
                     np.linalg.norm(fitElastic.coef_, ord=1)
                l2 = alpha * (1 - self.lambd) * \
                     np.linalg.norm(fitElastic.coef_, ord=2) * np.sqrt(len(l))
                coef = fitElastic.coef_
            # determine if coefficients should be 0, step 2 in 3.2 of SGL paper

            lossFunction += (l1 + l2)
            all_coef.extend(coef)
            coef = coef.reshape(-1, 1)  # added this 2021/04/29
            Yprime -= np.dot(self.X[:, l], coef)  # have to subtract all other group fits
            # intercept += fit.intercept_
        self.coef_ = np.array(all_coef)
        self.intercept = intercept

        return lossFunction + 1/(2 * len(self.Y)) * \
               np.linalg.norm(self.Y - np.dot(self.X, self.coef_),
                                      ord=2)**2

    def score(self, X, Y):
        y_pred = self.predict(X).T
        return np.sqrt(mean_squared_error(Y, y_pred))
        return r2_score(Y, y_pred)

    def get_params(self, deep = False):
        return {'groups': self.groups,
                'coef_': self.coef_,
                'lambd': self.lambd,
                'alpha': self.alpha,
                'intercept': self.intercept}

class SparseGroupLassoWithChosenGroups(object):
    def __init__(self, X, Y, groups, chosen_groups, lambd):
        self.groups = groups
        self.chosen_groups = chosen_groups
        self.X = X
        self.Y = Y
        self.lambd = lambd # amount of L1 to mix in
        self.thinned_X = self.select_groups()

    def select_groups(self):
        groupsToDelete = list(set(self.groups) - set(self.chosen_groups))
        toDelete = []
        toKeep = []
        for i in range(len(self.groups)):
            if self.groups[i] in groupsToDelete:
                toDelete.append(i)
            else:
                toKeep.append(i)
        returnX = np.delete(arr=self.X, obj=toDelete, axis=1)
        self.keep = np.array(toKeep)
        return returnX

    def evaluate(self):
        lossFunction = 0
        intercept = 0.
        all_coef = []
        alpha = self.alpha[0]
        # first within the group
        for iGroup in self.chosen_groups:
            l = np.where(self.groups == iGroup)[0]
            fit = Lasso(fit_intercept=True, alpha = alpha).fit(self.X[:, l],
                                                  self.Y)
            lossFunction += alpha * (self.lambd) * \
                            np.linalg.norm(fit.coef_, ord=1) #! not sure about this
            lossFunction += alpha * (1-self.lambd) * \
                            np.linalg.norm(fit.coef_, ord = 2) * np.sqrt(len(l))
            all_coef.extend(fit.coef_)
            intercept += fit.intercept_
        self.coef_ = np.array(all_coef)
        self.intercept = intercept

        return lossFunction + 1/(2 * len(self.Y)) * \
               np.linalg.norm(self.Y - self.thinned_X @ np.array(all_coef) \
                                      - self.intercept,
                                      ord=2)**2