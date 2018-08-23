# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection.base import SelectorMixin, TransformerMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif

from sklearn.preprocessing import StandardScaler

import numpy as np
import copy

class FeatureOrderer(BaseEstimator, TransformerMixin):
    """

    """

    def __init__(self, score_func):

        """

        :param estimator:
        """

        self.score_func = score_func
        self.order = None

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """

        # Find out if random or not
        if str(self.score_func) == 'random':
            score_func_ret = np.random.permutation(X.shape[1])
        else:
            score_func_ret = self.score_func(X, y)

        # Give scores based to the score function
        if isinstance(score_func_ret, (list, tuple)):
            self.scores_, self.pvalues_ = score_func_ret
            self.pvalues_ = np.asarray(self.pvalues_)
        else:
            self.scores_ = score_func_ret
            self.pvalues_ = None

        return self


    def transform(self, X, y=None):
        """

        :param X:
        :param y:
        :return:
        """

        order = np.argsort(-self.scores_)

        return X[:, order]


