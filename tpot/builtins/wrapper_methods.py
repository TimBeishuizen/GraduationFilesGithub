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
from sklearn.feature_selection.base import SelectorMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

from .order_methods import FeatureOrderer

import numpy as np
import copy


class WrapperSelector(BaseEstimator, SelectorMixin):
    """

    """

    def __init__(self, threshold=0.01, score_func=GaussianNB(), order_func=None, cv_groups=5):
        self.score_func = score_func
        self.threshold = threshold
        self.order_func = order_func
        self.cv_groups = cv_groups
        self.selected_features = None
        self.score = 0

    def _get_support_mask(self):
        """

        :return:
        """

        if self.selected_features == None:
            raise ValueError("First fit the model before transform")

        return self.selected_features

    def _compute_score(self, X, y):
        """

        :param X: The dataset input
        :param y: The output
        :return: The score using both
        """

        if X.shape[0] == 0:
            return 0
        else:
            return np.mean(cross_val_score(estimator=self.score_func, X=X, y=y, cv=self.cv_groups))



class ForwardSelector(WrapperSelector):
    """

    """

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """

        # Loop through all features to find the best
        self.selected_features = np.full(X.shape[1], False, dtype=bool)

        self._forward_selection(X, y)

        return self

    def _forward_selection(self, X, y):
        """

        :param X:
        :param y:
        :param curr_loc:
        :return:
        """

        for i in range(X.shape[1]):

            # If feature already used (useful for further implementations)
            if self.selected_features[i]:
                continue

            # Create a new candidate feature set
            candidate_feature_set = copy.copy(self.selected_features)
            candidate_feature_set[i] = True

            # Compute the new score with the added feature
            new_score = self._compute_score(X[:, candidate_feature_set], y)

            # Add feature if chance is big enough
            if new_score - self.threshold > self.score:
                self.selected_features[i] = True
                self.score = new_score

        return


class BackwardSelector(WrapperSelector):
    """

    """

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """

        # Loop through all features to remove the worst
        self.selected_features = np.full(X.shape[1], True, dtype=bool)

        self._backward_selection(X, y)

        return self

    def _backward_selection(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """

        # Loop through processed features to remove the worst
        for i in reversed(range(X.shape[1])):

            # If already false (useful for further implementations)
            if not self.selected_features[i]:
                continue

            # Create a new candidate feature set
            candidate_feature_set = copy.copy(self.selected_features)
            candidate_feature_set[i] = False

            # Compute the new score with the removed feature
            new_score = self._compute_score(X[:, candidate_feature_set], y)

            # Remove feature if chance is big enough
            if new_score + self.threshold > self.score:
                self.selected_features[i] = False
                self.score = new_score

        return


class PTA(WrapperSelector):
    """

    """

    def __init__(self, threshold=0.01, l=5, r=2, score_func=GaussianNB(), cv_groups=5):
        """

        :param threshold:
        :param l:
        :param r:
        :param score_func:
        :param cv_groups:
        """

        super().__init__(threshold=threshold, score_func=score_func, cv_groups=cv_groups)

        self.l = l
        self.r = r

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """

        # Loop through all features to find the best
        self.selected_features = np.full(X.shape[1], False, dtype=bool)

        curr_loc = 0

        # While not all locations are yet investigated, use forward and backward selection
        while curr_loc < X.shape[0]:
            curr_loc = self._forward_selection(X, y, curr_loc)
            self._backward_selection(X, y, curr_loc)

        return self

    def _forward_selection(self, X, y, curr_loc):
        """

        :param X:
        :param y:
        :param curr_loc:
        :return:
        """

        curr_l = 0

        for i in range(curr_loc, X.shape[1]):

            # Create a new candidate feature set
            candidate_feature_set = copy.copy(self.selected_features)
            candidate_feature_set[i] = True

            # Compute the new score with the added feature
            new_score = self._compute_score(X[:, candidate_feature_set], y)

            # Add feature if chance is big enough
            if new_score - self.threshold > self.score:
                self.selected_features[i] = True
                self.score = new_score
                curr_l += 1

            # Start backward selection if l are picked
            if curr_l == self.l:
                return i

        return X.shape[0] + 1

    def _backward_selection(self, X, y, curr_loc):
        """

        :param X:
        :param y:
        :return:
        """

        curr_r = 0

        # Loop through processed features to remove the worst
        for i in reversed(range(curr_loc)):

            # If already false (useful for further implementations)
            if not self.selected_features[i]:
                continue

            # Create a new candidate feature set
            candidate_feature_set = copy.copy(self.selected_features)
            candidate_feature_set[i] = False

            # Compute the new score with the removed feature
            new_score = self._compute_score(X[:, candidate_feature_set], y)

            # Remove feature if chance is big enough
            if new_score + self.threshold > self.score:
                self.selected_features[i] = False
                self.score = new_score
                curr_r += 1

            # Stop backward selection when r are removed
            if curr_r == self.r:
                return

        return


class FloatingSelector(ForwardSelector, BackwardSelector):
    """

    """

    def __init__(self, threshold=0.01, max_iter=100, score_func=GaussianNB(), cv_groups=5):
        """

        :param threshold:
        :param max_iter:
        :param score_func:
        :param cv_groups:
        """

        super(ForwardSelector, self).__init__(threshold=threshold, score_func=score_func, cv_groups=cv_groups)

        self.max_iter = max_iter

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """

        # Loop through all features to find the best
        self.selected_features = np.full(X.shape[1], False, dtype=bool)

        # Continue until the number of maximum iterations is reached
        for i in range(self.max_iter):

            current_set = copy.copy(self.selected_features)

            self._forward_selection(X, y)
            self._backward_selection(X, y)

            # If no change was present this iteration
            if np.array_equal(current_set, self.selected_features):
                return self

        return self
