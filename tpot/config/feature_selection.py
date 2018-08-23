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

# All feature selection tools
fs_config = ['tpot.operator_utils.TPOT_RFE',
             'tpot.operator_utils.TPOT_SelectFromModel',
             'tpot.operator_utils.TPOT_SelectFwe',
             'tpot.operator_utils.TPOT_SelectPercentile',
             'tpot.operator_utils.TPOT_VarianceThreshold',
             'tpot.operator_utils.TPOT_SelectKBest',
             'tpot.operator_utils.TPOT_SelectKFromModel',
             'tpot.operator_utils.TPOT_ForwardSelector',
             'tpot.operator_utils.TPOT_PTA',
             'tpot.operator_utils.TPOT_FloatingSelector',
             ]

order_config = ['tpot.operator_utils.TPOT_FeatureOrderer']

fs_pipeline_steps = ['rfe',
             'selectfrommodel',
             'selectfwe',
             'selectpercentile',
             'variancethreshold',
             'selectkbest',
             'selectkfrommodel',
             'forwardselector',
             'pta',
             'floatingselector',
             ]