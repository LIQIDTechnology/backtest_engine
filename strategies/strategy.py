from pathlib import Path
from typing import Union
from functools import lru_cache

import numpy as np
import pandas as pd
import datetime as dt

from portfolio import Portfolio
from instrument import Instrument


class Strategy(Portfolio):

    def __init__(self, config_path: Union[str, Path], scale_unit: float):
        super().__init__(config_path)
        self.unit3_scale = scale_unit

    def check_rebal(self, t):
        bool = False
        for cluster in self.unit3_ls:
            weight = self.details_np[t-1, self.unit3_idx[cluster]].sum()
            cluster_wt = self.unit3_weights[cluster]
            thres = self.unit3_thres[cluster]
            if cluster_wt - thres < weight < cluster_wt + thres:
                self.details_np[t, self.unit3_thres_breach[cluster]] = 0
            else:
                self.details_np[t, self.unit3_thres_breach[cluster]] = 1
                bool = True
                self.details_np[t, self.rebalance_col] = 1
        return bool

    def routine(self, t):
        self.reset_weights(t - 1) if self.details.index[t] == self.start_date else None  # INIT WEIGHTS
        self.calc_portfolio_ret(t)
        self.reset_weights(t) if self.check_rebal(t) else self.calc_weights(t)