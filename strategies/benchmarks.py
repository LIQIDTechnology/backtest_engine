import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
from typing import Union
from portfolio import Portfolio


class Benchmark(Portfolio):

    def __init__(self, config_path: Union[str, Path]):
        super().__init__(config_path)
        self.bm_type = self.config['strategy']['benchmark']

    def check_rebal(self, t):
        """
        This functions evaluates day T for a rebalancing.

        1. QUARTERLY
        2. MONTHLY
        3. ANNUALY
        4. BREACH OF 5%
        """
        today = self.details.index[t]
        if today == self.start_date:
            rebal_bool = False
        else:
            td_month = self.details.index[t].month
            ytd_month = self.details.index[t-1].month

            if self.bm_type == "quarterly":
                rebal_bool = True if td_month != ytd_month and td_month in (3, 6, 9, 12) else False
            elif self.bm_type == "monthly":
                rebal_bool = True if td_month != ytd_month else False
            elif self.bm_type == "annually":
                rebal_bool = True if td_month != ytd_month and td_month == 1 else False
            elif self.bm_type == "nothing":
                rebal_bool = False
            elif self.bm_type == "five_percent":
                for cluster in self.unit1_ls:
                    weight = self.details_np[t-1, self.unit1_idx[cluster]].sum()
                    cluster_wt = self.unit1_weights[cluster]
                    thres = 0.05
                    if cluster_wt - thres < weight < cluster_wt + thres:
                        pass
                    else:
                        self.details_np[t, self.unit1_thres_breach[cluster]] = 1
                        self.details_np[t, self.unit1_rebalance_col] = 1
                        self.details_np[t, self.rebalance_col] = 1
                rebal_bool = True if np.nansum(self.details_np[t, self.unit1_col]) >= 1 else False
                self.details_np[t, self.rebalance_col] = 1 if rebal_bool else None
        self.details_np[t, self.rebalance_col] = 1 if rebal_bool else None
        return rebal_bool

    def routine(self, t):
        self.reset_weights(t-1) if self.details.index[t-1] == self.start_date else None  # INIT WEIGHTS
        self.calc_portfolio_ret(t)
        self.reset_weights(t) if self.check_rebal(t) else self.calc_weights(t)
        # self.details.index[t-1]
        # (self.details[self.inst_col].notna().idxmax())



