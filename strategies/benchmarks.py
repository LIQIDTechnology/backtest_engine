import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
from typing import Union
from portfolio import Portfolio


class Strategy(Portfolio):

    def __init__(self, config_path: Union[str, Path], scale_unit: float, bm_type: str):
        super().__init__(config_path)
        self.unit3_scale = scale_unit
        self.bm_type = bm_type

    def check_rebal(self, t, bm_type):
        """
        Triggers Rebalancing Quarterly
        """
        today = self.details.index[t]
        if today == self.start_date:
            rebal_bool = False
        else:
            td_month = self.details.index[t].month
            ytd_month = self.details.index[t-1].month

            if bm_type == "quarterly":
                rebal_bool = True if td_month != ytd_month and td_month in (3, 6, 9, 12) else False
            elif bm_type == "monthly":
                rebal_bool = True if td_month != ytd_month else False
            elif bm_type == "annually":
                rebal_bool = True if td_month != ytd_month and td_month == 1 else False
            elif bm_type == "nothing":
                rebal_bool = False

        self.details_np[t, self.rebalance_col] = 1 if rebal_bool else 0

        return rebal_bool

    def check_rebal_monthly(self, t):
        """
        Triggers Rebalancing Monthly
        """
        today = self.details.index[t]
        if today == self.start_date:
            rebal_bool = False
        else:
            td_month = self.details.index[t].month
            ytd_month = self.details.index[t-1].month

        self.details_np[t, self.rebalance_col] = 1 if bool else 0

        return rebal_bool

    def routine(self, t):
        self.reset_weights(t - 1) if self.details.index[t] == self.start_date else None  # INIT WEIGHTS
        self.calc_portfolio_ret(t)
        self.reset_weights(t) if self.check_rebal(t, self.bm_type) else self.calc_weights(t)
