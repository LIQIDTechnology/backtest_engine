from pathlib import Path
from typing import Union
from functools import lru_cache

import numpy as np
import pandas as pd
import datetime as dt

from portfolio import Portfolio
from instrument import Instrument


class Strategy(Portfolio):
    """
    Vanilla Strategy, where the Rebalancing Logic is steered over the Threshold Scaling Unit

    """

    def __init__(self, config_path: Union[str, Path], scale_unit: float):
        """
        Declaring variables upon Object Initialization
        """
        super().__init__(config_path)
        self.unit3_scale = scale_unit

    def set_unit1_threshold(self):
        """
        Set Unit I Threshold, set to 5%
        """
        for cluster in self.unit1_ls:
            self.unit1_thres[cluster] = 0.05

    def set_unit2_threshold(self):
        """
        Set UNIT II Threshold, logic derived from current methodology
        """
        for cluster in self.unit2_ls:
            if cluster in ["GOLD", "CASH", "COM"]:
                self.unit2_thres[cluster] = 0.01
            else:
                curr_risk_class = self.strategy_risk_class
                next_risk_class = str(int(self.strategy_risk_class) + 10)  # e.g. 50 + 10
                weight_rk_curr = self.instruments_table[self.instruments_table["UNIT II"] == cluster][curr_risk_class].sum()
                weight_rk_next = self.instruments_table[self.instruments_table["UNIT II"] == cluster][next_risk_class].sum()
                self.unit2_thres[cluster] = abs(weight_rk_curr - weight_rk_next) / 2

    def set_unit3_threshold(self):
        """
        Set UNIT III Threshold, logic derived from current methodology
        """
        for cluster in self.unit3_ls:
            unit2_cluster = self.unit3to2_map[cluster]
            unit2_wt = self.unit2_weights[unit2_cluster]
            unit3_wt = self.unit3_weights[cluster]
            self.unit3_thres[cluster] = unit3_wt / unit2_wt * self.unit3_scale

    def check_rebal_cluster(self, cluster_str, t):
        """
        Method to identify breaches on Thresholds on an explicit UNIT
        """
        unit_ls = getattr(self, f"{cluster_str}_ls")
        unit_idx = getattr(self, f"{cluster_str}_idx")
        unit_weights = getattr(self, f"{cluster_str}_weights")
        unit_thres = getattr(self, f"{cluster_str}_thres")
        unit_thres_breach = getattr(self, f"{cluster_str}_thres_breach")
        unit_rebalance_col = getattr(self, f"{cluster_str}_rebalance_col")

        for cluster in unit_ls:
            weight = self.details_np[t - 1, unit_idx[cluster]].sum()
            cluster_wt = unit_weights[cluster]
            thres = unit_thres[cluster]
            if cluster_wt - thres < weight < cluster_wt + thres:
                pass
            else:
                self.details_np[t, unit_thres_breach[cluster]] = 1
                self.details_np[t, unit_rebalance_col] = 1

    def check_rebal(self, t):
        """
        Daily Method to identify breaches on Thresholds on all UNIT Levels
        """
        self.check_rebal_cluster("unit1", t)
        self.check_rebal_cluster("unit2", t)
        self.check_rebal_cluster("unit3", t)

        rebal_ls = [*self.unit1_col, *self.unit2_col, *self.unit3_col]
        rebal_bool = True if np.nansum(self.details_np[t, rebal_ls]) >= 1 else False
        self.details_np[t, self.rebalance_col] = 1 if rebal_bool else None

        return rebal_bool

    def routine(self, t):
        """
        Daily Routine what is calculated on each day / over each row
        """
        if self.details.index[t] == self.start_date:  # INIT WEIGHTS
            self.details_np[0, self.pf_ret_col] = 0
            self.details_np[0, self.pf_cum_ret_col] = 0
            self.details_np[0, self.hyp_amount_inv_col] = self.amount_inv
            self.reset_weights(t - 1)
        self.calc_portfolio_ret(t)
        self.reset_weights(t) if self.check_rebal(t) else self.calc_weights(t)


if __name__ == "__main__":
    # Example Execution of Strategy Object
    strategy = Strategy(config_path=Path(__file__).parents[1] / 'config/config_strategy.ini', scale_unit=0.020925644969531886)
    strategy.manage_portfolio()  # Returns a KPI Dictionary
    strategy.export_files()  # Exports Detail Sheet (also triggered in manage_portfolio

    # Output KPIs into DataFrame
    kpi_dic = strategy.get_kpi()  # Returns a KPI Dictionary
    kpi_df = pd.DataFrame([kpi_dic])  # Convert to DataFrame
    folderpath = strategy.root_path
    filename = "kpi_summary_optim.csv"
    kpi_df.to_csv(folderpath / filename)