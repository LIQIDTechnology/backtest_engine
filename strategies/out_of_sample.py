from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import datetime as dt

from portfolio import Portfolio
from threshold_optim import ThresholdOptimizer


class StrategyOutOfSample(Portfolio):

    def __init__(self, config_path: Union[str, Path], scale_unit: float, optim_type: str):
        super().__init__(config_path)
        self.unit3_scale = scale_unit
        self.optim_type = optim_type

    def set_unit1_threshold(self):
        for cluster in self.unit1_ls:
            self.unit1_thres[cluster] = 0.05

    def set_unit2_threshold(self):

        for cluster in self.unit2_ls:
            if cluster in ["GOLD", "CASH", "COM"]:
                self.unit2_thres[cluster] = 0.01
            else:
                weight_rk_curr = self.instruments_table[self.instruments_table["UNIT II"] == cluster][self.strategy_risk_class].sum()
                weight_rk_next = self.instruments_table[self.instruments_table["UNIT II"] == cluster][str(int(self.strategy_risk_class) + 10)].sum()
                self.unit2_thres[cluster] = abs(weight_rk_curr - weight_rk_next) / 2

    def set_unit3_threshold(self):
        # THRESHOLD UNIT III
        for cluster in self.unit3_ls:
            unit2_cluster = self.unit3to2_map[cluster]
            unit2_wt = self.unit2_weights[unit2_cluster]
            unit3_wt = self.unit3_weights[cluster]
            self.unit3_thres[cluster] = unit3_wt / unit2_wt * self.unit3_scale

    def check_rebal_cluster(self, cluster_str, t):
        """

        @param cluster_str:
        @param t:
        @return:
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

        @param t:
        @return:
        """
        self.check_rebal_cluster("unit1", t)
        self.check_rebal_cluster("unit2", t)
        self.check_rebal_cluster("unit3", t)

        rebal_ls = [*self.unit1_col, *self.unit2_col, *self.unit3_col]
        rebal_bool = True if np.nansum(self.details_np[t, rebal_ls]) >= 1 else False
        self.details_np[t, self.rebalance_col] = 1 if rebal_bool else None

        return rebal_bool

    def routine(self, t):
        self.reset_weights(t-1) if self.details.index[t] == self.start_date else None  # INIT WEIGHTS
        self.calc_portfolio_ret(t)
        self.reset_weights(t) if self.check_rebal(t) else self.calc_weights(t)
        if self.optim_type == 'expanding':
            start_date = self.start_date
            if self.details.index[t].year != self.details.index[t - 1].year:
                study = ThresholdOptimizer(start_date=start_date, end_date=self.details.index[t])
                res = study.threshold_optimum()
                print(self.details.index[t].year, res.x)
                self.unit3_scale = res.x
        elif self.optim_type == 'rolling':
            check_ls = []
            for k in range(1, 200):
                year = self.start_date.year + k
                month = self.start_date.month
                day = self.start_date.day
                check_ls.append(dt.date(year, month, day))

            if self.details.index[t] in check_ls:
                end_date = self.details.index[t]
                year = self.details.index[t-1].year
                month = 1
                day = 1
                start_date = dt.date(year, month, day)
                study = ThresholdOptimizer(start_date=start_date, end_date=self.details.index[t])
                res = study.threshold_optimum()
                print(self.details.index[t].year, res.x)
                self.unit3_scale = res.x


if __name__ == "__main__":
    # Example Execution of StrategyOutOfSample Object
    config_path = Path(__file__).parents[1] / 'config/config_strategy.ini'
    start_scale_value = 0.05
    # expanding, rolling
    strategy = StrategyOutOfSample(config_path=config_path, scale_unit=start_scale_value, optim_type='rolling')
    strategy.manage_portfolio()
    strategy.export_files()

    # Output KPIs into DataFrame
    kpi_dic = strategy.get_kpi()
    kpi_df = pd.DataFrame([kpi_dic])
    folderpath = strategy.root_path
    filename = "kpi_summary_out_of_sample.csv"
    kpi_df.to_csv(folderpath / filename)
