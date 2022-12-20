import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
from typing import Union
from portfolio import Portfolio
from statistics import Statistic


class Benchmark(Portfolio):
    """
    Benchmark Strategy, includes strategies with simple rebalancing logics

    """
    def __init__(self, config_path: Union[str, Path]):
        """
        Declaring variables upon Object Initialization
        """
        super().__init__(config_path)
        self.bm_type = None
        self.strategy_name = f"{self.bm_type}_rebalance"
        self.unit1_ls = None
        self.unit1_idx = {}
        self.unit1_weights = {}
        self.unit1_thres_breach = {}
        self.unit1_rebalance_col = None
        self.unit1_col = None
        self.unit1_thres = {}

    def init_details_strat(self):
        self.set_unit_ls()
        strat_ls = self.unit1_ls
        other_ls = ["UNIT1 Rebalance"]

        res_ls = [x for x in [*strat_ls, *other_ls] if x is not None]
        return res_ls

    def wake_up_strat(self):
        self.unit1_col = [self.details.columns.get_loc(col) for col in self.unit1_ls]
        self.set_unit1_threshold()

    def set_unit_ls(self):
        self.unit1_ls = list(set(inst.unit1 for inst in self.instrument_ls if inst.unit1 is not None))

    def set_unit1_threshold(self):
        """
        Set Unit I Threshold, set to 5%
        """

        def get_cluster_weight(instruments_table, rc, cluster):
            """This function gets the weight on Cluster Level"""
            wts_sum = instruments_table[instruments_table["UNIT I"] == cluster][rc].sum()
            return wts_sum

        for cluster in self.unit1_ls:
            this_rc = int(self.strategy_risk_class)
            this_rc = 90 if this_rc == 100 else this_rc
            next_rc = this_rc + 10
            curr_cluster_weight = get_cluster_weight(self.instruments_table, str(this_rc), cluster)
            next_cluster_weight = get_cluster_weight(self.instruments_table, str(next_rc), cluster)
            self.unit1_thres[cluster] = abs(curr_cluster_weight - next_cluster_weight) / 2

    def check_cash(self, t):
        cash_wt_col = self.details.columns.get_loc("LS01TREU Index Weight")

        cash_wt_tm1 = self.details_np[t-1, cash_wt_col]

        rebal_bool = True if cash_wt_tm1 < 0.005 else False

        return rebal_bool

    def check_rebal(self, t):
        """
        Daily Method to identify Rebalance Triggers based on Benchmark Type

        1. QUARTERLY
        2. MONTHLY
        3. ANNUALY
        4. BREACH OF 5%
        """
        today = self.details.index[t]  # Today
        rebal_bool = None
        if today == self.start_date:
            rebal_bool = False
        else:
            td_month = self.details.index[t].month  # Today's Month
            try:
                ytd_month = self.details.index[t - 1].month  # Yesterday's Month
            except IndexError:
                ytd_month = td_month

            if self.bm_type == "quarterly":
                rebal_bool = True if td_month != ytd_month and td_month in (3, 6, 9, 12) else False
                rebal_bool = True if rebal_bool else self.check_cash(t)
            elif self.bm_type == "monthly":
                rebal_bool = True if td_month != ytd_month else False
                rebal_bool = True if rebal_bool else self.check_cash(t)
            elif self.bm_type == "annually":
                rebal_bool = True if td_month != ytd_month and td_month == 1 else False
                rebal_bool = True if rebal_bool else self.check_cash(t)
            elif self.bm_type == "never":
                rebal_bool = False
                rebal_bool = True if rebal_bool else self.check_cash(t)
            elif self.bm_type == "five_percent":
                def create_map(instrument_ls, cluster, details_df, level):
                    cluster_idx = {}
                    for cl in cluster:
                        cluster_ls = []
                        for inst in instrument_ls:
                            dir(inst)
                            if getattr(inst, f"unit{level}") == cl:
                                cluster_ls.append(details_df.columns.get_loc(f"{inst.ticker} Weight"))
                        cluster_idx[cl] = cluster_ls
                    return cluster_idx

                self.unit1_idx = create_map(self.instrument_ls, self.unit1_ls, self.details, 1)

                for cluster in self.unit1_ls:
                    wts_sum = self.instruments_table[self.instruments_table["UNIT I"] == cluster][self.strategy_risk_class].sum()
                    self.unit1_weights[cluster] = wts_sum

                self.unit1_thres_breach = {cluster: self.details.columns.get_loc(cluster) for cluster in self.unit1_ls}
                self.unit1_rebalance_col = self.details.columns.get_loc("UNIT1 Rebalance")

                for cluster in self.unit1_ls:
                    weight = self.details_np[t-1, self.unit1_idx[cluster]].sum()
                    cluster_wt = self.unit1_weights[cluster]
                    thres = 0.05
                    if cluster_wt - thres < weight < cluster_wt + thres:
                        pass  # If within Threshold Range do Nothing
                    else:
                        self.details_np[t, self.unit1_thres_breach[cluster]] = 1
                        self.details_np[t, self.unit1_rebalance_col] = 1
                        self.details_np[t, self.rebalance_col] = 1
                rebal_bool = True if np.nansum(self.details_np[t, self.unit1_col]) >= 1 else False
                rebal_bool = True if rebal_bool else self.check_cash(t)
                self.details_np[t, self.rebalance_col] = 1 if rebal_bool else None
        self.details_np[t, self.rebalance_col] = 1 if rebal_bool else None
        return rebal_bool

    def routine(self, t):
        """
        Daily Routine what is calculated on each day / over each row
        """
        if self.details.index[t] == self.start_date:  # INIT WEIGHTS
            self.reset_weights(t - 1)
            self.details_np[0, self.pf_ret_col] = 0
            self.details_np[0, self.pf_cum_ret_col] = 0
            self.details_np[0, self.inv_col_np] = self.details_np[0, self.wts_col_np] * self.amount_inv
            self.details_np[0, self.hyp_amount_inv_col] = self.details_np[0, self.inv_col_np].sum()
        self.calc_portfolio_ret(t)
        # self.reset_weights(t) if self.check_rebal(t) else self.calc_weights(t)


if __name__ == "__main__":
    # Example Execution of Benchmark Object
    config_folder = Path(__file__).parents[1]
    config_path = config_folder / 'config/config_benchmark.ini'  # same for all benchmark types
    # quarterly, monthly, annually, nothing, five_percent"
    benchmark = Benchmark(config_path=config_path)
    bm_list = ["quarterly", "monthly", "annually", "nothing", "five_percent"]
    for bm in bm_list:
        benchmark.bm_type = bm
        benchmark.strategy_name = f"{bm}_rebalance"
        root_name = benchmark.strategy_name
        benchmark.strategy_risk_class = "100"
        benchmark.manage_portfolio()  # Returns a KPI Dictionary
        benchmark.export_files()  # Exports Detail Sheet (also triggered in manage_portfolio)
        kpi_dic = Statistic(benchmark).get_kpi()

    # Output KPIs into DataFrame
    # kpi_dic = benchmark_never.get_kpi()  # Returns a KPI Dictionary
    # kpi_df = pd.DataFrame([kpi_dic])  # Convert to DataFrame
    # folderpath = benchmark_never.root_path
    # filename = "benchmark_never.csv"
    # kpi_df.to_csv(folderpath / filename)
    # print(f'KPIs exported to {folderpath / filename}')
