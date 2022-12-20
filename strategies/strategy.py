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

        self.unit1_thres = {}
        self.unit2_thres = {}
        self.unit3_thres = {}

        self.unit1_thres_breach = {}
        self.unit2_thres_breach = {}
        self.unit3_thres_breach = {}

        self.unit1_weights = {}
        self.unit2_weights = {}
        self.unit3_weights = {}

        self.unit1_idx = {}
        self.unit2_idx = {}
        self.unit3_idx = {}

        self.unit3to2_map = {}
        self.unit2to1_map = {}

        self.unit1_col = None
        self.unit2_col = None
        self.unit3_col = None

        self.unit1_rebalance_col = None
        self.unit2_rebalance_col = None
        self.unit3_rebalance_col = None

        self.unit3_ls = None
        self.unit2_ls = None
        self.unit1_ls = None

    def init_details_strat(self):
        self.set_unit_ls()
        strat_ls = [*self.unit3_ls, *self.unit2_ls, *self.unit1_ls]
        other_ls = ["UNIT1 Rebalance", "UNIT2 Rebalance", "UNIT3 Rebalance"]

        res_ls = [x for x in [*strat_ls, *other_ls] if x is not None]
        return res_ls

    def wake_up_strat(self):
        self.unit1_col = [self.details.columns.get_loc(col) for col in self.unit1_ls]
        self.unit2_col = [self.details.columns.get_loc(col) for col in self.unit2_ls]
        self.unit3_col = [self.details.columns.get_loc(col) for col in self.unit3_ls]

        self.unit1_rebalance_col = self.details.columns.get_loc("UNIT1 Rebalance")
        self.unit2_rebalance_col = self.details.columns.get_loc("UNIT2 Rebalance")
        self.unit3_rebalance_col = self.details.columns.get_loc("UNIT3 Rebalance")

        self.unit1_thres_breach = {cluster: self.details.columns.get_loc(cluster) for cluster in self.unit1_ls}
        self.unit2_thres_breach = {cluster: self.details.columns.get_loc(cluster) for cluster in self.unit2_ls}
        self.unit3_thres_breach = {cluster: self.details.columns.get_loc(cluster) for cluster in self.unit3_ls}

        self.set_cluster_column_idx()
        self.set_cluster_weight()
        self.set_unit_mapping()
        self.set_unit1_threshold()
        self.set_unit2_threshold()
        self.set_unit3_threshold()

    def set_unit_mapping(self):
        """
        This function creates mapping dictionaries between
        Unit III - Unit II
        Unit II - Unit I
        """
        self.unit3to2_map = dict(zip(self.instruments_table["UNIT III"], self.instruments_table["UNIT II"]))
        self.unit2to1_map = dict(zip(self.instruments_table["UNIT II"], self.instruments_table["UNIT I"]))

    def set_cluster_weight(self):
        """
        This function sets the weight on Cluster Level
        """
        for cluster in self.unit1_ls:
            wts_sum = self.instruments_table[self.instruments_table["UNIT I"] == cluster][self.strategy_risk_class].sum()
            self.unit1_weights[cluster] = wts_sum
        for cluster in self.unit2_ls:
            wts_sum = self.instruments_table[self.instruments_table["UNIT II"] == cluster][self.strategy_risk_class].sum()
            self.unit2_weights[cluster] = wts_sum
        for cluster in self.unit3_ls:
            wts_sum = self.instruments_table[self.instruments_table["UNIT III"] == cluster][self.strategy_risk_class].sum()
            self.unit3_weights[cluster] = wts_sum

    def set_cluster_column_idx(self):
        """
        This function curates the cluster dictionaries,
        which contain the columns (indices) for each cluster (UNIT I-III).
        """
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
        self.unit2_idx = create_map(self.instrument_ls, self.unit2_ls, self.details, 2)
        self.unit3_idx = create_map(self.instrument_ls, self.unit3_ls, self.details, 3)

    def set_unit_ls(self):
        self.unit1_ls = list(set(inst.unit1 for inst in self.instrument_ls if inst.unit1 is not None))
        self.unit2_ls = list(set(inst.unit2 for inst in self.instrument_ls if inst.unit2 is not None))
        self.unit3_ls = list(set(inst.unit3 for inst in self.instrument_ls if inst.unit3 is not None))

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

    def set_unit2_threshold(self):
        """
        Set UNIT II Threshold, logic derived from current methodology
        """
        def get_cluster_weight(instruments_table, rc, cluster):
            """This function gets the weight on Cluster Level"""
            wts_sum = instruments_table[instruments_table["UNIT II"] == cluster][rc].sum()
            return wts_sum

        for cluster in self.unit2_ls:
            if cluster in ["GOLD", "CASH", "COM"]:
                self.unit2_thres[cluster] = 0.01
            else:
                this_rc = int(self.strategy_risk_class)

                next_rc = this_rc + 10
                curr_cluster_weight = get_cluster_weight(self.instruments_table, str(this_rc), cluster)
                try:
                    next_cluster_weight = get_cluster_weight(self.instruments_table, str(next_rc), cluster)
                except KeyError:
                    prev_rc = this_rc - 10
                    next_cluster_weight = get_cluster_weight(self.instruments_table, str(prev_rc), cluster)

                thres = abs(curr_cluster_weight - next_cluster_weight) / 2

                if cluster == 'BONDS HY':
                    thres = curr_cluster_weight * 0.1 if thres / curr_cluster_weight < 0.1 else thres
                elif cluster == 'EQU EM':
                    thres = curr_cluster_weight * 0.09 if thres / curr_cluster_weight < 0.09 else thres

                self.unit2_thres[cluster] = thres

    def set_unit3_threshold(self):
        """
        Set UNIT III Threshold, logic derived from current methodology
        """
        for cluster in self.unit3_ls:
            unit2_cluster = self.unit3to2_map[cluster]
            unit2_wt = self.unit2_weights[unit2_cluster]
            unit3_wt = self.unit3_weights[cluster]
            self.unit3_thres[cluster] = abs(unit3_wt / unit2_wt * self.unit3_scale)

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
            cluster_wt = unit_weights[cluster]
            weight = self.details_np[t - 1, unit_idx[cluster]].sum()
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
            self.reset_weights(t - 1)
            self.details_np[0, self.pf_ret_col] = 0
            self.details_np[0, self.pf_cum_ret_col] = 0
            self.details_np[0, self.inv_col_np] = self.details_np[0, self.wts_col_np] * self.amount_inv
            self.details_np[0, self.hyp_amount_inv_col] = self.details_np[0, self.inv_col_np].sum()
        self.calc_portfolio_ret(t)

if __name__ == "__main__":
    # Example Execution of Strategy Object
    strategy = Strategy(config_path=Path(__file__).parents[1] / 'config/config_strategy.ini', scale_unit = 0.020925644969531886)
    strategy.manage_portfolio()  # Returns a KPI Dictionary
    strategy.export_files()  # Exports Detail Sheet (also triggered in manage_portfolio

    # Output KPIs into DataFrame
    kpi_dic = strategy.get_kpi()  # Returns a KPI Dictionary
    kpi_df = pd.DataFrame([kpi_dic])  # Convert to DataFrame
    folderpath = strategy.root_path
    filename = "kpi_summary_optim.csv"
    kpi_df.to_csv(folderpath / filename)