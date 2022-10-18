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

        self.inst_col = None
        self.wts_col = None
        self.ret_col = None
        self.inst_col_np = None
        self.wts_col_np = None
        self.ret_col_np = None
        self.pf_ret_col = None

        self.unit3_ls = None
        self.unit2_ls = None
        self.unit1_ls = None

    def init_details(self):
        self.inst_col = [f"{inst.ticker}" for inst in self.instrument_ls]
        self.wts_col = [f"{inst.ticker} Weight" for inst in self.instrument_ls]
        self.ret_col = [f"{inst.ticker} Return" for inst in self.instrument_ls]
        self.unit3_ls = self.config["cluster"]["unit3"].split(",")
        self.unit2_ls = self.config["cluster"]["unit2"].split(",")
        self.unit1_ls = self.config["cluster"]["unit1"].split(",")

        inst_ls = [*self.inst_col, *self.ret_col, *self.wts_col, *self.unit3_ls, *self.unit2_ls, *self.unit1_ls]
        other_ls = ["Rebalance?", "Portfolio Return"]

        self.details = pd.DataFrame(columns=[*inst_ls, *other_ls])

        # NDARRAY COLUMN INDICES
        self.inst_col_np = [self.details.columns.get_loc(col) for col in self.inst_col]
        self.wts_col_np = [self.details.columns.get_loc(col) for col in self.wts_col]
        self.ret_col_np = [self.details.columns.get_loc(col) for col in self.ret_col]
        self.pf_ret_col = self.details.columns.get_loc("Portfolio Return")
        self.rebalance_col = self.details.columns.get_loc("Rebalance?")
        self.unit1_thres_breach = {cluster: self.details.columns.get_loc(cluster) for cluster in self.unit1_ls}
        self.unit2_thres_breach = {cluster: self.details.columns.get_loc(cluster) for cluster in self.unit2_ls}
        self.unit3_thres_breach = {cluster: self.details.columns.get_loc(cluster) for cluster in self.unit3_ls}

        # INSTRUMENT WEIGHT TO CLUSTER MAPPING UNIT I-III
        for cluster in self.unit1_ls:
            cluster_ls = []
            for inst in self.instrument_ls:
                if inst.unit1 == cluster:
                    cluster_ls.append(self.details.columns.get_loc(f"{inst.ticker} Weight"))
            self.unit1_idx[cluster] = cluster_ls

        for cluster in self.unit2_ls:
            cluster_ls = []
            for inst in self.instrument_ls:
                if inst.unit2 == cluster:
                    cluster_ls.append(self.details.columns.get_loc(f"{inst.ticker} Weight"))
            self.unit2_idx[cluster] = cluster_ls

        for cluster in self.unit3_ls:
            cluster_ls = []
            for inst in self.instrument_ls:
                if inst.unit3 == cluster:
                    cluster_ls.append(self.details.columns.get_loc(f"{inst.ticker} Weight"))
            self.unit3_idx[cluster] = cluster_ls

        # WEIGHTS BY UNIT I-III
        for cluster in self.unit1_ls:
            wts_sum = self.instruments_table[self.instruments_table["UNIT I"] == cluster]["Weight"].sum()
            self.unit1_weights[cluster] = wts_sum
        for cluster in self.unit2_ls:
            wts_sum = self.instruments_table[self.instruments_table["UNIT II"] == cluster]["Weight"].sum()
            self.unit2_weights[cluster] = wts_sum
        for cluster in self.unit3_ls:
            wts_sum = self.instruments_table[self.instruments_table["UNIT III"] == cluster]["Weight"].sum()
            self.unit3_weights[cluster] = wts_sum

        # UNIT I-III MAPPINGS
        self.unit3to2_map = dict(zip(self.instruments_table["UNIT III"], self.instruments_table["UNIT II"]))
        self.unit2to1_map = dict(zip(self.instruments_table["UNIT II"], self.instruments_table["UNIT I"]))

        # THRESHOLD UNIT III
        for cluster in self.unit3_ls:
            unit2_cluster = self.unit3to2_map[cluster]
            unit2_wt = self.unit2_weights[unit2_cluster]
            unit3_wt = self.unit3_weights[cluster]
            self.unit3_thres[cluster] = unit3_wt / unit2_wt * self.unit3_scale

    def reset_weights(self, t):
        self.details_np[t, self.wts_col_np] = [inst.weight for inst in self.instrument_ls]
        return self.details_np[t, self.wts_col_np]

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

    def calc_weights(self, t):
        pf_ret_t = self.details_np[t, self.pf_ret_col]
        inst_ret_t_arr = self.details_np[t, self.ret_col_np]
        wts_tm1_arr = self.details_np[t-1, self.wts_col_np]
        wts_t_arr = wts_tm1_arr * (1 + inst_ret_t_arr) / (1 + pf_ret_t)
        self.details_np[t, self.wts_col_np] = wts_t_arr
        return wts_t_arr

    def set_weights(self, t):
        self.reset_weights(t) if self.check_rebal(t) else self.calc_weights(t)

    def get_portfolio_ret(self, t):
        wts_array = self.details_np[t-1, self.wts_col_np]
        ret_array = self.details_np[t, self.ret_col_np]
        pf_ret = np.dot(wts_array, ret_array)
        self.details_np[t, self.pf_ret_col] = pf_ret
        return pf_ret

    def routine(self, t):
        self.reset_weights(t - 1) if self.details.index[t] == self.start_date else None  # INIT WEIGHTS
        self.get_portfolio_ret(t)
        self.set_weights(t)
