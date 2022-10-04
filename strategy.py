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
        self.unit_03_scale = scale_unit
        # self.weights = {unit: float(self.config["weights"][unit]) for unit in self.config["weights"]}

        self.cluster_index_map = {}
        self.unit3_weights = {}
        self.unit3_thres = {}
        self.unit3cluster_rebal_row = {}

        self.threshold = self.init_threshold()
        self.inst_col = None
        self.wts_col = None
        # self.wts_tm1_col = None
        self.ret_col = None
        self.inst_col_np = None
        self.wts_col_np = None
        # self.wts_tm1_col_np = None
        self.ret_col_np = None

        self.pf_ret_col = None




    def init_details(self):
        self.inst_col = [f"{inst.ticker}" for inst in self.instrument_ls]
        self.wts_col = [f"{inst.ticker} Weight" for inst in self.instrument_ls]
        # self.wts_tm1_col = [f"{inst.ticker} Weight_tm1" for inst in self.instrument_ls]
        self.ret_col = [f"{inst.ticker} Return" for inst in self.instrument_ls]
        # inst_ls = [*self.inst_col, *self.ret_col,  *self.wts_col, *self.wts_tm1_col]
        inst_ls = [*self.inst_col, *self.ret_col,  *self.wts_col]
        unit_3_ls = self.config["cluster"]["unit3"].split(",")
        other_ls = ["Portfolio Return", *unit_3_ls, "Rebalance?"]

        self.details = pd.DataFrame(columns=[*inst_ls, *other_ls])

        # NDARRAY INDICES
        self.inst_col_np = [self.details.columns.get_loc(col) for col in self.inst_col]
        self.wts_col_np = [self.details.columns.get_loc(col) for col in self.wts_col]
        # self.wts_tm1_col_np = [self.details.columns.get_loc(col) for col in self.wts_tm1_col]
        self.ret_col_np = [self.details.columns.get_loc(col) for col in self.ret_col]
        self.pf_ret_col = self.details.columns.get_loc("Portfolio Return")
        self.rebalance_col = self.details.columns.get_loc("Rebalance?")
        # self.Equities Asia Pacific ex Japan,
        # Equities Emerging Markets World,
        # Equities Europe,
        # Equities Japan,
        # Equities North America

        # self.apaxex
        for cluster in self.config["cluster"]["unit3"].split(","):
            self.unit3cluster_rebal_row[cluster] = self.details.columns.get_loc(cluster)

        for cluster in self.config["cluster"]["unit3"].split(","):
            cluster_ls = []
            for inst in self.instrument_ls:
                if inst.unit3 == cluster:
                    cluster_ls.append(self.details.columns.get_loc(f"{inst.ticker} Weight"))

            self.cluster_index_map[cluster] = cluster_ls

    def init_threshold(self):
        # self.unit3_weights = {unit: float(self.config["weight"][unit]) for unit in self.config["weight"]}

        for cluster in self.config["cluster"]["unit3"].split(","):
            # clstera asia ex jp = em
            unit2_cluster = self.instruments_table[self.instruments_table["UNIT III"] == cluster]["UNIT II"].values[0]
            # get all weights of em
            all_wts = self.instruments_table[self.instruments_table["UNIT II"] == unit2_cluster]["Weight"].sum()


            unit3_cluster = self.instruments_table[self.instruments_table["UNIT III"]==cluster]["Weight"].values[0]
            self.unit3_weights[cluster] = unit3_cluster
            self.unit3_thres[cluster] = unit3_cluster / all_wts * self.unit_03_scale


        return self.unit3_thres

    def get_portfolio_ret(self, t: dt.date, tm1: dt.date):
        wts_array = np.array(self.details.loc[tm1, self.wts_col])
        ret_array = np.array(self.details.loc[t, self.ret_col])
        pf_ret = np.dot(wts_array, ret_array)
        self.details.loc[t, "Portfolio Return"] = pf_ret
        return pf_ret

    def reset_weights(self, t):
        self.details.loc[t, self.wts_col] = [inst.weight for inst in self.instrument_ls]

    def calc_weights(self, t, tm1):
        pf_ret_t = self.details.loc[t, "Portfolio Return"]
        inst_ret_t_arr = np.array(self.details.loc[t, self.ret_col])
        wts_tm1_arr = np.array(self.details.loc[tm1, self.wts_col])

        wts_t_arr = wts_tm1_arr * (1 + inst_ret_t_arr) / (1 + pf_ret_t)

        self.details.loc[t, self.wts_col] = wts_t_arr
        return wts_t_arr

    def set_weights(self, t: dt.date, tm1: dt.date):
        self.reset_weights(t) if self.check_rebal(t, tm1) else self.calc_weights(t, tm1)

    def get_prices(self, t):
        scol = self.instrument_ls[0].ticker
        ecol = self.instrument_ls[-1].ticker
        self.details.loc[t, scol:ecol] = self.prices_table.loc[t, scol:ecol]

    def check_rebal(self, t, tm1):
        tm1 = self.calendar.bday_add(t, days=-1)
        check_arr = np.array(self.details.loc[tm1, ["NDUECAPF Index Weight", "NDUEEGF Index Weight", "MSDEE15N Index Weight", "NDDLJN Index Weight", "NDDUUS Index Weight"]])
        check_arr_dic = {}
        truth = []

        for unit in self.config["weights"]:
            check_arr_dic[unit] = check_arr[list(self.weights.keys()).index(unit)]
            truth.append(self.weights[unit] - self.threshold[unit] >= check_arr_dic[unit] >= self.weights[unit] + self.threshold[unit])

        if any(item is True for item in truth):
            self.details.loc[t, "Rebalance?"] = 0
            return False
        else:
            self.details.loc[t, "Rebalance?"] = 1
            return True
        return False

    def get_return(self, t: dt.date):
        tm1 = self.calendar.bday_add(t, days=-1)
        scol = f"{self.instrument_ls[0].ticker}"
        ecol = f"{self.instrument_ls[-1].ticker}"
        val_t_arr = np.array(self.prices_table.loc[t, scol:ecol])
        val_tm1_arr = np.array(self.prices_table.loc[tm1, scol:ecol])
        ret_arr = val_t_arr / val_tm1_arr - 1

        scol_ret = f"{self.instrument_ls[0].ticker} Return"
        ecol_ret = f"{self.instrument_ls[-1].ticker} Return"

        self.details.loc[t, scol_ret:ecol_ret] = ret_arr
        self.details.loc[t, "EUR Curncy Return"] = 0

    def routine(self, t: dt.date):
        tm1 = self.calendar.bday_add(t, days=-1)
        self.reset_weights(tm1) if self.start_date == t else None
        self.get_portfolio_ret(t, tm1)
        self.set_weights(t, tm1)

    def reset_weights_np(self, t):
        self.details_np[t, self.wts_col_np] = [inst.weight for inst in self.instrument_ls]

    def check_rebal_np(self, t):

        unit3_cluster = self.config["cluster"]["unit3"].split(",")
        bool = False
        for cluster in unit3_cluster:
            weight = self.details_np[t-1, self.cluster_index_map[cluster]].sum()
            cluster_wt = self.unit3_weights[cluster]
            thres = self.unit3_thres[cluster]
            if cluster_wt - thres < weight < cluster_wt + thres:
                self.details_np[t, self.unit3cluster_rebal_row[cluster]] = 0
            else:
                self.details_np[t, self.unit3cluster_rebal_row[cluster]] = 1
                bool = True
                self.details_np[t, self.rebalance_col] = 1

        return bool

        # europe_wts
        # check_arr_dic = {}
        # truth = []

        # for unit in self.config["weights"]:
        #     check_arr_dic[unit] = check_arr[list(self.weights.keys()).index(unit)]
        #     truth.append(self.weights[unit] - self.threshold[unit] >= check_arr_dic[unit] >= self.weights[unit] + self.threshold[unit])
        #
        # if any(item is True for item in truth):
        #     self.details.loc[t, "Rebalance?"] = 0
        #     return False
        # else:
        #     self.details.loc[t, "Rebalance?"] = 1
        #     return True
        return False

    def calc_weights_np(self, t):
        pf_ret_t = self.details_np[t, self.pf_ret_col]
        inst_ret_t_arr = self.details_np[t, self.ret_col_np]
        wts_tm1_arr = self.details_np[t-1, self.wts_col_np]

        wts_t_arr = wts_tm1_arr * (1 + inst_ret_t_arr) / (1 + pf_ret_t)

        self.details_np[t, self.wts_col_np] = wts_t_arr
        return wts_t_arr

    def set_weights_np(self, t):
        self.reset_weights_np(t) if self.check_rebal_np(t) else self.calc_weights_np(t)

    def get_portfolio_ret_np(self, t):

        wts_array = self.details_np[t-1, self.wts_col_np]
        ret_array = self.details_np[t, self.ret_col_np]
        pf_ret = np.dot(wts_array, ret_array)

        self.details_np[t, self.pf_ret_col] = pf_ret
        return pf_ret

    def routine_np(self, t):

        self.reset_weights_np(t-1) if self.details.index[t] == self.start_date else None  # INIT WEIGHTS
        self.get_portfolio_ret_np(t)
        self.set_weights_np(t)
