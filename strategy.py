from pathlib import Path
from typing import Union
from functools import lru_cache

import numpy as np
import pandas as pd
import datetime as dt

from portfolio import Portfolio
from instrument import Instrument


class Strategy(Portfolio):

    def __init__(self, config_path: Union[str, Path]):
        super().__init__(config_path)
        self.unit_03_scale = float(self.config["params"]["unit_03_scale"])
        self.weights = {unit: float(self.config["weights"][unit]) for unit in self.config["weights"]}
        self.threshold = self.init_threshold()

    def init_details(self):
        inst_ls = [self.config["instruments"][inst] for inst in self.config["instruments"]]
        ret_ls = [f"{self.config['instruments'][inst]} Return" for inst in self.config["instruments"]]
        wts_ls = [f"{self.config['instruments'][inst]} Weight" for inst in self.config["instruments"]]
        other_ls = ["Portfolio Return", 'apac ex jp', 'em world', 'europe', 'japan', 'north america', "Rebalance?"]
        self.details = pd.DataFrame(columns=inst_ls + ret_ls + wts_ls + other_ls)

    def init_threshold(self
                       ):
        constraint = {unit: float(self.config["constraint"][unit]) for unit in self.config["constraint"]}
        threshold = {}
        for unit in self.config["weights"]:
            wt = self.weights[unit]
            const = constraint[unit]
            thres = wt / const * self.unit_03_scale
            threshold[unit] = thres
        return threshold

    def get_portfolio_ret(self, t: dt.date):
        tm1 = self.calendar.bday_add(t, days=-1)
        wts_array = np.array([self.details.loc[tm1, f"{inst.ticker} Weight"] for inst in self.instrument_ls])
        ret_array = np.array([self.details.loc[t, f"{inst.ticker} Return"] for inst in self.instrument_ls])
        pf_ret = np.dot(wts_array, ret_array)
        self.details.loc[t, "Portfolio Return"] = pf_ret
        return pf_ret

    def reset_weights(self, t):
        scol = f"{self.instrument_ls[0].ticker} Weight"
        ecol = f"{self.instrument_ls[-1].ticker} Weight"
        self.details.loc[t, scol:ecol] = [inst.weight for inst in self.instrument_ls]

    def calc_weights(self, t):
        tm1 = self.calendar.bday_add(t, days=-1)
        pf_ret_t = self.details.loc[t, "Portfolio Return"]

        scol_inst_ret = f"{self.instrument_ls[0].ticker} Return"
        ecol_inst_ret = f"{self.instrument_ls[-1].ticker} Return"
        inst_ret_t_arr = np.array(self.details.loc[t, scol_inst_ret:ecol_inst_ret])

        scol_wts = f"{self.instrument_ls[0].ticker} Weight"
        ecol_wts = f"{self.instrument_ls[-1].ticker} Weight"
        wts_tm1_arr = np.array(self.details.loc[tm1, scol_wts:ecol_wts])
        wts_t_arr = wts_tm1_arr * (1 + inst_ret_t_arr) / (1 + pf_ret_t)
        self.details.loc[t, scol_wts:ecol_wts] = wts_t_arr
        return wts_t_arr

    def get_weights(self, t: dt.date):
        if self.check_rebal(t):
            self.reset_weights(t)
        else:
            self.calc_weights(t)

    def get_prices(self, t):
        scol = self.instrument_ls[0].ticker
        ecol = self.instrument_ls[-1].ticker
        self.details.loc[t, scol:ecol] = self.prices_table.loc[t, scol:ecol]

    def check_rebal(self, t):
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
        print(t)
        tm1 = self.calendar.bday_add(t, days=-1)
        self.reset_weights(tm1) if self.start_date == t else None
        self.calc_benchmark(t, rk=10)
        self.get_prices(t)
        self.get_return(t)
        self.get_portfolio_ret(t)
        self.get_weights(t)
        #self.export_files()