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
        self.params = None

    def init_details(self):
        inst_ls = [self.config["instruments"][inst] for inst in self.config["instruments"]]
        ret_ls = [f"{self.config['instruments'][inst]} Return" for inst in self.config["instruments"]]
        wts_ls = [f"{self.config['instruments'][inst]} Weight" for inst in self.config["instruments"]]
        other_ls = ["Portfolio Return"]
        self.details = pd.DataFrame(columns=inst_ls + ret_ls + wts_ls + other_ls)

    def get_portfolio_ret(self, t: dt.date):
        tm1 = self.calendar.bday_add(self.start_date, days=-1)
        wts_array = np.array([self.details.loc[tm1, f"{inst.ticker} Weight"] for inst in self.instrument_ls])
        ret_array = np.array([self.details.loc[t, f"{inst.ticker} Return"] for inst in self.instrument_ls])
        pf_ret = np.dot(wts_array, ret_array)
        self.details.loc[t, "Portfolio Return"] = pf_ret

    def reset_weights(self, t):
        scol = f"{self.instrument_ls[0].ticker} Weight"
        ecol = f"{self.instrument_ls[-1].ticker} Weight"
        self.details.loc[t, scol:ecol] = [inst.weight for inst in self.instrument_ls]

    def get_weights(self, t: dt.date):
        tm1 = self.calendar.bday_add(t, days=-1)
        pf_ret_t = self.details.loc[t, "Portfolio Return"]

        scol_inst_ret = f"{self.instrument_ls[0].ticker} Return"
        ecol_inst_ret = f"{self.instrument_ls[-1].ticker} Return"
        inst_ret_t_arr = np.array(self.details.loc[t, scol_inst_ret:ecol_inst_ret])

        scol_wts = f"{self.instrument_ls[0].ticker} Weight"
        ecol_wts = f"{self.instrument_ls[-1].ticker} Weight"
        wt_tm1_arr = np.array(self.details.loc[tm1, scol_wts:ecol_wts])

        self.details.loc[t, scol_wts:ecol_wts] = wt_tm1_arr * (1 + inst_ret_t_arr) / (1 + pf_ret_t)

    def get_prices(self, t):
        scol = self.instrument_ls[0].ticker
        ecol = self.instrument_ls[-1].ticker
        self.details.loc[t, scol:ecol] = self.prices_table.loc[t, scol:ecol]

    def check_rebal(self, t):
        tm1 = self.calendar.bday_add(self.start_date, days=-1)
        reset_bool = True if self.details.loc[t, "Reset"] == 1 else False
        self.reset_weights(tm1) if t == self.start_date or reset_bool else None

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
        self.check_rebal(t)
        self.get_prices(t)
        self.get_return(t)
        self.get_portfolio_ret(t)
        self.get_weights(t)
        self.check_rebal(t)
        print(t)



