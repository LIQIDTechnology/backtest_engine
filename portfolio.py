import configparser
from pathlib import Path
from typing import Union
from abc import abstractmethod
from loguru import logger
import datetime as dt

import numpy as np
import pandas as pd

from ex_calendar import Calendar
from instrument import Instrument


class Portfolio(object):

    def __init__(self, config_path: Union[str, Path]):
        self.config = self.load_config(config_path)
        self.root_path = self.config["paths"]["root_path"]

        # Portfolio CONFIGURATION
        self.calendar = Calendar(self.config["strategy"]["calendar"])
        self.strategy_name = self.config["strategy"]["strategy name"]
        self.start_date = dt.datetime.strptime(self.config["strategy"]["start date"], "%Y-%m-%d").date()
        self.end_date = dt.datetime.strptime(self.config["strategy"]["end date"], "%Y-%m-%d").date()
        self.trading_cost = float(self.config["strategy"]["trading costs"])

        # PORTFOLIO MANAGEMENT
        self.prices_table = self.init_prices(path=self.root_path / Path("prices.csv"), index_col="Date")
        self.instruments_table = pd.read_csv(self.root_path / Path("instruments.csv"), index_col="Instrument Ticker")
        self.instrument_ls = self.init_instruments()
        self.details = pd.DataFrame([])
        self.details_np = None

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
        self.rebalance_col = None

        self.unit3_ls = self.config["cluster"]["unit3"].split(",")
        self.unit2_ls = self.config["cluster"]["unit2"].split(",")
        self.unit1_ls = self.config["cluster"]["unit1"].split(",")

    def set_column_idx(self):
        """
        This function contains the column (indices) of the details sheet.

        """
        self.inst_col_np = [self.details.columns.get_loc(col) for col in self.inst_col]
        self.wts_col_np = [self.details.columns.get_loc(col) for col in self.wts_col]
        self.ret_col_np = [self.details.columns.get_loc(col) for col in self.ret_col]
        self.pf_ret_col = self.details.columns.get_loc("Portfolio Return")
        self.rebalance_col = self.details.columns.get_loc("Rebalance?")

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

    def set_cluster_weight(self):
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

    def set_unit_mapping(self):
        # UNIT I-III MAPPINGS
        self.unit3to2_map = dict(zip(self.instruments_table["UNIT III"], self.instruments_table["UNIT II"]))
        self.unit2to1_map = dict(zip(self.instruments_table["UNIT II"], self.instruments_table["UNIT I"]))

    def set_unit3_threshold(self):
        # THRESHOLD UNIT III
        for cluster in self.unit3_ls:
            unit2_cluster = self.unit3to2_map[cluster]
            unit2_wt = self.unit2_weights[unit2_cluster]
            unit3_wt = self.unit3_weights[cluster]
            self.unit3_thres[cluster] = unit3_wt / unit2_wt * self.unit3_scale


    def init_instruments(self) -> list:
        """
        Creates a list with Instrument Objects from Instruments Table selecting the Risk
        """
        keep_col = self.config["instruments"]["configuration"].split(",")
        weight_col = [self.config["strategy"]["strategy risk class"]]
        keep_col = [*keep_col, *weight_col]
        instruments_table = self.instruments_table.loc[:, keep_col]
        instruments_table.rename(columns={weight_col[0]: "Weight"}, inplace=True)
        instruments_table = instruments_table.fillna(0)
        self.instruments_table = instruments_table[instruments_table["Weight"] != 0]

        instrument_ls = [Instrument(self.instruments_table.loc[inst, :]) for inst in self.instruments_table.index]
        return instrument_ls

    @staticmethod
    def init_prices(path: Path, index_col: str) -> pd.DataFrame:
        tbl = pd.read_csv(path)
        tbl[index_col] = tbl[index_col].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").date())
        tbl = tbl.set_index(index_col)
        return tbl

    def manage_portfolio(self):
        self.wake_up()

        end_date = self.end_date if self.end_date is not None else dt.date.today()
        bday_range = self.calendar.bday_range(start=self.start_date, end=end_date)

        self.details_np = self.details.values  # Convert DataFrame into Numpy Matrix

        for day_idx in range(1, len(bday_range)+1):
            self.routine(day_idx)

        self.go_to_sleep()
        kpi_dic = self.get_kpi()
        return kpi_dic

    def get_kpi(self):
        days_per_year = 252

        # Count of Rebalancing
        rebal_count = self.details.loc[:,"Rebalance?"].sum()

        sharpe = 3
        sortino = 3

        # Max Drawdown
        levels = pd.Series(1.0).append(self.details["Cumulative Portfolio Return"][1:]+1)
        max = np.maximum.accumulate(levels)
        max_drawdown = (levels / max - 1).min()
        mdd = max_drawdown

        # Annual Return Vola
        array = np.array(self.details["Portfolio Return"][1:])
        std_pa = (((array + 1) ** 2).mean() ** days_per_year - (array + 1).mean() ** (2 * days_per_year)) ** (1 / 2)
        vola = std_pa

        # Total Return
        tot_ret = (self.details.loc[self.end_date, "Cumulative Portfolio Return"] - 1) * 100
        tot_ret_str = f'{"{:.2f}".format(tot_ret)} %'
        kpi_dic = {"Rebalancing Count": rebal_count,
                   "Annualized Volatility": vola,
                   "Maximum Drawdown": mdd,
                   "Total Return": tot_ret}
        return kpi_dic

    def wake_up(self):
        self.init_details()
        self.apply_product_cost()
        self.unit1_thres_breach = {cluster: self.details.columns.get_loc(cluster) for cluster in self.unit1_ls}
        self.unit2_thres_breach = {cluster: self.details.columns.get_loc(cluster) for cluster in self.unit2_ls}
        self.unit3_thres_breach = {cluster: self.details.columns.get_loc(cluster) for cluster in self.unit3_ls}

        self.set_column_idx()
        self.set_cluster_column_idx()
        self.set_cluster_weight()
        self.set_unit_mapping()
        self.set_unit3_threshold()

    def init_details(self):
        self.inst_col = [f"{inst.ticker}" for inst in self.instrument_ls]
        self.wts_col = [f"{inst.ticker} Weight" for inst in self.instrument_ls]
        self.ret_col = [f"{inst.ticker} Return" for inst in self.instrument_ls]

        inst_ls = [*self.inst_col, *self.ret_col, *self.wts_col, *self.unit3_ls, *self.unit2_ls, *self.unit1_ls]
        other_ls = ["Rebalance?", "Portfolio Return"]

        self.details = pd.DataFrame(columns=[*inst_ls, *other_ls])

        end_date = self.end_date if self.end_date is not None else dt.date.today()
        start_date = self.calendar.bday_add(self.start_date, days=-1)
        bday_range = self.calendar.bday_range(start=start_date, end=end_date)

        # Import Prices Table into Details Sheet
        inst_col_ls = [inst.ticker for inst in self.instrument_ls]
        self.details = pd.concat([self.details, self.prices_table.loc[bday_range, inst_col_ls]])

        # Deduce Return Table in Details Sheet
        # TODO: to be replaced
        inst_ret_col_ls = [f"{inst.ticker} Return" for inst in self.instrument_ls]
        price_mat_t = np.array(self.details.loc[self.details.index[1]:self.details.index[-1], inst_col_ls])
        price_mat_tm1 = np.array(self.details.loc[self.details.index[0]:self.details.index[-2], inst_col_ls])
        ret_mat = price_mat_t / price_mat_tm1 - 1
        ret_df = pd.DataFrame(ret_mat)
        ret_df.columns = inst_ret_col_ls
        ret_df.index = bday_range[1:]
        self.details.loc[bday_range[1:], inst_ret_col_ls] = ret_df
        self.details.loc[:, "EUR Curncy Return"] = 0

    def apply_product_cost(self):
        # Apply Product Cost
        for inst in self.instrument_ls:
            ret_series = self.details.loc[:, f"{inst.ticker} Return"]
            prod_cost = inst.product_cost
            cost_factor = (1 - prod_cost) ** (1 / 252)
            new_ret_series = (1 + ret_series) * cost_factor - 1
            self.details.loc[:, f"{inst.ticker} Return"] = new_ret_series

    @abstractmethod
    def routine(self, day: int):
        pass

    # PORTFOLIO FUNCTIONS
    def reset_weights(self, t):
        self.details_np[t, self.wts_col_np] = [inst.weight for inst in self.instrument_ls]
        return self.details_np[t, self.wts_col_np]

    def calc_weights(self, t):
        pf_ret_t = self.details_np[t, self.pf_ret_col]
        inst_ret_t_arr = self.details_np[t, self.ret_col_np]
        wts_tm1_arr = self.details_np[t-1, self.wts_col_np]
        wts_t_arr = wts_tm1_arr * (1 + inst_ret_t_arr) / (1 + pf_ret_t)
        self.details_np[t, self.wts_col_np] = wts_t_arr
        return wts_t_arr

    def calc_portfolio_ret(self, t):

        wts_array_tm1 = self.details_np[t-1, self.wts_col_np]
        ret_array = self.details_np[t, self.ret_col_np]
        pf_ret = np.dot(wts_array_tm1, ret_array)

        if self.details.index[t] > self.start_date and self.check_rebal(t-1):
            wts_array_tm2 = self.details_np[t-2, self.wts_col_np]
            volume_traded = sum(abs(wts_array_tm2-wts_array_tm1))
            trading_cost = volume_traded * self.trading_cost
            pf_ret = pf_ret - trading_cost

        self.details_np[t, self.pf_ret_col] = pf_ret
        return pf_ret



    @abstractmethod
    def check_rebal(self, t):
        pass

    def go_to_sleep(self):
        """
        Generate Output Files, e.g Details Sheet or Fact Sheet
        """
        self.details = pd.DataFrame(self.details_np, columns=self.details.columns, index=self.details.index)
        self.details['Cumulative Portfolio Return'] = (1 + self.details["Portfolio Return"]).cumprod() - 1
        self.export_files()
        self.generate_fact_sheet()

    def export_files(self):
        """
        Exports the Details Sheet.
        """


        # self.details.to_csv(self.root_path / Path(f"{self.strategy_name}_details.csv"))
        # print(f'Details Sheet exported to {self.root_path / Path(f"{self.strategy_name}_details.csv")}')

    def generate_fact_sheet(self):
        """
        Generates the Fact Sheet.
        """
        pass

    def load_config(self, config_path: Union[str, Path]) -> configparser.RawConfigParser:
        """
        Loads the config file.

        Note:
            In case of a FileNotFoundError due to the nature of network drives this could also be a hint towards an
            incorrect or missing drive mapping.

        Args:
            config_path (Union[str, Path], optional): If provided, specifies the config file path to be loaded.

        Returns:
            configparser.RawConfigParser: The loaded config.

        Raises:
            FileNotFoundError: If the specified config file could not be found at the target location.
        """

        if not config_path.exists():
            error_msg = f'Attempting to load the config file {config_path}, while this file could not be found!'
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        config = configparser.RawConfigParser()
        config.read(str(config_path))
        return config

