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

    def init_instruments(self) -> list:
        """
        Creates a list with Instrument Objects from Instruments Table selecting the Risk
        """
        keep_col = self.config["instruments"]["configuration"].split(",")
        weight_col = [self.config["strategy"]["strategy risk class"]]
        keep_col = [*keep_col, *weight_col]
        instruments_table = self.instruments_table.loc[:, keep_col]
        instruments_table.rename(columns={weight_col[0]: "Weight"}, inplace=True)
        self.instruments_table = instruments_table[instruments_table["Weight"] != 0]
        instrument_ls = [Instrument(instruments_table.loc[inst, :]) for inst in instruments_table.index]
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

        # Apply Product Cost
        for inst in self.instrument_ls:
            ret_series = self.details.loc[:, f"{inst.ticker} Return"]
            prod_cost = inst.product_cost
            cost_factor = (1 - prod_cost) ** (1 / 252)
            new_ret_series = (1 + ret_series) * cost_factor - 1
            self.details.loc[:, f"{inst.ticker} Return"] = new_ret_series



    @abstractmethod
    def init_details(self):
        pass

    @abstractmethod
    def routine(self, day: int):
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

