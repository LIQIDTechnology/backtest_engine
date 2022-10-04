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
        self.calendar = Calendar(self.config["strategy"]["calendar"])
        self.strategy_name = self.config["strategy"]["strategy name"]
        self.start_date = dt.datetime.strptime(self.config["strategy"]["start date"], "%Y-%m-%d").date()
        self.end_date = dt.datetime.strptime(self.config["strategy"]["end date"], "%Y-%m-%d").date()
        self.root_path = self.config["paths"]["root_path"]
        self.prices_table = self.load_prices(path=self.root_path / Path("prices.csv"), index_col="Date")
        self.instruments_table = pd.read_csv(self.root_path / Path("instruments.csv"), index_col="Instrument Ticker")
        self.details = pd.DataFrame([])
        self.details_np = None
        self.instrument_ls = [Instrument(self.instruments_table.loc[self.config["instruments"][inst], :])
                              for inst in self.config["instruments"]]

    @staticmethod
    def load_prices(path: Path, index_col: str) -> pd.DataFrame:
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

    def wake_up(self):
        self.init_details()

        end_date = self.end_date if self.end_date is not None else dt.date.today()
        start_date = self.calendar.bday_add(self.start_date, days=-1)
        bday_range = self.calendar.bday_range(start=start_date, end=end_date)

        # Import Prices Table
        inst_col_ls = [inst.ticker for inst in self.instrument_ls]
        self.details = pd.concat([self.details, self.prices_table.loc[bday_range, inst_col_ls]])

        # Deduce Return Table
        inst_ret_col_ls = [f"{inst.ticker} Return" for inst in self.instrument_ls]
        price_mat_t = np.array(self.details.loc[self.details.index[1]:self.details.index[-1], inst_col_ls])
        price_mat_tm1 = np.array(self.details.loc[self.details.index[0]:self.details.index[-2], inst_col_ls])
        ret_mat = price_mat_t / price_mat_tm1 - 1
        ret_df = pd.DataFrame(ret_mat)
        ret_df.columns = inst_ret_col_ls
        ret_df.index = bday_range[1:]
        self.details.loc[bday_range[1:], inst_ret_col_ls] = ret_df
        self.details.loc[:, "EUR Curncy Return"] = 0

    @abstractmethod
    def routine(self, day: int):
        pass

    def go_to_sleep(self):
        """
        Generate Output Files, e.g Details Sheet or Fact Sheet
        """
        self.export_files()
        self.generate_fact_sheet()

    @abstractmethod
    def init_details(self):
        pass

    def export_files(self):
        """
        Exports the Details Sheet.
        """
        self.details = pd.DataFrame(self.details_np, columns=self.details.columns, index=self.details.index)

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

