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
        self.instrument_ls = []

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
        for day in bday_range:
            self.routine(day)
        self.go_to_sleep()

    def wake_up(self):
        self.init_details()
        for inst in self.config["instruments"]:
            inst_ticker = self.config["instruments"][inst]
            self.instrument_ls.append(Instrument(self.instruments_table.loc[inst_ticker, :]))

    @abstractmethod
    def routine(self, day: dt.date):
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
        self.details.to_csv(self.root_path / Path(f"{self.strategy_name}_details.csv"))
        print(f'Details Sheet exported to {self.root_path / Path(f"{self.strategy_name}_details.csv")}')

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

