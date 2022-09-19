import configparser
from pathlib import Path
from typing import Union
from abc import abstractmethod
from loguru import logger

import numpy as np
import pandas as pd
from calendar import Calendar


class Portfolio(object):

    def __init__(self, config_path: Union[str, Path]):
        self.logger = logger
        self.config = self.load_config(config_path)
        self.calendar = Calendar(self.config["strategy"]["g"])

    def manage_portfolio(self):
        self.wake_up()
        for days in self.calendar:
            self.routine()
        self.go_to_sleep()

    @abstractmethod
    def wake_up(self):
        pass

    @abstractmethod
    def routine(self):
        pass

    @abstractmethod
    def go_to_sleep(self):
        """
        Generate Output Files, e.g Details Sheet or Fact Sheet
        """
        self.export_files()
        self.generate_fact_sheet()

    def export_files(self):
        pass

    def generate_fact_sheet(self):
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

