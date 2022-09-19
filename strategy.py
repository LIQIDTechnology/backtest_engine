from pathlib import Path
from typing import Union
from portfolio import Portfolio
from instruments import Instruments
import datetime as dt
import numpy as np
import pandas as pd
from loguru import logger


class Strategy(Portfolio):

    def __init__(self, config_path: Union[str, Path]):
        super().__init__(config_path)
        self.instruments = None  # List of Instrument Objects
        self.params = None

    def wake_up(self):
        self.instruments = [Instruments(instr) for instr in self.config["instruments"]]
        pass

    def routine(self):
        pass

    def go_to_sleep(self):
        pass

    def run(self):
        self.manage_portfolio()






