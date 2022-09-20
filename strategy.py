from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import datetime as dt

from portfolio import Portfolio


class Strategy(Portfolio):

    def __init__(self, config_path: Union[str, Path]):
        super().__init__(config_path)
        self.start_date = self.config["strategy"]["start date"]
        self.params = None

    def init_details(self):
        self.details = pd.DataFrame(columns=[self.config["instruments"][inst] for inst in self.config["instruments"]])

    def routine(self):
        #Test
        print(self.details)
        print(self.instrument_ls)

    def go_to_sleep(self):
        pass