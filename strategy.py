from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import datetime as dt

from portfolio import Portfolio


class Strategy(Portfolio):

    def __init__(self, config_path: Union[str, Path]):
        super().__init__(config_path)
        self.start_date = dt.datetime.strptime(self.config["strategy"]["start date"], "%Y-%m-%d").date()
        self.end_date = dt.datetime.strptime(self.config["strategy"]["end date"], "%Y-%m-%d").date()
        self.params = None

    def init_details(self):
        self.details = pd.DataFrame(columns=[self.config["instruments"][inst] for inst in self.config["instruments"]])

    def routine(self, t: dt.date):
        self.details.loc[t, self.instrument_ls[0].ticker] = self.prices_table.loc[t, self.instrument_ls[0].ticker]
