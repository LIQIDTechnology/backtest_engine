import configparser
from pathlib import Path
from typing import Union
from abc import abstractmethod
import datetime as dt

import numpy as np
import pandas as pd

from ex_calendar import Calendar
from instrument import Instrument


class Portfolio(object):
    """
    Portfolio Class containing necessary methods for any Strategy

    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Declaring variables upon Object Initialization
        """
        self.config = self.load_config(config_path)
        self.root_path = Path(self.config["paths"]["root_path"])
        self.instrument_path = Path(self.config["paths"]["instrument_path"])
        self.prices_path = Path(self.config["paths"]["prices_path"])

        # PORTFOLIO CONFIGURATION
        self.calendar = Calendar(self.config["strategy"]["calendar"])
        self.strategy_name = self.config["strategy"]["strategy name"]
        self.strategy_risk_class = self.config["strategy"]["strategy risk class"]
        self.start_date = dt.datetime.strptime(self.config["strategy"]["start date"], "%Y-%m-%d").date()
        self.end_date = dt.datetime.strptime(self.config["strategy"]["end date"], "%Y-%m-%d").date()
        self.trading_cost = float(self.config["strategy"]["trading costs"])
        self.liqid_fee = float(self.config["strategy"]["liqid fee"])
        self.amount_inv = float(self.config["strategy"]["amount invested"])
        self.trading_days = float(self.config["strategy"]["trading days"])

        # PORTFOLIO MANAGEMENT
        self.instruments_table = pd.DataFrame([])
        self.instrument_ls = []
        self.prices_table = pd.DataFrame([])
        self.details = None
        self.details_np = None

        self.inst_col = None
        self.ret_col = None
        self.inv_col = None
        self.wts_col = None
        self.inst_col_np = None
        self.wts_col_np = None
        self.ret_col_np = None
        self.pf_ret_col = None
        self.pf_cum_ret_col = None
        self.liqid_cost_col = None
        self.trading_vol_col = None
        self.trading_cost_col = None
        self.hyp_amount_inv_col = None
        self.hyp_liqid_cost_col = None
        self.hyp_trading_vol_col = None
        self.hyp_trading_cost_col = None
        self.rebalance_col = None

    def set_column_idx(self):
        """
        This function contains the column (indices) of the details sheet.
        Numpy Matrices can only be read & queried via rows and columns
        DataFrames can be also read & queried via row name and column names

        As the calculation is run over numpy matrices is necessary to have a column name to index mapping
        """
        self.inst_col_np = [self.details.columns.get_loc(col) for col in self.inst_col]
        self.ret_col_np = [self.details.columns.get_loc(col) for col in self.ret_col]
        self.inv_col_np = [self.details.columns.get_loc(col) for col in self.inv_col]
        self.wts_col_np = [self.details.columns.get_loc(col) for col in self.wts_col]
        self.pf_ret_col = self.details.columns.get_loc("Portfolio Return")
        self.pf_cum_ret_col = self.details.columns.get_loc("Cumulative Portfolio Return")
        self.liqid_cost_col = self.details.columns.get_loc("LIQID Cost")
        self.trading_vol_col = self.details.columns.get_loc("Trading Volume")
        self.trading_cost_col = self.details.columns.get_loc("Trading Cost")
        self.hyp_amount_inv_col = self.details.columns.get_loc("Hyp Amount Invested")
        self.hyp_liqid_cost_col = self.details.columns.get_loc("Hyp LIQID Cost")
        self.hyp_trading_vol_col = self.details.columns.get_loc("Hyp Trading Volume")
        self.hyp_trading_cost_col = self.details.columns.get_loc("Hyp Trading Cost")
        self.rebalance_col = self.details.columns.get_loc("Rebalance")

    def init_instruments(self) -> list:
        """
        Creates a list with Instrument Objects from Instruments Table selecting the Risk
        """
        self.instruments_table = pd.read_csv(self.instrument_path, index_col="Instrument Ticker")
        self.instruments_table = self.instruments_table.fillna(0)
        self.instruments_table = self.instruments_table[self.instruments_table[self.strategy_risk_class] != 0]

        instrument_ls = [Instrument(self.instruments_table.loc[inst, :], self.strategy_risk_class)
                         for inst in self.instruments_table.index]
        self.instrument_ls = instrument_ls
        return instrument_ls

    def init_prices(self, path: Path, index_col: str) -> pd.DataFrame:
        """
        This function loads in the prices.
        In case there is a substitution configured for an instrument,
        it will be substituted until the original is available. ('available from' in instruments config)
        """
        tbl = pd.read_csv(path)
        tbl[index_col] = tbl[index_col].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").date())
        tbl = tbl.set_index(index_col)
        start_date = self.start_date + dt.timedelta(days=-1)
        price_ret_df = tbl.loc[start_date:, [inst.ticker for inst in self.instrument_ls]]

        # SUBSTITUTE
        for inst in self.instrument_ls:
            if inst.substitute_bool and inst.available_from > start_date:
                end_date = self.end_date if inst.available_from > self.end_date \
                    else inst.available_from + dt.timedelta(days=-1)
                price_ret_df.loc[start_date:end_date, inst.ticker] = tbl.loc[start_date:end_date, inst.substitute_ticker]
        self.prices_table = price_ret_df
        return price_ret_df

    def manage_portfolio(self):
        """
        This method contains the central for loop for which calculation per day/row is called.
        """
        self.wake_up()

        end_date = self.end_date if self.end_date is not None else dt.date.today()
        bday_range = self.calendar.bday_range(start=self.start_date, end=end_date)
        self.details_np = self.details.values  # Convert DataFrame into Numpy Matrix for performance

        # print([inst.weight * 100000 for inst in self.instrument_ls])

        for day_idx in range(1, len(bday_range)+1):
            self.routine(day_idx)

        self.go_to_sleep()


    @abstractmethod
    def wake_up_strat(self):
        pass

    def wake_up(self):
        """
        This method initialises all necessary variables
        """

        self.init_instruments()
        self.init_prices(self.prices_path, index_col="Date")
        self.init_details()
        self.apply_product_cost()
        self.set_column_idx()
        self.wake_up_strat()

    @abstractmethod
    def init_details_strat(self):
        return []

    def init_details(self):
        """
        This method is initialising the details sheet with column headers and rows (date range).
        It is also responsible for importing the prices from the price sheet and deducing the return series.
        """
        self.inst_col = [f"{inst.ticker}" for inst in self.instrument_ls]
        self.ret_col = [f"{inst.ticker} Return" for inst in self.instrument_ls]
        self.inv_col = [f"{inst.ticker} Invested" for inst in self.instrument_ls]
        self.wts_col = [f"{inst.ticker} Weight" for inst in self.instrument_ls]

        inst_ls = [*self.inst_col, *self.ret_col, *self.inv_col, *self.wts_col]
        strat_ls = self.init_details_strat()
        other_ls = ["Rebalance",
                    "LIQID Cost",
                    "Trading Volume",
                    "Trading Cost",
                    "Portfolio Return",
                    "Cumulative Portfolio Return",
                    "Hyp Amount Invested",
                    "Hyp LIQID Cost",
                    "Hyp Trading Volume",
                    "Hyp Trading Cost"]

        self.details = pd.DataFrame(columns=[*inst_ls, *strat_ls, *other_ls])
        self.details.index.name = 'Date'

        end_date = self.end_date if self.end_date is not None else dt.date.today()
        start_date = self.calendar.bday_add(self.start_date, days=-1)
        bday_range = self.calendar.bday_range(start=start_date, end=end_date)

        # Import Prices Table into Details Sheet
        prices_table = self.prices_table.loc[bday_range, self.inst_col]
        self.details = pd.concat([self.details, prices_table])
        pd.concat([prices_table, self.details])

        # Deduce Return Table in Details Sheet
        rename_dic = {col: f"{col} Return" for col in self.inst_col}
        ret_mat = self.details.loc[:, self.inst_col] - 1
        ret_mat.rename(columns=rename_dic, inplace=True)
        self.details.loc[:, self.ret_col] = ret_mat

    def apply_product_cost(self):
        """
        This function applies the product cost on the daily return
        """
        for inst in self.instrument_ls:
            ret_series = self.details.loc[:, f"{inst.ticker} Return"]
            prod_cost = inst.product_cost

            ####### martin version
            cost_factor = (1 + prod_cost) ** (1 / self.trading_days) - 1
            new_ret_series = ret_series - cost_factor

            ####### truc version
            # cost_factor = (1 - prod_cost) ** (1 / self.trading_days)
            # new_ret_series = (1 + ret_series) * cost_factor - 1

            self.details.loc[:, f"{inst.ticker} Return"] = new_ret_series

    @abstractmethod
    def routine(self, day: int):
        """
        This function has to implemented in the child class
        """
        pass

    # PORTFOLIO FUNCTIONS
    def reset_weights(self, t):
        """
        This function resets the weights to the starting allocation.
        """
        self.details_np[t, self.wts_col_np] = [inst.weight for inst in self.instrument_ls]
        return self.details_np[t, self.wts_col_np]

    def calc_weights(self, t):
        """
        This function calculates the new weights after one return period (day).
        """
        pf_ret_t = self.details_np[t, self.pf_ret_col]
        inst_ret_t_arr = self.details_np[t, self.ret_col_np]
        wts_tm1_arr = self.details_np[t-1, self.wts_col_np]
        wts_t_arr = wts_tm1_arr * (1 + inst_ret_t_arr) / (1 + pf_ret_t)
        self.details_np[t, self.wts_col_np] = wts_t_arr
        return wts_t_arr

    def apply_rebalance(self, t):
        """
        8 Uhr morgens, Portfolio hat über Nacht gerechnet, Instrumente werden gehandelt.
        """

        today = self.details.index[t]
        # Apply Trading Costs for Rebalancing

        # self.reset_weights(t)  # Target Weight
        # old_alloc = self.details_np[t, self.inv_col_np]
        # self.details_np[t, self.inv_col_np] = self.details_np[t, self.wts_col_np] * hyp_amount_inv_t
        trading_cost = 0
        volume = 0
        hyp_amount_inv_tm1 = self.details_np[t - 1, self.hyp_amount_inv_col]
        for inst in self.instrument_ls:
            # inst_weight_col = self.details.columns.get_loc(f"{inst.ticker} Weight")
            inst_inv_col = self.details.columns.get_loc(f"{inst.ticker} Invested")

            inv_target = inst.weight * hyp_amount_inv_tm1
            inv_current = self.details_np[t - 1, inst_inv_col]

            diff = abs(inv_target - inv_current)
            inst_trading_cost = diff * inst.trading_cost
            # inst_trading_cost = diff * 0

            inv_actual = inv_target - inst_trading_cost
            self.details_np[t, inst_inv_col] = inv_actual
            volume += diff
            trading_cost += inst_trading_cost

        hyp_amount_inv_t = self.details_np[t, self.inv_col_np].sum()
        self.details_np[t, self.hyp_amount_inv_col] = hyp_amount_inv_tm1
        self.details_np[t, self.wts_col_np] = self.details_np[t, self.inv_col_np] / hyp_amount_inv_t

        volume_traded = volume / hyp_amount_inv_tm1
        trading_volume = trading_cost / volume
        self.details_np[t, self.trading_vol_col] = volume_traded
        self.details_np[t, self.trading_cost_col] = trading_volume
        self.details_np[t, self.hyp_trading_vol_col] = volume
        self.details_np[t, self.hyp_trading_cost_col] = trading_cost
        # pf_ret = hyp_amount_inv_t / hyp_amount_inv_tm1 - 1

    def apply_liqid_fee(self, t):
        # Apply LIQID Fee
        month_today = self.details.index[t].month
        try:
            month_tmr = self.details.index[t + 1].month
        except IndexError:
            month_tmr = month_today

        if month_today != month_tmr and month_today in (3, 6, 9, 12):
            quarterly_cost = self.liqid_fee / 4
            hyp_amount_inv_t = self.details_np[t, self.inv_col_np].sum()
            cost_to_deduct = hyp_amount_inv_t * quarterly_cost

            cash_ticker = 'LS01TREU Index'
            cash_inv_col = self.details.columns.get_loc(f"{cash_ticker} Invested")
            cash_current = self.details_np[t, cash_inv_col]
            cash_after_cost = cash_current - cost_to_deduct
            self.details_np[t, cash_inv_col] = cash_after_cost

            # hyp_amount_inv_t_after = self.details_np[t, self.inv_col_np].sum()

            # pf_ret = hyp_amount_inv_t / hyp_amount_inv_tm1 - 1

            # self.details_np[t, self.wts_col_np] = self.details_np[t, self.inv_col_np] / hyp_amount_inv_t
            # self.details_np[t, self.hyp_liqid_cost_col] = cost_to_deduct

    def calc_portfolio_ret(self, t):
        """
        This function calculates the Portfolio Ret
        """
        if self.details.index[t] > self.start_date and self.check_rebal(t):
            self.apply_rebalance(t)
            inst_inv_arr_tm1 = self.details_np[t, self.inv_col_np]  # Neue Gewichte am Morgen
            hyp_amount_inv_tm1 = self.details_np[t, self.hyp_amount_inv_col]
        else:
            inst_inv_arr_tm1 = self.details_np[t-1, self.inv_col_np]
            hyp_amount_inv_tm1 = self.details_np[t-1, self.hyp_amount_inv_col]

        ret_array_t = self.details_np[t, self.ret_col_np]
        inst_inv_arr_t = inst_inv_arr_tm1 * (1 + ret_array_t)
        self.details_np[t, self.inv_col_np] = inst_inv_arr_t

        self.apply_liqid_fee(t)  # if EOM

        hyp_amount_inv_t = self.details_np[t, self.inv_col_np].sum()
        self.details_np[t, self.hyp_amount_inv_col] = hyp_amount_inv_t
        pf_ret = hyp_amount_inv_t / hyp_amount_inv_tm1 - 1
        self.details_np[t, self.pf_ret_col] = pf_ret

        # calc_weights
        wts_arr_t = self.details_np[t, self.inv_col_np] / hyp_amount_inv_t
        self.details_np[t, self.wts_col_np] = wts_arr_t

        # Cumulative Portfolio Return
        pf_cum_ret_tm1 = self.details_np[t-1, self.pf_cum_ret_col]
        pf_cum_ret_t = (pf_cum_ret_tm1 + 1) * (pf_ret + 1) - 1
        self.details_np[t, self.pf_cum_ret_col] = pf_cum_ret_t

        return pf_ret

    @abstractmethod
    def check_rebal(self, t):
        """
        This function has to implemented in the child class
        """
        pass

    def go_to_sleep(self):
        """
        This method handles post calculation processes
        Generate Output Files, e.g Details Sheet & Fact Sheet (Future)
        """
        self.details = pd.DataFrame(self.details_np, columns=self.details.columns, index=self.details.index)
        # self.details['Cumulative Portfolio Return'] = (1 + self.details["Portfolio Return"]).cumprod() - 1
        # self.export_files()
        # self.generate_fact_sheet()

    def export_files(self):
        """
        This function exports the Details Sheet.
        """
        self.details.to_csv(self.root_path / Path(f"{self.strategy_name}_details.csv"))
        print(f'Details Sheet exported to {self.root_path / Path(f"{self.strategy_name}_details.csv")}')

    def qplix_upload(self):
        """
        This function generates the qplix ticker upload file
        """
        masterdata_col = self.config['qplix']['masterdata_col'].split(",")
        masterdata_df = pd.DataFrame(columns=masterdata_col)
        masterdata_dic = {"Ticker": self.strategy_name, "Name": self.strategy_name, "CurrencyCode": "EUR",
                          "Source": "Internal"}
        tmp_df = pd.DataFrame([masterdata_dic])
        masterdata_df = pd.concat([masterdata_df, tmp_df])

        quotes_df = self.details.loc[:, ["Hyp Amount Invested"]] / self.amount_inv
        quotes_df.rename(columns={"Hyp Amount Invested": "ClosePrice"}, inplace=True)
        quotes_df["Ticker"] = self.strategy_name
        quotes_df.reset_index(inplace=True)
        quotes_df.set_index("Ticker", inplace=True)

        with pd.ExcelWriter(self.root_path / f'QPLIX_Upload_{self.strategy_name}.xlsx') as writer:
            masterdata_df.to_excel(writer, sheet_name='Masterdata', index=False)
            quotes_df.to_excel(writer, sheet_name='Quotes')

    def generate_fact_sheet(self):
        """
        This function generates the Fact Sheet.
        """
        pass

    def load_config(self, config_path: Union[str, Path]) -> configparser.RawConfigParser:
        """
        This function loads the config file.
        """
        if not config_path.exists():
            error_msg = f'Attempting to load the config file {config_path}, while this file could not be found!'
            print(error_msg)
            raise FileNotFoundError(error_msg)
        
        config = configparser.RawConfigParser()
        config.read(str(config_path))
        return config

