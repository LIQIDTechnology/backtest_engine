from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from strategies.strategy import Strategy
import datetime as dt
from statistics import Statistic
from strategies.benchmarks import Benchmark
from multiprocessing import Pool


class ThresholdOptimizer(object):

    def __init__(self):
        self.start_date = None
        self.end_date = None
        self.risk_class = None
        self.result = []

    def objective_total_return(self, scale_unit):
        """
        Objective Function: Total Return of Strategy
        """
        config_path = Path('config') / 'config_strategy.ini'
        config_path = Path(__file__).parents[0] / 'config/config_strategy.ini'
        strategy = Strategy(config_path=config_path, scale_unit=scale_unit)
        strategy.strategy_risk_class = self.risk_class
        strategy.end_date = self.start_date
        strategy.end_date = self.end_date
        strategy.manage_portfolio()
        # strategy.export_files()
        tot_ret = (strategy.details.loc[strategy.end_date, "Cumulative Portfolio Return"])
        # print(f'Scale Unit: {scale_unit}', f'Total Return: {"{:.5f}".format(tot_ret)}, {self.start_date}, {self.end_date}', flush=True)
        return -tot_ret

    def threshold_optimum(self) -> dict:
        """"""
        bounds_ls = [[0, 0.05], [0.05, 0.1], [0, 0.1]]
        # bounds_ls = [[0, 0.05], [0.05, 0.1], [0, 0.1]]
        # bounds_ls = [[0, 0.2]]
        tr_max = -9999
        result = None
        for bounds in bounds_ls:
            res = scipy.optimize.minimize_scalar(self.objective_total_return, bounds=bounds, method="bounded")
            tr = -self.objective_total_return(res.x)
            if tr > tr_max:
                tr_max = tr
                result = res
            else:
                pass
        return result

    def study_insample(self, start_date: dt.date, end_date: dt.date, rc: str):
        config_path = Path(__file__).parents[0] / 'config/config_strategy.ini'
        tmp_strat = Strategy(config_path=config_path, scale_unit=0.1)
        tmp_strat.manage_portfolio()
        dates_ls = tmp_strat.details.index

        val, idx = min((val, idx) for (idx, val) in enumerate(abs(tmp_strat.details.index - start_date)))
        start_date = tmp_strat.details.index[idx]
        val, idx = min((val, idx) for (idx, val) in enumerate(abs(tmp_strat.details.index - end_date)))
        end_date = tmp_strat.details.index[idx]

        self.start_date = start_date
        self.end_date = end_date
        self.risk_class = rc
        res = self.threshold_optimum()
        sc_optim = res.x
        return sc_optim

    def insample_all_rc(self, start_date, end_date):
        """This functions runs all In Sample Optimizing Strategies."""

        config_path = Path(__file__).parents[0] / 'config/config_strategy.ini'
        tmp_strat = Strategy(config_path=config_path, scale_unit=0.1)
        tmp_strat.manage_portfolio()

        val, idx = min((val, idx) for (idx, val) in enumerate(abs(tmp_strat.details.index - start_date)))
        start_date = tmp_strat.details.index[idx]
        val, idx = min((val, idx) for (idx, val) in enumerate(abs(tmp_strat.details.index - end_date)))
        end_date = tmp_strat.details.index[idx]

        print(f"Optim_{start_date}_{end_date}")

        columns = [str(x * 10) for x in range(1, 11)]
        rows = ["Optimale S-Unit",
                "Total Return",
                "Volatility",
                "MDD",
                "# Rebalancing Unit 1",
                "# Rebalancing Unit 2",
                "# Rebalancing Unit 3",
                "Trading Kosten",
                "Monthly Rebal. Total Return",
                "Unit1 Rebal. Total Return"]

        global_df = pd.DataFrame(columns=columns, index=rows)
        # global_df = pd.read_csv('/Volumes/GoogleDrive/.shortcut-targets-by-id/19JZlTc1zpxipptIN5dg3E-SGtYp5rg7r/02_Backtesting/04_Python Code/source/slide_insample_copy.csv', index_col="Unnamed: 0")
        self.start_date = start_date
        self.end_date = end_date

        for rc in range(1, 11):
            rc_str = str(rc * 10)
            self.risk_class = rc_str
            # print(self.risk_class)
            # try:
            res = self.threshold_optimum()
            sc_optim = res.x
            # sc_optim = float(global_df.loc["Optimale S-Unit", rc_str])

            strategy = Strategy(config_path=Path(__file__).parents[0] / 'config/config_strategy.ini', scale_unit=sc_optim)
            strategy.start_date = self.start_date
            strategy.end_date = self.end_date
            strategy.strategy_risk_class = rc_str
            strategy.strategy_name = f"{strategy.strategy_name}_{rc_str}_{start_date}_{end_date}"
            strategy.manage_portfolio()
            strategy.export_files()
            kpi_dic = Statistic(strategy).get_kpi()
            # kpi_df = pd.DataFrame([kpi_dic]).transpose()
            # kpi_df.rename(columns={0: kpi_dic['Strategy']}, inplace=True)
            # kpi_df.drop("Strategy", inplace=True)
            # kpi_df.to_csv(strategy.root_path / "kpi_slide_insample.csv")

            global_df.loc["Optimale S-Unit", self.risk_class] = sc_optim
            global_df.loc["Total Return", self.risk_class] = kpi_dic["Total Return"]
            global_df.loc["Volatility", self.risk_class] = kpi_dic["Volatility"]
            global_df.loc["MDD", self.risk_class] = kpi_dic["Maximum Drawdown"]
            global_df.loc["# Rebalancing Unit 1", self.risk_class] = kpi_dic["Rebalancing Count UNIT1"]
            global_df.loc["# Rebalancing Unit 2", self.risk_class] = kpi_dic["Rebalancing Count UNIT2"]
            global_df.loc["# Rebalancing Unit 3", self.risk_class] = kpi_dic["Rebalancing Count UNIT3"]
            global_df.loc["Trading Kosten", self.risk_class] = kpi_dic["Trading Cost"]
            # except AttributeError:
            #     None


            # Benchmark
            for bm in ["monthly", "five_percent"]:
                benchmark = Benchmark(config_path=Path('config') / 'config_benchmark.ini')
                benchmark.bm_type = bm
                benchmark.strategy_name = f"{bm}_rebalance"
                root_name = benchmark.strategy_name
                benchmark.strategy_name = f"{root_name}_{self.risk_class}"

                benchmark.strategy_risk_class = self.risk_class
                benchmark.start_date = self.start_date
                benchmark.end_date = self.end_date
                benchmark.manage_portfolio()
                kpi_dic = Statistic(benchmark).get_kpi()
                if bm == "monthly":
                    global_df.loc["Monthly Rebal. Total Return", self.risk_class] = kpi_dic["Total Return"]
                else:
                    global_df.loc["Unit1 Rebal. Total Return", self.risk_class] = kpi_dic["Total Return"]

            global_df.to_csv(strategy.root_path / f"insample_summary_{self.start_date}_{self.end_date}.csv")


    def insample_one_rc(self, start_date, end_date, rc):
        """This functions runs all In Sample Optimizing Strategies."""

        config_path = Path(__file__).parents[0] / 'config/config_strategy.ini'
        tmp_strat = Strategy(config_path=config_path, scale_unit=0.1)
        tmp_strat.manage_portfolio()

        val, idx = min((val, idx) for (idx, val) in enumerate(abs(tmp_strat.details.index - start_date)))
        start_date = tmp_strat.details.index[idx]
        val, idx = min((val, idx) for (idx, val) in enumerate(abs(tmp_strat.details.index - end_date)))
        end_date = tmp_strat.details.index[idx]

        print(f"Optim_{start_date}_{end_date}")

        columns = [str(x * 10) for x in range(1, 11)]
        rows = ["Optimale S-Unit",
                "Total Return",
                "Volatility",
                "MDD",
                "# Rebalancing Unit 1",
                "# Rebalancing Unit 2",
                "# Rebalancing Unit 3",
                "Trading Kosten",
                "Monthly Rebal. Total Return",
                "Unit1 Rebal. Total Return"]

        global_df = pd.DataFrame(columns=columns, index=rows)
        # global_df = pd.read_csv('/Volumes/GoogleDrive/.shortcut-targets-by-id/19JZlTc1zpxipptIN5dg3E-SGtYp5rg7r/02_Backtesting/04_Python Code/source/slide_insample_copy.csv', index_col="Unnamed: 0")
        self.start_date = start_date
        self.end_date = end_date


        rc_str = str(rc * 10)
        self.risk_class = rc_str
        # print(self.risk_class)
        # try:
        res = self.threshold_optimum()
        sc_optim = res.x
        # sc_optim = float(global_df.loc["Optimale S-Unit", rc_str])

        strategy = Strategy(config_path=Path(__file__).parents[0] / 'config/config_strategy.ini', scale_unit=sc_optim)
        strategy.start_date = self.start_date
        strategy.end_date = self.end_date
        strategy.strategy_risk_class = rc_str
        strategy.strategy_name = f"{strategy.strategy_name}_{rc_str}_{start_date}_{end_date}"
        strategy.manage_portfolio()
        strategy.export_files()
        kpi_dic = Statistic(strategy).get_kpi()
        # kpi_df = pd.DataFrame([kpi_dic]).transpose()
        # kpi_df.rename(columns={0: kpi_dic['Strategy']}, inplace=True)
        # kpi_df.drop("Strategy", inplace=True)
        # kpi_df.to_csv(strategy.root_path / "kpi_slide_insample.csv")

        global_df.loc["Optimale S-Unit", self.risk_class] = sc_optim
        global_df.loc["Total Return", self.risk_class] = kpi_dic["Total Return"]
        global_df.loc["Volatility", self.risk_class] = kpi_dic["Volatility"]
        global_df.loc["MDD", self.risk_class] = kpi_dic["Maximum Drawdown"]
        global_df.loc["# Rebalancing Unit 1", self.risk_class] = kpi_dic["Rebalancing Count UNIT1"]
        global_df.loc["# Rebalancing Unit 2", self.risk_class] = kpi_dic["Rebalancing Count UNIT2"]
        global_df.loc["# Rebalancing Unit 3", self.risk_class] = kpi_dic["Rebalancing Count UNIT3"]
        global_df.loc["Trading Kosten", self.risk_class] = kpi_dic["Trading Cost"]
        # except AttributeError:
        #     None


        # Benchmark
        for bm in ["monthly", "five_percent"]:
            benchmark = Benchmark(config_path=Path('config') / 'config_benchmark.ini')
            benchmark.bm_type = bm
            benchmark.strategy_name = f"{bm}_rebalance"
            root_name = benchmark.strategy_name
            benchmark.strategy_name = f"{root_name}_{self.risk_class}"

            benchmark.strategy_risk_class = self.risk_class
            benchmark.start_date = self.start_date
            benchmark.end_date = self.end_date
            benchmark.manage_portfolio()
            kpi_dic = Statistic(benchmark).get_kpi()
            if bm == "monthly":
                global_df.loc["Monthly Rebal. Total Return", self.risk_class] = kpi_dic["Total Return"]
            else:
                global_df.loc["Unit1 Rebal. Total Return", self.risk_class] = kpi_dic["Total Return"]

        global_df.to_csv(strategy.root_path / f"insample_summary_{rc}_{self.start_date}_{self.end_date}.csv")

    def run_insample(self):
        # rolling
        # window_ls = [(dt.date(2001+k, 1, 18), dt.date(2001+10+k, 1, 18)) for k in range(0, 12)]
        # window_ls = [window_ls[1]]
        # with Pool() as pool:
        #     pool.starmap(self.insample_all_rc, window_ls)


        # for window in window_ls:
        #     self.insample_all_rc(window[0], window[1])


        # expanding
        # start_date = dt.date(2001, 1, 18)
        # expanding_window_ls = [(start_date, dt.date(2001 + 10 + k, 1, 18)) for k in range(0, 12)][1:]
        # with Pool() as pool:
        #     pool.starmap(self.insample_all_rc, expanding_window_ls)

        # for window in expanding_window_ls:
        #     self.insample_all_rc(window[0], window[1])
        # expanding_window_ls = [expanding_window_ls[0]]
        # for window in window_ls:
        #     self.insample_one_rc(window[0], window[1], 10)

        # complete
        start_date = dt.date(2005, 1, 4)
        end_date = dt.date(2005, 12, 30)
        self.insample_all_rc(start_date, end_date)
        start_date = dt.date(2016, 1, 5)
        end_date = dt.date(2016, 12, 30)
        self.insample_all_rc(start_date, end_date)


if __name__ == "__main__":
    study = ThresholdOptimizer()
    study.run_insample()

