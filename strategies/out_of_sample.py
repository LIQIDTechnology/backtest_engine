from pathlib import Path
from typing import Union
from functools import lru_cache

import numpy as np
import pandas as pd
import datetime as dt

from portfolio import Portfolio
from instrument import Instrument
from threshold_optim import ThresholdOptimizer
from statistics import Statistic
from strategies.benchmarks import Benchmark


class OutOfSample(Portfolio):
    """
    Vanilla Strategy, where the Rebalancing Logic is steered over the Threshold Scaling Unit

    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Declaring variables upon Object Initialization
        """
        super().__init__(config_path)
        self.mode = None
        self.window = int(self.config["strategy"]["window"])
        self.unit3_scale = None
        self.scaling_unit_col = None

        self.unit1_thres = {}
        self.unit2_thres = {}
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

        self.unit1_col = None
        self.unit2_col = None
        self.unit3_col = None

        self.unit1_rebalance_col = None
        self.unit2_rebalance_col = None
        self.unit3_rebalance_col = None

        self.unit3_ls = None
        self.unit2_ls = None
        self.unit1_ls = None

        self.eval_scaling_unit_ls = []

    def init_details_strat(self):
        self.set_unit_ls()
        strat_ls = [*self.unit3_ls, *self.unit2_ls, *self.unit1_ls]
        other_ls = ["UNIT1 Rebalance", "UNIT2 Rebalance", "UNIT3 Rebalance", "Scaling Unit"]

        res_ls = [x for x in [*strat_ls, *other_ls] if x is not None]
        return res_ls

    def wake_up_strat(self):
        self.scaling_unit_col = self.details.columns.get_loc("Scaling Unit")

        self.unit1_col = [self.details.columns.get_loc(col) for col in self.unit1_ls]
        self.unit2_col = [self.details.columns.get_loc(col) for col in self.unit2_ls]
        self.unit3_col = [self.details.columns.get_loc(col) for col in self.unit3_ls]

        self.unit1_rebalance_col = self.details.columns.get_loc("UNIT1 Rebalance")
        self.unit2_rebalance_col = self.details.columns.get_loc("UNIT2 Rebalance")
        self.unit3_rebalance_col = self.details.columns.get_loc("UNIT3 Rebalance")

        self.unit1_thres_breach = {cluster: self.details.columns.get_loc(cluster) for cluster in self.unit1_ls}
        self.unit2_thres_breach = {cluster: self.details.columns.get_loc(cluster) for cluster in self.unit2_ls}
        self.unit3_thres_breach = {cluster: self.details.columns.get_loc(cluster) for cluster in self.unit3_ls}

        self.set_cluster_column_idx()
        self.set_cluster_weight()
        self.set_unit_mapping()
        self.set_unit1_threshold()
        self.set_unit2_threshold()

        next_ls = [dt.date(self.start_date.year+k, self.start_date.month, self.start_date.day) for k in range(0, 20)]
        for next_date in next_ls:
            val, idx = min((val, idx) for (idx, val) in enumerate(abs(self.details.index - next_date)))
            self.eval_scaling_unit_ls.append(idx)
        self.eval_scaling_unit_ls = list(set(self.eval_scaling_unit_ls))
        self.eval_scaling_unit_ls.remove(max(self.eval_scaling_unit_ls))

        # self.set_unit3_threshold()

    def set_unit_mapping(self):
        """
        This function creates mapping dictionaries between
        Unit III - Unit II
        Unit II - Unit I
        """
        self.unit3to2_map = dict(zip(self.instruments_table["UNIT III"], self.instruments_table["UNIT II"]))
        self.unit2to1_map = dict(zip(self.instruments_table["UNIT II"], self.instruments_table["UNIT I"]))

    def set_cluster_weight(self):
        """
        This function sets the weight on Cluster Level
        """
        for cluster in self.unit1_ls:
            wts_sum = self.instruments_table[self.instruments_table["UNIT I"] == cluster][self.strategy_risk_class].sum()
            self.unit1_weights[cluster] = wts_sum
        for cluster in self.unit2_ls:
            wts_sum = self.instruments_table[self.instruments_table["UNIT II"] == cluster][self.strategy_risk_class].sum()
            self.unit2_weights[cluster] = wts_sum
        for cluster in self.unit3_ls:
            wts_sum = self.instruments_table[self.instruments_table["UNIT III"] == cluster][self.strategy_risk_class].sum()
            self.unit3_weights[cluster] = wts_sum

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

    def set_unit_ls(self):
        self.unit1_ls = list(set(inst.unit1 for inst in self.instrument_ls if inst.unit1 is not None))
        self.unit2_ls = list(set(inst.unit2 for inst in self.instrument_ls if inst.unit2 is not None))
        self.unit3_ls = list(set(inst.unit3 for inst in self.instrument_ls if inst.unit3 is not None))

    def set_unit1_threshold(self):
        """
        Set Unit I Threshold, set to 5%
        """

        def get_cluster_weight(instruments_table, rc, cluster):
            """This function gets the weight on Cluster Level"""
            wts_sum = instruments_table[instruments_table["UNIT I"] == cluster][rc].sum()
            return wts_sum

        for cluster in self.unit1_ls:
            this_rc = int(self.strategy_risk_class)
            this_rc = 90 if this_rc == 100 else this_rc
            next_rc = this_rc + 10
            curr_cluster_weight = get_cluster_weight(self.instruments_table, str(this_rc), cluster)
            next_cluster_weight = get_cluster_weight(self.instruments_table, str(next_rc), cluster)
            self.unit1_thres[cluster] = abs(curr_cluster_weight - next_cluster_weight) / 2

    def set_unit2_threshold(self):
        """
        Set UNIT II Threshold, logic derived from current methodology
        """
        def get_cluster_weight(instruments_table, rc, cluster):
            """This function gets the weight on Cluster Level"""
            wts_sum = instruments_table[instruments_table["UNIT II"] == cluster][rc].sum()
            return wts_sum

        for cluster in self.unit2_ls:
            if cluster in ["GOLD", "CASH", "COM"]:
                self.unit2_thres[cluster] = 0.01
            else:
                this_rc = int(self.strategy_risk_class)

                next_rc = this_rc + 10
                curr_cluster_weight = get_cluster_weight(self.instruments_table, str(this_rc), cluster)
                try:
                    next_cluster_weight = get_cluster_weight(self.instruments_table, str(next_rc), cluster)
                except KeyError:
                    prev_rc = this_rc - 10
                    next_cluster_weight = get_cluster_weight(self.instruments_table, str(prev_rc), cluster)

                thres = abs(curr_cluster_weight - next_cluster_weight) / 2

                if cluster == 'BONDS HY':
                    thres = curr_cluster_weight * 0.1 if thres / curr_cluster_weight < 0.1 else thres
                elif cluster == 'EQU EM':
                    thres = curr_cluster_weight * 0.09 if thres / curr_cluster_weight < 0.09 else thres

                self.unit2_thres[cluster] = thres

    def get_optim_unit3(self, t):

        # study = ThresholdOptimizer()
        # print(t)
        tbl = pd.read_csv(self.prices_path)
        tbl["Date"] = tbl["Date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").date())
        tbl = tbl.set_index("Date")

        if self.mode == "oos_rolling":
            today = self.details.index[t]
            start_date = dt.date(today.year - 10, 1, 18)
            val, idx = min((val, idx) for (idx, val) in enumerate(abs(tbl.index - start_date)))
            start_date = tbl.index[idx]
        elif self.mode == "oos_expanding":
            today = self.details.index[t]
            start_date = dt.date(2001, 1, 18)
        elif self.mode == "is_future-rolling":

            today = self.details.index[t]

            start_date = dt.date(today.year - 10, 1, 18)
            val, idx = min((val, idx) for (idx, val) in enumerate(abs(tbl.index - start_date)))
            start_date = tbl.index[idx]

            start_date_dic_map = {"2001-01-18": "2002-01-18",
                                  "2002-01-18": "2003-01-17",
                                  "2003-01-17": "2004-01-19",
                                  "2004-01-19": "2005-01-18",
                                  "2005-01-18": "2006-01-18",
                                  "2006-01-18": "2007-01-18",
                                  "2007-01-18": "2008-01-18",
                                  "2008-01-18": "2009-01-19",
                                  "2009-01-19": "2010-01-18",
                                  "2010-01-18": "2011-01-18",
                                  "2011-01-18": "2012-01-18",
                                  "2012-01-18": "2022-01-18"}

            start_date = start_date_dic_map[start_date.strftime("%Y-%m-%d")]

            end_date_dic_map = {"2011-01-18": "2012-01-18",
                                "2012-01-18": "2013-01-18",
                                "2013-01-18": "2014-01-17",
                                "2014-01-17": "2015-01-19",
                                "2015-01-19": "2016-01-18",
                                "2016-01-18": "2017-01-18",
                                "2017-01-18": "2018-01-18",
                                "2018-01-18": "2019-01-18",
                                "2019-01-18": "2020-01-17",
                                "2020-01-17": "2021-01-18",
                                "2021-01-18": "2022-01-18"}

            today = end_date_dic_map[today.strftime("%Y-%m-%d")]
            today = dt.datetime.strptime(today, "%Y-%m-%d", ).date()

            pass
        elif self.mode == "is-future-expanding":
            today = self.details.index[t]
            start_date = dt.date(2001, 1, 18)

            end_date_dic_map = {"2011-01-18": "2012-01-18",
                                "2012-01-18": "2013-01-18",
                                "2013-01-18": "2014-01-17",
                                "2014-01-17": "2015-01-19",
                                "2015-01-19": "2016-01-18",
                                "2016-01-18": "2017-01-18",
                                "2017-01-18": "2018-01-18",
                                "2018-01-18": "2019-01-18",
                                "2019-01-18": "2020-01-17",
                                "2020-01-17": "2021-01-18",
                                "2021-01-18": "2022-01-18"}
            today = end_date_dic_map[today.strftime("%Y-%m-%d")]
            today = dt.datetime.strptime(today, "%Y-%m-%d", ).date()

        print(start_date, today)

        filepath = Path("/Volumes/GoogleDrive/.shortcut-targets-by-id/19JZlTc1zpxipptIN5dg3E-SGtYp5rg7r/02_Backtesting/04_Python Code/results")
        filename = f"insample_summary_{start_date}_{today}.csv"
        sc_optim_summary = pd.read_csv(filepath / filename, index_col=0)

        # start_date = dt.date(today.year-self.window, 1, 18)
        # end_date = today
        # sc_optim = study.study_insample(start_date, end_date, self.strategy_risk_class)
        sc_optim = float(sc_optim_summary.loc["Optimale S-Unit", self.strategy_risk_class])
        return sc_optim

    def set_unit3_threshold(self, t):
        """
        Set UNIT III Threshold, logic derived from current methodology
        """

        self.unit3_scale = self.get_optim_unit3(t)

        for cluster in self.unit3_ls:
            unit2_cluster = self.unit3to2_map[cluster]
            unit2_wt = self.unit2_weights[unit2_cluster]
            unit3_wt = self.unit3_weights[cluster]
            self.unit3_thres[cluster] = abs(unit3_wt / unit2_wt * self.unit3_scale)

    def check_rebal_cluster(self, cluster_str, t):
        """
        Method to identify breaches on Thresholds on an explicit UNIT
        """
        unit_ls = getattr(self, f"{cluster_str}_ls")
        unit_idx = getattr(self, f"{cluster_str}_idx")
        unit_weights = getattr(self, f"{cluster_str}_weights")
        unit_thres = getattr(self, f"{cluster_str}_thres")
        unit_thres_breach = getattr(self, f"{cluster_str}_thres_breach")
        unit_rebalance_col = getattr(self, f"{cluster_str}_rebalance_col")

        for cluster in unit_ls:
            cluster_wt = unit_weights[cluster]
            weight = self.details_np[t - 1, unit_idx[cluster]].sum()
            thres = unit_thres[cluster]

            if cluster_wt - thres < weight < cluster_wt + thres:
                pass
            else:
                self.details_np[t, unit_thres_breach[cluster]] = 1
                self.details_np[t, unit_rebalance_col] = 1

    def check_rebal(self, t):
        """
        Daily Method to identify breaches on Thresholds on all UNIT Levels
        """
        self.check_rebal_cluster("unit1", t)
        self.check_rebal_cluster("unit2", t)
        self.check_rebal_cluster("unit3", t)

        rebal_ls = [*self.unit1_col, *self.unit2_col, *self.unit3_col]
        rebal_bool = True if np.nansum(self.details_np[t, rebal_ls]) >= 1 else False
        self.details_np[t, self.rebalance_col] = 1 if rebal_bool else None

        return rebal_bool

    def routine(self, t):
        """
        Daily Routine what is calculated on each day / over each row
        """

        self.set_unit3_threshold(t) if t in self.eval_scaling_unit_ls else None
        self.details_np[t, self.scaling_unit_col] = self.unit3_scale
        self.details.loc[self.details.index[t], "Scaling Unit"] = self.unit3_scale
        if self.details.index[t] == self.start_date:  # INIT WEIGHTS
            self.reset_weights(t - 1)
            self.details_np[0, self.pf_ret_col] = 0
            self.details_np[0, self.pf_cum_ret_col] = 0
            self.details_np[0, self.inv_col_np] = self.details_np[0, self.wts_col_np] * self.amount_inv
            self.details_np[0, self.hyp_amount_inv_col] = self.details_np[0, self.inv_col_np].sum()
        self.calc_portfolio_ret(t)


if __name__ == "__main__":

    mode_ls = ["oos_rolling", "oos_expanding", "is_future-rolling", "is-future-expanding"]
    # mode_ls = ["is-future-expanding"]

    for mode in mode_ls:

        columns = [str(x * 10) for x in range(1, 11)]
        rows = ["Total Return",
                "Volatility",
                "MDD",
                "# Rebalancing Unit 1",
                "# Rebalancing Unit 2",
                "# Rebalancing Unit 3",
                "Trading Kosten",
                "Monthly Rebal. Total Return",
                "Unit1 Rebal. Total Return",
                "2011 TR",
                "2012 TR",
                "2013 TR",
                "2014 TR",
                "2015 TR",
                "2016 TR",
                "2017 TR",
                "2018 TR",
                "2019 TR",
                "2020 TR",
                "2021 TR",
                "2022 TR"]

        global_df = pd.DataFrame(columns=columns, index=rows)

        for rc in range(1, 11):

            rc_str = str(rc*10)

            strategy = OutOfSample(config_path=Path(__file__).parents[1] / 'config/config_out_of_sample.ini')
            strategy.strategy_risk_class = rc_str
            strategy.strategy_name = f"{mode}_{rc_str}"
            strategy.mode = mode
            strategy.manage_portfolio()
            strategy.export_files()  # Exports Detail Sheet (also triggered in manage_portfolio)
            kpi_dic = Statistic(strategy).get_kpi()

            global_df.loc["Total Return", rc_str] = kpi_dic["Total Return"]
            global_df.loc["Volatility", rc_str] = kpi_dic["Volatility"]
            global_df.loc["MDD", rc_str] = kpi_dic["Maximum Drawdown"]
            global_df.loc["# Rebalancing Unit 1", rc_str] = kpi_dic["Rebalancing Count UNIT1"]
            global_df.loc["# Rebalancing Unit 2", rc_str] = kpi_dic["Rebalancing Count UNIT2"]
            global_df.loc["# Rebalancing Unit 3", rc_str] = kpi_dic["Rebalancing Count UNIT3"]
            global_df.loc["Trading Kosten", rc_str] = kpi_dic["Trading Cost"]
            global_df.loc["2011 TR", rc_str] = kpi_dic["TR 2011"]
            global_df.loc["2012 TR", rc_str] = kpi_dic["TR 2012"]
            global_df.loc["2013 TR", rc_str] = kpi_dic["TR 2013"]
            global_df.loc["2014 TR", rc_str] = kpi_dic["TR 2014"]
            global_df.loc["2015 TR", rc_str] = kpi_dic["TR 2015"]
            global_df.loc["2016 TR", rc_str] = kpi_dic["TR 2016"]
            global_df.loc["2017 TR", rc_str] = kpi_dic["TR 2017"]
            global_df.loc["2018 TR", rc_str] = kpi_dic["TR 2018"]
            global_df.loc["2019 TR", rc_str] = kpi_dic["TR 2019"]
            global_df.loc["2020 TR", rc_str] = kpi_dic["TR 2020"]
            global_df.loc["2021 TR", rc_str] = kpi_dic["TR 2021"]
            global_df.loc["2022 TR", rc_str] = kpi_dic["TR 2022"]

            # for bm in ["monthly", "five_percent"]:
            #     benchmark = Benchmark(config_path=Path(__file__).parents[0] / 'config_benchmark.ini')
            #     benchmark.bm_type = bm
            #     benchmark.strategy_name = f"{bm}_rebalance"
            #     root_name = benchmark.strategy_name
            #     benchmark.strategy_name = f"{root_name}_{rc_str}"
            #
            #     benchmark.strategy_risk_class = rc_str
            #     benchmark.start_date = strategy.start_date
            #     benchmark.end_date = strategy.end_date
            #     benchmark.manage_portfolio()
            #     kpi_dic = Statistic(benchmark).get_kpi()
            #     if bm == "monthly":
            #         global_df.loc["Monthly Rebal. Total Return", rc_str] = kpi_dic["Total Return"]
            #     else:
            #         global_df.loc["Unit1 Rebal. Total Return", rc_str] = kpi_dic["Total Return"]

        global_df.to_csv(strategy.root_path / f"{mode}_summary_{strategy.start_date}_{strategy.end_date}.csv")