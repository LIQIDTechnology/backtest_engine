import pandas as pd
import numpy as np
import datetime as dt
from portfolio import Portfolio


class Statistic(object):
    def __init__(self, strategy: Portfolio):
        self.strategy = strategy
        self.kpi_dic = {}
        self.calc_kpi()

    def calc_total_return(self):
        tot_ret = (self.strategy.details.loc[self.strategy.end_date, "Cumulative Portfolio Return"])
        tot_ret_str = f'{"{:.2f}".format(tot_ret * 100)} %'
        return tot_ret_str

    def calc_average_annual_return(self):
        pf_ret = self.strategy.details["Portfolio Return"].values[1:]
        arth_avg_pd = pf_ret.mean()
        arth_avg_pa = (arth_avg_pd + 1) ** self.strategy.trading_days - 1
        arth_avg_pd_str = f'{"{:.2f}".format(arth_avg_pa * 100)} %'
        return arth_avg_pd_str

    def calc_volatility(self):
        returns = self.strategy.details["Portfolio Return"]
        std = returns.std()
        vola = std * np.sqrt(self.strategy.trading_days)

        # Print the annual volatility
        vola_str = f'{"{:.2f}".format(vola * 100)} %'
        return vola_str

    def calc_mdd(self):
        levels = pd.concat([pd.Series(1.0), self.strategy.details["Cumulative Portfolio Return"][1:] + 1])
        max = np.maximum.accumulate(levels)
        max_drawdown = (levels / max - 1).min()
        mdd = max_drawdown
        mdd_str = f'{"{:.2f}".format(mdd * 100)} %'
        return mdd_str

    def calc_trading_cost(self):
        return self.strategy.details["Hyp Trading Cost"].sum()

    def calc_kpi(self):
        self.kpi_dic.update({"Strategy": self.strategy.strategy_name})
        self.kpi_dic.update({"Total Return": self.calc_total_return()})
        self.kpi_dic.update({"Rebalancing Count": self.strategy.details.loc[:, "Rebalance"].sum()})
        self.kpi_dic.update({"Average Return p.a.": self.calc_average_annual_return()})
        self.kpi_dic.update({"Volatility": self.calc_volatility()})
        self.kpi_dic.update({"Maximum Drawdown": self.calc_mdd()})
        self.kpi_dic.update({"Start Date": self.strategy.start_date})
        self.kpi_dic.update({"End Date": self.strategy.end_date})
        self.kpi_dic.update({"Trading Cost": self.calc_trading_cost()})
        try:
            self.kpi_dic.update({"Scaling Unit": self.strategy.unit3_scale})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count UNIT1": self.strategy.details.loc[:, "UNIT1 Rebalance"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count UNIT2": self.strategy.details.loc[:, "UNIT2 Rebalance"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count UNIT3": self.strategy.details.loc[:, "UNIT3 Rebalance"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Threshold L": self.strategy.unit1_thres['L']})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Threshold H": self.strategy.unit1_thres['H']})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Threshold Bonds IG": self.strategy.unit2_thres['BONDS IG']})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Threshold Bonds HY": self.strategy.unit2_thres['BONDS HY']})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Threshold EQU DM": self.strategy.unit2_thres['EQU DM']})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Threshold EQU EM": self.strategy.unit2_thres['EQU EM']})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Threshold Com": self.strategy.unit2_thres['COM']})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Threshold Gold": self.strategy.unit2_thres['GOLD']})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Threshold Cash": self.strategy.unit2_thres['CASH']})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Threshold Equities Asia Pacific ex Japan": self.strategy.unit3_thres['Equities Asia Pacific ex Japan']})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Threshold Equities Emerging Markets World": self.strategy.unit3_thres['Equities Emerging Markets World']})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Threshold Equities Europe": self.strategy.unit3_thres['Equities Europe']})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Threshold Equities Japan": self.strategy.unit3_thres['Equities Japan']})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Threshold Equities North America": self.strategy.unit3_thres['Equities North America']})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count L": self.strategy.details.loc[:, "L"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count H": self.strategy.details.loc[:, "H"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count Bonds IG": self.strategy.details.loc[:, "BONDS IG"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count Bonds HY": self.strategy.details.loc[:, "BONDS HY"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count EQU DM": self.strategy.details.loc[:, "EQU DM"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count EQU EM": self.strategy.details.loc[:, "EQU EM"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count Com": self.strategy.details.loc[:, "COM"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count Gold": self.strategy.details.loc[:, "GOLD"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count Cash": self.strategy.details.loc[:, "CASH"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count Equities Asia Pacific Ex Japan": self.strategy.details.loc[:, "Equities Asia Pacific ex Japan"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count Equities Emerging Markets World": self.strategy.details.loc[:, "Equities Emerging Markets World"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count Equities Europe": self.strategy.details.loc[:, "Equities Europe"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count Equities Japan": self.strategy.details.loc[:, "Equities Japan"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        try:
            self.kpi_dic.update({"Rebalancing Count Equities North America": self.strategy.details.loc[:, "Equities North America"].sum()})
        except (AttributeError, KeyError) as e:
            pass
        year_ls = [x.year for x in self.strategy.details.index]
        year_ls = list(dict.fromkeys(year_ls))
        for year in year_ls:
            try:
                self.kpi_dic.update({f"TR {year}": self.get_total_return_year(year)})
            except (AttributeError, KeyError) as e:
                pass

        try:
            self.kpi_dic.update({"Rebalancing Count Equities North America": self.strategy.details.loc[:, "Equities North America"].sum()})
        except (AttributeError, KeyError) as e:
            pass

    def get_total_return_year(self, year):
        year_ls = [x.year for x in self.strategy.details.index]
        year_ls = list(dict.fromkeys(year_ls))
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)

        year_df = self.strategy.details.loc[start_date:end_date, ["Portfolio Return", "Cumulative Portfolio Return"]]


        year_df["Temp"] = (year_df["Portfolio Return"] + 1).cumprod() - 1
        return year_df["Temp"][-1]

    def get_total_return(self):
        tot_ret = (self.strategy.details.loc[self.strategy.end_date, "Cumulative Portfolio Return"])
        return tot_ret

    def get_total_rebal(self):

        tot_rebal = self.strategy.details.loc[:, "Rebalance"].sum()
        return tot_rebal

    def get_kpi(self):
        """
        Returns the KPIs in a summarized KPIs
        """
        return self.kpi_dic