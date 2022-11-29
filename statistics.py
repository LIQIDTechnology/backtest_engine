import pandas as pd


class Statistics(object):
    def __init__(self, details: pd.DataFrame):
        self.details = details
    days_per_year = 252


    def count_rebalancing(self):
        #  Count of Rebalancing
        rebal_count_unit1 = self.details.loc[:, "UNIT1 Rebalance"].sum()
        rebal_count_unit2 = self.details.loc[:, "UNIT2 Rebalance"].sum()
        rebal_count_unit3 = self.details.loc[:, "UNIT3 Rebalance"].sum()
        rebal_count = self.details.loc[:, "Rebalance"].sum()

    def sharpe_ratio(self):
        # Sharpe Ratio Proxy
        pf_ret = self.details["Portfolio Return"].values[1:]
        avg_ret = np.average(pf_ret)
        sr_1 = ((1 + avg_ret) ** days_per_year) - 1
        stdev_pf_ret = np.std(pf_ret, ddof=1)
        sr_2 = stdev_pf_ret * np.sqrt(days_per_year)
        sr = sr_1 / sr_2
        sr_str = f'{"{:.5f}".format(sr)}'

        # Max Drawdown
        levels = pd.concat([pd.Series(1.0), self.details["Cumulative Portfolio Return"][1:] + 1])
        max = np.maximum.accumulate(levels)
        max_drawdown = (levels / max - 1).min()
        mdd = max_drawdown
        mdd_str = f'{"{:.2f}".format(mdd * 100)} %'

        ## Average annualised return
        arth_avg_pd = pf_ret.mean()
        arth_avg_pa = (arth_avg_pd + 1) ** days_per_year - 1
        arth_avg_pd_str = f'{"{:.4f}".format(arth_avg_pd * 100)} %'

        # Annual Return Vola
        array = np.array(self.details["Portfolio Return"][1:])
        std_pa = (((array + 1) ** 2).mean() ** days_per_year - (array + 1).mean() ** (2 * days_per_year)) ** (1 / 2)
        vola = std_pa
        vola_str = f'{"{:.2f}".format(vola * 100)} %'

        # Total Return
        tot_ret = (self.details.loc[self.end_date, "Cumulative Portfolio Return"])
        tot_ret_str = f'{"{:.2f}".format(tot_ret * 100)} %'

    def get_kpi(self):
        """
        Returns the KPIs in a summarized KPIs
        """
        kpi_dic = {"Strategy": self.strategy_name,
                   "Total Return": tot_ret_str,
                   "Sharpe Ratio Proxy": sr_str,
                   "Rebalancing Count": rebal_count,
                   "Rebalancing Count UNIT1": rebal_count_unit1,
                   "Rebalancing Count UNIT2": rebal_count_unit2,
                   "Rebalancing Count UNIT3": rebal_count_unit3,
                   "Maximum Drawdown": mdd_str,
                   "Average annualised return": arth_avg_pd_str,
                   "Annualized Volatility": vola_str}
        weight1_dic = {cluster: self.details_np[-1, self.unit1_idx[cluster]].sum() for cluster in self.unit1_ls}
        weight2_dic = {cluster: self.details_np[-1, self.unit2_idx[cluster]].sum() for cluster in self.unit2_ls}
        weight3_dic = {cluster: self.details_np[-1, self.unit3_idx[cluster]].sum() for cluster in self.unit3_ls}

        ret_dic = {**kpi_dic, **weight1_dic, **weight2_dic, **weight3_dic}
        return ret_dic