import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
from typing import Union
from portfolio import Portfolio


class Benchmark(Portfolio):
    """
    Benchmark Strategy, includes strategies with simple rebalancing logics

    """
    def __init__(self, config_path: Union[str, Path], benchmark_type: str):
        """
        Declaring variables upon Object Initialization
        """
        super().__init__(config_path)
        self.bm_type = benchmark_type
        self.strategy_name = f"{benchmark_type}_rebalance"

    def check_rebal(self, t):
        """
        Daily Method to identify Rebalance Triggers based on Benchmark Type

        1. QUARTERLY
        2. MONTHLY
        3. ANNUALY
        4. BREACH OF 5%
        """
        today = self.details.index[t]  # Today
        if today == self.start_date:
            rebal_bool = False
        else:
            td_month = self.details.index[t].month  # Today's Month
            ytd_month = self.details.index[t-1].month  # Yesterday's Month

            if self.bm_type == "quarterly":
                rebal_bool = True if td_month != ytd_month and td_month in (3, 6, 9, 12) else False
            elif self.bm_type == "monthly":
                rebal_bool = True if td_month != ytd_month else False
            elif self.bm_type == "annually":
                rebal_bool = True if td_month != ytd_month and td_month == 1 else False
            elif self.bm_type == "never":
                rebal_bool = False
            elif self.bm_type == "five_percent":
                for cluster in self.unit1_ls:
                    weight = self.details_np[t-1, self.unit1_idx[cluster]].sum()
                    cluster_wt = self.unit1_weights[cluster]
                    thres = 0.05
                    if cluster_wt - thres < weight < cluster_wt + thres:
                        pass  # If within Threshold Range do Nothing
                    else:
                        self.details_np[t, self.unit1_thres_breach[cluster]] = 1
                        self.details_np[t, self.unit1_rebalance_col] = 1
                        self.details_np[t, self.rebalance_col] = 1
                rebal_bool = True if np.nansum(self.details_np[t, self.unit1_col]) >= 1 else False
                self.details_np[t, self.rebalance_col] = 1 if rebal_bool else None
        self.details_np[t, self.rebalance_col] = 1 if rebal_bool else None
        return rebal_bool

    def routine(self, t):
        """
        Daily Routine what is calculated on each day / over each row
        """
        self.reset_weights(t-1) if self.details.index[t] == self.start_date else None  # INIT WEIGHTS
        self.calc_portfolio_ret(t)
        self.reset_weights(t) if self.check_rebal(t) else self.calc_weights(t)


if __name__ == "__main__":
    # Example Execution of Strategy Object
    config_folder = Path(__file__).parents[1]
    config_path = config_folder / 'config/config_benchmark.ini'  # same for all benchmark types
    # quarterly, monthly, annually, nothing, five_percent"
    benchmark_never = Benchmark(config_path=config_path, benchmark_type='never')
    benchmark_never.manage_portfolio()  # Returns a KPI Dictionary
    benchmark_never.export_files()  # Exports Detail Sheet (also triggered in manage_portfolio)

    # Output KPIs into DataFrame
    kpi_dic = benchmark_never.get_kpi()  # Returns a KPI Dictionary
    kpi_df = pd.DataFrame([kpi_dic])  # Convert to DataFrame
    folderpath = benchmark_never.root_path
    filename = "benchmark_never.csv"
    kpi_df.to_csv(folderpath / filename)
    print(f'KPIs exported to {folderpath / filename}')



