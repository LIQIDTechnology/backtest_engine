from pathlib import Path

import pandas as pd

from strategies.benchmarks import Benchmark
from strategies.strategy import Strategy

if __name__ == "__main__":
    # res = scipy.optimize.minimize_scalar(f, bounds=(0.2, 0.25), method='bounded')
    # print(res.x)

    config_path = Path('config') / 'config.ini'
    strategy = Strategy(config_path=config_path, scale_unit=0.0845)

    benchmark_unit1_only = Benchmark(config_path=Path('config') / 'config_unit1-only.ini')
    benchmark_never = Benchmark(config_path=Path('config') / 'config_never.ini')
    benchmark_monthly = Benchmark(config_path=Path('config') / 'config_monthly.ini')
    # benchmark_annualy = Benchmark(config_path=Path('config') / 'config_annualy.ini')
    # benchmark_quarterly = Benchmark(config_path=Path('config') / 'config_quarterly.ini')

    kpi_dic1 = benchmark_unit1_only.manage_portfolio()
    kpi_dic2 = benchmark_never.manage_portfolio()
    kpi_dic3 = benchmark_monthly.manage_portfolio()
    kpi_dic4 = strategy.manage_portfolio()


    kpi1_df = pd.DataFrame([kpi_dic1])
    kpi2_df = pd.DataFrame([kpi_dic2])
    kpi3_df = pd.DataFrame([kpi_dic3])
    kpi4_df = pd.DataFrame([kpi_dic4])


    kpi = pd.concat([kpi1_df, kpi2_df, kpi3_df, kpi4_df])
    folderpath = Path("/Volumes/GoogleDrive/.shortcut-targets-by-id/19JZlTc1zpxipptIN5dg3E-SGtYp5rg7r/02_Backtesting/02_Daten/input")
    filename = "kpi_summary.csv"
    kpi.to_csv(folderpath / filename)
