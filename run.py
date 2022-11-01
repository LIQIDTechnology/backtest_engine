from pathlib import Path

import pandas as pd

from strategies.benchmarks import Benchmark
from strategies.strategy import Strategy

if __name__ == "__main__":
    # res = scipy.optimize.minimize_scalar(f, bounds=(0.2, 0.25), method='bounded')
    # print(res.x)

    benchmark_unit1_only = Benchmark(config_path=Path('config') / 'config_unit1.ini')
    benchmark_never = Benchmark(config_path=Path('config') / 'config_never.ini')
    benchmark_monthly = Benchmark(config_path=Path('config') / 'config_monthly.ini')
    benchmark_quarterly = Benchmark(config_path=Path('config') / 'config_quarterly.ini')
    benchmark_annually = Benchmark(config_path=Path('config') / 'config_annually.ini')
    strategy = Strategy(config_path=Path('config') / 'config.ini', scale_unit=0.020925644969531886)
    strategy1 = Strategy(config_path=Path('config') / 'config.ini', scale_unit=0.03305355721784304)
    strategy2 = Strategy(config_path=Path('config') / 'config.ini', scale_unit=0.08646006000121585)
    strategy3 = Strategy(config_path=Path('config') / 'config.ini', scale_unit=0.19999577787513406)
    strategy4 = Strategy(config_path=Path('config') / 'config.ini', scale_unit=0.2999946842693384)
    strategy5 = Strategy(config_path=Path('config') / 'config.ini', scale_unit=0.34999468501095826)

    kpi_dic1 = benchmark_unit1_only.manage_portfolio()
    kpi_dic2 = benchmark_never.manage_portfolio()
    kpi_dic3 = benchmark_monthly.manage_portfolio()
    kpi_dic4 = benchmark_quarterly.manage_portfolio()
    kpi_dic5 = benchmark_annually.manage_portfolio()
    kpi_dic6 = strategy.manage_portfolio()
    kpi_dic7 = strategy1.manage_portfolio()
    kpi_dic8 = strategy2.manage_portfolio()
    kpi_dic9 = strategy3.manage_portfolio()
    kpi_dic10 = strategy4.manage_portfolio()
    kpi_dic11 = strategy5.manage_portfolio()

    kpi1_df = pd.DataFrame([kpi_dic1])
    kpi2_df = pd.DataFrame([kpi_dic2])
    kpi3_df = pd.DataFrame([kpi_dic3])
    kpi4_df = pd.DataFrame([kpi_dic4])
    kpi5_df = pd.DataFrame([kpi_dic5])
    kpi6_df = pd.DataFrame([kpi_dic6])

    # kpi = pd.concat([kpi1_df, kpi2_df, kpi3_df, kpi4_df])
    kpi = pd.concat([kpi1_df, kpi2_df, kpi3_df, kpi4_df, kpi5_df, kpi6_df])
    folderpath = Path("/Volumes/GoogleDrive/My Drive/0003_QPLIX/004_Backtest Engine/input")
    filename = "kpi_summary.csv"
    kpi.to_csv(folderpath / filename)
