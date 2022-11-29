from pathlib import Path
import pandas as pd
from strategies.benchmarks import Benchmark
from strategies.strategy import Strategy
from strategies.out_of_sample import StrategyOutOfSample
from threshold_optim import ThresholdOptimizer
import datetime as dt

if __name__ == "__main__":
    # Benchmark (quarterly, monthly, annually, never, five_percent)
    benchmark_unit1_only = Benchmark(config_path=Path('config') / 'config_benchmark.ini', benchmark_type='five_percent')
    benchmark_unit1_only.manage_portfolio()
    benchmark_unit1_only.export_files()
    benchmark_never = Benchmark(config_path=Path('config') / 'config_benchmark.ini', benchmark_type='never')
    benchmark_never.manage_portfolio()
    benchmark_never.export_files()
    benchmark_monthly = Benchmark(config_path=Path('config') / 'config_benchmark.ini', benchmark_type='monthly')
    benchmark_monthly.manage_portfolio()
    benchmark_monthly.export_files()
    benchmark_quarterly = Benchmark(config_path=Path('config') / 'config_benchmark.ini', benchmark_type='quarterly')
    benchmark_quarterly.manage_portfolio()
    benchmark_quarterly.export_files()
    benchmark_annually = Benchmark(config_path=Path('config') / 'config_benchmark.ini', benchmark_type='annually')
    benchmark_annually.manage_portfolio()
    benchmark_annually.export_files()

    # Optimal Strategy
    # study = ThresholdOptimizer(start_date=dt.date(2001, 1, 18), end_date=dt.date(2022, 10, 18))
    # res = study.threshold_optimum()
    # optim_strategy = Strategy(config_path=Path('config') / 'config_strategy.ini', scale_unit=res.x)
    # optim_strategy.manage_portfolio()
    #
    # # Out of Sample
    # # Rolling
    # start_scale_value = 0.05
    # strategy = StrategyOutOfSample(config_path=Path('config') / 'config_strategy.ini', scale_unit=start_scale_value, optim_type='rolling')
    # strategy.manage_portfolio()
    # strategy.export_files()
    # # Expanding
    # start_scale_value = 0.05
    # strategy = StrategyOutOfSample(config_path=Path('config') / 'config_strategy.ini', scale_unit=start_scale_value, optim_type='expanding')
    # strategy.manage_portfolio()
    # strategy.export_files()

    # Output KPIs into DataFrame
    strategy_obj_ls = []
    # strategy_obj_ls = [optim_strategy, benchmark_unit1_only, benchmark_never, benchmark_monthly, benchmark_quarterly, benchmark_annually]
    strategy_obj_ls = [benchmark_unit1_only, benchmark_never, benchmark_monthly, benchmark_quarterly, benchmark_annually]
    kpi_df_ls = []
    for strategy in strategy_obj_ls:
        kpi_dic = strategy.get_kpi()
        kpi_df = pd.DataFrame([kpi_dic])
        kpi_df_ls.append(kpi_df)
    kpi_all_df = pd.concat(kpi_df_ls)
    # kpi_all_df.insert(0, 'Optimal Threshold', None)
    kpi_all_df.set_index("Strategy", inplace=True)
    # kpi_all_df.loc[optim_strategy.strategy_name, "Optimal Threshold"] = res.x
    folderpath = benchmark_monthly.root_path
    filename = "kpi_summary.csv"
    kpi_all_df.to_csv(folderpath / filename)
    print(f'KPI Summary exported to {folderpath / filename}')
