from pathlib import Path
import pandas as pd
from strategies.benchmarks import Benchmark
from strategies.strategy import Strategy
from threshold_optim import ThresholdOptimizer

if __name__ == "__main__":
    # Benchmark (quarterly, monthly, annually, never, five_percent)
    benchmark_unit1_only = Benchmark(config_path=Path('config') / 'config_benchmark.ini', benchmark_type='five_percent')
    benchmark_unit1_only.manage_portfolio()
    benchmark_never = Benchmark(config_path=Path('config') / 'config_benchmark.ini', benchmark_type='never')
    benchmark_never.manage_portfolio()
    benchmark_monthly = Benchmark(config_path=Path('config') / 'config_benchmark.ini', benchmark_type='monthly')
    benchmark_monthly.manage_portfolio()
    benchmark_quarterly = Benchmark(config_path=Path('config') / 'config_benchmark.ini', benchmark_type='quarterly')
    benchmark_quarterly.manage_portfolio()
    benchmark_annually = Benchmark(config_path=Path('config') / 'config_benchmark.ini', benchmark_type='annually')
    benchmark_annually.manage_portfolio()

    # Optimal Strategy
    study = ThresholdOptimizer()
    res = study.threshold_optimum()
    optim_strategy = Strategy(config_path=Path('config') / 'config.ini', scale_unit=res.x)
    optim_strategy.manage_portfolio()

    # Output KPIs into DataFrame
    strategy_obj_ls = []
    strategy_obj_ls = [optim_strategy, benchmark_unit1_only, benchmark_never, benchmark_monthly, benchmark_quarterly, benchmark_annually]
    kpi_df_ls = []
    for strategy in strategy_obj_ls:
        kpi_dic = strategy.get_kpi()
        kpi_df = pd.DataFrame([kpi_dic])
        kpi_df_ls.append(kpi_df)
    kpi_all_df = pd.concat(kpi_df_ls)
    kpi_all_df.insert(0, 'Optimal Threshold', None)
    kpi_all_df.set_index("Strategy", inplace=True)
    kpi_all_df.loc[optim_strategy.strategy_name, "Optimal Threshold"] = res.x
    folderpath = benchmark_unit1_only.root_path
    filename = "kpi_summary.csv"
    kpi_all_df.to_csv(folderpath / filename)
    print(f'KPI Summary exported to {folderpath / filename}')
