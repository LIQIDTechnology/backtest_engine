from pathlib import Path
import pandas as pd
from strategies.benchmarks import Benchmark
from strategies.strategy import Strategy
from strategies.out_of_sample import StrategyOutOfSample
from threshold_optim import ThresholdOptimizer
import datetime as dt

if __name__ == "__main__":
    # Benchmark (quarterly, monthly, annually, never, five_percent)

    bm_type_ls = ['quarterly', 'monthly', 'annually', 'never', 'five_percent']
    bm_type_ls = ['monthly']

    for bm in bm_type_ls:
        benchmark = Benchmark(config_path=Path('config') / 'config_benchmark.ini')
        benchmark.bm_type = bm
        benchmark.strategy_name = f"{bm}_rebalance"
        root_name = benchmark.strategy_name
        kpi_df_ls = []
        for rc in range(1, 11):
            benchmark.strategy_name = f"{root_name}_{rc * 10}"
            benchmark.strategy_risk_class = str(rc * 10)
            benchmark.manage_portfolio()
            benchmark.export_files()
            # benchmark.qplix_upload()
            kpi_dic = benchmark.get_kpi()
            kpi_df = pd.DataFrame([kpi_dic])
            kpi_df_ls.append(kpi_df)
    kpi_all_df = pd.concat(kpi_df_ls)
    kpi_all_df.set_index("Strategy", inplace=True)
    folderpath = benchmark.root_path
    filename = f"kpi_summary_{bm}.csv"

    # kpi_df_ls = []
    # scale_unit = 0.09
    # optim_strategy = Strategy(config_path=Path('config') / 'config_strategy.ini', scale_unit=scale_unit)
    # root_name = optim_strategy.strategy_name
    # for rc in range(2, 3):
    #     optim_strategy.strategy_name = f"{root_name}_{rc * 10}"
    #     optim_strategy.strategy_risk_class = str(rc * 10)
    #     optim_strategy.manage_portfolio()
    #     optim_strategy.export_files()
    #     kpi_dic = optim_strategy.get_kpi()
    #     kpi_df = pd.DataFrame([kpi_dic])
    #     kpi_df_ls.append(kpi_df)
    # kpi_all_df = pd.concat(kpi_df_ls)
    # kpi_all_df.set_index("Strategy", inplace=True)
    # folderpath = optim_strategy.root_path
    # filename = f"kpi_summary_optimal.csv"


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
    # strategy_obj_ls = []
    # # strategy_obj_ls = [optim_strategy, benchmark_unit1_only, benchmark_never, benchmark_monthly, benchmark_quarterly, benchmark_annually]
    # # strategy_obj_ls = [benchmark_unit1_only, benchmark_never, benchmark_monthly, benchmark_quarterly, benchmark_annually]
    # # strategy_obj_ls = [benchmark_never]
    # # kpi_df_ls = []
    # # for strategy in strategy_obj_ls:
    # #     kpi_dic = strategy.get_kpi()
    # #     kpi_df = pd.DataFrame([kpi_dic])
    # #     kpi_df_ls.append(kpi_df)
    # kpi_all_df = pd.concat(kpi_df_ls)
    # # kpi_all_df.insert(0, 'Optimal Threshold', None)
    # kpi_all_df.set_index("Strategy", inplace=True)
    # # kpi_all_df.loc[optim_strategy.strategy_name, "Optimal Threshold"] = res.x
    # folderpath = benchmark.root_path
    # filename = f"kpi_summary_{bm_type}.csv"
    kpi_all_df.to_csv(folderpath / filename)
    print(f'KPI Summary exported to {folderpath / filename}')
