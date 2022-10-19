from pathlib import Path

import scipy.optimize

# from strategies.strategy_mvp2 import Strategy
# from strategies.benchmark_monthly import Strategy
from strategies.benchmarks import Strategy
import numpy as np


def sr_martin(pf_ret):
    avg_ret = np.average(pf_ret)
    sr_1 = ((1 + avg_ret) ** 252) - 1
    stdev_pf_ret = np.std(pf_ret, ddof=1)
    sr_2 = stdev_pf_ret * np.sqrt(252)
    sr = sr_1 / sr_2
    return sr


def f(scale_unit):
    config_path = Path('config') / 'config.ini'
    strategy = Strategy(config_path=config_path, scale_unit=scale_unit)
    strategy.manage_portfolio()
    pf_ret = strategy.details["Portfolio Return"].values[1:]
    sr = sr_martin(pf_ret)
    print(f'Scale Unit: {scale_unit}')
    print(f'Threshold: {"{:.2f}".format(scale_unit*100)} %')
    print(f'Sharpe Ratio: {"{:.5f}".format(sr)}')
    return -sr


if __name__ == "__main__":
    # res = scipy.optimize.minimize_scalar(f, bounds=(0.2, 0.25), method='bounded')
    # print(res.x)
    scale_unit = 0.03
    config_path = Path('config') / 'config.ini'
    strategy = Strategy(config_path=config_path, scale_unit=scale_unit)
    strategy.manage_portfolio()
