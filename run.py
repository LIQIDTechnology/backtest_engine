import cProfile
import pstats
from pathlib import Path

import scipy.optimize

from strategy import Strategy
import numpy as np
from scipy import optimize


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
    print(f'Threshold: {"{:.2f}".format(scale_unit*100)} %')
    print(f'Sharpe Ratio: {"{:.5f}".format(sr)}')
    return -sr


if __name__ == "__main__":
    f(0.03)
