import cProfile
import pstats
from pathlib import Path

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
    return sr


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    # # define the starting point as a random sample from the domain
    # pt = 0.03
    # # perform the l-bfgs-b algorithm search
    # result = optimize.minimize(f, pt, method='L-BFGS-B')
    # print(result)
    # sr_ls = []
    #
    # for k in range(0, 101):
    #     sr = f(k/100)
    #     print(k, sr)
    #     sr_ls.append(sr)

    sr = f(0.03)
    print(sr)

    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
