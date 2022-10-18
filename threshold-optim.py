from pathlib import Path
from strategies.strategy import Strategy


class ThresholdOptimizer(object):

    def __init__(self, params):
        pass

    def sharpe_ratio(self, pf_ret):
        return 3

    def f(self, scale_unit):
        config_path = Path('config') / 'config.ini'
        strategy = Strategy(config_path=config_path, scale_unit=scale_unit)
        strategy.manage_portfolio()
        pf_ret = strategy.details["Portfolio Return"].values[1:]
        sr = self.sharpe_ratio(pf_ret)
        return -sr

    def threshold_optimum(self) -> dict:
        self.f(0.03)
        result = {}
        return result


if __name__ == "__main__":

    params = ["which strategy"]
    study = ThresholdOptimizer(params=params)
    res = study.threshold_optimum()


