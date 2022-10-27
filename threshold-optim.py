from pathlib import Path
import scipy

from strategies.strategy import Strategy
import numpy as np

class ThresholdOptimizer(object):

    def __init__(self):
        pass

    def sr_martin(self, pf_ret):
        avg_ret = np.average(pf_ret)
        sr_1 = ((1 + avg_ret) ** 252) - 1
        stdev_pf_ret = np.std(pf_ret, ddof=1)
        sr_2 = stdev_pf_ret * np.sqrt(252)
        sr = sr_1 / sr_2
        return sr

    def objective(self, scale_unit):
        print(scale_unit)
        scale_unit = scale_unit[0]
        config_path = Path('config') / 'config.ini'
        strategy = Strategy(config_path=config_path, scale_unit=scale_unit)

        strategy.manage_portfolio()
        pf_ret = strategy.details["Portfolio Return"].values[1:]
        sr = self.sr_martin(pf_ret)
        print(f'Scale Unit: {scale_unit}', f'Sharpe Ratio: {"{:.5f}".format(sr)}')
        # print(f'Threshold: {"{:.2f}".format(scale_unit * 100)} %')
        # print(f'Sharpe Ratio: {"{:.5f}".format(sr)}')
        return -sr

    def objective_scalar(self, scale_unit):
        config_path = Path('config') / 'config.ini'
        strategy = Strategy(config_path=config_path, scale_unit=scale_unit)

        strategy.manage_portfolio()
        pf_ret = strategy.details["Portfolio Return"].values[1:]
        sr = self.sr_martin(pf_ret)
        print(f'Scale Unit: {scale_unit}', f'Sharpe Ratio: {"{:.5f}".format(sr)}')
        # print(f'Threshold: {"{:.2f}".format(scale_unit * 100)} %')
        # print(f'Sharpe Ratio: {"{:.5f}".format(sr)}')
        return -sr

    def threshold_optimum(self) -> dict:
        # self.f(0.03)
        init_wt = 0.1
        # b = (0, 1)
        b = (0, 0.1)
        bnds = [b, b]
        self.objective_scalar(0.089)
        res = scipy.optimize.minimize_scalar(self.objective_scalar, bounds=b, method="bounded")

        self.objective_scalar(0.3204861292689471)
        sr_max = 0
        step_max = 0
        for step in range(0, 300):
            res = - self.objective_scalar(step / 1000)
            step_max = step / 1000 if sr_max < res else step_max
            sr_max = res if sr_max < res else sr_max
            print(step_max)

            # print(k/1000)
        - self.objective(120 / 1000)

        res = scipy.optimize.minimize(self.objective, init_wt, method="nelder-mead")
        res = scipy.optimize.minimize(self.objective, init_wt, bounds=bnds, method='SLSQP')  # cannot use bounds, now can

        res = scipy.optimize.basinhopping(self.objective, init_wt,)


        res

        # def constraint1()
        # con1 = {'type': 'ineq', 'fun': constraint1}
        print(res.x[0])
        print(self.objective(res.x[0]))

        self.objective(0.086)

        result = {}
        return result


if __name__ == "__main__":
    study = ThresholdOptimizer()
    res = study.threshold_optimum()
