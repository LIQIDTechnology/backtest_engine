from pathlib import Path
import scipy

from strategies.strategy import Strategy
import numpy as np
import gurobipy as gb

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
        scale_unit_jo = scale_unit[0]
        config_path = Path('config') / 'config.ini'
        strategy = Strategy(config_path=config_path, scale_unit=scale_unit_jo)

        strategy.manage_portfolio()
        pf_ret = strategy.details["Portfolio Return"].values[1:]
        sr = self.sr_martin(pf_ret)
        print(f'Scale Unit: {scale_unit_jo}', f'Sharpe Ratio: {"{:.5f}".format(sr)}')
        # print(f'Threshold: {"{:.2f}".format(scale_unit * 100)} %')
        # print(f'Sharpe Ratio: {"{:.5f}".format(sr)}')
        return -sr

    def threshold_optimum(self) -> dict:
        # self.f(0.03)
        init_wt = 0.1
        # b = (0, 1)
        b = (0, 0.5)
        bnds = [b, b]

        # bnds = (b)
        # res = scipy.optimize.minimize(self.objective, init_wt, method="SLSQP", bounds=bnds)
        # res = scipy.optimize.minimize_scalar(self.objective, bounds=bnds, method="golden")
        res = scipy.optimize.brute(self.objective, ranges=bnds)
        res = scipy.optimize.minimize(self.objective, init_wt, method="nelder-mead")
        res = scipy.optimize.dual_annealing(self.objective, bounds=bnds, maxiter=100)
        sr_max = 0
        k_max = 0
        for k in range(0, 300):
            res = - self.objective(k / 1000)
            sr_max = res if sr_max < res else sr_max
            k_max = k / 1000 if sr_max < res else k_max

            # print(k/1000)
        self.objective(120 / 1000)

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
