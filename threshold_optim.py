from pathlib import Path
import scipy

from strategies.strategy import Strategy
import numpy as np


class ThresholdOptimizer(object):

    def __init__(self):
        pass

    # def sr_martin(self, pf_ret):
    #     avg_ret = np.average(pf_ret)
    #     sr_1 = ((1 + avg_ret) ** 252) - 1
    #     stdev_pf_ret = np.std(pf_ret, ddof=1)
    #     sr_2 = stdev_pf_ret * np.sqrt(252)
    #     sr = sr_1 / sr_2
    #     return sr
    #
    # def objective(self, scale_unit):
    #     print(scale_unit)
    #     scale_unit = scale_unit[0]
    #     config_path = Path('config') / 'config.ini'
    #     strategy = Strategy(config_path=config_path, scale_unit=scale_unit)
    #
    #     strategy.manage_portfolio()
    #     pf_ret = strategy.details["Portfolio Return"].values[1:]
    #     sr = self.sr_martin(pf_ret)
    #     print(f'Scale Unit: {scale_unit}', f'Sharpe Ratio: {"{:.5f}".format(sr)}')
    #     # print(f'Threshold: {"{:.2f}".format(scale_unit * 100)} %')
    #     # print(f'Sharpe Ratio: {"{:.5f}".format(sr)}')
    #     return -sr
    #
    # def objective_scalar(self, scale_unit):
    #     config_path = Path('config') / 'config.ini'
    #     strategy = Strategy(config_path=config_path, scale_unit=scale_unit)
    #
    #     strategy.manage_portfolio()
    #     pf_ret = strategy.details["Portfolio Return"].values[1:]
    #     sr = self.sr_martin(pf_ret)
    #     print(f'Scale Unit: {scale_unit}', f'Sharpe Ratio: {"{:.5f}".format(sr)}')
    #     # print(f'Threshold: {"{:.2f}".format(scale_unit * 100)} %')
    #     # print(f'Sharpe Ratio: {"{:.5f}".format(sr)}')
    #     return -sr

    def objective_total_return(self, scale_unit):
        """
        Objective Function: Total Return of Strategy
        """
        config_path = Path('config') / 'config.ini'
        strategy = Strategy(config_path=config_path, scale_unit=scale_unit)
        strategy.manage_portfolio()
        tot_ret = (strategy.details.loc[strategy.end_date, "Cumulative Portfolio Return"])
        print(f'Scale Unit: {scale_unit}', f'Total Return: {"{:.5f}".format(tot_ret)}')
        return -tot_ret

    def threshold_optimum(self) -> dict:
        """"""
        bounds_ls = [[0, 0.05], [0.05, 0.1], [0, 0.1], [0.1, 0.15], [0, 0.15], [0.15, 0.2]]
        tr_max = 0
        result = None
        for bounds in bounds_ls:
            res = scipy.optimize.minimize_scalar(self.objective_total_return, bounds=bounds, method="bounded")
            tr = -self.objective_total_return(res.x)
            if tr > tr_max:
                tr_max = tr
                result = res
            else:
                pass
        return result

        #
        #
        #
        # init_wt = 0.1
        # # b = (0, 1)
        # b = (0.00, 0.05)
        # b1 = (0.05, 0.1)
        # b2 = (0.015, 0.2)
        # b3 = (0.25, 0.3)
        # b4 = (0.3, 0.35)
        # bnds = [b]
        # bracket_ls = [b, b1, b2, b3, b4]
        # result_list = []
        # mybounds = MyBounds()
        # minimizer_kwargs = {"method": "BFGS"}
        # ret = scipy.optimize.basinhopping(self.objective_scalar, [init_wt], accept_test=mybounds, stepsize= 0.01,
        #                                   minimizer_kwargs=minimizer_kwargs, niter=200)
        # for brackets in bracket_ls:
        #     res = scipy.optimize.minimize_scalar(self.objective_scalar, bounds=brackets, method="bounded")
        #     result_list.append([brackets, res.x])
        # res = scipy.optimize.minimize_scalar(self.objective_scalar, bounds=[0, 0.1], method="bounded")
        # res.x
        # # self.objective_scalar(0.089)
        # res = scipy.optimize.minimize(self.objective_scalar, init_wt, method="nelder-mead")
        # res = scipy.optimize.brute(self.objective_scalar, [init_wt])
        # res = scipy.optimize.basinhopping(self.objective_scalar, [init_wt], stepsize=0.01, minimizer_kwargs={"method": "L-BFGS-B"})


if __name__ == "__main__":
    study = ThresholdOptimizer()
    res = study.threshold_optimum()
    print(res.x)

