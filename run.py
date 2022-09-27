from strategy import Strategy
from pathlib import Path
import scipy

if __name__ == "__main__":
    config_path = Path('config') / 'config.ini'
    strategy = Strategy(config_path=config_path)
    total_return = strategy.manage_portfolio(params=[0.03])
    for rk in range(10, 110, 10):
        results = scipy.optimize.minimize(-strategy.manage_portfolio(params=params), params, methods="")
