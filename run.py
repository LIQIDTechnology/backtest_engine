from pathlib import Path
from strategies.benchmarks import Benchmark
from strategies.strategy import Strategy

if __name__ == "__main__":
    # res = scipy.optimize.minimize_scalar(f, bounds=(0.2, 0.25), method='bounded')
    # print(res.x)
    scale_unit = 0.13
    config_path = Path('config') / 'config.ini'
    strategy = Strategy(config_path=config_path, scale_unit=scale_unit)
    strategy.manage_portfolio()
