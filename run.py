from strategy import Strategy
from pathlib import Path

if __name__ == "__main__":
    config_path = Path('config') / 'config.ini'
    strategy = Strategy(config_path=config_path)
    strategy.manage_portfolio()
