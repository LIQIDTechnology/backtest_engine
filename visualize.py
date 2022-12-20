from pathlib import Path
from strategies.strategy import Strategy
from statistics import Statistic
import seaborn as sns
import pandas as pd
import datetime as dt

# n = 1000
# sc_iter_ls = [x / n for x in range(0, n)]
# twenty = int(n/5)


for rc in range(10, 11):
    sc_file = pd.read_csv("/Volumes/GoogleDrive/.shortcut-targets-by-id/19JZlTc1zpxipptIN5dg3E-SGtYp5rg7r/02_Backtesting/04_Python Code/source/insample_summary_2011-01-18_2022-10-18.csv", index_col=0)
    sc_optim = float(sc_file.loc["Optimale S-Unit", f"{rc * 10}"])
    sc_iter_ls = [sc_optim - 0.03, sc_optim - 0.02, sc_optim - 0.01, sc_optim, sc_optim + 0.01, sc_optim + 0.02, sc_optim + 0.03]
    # sc_iter_ls = [sc_optim]
    visual_df = pd.DataFrame()
    for sc in sc_iter_ls:
        print(sc)
        optim_strategy = Strategy(config_path=Path('config') / 'config_strategy.ini', scale_unit=sc)
        folderpath = optim_strategy.root_path
        root_name = optim_strategy.strategy_name
        tot_dic = {}

        optim_strategy.strategy_name = f"{root_name}_{rc * 10}"
        optim_strategy.end_date = dt.date(2022, 1, 18)
        optim_strategy.strategy_risk_class = str(rc * 10)
        optim_strategy.manage_portfolio()
        total_return = Statistic(optim_strategy).get_total_return()
        total_rebal = Statistic(optim_strategy).get_total_rebal()
        print(f"sc{sc}, tr{total_return}")
        tot_dic.update({str(rc * 10): total_return})
        tot_dic.update({f"{str(rc * 10)}_reabl": total_rebal})
        tot_ret_df = pd.DataFrame([tot_dic], index=[sc])
        visual_df = pd.concat([visual_df, tot_ret_df])

    visual_df.to_csv(folderpath / f'visual_summary_{rc}_{optim_strategy.start_date}_{optim_strategy.end_date}.csv')


    # visual_df = pd.read_csv('/Volumes/GoogleDrive/.shortcut-targets-by-id/19JZlTc1zpxipptIN5dg3E-SGtYp5rg7r/02_Backtesting/04_Python Code/source/visual_summary.csv', index_col="Unnamed: 0")
    # Use seaborn to create a heatmap
    visual_plot = sns.heatmap(visual_df, annot=True, fmt="0.2f", cmap="YlGnBu")
    fig = visual_plot.get_figure()
# fig.savefig('/Volumes/GoogleDrive/.shortcut-targets-by-id/19JZlTc1zpxipptIN5dg3E-SGtYp5rg7r/02_Backtesting/04_Python Code/source/visual.png')
    fig.savefig(folderpath / f'visual_{rc}_{optim_strategy.start_date}_{optim_strategy.end_date}.png')
