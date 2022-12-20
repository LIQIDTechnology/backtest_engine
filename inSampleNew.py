from pathlib import Path
from strategies.strategy import Strategy
from statistics import Statistic
import seaborn as sns
import pandas as pd
import datetime as dt

# n = 1000
# sc_iter_ls = [x / n for x in range(0, n)]
# twenty = int(n/5)

columns = [str(x * 10) for x in range(1, 11)]
rows = ["Total Return",
        "Volatility",
        "MDD",
        "# Rebalancing Unit 1",
        "# Rebalancing Unit 2",
        "# Rebalancing Unit 3",
        "Trading Kosten",
        "Monthly Rebal. Total Return",
        "Unit1 Rebal. Total Return",
        "2011 TR",
        "2012 TR",
        "2013 TR",
        "2014 TR",
        "2015 TR",
        "2016 TR",
        "2017 TR",
        "2018 TR",
        "2019 TR",
        "2020 TR",
        "2021 TR",
        "2022 TR"]

global_df = pd.DataFrame(columns=columns, index=rows)


for rc in range(1, 11):
    sc_file = pd.read_csv(
        "/Volumes/GoogleDrive/.shortcut-targets-by-id/19JZlTc1zpxipptIN5dg3E-SGtYp5rg7r/02_Backtesting/04_Python Code/source/insample_summary_2011-01-18_2022-01-18.csv",
        index_col=0)
    sc_optim = float(sc_file.loc["Optimale S-Unit", f"{rc * 10}"])

    sc = sc_optim
    print(sc)
    optim_strategy = Strategy(config_path=Path('config') / 'config_strategy.ini', scale_unit=sc)
    folderpath = optim_strategy.root_path
    root_name = optim_strategy.strategy_name

    rc_str = str(rc * 10)
    optim_strategy.strategy_name = f"{root_name}_{rc * 10}"
    optim_strategy.start_date = dt.date(2011, 1, 18)
    optim_strategy.end_date = dt.date(2022, 1, 18)
    optim_strategy.strategy_risk_class = rc_str
    optim_strategy.manage_portfolio()
    kpi_dic = Statistic(optim_strategy).get_kpi()

    global_df.loc["Total Return", rc_str] = kpi_dic["Total Return"]
    global_df.loc["Volatility", rc_str] = kpi_dic["Volatility"]
    global_df.loc["MDD", rc_str] = kpi_dic["Maximum Drawdown"]
    global_df.loc["# Rebalancing Unit 1", rc_str] = kpi_dic["Rebalancing Count UNIT1"]
    global_df.loc["# Rebalancing Unit 2", rc_str] = kpi_dic["Rebalancing Count UNIT2"]
    global_df.loc["# Rebalancing Unit 3", rc_str] = kpi_dic["Rebalancing Count UNIT3"]
    global_df.loc["Trading Kosten", rc_str] = kpi_dic["Trading Cost"]
    try:
        global_df.loc["2001 TR", rc_str] = kpi_dic["TR 2001"]
        global_df.loc["2002 TR", rc_str] = kpi_dic["TR 2002"]
        global_df.loc["2003 TR", rc_str] = kpi_dic["TR 2003"]
        global_df.loc["2004 TR", rc_str] = kpi_dic["TR 2004"]
        global_df.loc["2005 TR", rc_str] = kpi_dic["TR 2005"]
        global_df.loc["2006 TR", rc_str] = kpi_dic["TR 2006"]
        global_df.loc["2007 TR", rc_str] = kpi_dic["TR 2007"]
        global_df.loc["2008 TR", rc_str] = kpi_dic["TR 2008"]
        global_df.loc["2009 TR", rc_str] = kpi_dic["TR 2009"]
        global_df.loc["2010 TR", rc_str] = kpi_dic["TR 2010"]
    except KeyError:
        pass

    global_df.loc["2011 TR", rc_str] = kpi_dic["TR 2011"]
    global_df.loc["2012 TR", rc_str] = kpi_dic["TR 2012"]
    global_df.loc["2013 TR", rc_str] = kpi_dic["TR 2013"]
    global_df.loc["2014 TR", rc_str] = kpi_dic["TR 2014"]
    global_df.loc["2015 TR", rc_str] = kpi_dic["TR 2015"]
    global_df.loc["2016 TR", rc_str] = kpi_dic["TR 2016"]
    global_df.loc["2017 TR", rc_str] = kpi_dic["TR 2017"]
    global_df.loc["2018 TR", rc_str] = kpi_dic["TR 2018"]
    global_df.loc["2019 TR", rc_str] = kpi_dic["TR 2019"]
    global_df.loc["2020 TR", rc_str] = kpi_dic["TR 2020"]
    global_df.loc["2021 TR", rc_str] = kpi_dic["TR 2021"]
    global_df.loc["2022 TR", rc_str] = kpi_dic["TR 2022"]

    print(kpi_dic["TR 2022"])

global_df.to_csv(optim_strategy.root_path / f"insample_split_summary_{optim_strategy.start_date}_{optim_strategy.end_date}.csv")