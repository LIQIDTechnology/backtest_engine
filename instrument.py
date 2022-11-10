import datetime as dt
class Instrument(object):

    def __init__(self, instrument: str = None, risk_class: int = None):
        print(instrument)
        self.ticker = instrument.name
        self.name = instrument["Instrument Name"]
        self.type = instrument["Type"]  # equity, bond, index, exchange traded fund, open fund
        self.currency = instrument["Currency"]  # Currency
        self.weight = instrument[risk_class]
        self.unit1 = instrument["UNIT I"]
        self.unit2 = instrument["UNIT II"]
        self.unit3 = instrument["UNIT III"]
        self.product_cost = instrument["Product Cost"]
        self.available_from = dt.datetime.strptime(instrument["Available from"], "%Y-%m-%d").date()
        self.substitute_bool = True if int(instrument["Substitute Bool"]) == 1 else False
        self.substitute_ticker = instrument["Substitute Index Ticker"]
        self.substitute_available_from = dt.datetime.strptime(instrument["Substitute available from"], "%Y-%m-%d").date() if self.substitute_bool else None
        self.substitute_available_from = instrument["Substitute available from"]

    def __str__(self):
        return f"Instrument({self.type}/{self.ticker}/{self.currency}/{self.weight}/{self.unit1}/{self.unit2}/{self.unit3}/{self.name})"

    def __repr__(self):
        return f"Instrument({self.type}/{self.ticker}/{self.currency}/{self.weight}/{self.unit1}/{self.unit2}/{self.unit3}/{self.name})"
