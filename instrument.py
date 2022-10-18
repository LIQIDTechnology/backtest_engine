class Instrument(object):

    def __init__(self, instrument: str = None):
        self.ticker = instrument.name
        self.name = instrument["Instrument Name"]
        self.type = instrument["Type"]  # equity, bond, index, exchange traded fund, open fund
        self.currency = instrument["Currency"]  # Currency
        self.cost = None  # TER p.a.
        self.weight = instrument["Weight"]
        self.unit1 = instrument["UNIT I"]
        self.unit2 = instrument["UNIT II"]
        self.unit3 = instrument["UNIT III"]
        self.product_cost = instrument["Product Cost"]

    def __str__(self):
        return f"Instrument({self.type}/{self.ticker}/{self.name}/{self.currency}/{self.weight}/{self.unit1}/{self.unit2}/{self.unit3})"

    def __repr__(self):
        return f"Instrument({self.type}/{self.ticker}/{self.name}/{self.currency}/{self.weight}/{self.unit1}/{self.unit2}/{self.unit3})"
