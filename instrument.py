class Instrument(object):

    def __init__(self, instrument: str = None):
        self.ticker = instrument.name
        self.name = instrument["Instrument Name"]
        self.type = instrument["Type"]  # equity, bond, index, exchange traded fund, open fund
        self.currency = instrument["Currency"]  # Currency
        self.cost = None  # TER p.a.
        self.weight = instrument["Weight"]

    def __str__(self):
        return f"Instrument({self.type}/{self.ticker}/{self.name}/{self.currency}/{self.weight})"

    def __repr__(self):
        return f"Instrument({self.type}/{self.ticker}/{self.name}/{self.currency}/{self.weight})"
