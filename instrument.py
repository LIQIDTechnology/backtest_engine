class Instrument(object):

    def __init__(self, instrument: str = None):
        self.ticker = instrument.name
        self.name = instrument["Instrument Name"]
        self.type = instrument["Type"]  # equity, bond, index, exchange traded fund, open fund
        self.cost = None  # TER p.a.
        self.currency = instrument["Currency"]  # Currency

    def __str__(self):
        return f"Instrument({'Index'}/{self.ticker}/{self.name}/{self.currency}/{self.cost})"

    def __repr__(self):
        return f"Instrument({'Index'}/{self.ticker}/{self.name}/{self.currency}/{self.cost})"
