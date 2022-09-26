import datetime as dt


class Calendar(object):
    def __init__(self, mic_str: str = None):
        self.mic_ls = mic_str

    def bday_range(self, start: dt, end: dt) -> list:
        numdays = end-start
        start.weekday()
        bday_range = [start + dt.timedelta(days=k) for k in range(0,(numdays.days)) if (start + dt.timedelta(days=k)).weekday() < 5]
        return bday_range

    def __repr__(self):
        return self.mic_ls
